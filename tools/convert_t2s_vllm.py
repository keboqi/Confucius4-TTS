#!/usr/bin/env python3
"""Convert Confucius4-TTS T2S weights into a vLLM-loadable directory."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

import safetensors.torch
import torch
import yaml
from huggingface_hub import hf_hub_download
from transformers import GPT2Config

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from confuciustts.llm.llm import Text2Semantic, Text2SemanticConfig


VLLM_WEIGHT_PREFIXES = (
    "semantic_embedding.",
    "semantic_position_embedding.",
    "transformer.",
    "final_norm.",
    "semantic_head.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Confucius Text2Semantic weights for vLLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="config/inference_config.yaml")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Local T2S checkpoint path or Hugging Face filename. Defaults to "
            "paths.t2s_checkpoint from the config."
        ),
    )
    parser.add_argument("--output", default="checkpoints/t2s-vllm")
    parser.add_argument("--repo-id", default="netease-youdao/Confucius4-TTS")
    return parser.parse_args()


def load_state_dict(checkpoint: str, repo_id: str) -> Dict[str, torch.Tensor]:
    ckpt_path = Path(checkpoint)
    if ckpt_path.exists():
        resolved = str(ckpt_path)
    else:
        resolved = hf_hub_download(repo_id, filename=checkpoint)

    if resolved.endswith(".safetensors"):
        return safetensors.torch.load_file(resolved, device="cpu")

    state = torch.load(resolved, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return {
        key.replace("t2s_model.", "", 1): value
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    }


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    model_cfg = Text2SemanticConfig(**cfg["t2s_model"])
    checkpoint = args.checkpoint or cfg["paths"]["t2s_checkpoint"]
    state_dict = load_state_dict(checkpoint, args.repo_id)

    model = Text2Semantic(model_cfg)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[convert_t2s_vllm] missing keys: {len(missing)}")
    if unexpected:
        print(f"[convert_t2s_vllm] unexpected keys: {len(unexpected)}")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    filtered = {
        key: value.contiguous()
        for key, value in model.state_dict().items()
        if key.startswith(VLLM_WEIGHT_PREFIXES)
    }
    safetensors.torch.save_file(filtered, str(output / "model.safetensors"))

    max_seq_len = (
        model_cfg.max_text_seq_lens
        + model_cfg.max_semantic_seq_lens
        + 1
    )
    hf_config = GPT2Config(
        vocab_size=model_cfg.semantic_vocab_size,
        n_positions=max_seq_len,
        n_ctx=max_seq_len,
        n_embd=model_cfg.model_dim,
        n_layer=model_cfg.num_layers,
        n_head=model_cfg.num_heads,
        gradient_checkpointing=False,
        use_cache=True,
    )
    hf_config.architectures = ["ConfuciusText2SemanticForCausalLM"]
    hf_config.model_type = "gpt2"
    hf_config.max_text_seq_lens = model_cfg.max_text_seq_lens
    hf_config.max_semantic_seq_lens = model_cfg.max_semantic_seq_lens
    hf_config.semantic_vocab_size = model_cfg.semantic_vocab_size
    hf_config.start_semantic_token = model_cfg.start_semantic_token
    hf_config.stop_semantic_token = model_cfg.stop_semantic_token
    hf_config.text2semantic_config = model_cfg.to_dict()

    with open(output / "config.json", "w", encoding="utf-8") as file:
        json.dump(hf_config.to_dict(), file, indent=2, sort_keys=True)
        file.write("\n")

    with open(output / "README.md", "w", encoding="utf-8") as file:
        file.write(
            "# Confucius4-TTS T2S vLLM export\n\n"
            "This directory contains only the autoregressive T2S weights used "
            "by the optional Confucius4-TTS vLLM backend.\n"
        )

    print(f"[convert_t2s_vllm] wrote {len(filtered)} tensors to {output}")


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
