#!/usr/bin/env python3
"""Gradio web UI for Confucius4-TTS zero-shot voice cloning."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import gradio as gr
import torch
import torchaudio

ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs" / "gradio"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

LANGUAGE_CHOICES = [
    "en - English",
    "zh - Chinese",
    "ja - Japanese",
    "ko - Korean",
    "de - German",
    "fr - French",
    "es - Spanish",
    "id - Indonesian",
    "it - Italian",
    "th - Thai",
    "pt - Portuguese",
    "ru - Russian",
    "ms - Malay",
    "vi - Vietnamese",
]


def _language_code(choice: str) -> str:
    return choice.split(" - ", 1)[0].strip()


def _resolve_repo_path(value: str, label: str, must_exist: bool = True) -> str:
    value = value.strip()
    if not value:
        raise gr.Error(f"{label} is required.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT_DIR / path
    path = path.resolve()
    if must_exist and not path.exists():
        raise gr.Error(f"{label} does not exist: {path}")
    if must_exist and not path.is_file():
        raise gr.Error(f"{label} must be a file: {path}")
    return str(path)


def _resolve_device(device_choice: str) -> str:
    if device_choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_choice == "cuda" and not torch.cuda.is_available():
        raise gr.Error("CUDA was selected, but torch.cuda.is_available() is false.")
    return device_choice


def _normalize_checkpoint(value: str) -> Optional[str]:
    value = value.strip()
    return value or None


@lru_cache(maxsize=4)
def _load_model(config_path: str, t2s_checkpoint: Optional[str], device: str) -> Any:
    try:
        from confuciustts.cli.inference import ConfuciusTTS
    except Exception as exc:
        raise gr.Error(
            "Could not import ConfuciusTTS. Reinstall the matching Torch packages with "
            "`pip install --force-reinstall torch==2.7.0 torchaudio==2.7.0 torchvision==0.22.0`, "
            "then run `pip install -r requirements.txt` again."
        ) from exc

    return ConfuciusTTS(
        config_path=config_path,
        t2s_checkpoint=t2s_checkpoint,
        device=device,
    )


def _output_path(prompt_wav: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(f"{prompt_wav}-{time.time_ns()}".encode("utf-8")).hexdigest()[:8]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return OUTPUT_DIR / f"confucius4tts-{stamp}-{digest}.wav"


def synthesize(
    prompt_wav: Optional[str],
    text: str,
    language: str,
    device_choice: str,
    config_path: str,
    t2s_checkpoint: str,
    temperature: float,
    top_p: float,
    top_k: int,
    num_beams: int,
    repetition_penalty: float,
    max_length: int,
    diffusion_steps: int,
    cfg_strength: float,
    max_text_tokens: int,
    cross_fade_duration: float,
    verbose: bool,
) -> tuple[str, str, str]:
    if not prompt_wav:
        raise gr.Error("Upload or record a reference audio file.")
    if not Path(prompt_wav).exists():
        raise gr.Error(f"Reference audio does not exist: {prompt_wav}")

    text = text.strip()
    if not text:
        raise gr.Error("Enter text to synthesize.")

    config = _resolve_repo_path(config_path, "Config")
    device = _resolve_device(device_choice)
    checkpoint = _normalize_checkpoint(t2s_checkpoint)
    lang = _language_code(language)

    model = _load_model(config, checkpoint, device)

    started = time.time()
    audio = model.generate(
        text=text,
        lang=lang,
        prompt_wav=prompt_wav,
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        num_beams=int(num_beams),
        repetition_penalty=float(repetition_penalty),
        max_length=int(max_length),
        n_timesteps=int(diffusion_steps),
        inference_cfg_rate=float(cfg_strength),
        max_text_tokens_per_segment=int(max_text_tokens),
        cross_fade_duration=float(cross_fade_duration),
        verbose=bool(verbose),
    )

    output = _output_path(prompt_wav)
    torchaudio.save(str(output), audio.detach().float().cpu(), model.sample_rate)

    generated_seconds = audio.shape[-1] / model.sample_rate
    elapsed = time.time() - started
    status = (
        f"Generated {generated_seconds:.2f}s of audio in {elapsed:.2f}s "
        f"on {device}. Saved to {output}."
    )
    return str(output), str(output), status


def clear_model_cache() -> str:
    _load_model.cache_clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return "Model cache cleared."


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Confucius4-TTS") as demo:
        gr.Markdown("# Confucius4-TTS")

        with gr.Row():
            with gr.Column(scale=1):
                prompt_wav = gr.Audio(
                    label="Reference audio",
                    type="filepath",
                )
                language = gr.Dropdown(
                    label="Language",
                    choices=LANGUAGE_CHOICES,
                    value=LANGUAGE_CHOICES[0],
                )
                device_choice = gr.Radio(
                    label="Device",
                    choices=["auto", "cuda", "cpu"],
                    value="auto",
                )

            with gr.Column(scale=2):
                text = gr.Textbox(
                    label="Text",
                    value="Hello, welcome to Confucius4-TTS.",
                    lines=6,
                    max_lines=14,
                )
                with gr.Row():
                    generate_btn = gr.Button("Generate", variant="primary")
                    clear_btn = gr.Button("Clear model cache")

        with gr.Accordion("Generation settings", open=False):
            with gr.Row():
                temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="Temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.8, step=0.05, label="Top-p")
                top_k = gr.Slider(1, 100, value=30, step=1, label="Top-k")
            with gr.Row():
                num_beams = gr.Slider(1, 8, value=3, step=1, label="Beams")
                repetition_penalty = gr.Slider(1.0, 20.0, value=10.0, step=0.5, label="Repetition penalty")
                diffusion_steps = gr.Slider(1, 80, value=25, step=1, label="Diffusion steps")
            with gr.Row():
                cfg_strength = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="CFG strength")
                max_length = gr.Slider(128, 2048, value=1520, step=16, label="Max semantic length")
                max_text_tokens = gr.Slider(20, 240, value=80, step=5, label="Segment token limit")
            cross_fade_duration = gr.Slider(0.0, 2.0, value=0.3, step=0.05, label="Cross fade seconds")

        with gr.Accordion("Model loading", open=False):
            config_path = gr.Textbox(
                label="Config path",
                value="config/inference_config.yaml",
            )
            t2s_checkpoint = gr.Textbox(
                label="T2S checkpoint filename override",
                value="",
                placeholder="Leave blank to use config/inference_config.yaml",
            )
            verbose = gr.Checkbox(label="Verbose inference logs", value=False)

        output_audio = gr.Audio(label="Generated audio", type="filepath")
        output_file = gr.File(label="Download WAV")
        status = gr.Textbox(label="Status", interactive=False)

        generate_btn.click(
            fn=synthesize,
            inputs=[
                prompt_wav,
                text,
                language,
                device_choice,
                config_path,
                t2s_checkpoint,
                temperature,
                top_p,
                top_k,
                num_beams,
                repetition_penalty,
                max_length,
                diffusion_steps,
                cfg_strength,
                max_text_tokens,
                cross_fade_duration,
                verbose,
            ],
            outputs=[output_audio, output_file, status],
        )
        clear_btn.click(fn=clear_model_cache, outputs=status)

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the Confucius4-TTS Gradio UI.")
    parser.add_argument("--server-name", default=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"))
    parser.add_argument("--server-port", type=int, default=int(os.getenv("GRADIO_SERVER_PORT", "7860")))
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--inbrowser", action="store_true")
    parser.add_argument("--root-path", default=os.getenv("GRADIO_ROOT_PATH"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    launch_kwargs = {
        "server_name": args.server_name,
        "server_port": args.server_port,
        "share": args.share,
        "inbrowser": args.inbrowser,
    }
    if args.root_path:
        launch_kwargs["root_path"] = args.root_path

    build_demo().queue(default_concurrency_limit=1).launch(**launch_kwargs)


if __name__ == "__main__":
    main()
