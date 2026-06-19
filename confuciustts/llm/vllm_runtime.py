"""Runtime wrapper for vLLM-backed Confucius Text2Semantic generation."""

from __future__ import annotations

import asyncio
import inspect
import os
import threading
import uuid
import warnings
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from confuciustts.llm.llm import Text2Semantic
from confuciustts.llm.vllm_model import PLACEHOLDER_TOKEN, PLACEHOLDER_TOKEN_ID
from confuciustts.llm.vllm_patch import register_confucius_vllm_model


_CONFUCIUS_VLLM_PLUGIN_NAME = "confucius4_tts"
_CONFUCIUS_VLLM_PLUGIN_REF = "confuciustts.llm.vllm_plugin:register"


class _BackgroundLoop:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=self.loop.run_forever,
            name="confucius-vllm-loop",
            daemon=True,
        )
        self.thread.start()

    def run(self, coro):
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result()


class Text2SemanticVLLM:
    """Use vLLM for autoregressive semantic token decoding.

    The original PyTorch Text2Semantic module is still used for prefix embedding
    and the one-shot latent pass required by Confucius's S2A model. vLLM handles
    the slow autoregressive semantic-token loop.
    """

    def __init__(
        self,
        torch_model: Text2Semantic,
        model_dir: str,
        gpu_memory_utilization: float = 0.25,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        attention_backend: Optional[str] = None,
        max_num_seqs: Optional[int] = None,
        max_model_len: Optional[int] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        _ensure_confucius_vllm_plugin_entrypoint()
        register_confucius_vllm_model()

        from vllm import SamplingParams
        from vllm.engine.arg_utils import AsyncEngineArgs

        try:
            from vllm.v1.engine.async_llm import AsyncLLM
        except Exception as exc:
            raise RuntimeError(
                "Confucius4-TTS vLLM mode requires a vLLM build with "
                "vllm.v1.engine.async_llm.AsyncLLM."
            ) from exc

        self.torch_model = torch_model
        self.model_dir = str(Path(model_dir))
        self.SamplingParams = SamplingParams
        self._loop: Optional[_BackgroundLoop] = None

        args: Dict[str, Any] = {
            "model": self.model_dir,
            "tensor_parallel_size": tensor_parallel_size,
            "dtype": dtype,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": True,
            "enable_mm_embeds": True,
            "async_scheduling": True,
            "skip_tokenizer_init": True,
        }
        if attention_backend:
            args["attention_backend"] = attention_backend
        if max_num_seqs is not None:
            args["max_num_seqs"] = max_num_seqs
        if max_model_len is not None:
            args["max_model_len"] = max_model_len
        if engine_kwargs:
            args.update(engine_kwargs)

        engine_args = AsyncEngineArgs(**_filter_kwargs(AsyncEngineArgs, args))
        self.llm = AsyncLLM.from_engine_args(engine_args)

        try:
            from vllm import TokensPrompt
        except Exception:
            from vllm.inputs import TokensPrompt

        self.TokensPrompt = TokensPrompt

    @property
    def device(self) -> torch.device:
        return next(self.torch_model.parameters()).device

    @torch.no_grad()
    def _build_prefix_embeddings(
        self,
        text_inputs: torch.Tensor,
        condition_vector: torch.Tensor,
    ) -> torch.Tensor:
        if text_inputs.shape[0] != 1:
            raise ValueError("The vLLM T2S wrapper currently expects batch size 1.")

        model = self.torch_model
        condition_emb = model.speaker_encoder(condition_vector).unsqueeze(1)
        text_emb = model.text_projector(text_inputs)
        text_emb = model.text_position_embedding(text_emb)

        bos = torch.full(
            (text_inputs.shape[0], 1),
            model.config.start_semantic_token,
            dtype=torch.long,
            device=text_inputs.device,
        )
        bos_emb = model.semantic_embedding(bos)
        return torch.cat([condition_emb, text_emb, bos_emb], dim=1)

    def _sampling_params(
        self,
        max_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        do_sample: bool,
        eos_token_id: int,
        num_beams: int,
        seed: Optional[int],
    ):
        kwargs: Dict[str, Any] = {
            "max_tokens": max_tokens,
            "temperature": temperature if do_sample else 0.0,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "stop_token_ids": [eos_token_id],
        }
        if seed is not None:
            kwargs["seed"] = int(seed)

        params = inspect.signature(self.SamplingParams).parameters
        if num_beams > 1:
            if do_sample and "best_of" in params:
                kwargs["best_of"] = num_beams
            elif not do_sample and "use_beam_search" in params:
                kwargs["use_beam_search"] = True
                kwargs["best_of"] = num_beams
            elif not do_sample and "best_of" in params:
                kwargs["best_of"] = num_beams
            else:
                warnings.warn(
                    "This vLLM version does not expose compatible beam/best-of "
                    "parameters; falling back to single-candidate sampling for "
                    "Confucius T2S.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        return self.SamplingParams(**_filter_kwargs(self.SamplingParams, kwargs))

    async def async_generate(
        self,
        text_inputs: torch.Tensor,
        condition_vector: torch.Tensor,
        max_length: int = 500,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        return_latent: bool = False,
        num_beams: int = 1,
        repetition_penalty: float = 1.0,
        seed: Optional[int] = None,
        **kwargs: Any,
    ):
        eos = (
            eos_token_id
            if eos_token_id is not None
            else self.torch_model.config.stop_semantic_token
        )
        prefix_embeds = self._build_prefix_embeddings(
            text_inputs,
            condition_vector,
        )
        max_tokens = max(1, int(max_length) - prefix_embeds.shape[1])
        sampling_params = self._sampling_params(
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            eos_token_id=eos,
            num_beams=num_beams,
            seed=seed,
        )

        prompt_kwargs = {
            "multi_modal_data": {
                "audio": {
                    "audio_embeds": [prefix_embeds.squeeze(0).detach().cpu()],
                }
            }
        }
        try:
            prompt = self.TokensPrompt(
                prompt_token_ids=[PLACEHOLDER_TOKEN_ID],
                **prompt_kwargs,
            )
        except TypeError:
            prompt = self.TokensPrompt(prompt=PLACEHOLDER_TOKEN, **prompt_kwargs)
        output_generator = self.llm.generate(
            prompt,
            sampling_params=sampling_params,
            request_id=uuid.uuid4().hex,
        )

        final_output = None
        async for output in output_generator:
            final_output = output
        if final_output is None:
            raise RuntimeError("vLLM returned no output for Confucius T2S generation.")

        token_ids = list(final_output.outputs[0].token_ids)
        if eos in token_ids:
            token_ids = token_ids[: token_ids.index(eos)]

        semantic_codes = torch.tensor(
            token_ids,
            dtype=torch.long,
            device=text_inputs.device,
        ).unsqueeze(0)

        if not return_latent:
            return semantic_codes

        latent = self._compute_latent(text_inputs, condition_vector, semantic_codes)
        return {"semantic_codes": semantic_codes, "latent": latent}

    def generate(self, *args: Any, **kwargs: Any):
        if self._loop is None:
            self._loop = _BackgroundLoop()
        return self._loop.run(self.async_generate(*args, **kwargs))

    @torch.no_grad()
    def _compute_latent(
        self,
        text_inputs: torch.Tensor,
        condition_vector: torch.Tensor,
        semantic_codes: torch.Tensor,
    ) -> torch.Tensor:
        model = self.torch_model
        device = text_inputs.device
        batch_size = text_inputs.shape[0]
        bos = torch.full(
            (batch_size, 1),
            model.config.start_semantic_token,
            dtype=semantic_codes.dtype,
            device=device,
        )
        eos = torch.full(
            (batch_size, 1),
            model.config.stop_semantic_token,
            dtype=semantic_codes.dtype,
            device=device,
        )
        semantic_with_bounds = torch.cat([bos, semantic_codes, eos], dim=1)
        inputs_embeds = model._prepare_embed_inputs(
            text_inputs=text_inputs,
            semantic_codes=semantic_with_bounds,
            condition_vector=condition_vector,
        )
        text_lengths = torch.full(
            (batch_size,),
            text_inputs.shape[1],
            dtype=torch.long,
            device=device,
        )
        semantic_lengths = torch.full(
            (batch_size,),
            semantic_codes.shape[1],
            dtype=torch.long,
            device=device,
        )
        cond_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
        text_mask = (
            torch.arange(text_inputs.shape[1], device=device).unsqueeze(0)
            < text_lengths.unsqueeze(1)
        )
        semantic_mask = (
            torch.arange(semantic_with_bounds.shape[1], device=device).unsqueeze(0)
            < (semantic_lengths + 2).unsqueeze(1)
        )
        attention_mask = torch.cat([cond_mask, text_mask, semantic_mask], dim=1)
        transformer_outputs = model.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = transformer_outputs.last_hidden_state
        return hidden_states[:, 1 + text_inputs.shape[1] : -2]


def _ensure_confucius_vllm_plugin_entrypoint() -> None:
    installed = False
    try:
        eps = entry_points()
        if hasattr(eps, "select"):
            plugin_eps = eps.select(group="vllm.general_plugins")
        else:
            plugin_eps = eps.get("vllm.general_plugins", ())
        installed = any(
            ep.name == _CONFUCIUS_VLLM_PLUGIN_NAME
            and ep.value == _CONFUCIUS_VLLM_PLUGIN_REF
            for ep in plugin_eps
        )
    except Exception:
        installed = False

    if not installed:
        raise RuntimeError(
            "Confucius4-TTS vLLM mode uses spawned vLLM engine workers. "
            "Install this repository first so vLLM can load the custom model "
            "plugin in child processes: run `pip install -e .` or "
            "`pip install -r requirements-vllm.txt` from the repository root."
        )

    enabled_plugins = os.environ.get("VLLM_PLUGINS")
    if enabled_plugins:
        names = {
            name.strip()
            for name in enabled_plugins.replace(";", ",").split(",")
            if name.strip()
        }
        if _CONFUCIUS_VLLM_PLUGIN_NAME not in names:
            os.environ["VLLM_PLUGINS"] = (
                f"{enabled_plugins},{_CONFUCIUS_VLLM_PLUGIN_NAME}"
            )


def _filter_kwargs(callable_obj: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    params = inspect.signature(callable_obj).parameters
    if any(param.kind == param.VAR_KEYWORD for param in params.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in params}
