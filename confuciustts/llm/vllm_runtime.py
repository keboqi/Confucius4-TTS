"""Runtime wrapper for vLLM-backed Confucius Text2Semantic generation."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import os
import sys
import tempfile
import threading
import traceback
import uuid
import warnings
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

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
        dtype: str = "float32",
        attention_backend: Optional[str] = None,
        max_num_seqs: Optional[int] = None,
        max_model_len: Optional[int] = None,
        prefix_mode: str = "auto",
        latent_mode: str = "auto",
        hidden_states_dir: Optional[str] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        _prepend_repo_to_pythonpath_for_workers()
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
        self.generate_timeout_seconds = _env_float(
            "CONFUCIUS_VLLM_GENERATE_TIMEOUT_SECONDS",
            120.0,
        )
        self._model_config = _load_model_config(self.model_dir)
        self.prefix_mode = self._resolve_prefix_mode(prefix_mode)
        self.latent_mode = "pytorch"
        self._latent_mode_requested = (latent_mode or "auto").strip().lower()
        self._hidden_states_connector = None
        self._hidden_states_dir = Path(
            hidden_states_dir
            or Path(tempfile.gettempdir()) / "confucius4tts-vllm-hidden-states"
        )
        self.placeholder_token_id = int(
            getattr(torch_model.config, "start_semantic_token", PLACEHOLDER_TOKEN_ID)
        )

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
        self.latent_mode = self._configure_vllm_latent_extraction(
            args,
            latent_mode,
        )
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
    def requires_torch_model_on_device(self) -> bool:
        return self.prefix_mode == "embeds" or self.latent_mode == "pytorch"

    @property
    def device(self) -> torch.device:
        return next(self.torch_model.parameters()).device

    def _resolve_prefix_mode(self, requested: str) -> str:
        normalized = (requested or "auto").strip().lower()
        if normalized not in {"auto", "embeds", "worker"}:
            raise ValueError("vLLM prefix mode must be one of: auto, embeds, worker.")
        supports_worker = bool(
            self._model_config.get("confucius_supports_worker_prefix", False)
        )
        if normalized == "auto":
            return "worker" if supports_worker else "embeds"
        if normalized == "worker" and not supports_worker:
            raise RuntimeError(
                "The converted T2S vLLM model does not include worker-side prefix "
                "weights. Re-run tools/convert_t2s_vllm.py, or use "
                "--vllm-prefix-mode embeds."
            )
        return normalized

    def _configure_vllm_latent_extraction(
        self,
        engine_args: Dict[str, Any],
        requested: str,
    ) -> str:
        normalized = (requested or "auto").strip().lower()
        if normalized not in {"auto", "vllm", "pytorch"}:
            raise ValueError("vLLM latent mode must be one of: auto, vllm, pytorch.")
        if normalized == "pytorch":
            return "pytorch"
        try:
            from vllm.config.kv_transfer import KVTransferConfig
            from vllm.distributed.kv_transfer.kv_connector.v1 import (
                example_hidden_states_connector,
            )
        except Exception as exc:
            if normalized == "vllm":
                raise RuntimeError(
                    "This vLLM build does not expose hidden-state extraction APIs."
                ) from exc
            warnings.warn(
                "This vLLM build does not expose hidden-state extraction APIs; "
                "falling back to the PyTorch latent pass.",
                RuntimeWarning,
                stacklevel=2,
            )
            return "pytorch"

        self._hidden_states_dir.mkdir(parents=True, exist_ok=True)
        num_layers = self._t2s_num_layers()
        engine_args["speculative_config"] = {
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {
                    "eagle_aux_hidden_state_layer_ids": [num_layers],
                },
            },
        }
        engine_args["kv_transfer_config"] = KVTransferConfig(
            kv_connector="ExampleHiddenStatesConnector",
            kv_role="kv_producer",
            kv_connector_extra_config={
                "shared_storage_path": str(self._hidden_states_dir),
                "allow_custom_save_path": True,
                "use_synchronization_lock": True,
            },
        )
        engine_args.setdefault("enable_chunked_prefill", False)
        self._hidden_states_connector = example_hidden_states_connector
        return "vllm"

    def _acoustic_semantic_vocab_size(self) -> int:
        return int(
            getattr(
                self.torch_model.config,
                "start_semantic_token",
                getattr(self.torch_model.config, "semantic_vocab_size", 8194) - 2,
            )
        )

    def _text_position_limit(self) -> Optional[int]:
        embedding = getattr(self.torch_model.text_position_embedding, "embedding", None)
        if embedding is not None:
            return int(embedding.num_embeddings)
        return getattr(self.torch_model.config, "max_text_seq_lens", None)

    def _semantic_position_limit(self) -> Optional[int]:
        embedding = getattr(
            self.torch_model.semantic_position_embedding,
            "embedding",
            None,
        )
        if embedding is not None:
            return int(embedding.num_embeddings)
        return getattr(self.torch_model.config, "max_semantic_seq_lens", None)

    def _t2s_num_layers(self) -> int:
        return int(
            getattr(
                self.torch_model.config,
                "num_layers",
                getattr(self.torch_model.config, "n_layer", 24),
            )
        )

    def _prefix_token_count(self, text_inputs: torch.Tensor) -> int:
        return 1 + int(text_inputs.shape[1]) + 1

    def _sanitize_generated_token_ids(
        self,
        token_ids: Sequence[int],
        eos_token_id: int,
    ) -> list[int]:
        acoustic_vocab_size = self._acoustic_semantic_vocab_size()
        bos_token_id = int(self.torch_model.config.start_semantic_token)
        cleaned: list[int] = []
        dropped: list[int] = []

        for token in token_ids:
            token = int(token)
            if token == eos_token_id:
                break
            if 0 <= token < acoustic_vocab_size:
                cleaned.append(token)
            elif token == bos_token_id:
                dropped.append(token)
            else:
                dropped.append(token)

        if dropped:
            warnings.warn(
                "vLLM generated non-acoustic semantic token IDs and they were "
                f"removed before S2A: {sorted(set(dropped))[:8]}",
                RuntimeWarning,
                stacklevel=2,
            )
        if not cleaned:
            raise RuntimeError(
                "vLLM generated no valid acoustic semantic tokens. "
                f"Raw token IDs started with: {list(token_ids)[:16]}"
            )
        return cleaned

    def _validate_text_token_ids(self, text_inputs: torch.Tensor) -> None:
        max_len = self._text_position_limit()
        if max_len is not None and text_inputs.shape[1] > max_len:
            raise ValueError(
                "Text prompt exceeds the T2S text position limit before vLLM "
                f"prefix construction: token_count={text_inputs.shape[1]}, "
                f"max_text_seq_lens={max_len}. Split the input text into "
                "smaller segments."
            )
        vocab_size = int(self.torch_model.text_projector.embed.num_embeddings)
        invalid = (text_inputs < 0) | (text_inputs >= vocab_size)
        if not invalid.any():
            return
        bad_ids = text_inputs[invalid].detach().cpu().unique().tolist()
        raise ValueError(
            "Text token IDs exceed the T2S text embedding vocabulary before "
            f"vLLM prefix construction: vocab_size={vocab_size}, "
            f"invalid_ids={bad_ids[:16]}"
        )

    @torch.no_grad()
    def _build_prefix_embeddings(
        self,
        text_inputs: torch.Tensor,
        condition_vector: torch.Tensor,
    ) -> torch.Tensor:
        if text_inputs.shape[0] != 1:
            raise ValueError("The vLLM T2S wrapper currently expects batch size 1.")
        self._validate_text_token_ids(text_inputs)

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

    def _build_prompt(
        self,
        text_inputs: torch.Tensor,
        condition_vector: torch.Tensor,
    ) -> tuple[Any, int]:
        if self.prefix_mode == "worker":
            self._validate_text_token_ids(text_inputs)
            prefix_len = self._prefix_token_count(text_inputs)
            prompt_kwargs = {
                "multi_modal_data": {
                    "audio": {
                        "text_inputs": [text_inputs.squeeze(0).detach().cpu()],
                        "condition_vector": [condition_vector.squeeze(0).detach().cpu()],
                    }
                }
            }
        else:
            prefix_embeds = self._build_prefix_embeddings(
                text_inputs,
                condition_vector,
            )
            prefix_len = int(prefix_embeds.shape[1])
            prompt_kwargs = {
                "multi_modal_data": {
                    "audio": {
                        "audio_embeds": [prefix_embeds.squeeze(0).detach().cpu()],
                    }
                }
            }

        try:
            prompt = self.TokensPrompt(
                prompt_token_ids=[self.placeholder_token_id],
                **prompt_kwargs,
            )
        except TypeError:
            prompt = self.TokensPrompt(prompt=PLACEHOLDER_TOKEN, **prompt_kwargs)
        return prompt, prefix_len

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
        extra_args: Optional[Dict[str, Any]] = None,
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
        if extra_args:
            if "extra_args" not in inspect.signature(self.SamplingParams).parameters:
                raise RuntimeError(
                    "This vLLM SamplingParams does not support extra_args; "
                    "cannot request hidden-state latent extraction."
                )
            kwargs["extra_args"] = extra_args

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
        prompt, prefix_len = self._build_prompt(
            text_inputs,
            condition_vector,
        )
        max_tokens = max(1, int(max_length) - prefix_len)
        semantic_limit = self._semantic_position_limit()
        if semantic_limit is not None:
            max_tokens = min(max_tokens, max(1, semantic_limit - 2))
        request_id = uuid.uuid4().hex
        hidden_states_path = (
            self._hidden_states_dir / f"{request_id}.safetensors"
            if return_latent and self.latent_mode == "vllm"
            else None
        )
        extra_args = None
        if hidden_states_path is not None:
            extra_args = {
                "kv_transfer_params": {
                    "hidden_states_path": str(hidden_states_path),
                    "include_output_tokens": True,
                }
            }
        try:
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
                extra_args=extra_args,
            )
        except RuntimeError:
            if self._latent_mode_requested == "vllm" or extra_args is None:
                raise
            warnings.warn(
                "This vLLM SamplingParams does not support hidden-state "
                "per-request options; falling back to the PyTorch latent pass.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.latent_mode = "pytorch"
            hidden_states_path = None
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
                extra_args=None,
            )

        output_generator = self.llm.generate(
            prompt,
            sampling_params=sampling_params,
            request_id=request_id,
        )

        final_output = None
        started = asyncio.get_running_loop().time()
        print(
            "[ConfuciusTTS] Submitted vLLM T2S request "
            f"id={request_id} max_tokens={max_tokens} prefix_len={prefix_len} "
            f"timeout={self.generate_timeout_seconds or 'disabled'}s",
            flush=True,
        )
        try:
            while True:
                try:
                    if self.generate_timeout_seconds and self.generate_timeout_seconds > 0:
                        output = await asyncio.wait_for(
                            output_generator.__anext__(),
                            timeout=float(self.generate_timeout_seconds),
                        )
                    else:
                        output = await output_generator.__anext__()
                except StopAsyncIteration:
                    break
                final_output = output
        except asyncio.TimeoutError as exc:
            await self._abort_request(request_id)
            raise TimeoutError(
                "vLLM T2S generation produced no output before timeout: "
                f"request_id={request_id}, "
                f"timeout={self.generate_timeout_seconds:.1f}s, "
                f"max_tokens={max_tokens}, prefix_len={prefix_len}. "
                "Set CONFUCIUS_VLLM_GENERATE_TIMEOUT_SECONDS=0 to disable this guard."
            ) from exc
        if final_output is None:
            raise RuntimeError("vLLM returned no output for Confucius T2S generation.")
        print(
            "[ConfuciusTTS] Completed vLLM T2S request "
            f"id={request_id} in {asyncio.get_running_loop().time() - started:.2f}s "
            f"tokens={len(final_output.outputs[0].token_ids)}",
            flush=True,
        )

        raw_token_ids = list(final_output.outputs[0].token_ids)
        if eos in raw_token_ids:
            raw_token_ids = raw_token_ids[: raw_token_ids.index(eos)]
        token_ids = self._sanitize_generated_token_ids(raw_token_ids, eos)

        semantic_codes = torch.tensor(
            token_ids,
            dtype=torch.long,
            device=text_inputs.device,
        ).unsqueeze(0)

        if not return_latent:
            return semantic_codes

        latent = None
        if self.latent_mode == "vllm":
            latent = self._extract_vllm_latent(
                final_output=final_output,
                hidden_states_path=hidden_states_path,
                prefix_len=prefix_len,
                raw_token_ids=raw_token_ids,
                token_ids=token_ids,
                device=text_inputs.device,
            )
        if latent is None:
            if self._latent_mode_requested == "vllm":
                raise RuntimeError(
                    "vLLM latent extraction was explicitly requested, but no "
                    "hidden-state latent tensor was returned."
                )
            latent = self._compute_latent(text_inputs, condition_vector, semantic_codes)
        return {"semantic_codes": semantic_codes, "latent": latent}

    async def async_generate_many(
        self,
        requests: Sequence[Dict[str, Any]],
    ) -> list[Any]:
        if not requests:
            return []
        tasks = [
            asyncio.create_task(self.async_generate(**request))
            for request in requests
        ]
        return await asyncio.gather(*tasks)

    def generate(self, *args: Any, **kwargs: Any):
        if self._loop is None:
            self._loop = _BackgroundLoop()
        return self._loop.run(self.async_generate(*args, **kwargs))

    def generate_many(self, requests: Sequence[Dict[str, Any]]) -> list[Any]:
        if self._loop is None:
            self._loop = _BackgroundLoop()
        return self._loop.run(self.async_generate_many(requests))

    async def _abort_request(self, request_id: str) -> None:
        for method_name in ("abort", "abort_request"):
            method = getattr(self.llm, method_name, None)
            if method is None:
                continue
            with contextlib.suppress(Exception):
                result = method(request_id)
                if inspect.isawaitable(result):
                    await result
                return

    @torch.no_grad()
    def _compute_latent(
        self,
        text_inputs: torch.Tensor,
        condition_vector: torch.Tensor,
        semantic_codes: torch.Tensor,
    ) -> torch.Tensor:
        model = self.torch_model
        device = text_inputs.device
        if next(model.parameters()).device != device:
            model.to(device)
        batch_size = text_inputs.shape[0]
        semantic_limit = self._semantic_position_limit()
        if semantic_limit is not None and semantic_codes.shape[1] + 2 > semantic_limit:
            raise ValueError(
                "Generated semantic sequence exceeds the T2S semantic position "
                f"limit before latent computation: token_count={semantic_codes.shape[1]}, "
                f"bounded_token_count={semantic_codes.shape[1] + 2}, "
                f"max_semantic_seq_lens={semantic_limit}."
            )
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

    def _normalize_vllm_hidden_states(
        self,
        hidden_states: torch.Tensor,
        min_tokens: int,
    ) -> torch.Tensor:
        while hidden_states.dim() > 2 and hidden_states.shape[0] == 1:
            hidden_states = hidden_states.squeeze(0)
        if hidden_states.dim() == 2:
            return hidden_states

        num_layers = self._t2s_num_layers()
        if hidden_states.dim() == 3:
            if hidden_states.shape[0] >= min_tokens and hidden_states.shape[1] <= num_layers + 1:
                return hidden_states[:, -1, :]
            if hidden_states.shape[1] >= min_tokens and hidden_states.shape[0] <= num_layers + 1:
                return hidden_states[-1, :, :]
            if hidden_states.shape[1] >= min_tokens:
                return hidden_states[0, :, :]

        raise RuntimeError(
            "Unsupported vLLM hidden-state tensor layout: "
            f"shape={tuple(hidden_states.shape)}, min_tokens={min_tokens}, "
            f"num_layers={num_layers}"
        )

    def _extract_vllm_latent(
        self,
        *,
        final_output: Any,
        hidden_states_path: Optional[Path],
        prefix_len: int,
        raw_token_ids: Sequence[int],
        token_ids: Sequence[int],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if self._hidden_states_connector is None:
            return None
        if list(raw_token_ids) != list(token_ids):
            warnings.warn(
                "vLLM hidden-state latent extraction skipped because generated "
                "tokens required sanitization; falling back to PyTorch latent pass.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        output_params = getattr(final_output, "kv_transfer_params", None) or {}
        resolved_path = output_params.get("hidden_states_path")
        path = Path(resolved_path) if resolved_path else hidden_states_path
        if path is None:
            return None
        try:
            obj = self._hidden_states_connector.load_hidden_states(str(path))
            hidden_states = obj["hidden_states"]
            start = prefix_len - 1
            end = start + len(token_ids)
            hidden_states = self._normalize_vllm_hidden_states(hidden_states, end)
            if end > hidden_states.shape[0]:
                raise RuntimeError(
                    "vLLM hidden-state tensor is shorter than expected: "
                    f"needed end={end}, shape={tuple(hidden_states.shape)}"
                )
            return hidden_states[start:end].to(device=device).unsqueeze(0)
        except Exception:
            traceback.print_exc()
            warnings.warn(
                "Failed to load vLLM hidden states; falling back to PyTorch latent pass.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        finally:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass


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


def _prepend_repo_to_pythonpath_for_workers() -> None:
    repo_root = str(Path(__file__).resolve().parents[2])
    if not sys.path or sys.path[0] != repo_root:
        sys.path.insert(0, repo_root)

    pythonpath = os.environ.get("PYTHONPATH")
    paths = pythonpath.split(os.pathsep) if pythonpath else []
    if not paths or paths[0] != repo_root:
        paths = [path for path in paths if path != repo_root]
        os.environ["PYTHONPATH"] = os.pathsep.join([repo_root, *paths])


def _load_model_config(model_dir: str) -> Dict[str, Any]:
    config_path = Path(model_dir) / "config.json"
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return {}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float.") from exc


def _filter_kwargs(callable_obj: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    params = inspect.signature(callable_obj).parameters
    if any(param.kind == param.VAR_KEYWORD for param in params.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in params}
