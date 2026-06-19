"""Registration and position handling for the Confucius vLLM adapter."""

from __future__ import annotations

from typing import Any


def register_confucius_vllm_model() -> None:
    """Register the custom T2S model and patch vLLM positions once."""
    from vllm import ModelRegistry

    from confuciustts.llm.vllm_model import ConfuciusText2SemanticForCausalLM

    try:
        ModelRegistry.register_model(
            "ConfuciusText2SemanticForCausalLM",
            ConfuciusText2SemanticForCausalLM,
        )
    except Exception as exc:
        if "already" not in str(exc).lower():
            raise

    _patch_gpu_model_runner_positions(ConfuciusText2SemanticForCausalLM)


def _patch_gpu_model_runner_positions(model_cls: type[Any]) -> None:
    """Make vLLM semantic positions start at BOS instead of prompt index 0.

    vLLM positions count the entire prompt, but Confucius's semantic position
    embedding must count only the semantic stream. The prompt passed to vLLM is
    [condition, text, BOS] as embeddings, so subtract prompt_len - 1. This makes
    condition/text positions negative, BOS position 0, and generated tokens 1..N.
    """
    import numpy as np
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    current_prepare = GPUModelRunner._prepare_inputs
    if getattr(current_prepare, "_confucius_position_patch", False):
        return

    def _prepare_inputs_with_confucius_positions(
        self,
        scheduler_output,
        num_scheduled_tokens,
        *args,
        **kwargs,
    ):
        result = current_prepare(
            self,
            scheduler_output,
            num_scheduled_tokens,
            *args,
            **kwargs,
        )

        model = self.get_model()
        if not isinstance(model, model_cls):
            return result

        total = scheduler_output.total_num_scheduled_tokens
        num_reqs = self.input_batch.num_reqs
        if total <= 0 or num_reqs <= 0:
            return result

        req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)
        offsets = np.zeros(num_reqs, dtype=np.int64)
        req_ids = list(self.input_batch.req_ids[:num_reqs])

        for index, req_id in enumerate(req_ids):
            request = self.requests[req_id]
            num_prompt_tokens = getattr(self.input_batch, "num_prompt_tokens", None)
            if num_prompt_tokens is not None:
                prompt_len = int(num_prompt_tokens[index])
            else:
                prompt_token_ids = getattr(request, "prompt_token_ids", ())
                prompt_len = len(prompt_token_ids)
            offsets[index] = -(prompt_len - 1)

        positions_np = self.positions.np[:total]
        np.add(positions_np, offsets[req_indices], out=positions_np)
        self.positions.copy_to_gpu(total)
        return result

    _prepare_inputs_with_confucius_positions._confucius_position_patch = True
    GPUModelRunner._prepare_inputs = _prepare_inputs_with_confucius_positions
