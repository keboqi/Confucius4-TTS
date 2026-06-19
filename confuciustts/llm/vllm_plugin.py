"""vLLM plugin entry point for the Confucius Text2Semantic adapter."""

from __future__ import annotations


def register() -> None:
    from confuciustts.llm.vllm_patch import register_confucius_vllm_model

    register_confucius_vllm_model()
