import argparse
import time
from pathlib import Path

import torch
import torchaudio

from confuciustts.cli.inference import ConfuciusTTS


def _parse_duration_csv(value: str | None) -> list[float] | None:
    if value is None or not value.strip():
        return None
    durations = [
        float(part.strip())
        for part in value.replace("\n", ",").split(",")
        if part.strip()
    ]
    if any(duration <= 0 for duration in durations):
        raise ValueError("--target-segment-durations values must be positive seconds.")
    return durations or None


def parse_args():
    p = argparse.ArgumentParser(
        description="ConfuciusTTS zero-shot TTS inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--text", type=str, required=True,
                   help="Input text to synthesize")
    p.add_argument("--lang", type=str, required=True,
                   help="Language code (e.g., zh, en, ja, ko)")
    p.add_argument("--prompt-wav", type=str, required=True,
                   help="Path to reference audio for voice cloning")
    p.add_argument("--output", type=str, required=True,
                   help="Output audio file path")
    p.add_argument("--config", type=str, default="config/inference_config.yaml",
                   help="Path to inference configuration YAML")
    p.add_argument("--t2s-checkpoint", type=str, default=None,
                   help="Override T2S checkpoint path from config")
    p.add_argument("--device", type=str, default="cuda",
                   help="Device for inference (cuda or cpu)")
    p.add_argument("--use-vllm", action=argparse.BooleanOptionalAction, default=None,
                   help="Use vLLM for the autoregressive T2S semantic decoder. Defaults to auto on CUDA when the converted model exists")
    p.add_argument("--vllm-model-dir", type=str, default=None,
                   help="Converted T2S vLLM directory from tools/convert_t2s_vllm.py")
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.25,
                   help="vLLM GPU memory utilization")
    p.add_argument("--vllm-tensor-parallel-size", type=int, default=1,
                   help="vLLM tensor parallel size")
    p.add_argument("--vllm-dtype", type=str, default="auto",
                   help="vLLM dtype argument")
    p.add_argument("--vllm-attention-backend", type=str, default=None,
                   help="Optional vLLM attention backend override, e.g. FLASHINFER or FLASH_ATTN")
    p.add_argument("--vllm-prefix-mode", default="auto",
                   choices=["auto", "embeds", "worker"],
                   help="How vLLM receives the T2S prefix. worker requires a freshly converted vLLM model")
    p.add_argument("--vllm-latent-mode", default="auto",
                   choices=["auto", "vllm", "pytorch"],
                   help="How vLLM mode obtains T2S latents for S2A")
    p.add_argument("--vllm-hidden-states-dir", default=None,
                   help="Directory for temporary vLLM hidden-state extraction files")
    p.add_argument("--compile-s2a", "--use-torch-compile", "--use_torch_compile",
                   action=argparse.BooleanOptionalAction, default=None, dest="compile_s2a",
                   help="Compile the S2A diffusion estimator with torch.compile. Defaults to enabled on CUDA")
    p.add_argument("--nvfp4", action=argparse.BooleanOptionalAction, default=False,
                   help="Selectively quantize S2A DiT attention/FFN linears to Blackwell NVFP4")
    p.add_argument("--use-bigvgan-cuda-kernel", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="Use BigVGAN's fused CUDA activation kernel. Defaults to enabled on CUDA")
    p.add_argument("--s2a-dtype", default="auto",
                   choices=["auto", "float32", "bfloat16", "float16", "bf16", "fp16", "fp32"],
                   help="S2A inference dtype")
    p.add_argument("--s2a-sdpa-backend", default="auto",
                   choices=["auto", "flash", "efficient", "math", "cudnn"],
                   help="Optional PyTorch SDPA backend override for S2A DiT attention")
    p.add_argument("--s2a-length-bucket-size", type=int, default=64,
                   help="Bucket multi-segment S2A batches by total mel length. 0 disables bucketing")
    p.add_argument("--profile-cuda", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Write torch profiler traces for S2A and BigVGAN stages")
    p.add_argument("--profile-dir", default="outputs/profiles",
                   help="Directory for CUDA profiler traces")
    p.add_argument("--gpu-stage-concurrency", type=int, default=1,
                   help="Maximum concurrent non-vLLM GPU stages per process")
    p.add_argument("--reference-cache-size", type=int, default=16,
                   help="Number of reference-audio conditioning entries to cache per process")
    p.add_argument("--temperature", type=float, default=0.8,
                   help="Sampling temperature for T2S generation (higher = more diverse)")
    p.add_argument("--top-p", type=float, default=0.8,
                   help="Nucleus sampling probability threshold")
    p.add_argument("--top-k", type=int, default=30,
                   help="Top-k sampling parameter")
    p.add_argument("--num-beams", type=int, default=3,
                   help="Beam search width (1 = greedy decoding)")
    p.add_argument("--repetition-penalty", type=float, default=10.0,
                   help="Penalty for repeating tokens (higher = less repetition)")
    p.add_argument("--max-length", type=int, default=1520,
                   help="Maximum semantic token sequence length")
    p.add_argument("--diffusion-steps", type=int, default=10,
                   help="Number of diffusion steps for S2A (more = higher quality, slower)")
    p.add_argument("--cfg-strength", type=float, default=0.7,
                   help="Classifier-free guidance scale (higher = stronger conditioning)")
    p.add_argument("--max-text-tokens", type=int, default=80,
                   help="Maximum tokenizer tokens per text segment")
    p.add_argument("--segment-render-batch-size", type=int, default=1,
                   help="Number of vLLM text segments to batch for S2A/vocoder rendering")
    p.add_argument("--target-duration-seconds", type=float, default=0.0,
                   help="Optional final output duration target in seconds, sample-fitted for millisecond precision. 0 disables duration control")
    p.add_argument("--target-segment-durations", type=str, default="",
                   help="Comma-separated per-segment duration targets in seconds, sample-fitted for millisecond precision")
    p.add_argument("--cross-fade-duration", type=float, default=0.3,
                   help="Cross-fade duration between generated segments")
    p.add_argument("--verbose", action="store_true",
                   help="Print processing information")
    return p.parse_args()


def main():
    args = parse_args()
    model = ConfuciusTTS(
        config_path=args.config,
        t2s_checkpoint=args.t2s_checkpoint,
        device=args.device,
        use_vllm=args.use_vllm,
        vllm_model_dir=args.vllm_model_dir,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_dtype=args.vllm_dtype,
        vllm_attention_backend=args.vllm_attention_backend,
        vllm_prefix_mode=args.vllm_prefix_mode,
        vllm_latent_mode=args.vllm_latent_mode,
        vllm_hidden_states_dir=args.vllm_hidden_states_dir,
        compile_s2a=args.compile_s2a,
        nvfp4=args.nvfp4,
        use_cuda_kernel=args.use_bigvgan_cuda_kernel,
        s2a_dtype=args.s2a_dtype,
        s2a_sdpa_backend=args.s2a_sdpa_backend,
        s2a_length_bucket_size=args.s2a_length_bucket_size,
        profile_cuda=args.profile_cuda,
        profile_dir=args.profile_dir,
        gpu_stage_concurrency=args.gpu_stage_concurrency,
        reference_cache_size=args.reference_cache_size,
    )
    t0 = time.time()
    audio = model.generate(
        text=args.text,
        lang=args.lang,
        prompt_wav=args.prompt_wav,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
        max_length=args.max_length,
        n_timesteps=args.diffusion_steps,
        inference_cfg_rate=args.cfg_strength,
        max_text_tokens_per_segment=args.max_text_tokens,
        segment_render_batch_size=args.segment_render_batch_size,
        target_duration_seconds=args.target_duration_seconds,
        target_segment_durations=_parse_duration_csv(args.target_segment_durations),
        cross_fade_duration=args.cross_fade_duration,
        verbose=args.verbose,
    )
    dt = time.time() - t0

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out), audio.cpu(), model.sample_rate)
    print(f"Saved {out} ({audio.shape[-1] / model.sample_rate:.3f}s, took {dt:.2f}s)")


if __name__ == "__main__":
    main()
