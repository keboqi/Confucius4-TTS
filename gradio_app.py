#!/usr/bin/env python3
"""Gradio web UI for Confucius4-TTS zero-shot voice cloning."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Any, Optional

import gradio as gr
import torch
import torchaudio

ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs" / "gradio"
SERVE_MODEL: Any = None
SERVE_DEVICE = "cuda"
SERVE_CONFIG_PATH: Optional[str] = None
SERVE_T2S_CHECKPOINT: Optional[str] = None
SERVE_COMPILE_S2A: Optional[bool] = None
SERVE_USE_BIGVGAN_CUDA_KERNEL: Optional[bool] = None

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


def _resolve_repo_dir(value: str, label: str, must_exist: bool = True) -> str:
    value = value.strip()
    if not value:
        raise gr.Error(f"{label} is required.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT_DIR / path
    path = path.resolve()
    if must_exist and not path.exists():
        raise gr.Error(f"{label} does not exist: {path}")
    if must_exist and not path.is_dir():
        raise gr.Error(f"{label} must be a directory: {path}")
    return str(path)


def _resolve_device(device_choice: str, probe_cuda: bool = True) -> str:
    if device_choice == "auto":
        if torch.cuda.is_available():
            if probe_cuda:
                _ensure_cuda_usable()
            return "cuda"
        return "cpu"
    if device_choice == "cuda" and not torch.cuda.is_available():
        raise gr.Error("CUDA was selected, but torch.cuda.is_available() is false.")
    if device_choice == "cuda" and probe_cuda:
        _ensure_cuda_usable()
    return device_choice


def _ensure_cuda_usable() -> None:
    try:
        device_name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        _ = torch.ones(1, device="cuda") + 1
        torch.cuda.synchronize()
    except Exception as exc:
        raise gr.Error(
            "CUDA is visible, but this PyTorch build cannot run kernels on the GPU. "
            f"Detected GPU: {locals().get('device_name', 'unknown')} "
            f"(sm_{locals().get('major', '?')}{locals().get('minor', '?')}). "
            "For Blackwell GPUs such as RTX PRO 6000, reinstall PyTorch with the CUDA 12.8 wheels: "
            "`pip install --force-reinstall -r requirements-cu128.txt`. "
            "If stable CUDA 12.8 wheels still fail, install the PyTorch nightly CUDA 12.8 build."
        ) from exc


def _normalize_optional_text(value: str) -> Optional[str]:
    value = value.strip()
    return value or None


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of: 1, true, yes, on, 0, false, no, off.")


def _env_optional_bool(*names: str) -> Optional[bool]:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return _env_bool(name)
    return None


def _normalize_checkpoint(value: str) -> Optional[str]:
    return _normalize_optional_text(value)


def _load_serving_model(
    config_path: str,
    t2s_checkpoint: Optional[str],
    device: str,
    vllm_model_dir: str,
    vllm_gpu_memory_utilization: float,
    vllm_tensor_parallel_size: int,
    vllm_dtype: str,
    vllm_attention_backend: Optional[str],
    compile_s2a: Optional[bool],
    use_bigvgan_cuda_kernel: Optional[bool],
) -> Any:
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
        use_vllm=True,
        vllm_model_dir=vllm_model_dir,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=vllm_tensor_parallel_size,
        vllm_dtype=vllm_dtype,
        vllm_attention_backend=vllm_attention_backend,
        compile_s2a=compile_s2a,
        use_cuda_kernel=use_bigvgan_cuda_kernel,
    )


def _vllm_model() -> Any:
    if SERVE_MODEL is None:
        raise gr.Error("The vLLM TTS model is not loaded.")
    return SERVE_MODEL


def _output_path(prompt_wav: str, backend: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(f"{prompt_wav}-{time.time_ns()}".encode("utf-8")).hexdigest()[:8]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return OUTPUT_DIR / f"confucius4tts-{backend}-{stamp}-{digest}.wav"


def _format_timing_status(
    output: Path,
    audio: torch.Tensor,
    sample_rate: int,
    timing_info: dict[str, Any],
    save_elapsed: float,
    request_elapsed: float,
) -> str:
    generated_seconds = audio.shape[-1] / sample_rate
    lines = [
        (
            f"{timing_info['backend']} generated {generated_seconds:.2f}s "
            f"of audio in {request_elapsed:.2f}s on {SERVE_DEVICE}."
        ),
        (
            f"segments={timing_info['segments']}, "
            f"semantic_tokens={timing_info['semantic_tokens']}, "
            f"saved={output}"
        ),
        "",
        "Timings:",
    ]
    target_duration = timing_info.get("target_duration_seconds")
    if target_duration is not None and target_duration > 0:
        lines.insert(
            2,
            (
                f"target_duration={target_duration:.3f}s, "
                f"delta={(generated_seconds - target_duration) * 1000:+.1f}ms"
            ),
        )
    for name, seconds in timing_info["steps"].items():
        lines.append(f"{name}: {seconds:.3f}s")
    lines.append(f"model_total: {timing_info['total']:.3f}s")
    lines.append(f"save_wav: {save_elapsed:.3f}s")
    lines.append(f"request_total: {request_elapsed:.3f}s")
    return "\n".join(lines)


def _original_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    paths = pythonpath.split(os.pathsep) if pythonpath else []
    root = str(ROOT_DIR)
    if root not in paths:
        env["PYTHONPATH"] = os.pathsep.join([root, *paths])
    return env


def _format_subprocess_error(process: subprocess.CompletedProcess[str]) -> str:
    parts = [
        f"Original PyTorch generation failed with exit code {process.returncode}."
    ]
    if process.stdout.strip():
        parts.extend(["", "stdout:", process.stdout.strip()])
    if process.stderr.strip():
        parts.extend(["", "stderr:", process.stderr.strip()])
    return "\n".join(parts)


def _parse_duration_csv(value: str) -> Optional[list[float]]:
    value = (value or "").strip()
    if not value:
        return None
    try:
        durations = [
            float(part.strip())
            for part in value.replace("\n", ",").split(",")
            if part.strip()
        ]
    except ValueError as exc:
        raise gr.Error("Segment durations must be comma-separated numbers.") from exc
    if any(duration <= 0 for duration in durations):
        raise gr.Error("Segment durations must be positive seconds.")
    return durations or None


def _synthesize_original_subprocess(
    prompt_wav: str,
    text: str,
    lang: str,
    temperature: float,
    top_p: float,
    top_k: int,
    num_beams: int,
    repetition_penalty: float,
    max_length: int,
    diffusion_steps: int,
    cfg_strength: float,
    max_text_tokens: int,
    target_duration_seconds: float,
    target_segment_durations: str,
    cross_fade_duration: float,
    verbose: bool,
) -> tuple[str, str, str]:
    if SERVE_CONFIG_PATH is None:
        raise gr.Error("The original PyTorch TTS model is not configured.")

    output = _output_path(prompt_wav, "pytorch")
    command = [
        sys.executable,
        "-m",
        "confuciustts.cli.run_inference",
        "--text",
        text,
        "--lang",
        lang,
        "--prompt-wav",
        prompt_wav,
        "--output",
        str(output),
        "--config",
        SERVE_CONFIG_PATH,
        "--device",
        SERVE_DEVICE,
        "--temperature",
        str(float(temperature)),
        "--top-p",
        str(float(top_p)),
        "--top-k",
        str(int(top_k)),
        "--num-beams",
        str(int(num_beams)),
        "--repetition-penalty",
        str(float(repetition_penalty)),
        "--max-length",
        str(int(max_length)),
        "--diffusion-steps",
        str(int(diffusion_steps)),
        "--cfg-strength",
        str(float(cfg_strength)),
        "--max-text-tokens",
        str(int(max_text_tokens)),
        "--target-duration-seconds",
        str(float(target_duration_seconds or 0.0)),
        "--cross-fade-duration",
        str(float(cross_fade_duration)),
    ]
    target_segment_durations = (target_segment_durations or "").strip()
    if target_segment_durations:
        command.extend(["--target-segment-durations", target_segment_durations])
    if SERVE_T2S_CHECKPOINT is not None:
        command.extend(["--t2s-checkpoint", SERVE_T2S_CHECKPOINT])
    if SERVE_COMPILE_S2A is True:
        command.append("--compile-s2a")
    elif SERVE_COMPILE_S2A is False:
        command.append("--no-compile-s2a")
    if SERVE_USE_BIGVGAN_CUDA_KERNEL is True:
        command.append("--use-bigvgan-cuda-kernel")
    elif SERVE_USE_BIGVGAN_CUDA_KERNEL is False:
        command.append("--no-use-bigvgan-cuda-kernel")
    if verbose:
        command.append("--verbose")

    started = time.perf_counter()
    process = subprocess.run(
        command,
        cwd=str(ROOT_DIR),
        env=_original_subprocess_env(),
        capture_output=True,
        text=True,
    )
    request_elapsed = time.perf_counter() - started
    if process.returncode != 0:
        raise gr.Error(_format_subprocess_error(process))
    if not output.exists():
        raise gr.Error(
            "Original PyTorch generation finished without creating the output file."
        )

    with wave.open(str(output), "rb") as wav_file:
        generated_seconds = wav_file.getnframes() / wav_file.getframerate()
    status_lines = [
        (
            f"Original PyTorch T2S generated {generated_seconds:.2f}s "
            f"of audio in {request_elapsed:.2f}s on {SERVE_DEVICE}."
        ),
        f"saved={output}",
        "",
        "Subprocess output:",
        process.stdout.strip() or "(no stdout)",
    ]
    if process.stderr.strip():
        status_lines.extend(["", "Subprocess stderr:", process.stderr.strip()])
    return str(output), str(output), "\n".join(status_lines)


def _synthesize(
    use_vllm: bool,
    prompt_wav: Optional[str],
    text: str,
    language: str,
    temperature: float,
    top_p: float,
    top_k: int,
    num_beams: int,
    repetition_penalty: float,
    max_length: int,
    diffusion_steps: int,
    cfg_strength: float,
    max_text_tokens: int,
    segment_render_batch_size: int,
    target_duration_seconds: float,
    target_segment_durations: str,
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

    lang = _language_code(language)
    target_segment_duration_values = _parse_duration_csv(target_segment_durations)
    if not use_vllm:
        return _synthesize_original_subprocess(
            prompt_wav=prompt_wav,
            text=text,
            lang=lang,
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            num_beams=int(num_beams),
            repetition_penalty=float(repetition_penalty),
            max_length=int(max_length),
            diffusion_steps=int(diffusion_steps),
            cfg_strength=float(cfg_strength),
            max_text_tokens=int(max_text_tokens),
            target_duration_seconds=float(target_duration_seconds or 0.0),
            target_segment_durations=target_segment_durations,
            cross_fade_duration=float(cross_fade_duration),
            verbose=bool(verbose),
        )

    model = _vllm_model()
    backend_slug = "vllm"

    started = time.perf_counter()
    audio, timing_info = model.generate(
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
        segment_render_batch_size=int(segment_render_batch_size),
        target_duration_seconds=float(target_duration_seconds or 0.0),
        target_segment_durations=target_segment_duration_values,
        cross_fade_duration=float(cross_fade_duration),
        verbose=bool(verbose),
        use_vllm=use_vllm,
        return_timings=True,
    )

    output = _output_path(prompt_wav, backend_slug)
    save_started = time.perf_counter()
    if torch.cuda.is_available() and model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    audio_cpu = audio.detach().float().cpu()
    torchaudio.save(str(output), audio_cpu, model.sample_rate)
    save_elapsed = time.perf_counter() - save_started

    status = _format_timing_status(
        output=output,
        audio=audio,
        sample_rate=model.sample_rate,
        timing_info=timing_info,
        save_elapsed=save_elapsed,
        request_elapsed=time.perf_counter() - started,
    )
    return str(output), str(output), status


def synthesize_vllm(*args: Any) -> tuple[str, str, str]:
    return _synthesize(True, *args)


def synthesize_original(*args: Any) -> tuple[str, str, str]:
    return _synthesize(False, *args)


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

            with gr.Column(scale=2):
                text = gr.Textbox(
                    label="Text",
                    value="Hello, welcome to Confucius4-TTS.",
                    lines=6,
                    max_lines=14,
                )
                with gr.Row():
                    generate_vllm_btn = gr.Button("Generate with vLLM", variant="primary")
                    generate_original_btn = gr.Button("Generate original")

        with gr.Accordion("Generation settings", open=False):
            with gr.Row():
                temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="Temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.8, step=0.05, label="Top-p")
                top_k = gr.Slider(1, 100, value=30, step=1, label="Top-k")
            with gr.Row():
                num_beams = gr.Slider(1, 8, value=3, step=1, label="Beams")
                repetition_penalty = gr.Slider(1.0, 20.0, value=10.0, step=0.5, label="Repetition penalty")
                diffusion_steps = gr.Slider(1, 80, value=10, step=1, label="Diffusion steps")
            with gr.Row():
                cfg_strength = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="CFG strength")
                max_length = gr.Slider(128, 2048, value=1520, step=16, label="Max semantic length")
                max_text_tokens = gr.Slider(20, 240, value=80, step=5, label="Segment token limit")
            with gr.Row():
                segment_render_batch_size = gr.Slider(1, 16, value=1, step=1, label="Render batch size")
                target_duration_seconds = gr.Number(
                    value=0.0,
                    precision=3,
                    label="Target duration seconds",
                )
                cross_fade_duration = gr.Slider(0.0, 2.0, value=0.3, step=0.05, label="Cross fade seconds")
            target_segment_durations = gr.Textbox(
                label="Segment durations CSV",
                value="",
                lines=1,
                max_lines=2,
            )

        with gr.Accordion("Server settings", open=False):
            verbose = gr.Checkbox(label="Verbose inference logs", value=False)

        with gr.Row():
            with gr.Column():
                vllm_audio = gr.Audio(label="vLLM audio", type="filepath")
                vllm_file = gr.File(label="Download vLLM WAV")
                vllm_status = gr.Textbox(label="vLLM timing", interactive=False, lines=14)
            with gr.Column():
                original_audio = gr.Audio(label="Original audio", type="filepath")
                original_file = gr.File(label="Download original WAV")
                original_status = gr.Textbox(label="Original timing", interactive=False, lines=14)

        generation_inputs = [
            prompt_wav,
            text,
            language,
            temperature,
            top_p,
            top_k,
            num_beams,
            repetition_penalty,
            max_length,
            diffusion_steps,
            cfg_strength,
            max_text_tokens,
            segment_render_batch_size,
            target_duration_seconds,
            target_segment_durations,
            cross_fade_duration,
            verbose,
        ]

        generate_vllm_btn.click(
            fn=synthesize_vllm,
            inputs=[
                *generation_inputs,
            ],
            outputs=[vllm_audio, vllm_file, vllm_status],
        )
        generate_original_btn.click(
            fn=synthesize_original,
            inputs=[
                *generation_inputs,
            ],
            outputs=[original_audio, original_file, original_status],
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the Confucius4-TTS Gradio UI.")
    parser.add_argument("--server-name", default=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"))
    parser.add_argument("--server-port", type=int, default=int(os.getenv("GRADIO_SERVER_PORT", "7860")))
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--inbrowser", action="store_true")
    parser.add_argument("--root-path", default=os.getenv("GRADIO_ROOT_PATH"))
    parser.add_argument("--config", default=os.getenv("CONFUCIUS_TTS_CONFIG", "config/inference_config.yaml"))
    parser.add_argument("--t2s-checkpoint", default=os.getenv("CONFUCIUS_T2S_CHECKPOINT", ""))
    parser.add_argument("--device", default=os.getenv("CONFUCIUS_TTS_DEVICE", "cuda"),
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--vllm-model-dir", default=os.getenv("CONFUCIUS_T2S_VLLM_DIR", "checkpoints/t2s-vllm"))
    parser.add_argument("--vllm-gpu-memory-utilization", type=float,
                        default=float(os.getenv("CONFUCIUS_VLLM_GPU_MEMORY_UTILIZATION", "0.25")))
    parser.add_argument("--vllm-tensor-parallel-size", type=int,
                        default=int(os.getenv("CONFUCIUS_VLLM_TENSOR_PARALLEL_SIZE", "1")))
    parser.add_argument("--vllm-dtype", default=os.getenv("CONFUCIUS_VLLM_DTYPE", "float32"))
    parser.add_argument("--vllm-attention-backend",
                        default=os.getenv("CONFUCIUS_VLLM_ATTENTION_BACKEND", ""))
    parser.add_argument("--compile-s2a", "--use-torch-compile", "--use_torch_compile",
                        action=argparse.BooleanOptionalAction, dest="compile_s2a",
                        default=_env_optional_bool(
                            "CONFUCIUS_USE_TORCH_COMPILE",
                            "CONFUCIUS_COMPILE_S2A",
                        ),
                        help="Compile the S2A diffusion estimator with torch.compile. Defaults to enabled on CUDA.")
    parser.add_argument("--use-bigvgan-cuda-kernel", action=argparse.BooleanOptionalAction,
                        default=_env_optional_bool(
                            "CONFUCIUS_USE_BIGVGAN_CUDA_KERNEL",
                            "CONFUCIUS_BIGVGAN_USE_CUDA_KERNEL",
                        ),
                        help="Use BigVGAN's fused CUDA activation kernel. Defaults to enabled on CUDA.")
    parser.add_argument("--concurrency-limit", type=int,
                        default=int(os.getenv("GRADIO_CONCURRENCY_LIMIT", "100")))
    return parser.parse_args()


def main() -> None:
    global SERVE_COMPILE_S2A, SERVE_CONFIG_PATH, SERVE_DEVICE, SERVE_MODEL
    global SERVE_T2S_CHECKPOINT, SERVE_USE_BIGVGAN_CUDA_KERNEL

    args = parse_args()
    config_path = _resolve_repo_path(args.config, "Config")
    vllm_model_dir = _resolve_repo_dir(args.vllm_model_dir, "T2S vLLM directory")
    # Avoid creating a CUDA context before the vLLM engine has forked.
    SERVE_DEVICE = _resolve_device(args.device, probe_cuda=False)
    if SERVE_DEVICE != "cuda":
        raise gr.Error(
            "The Gradio serving entry point requires CUDA because it always "
            "uses the vLLM T2S backend."
        )
    if args.concurrency_limit < 1:
        raise gr.Error("--concurrency-limit must be at least 1.")
    t2s_checkpoint = _normalize_checkpoint(args.t2s_checkpoint)
    vllm_attention_backend = _normalize_optional_text(args.vllm_attention_backend)
    SERVE_CONFIG_PATH = config_path
    SERVE_T2S_CHECKPOINT = t2s_checkpoint
    SERVE_COMPILE_S2A = args.compile_s2a
    SERVE_USE_BIGVGAN_CUDA_KERNEL = args.use_bigvgan_cuda_kernel
    compile_s2a_label = "auto" if SERVE_COMPILE_S2A is None else str(SERVE_COMPILE_S2A)
    bigvgan_kernel_label = (
        "auto"
        if SERVE_USE_BIGVGAN_CUDA_KERNEL is None
        else str(SERVE_USE_BIGVGAN_CUDA_KERNEL)
    )

    print(
        "[Confucius4-TTS] Loading always-on vLLM T2S backend "
        f"from {vllm_model_dir} on {SERVE_DEVICE} "
        f"(compile_s2a={compile_s2a_label}, "
        f"use_bigvgan_cuda_kernel={bigvgan_kernel_label})..."
    )
    SERVE_MODEL = _load_serving_model(
        config_path=config_path,
        t2s_checkpoint=t2s_checkpoint,
        device=SERVE_DEVICE,
        vllm_model_dir=vllm_model_dir,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_dtype=args.vllm_dtype,
        vllm_attention_backend=vllm_attention_backend,
        compile_s2a=args.compile_s2a,
        use_bigvgan_cuda_kernel=args.use_bigvgan_cuda_kernel,
    )
    print("[Confucius4-TTS] vLLM-backed TTS model is ready.")

    launch_kwargs = {
        "server_name": args.server_name,
        "server_port": args.server_port,
        "share": args.share,
        "inbrowser": args.inbrowser,
    }
    if args.root_path:
        launch_kwargs["root_path"] = args.root_path

    build_demo().queue(default_concurrency_limit=args.concurrency_limit).launch(**launch_kwargs)


if __name__ == "__main__":
    main()
