#!/usr/bin/env python3
"""FastAPI service for Confucius4-TTS optimized for programmatic clients."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import threading
import time
import traceback
import uuid
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torchaudio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import gradio_app as serving

API_VERSION = "0.1.0"
DEFAULT_PROMPT_WAV = "resources/voice.mp3"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "api"
API_OUTPUT_DIR = DEFAULT_OUTPUT_DIR
REFERENCE_UPLOAD_SUFFIXES = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}
LANGUAGE_CODES = tuple(
    serving._language_code(choice) for choice in serving.LANGUAGE_CHOICES
)


@dataclass(frozen=True)
class ApiSettings:
    config: str
    t2s_checkpoint: str
    device: str
    vllm_model_dir: str
    vllm_gpu_memory_utilization: float
    vllm_tensor_parallel_size: int
    vllm_dtype: str
    vllm_attention_backend: str
    vllm_prefix_mode: str
    vllm_latent_mode: str
    vllm_hidden_states_dir: str
    compile_s2a: Optional[bool]
    use_bigvgan_cuda_kernel: Optional[bool]
    s2a_dtype: str
    s2a_sdpa_backend: str
    s2a_length_bucket_size: int
    compile_cache: bool
    compile_cache_dir: str
    warmup: bool
    warmup_mode: str
    warmup_prompt_wav: str
    warmup_text: str
    warmup_extra_text: str
    warmup_lang: str
    warmup_diffusion_steps: int
    profile_cuda: bool
    profile_dir: str
    gpu_stage_concurrency: int
    reference_cache_size: int
    postprocess_concurrency: int
    detailed_timings: bool
    output_dir: str
    cors_origins: tuple[str, ...]

    @classmethod
    def from_env(cls) -> "ApiSettings":
        return cls(
            config=os.getenv("CONFUCIUS_TTS_CONFIG", "config/inference_config.yaml"),
            t2s_checkpoint=os.getenv("CONFUCIUS_T2S_CHECKPOINT", ""),
            device=os.getenv("CONFUCIUS_TTS_DEVICE", "cuda"),
            vllm_model_dir=os.getenv("CONFUCIUS_T2S_VLLM_DIR", "checkpoints/t2s-vllm"),
            vllm_gpu_memory_utilization=float(
                os.getenv("CONFUCIUS_VLLM_GPU_MEMORY_UTILIZATION", "0.25")
            ),
            vllm_tensor_parallel_size=int(
                os.getenv("CONFUCIUS_VLLM_TENSOR_PARALLEL_SIZE", "1")
            ),
            vllm_dtype=os.getenv("CONFUCIUS_VLLM_DTYPE", "auto"),
            vllm_attention_backend=os.getenv("CONFUCIUS_VLLM_ATTENTION_BACKEND", ""),
            vllm_prefix_mode=os.getenv("CONFUCIUS_VLLM_PREFIX_MODE", "auto"),
            vllm_latent_mode=os.getenv("CONFUCIUS_VLLM_LATENT_MODE", "auto"),
            vllm_hidden_states_dir=os.getenv("CONFUCIUS_VLLM_HIDDEN_STATES_DIR", ""),
            compile_s2a=serving._env_optional_bool(
                "CONFUCIUS_USE_TORCH_COMPILE",
                "CONFUCIUS_COMPILE_S2A",
            ),
            use_bigvgan_cuda_kernel=serving._env_optional_bool(
                "CONFUCIUS_USE_BIGVGAN_CUDA_KERNEL",
                "CONFUCIUS_BIGVGAN_USE_CUDA_KERNEL",
            ),
            s2a_dtype=os.getenv("CONFUCIUS_S2A_DTYPE", "auto"),
            s2a_sdpa_backend=os.getenv("CONFUCIUS_S2A_SDPA_BACKEND", "auto"),
            s2a_length_bucket_size=int(os.getenv("CONFUCIUS_S2A_LENGTH_BUCKET_SIZE", "64")),
            compile_cache=serving._env_bool("CONFUCIUS_COMPILE_CACHE", True),
            compile_cache_dir=os.getenv(
                "CONFUCIUS_COMPILE_CACHE_DIR",
                "outputs/compile-cache/torchinductor",
            ),
            warmup=serving._env_bool("CONFUCIUS_WARMUP", True),
            warmup_mode=os.getenv("CONFUCIUS_WARMUP_MODE", "background"),
            warmup_prompt_wav=os.getenv("CONFUCIUS_WARMUP_PROMPT_WAV", "resources/voice.mp3"),
            warmup_text=os.getenv(
                "CONFUCIUS_WARMUP_TEXT",
                "Hello, welcome to Confucius4-TTS.",
            ),
            warmup_extra_text=os.getenv(
                "CONFUCIUS_WARMUP_EXTRA_TEXT",
                "this is the second warmup.",
            ),
            warmup_lang=os.getenv("CONFUCIUS_WARMUP_LANG", "en"),
            warmup_diffusion_steps=int(os.getenv("CONFUCIUS_WARMUP_DIFFUSION_STEPS", "10")),
            profile_cuda=serving._env_bool("CONFUCIUS_PROFILE_CUDA", False),
            profile_dir=os.getenv("CONFUCIUS_PROFILE_DIR", "outputs/profiles"),
            gpu_stage_concurrency=int(os.getenv("CONFUCIUS_GPU_STAGE_CONCURRENCY", "1")),
            reference_cache_size=int(os.getenv("CONFUCIUS_REFERENCE_CACHE_SIZE", "16")),
            postprocess_concurrency=int(os.getenv("CONFUCIUS_POSTPROCESS_CONCURRENCY", "2")),
            detailed_timings=serving._env_bool("CONFUCIUS_DETAILED_TIMINGS", False),
            output_dir=os.getenv("CONFUCIUS_API_OUTPUT_DIR", "outputs/api"),
            cors_origins=_parse_cors_origins(os.getenv("CONFUCIUS_API_CORS_ORIGINS", "")),
        )


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    lang: str = Field("en", description="Language code such as en, zh, ja, ko.")
    prompt_wav: Optional[str] = Field(
        DEFAULT_PROMPT_WAV,
        description=(
            "Server-side reference audio path. Omit or leave empty to use "
            f"{DEFAULT_PROMPT_WAV}."
        ),
    )
    temperature: float = Field(0.8, ge=0.0)
    top_p: float = Field(0.8, gt=0.0, le=1.0)
    top_k: int = Field(30, ge=0)
    num_beams: int = Field(3, ge=1)
    repetition_penalty: float = Field(10.0, gt=0.0)
    max_length: int = Field(1520, ge=1)
    diffusion_steps: int = Field(10, ge=1)
    cfg_strength: float = Field(0.7, ge=0.0)
    max_text_tokens: int = Field(80, ge=1)
    segment_render_batch_size: int = Field(1, ge=1)
    target_duration_seconds: float = Field(0.0, ge=0.0)
    target_segment_durations: Optional[list[Optional[float]]] = Field(
        None,
        description="Optional per-segment duration targets in seconds.",
    )
    cross_fade_duration: float = Field(0.3, ge=0.0)
    output_format: str = Field(
        "mp3",
        description="Response audio format: mp3 by default, or wav.",
    )
    verbose: bool = False
    include_audio_base64: bool = False


class TTSResponse(BaseModel):
    request_id: str
    lang: str
    sample_rate: int
    output_format: str
    duration_seconds: float
    elapsed_seconds: float
    audio_path: str
    audio_url: str
    wav_path: str
    wav_url: str
    audio_base64: Optional[str] = None
    timings: Optional[dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    device: str
    sample_rate: Optional[int]
    vllm_loaded: bool
    output_dir: str


def _parse_cors_origins(value: str) -> tuple[str, ...]:
    return tuple(origin.strip() for origin in value.split(",") if origin.strip())


def _resolve_repo_dir_or_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path.resolve()


def _configure_serving(settings: ApiSettings) -> None:
    global API_OUTPUT_DIR

    if serving.SERVE_MODEL is not None:
        return

    config_path = serving._resolve_repo_path(settings.config, "Config")
    vllm_model_dir = serving._resolve_repo_dir(settings.vllm_model_dir, "T2S vLLM directory")
    serve_device = serving._resolve_device(settings.device, probe_cuda=False)
    if serve_device != "cuda":
        raise RuntimeError(
            "The FastAPI serving entry point requires CUDA because it uses "
            "the vLLM T2S backend."
        )
    if settings.s2a_length_bucket_size < 0:
        raise ValueError("--s2a-length-bucket-size must be zero or greater.")
    if settings.warmup_diffusion_steps < 1:
        raise ValueError("--warmup-diffusion-steps must be at least 1.")
    if settings.warmup_mode not in {"background", "foreground"}:
        raise ValueError("--warmup-mode must be one of: background, foreground.")
    if settings.gpu_stage_concurrency < 1:
        raise ValueError("--gpu-stage-concurrency must be at least 1.")
    if settings.reference_cache_size < 0:
        raise ValueError("--reference-cache-size must be zero or greater.")
    if settings.postprocess_concurrency < 1:
        raise ValueError("--postprocess-concurrency must be at least 1.")

    API_OUTPUT_DIR = _resolve_repo_dir_or_path(settings.output_dir)
    API_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    serving.OUTPUT_DIR = API_OUTPUT_DIR

    t2s_checkpoint = serving._normalize_checkpoint(settings.t2s_checkpoint)
    vllm_attention_backend = serving._normalize_optional_text(settings.vllm_attention_backend)
    vllm_hidden_states_dir = serving._normalize_optional_text(settings.vllm_hidden_states_dir)

    serving.SERVE_CONFIG_PATH = config_path
    serving.SERVE_T2S_CHECKPOINT = t2s_checkpoint
    serving.SERVE_DEVICE = serve_device
    serving.SERVE_COMPILE_S2A = settings.compile_s2a
    serving.SERVE_USE_BIGVGAN_CUDA_KERNEL = settings.use_bigvgan_cuda_kernel
    serving.SERVE_S2A_DTYPE = settings.s2a_dtype
    serving.SERVE_S2A_SDPA_BACKEND = settings.s2a_sdpa_backend
    serving.SERVE_S2A_LENGTH_BUCKET_SIZE = max(0, int(settings.s2a_length_bucket_size))
    serving.SERVE_PROFILE_CUDA = bool(settings.profile_cuda)
    serving.SERVE_PROFILE_DIR = settings.profile_dir
    serving.SERVE_VLLM_PREFIX_MODE = settings.vllm_prefix_mode
    serving.SERVE_VLLM_LATENT_MODE = settings.vllm_latent_mode
    serving.SERVE_VLLM_HIDDEN_STATES_DIR = vllm_hidden_states_dir
    serving.SERVE_GPU_STAGE_CONCURRENCY = int(settings.gpu_stage_concurrency)
    serving.SERVE_REFERENCE_CACHE_SIZE = int(settings.reference_cache_size)
    serving.SERVE_DETAILED_TIMINGS = bool(settings.detailed_timings)
    serving.SERVE_POSTPROCESS_SEMAPHORE = threading.Semaphore(
        int(settings.postprocess_concurrency)
    )
    serving.SERVE_STREAM_SEMAPHORE = threading.Semaphore(
        int(settings.gpu_stage_concurrency)
    )

    compile_cache_dir = serving._configure_compile_cache(
        enabled=bool(settings.compile_cache),
        cache_dir=settings.compile_cache_dir,
    )
    compile_s2a_label = "auto" if settings.compile_s2a is None else str(settings.compile_s2a)
    bigvgan_kernel_label = (
        "auto"
        if settings.use_bigvgan_cuda_kernel is None
        else str(settings.use_bigvgan_cuda_kernel)
    )
    print(
        "[Confucius4-TTS] Loading FastAPI vLLM T2S backend "
        f"from {vllm_model_dir} on {serve_device} "
        f"(compile_s2a={compile_s2a_label}, "
        f"use_bigvgan_cuda_kernel={bigvgan_kernel_label}, "
        f"vllm_prefix_mode={settings.vllm_prefix_mode}, "
        f"vllm_latent_mode={settings.vllm_latent_mode}, "
        f"gpu_stage_concurrency={settings.gpu_stage_concurrency}, "
        f"reference_cache_size={settings.reference_cache_size}, "
        f"detailed_timings={settings.detailed_timings}, "
        f"s2a_dtype={settings.s2a_dtype}, "
        f"s2a_sdpa_backend={settings.s2a_sdpa_backend}, "
        f"s2a_length_bucket_size={serving.SERVE_S2A_LENGTH_BUCKET_SIZE}, "
        f"output_dir={API_OUTPUT_DIR}, "
        f"compile_cache_dir={compile_cache_dir or 'disabled'}, "
        f"warmup={bool(settings.warmup)})...",
        flush=True,
    )
    serving.SERVE_MODEL = serving._load_serving_model(
        config_path=config_path,
        t2s_checkpoint=t2s_checkpoint,
        device=serve_device,
        vllm_model_dir=vllm_model_dir,
        vllm_gpu_memory_utilization=settings.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=settings.vllm_tensor_parallel_size,
        vllm_dtype=settings.vllm_dtype,
        vllm_attention_backend=vllm_attention_backend,
        vllm_prefix_mode=settings.vllm_prefix_mode,
        vllm_latent_mode=settings.vllm_latent_mode,
        vllm_hidden_states_dir=vllm_hidden_states_dir,
        compile_s2a=settings.compile_s2a,
        use_bigvgan_cuda_kernel=settings.use_bigvgan_cuda_kernel,
        s2a_dtype=settings.s2a_dtype,
        s2a_sdpa_backend=settings.s2a_sdpa_backend,
        s2a_length_bucket_size=serving.SERVE_S2A_LENGTH_BUCKET_SIZE,
        profile_cuda=settings.profile_cuda,
        profile_dir=settings.profile_dir,
        gpu_stage_concurrency=settings.gpu_stage_concurrency,
        reference_cache_size=settings.reference_cache_size,
    )
    print("[Confucius4-TTS] FastAPI vLLM-backed TTS model is ready.", flush=True)

def _run_startup_warmup(settings: ApiSettings) -> None:
    print("[Confucius4-TTS] Running FastAPI startup warmup generation...", flush=True)
    warmup_started = time.perf_counter()
    warmup_ok = serving._warmup_serving_model(
        serving.SERVE_MODEL,
        prompt_wav=settings.warmup_prompt_wav,
        text=settings.warmup_text,
        lang=settings.warmup_lang,
        diffusion_steps=settings.warmup_diffusion_steps,
        cfg_strength=0.7,
        extra_text=serving._normalize_optional_text(settings.warmup_extra_text),
    )
    warmup_status = "completed" if warmup_ok else "finished with errors"
    print(
        "[Confucius4-TTS] FastAPI startup warmup "
        f"{warmup_status} in {time.perf_counter() - warmup_started:.2f}s.",
        flush=True,
    )


def _start_background_warmup(settings: ApiSettings) -> threading.Thread:
    thread = threading.Thread(
        target=_run_startup_warmup,
        args=(settings,),
        name="confucius-fastapi-warmup",
        daemon=True,
    )
    thread.start()
    return thread


def _run_or_schedule_warmup(app: FastAPI, settings: ApiSettings) -> None:
    if settings.warmup:
        if settings.warmup_mode == "foreground":
            _run_startup_warmup(settings)
        else:
            print(
                "[Confucius4-TTS] FastAPI startup warmup will run in the "
                "background after the HTTP service starts.",
                flush=True,
            )
            app.state.warmup_thread = _start_background_warmup(settings)
    else:
        print(
            "[Confucius4-TTS] FastAPI startup warmup disabled; "
            "the first synthesis request may be slower.",
            flush=True,
        )


def _model() -> Any:
    if serving.SERVE_MODEL is None:
        raise RuntimeError("The TTS model is not loaded.")
    return serving.SERVE_MODEL


def _normalize_language(value: str) -> str:
    language = (value or "en").strip()
    if " - " in language:
        language = serving._language_code(language)
    language = language.lower()
    if language not in LANGUAGE_CODES:
        raise ValueError(
            "Unsupported language code "
            f"{language!r}; expected one of: {', '.join(LANGUAGE_CODES)}."
        )
    return language


def _normalize_reference_path(prompt_wav: Optional[str]) -> str:
    try:
        return serving._reference_wav_or_default(prompt_wav)
    except Exception as exc:
        raise ValueError(str(exc)) from exc


def _normalize_segment_durations(
    durations: Optional[list[Optional[float]]],
) -> Optional[list[Optional[float]]]:
    if durations is None:
        return None
    if not durations:
        return None
    normalized: list[Optional[float]] = []
    for value in durations:
        if value is None:
            normalized.append(None)
            continue
        duration = float(value)
        if duration <= 0:
            raise ValueError("target_segment_durations values must be positive seconds.")
        normalized.append(duration)
    return normalized


def _normalize_output_format(value: str) -> str:
    normalized = (value or "mp3").strip().lower()
    if normalized not in {"mp3", "wav"}:
        raise ValueError("output_format must be one of: mp3, wav.")
    return normalized


def _new_request_id() -> str:
    return uuid.uuid4().hex


def _new_output_path(request_id: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return API_OUTPUT_DIR / f"confucius4tts-api-{stamp}-{request_id[:12]}.wav"


def _audio_url(output_path: Path) -> str:
    return f"/v1/audio/{output_path.name}"


def _audio_media_type(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".mp3":
        return "audio/mpeg"
    if suffix == ".wav":
        return "audio/wav"
    return "application/octet-stream"


def _timings_payload(timing_info: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if timing_info is None:
        return None
    payload = dict(timing_info)
    steps = payload.get("steps")
    if steps is not None:
        payload["steps"] = dict(steps)
    return payload


def _synthesize_sync(
    request: TTSRequest,
    *,
    prompt_wav_override: Optional[str] = None,
) -> TTSResponse:
    text = request.text.strip()
    if not text:
        raise ValueError("text is required.")

    lang = _normalize_language(request.lang)
    prompt_wav = _normalize_reference_path(prompt_wav_override or request.prompt_wav)
    output_format = _normalize_output_format(request.output_format)
    target_segment_durations = _normalize_segment_durations(
        request.target_segment_durations
    )
    model = _model()
    request_id = _new_request_id()
    wav_output = _new_output_path(request_id)

    started = time.perf_counter()
    result = model.generate(
        text=text,
        lang=lang,
        prompt_wav=prompt_wav,
        temperature=float(request.temperature),
        top_p=float(request.top_p),
        top_k=int(request.top_k),
        num_beams=int(request.num_beams),
        repetition_penalty=float(request.repetition_penalty),
        max_length=int(request.max_length),
        n_timesteps=int(request.diffusion_steps),
        inference_cfg_rate=float(request.cfg_strength),
        max_text_tokens_per_segment=int(request.max_text_tokens),
        segment_render_batch_size=int(request.segment_render_batch_size),
        target_duration_seconds=float(request.target_duration_seconds or 0.0),
        target_segment_durations=target_segment_durations,
        cross_fade_duration=float(request.cross_fade_duration),
        verbose=bool(request.verbose),
        use_vllm=True,
        return_timings=serving.SERVE_DETAILED_TIMINGS,
    )
    if serving.SERVE_DETAILED_TIMINGS:
        audio, timing_info = result
    else:
        audio = result
        timing_info = None

    wav_output.parent.mkdir(parents=True, exist_ok=True)
    with (serving.SERVE_POSTPROCESS_SEMAPHORE or nullcontext()):
        save_started = time.perf_counter()
        if (
            serving.SERVE_DETAILED_TIMINGS
            and torch.cuda.is_available()
            and model.device.type == "cuda"
        ):
            torch.cuda.synchronize(model.device)
        audio_cpu = audio.detach().float().cpu()
        torchaudio.save(str(wav_output), audio_cpu, model.sample_rate)
        save_elapsed = time.perf_counter() - save_started
        mp3_started = time.perf_counter()
        mp3_output = serving._save_mp3_preview(wav_output, audio_cpu, model.sample_rate)
        mp3_elapsed = time.perf_counter() - mp3_started

    elapsed = time.perf_counter() - started
    duration_seconds = audio_cpu.shape[-1] / model.sample_rate
    timings = _timings_payload(timing_info)
    if timings is not None:
        timings["save_wav"] = save_elapsed
        timings["save_mp3"] = mp3_elapsed
        timings["request_total"] = elapsed
    selected_output = mp3_output if output_format == "mp3" else wav_output

    audio_base64 = None
    if request.include_audio_base64:
        audio_base64 = base64.b64encode(selected_output.read_bytes()).decode("ascii")

    return TTSResponse(
        request_id=request_id,
        lang=lang,
        sample_rate=int(model.sample_rate),
        output_format=output_format,
        duration_seconds=float(duration_seconds),
        elapsed_seconds=float(elapsed),
        audio_path=str(selected_output),
        audio_url=_audio_url(selected_output),
        wav_path=str(wav_output),
        wav_url=_audio_url(wav_output),
        audio_base64=audio_base64,
        timings=timings,
    )


async def _run_synthesis(
    request: TTSRequest,
    *,
    prompt_wav_override: Optional[str] = None,
) -> TTSResponse:
    try:
        return await run_in_threadpool(
            _synthesize_sync,
            request,
            prompt_wav_override=prompt_wav_override,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _copy_request(request: TTSRequest, **updates: Any) -> TTSRequest:
    if hasattr(request, "model_copy"):
        return request.model_copy(update=updates)
    return request.copy(update=updates)


def _parse_payload(payload: str) -> TTSRequest:
    try:
        if hasattr(TTSRequest, "model_validate_json"):
            return TTSRequest.model_validate_json(payload)
        return TTSRequest.parse_raw(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {exc}") from exc


def _resolve_download_path(file_name: str) -> Path:
    if not file_name or Path(file_name).name != file_name:
        raise HTTPException(status_code=400, detail="Invalid file name.")
    path = (API_OUTPUT_DIR / file_name).resolve()
    try:
        path.relative_to(API_OUTPUT_DIR.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid file path.") from exc
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Audio file not found.")
    return path


async def _save_uploaded_reference(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in REFERENCE_UPLOAD_SUFFIXES:
        suffix = ".wav"
    reference_dir = API_OUTPUT_DIR / "references"
    reference_dir.mkdir(parents=True, exist_ok=True)
    reference_path = reference_dir / f"reference-{uuid.uuid4().hex}{suffix}"
    with reference_path.open("wb") as handle:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    await upload.close()
    return str(reference_path)


def _health_payload() -> HealthResponse:
    model = serving.SERVE_MODEL
    return HealthResponse(
        status="ok" if model is not None else "starting",
        version=API_VERSION,
        model_loaded=model is not None,
        device=serving.SERVE_DEVICE,
        sample_rate=None if model is None else int(model.sample_rate),
        vllm_loaded=bool(model is not None and getattr(model, "t2s_vllm", None) is not None),
        output_dir=str(API_OUTPUT_DIR),
    )


def _ui_html() -> str:
    language_options = "\n".join(
        f'<option value="{code}"{" selected" if code == "en" else ""}>{code}</option>'
        for code in LANGUAGE_CODES
    )
    defaults = json.dumps(
        {
            "prompt_wav": DEFAULT_PROMPT_WAV,
            "temperature": 0.8,
            "top_p": 0.8,
            "top_k": 30,
            "num_beams": 3,
            "repetition_penalty": 10.0,
            "max_length": 1520,
            "diffusion_steps": 10,
            "cfg_strength": 0.7,
            "max_text_tokens": 80,
            "segment_render_batch_size": 1,
            "target_duration_seconds": 0,
            "cross_fade_duration": 0.3,
            "output_format": "mp3",
        }
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Confucius4-TTS API Tester</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --border: #d7dce2;
      --text: #17202a;
      --muted: #5e6a78;
      --accent: #2f6fed;
      --accent-dark: #1f55bd;
      --danger: #b42318;
      --ok: #117047;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
    }}
    main {{
      width: min(1180px, calc(100vw - 32px));
      margin: 24px auto;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 16px;
    }}
    header {{
      grid-column: 1 / -1;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }}
    h1 {{
      margin: 0;
      font-size: 22px;
      line-height: 1.2;
      font-weight: 650;
      letter-spacing: 0;
    }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    section, aside {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .wide {{ grid-column: 1 / -1; }}
    label {{
      display: grid;
      gap: 6px;
      min-width: 0;
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
    }}
    input, select, textarea, button {{
      width: 100%;
      min-width: 0;
      border-radius: 6px;
      border: 1px solid var(--border);
      font: inherit;
    }}
    input, select, textarea {{
      background: #fff;
      color: var(--text);
      padding: 9px 10px;
    }}
    textarea {{
      min-height: 190px;
      resize: vertical;
      line-height: 1.45;
    }}
    input[type="file"] {{ padding: 7px; }}
    details {{
      margin-top: 14px;
      border-top: 1px solid var(--border);
      padding-top: 14px;
    }}
    summary {{
      cursor: pointer;
      color: var(--text);
      font-weight: 650;
      margin-bottom: 12px;
    }}
    .actions {{
      display: flex;
      gap: 10px;
      margin-top: 14px;
    }}
    button {{
      cursor: pointer;
      padding: 10px 12px;
      font-weight: 650;
      background: #fff;
      color: var(--text);
    }}
    button.primary {{
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }}
    button.primary:hover {{ background: var(--accent-dark); }}
    button:disabled {{ cursor: wait; opacity: 0.65; }}
    .status {{
      min-height: 46px;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 10px;
      color: var(--muted);
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }}
    .status.ok {{ color: var(--ok); }}
    .status.error {{ color: var(--danger); }}
    audio {{
      width: 100%;
      margin-top: 12px;
    }}
    dl {{
      display: grid;
      grid-template-columns: 120px minmax(0, 1fr);
      gap: 8px 10px;
      margin: 14px 0 0;
    }}
    dt {{ color: var(--muted); font-weight: 650; }}
    dd {{
      margin: 0;
      overflow-wrap: anywhere;
    }}
    pre {{
      margin: 14px 0 0;
      max-height: 280px;
      overflow: auto;
      border-radius: 6px;
      border: 1px solid var(--border);
      background: #f9fafb;
      padding: 10px;
      font-size: 12px;
    }}
    @media (max-width: 900px) {{
      main {{ grid-template-columns: 1fr; }}
      header {{ align-items: flex-start; flex-direction: column; }}
    }}
    @media (max-width: 620px) {{
      main {{ width: min(100vw - 20px, 1180px); margin: 10px auto; }}
      .grid {{ grid-template-columns: 1fr; }}
      .wide {{ grid-column: auto; }}
      .actions {{ flex-direction: column; }}
      dl {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Confucius4-TTS API Tester</h1>
      <nav><a href="/docs">Docs</a></nav>
    </header>

    <section>
      <form id="tts-form" novalidate>
        <div class="grid">
          <label class="wide">Text
            <textarea id="text" required>Hello, this is a test of zero-shot voice cloning.</textarea>
          </label>
          <label>Language
            <select id="lang">{language_options}</select>
          </label>
          <label>Server Reference
            <input id="prompt_wav" value="{DEFAULT_PROMPT_WAV}">
          </label>
          <label>Output
            <select id="output_format">
              <option value="mp3" selected>MP3</option>
              <option value="wav">WAV</option>
            </select>
          </label>
          <label class="wide">Upload Reference
            <input id="prompt_file" type="file" accept="audio/*">
          </label>
        </div>

        <details>
          <summary>Generation Parameters</summary>
          <div class="grid">
            <label>Temperature <input id="temperature" type="number" step="0.05" min="0" value="0.8"></label>
            <label>Top P <input id="top_p" type="number" step="0.01" min="0" max="1" value="0.8"></label>
            <label>Top K <input id="top_k" type="number" step="1" min="0" value="30"></label>
            <label>Beams <input id="num_beams" type="number" step="1" min="1" value="3"></label>
            <label>Repetition Penalty <input id="repetition_penalty" type="number" step="0.1" min="0.1" value="10"></label>
            <label>Max Length <input id="max_length" type="number" step="1" min="1" value="1520"></label>
            <label>Diffusion Steps <input id="diffusion_steps" type="number" step="1" min="1" value="10"></label>
            <label>CFG Strength <input id="cfg_strength" type="number" step="0.05" min="0" value="0.7"></label>
            <label>Text Tokens <input id="max_text_tokens" type="number" step="1" min="1" value="80"></label>
            <label>Render Batch <input id="segment_render_batch_size" type="number" step="1" min="1" value="1"></label>
            <label>Target Seconds <input id="target_duration_seconds" type="number" step="0.1" min="0" value="0"></label>
            <label>Cross Fade <input id="cross_fade_duration" type="number" step="0.05" min="0" value="0.3"></label>
            <label class="wide">Segment Seconds
              <input id="target_segment_durations" placeholder="1.2, 1.4, 1.1">
            </label>
          </div>
        </details>

        <div class="actions">
          <button class="primary" id="generate" type="submit">Generate</button>
          <button id="reset" type="button">Reset</button>
        </div>
      </form>
    </section>

    <aside>
      <div id="status" class="status">Ready</div>
      <audio id="audio" controls></audio>
      <dl>
        <dt>Request</dt><dd id="request-id">-</dd>
        <dt>Duration</dt><dd id="duration">-</dd>
        <dt>Elapsed</dt><dd id="elapsed">-</dd>
        <dt>Download</dt><dd id="download">-</dd>
      </dl>
      <pre id="json">{{}}</pre>
    </aside>
  </main>

  <script>
    const defaults = {defaults};
    const form = document.getElementById('tts-form');
    const statusEl = document.getElementById('status');
    const audioEl = document.getElementById('audio');
    const jsonEl = document.getElementById('json');
    const generateBtn = document.getElementById('generate');
    const fields = [
      'temperature', 'top_p', 'top_k', 'num_beams', 'repetition_penalty',
      'max_length', 'diffusion_steps', 'cfg_strength', 'max_text_tokens',
      'segment_render_batch_size', 'target_duration_seconds', 'cross_fade_duration'
    ];

    function setStatus(text, kind = '') {{
      statusEl.className = 'status' + (kind ? ' ' + kind : '');
      statusEl.textContent = text;
    }}

    function numberValue(id) {{
      const value = document.getElementById(id).value;
      return value === '' ? defaults[id] : Number(value);
    }}

    function payload() {{
      const data = {{
        text: document.getElementById('text').value,
        lang: document.getElementById('lang').value,
        prompt_wav: document.getElementById('prompt_wav').value || defaults.prompt_wav,
        output_format: document.getElementById('output_format').value || defaults.output_format,
      }};
      for (const id of fields) data[id] = numberValue(id);
      const segmentDurations = document.getElementById('target_segment_durations').value
        .split(',')
        .map((item) => item.trim())
        .filter(Boolean)
        .map(Number);
      if (segmentDurations.length) data.target_segment_durations = segmentDurations;
      return data;
    }}

    function renderResult(result) {{
      const audioUrl = result.audio_url + '?t=' + Date.now();
      audioEl.src = audioUrl;
      document.getElementById('request-id').textContent = result.request_id || '-';
      document.getElementById('duration').textContent = result.duration_seconds
        ? result.duration_seconds.toFixed(2) + 's'
        : '-';
      document.getElementById('elapsed').textContent = result.elapsed_seconds
        ? result.elapsed_seconds.toFixed(2) + 's'
        : '-';
      document.getElementById('download').innerHTML = result.audio_url
        ? '<a href="' + result.audio_url + '">' + (result.output_format || 'audio').toUpperCase() + '</a>'
        : '-';
      jsonEl.textContent = JSON.stringify(result, null, 2);
    }}

    async function generate() {{
      const data = payload();
      const file = document.getElementById('prompt_file').files[0];
      let response;
      if (file) {{
        const formData = new FormData();
        formData.append('payload', JSON.stringify(data));
        formData.append('prompt_wav', file);
        response = await fetch('/v1/tts/upload', {{ method: 'POST', body: formData }});
      }} else {{
        response = await fetch('/v1/tts', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify(data),
        }});
      }}
      const text = await response.text();
      let result;
      try {{
        result = JSON.parse(text);
      }} catch (_error) {{
        throw new Error(text || response.statusText);
      }}
      if (!response.ok) {{
        const detail = typeof result.detail === 'string'
          ? result.detail
          : JSON.stringify(result.detail || result);
        throw new Error(detail);
      }}
      return result;
    }}

    form.addEventListener('submit', async (event) => {{
      event.preventDefault();
      generateBtn.disabled = true;
      setStatus('Generating...');
      try {{
        const result = await generate();
        renderResult(result);
        setStatus('Complete', 'ok');
      }} catch (error) {{
        setStatus(error.message || String(error), 'error');
      }} finally {{
        generateBtn.disabled = false;
      }}
    }});

    document.getElementById('reset').addEventListener('click', () => {{
      form.reset();
      document.getElementById('prompt_wav').value = defaults.prompt_wav;
      document.getElementById('output_format').value = defaults.output_format;
      audioEl.removeAttribute('src');
      document.getElementById('request-id').textContent = '-';
      document.getElementById('duration').textContent = '-';
      document.getElementById('elapsed').textContent = '-';
      document.getElementById('download').textContent = '-';
      jsonEl.textContent = '{{}}';
      setStatus('Ready');
    }});
  </script>
</body>
</html>"""


def create_app(
    settings: Optional[ApiSettings] = None,
    *,
    load_model_in_lifespan: bool = True,
) -> FastAPI:
    settings = settings or ApiSettings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if load_model_in_lifespan:
            _configure_serving(settings)
        _run_or_schedule_warmup(app, settings)
        yield

    app = FastAPI(
        title="Confucius4-TTS API",
        version=API_VERSION,
        lifespan=lifespan,
    )
    app.state.settings = settings

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.get("/", tags=["meta"])
    async def root() -> dict[str, str]:
        return {
            "name": "Confucius4-TTS API",
            "version": API_VERSION,
            "health": "/health",
            "ui": "/ui",
            "tts_json": "/v1/tts",
            "tts_audio": "/v1/tts/audio",
            "docs": "/docs",
        }

    @app.get("/ui", response_class=HTMLResponse, tags=["meta"])
    async def ui() -> HTMLResponse:
        return HTMLResponse(_ui_html())

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    async def health() -> HealthResponse:
        return _health_payload()

    @app.post("/v1/tts", response_model=TTSResponse, tags=["tts"])
    async def synthesize_json(request: TTSRequest) -> TTSResponse:
        return await _run_synthesis(request)

    @app.post("/v1/tts/audio", tags=["tts"])
    async def synthesize_audio(request: TTSRequest) -> FileResponse:
        response = await _run_synthesis(_copy_request(request, include_audio_base64=False))
        return FileResponse(
            response.audio_path,
            media_type=_audio_media_type(response.audio_path),
            filename=Path(response.audio_path).name,
            headers={
                "X-Confucius-Request-ID": response.request_id,
                "X-Confucius-Output-Format": response.output_format,
                "X-Confucius-Sample-Rate": str(response.sample_rate),
                "X-Confucius-Duration-Seconds": f"{response.duration_seconds:.6f}",
                "X-Confucius-Elapsed-Seconds": f"{response.elapsed_seconds:.6f}",
            },
        )

    @app.post("/v1/tts/upload", response_model=TTSResponse, tags=["tts"])
    async def synthesize_with_uploaded_reference(
        payload: str = Form(..., description="JSON-encoded TTSRequest."),
        prompt_wav: Optional[UploadFile] = File(
            None,
            description=f"Optional reference audio file. Omit to use {DEFAULT_PROMPT_WAV}.",
        ),
    ) -> TTSResponse:
        request = _parse_payload(payload)
        reference_path = None
        if prompt_wav is not None:
            reference_path = await _save_uploaded_reference(prompt_wav)
        return await _run_synthesis(request, prompt_wav_override=reference_path)

    @app.post("/v1/tts/upload/audio", tags=["tts"])
    async def synthesize_uploaded_reference_audio(
        payload: str = Form(..., description="JSON-encoded TTSRequest."),
        prompt_wav: Optional[UploadFile] = File(
            None,
            description=f"Optional reference audio file. Omit to use {DEFAULT_PROMPT_WAV}.",
        ),
    ) -> FileResponse:
        request = _copy_request(_parse_payload(payload), include_audio_base64=False)
        reference_path = None
        if prompt_wav is not None:
            reference_path = await _save_uploaded_reference(prompt_wav)
        response = await _run_synthesis(request, prompt_wav_override=reference_path)
        return FileResponse(
            response.audio_path,
            media_type=_audio_media_type(response.audio_path),
            filename=Path(response.audio_path).name,
            headers={
                "X-Confucius-Request-ID": response.request_id,
                "X-Confucius-Output-Format": response.output_format,
                "X-Confucius-Sample-Rate": str(response.sample_rate),
                "X-Confucius-Duration-Seconds": f"{response.duration_seconds:.6f}",
                "X-Confucius-Elapsed-Seconds": f"{response.elapsed_seconds:.6f}",
            },
        )

    @app.get("/v1/audio/{file_name}", tags=["tts"])
    async def download_audio(file_name: str) -> FileResponse:
        path = _resolve_download_path(file_name)
        return FileResponse(
            str(path),
            media_type=_audio_media_type(path),
            filename=path.name,
        )

    return app


def parse_args() -> argparse.Namespace:
    defaults = ApiSettings.from_env()
    parser = argparse.ArgumentParser(description="Launch the Confucius4-TTS FastAPI service.")
    parser.add_argument("--host", "--server-name", dest="host",
                        default=os.getenv("CONFUCIUS_API_HOST", "127.0.0.1"))
    parser.add_argument("--port", "--server-port", dest="port", type=int,
                        default=int(os.getenv("CONFUCIUS_API_PORT", "8000")))
    parser.add_argument("--root-path", default=os.getenv("CONFUCIUS_API_ROOT_PATH", ""))
    parser.add_argument("--log-level", default=os.getenv("CONFUCIUS_API_LOG_LEVEL", "info"))
    parser.add_argument("--cors-origin", dest="cors_origins", action="append",
                        default=list(defaults.cors_origins),
                        help="Allowed CORS origin. Repeat or set CONFUCIUS_API_CORS_ORIGINS.")
    parser.add_argument("--output-dir", default=defaults.output_dir,
                        help="Directory for generated WAV files and uploaded references.")
    parser.add_argument("--config", default=defaults.config)
    parser.add_argument("--t2s-checkpoint", default=defaults.t2s_checkpoint)
    parser.add_argument("--device", default=defaults.device, choices=["auto", "cuda", "cpu"])
    parser.add_argument("--vllm-model-dir", default=defaults.vllm_model_dir)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float,
                        default=defaults.vllm_gpu_memory_utilization)
    parser.add_argument("--vllm-tensor-parallel-size", type=int,
                        default=defaults.vllm_tensor_parallel_size)
    parser.add_argument("--vllm-dtype", default=defaults.vllm_dtype)
    parser.add_argument("--vllm-attention-backend", default=defaults.vllm_attention_backend)
    parser.add_argument("--vllm-prefix-mode", default=defaults.vllm_prefix_mode,
                        choices=["auto", "embeds", "worker"])
    parser.add_argument("--vllm-latent-mode", default=defaults.vllm_latent_mode,
                        choices=["auto", "vllm", "pytorch"])
    parser.add_argument("--vllm-hidden-states-dir", default=defaults.vllm_hidden_states_dir)
    parser.add_argument("--compile-s2a", "--use-torch-compile", "--use_torch_compile",
                        action=argparse.BooleanOptionalAction, dest="compile_s2a",
                        default=defaults.compile_s2a)
    parser.add_argument("--use-bigvgan-cuda-kernel", action=argparse.BooleanOptionalAction,
                        default=defaults.use_bigvgan_cuda_kernel)
    parser.add_argument("--s2a-dtype", default=defaults.s2a_dtype,
                        choices=["auto", "float32", "bfloat16", "float16", "bf16", "fp16", "fp32"])
    parser.add_argument("--s2a-sdpa-backend", default=defaults.s2a_sdpa_backend,
                        choices=["auto", "flash", "efficient", "math", "cudnn"])
    parser.add_argument("--s2a-length-bucket-size", type=int,
                        default=defaults.s2a_length_bucket_size)
    parser.add_argument("--compile-cache", action=argparse.BooleanOptionalAction,
                        default=defaults.compile_cache)
    parser.add_argument("--compile-cache-dir", default=defaults.compile_cache_dir)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction,
                        default=defaults.warmup)
    parser.add_argument("--warmup-mode", default=defaults.warmup_mode,
                        choices=["background", "foreground"],
                        help="Run warmup after HTTP startup or block startup until warmup completes.")
    parser.add_argument("--warmup-prompt-wav", default=defaults.warmup_prompt_wav)
    parser.add_argument("--warmup-text", default=defaults.warmup_text)
    parser.add_argument("--warmup-extra-text", default=defaults.warmup_extra_text)
    parser.add_argument("--warmup-lang", default=defaults.warmup_lang,
                        choices=LANGUAGE_CODES)
    parser.add_argument("--warmup-diffusion-steps", type=int,
                        default=defaults.warmup_diffusion_steps)
    parser.add_argument("--profile-cuda", action=argparse.BooleanOptionalAction,
                        default=defaults.profile_cuda)
    parser.add_argument("--profile-dir", default=defaults.profile_dir)
    parser.add_argument("--gpu-stage-concurrency", type=int,
                        default=defaults.gpu_stage_concurrency)
    parser.add_argument("--reference-cache-size", type=int,
                        default=defaults.reference_cache_size)
    parser.add_argument("--postprocess-concurrency", type=int,
                        default=defaults.postprocess_concurrency)
    parser.add_argument("--detailed-timings", action=argparse.BooleanOptionalAction,
                        default=defaults.detailed_timings)
    return parser.parse_args()


def _settings_from_args(args: argparse.Namespace) -> ApiSettings:
    return ApiSettings(
        config=args.config,
        t2s_checkpoint=args.t2s_checkpoint,
        device=args.device,
        vllm_model_dir=args.vllm_model_dir,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_dtype=args.vllm_dtype,
        vllm_attention_backend=args.vllm_attention_backend,
        vllm_prefix_mode=args.vllm_prefix_mode,
        vllm_latent_mode=args.vllm_latent_mode,
        vllm_hidden_states_dir=args.vllm_hidden_states_dir,
        compile_s2a=args.compile_s2a,
        use_bigvgan_cuda_kernel=args.use_bigvgan_cuda_kernel,
        s2a_dtype=args.s2a_dtype,
        s2a_sdpa_backend=args.s2a_sdpa_backend,
        s2a_length_bucket_size=args.s2a_length_bucket_size,
        compile_cache=args.compile_cache,
        compile_cache_dir=args.compile_cache_dir,
        warmup=args.warmup,
        warmup_mode=args.warmup_mode,
        warmup_prompt_wav=args.warmup_prompt_wav,
        warmup_text=args.warmup_text,
        warmup_extra_text=args.warmup_extra_text,
        warmup_lang=args.warmup_lang,
        warmup_diffusion_steps=args.warmup_diffusion_steps,
        profile_cuda=args.profile_cuda,
        profile_dir=args.profile_dir,
        gpu_stage_concurrency=args.gpu_stage_concurrency,
        reference_cache_size=args.reference_cache_size,
        postprocess_concurrency=args.postprocess_concurrency,
        detailed_timings=args.detailed_timings,
        output_dir=args.output_dir,
        cors_origins=tuple(args.cors_origins or ()),
    )


def main() -> None:
    args = parse_args()
    settings = _settings_from_args(args)
    # Match gradio_app.py startup ordering: initialize the vLLM-backed model
    # before starting the web server event loop. Creating AsyncLLM inside
    # Uvicorn's lifespan loop can leave the first vLLM generation waiting on
    # the wrong async loop/thread.
    _configure_serving(settings)
    app_instance = create_app(settings, load_model_in_lifespan=False)

    import uvicorn

    uvicorn.run(
        app_instance,
        host=args.host,
        port=args.port,
        root_path=args.root_path,
        log_level=args.log_level,
    )


app = create_app()


if __name__ == "__main__":
    main()
