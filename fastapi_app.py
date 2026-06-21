#!/usr/bin/env python3
"""FastAPI service for Confucius4-TTS optimized for programmatic clients."""

from __future__ import annotations

import argparse
import base64
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
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import gradio_app as serving

API_VERSION = "0.1.0"
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
        None,
        description="Server-side reference audio path. Omit to use resources/voice.mp3.",
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
    verbose: bool = False
    include_audio_base64: bool = False


class TTSResponse(BaseModel):
    request_id: str
    lang: str
    sample_rate: int
    duration_seconds: float
    elapsed_seconds: float
    audio_path: str
    audio_url: str
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

    if settings.warmup:
        serving._warmup_serving_model(
            serving.SERVE_MODEL,
            prompt_wav=settings.warmup_prompt_wav,
            text=settings.warmup_text,
            lang=settings.warmup_lang,
            diffusion_steps=settings.warmup_diffusion_steps,
            cfg_strength=0.7,
            extra_text=serving._normalize_optional_text(settings.warmup_extra_text),
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


def _new_request_id() -> str:
    return uuid.uuid4().hex


def _new_output_path(request_id: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return API_OUTPUT_DIR / f"confucius4tts-api-{stamp}-{request_id[:12]}.wav"


def _audio_url(output_path: Path) -> str:
    return f"/v1/audio/{output_path.name}"


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
    target_segment_durations = _normalize_segment_durations(
        request.target_segment_durations
    )
    model = _model()
    request_id = _new_request_id()
    output = _new_output_path(request_id)

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

    output.parent.mkdir(parents=True, exist_ok=True)
    with (serving.SERVE_POSTPROCESS_SEMAPHORE or nullcontext()):
        save_started = time.perf_counter()
        if (
            serving.SERVE_DETAILED_TIMINGS
            and torch.cuda.is_available()
            and model.device.type == "cuda"
        ):
            torch.cuda.synchronize(model.device)
        audio_cpu = audio.detach().float().cpu()
        torchaudio.save(str(output), audio_cpu, model.sample_rate)
        save_elapsed = time.perf_counter() - save_started

    elapsed = time.perf_counter() - started
    duration_seconds = audio_cpu.shape[-1] / model.sample_rate
    timings = _timings_payload(timing_info)
    if timings is not None:
        timings["save_wav"] = save_elapsed
        timings["request_total"] = elapsed

    audio_base64 = None
    if request.include_audio_base64:
        audio_base64 = base64.b64encode(output.read_bytes()).decode("ascii")

    return TTSResponse(
        request_id=request_id,
        lang=lang,
        sample_rate=int(model.sample_rate),
        duration_seconds=float(duration_seconds),
        elapsed_seconds=float(elapsed),
        audio_path=str(output),
        audio_url=_audio_url(output),
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


def create_app(settings: Optional[ApiSettings] = None) -> FastAPI:
    settings = settings or ApiSettings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _configure_serving(settings)
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
            "tts_json": "/v1/tts",
            "tts_audio": "/v1/tts/audio",
            "docs": "/docs",
        }

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
            media_type="audio/wav",
            filename=Path(response.audio_path).name,
            headers={
                "X-Confucius-Request-ID": response.request_id,
                "X-Confucius-Sample-Rate": str(response.sample_rate),
                "X-Confucius-Duration-Seconds": f"{response.duration_seconds:.6f}",
                "X-Confucius-Elapsed-Seconds": f"{response.elapsed_seconds:.6f}",
            },
        )

    @app.post("/v1/tts/upload", response_model=TTSResponse, tags=["tts"])
    async def synthesize_with_uploaded_reference(
        payload: str = Form(..., description="JSON-encoded TTSRequest."),
        prompt_wav: UploadFile = File(...),
    ) -> TTSResponse:
        reference_path = await _save_uploaded_reference(prompt_wav)
        request = _parse_payload(payload)
        return await _run_synthesis(request, prompt_wav_override=reference_path)

    @app.post("/v1/tts/upload/audio", tags=["tts"])
    async def synthesize_uploaded_reference_audio(
        payload: str = Form(..., description="JSON-encoded TTSRequest."),
        prompt_wav: UploadFile = File(...),
    ) -> FileResponse:
        reference_path = await _save_uploaded_reference(prompt_wav)
        request = _copy_request(_parse_payload(payload), include_audio_base64=False)
        response = await _run_synthesis(request, prompt_wav_override=reference_path)
        return FileResponse(
            response.audio_path,
            media_type="audio/wav",
            filename=Path(response.audio_path).name,
            headers={
                "X-Confucius-Request-ID": response.request_id,
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
            media_type="audio/wav",
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
    app_instance = create_app(settings)

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
