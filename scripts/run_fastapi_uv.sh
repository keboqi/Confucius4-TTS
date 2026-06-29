#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

VENV_DIR="${VENV_DIR:-.venv-fastapi}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
CONFIG_PATH="${CONFIG_PATH:-config/inference_config.yaml}"
VLLM_MODEL_DIR="${VLLM_MODEL_DIR:-checkpoints/t2s-vllm}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
DEVICE="${DEVICE:-cuda}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.25}"
NVFP4="${CONFUCIUS_NVFP4:-0}"
WARMUP="${WARMUP:-background}"
INSTALL_FFMPEG="${INSTALL_FFMPEG:-1}"
INSTALL_UV="${INSTALL_UV:-1}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}"
INSTALL_TORCHCODEC="${INSTALL_TORCHCODEC:-1}"
CONVERT_VLLM="${CONVERT_VLLM:-auto}"
UV_LINK_MODE="${UV_LINK_MODE:-copy}"
FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEEL:-https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.10-cp312/flash_attn-2.8.3%2Bcu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl}"
TORCHCODEC_SPEC="${TORCHCODEC_SPEC:-torchcodec==0.9.*}"
UV_BIN="${UV_BIN:-uv}"

export UV_LINK_MODE

log() {
    printf '[confucius4-tts-fastapi] %s\n' "$*"
}

run_with_sudo() {
    if [[ "$(id -u)" -eq 0 ]]; then
        "$@"
    else
        sudo "$@"
    fi
}

ensure_uv() {
    if command -v "${UV_BIN}" >/dev/null 2>&1; then
        return
    fi
    if [[ "${INSTALL_UV}" != "1" ]]; then
        printf 'uv is not installed. Install uv or set UV_BIN=/path/to/uv.\n' >&2
        exit 1
    fi
    if ! command -v curl >/dev/null 2>&1; then
        printf 'curl is required to bootstrap uv. Install curl or uv first.\n' >&2
        exit 1
    fi
    log "Installing uv with the standalone installer"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
    if ! command -v uv >/dev/null 2>&1; then
        printf 'uv install completed, but uv is still not on PATH.\n' >&2
        exit 1
    fi
    UV_BIN="$(command -v uv)"
}

ensure_ffmpeg() {
    if command -v ffmpeg >/dev/null 2>&1; then
        return
    fi
    if [[ "${INSTALL_FFMPEG}" != "1" ]]; then
        log "ffmpeg is not installed; continuing because INSTALL_FFMPEG=${INSTALL_FFMPEG}"
        return
    fi
    if ! command -v apt-get >/dev/null 2>&1; then
        printf 'ffmpeg is missing and apt-get is unavailable. Install ffmpeg manually.\n' >&2
        exit 1
    fi
    log "Installing ffmpeg with apt-get"
    run_with_sudo apt-get update
    run_with_sudo apt-get install -y ffmpeg
}

create_venv() {
    local venv_python="${VENV_DIR}/bin/python"
    if [[ -x "${venv_python}" ]]; then
        log "Using existing virtual environment: ${VENV_DIR}"
        return
    fi
    log "Creating isolated uv virtual environment: ${VENV_DIR} (Python ${PYTHON_VERSION})"
    "${UV_BIN}" venv --python "${PYTHON_VERSION}" "${VENV_DIR}"
}

venv_python() {
    printf '%s/bin/python' "${VENV_DIR}"
}

install_python_dependencies() {
    local py
    py="$(venv_python)"

    log "Installing core requirements into ${VENV_DIR}"
    "${UV_BIN}" pip install --python "${py}" -r requirements.txt

    log "Installing CUDA 12.8 PyTorch wheel set"
    "${UV_BIN}" pip install --python "${py}" --reinstall -r requirements-cu128.txt

    log "Installing vLLM requirements and editable Confucius4-TTS package"
    "${UV_BIN}" pip install --python "${py}" -r requirements-vllm.txt

    case "${NVFP4}" in
        1|true|yes|on)
            log "Installing optional NVFP4 dependencies"
            "${UV_BIN}" pip install --python "${py}" -r requirements-nvfp4.txt
            ;;
    esac

    log "Pinning numpy below 2 for the current stack"
    "${UV_BIN}" pip install --python "${py}" "numpy<2"

    if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
        log "Installing FlashAttention wheel"
        "${UV_BIN}" pip install --python "${py}" "${FLASH_ATTN_WHEEL}"
    else
        log "Skipping FlashAttention wheel because INSTALL_FLASH_ATTN=${INSTALL_FLASH_ATTN}"
    fi

    if [[ "${INSTALL_TORCHCODEC}" == "1" ]]; then
        log "Installing torchcodec"
        "${UV_BIN}" pip install --python "${py}" "${TORCHCODEC_SPEC}"
    else
        log "Skipping torchcodec because INSTALL_TORCHCODEC=${INSTALL_TORCHCODEC}"
    fi
}

convert_vllm_model() {
    local py
    py="$(venv_python)"

    case "${CONVERT_VLLM}" in
        1|true|yes|on)
            ;;
        0|false|no|off)
            log "Skipping vLLM conversion because CONVERT_VLLM=${CONVERT_VLLM}"
            return
            ;;
        auto)
            if [[ -d "${VLLM_MODEL_DIR}" ]]; then
                log "Using existing converted vLLM model directory: ${VLLM_MODEL_DIR}"
                return
            fi
            ;;
        *)
            printf 'CONVERT_VLLM must be one of: auto, 1, 0.\n' >&2
            exit 1
            ;;
    esac

    log "Converting T2S checkpoint for vLLM: ${VLLM_MODEL_DIR}"
    "${py}" tools/convert_t2s_vllm.py \
        --config "${CONFIG_PATH}" \
        --output "${VLLM_MODEL_DIR}"
}

start_fastapi() {
    local py
    py="$(venv_python)"
    local server_args=(
        --host "${HOST}"
        --port "${PORT}"
        --device "${DEVICE}"
        --config "${CONFIG_PATH}"
        --vllm-model-dir "${VLLM_MODEL_DIR}"
        --vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
    )

    case "${WARMUP}" in
        1|true|yes|on|background)
            server_args+=(--warmup --warmup-mode background)
            ;;
        foreground|blocking)
            server_args+=(--warmup --warmup-mode foreground)
            ;;
        0|false|no|off)
            server_args+=(--no-warmup)
            ;;
        auto)
            ;;
        *)
            printf 'WARMUP must be one of: background, foreground, auto, 1, 0.\n' >&2
            exit 1
            ;;
    esac

    log "Starting FastAPI backend on ${HOST}:${PORT}"
    exec "${py}" fastapi_app.py "${server_args[@]}" "$@"
}

ensure_uv
ensure_ffmpeg
create_venv
install_python_dependencies
convert_vllm_model
start_fastapi "$@"
