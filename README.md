<div align="center">
    <img src="./resources/Confucius4-TTS.png" alt="Confucius4-TTS" width="35%">
    <h1>Confucius4-TTS: a Multilingual and Cross-Lingual Zero-Shot TTS Engine</h1>
    <p><b>One voice. Any language.</b></p>
</div>

<div align="center">
    <a href="./README.zh.md"><img src="https://img.shields.io/badge/README-中文版本-red"></a>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="./LICENSE"><img src="https://img.shields.io/badge/code_license-Apache%202.0-blue"></a>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://confucius4-tts.youdao.com/gradio"><img src="https://img.shields.io/badge/Demo-在线体验-orange"></a>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://2901733926.github.io/Confucius4-TTS/"><img src="https://img.shields.io/badge/GitHub.io-Demo_Page-blue?logo=GitHub&style=flat-square"></a>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://huggingface.co/netease-youdao/Confucius4-TTS"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Confucius4TTS-yellow"></a>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://modelscope.cn/models/netease-youdao/Confucius4-TTS"><img src="https://img.shields.io/badge/ModelScope-Confucius4TTS-purple"></a>
    &nbsp;&nbsp;&nbsp;&nbsp;
</div>
<br>

Confucius4-TTS is an advanced LLM-based text-to-speech (TTS) system designed for multilingual and cross-lingual speech synthesis. Built on a speech encoder + large language model (LLM) architecture, Confucius4-TTS enables high-quality speech generation while preserving speaker identity across languages. You can try our online demo at **[https://confucius4-tts.youdao.com/gradio](https://confucius4-tts.youdao.com/gradio)**.

**✨ Key Features**

- **14 Languages Supported**: Chinese, English, Japanese, Korean, German, French, Spanish, Indonesian, Italian, Thai, Portuguese, Russian, Malay and Vietnamese *(more coming soon)*
- **Unconstrained Voice Cloning**: No reference transcript required
- **Cross-Lingual Voice Transfer**: Unaccented speech synthesis across 14 languages
- **Zero-Shot Voice Transfer**: Clone voices without additional training
- **Seamless Emotion Transfer**: Clone the feeling, not just the voice
- **Robust Generalization**: Stable performance in real-world multilingual scenarios

With strong cross-lingual generalization, Confucius4-TTS allows users to seamlessly switch languages while keeping the same voice, delivering fluent, natural, and expressive speech.

<div align="center">

Video Demo

<table border="0">
  <tr>
    <td>
      <video src="https://github.com/user-attachments/assets/2e2a4fc2-c8ef-4f12-89f7-55a6221200f1" controls width="100%"></video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/dacd356d-3bf5-4b06-9a2c-6ad5c24eb035" controls width="100%"></video>
    </td>
  </tr>
  <tr>
    <td>
      <video src="https://github.com/user-attachments/assets/e00ae5e1-fbb0-4137-af13-dd53599196a5" controls width="100%"></video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/f6c1aabb-4258-40ba-b945-81c3eeed67c3" controls width="100%"></video>
    </td>
  </tr>
</table>
</div>

## Contents

- [Installation](#-installation)
- [Inference](#-inference)
- [Training](#-training)
- [Performance](#-performance)
- [Citation](#citation)

## 🛠 Installation

### Requirements

- Python 3.10
- CUDA 12.6

### Setup

1. Clone the repository:

```bash
git clone https://github.com/netease-youdao/Confucius4-TTS.git
cd Confucius4-TTS
```

2. Create and activate a conda environment:

```bash
conda create -n confuciustts python=3.10 -y
conda activate confuciustts
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

For RTX PRO 6000 Blackwell deployment, install the CUDA 12.8 PyTorch wheels
after the base dependencies:

```bash
pip install -r requirements.txt
pip install --force-reinstall -r requirements-cu128.txt
```

## 🚀 Inference

Use the provided `example.py` script for zero-shot TTS synthesis:

```bash
python example.py \
    --prompt_wav path/to/reference.wav \
    --text "Hello, this is a test of zero-shot voice cloning." \
    --lang en \
    --out output.wav \
    --config config/inference_config.yaml
```

### Gradio vLLM Service

Launch the local Gradio interface. The Gradio entry point loads a single
long-lived vLLM-backed T2S engine at startup and reuses it for vLLM requests.
The "Generate original" button runs the original PyTorch backend in an isolated
subprocess so it does not share runtime state with the vLLM engine:

```bash
sudo apt install ffmpeg
pip install -r requirements.txt
pip install --force-reinstall -r requirements-cu128.txt
pip install -r requirements-vllm.txt
pip install "numpy<2"
pip install "https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.10-cp312/flash_attn-2.8.3%2Bcu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
pip install "torchcodec==0.9.*"

python tools/convert_t2s_vllm.py \
    --config config/inference_config.yaml \
    --output checkpoints/t2s-vllm

python gradio_app.py \
    --server-name 0.0.0.0 \
    --server-port 7860 \
    --device cuda \
    --config config/inference_config.yaml \
    --vllm-model-dir checkpoints/t2s-vllm \
    --vllm-gpu-memory-utilization 0.25 \
    --concurrency-limit 100
```

The optimized serving settings are defaults: vLLM uses automatic dtype
selection, S2A uses automatic reduced precision on CUDA, S2A length bucketing is
enabled, reference conditioning caches 16 audio prompts, and synchronized CUDA
stage timings stay off unless requested.

S2A diffusion uses `torch.compile` by default on CUDA. This can improve
repeated-generation throughput after the first compile/warmup pass, but startup
and the first request will take longer. Use `--no-compile-s2a` to disable it.
The launcher also accepts the IndexTTS-style alias `--use-torch-compile`.
The Gradio launcher enables PyTorch Inductor's persistent FX graph and
AOTAutograd caches by default under `outputs/compile-cache/torchinductor`, so
compiled S2A artifacts can be reused after a service restart. Override the
location with `--compile-cache-dir` or disable the launcher default with
`--no-compile-cache`.
The S2A compile path keeps Inductor kernels enabled but disables Inductor CUDA
graph capture because prompt/text-dependent dynamic shapes can otherwise create
many graph recordings and fail inside Gradio worker threads.

The Gradio service also runs a real startup generation by default, using
`resources/voice.mp3` and `Hello, welcome to Confucius4-TTS.`. This prewarms
audio loading, reference conditioning, the vLLM request path, S2A, and BigVGAN
before the first user clicks "Generate with vLLM"; use `--no-warmup` if you
prefer faster server startup over faster first-request latency. Override the
warmup reference or text with `--warmup-prompt-wav` and `--warmup-text`.

S2A DiT attention uses PyTorch SDPA. By default PyTorch chooses the backend, but
you can force a backend for validation with `--s2a-sdpa-backend flash`,
`efficient`, `math`, or `cudnn`. S2A dtype defaults to `auto`, which chooses
`bfloat16` on CUDA when supported and falls back to `float32` otherwise; use
`--s2a-dtype float32` if you need the previous conservative path. For
multi-segment rendering, `--s2a-length-bucket-size` defaults to `64` to group
similarly sized S2A batches and reduce padding. Use `--profile-cuda` to write torch
profiler traces for the S2A and BigVGAN stages under `outputs/profiles/`.

BigVGAN uses NVIDIA's fused CUDA activation kernel automatically on CUDA,
matching the IndexTTS fast path. Use `--no-use-bigvgan-cuda-kernel` to disable
it, or set `CONFUCIUS_USE_BIGVGAN_CUDA_KERNEL=0`. If the kernel cannot be built
or loaded, Confucius falls back to the plain torch BigVGAN implementation.
On RTX PRO 6000 / Blackwell, the vendored BigVGAN loader detects the active CUDA
device and builds an architecture-specific extension such as `sm_120`, avoiding
older hard-coded `compute_70` flags that recent CUDA toolchains reject.

Duration targets are specified in seconds and the final waveform is fitted at
sample precision, so millisecond-level dubbing targets are preserved. The
S2A/vocoder render batch size defaults to `1`, which is currently fastest in
testing; larger values remain available for experiments.

`requirements-vllm.txt` also installs this repository in editable mode. This
registers the Confucius custom T2S model as a `vllm.general_plugins` entry
point, which is required because the Gradio service uses spawned vLLM engine
workers.

New vLLM exports include the T2S prefix modules inside the vLLM worker. Re-run
`tools/convert_t2s_vllm.py` after upgrading to enable `--vllm-prefix-mode auto`
to avoid building large prefix embeddings in the serving process. Older exports
still work, but automatically fall back to `embeds` mode.

The vLLM backend can also avoid the duplicate PyTorch T2S transformer latent
pass when the installed vLLM build exposes hidden-state extraction. The default
`--vllm-latent-mode auto` tries that path and falls back to the PyTorch latent
pass if unsupported; use `--vllm-latent-mode pytorch` for the previous behavior
or `vllm` to require the new path.

Non-vLLM GPU stages are throttled with `--gpu-stage-concurrency` so Wav2Vec2,
CAMPPlus, S2A, and BigVGAN do not all run concurrently across many Gradio
requests. Reference conditioning is cached by audio content hash; tune the LRU
size with `--reference-cache-size`.

Gradio no longer requests synchronized per-stage CUDA timings by default. Use
`--detailed-timings` only when profiling; normal serving reports request-level
latency without forcing `_StepTimer` CUDA synchronizations.

By default vLLM selects the first compatible attention backend. On Blackwell
this normally means FlashInfer first, then FlashAttention, then Triton. The
vLLM requirements pin `flashinfer-python`. If you explicitly need to test
`FLASH_ATTN`, use a matching prebuilt wheel from
[mjun0812/flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels)
instead of source-building `flash-attn`. Match the wheel to the deployment
Python, CUDA/PyTorch, CXX11 ABI, and Linux platform tuple. For example, PyTorch
`+cu128` with a CUDA 13.0 `nvcc` will make `pip install flash-attn` fall back
to a source build and fail with a CUDA version mismatch.

To force an attention backend for testing, add one of:

```bash
--vllm-attention-backend FLASHINFER
--vllm-attention-backend FLASH_ATTN
```

You can verify the FlashInfer install with:

```bash
flashinfer show-config
```

The UI accepts a reference audio file, synthesis text, language selection,
and advanced generation settings. Generated WAV files are saved under
`outputs/gradio/`.

The service defaults to vLLM `auto` dtype for throughput. Use
`--vllm-dtype float32` if a deployment needs the previous conservative T2S
precision path.

### vLLM T2S Backend

The vLLM path accelerates the autoregressive Text2Semantic stage. Reference
audio encoding, S2A diffusion, and BigVGAN remain in PyTorch; S2A diffusion can
wrap its DiT estimator with `torch.compile`, while BigVGAN can use its fused
CUDA activation kernel for faster vocoding. Both optimizations are enabled by
default on CUDA.

The converted T2S directory is required before starting the service:

```bash
python tools/convert_t2s_vllm.py \
    --config config/inference_config.yaml \
    --output checkpoints/t2s-vllm
```

For API servers or Gradio queues, concurrent requests can share the same vLLM
engine so semantic decoding is batched by vLLM.

If `transformers` fails while importing `Wav2Vec2BertModel` with an error like
`operator torchvision::nms does not exist`, the Python environment has a
Torch/TorchVision mismatch. Reinstall the matching CUDA 12.8 package set:

```bash
pip install -r requirements.txt
pip install --force-reinstall -r requirements-cu128.txt
```

On Blackwell GPUs such as RTX PRO 6000, a PyTorch build that only supports
older architectures can fail during generation with
`CUDA error: no kernel image is available for execution on the device`.
Use the CUDA 12.8 PyTorch wheel set:

```bash
pip install -r requirements.txt
pip install --force-reinstall -r requirements-cu128.txt
```

If the stable CUDA 12.8 wheel still does not include your GPU architecture, use
the PyTorch nightly CUDA 12.8 wheel:

```bash
pip install -r requirements.txt
pip install --pre --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

You can also use the Python API directly:

```python
import torch
import torchaudio
from confuciustts.cli.inference import ConfuciusTTS

model = ConfuciusTTS(
    config_path="config/inference_config.yaml",
    device="cuda" if torch.cuda.is_available() else "cpu",
)

audio = model.generate(
    text="Hello, welcome to Confucius4-TTS.",
    lang="en",
    prompt_wav="path/to/reference.wav",
    verbose=True,
)

torchaudio.save("output.wav", audio.cpu(), model.sample_rate)
```

## 🚀 Fine-Tuning

Confucius4-TTS follows a "speech encoder + LLM" architecture. The training pipeline covers two modules:
- **Text2Semantic (T2S)**: generates semantic token sequences from text and speaker conditioning.
- **Semantic2Acoustic (S2A)**: a flow-matching model that converts semantic tokens into mel spectrograms.

### 1. Prepare Pretrained Models

Download the two external models:

```bash
# Wav2Vec2-BERT (speaker conditioning & semantic feature extraction)
huggingface-cli download facebook/w2v-bert-2.0 \
    --local-dir pretrained/w2v-bert-2.0

# Amphion MaskGCT (semantic codec implementation)
git clone https://github.com/open-mmlab/Amphion.git external/Amphion
```

After downloading, your directory should look like:

```
checkpoints/
├── t2s_model.safetensors        # pretrained T2S weights
├── s2a_model.pt                 # pretrained S2A weights
├── wav2vec2bert_stats.pt        # semantic feature normalization statistics
├── special_tokens_map.json      # tokenizer files
├── tokenizer.json
├── tokenizer.model
└── tokenizer_config.json
pretrained/
├── w2v-bert-2.0/                # Wav2Vec2-BERT model
└── campplus/
    └── campplus_cn_common.bin   # CAMPPlus speaker encoder checkpoint
external/
└── Amphion/                     # MaskGCT semantic codec implementation
```

### 2. Prepare Training Data

Training data is provided as **TSV files** (tab-separated, no header) with the following 5 columns:

| Column | Description |
|---|---|
| `lang` | Language code (e.g. `zh`, `en`, `ja`) |
| `wav_path` | Path to the target audio |
| `norm_text` | Normalized text |
| `semantic_ids_path` | Pre-extracted semantic tokens (`.npy` file path) |
| `ref_audio_paths` | Reference audio path(s), comma-separated for multiple |

Configure the train/validation paths in `config/train_t2s.yaml`:

```yaml
data:
  train_data_path:
    - data/train.tsv
  val_data_path:
    - data/val.tsv
```

### 3. Launch T2S Training

Set the pretrained T2S checkpoint path in `config/train_t2s.yaml`:

```yaml
paths:
  t2s_checkpoint: checkpoints/t2s_model.safetensors
```

**Single-node training:**

```bash
python -m confuciustts.cli.train_t2s -c config/train_t2s.yaml
```

### 4. Launch S2A Training

Set the checkpoint paths in `config/train_s2a.yaml`. `t2s_checkpoint` points to the frozen T2S backbone; `s2a_checkpoint` is optional and can be used to resume from a pretrained S2A model:

```yaml
paths:
  t2s_checkpoint: checkpoints/t2s_model.safetensors
  s2a_checkpoint: checkpoints/s2a_model.pt   # optional: resume from pretrained S2A
```

**Single-node training:**

```bash
python -m confuciustts.cli.train_s2a -c config/train_s2a.yaml
```

During S2A training, the T2S model, speaker encoder (Wav2Vec2-BERT), and style encoder (CAMPPlus) are all frozen. Only the flow-matching S2A model is trained.

## 📊 Performance

Confucius4-TTS achieves competitive results on multilingual and cross-lingual zero-shot TTS benchmarks, with strong intelligibility and speaker similarity across multiple languages.

> Lower is better for WER/CER (↓), and higher is better for SIM (↑).

### CV3-eval Cross-lingual

<details>
<summary><b>CV3-eval Cross-lingual Results (click to expand)</b></summary>

| Direction | Metric | Confucius4-TTS | F5-TTS† | Spark-TTS | CosyVoice2† | CosyVoice3-0.5B† | CosyVoice3-0.5B + DiffRO† | CosyVoice3-1.5B† | CosyVoice3-1.5B + DiffRO† |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| en→zh | WER↓ | **6.71** | 11.60 | 12.40 | 13.50 | 8.48 | 5.16 | 8.01 | 5.09 |
| ja→zh | WER↓ | 4.93 | – | – | 48.10 | 6.86 | 3.22 | 6.78 | **3.05** |
| ko→zh | WER↓ | 1.46 | – | – | 7.70 | 5.24 | **1.03** | 3.30 | 1.06 |
| zh→en | WER↓ | **3.19** | 5.57 | 7.36 | 17.10 | 6.83 | 4.41 | 5.39 | 4.20 |
| ja→en | WER↓ | **3.44** | – | – | 11.20 | 5.86 | 4.78 | 5.94 | 4.19 |
| ko→en | WER↓ | **3.42** | – | – | 13.10 | 18.30 | 7.91 | 13.70 | 7.08 |

† Requires reference text.

</details>

### X-Voice Benchmark

<details>
<summary><b>X-Voice Cross-lingual Results (click to expand)</b></summary>

| Direction | Metric | Confucius4-TTS | X-Voice | OmniVoice† | IndexTTS2 |
|---|---|---:|---:|---:|---:|
| de→zh | WER↓ | **2.86** | 3.07 | 13.10 | 3.46 |
|  | SIM↑ | 0.569 | 0.516 | **0.691** | 0.544 |
| en→zh | WER↓ | 3.27 | **3.06** | 4.03 | 3.78 |
|  | SIM↑ | 0.504 | 0.443 | **0.544** | 0.485 |
| fr→zh | WER↓ | **2.74** | 3.01 | 18.10 | 3.53 |
|  | SIM↑ | 0.550 | 0.518 | **0.686** | 0.543 |
| ja→zh | WER↓ | 3.50 | **3.39** | 79.10 | 4.11 |
|  | SIM↑ | 0.637 | 0.629 | **0.709** | 0.650 |
| ko→zh | WER↓ | **2.86** | 3.13 | 11.88 | 2.90 |
|  | SIM↑ | 0.649 | 0.655 | **0.718** | 0.650 |
| th→zh | WER↓ | 2.87 | **2.79** | 3.30 | 3.08 |
|  | SIM↑ | 0.623 | 0.614 | **0.661** | 0.622 |
| vi→zh | WER↓ | **2.75** | 2.78 | 10.51 | 2.98 |
|  | SIM↑ | 0.640 | 0.641 | **0.701** | 0.641 |

† Requires reference text.

</details>

### Seed-TTS-eval

<details>
<summary><b>Seed-TTS-eval English & Chinese Zero-shot Results (click to expand)</b></summary>

| Language | Metric | Confucius4-TTS | Qwen3-TTS | FishAudio S2† | OmniVoice† | VoxCPM2† | X-Voice |
|---|---|---:|---:|---:|---:|---:|---:|
| English | WER↓ | 1.49 | 1.24 | **0.99** | 1.60 | 1.84 | 1.91 |
|  | SIM↑ | 0.70 | 0.714 | – | 0.741 | **0.753** | 0.627 |
| Chinese | CER↓ | 0.94 | 0.77 | **0.54** | 0.84 | 0.97 | 1.47 |
|  | SIM↑ | 0.765 | 0.770 | – | 0.777 | **0.795** | 0.746 |

† Requires reference text.

</details>

### MiniMax-Multilingual-Test

<details>
<summary><b>MiniMax-Multilingual-Test Results (click to expand)</b></summary>

| Language | Metric | Confucius4-TTS | ElevenLab | Qwen3-TTS | FishAudio S2† | OmniVoice† | VoxCPM2† | X-Voice |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| German | WER↓ | **0.47** | 0.57 | 1.24 | 0.55 | 0.96 | 0.68 | 2.00 |
|  | SIM↑ | 0.775 | 0.614 | 0.768 | 0.767 | **0.812** | 0.803 | 0.763 |
| French | WER↓ | 3.66 | 5.22 | **2.86** | 3.05 | 3.35 | 4.53 | 4.73 |
|  | SIM↑ | 0.723 | 0.535 | 0.716 | 0.698 | **0.801** | 0.735 | 0.746 |
| Indonesian | WER↓ | 1.12 | **1.06** | – | 1.46 | 1.97 | 1.08 | 1.47 |
|  | SIM↑ | 0.765 | 0.660 | – | 0.763 | **0.805** | 0.800 | 0.725 |
| Korean | WER↓ | 1.84 | 1.87 | 1.76 | **1.18** | 2.65 | 1.96 | 2.27 |
|  | SIM↑ | 0.812 | 0.700 | 0.790 | 0.817 | 0.828 | **0.833** | 0.788 |
| Thai | WER↓ | **1.56** | 73.94 | – | 4.23 | 3.98 | 2.96 | 4.71 |
|  | SIM↑ | 0.773 | 0.588 | – | 0.786 | **0.841** | 0.840 | 0.791 |
| Japanese | WER↓ | 4.14 | 10.65 | 3.82 | **2.76** | 4.03 | 4.63 | 7.13 |
|  | SIM↑ | 0.788 | 0.738 | 0.771 | 0.796 | **0.828** | **0.828** | 0.765 |
| Vietnamese | WER↓ | 1.61 | 73.42 | – | 7.41 | **1.37** | 3.31 | 1.40 |
|  | SIM↑ | 0.751 | 0.369 | – | 0.740 | 0.805 | **0.806** | 0.672 |
| Italian | WER↓ | 1.30 | 1.74 | **0.95** | 1.27 | 2.07 | 1.56 | 2.27 |
|  | SIM↑ | 0.787 | 0.579 | 0.752 | 0.747 | **0.812** | 0.780 | 0.780 |
| Portuguese | WER↓ | 2.48 | 1.33 | 1.53 | **1.14** | 2.51 | 1.94 | 2.61 |
|  | SIM↑ | 0.796 | 0.711 | 0.805 | 0.781 | **0.859** | 0.837 | 0.794 |
| Spanish | WER↓ | 1.02 | 1.08 | 1.13 | **0.91** | 1.03 | 1.44 | 2.91 |
|  | SIM↑ | 0.778 | 0.615 | 0.814 | 0.776 | 0.804 | **0.831** | 0.747 |
| Russian | WER↓ | 4.64 | 3.88 | 3.21 | 2.40 | **2.23** | 3.63 | 6.49 |
|  | SIM↑ | 0.787 | 0.675 | 0.784 | 0.790 | 0.783 | **0.811** | 0.799 |

† Requires reference text.

</details>

---

## Acknowledgements

Confucius4-TTS builds on the following open-source projects:

- **[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)** — Speaker encoder (ECAPA-TDNN) and text embedding projector architectures
- **[CosyVoice](https://github.com/FunAudioLLM/CosyVoice)** — Text normalization pipeline
- **[Amphion / MaskGCT](https://github.com/open-mmlab/Amphion)** — Semantic codec implementation
- **[w2v-BERT 2.0](https://huggingface.co/facebook/w2v-bert-2.0)** — Semantic feature extraction and speaker conditioning
- **[Seed-VC](https://github.com/Plachtaa/seed-vc)** — Flow matching architecture reference
- **[BigVGAN](https://github.com/NVIDIA/BigVGAN)** — High-fidelity neural vocoder for mel-spectrogram to waveform synthesis

---

## Citation

If you find Confucius4-TTS useful in your research or project, please consider citing:

```bibtex
@misc{confucius4tts_2026,
  title        = {Confucius4-TTS: A Multilingual and Cross-Lingual Zero-Shot TTS Engine},
  author       = {{NetEase Youdao}},
  year         = {2026},
  howpublished = {\url{https://github.com/netease-youdao/Confucius4-TTS}},
  note         = {GitHub repository}
}
```
