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
