<div align="center">
    <img src="./resources/Confucius4-TTS.png" alt="Confucius4-TTS" width="35%">
    <h1>Confucius4-TTS: 多语种跨语种零样本TTS</h1>
    <p><b>一种音色，任意语言。</b></p>
</div>

<div align="center">
    <a href="./README.md"><img src="https://img.shields.io/badge/README-EN-red"></a>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="./LICENSE"><img src="https://img.shields.io/badge/code_license-Apache%202.0-blue"></a>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://confucius4-tts.youdao.com/gradio"><img src="https://img.shields.io/badge/Demo-在线体验-purple"></a>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://2901733926.github.io/Confucius4-TTS/"><img src="https://img.shields.io/badge/GitHub.io-Demo_Page-blue?logo=GitHub&style=flat-square"></a>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://huggingface.co/netease-youdao/Confucius4-TTS"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow"></a>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://modelscope.cn/models/netease-youdao/Confucius4-TTS"><img src="https://img.shields.io/badge/ModelScope-Model-blue"></a>
    &nbsp;&nbsp;&nbsp;&nbsp;
</div>
<br>
Confucius4-TTS 是一款基于大语言模型（LLM）的先进文本转语音（TTS）系统，专为多语种和跨语种语音合成而设计。基于语音编码器 + 大语言模型（LLM）架构构建，能够在保持说话人音色一致的同时，实现跨语种的高质量语音生成。
在线 Demo 页面体验：[https://confucius4-tts.youdao.com/gradio]

**✨ 核心特性**

- **支持 14 种语言**：中文、英文、日语、韩语、德语、法语、西班牙语、印尼语、意大利语、泰语、葡萄牙语、俄语、马来语、越南语 *（更多语言即将推出）*
- **无约束声音克隆**：无需参考文本
- **跨语种声音迁移**：跨 14 种语言的无口音语音合成
- **零样本声音迁移**：无需额外训练即可克隆声音
- **无缝情感迁移**：克隆情感，而非仅仅是声音
- **强泛化能力**：在真实多语种场景中表现稳定

凭借强大的跨语种泛化能力，Confucius4-TTS 允许用户在保持相同音色的同时无缝切换语言，提供流畅、自然且富有表现力的语音。

## Contents

- [环境安装](#-环境安装)
- [推理](#-推理)
- [训练](#-训练)
- [性能](#-性能)
- [引用](#引用)

## 🛠 环境安装

### 环境要求

- Python 3.10
- CUDA 12.6

### 安装步骤

1. 克隆仓库：

```bash
git clone https://github.com/netease-youdao/Confucius4-TTS.git
cd Confucius4-TTS
```

2. 创建并激活 conda 环境：

```bash
conda create -n confuciustts python=3.10 -y
conda activate confuciustts
```

3. 安装依赖：

```bash
pip install -r requirements.txt
```

## 🚀 推理

使用提供的 `example.py` 脚本进行zero-shot TTS 合成：

```bash
python example.py \
    --prompt_wav path/to/reference.wav \
    --text "要合成的文本" \
    --lang zh \
    --out output.wav \
    --config config/inference_config.yaml
```

也可以直接使用 Python API：

```python
import torch
import torchaudio
from confuciustts.cli.inference import ConfuciusTTS

model = ConfuciusTTS(
    config_path="config/inference_config.yaml",
    device="cuda" if torch.cuda.is_available() else "cpu",
)

audio = model.generate(
    text="你好，欢迎使用 Confucius4-TTS。",
    lang="zh",
    prompt_wav="path/to/reference.wav",
    verbose=True,
)

torchaudio.save("output.wav", audio.cpu(), model.sample_rate)
```

### Gradio vLLM 服务

启动本地 Gradio 界面。Gradio 入口点在启动时加载一个长时间运行的、基于 vLLM 的 T2S 引擎，并将其复用于 vLLM 请求。“Generate original” 按钮在一个独立的子进程中运行原始的 PyTorch 后端，因此它不会与 vLLM 引擎共享运行时状态：

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

### FastAPI vLLM 服务

对于编程式客户端，请使用 FastAPI 服务。它复用了与 Gradio 服务相同的长时间运行的 vLLM 模型设置，但主 API 路径直接保存 WAV 输出，并跳过了仅用于 UI 的 MP3 预览：

```bash
bash scripts/run_fastapi_uv.sh
```

该脚本使用 `uv` 创建了一个独立的 `.venv-fastapi` 虚拟环境，安装与 Gradio 快速启动使用的相同的 CUDA 和 vLLM 软件栈，当缺失 `checkpoints/t2s-vllm` 时进行转换，然后在 `0.0.0.0:8000` 上启动 `fastapi_app.py`。启动预热默认在后台运行，因此模型加载完成后 HTTP 服务即可立即可用，同时服务器仍会为其后续的请求进行预热。这可以将后端 TTS 服务的依赖从主机 Python 环境中隔离出来。

常见的环境变量覆盖项：

```bash
HOST=127.0.0.1 PORT=8010 VENV_DIR=.venv-confucius-api bash scripts/run_fastapi_uv.sh
WARMUP=foreground bash scripts/run_fastapi_uv.sh
WARMUP=0 bash scripts/run_fastapi_uv.sh
CONVERT_VLLM=1 bash scripts/run_fastapi_uv.sh
INSTALL_FFMPEG=0 INSTALL_FLASH_ATTN=0 bash scripts/run_fastapi_uv.sh --no-warmup
```

返回输出路径和下载 URL 的 JSON 响应：

```bash
curl -X POST http://127.0.0.1:8000/v1/tts \
    -H "Content-Type: application/json" \
    -d '{
      "text": "Hello, this is a test of zero-shot voice cloning.",
      "lang": "en"
    }'
```

当省略或清空 `prompt_wav` 时，API 会使用 `resources/voice.mp3` 作为默认参考声音。API 默认返回 MP3，这与 Gradio 的预览输出相匹配。当客户端需要 WAV 文件时，请设置 `"output_format": "wav"`。
FastAPI 预热默认使用与浏览器测试页面相同的文本，即 `Hello, this is a test of zero-shot voice cloning.`，并且 `GET /health` 会报告当前的预热状态。

直接获取 MP3 响应：

```bash
curl -X POST http://127.0.0.1:8000/v1/tts/audio \
    -H "Content-Type: application/json" \
    -d '{"text":"Hello from the API.","lang":"en"}' \
    --output output.mp3
```

直接获取 WAV 响应：

```bash
curl -X POST http://127.0.0.1:8000/v1/tts/audio \
    -H "Content-Type: application/json" \
    -d '{"text":"Hello from the API.","lang":"en","output_format":"wav"}' \
    --output output.wav
```

上传带有 JSON 数据的参考音频文件：

```bash
curl -X POST http://127.0.0.1:8000/v1/tts/upload/audio \
    -F 'payload={"text":"Hello with an uploaded voice.","lang":"en"}' \
    -F prompt_wav=@resources/voice.mp3 \
    --output output.mp3
```

生成的文件默认存储在 `outputs/api` 下。可以通过 `--output-dir` 或 `CONFUCIUS_API_OUTPUT_DIR` 覆盖它。该服务还暴露了用于浏览器测试的 `GET /ui`，`GET /health`，位于 `/docs` 的交互式文档，以及用于访问之前生成的 WAV 文件的 `GET /v1/audio/{file_name}`。

优化的服务设置默认为：vLLM 使用自动 dtype 选择，S2A 在 CUDA 上自动使用降低精度计算，S2A 启用长度分桶（length bucketing），参考条件缓存 100 个音频提示，并且除非请求否则关闭同步的 CUDA 阶段计时。

S2A diffusion 默认在 CUDA 上使用 `torch.compile`。这可以提高首次编译/预热后的重复生成吞吐量，但启动和首次请求将花费更长时间。使用 `--no-compile-s2a` 来禁用它。启动器还接受 IndexTTS 风格的别名 `--use-torch-compile`。
Gradio 启动器默认在 `outputs/compile-cache/torchinductor` 下启用 PyTorch Inductor 的持久化 FX 图和 AOTAutograd 缓存，因此编译的 S2A 制品可以在服务重启后复用。通过 `--compile-cache-dir` 覆盖缓存位置或通过 `--no-compile-cache` 禁用该启动器默认行为。
S2A 编译路径保持启用了 Inductor kernels，但禁用了 Inductor CUDA 图捕获（CUDA graph capture），因为依赖于提示音/文本的动态形状可能会导致创建过多的图形记录并在 Gradio 工作线程内失败。

Gradio 服务也默认运行一次真正的启动生成，使用 `resources/voice.mp3` 和 `Hello, welcome to Confucius4-TTS.`。这在首个用户点击“Generate with vLLM”之前预热音频加载、参考条件、vLLM 请求路径、S2A 和 BigVGAN；如果您更倾向于快速的服务器启动而非快速的首次请求延迟，请使用 `--no-warmup`。可以使用 `--warmup-prompt-wav` 和 `--warmup-text` 覆盖预热参考音频或文本。

S2A DiT 注意力机制使用 PyTorch SDPA。默认情况下 PyTorch 会选择后端，但您可以强制指定后端用于验证，如 `--s2a-sdpa-backend flash`、`efficient`、`math` 或 `cudnn`。S2A 数据类型默认为 `auto`，它会在支持的 CUDA 设备上选择 `bfloat16` 并在其他情况下回退到 `float32`；如果您需要以前的保守路径，请使用 `--s2a-dtype float32`。对于多段合成渲染，`--s2a-length-bucket-size` 默认为 `64`，以便将大小相近的 S2A 批处理进行分组并减少填充。使用 `--profile-cuda` 可将 S2A 和 BigVGAN 阶段的 torch 分析器跟踪写入 `outputs/profiles/`。

BigVGAN 自动在 CUDA 上使用 NVIDIA 的融合 CUDA 激活核，这与 IndexTTS 的快速路径相匹配。使用 `--no-use-bigvgan-cuda-kernel` 将其禁用，或设置 `CONFUCIUS_USE_BIGVGAN_CUDA_KERNEL=0`。如果无法构建或加载该内核，Confucius 会回退到纯 torch 的 BigVGAN 实现。
在 RTX PRO 6000 / Blackwell 上，随附的 BigVGAN 加载器会检测当前活动的 CUDA 设备并构建特定于架构的扩展，例如 `sm_120`，从而避免最新的 CUDA 工具链拒绝旧有的硬编码 `compute_70` 标志。

目标时长以秒为单位指定，并且最终波形在采样精度级别拟合，因此保留了毫秒级的配音目标。S2A/声码器的渲染批大小默认为 `1`，这在目前的测试中是最快的；仍保留了更大的值用于实验。

`requirements-vllm.txt` 也会以可编辑模式安装此仓库。这会将 Confucius 的自定义 T2S 模型注册为 `vllm.general_plugins` 入口点，这是必须的，因为 Gradio 服务使用的是生成的 vLLM 引擎工作进程。

新的 vLLM 导出会将 T2S 前缀模块包含在 vLLM worker 内部。升级后请重新运行 `tools/convert_t2s_vllm.py` 以启用 `--vllm-prefix-mode auto`，避免在服务进程中构建大型前缀嵌入。旧的导出仍然有效，但会自动回退到 `embeds` 模式。

如果安装的 vLLM 构建版本暴露了隐藏状态提取，vLLM 后端也可以避免重复的 PyTorch T2S transformer 潜层计算。默认的 `--vllm-latent-mode auto` 会尝试该路径，如果不支持则回退到 PyTorch 的潜层计算；使用 `--vllm-latent-mode pytorch` 保持先前的行为，或使用 `vllm` 强制使用新路径。

非 vLLM 的 GPU 阶段受 `--gpu-stage-concurrency` 限制，因此 Wav2Vec2、CAMPPlus、S2A 和 BigVGAN 不会在大量 Gradio 请求中全部并发运行。参考条件通过音频内容哈希进行缓存；使用 `--reference-cache-size` 调整 LRU 大小。

Gradio 不再默认请求同步每个阶段的 CUDA 耗时。仅在进行性能分析时使用 `--detailed-timings`；正常的服务会在不强制 `_StepTimer` CUDA 同步的情况下报告请求级延迟。

默认情况下，vLLM 选择第一个兼容的注意力机制后端。在 Blackwell 架构上，这通常意味着首先选择 FlashInfer，然后是 FlashAttention，最后是 Triton。vLLM 需求固定了 `flashinfer-python` 版本。如果您明确需要测试 `FLASH_ATTN`，请从
[mjun0812/flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels)
使用匹配的预构建 wheel 包，而不是从源码构建 `flash-attn`。需将 wheel 包与部署环境的 Python、CUDA/PyTorch、CXX11 ABI 以及 Linux 平台相匹配。例如，带有 CUDA 13.0 `nvcc` 的 PyTorch `+cu128` 环境会导致 `pip install flash-attn` 回退至源码构建，并因 CUDA 版本不匹配而失败。

为了强制指定测试使用的注意力机制后端，可添加以下参数之一：

```bash
--vllm-attention-backend FLASHINFER
--vllm-attention-backend FLASH_ATTN
```

您可以通过以下命令验证 FlashInfer 是否安装成功：

```bash
flashinfer show-config
```

UI 界面接受参考音频文件、合成文本、语言选择以及高级生成设置。生成的 WAV 文件保存在 `outputs/gradio/` 目录下。

为提升吞吐量，服务默认将 vLLM 设置为 `auto` dtype。如果部署环境需要以前保守的 T2S 精度路径，请使用 `--vllm-dtype float32`。

### vLLM T2S 后端

vLLM 路径加速了自回归的 Text2Semantic 阶段。参考音频编码、S2A diffusion 及 BigVGAN 仍在 PyTorch 中运行；S2A diffusion 可以通过 `torch.compile` 封装其 DiT 估计器，同时 BigVGAN 可以使用其融合的 CUDA 激活内核来加速声码器。这两种优化都在 CUDA 上默认启用。

启动服务前需要转换后的 T2S 目录：

```bash
python tools/convert_t2s_vllm.py \
    --config config/inference_config.yaml \
    --output checkpoints/t2s-vllm
```

对于 API 服务器或 Gradio 队列，并发请求可以共享同一个 vLLM 引擎，因此语义解码由 vLLM 进行批处理。

如果 `transformers` 在导入 `Wav2Vec2BertModel` 时失败并报错如 `operator torchvision::nms does not exist`，则说明 Python 环境中的 Torch/TorchVision 版本不匹配。请重新安装匹配的 CUDA 12.8 软件包系列：

```bash
pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## 🚀 微调

Confucius4-TTS 采用「语音编码器 + LLM」架构，训练流程涵盖以下两个模块：
- **Text2Semantic（T2S）**：根据文本与说话人条件生成语义 token 序列。
- **Semantic2Acoustic（S2A）**：流匹配模型，将语义 token 转换为梅尔频谱图。

### 1. 准备预训练模型

下载两个外部模型：

```bash
# Wav2Vec2-BERT（说话人条件化 & 语义特征提取）
huggingface-cli download facebook/w2v-bert-2.0 \
    --local-dir pretrained/w2v-bert-2.0

# Amphion MaskGCT（语义编解码器实现）
git clone https://github.com/open-mmlab/Amphion.git external/Amphion
```

下载完成后，目录结构如下：

```
checkpoints/
├── t2s_model.safetensors        # T2S 预训练权重
├── s2a_model.pt                 # S2A 预训练权重
├── wav2vec2bert_stats.pt        # 语义特征归一化统计量
├── special_tokens_map.json      # 分词器文件
├── tokenizer.json
├── tokenizer.model
└── tokenizer_config.json
pretrained/
├── w2v-bert-2.0/                # Wav2Vec2-BERT 模型
└── campplus/
    └── campplus_cn_common.bin   # CAMPPlus 说话人编码器权重
external/
└── Amphion/                     # MaskGCT 语义编解码器实现
```

### 2. 准备训练数据

训练数据为 **TSV 文件**（制表符分隔），不含表头，包含以下 5 列：

| 列名 | 说明 |
|---|---|
| `lang` | 语言代码（如 `zh`、`en`、`ja`） |
| `wav_path` | 目标音频路径 |
| `norm_text` | 归一化后的文本 |
| `semantic_ids_path` | 预提取的语义 token（`.npy` 文件路径） |
| `ref_audio_paths` | 参考音频路径，支持多个用逗号分隔 |

在 `config/train_t2s.yaml` 中配置训练/验证集路径：

```yaml
data:
  train_data_path:
    - data/train.tsv
  val_data_path:
    - data/val.tsv
```

### 3. 启动 T2S 训练

在 `config/train_t2s.yaml` 中设置预训练 T2S 权重路径：

```yaml
paths:
  t2s_checkpoint: checkpoints/t2s_model.safetensors
```

**单机训练：**

```bash
python -m confuciustts.cli.train_t2s -c config/train_t2s.yaml
```

### 4. 启动 S2A 训练

在 `config/train_s2a.yaml` 中设置权重路径。`t2s_checkpoint` 指向冻结的 T2S 骨干网络；`s2a_checkpoint` 为可选项，用于从预训练 S2A 模型继续训练：

```yaml
paths:
  t2s_checkpoint: checkpoints/t2s_model.safetensors
  s2a_checkpoint: checkpoints/s2a_model.pt   # 可选：从预训练 S2A 权重继续训练
```

**单机训练：**

```bash
python -m confuciustts.cli.train_s2a -c config/train_s2a.yaml
```

S2A 训练过程中，T2S 模型、说话人编码器（Wav2Vec2-BERT）和风格编码器（CAMPPlus）均处于冻结状态，只有流匹配 S2A 模型参与训练。

## 📊 性能

Confucius4-TTS 在多语种及跨语种零样本 TTS 基准测试中表现优异，兼具高可懂度与说话人相似度。

> WER/CER 越低越好（↓），SIM 越高越好（↑）。

### CV3-eval 跨语种

<details>
<summary><b>CV3-eval 跨语种结果（点击展开）</b></summary>

| Direction | Metric | Confucius4-TTS | F5-TTS† | Spark-TTS | CosyVoice2† | CosyVoice3-0.5B† | CosyVoice3-0.5B + DiffRO† | CosyVoice3-1.5B† | CosyVoice3-1.5B + DiffRO† |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| en→zh | WER↓ | **6.71** | 11.60 | 12.40 | 13.50 | 8.48 | 5.16 | 8.01 | 5.09 |
| ja→zh | WER↓ | 4.93 | – | – | 48.10 | 6.86 | 3.22 | 6.78 | **3.05** |
| ko→zh | WER↓ | 1.46 | – | – | 7.70 | 5.24 | **1.03** | 3.30 | 1.06 |
| zh→en | WER↓ | **3.19** | 5.57 | 7.36 | 17.10 | 6.83 | 4.41 | 5.39 | 4.20 |
| ja→en | WER↓ | **3.44** | – | – | 11.20 | 5.86 | 4.78 | 5.94 | 4.19 |
| ko→en | WER↓ | **3.42** | – | – | 13.10 | 18.30 | 7.91 | 13.70 | 7.08 |

† 需要参考文本。

</details>

### X-Voice Benchmark

<details>
<summary><b>X-Voice 跨语种结果（点击展开）</b></summary>

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

† 需要参考文本。

</details>

### Seed-TTS-eval

<details>
<summary><b>Seed-TTS-eval 中英文测试集结果（点击展开）</b></summary>

| Language | Metric | Confucius4-TTS | Qwen3-TTS | FishAudio S2† | OmniVoice† | VoxCPM2† | X-Voice |
|---|---|---:|---:|---:|---:|---:|---:|
| English | WER↓ | 1.49 | 1.24 | **0.99** | 1.60 | 1.84 | 1.91 |
|  | SIM↑ | 0.70 | 0.714 | – | 0.741 | **0.753** | 0.627 |
| Chinese | CER↓ | 0.94 | 0.77 | **0.54** | 0.84 | 0.97 | 1.47 |
|  | SIM↑ | 0.765 | 0.770 | – | 0.777 | **0.795** | 0.746 |

† 需要参考文本。

</details>

### MiniMax-Multilingual-Test

<details>
<summary><b>MiniMax-Multilingual-Test 结果（点击展开）</b></summary>

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

† 需要参考文本。

</details>

---

## 致谢

Confucius4-TTS 基于以下开源项目构建：

- **[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)** — 说话人编码器（ECAPA-TDNN）及文本嵌入投影层架构
- **[CosyVoice](https://github.com/FunAudioLLM/CosyVoice)** — 文本归一化流程
- **[Amphion / MaskGCT](https://github.com/open-mmlab/Amphion)** — 语义编解码器实现
- **[w2v-BERT 2.0](https://huggingface.co/facebook/w2v-bert-2.0)** — 语义特征提取与说话人条件化
- **[Seed-VC](https://github.com/Plachtaa/seed-vc)** — Flow matching 架构参考
- **[BigVGAN](https://github.com/NVIDIA/BigVGAN)** — 高保真神经声码器，用于梅尔频谱图到波形的合成

---

## 引用

如果您在研究或项目中使用了 Confucius4-TTS，请考虑引用：

```bibtex
@misc{confucius4tts_2026,
  title        = {Confucius4-TTS: A Multilingual and Cross-Lingual Zero-Shot TTS Engine},
  author       = {{NetEase Youdao}},
  year         = {2026},
  howpublished = {\url{https://github.com/netease-youdao/Confucius4-TTS}},
  note         = {GitHub repository}
}
```
