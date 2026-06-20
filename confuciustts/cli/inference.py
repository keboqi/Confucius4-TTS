import argparse
import os
import sys
import threading
import time
import traceback
import warnings
from collections import OrderedDict
from typing import Any, Optional

import safetensors.torch
import torch
import torchaudio
import yaml
from transformers import AutoTokenizer, SeamlessM4TFeatureExtractor, Wav2Vec2BertModel
from huggingface_hub import hf_hub_download

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from external.bigvgan.bigvgan import BigVGAN
from external.campplus import CAMPPlus

from confuciustts.flow.flow import MaskedDiffWithXvec, MaskedDiffWithXvecConfig
from confuciustts.frontend.text_normalizer import TextNormalizer
from confuciustts.llm.llm import Text2Semantic, Text2SemanticConfig
from confuciustts.utils.audio_features import mel_spectrogram
from confuciustts.utils.audio_post import cross_fade_concat
from confuciustts.utils.text_utils import get_language_token


class _StepTimer:
    def __init__(
        self,
        owner: "ConfuciusTTS",
        timings: Optional[OrderedDict[str, float]],
        name: str,
    ) -> None:
        self.owner = owner
        self.timings = timings
        self.name = name
        self.started = 0.0

    def __enter__(self) -> "_StepTimer":
        if self.timings is not None:
            self.owner._sync_device()
            self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.timings is not None:
            self.owner._sync_device()
            elapsed = time.perf_counter() - self.started
            self.timings[self.name] = self.timings.get(self.name, 0.0) + elapsed


class ConfuciusTTS:
    """Zero-shot multilingual TTS system based on a two-stage architecture.

    Workflow:
    1. Load reference audio to extract style and semantic conditioning
    2. Text → Semantic tokens (T2S model, LLM-based)
    3. Semantic tokens → Mel-spectrogram (S2A model, flow matching)
    4. Mel → Waveform (BigVGAN vocoder)

    Args:
        config_path: Path to YAML configuration file with model paths and audio parameters
        t2s_checkpoint: Optional path to T2S checkpoint (overrides config)
        device: Device for inference ("cuda" or "cpu")
        compile_s2a: Compile the S2A diffusion estimator with torch.compile. None enables it on CUDA.
        use_cuda_kernel: Use BigVGAN's fused CUDA activation kernel. None enables it on CUDA.
    """
    def __init__(
        self,
        config_path: str = "config/inference_config.yaml",
        t2s_checkpoint: Optional[str] = None,
        device: str = "cuda",
        use_vllm: bool = False,
        vllm_model_dir: Optional[str] = None,
        vllm_gpu_memory_utilization: float = 0.25,
        vllm_tensor_parallel_size: int = 1,
        vllm_dtype: str = "float32",
        vllm_attention_backend: Optional[str] = None,
        compile_s2a: Optional[bool] = None,
        use_cuda_kernel: Optional[bool] = None,
    ):
        self.device = torch.device(device)
        self.compile_s2a = self._resolve_compile_s2a(compile_s2a)
        self.use_cuda_kernel = self._resolve_bigvgan_cuda_kernel(use_cuda_kernel)
        self.t2s_vllm = None
        self._pytorch_t2s_lock = threading.Lock()

        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        paths = self.cfg["paths"]
        if t2s_checkpoint is not None:
            paths["t2s_checkpoint"] = t2s_checkpoint
        paths.setdefault("t2s_checkpoint", "checkpoints/model.safetensors")

        self.sample_rate = self.cfg["audio"]["target_sample_rate"]
        self.n_mels = self.cfg["audio"]["n_mels"]
        self.n_fft = self.cfg["audio"]["n_fft"]
        self.hop_length = self.cfg["audio"]["hop_length"]
        self.win_length = self.cfg["audio"]["win_length"]
        self.fmin = self.cfg["audio"]["fmin"]
        self.fmax = self.cfg["audio"]["fmax"]

        self.normalizer = TextNormalizer()

        if use_vllm:
            # Start vLLM before any CUDA context is created so forked engine
            # processes inherit the custom Confucius model registration.
            self._load_t2s_model(paths, move_to_device=False)
            if vllm_model_dir is None:
                vllm_model_dir = paths.get("t2s_vllm_dir", "./checkpoints/t2s-vllm")
            from confuciustts.llm.vllm_runtime import Text2SemanticVLLM

            self.t2s_vllm = Text2SemanticVLLM(
                self.t2s_model,
                model_dir=vllm_model_dir,
                gpu_memory_utilization=vllm_gpu_memory_utilization,
                tensor_parallel_size=vllm_tensor_parallel_size,
                dtype=vllm_dtype,
                attention_backend=vllm_attention_backend,
            )
            self.t2s_model.to(self.device)

        self.feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(paths["w2v_bert_path"])
        self.w2v_model = Wav2Vec2BertModel.from_pretrained(paths["w2v_bert_path"]).eval().to(self.device)
        stats = torch.load(paths["w2v_stat"], map_location="cpu")
        self.semantic_mean = stats["mean"].to(self.device)
        self.semantic_std = torch.sqrt(stats["var"]).to(self.device)

        spk_cfg = paths["style_encoder"]
        self.style_encoder = CAMPPlus(**spk_cfg.get("init_args", {}))
        style_encoder_path = hf_hub_download(
            "funasr/campplus", filename=spk_cfg["checkpoint"]
        )
        spk_state = torch.load(style_encoder_path, map_location="cpu")
        if isinstance(spk_state, dict) and "state_dict" in spk_state:
            spk_state = spk_state["state_dict"]
        self.style_encoder.load_state_dict(spk_state, strict=False)
        self.style_encoder.eval().to(self.device)

        if not use_vllm:
            self._load_t2s_model(paths, move_to_device=True)

        s2a_config = MaskedDiffWithXvecConfig(**self.cfg["s2a_model"])
        self.s2a_model = MaskedDiffWithXvec(s2a_config)
        s2a_model_path = hf_hub_download(
            "netease-youdao/Confucius4-TTS", filename=paths["s2a_checkpoint"]
        )
        self.s2a_model.load_state_dict(
            torch.load(s2a_model_path, map_location="cpu", weights_only=False)
        )
        self.s2a_model.eval().to(self.device)
        for param in self.s2a_model.parameters():
            param.requires_grad = False
        if self.compile_s2a:
            self._compile_s2a_estimator()

        self._preload_bigvgan_cuda_kernel()
        self.bigvgan = self._load_bigvgan(paths["vocoder_path"])
        self.bigvgan.remove_weight_norm()
        self.bigvgan.eval().to(self.device)
        for param in self.bigvgan.parameters():
            param.requires_grad = False

    def _load_t2s_model(self, paths: dict[str, Any], move_to_device: bool) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(paths["tokenizer_path"])
        t2s_config = Text2SemanticConfig(**self.cfg["t2s_model"])
        self.t2s_model = Text2Semantic(t2s_config)
        self.t2s_model.config.vocab_size = t2s_config.semantic_vocab_size

        t2s_model_path = hf_hub_download(
            "netease-youdao/Confucius4-TTS", filename=paths["t2s_checkpoint"]
        )
        self.t2s_model.load_state_dict(
            safetensors.torch.load_file(t2s_model_path, device="cpu")
        )
        self.t2s_model.eval()
        if move_to_device:
            self.t2s_model.to(self.device)

    def _sync_device(self) -> None:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def _resolve_bigvgan_cuda_kernel(self, requested: Optional[bool]) -> bool:
        if self.device.type != "cuda":
            if requested:
                print("[ConfuciusTTS] BigVGAN CUDA kernel requested but disabled on non-CUDA device.")
            return False
        return True if requested is None else bool(requested)

    def _resolve_compile_s2a(self, requested: Optional[bool]) -> bool:
        if requested is False:
            return False
        if not hasattr(torch, "compile"):
            if requested:
                raise RuntimeError("torch.compile is not available in this PyTorch build.")
            print("[ConfuciusTTS] torch.compile is not available; S2A compile disabled.")
            return False
        if self.device.type != "cuda":
            if requested:
                raise RuntimeError("S2A torch.compile is only enabled for CUDA inference.")
            return False
        return True

    def _preload_bigvgan_cuda_kernel(self) -> None:
        if not self.use_cuda_kernel:
            return
        try:
            from external.bigvgan.alias_free_activation.cuda import load

            anti_alias_activation_cuda = load.load()
            print("[ConfuciusTTS] Preloaded custom CUDA kernel for BigVGAN:", anti_alias_activation_cuda)
        except Exception:
            traceback.print_exc()
            print("[ConfuciusTTS] Failed to load custom CUDA kernel for BigVGAN. Falling back to torch.")
            self.use_cuda_kernel = False

    def _load_bigvgan(self, vocoder_path: str) -> BigVGAN:
        try:
            return BigVGAN.from_pretrained(
                vocoder_path,
                use_cuda_kernel=self.use_cuda_kernel,
            )
        except Exception:
            if not self.use_cuda_kernel:
                raise
            traceback.print_exc()
            print("[ConfuciusTTS] Failed to initialize BigVGAN with CUDA kernel. Falling back to torch.")
            self.use_cuda_kernel = False
            return BigVGAN.from_pretrained(vocoder_path, use_cuda_kernel=False)

    def _s2a_estimator_max_seq_len(self) -> Optional[int]:
        estimator = self.s2a_model.decoder.estimator
        estimator = getattr(estimator, "_orig_mod", estimator)
        return getattr(estimator, "max_seq_len", None)

    def _compile_s2a_estimator(self) -> None:
        print("[ConfuciusTTS] Compiling S2A diffusion estimator with torch.compile...")
        self.s2a_model.enable_torch_compile(
            fullgraph=True,
            dynamic=True,
            mode="reduce-overhead",
        )
        print("[ConfuciusTTS] S2A diffusion estimator compile wrapper is ready.")

    def _select_t2s_backend(self, use_vllm: Optional[bool]):
        if use_vllm is False:
            return self.t2s_model, "pytorch"
        if self.t2s_vllm is not None:
            return self.t2s_vllm, "vllm"
        if use_vllm is True:
            raise RuntimeError("vLLM T2S was requested, but the vLLM backend is not loaded.")
        return self.t2s_model, "pytorch"

    def _load_prompt(self, prompt_wav: str):
        """Load and resample reference audio to 16kHz and target sample rate.

        Args:
            prompt_wav: Path to reference audio file

        Returns:
            Tuple of (wav_16k, wav_tgt) resampled to 16kHz and target sample rate
        """
        wav, sr = torchaudio.load(prompt_wav)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav_16k = wav if sr == 16000 else torchaudio.functional.resample(wav, sr, 16000)
        wav_tgt = wav if sr == self.sample_rate else torchaudio.functional.resample(wav, sr, self.sample_rate)
        return wav_16k, wav_tgt

    def _ref_mel(self, wav_tgt: torch.Tensor) -> torch.Tensor:
        """Extract mel-spectrogram from reference audio for S2A conditioning.

        Args:
            wav_tgt: Waveform at target sample rate, shape (C, T)

        Returns:
            Mel-spectrogram with shape (1, T_mel, n_mels)
        """
        mel = mel_spectrogram(
            wav_tgt.to(self.device).float(),
            sample_rate=self.sample_rate,
            n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
            n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax,
        )
        return mel.transpose(1, 2).contiguous()

    def _extract_semantic(self, wav_16k: torch.Tensor) -> torch.Tensor:
        """Extract normalized semantic features from reference audio using Wav2Vec2-BERT.

        Args:
            wav_16k: Waveform at 16kHz, shape (1, T)

        Returns:
            Normalized hidden states from layer 17, shape (1, T_feat, D)
        """
        inputs = self.feature_extractor(
            wav_16k.squeeze(0).cpu().numpy(), sampling_rate=16000, return_tensors="pt"
        )
        input_features = inputs["input_features"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        outputs = self.w2v_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feats = outputs.hidden_states[17]  # Layer 17 hidden states
        return (feats - self.semantic_mean) / self.semantic_std

    def _extract_style(self, wav_16k: torch.Tensor) -> torch.Tensor:
        """Extract speaker style embedding using CAMPPlus encoder.

        Args:
            wav_16k: Waveform at 16kHz, shape (1, T)

        Returns:
            Style embedding, shape (1, D_style)
        """
        fbank = torchaudio.compliance.kaldi.fbank(
            wav_16k, num_mel_bins=80, sample_frequency=16000, dither=0.0
        )
        fbank = fbank - fbank.mean(dim=0, keepdim=True)
        return self.style_encoder(fbank.unsqueeze(0).to(self.device))

    def _text_vocab_size(self) -> int:
        return int(self.t2s_model.text_projector.embed.num_embeddings)

    def _validate_text_token_ids(
        self,
        token_ids: torch.Tensor,
        text: str,
        lang: str,
    ) -> None:
        vocab_size = self._text_vocab_size()
        if token_ids.numel() == 0:
            raise ValueError(f"Tokenizer returned no IDs for {lang!r} text: {text!r}")
        invalid = (token_ids < 0) | (token_ids >= vocab_size)
        if not invalid.any():
            return
        bad_ids = token_ids[invalid].detach().cpu().unique().tolist()
        raise ValueError(
            f"Tokenizer produced IDs outside the T2S text vocabulary for lang={lang!r}: "
            f"vocab_size={vocab_size}, invalid_ids={bad_ids[:16]}, text={text!r}. "
            "This would otherwise trigger a CUDA embedding index assert."
        )

    def _tokenize_segment(self, text: str, lang: str) -> torch.Tensor:
        lang_token = get_language_token(lang)
        formatted = f"You are a helpful assistant. {lang_token}:{text}"
        token_ids = self.tokenizer.encode(formatted, return_tensors="pt")
        self._validate_text_token_ids(token_ids, text, lang)
        return token_ids.to(self.device)

    def _sanitize_t2s_semantic_output(
        self,
        semantic_codes: torch.Tensor,
        lm_latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        codebook_size = int(self.s2a_model.input_embedding.embedding.num_embeddings)
        invalid = (semantic_codes < 0) | (semantic_codes >= codebook_size)
        if not invalid.any():
            return semantic_codes, lm_latent
        if semantic_codes.shape[0] != 1:
            raise ValueError(
                "Cannot sanitize batched T2S semantic output with variable invalid "
                f"positions: shape={tuple(semantic_codes.shape)}"
            )
        valid_mask = ~invalid[0]
        if not valid_mask.any():
            bad_ids = semantic_codes.detach().cpu().unique().tolist()
            raise ValueError(
                "T2S generated no valid acoustic semantic tokens for S2A: "
                f"codebook_size={codebook_size}, generated_ids={bad_ids[:16]}"
            )
        bad_ids = semantic_codes[invalid].detach().cpu().unique().tolist()
        warnings.warn(
            "T2S generated non-acoustic semantic token IDs and they were removed "
            f"before S2A: codebook_size={codebook_size}, invalid_ids={bad_ids[:16]}",
            RuntimeWarning,
            stacklevel=2,
        )
        return semantic_codes[:, valid_mask], lm_latent[:, valid_mask]

    def _generate_t2s(
        self,
        token_ids: torch.Tensor,
        semantic_features: torch.Tensor,
        temperature: float,
        top_p: float,
        top_k: int,
        num_beams: int,
        repetition_penalty: float,
        max_length: int,
        use_vllm: Optional[bool],
    ) -> tuple[Any, str]:
        t2s_backend, backend_name = self._select_t2s_backend(use_vllm)
        kwargs = {
            "text_inputs": token_ids,
            "condition_vector": semantic_features,
            "max_length": max_length,
            "num_beams": num_beams,
            "do_sample": True,
            "top_p": top_p,
            "top_k": top_k,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "early_stopping": True,
            "return_latent": True,
        }
        if backend_name == "pytorch":
            with self._pytorch_t2s_lock:
                return t2s_backend.generate(**kwargs), backend_name
        return t2s_backend.generate(**kwargs), backend_name

    def _duration_seconds_to_frames(self, duration_seconds: Optional[float]) -> Optional[int]:
        target_samples = self._duration_seconds_to_samples(duration_seconds)
        if target_samples is None:
            return None
        return max(1, int(round(target_samples / self.hop_length)))

    def _duration_seconds_to_samples(self, duration_seconds: Optional[float]) -> Optional[int]:
        if duration_seconds is None or duration_seconds <= 0:
            return None
        return max(1, int(round(duration_seconds * self.sample_rate)))

    def _fit_audio_to_duration(
        self,
        audio: torch.Tensor,
        target_duration_seconds: Optional[float],
    ) -> torch.Tensor:
        target_samples = self._duration_seconds_to_samples(target_duration_seconds)
        if target_samples is None:
            return audio
        current_samples = audio.shape[-1]
        if current_samples == target_samples:
            return audio
        if current_samples > target_samples:
            return audio[..., :target_samples].contiguous()
        pad_shape = (*audio.shape[:-1], target_samples - current_samples)
        padding = torch.zeros(pad_shape, device=audio.device, dtype=audio.dtype)
        return torch.cat([audio, padding], dim=-1)

    def _segment_duration_targets(
        self,
        segments: list[str],
        target_duration_seconds: Optional[float],
        target_segment_durations: Optional[list[Optional[float]]],
        cross_fade_duration: float,
    ) -> list[Optional[float]]:
        if target_segment_durations is not None:
            if len(target_segment_durations) != len(segments):
                raise ValueError(
                    "target_segment_durations must match the number of text segments "
                    f"({len(segments)}), got {len(target_segment_durations)}."
                )
            return [
                None if duration is None or duration <= 0 else float(duration)
                for duration in target_segment_durations
            ]

        if target_duration_seconds is None or target_duration_seconds <= 0:
            return [None] * len(segments)

        inter_segment_silence = 0.0
        if len(segments) > 1 and cross_fade_duration > 0:
            silence_samples = int(cross_fade_duration * self.sample_rate) // 3
            inter_segment_silence = (silence_samples / self.sample_rate) * (len(segments) - 1)
        available_duration = float(target_duration_seconds) - inter_segment_silence
        min_duration = len(segments) * self.hop_length / self.sample_rate
        if available_duration < min_duration:
            raise ValueError(
                "target_duration_seconds is too short for the segmented output after "
                "inter-segment silence. Reduce cross_fade_duration or increase the target."
            )

        weights = [
            max(1, len(self.tokenizer.tokenize(segment)))
            for segment in segments
        ]
        weight_total = sum(weights)
        return [
            available_duration * weight / weight_total
            for weight in weights
        ]

    def _log_t2s_output(
        self,
        semantic_codes: torch.Tensor,
        lm_latent: torch.Tensor,
        verbose: bool,
    ) -> None:
        if not (verbose or os.getenv("CONFUCIUS_VLLM_DEBUG_GENERATION")):
            return
        codes_cpu = semantic_codes.detach().cpu().flatten()
        unique, counts = torch.unique(codes_cpu, return_counts=True)
        order = torch.argsort(counts, descending=True)[:10]
        top_tokens = [
            (int(unique[idx]), int(counts[idx]))
            for idx in order
        ]
        print(
            "[ConfuciusTTS] T2S output: "
            f"tokens={semantic_codes.shape[1]}, "
            f"latent={tuple(lm_latent.shape)}, "
            f"first={codes_cpu[:16].tolist()}, "
            f"last={codes_cpu[-16:].tolist()}, "
            f"top={top_tokens}",
            flush=True,
        )

    @torch.no_grad()
    def _prepare_s2a_condition(
        self,
        t2s_out: dict[str, torch.Tensor],
        reference_mel: torch.Tensor,
        verbose: bool,
        target_duration_seconds: Optional[float] = None,
    ) -> tuple[torch.Tensor, int, int, int, Optional[int]]:
        semantic_codes = t2s_out["semantic_codes"]  # (B, T_semantic)
        lm_latent = t2s_out["latent"]  # (B, T_semantic, D_hidden)
        semantic_codes, lm_latent = self._sanitize_t2s_semantic_output(
            semantic_codes,
            lm_latent,
        )
        self._log_t2s_output(semantic_codes, lm_latent, verbose)

        semantic_tokens = int(semantic_codes.shape[1])
        target_samples = self._duration_seconds_to_samples(target_duration_seconds)
        target_length = self._duration_seconds_to_frames(target_duration_seconds)
        if target_length is None:
            target_length = int(semantic_tokens * 1.72)
        target_lengths = torch.tensor([target_length], device=self.device)

        semantic_emb = self.s2a_model.input_embedding(semantic_codes).transpose(1, 2)
        combined_features = torch.cat([lm_latent, semantic_emb], dim=-1)
        text_cond = self.s2a_model.encoder_proj(combined_features)
        cond_target, _ = self.s2a_model.length_regulator(text_cond, target_lengths)

        prompt_length = reference_mel.size(1)
        prompt_condition = self.s2a_model.prompt_cond.expand(1, prompt_length, -1)
        cat_condition = torch.cat([prompt_condition, cond_target], dim=1)
        total_length = prompt_length + target_length
        max_seq_len = self._s2a_estimator_max_seq_len()
        if max_seq_len is not None and total_length > max_seq_len:
            raise ValueError(
                f"Requested S2A duration needs {total_length} mel frames including "
                f"the prompt, but this decoder supports at most {max_seq_len}."
            )
        return cat_condition.squeeze(0), target_length, total_length, semantic_tokens, target_samples

    @torch.no_grad()
    def _decode_s2a_condition_batch(
        self,
        prepared_conditions: list[tuple[torch.Tensor, int, int, int, Optional[int]]],
        reference_mel: torch.Tensor,
        style_embedding: torch.Tensor,
        n_timesteps: int,
        inference_cfg_rate: float,
        timings: Optional[OrderedDict[str, float]] = None,
    ) -> tuple[list[torch.Tensor], int]:
        if not prepared_conditions:
            return [], 0

        max_total_length = max(total_length for _, _, total_length, _, _ in prepared_conditions)
        condition_dim = prepared_conditions[0][0].shape[-1]
        batch_size = len(prepared_conditions)
        cat_condition = torch.zeros(
            batch_size,
            max_total_length,
            condition_dim,
            device=self.device,
            dtype=prepared_conditions[0][0].dtype,
        )
        total_lengths = []
        target_lengths = []
        target_samples = []
        semantic_token_total = 0
        for index, (
            condition,
            target_length,
            total_length,
            semantic_tokens,
            target_sample_count,
        ) in enumerate(prepared_conditions):
            cat_condition[index, :total_length] = condition
            total_lengths.append(total_length)
            target_lengths.append(target_length)
            target_samples.append(target_sample_count)
            semantic_token_total += semantic_tokens

        total_lengths_tensor = torch.tensor(total_lengths, device=self.device, dtype=torch.long)
        prompt_length = reference_mel.size(1)
        prompt_batch = reference_mel.transpose(1, 2).expand(batch_size, -1, -1).contiguous()
        style_batch = style_embedding.expand(batch_size, -1).contiguous()

        with _StepTimer(self, timings, "s2a"):
            mel = self.s2a_model.decoder.forward(
                mu=cat_condition,
                x_lens=total_lengths_tensor,
                prompt=prompt_batch,
                spks=style_batch,
                n_timesteps=n_timesteps,
                inference_cfg_rate=inference_cfg_rate,
                temperature=1.0,
            )
            mel = mel[:, :, prompt_length:]
            for index, target_length in enumerate(target_lengths):
                if target_length < mel.shape[-1]:
                    mel[index, :, target_length:] = 0

        with _StepTimer(self, timings, "vocoder"):
            audio_batch = self.bigvgan(mel.float().to(self.device)).squeeze(1)

        samples_per_frame = audio_batch.shape[-1] / mel.shape[-1]
        chunks = []
        for index, (target_length, target_sample_count) in enumerate(zip(target_lengths, target_samples)):
            audio_length = (
                target_sample_count
                if target_sample_count is not None
                else int(round(target_length * samples_per_frame))
            )
            chunk = audio_batch[index : index + 1, :audio_length].contiguous()
            if chunk.shape[-1] < audio_length:
                pad = torch.zeros(
                    chunk.shape[0],
                    audio_length - chunk.shape[-1],
                    device=chunk.device,
                    dtype=chunk.dtype,
                )
                chunk = torch.cat([chunk, pad], dim=-1)
            chunks.append(chunk)
        return chunks, semantic_token_total

    @torch.no_grad()
    def _synth_audio_from_t2s(
        self,
        t2s_out: dict[str, torch.Tensor],
        reference_mel: torch.Tensor,
        style_embedding: torch.Tensor,
        n_timesteps: int,
        inference_cfg_rate: float,
        verbose: bool,
        target_duration_seconds: Optional[float] = None,
        timings: Optional[OrderedDict[str, float]] = None,
    ) -> tuple[torch.Tensor, int]:
        prepared = self._prepare_s2a_condition(
            t2s_out=t2s_out,
            reference_mel=reference_mel,
            verbose=verbose,
            target_duration_seconds=target_duration_seconds,
        )
        chunks, semantic_token_total = self._decode_s2a_condition_batch(
            prepared_conditions=[prepared],
            reference_mel=reference_mel,
            style_embedding=style_embedding,
            n_timesteps=n_timesteps,
            inference_cfg_rate=inference_cfg_rate,
            timings=timings,
        )
        return chunks[0], semantic_token_total

    @torch.no_grad()
    def _synth_segment(
        self,
        text: str,
        lang: str,
        semantic_features: torch.Tensor,
        style_embedding: torch.Tensor,
        reference_mel: torch.Tensor,
        temperature: float,
        top_p: float,
        top_k: int,
        num_beams: int,
        repetition_penalty: float,
        max_length: int,
        n_timesteps: int,
        inference_cfg_rate: float,
        verbose: bool,
        target_duration_seconds: Optional[float] = None,
        use_vllm: Optional[bool] = None,
        timings: Optional[OrderedDict[str, float]] = None,
    ) -> tuple[torch.Tensor, int]:
        """Synthesize audio for a single text segment using T2S and S2A models.

        Args:
            text: Input text segment
            lang: Language code (e.g., "zh", "en")
            semantic_features: Conditioning features from reference audio, shape (1, T_feat, D)
            style_embedding: Speaker style vector, shape (1, D_style)
            reference_mel: Reference mel-spectrogram, shape (1, T_mel, n_mels)
            temperature: Sampling temperature for T2S generation
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            num_beams: Beam search width
            repetition_penalty: Penalty for repeating tokens
            max_length: Maximum sequence length for T2S generation
            n_timesteps: Number of diffusion steps for S2A
            inference_cfg_rate: Classifier-free guidance rate
            verbose: Print debug info

        Returns:
            Generated waveform, shape (1, T_audio)
        """
        with _StepTimer(self, timings, "text_tokenize"):
            token_ids = self._tokenize_segment(text, lang)

        _, backend_name = self._select_t2s_backend(use_vllm)
        with _StepTimer(self, timings, f"t2s_{backend_name}"):
            t2s_out, _ = self._generate_t2s(
                token_ids=token_ids,
                semantic_features=semantic_features,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                max_length=max_length,
                use_vllm=use_vllm,
            )
        return self._synth_audio_from_t2s(
            t2s_out=t2s_out,
            reference_mel=reference_mel,
            style_embedding=style_embedding,
            n_timesteps=n_timesteps,
            inference_cfg_rate=inference_cfg_rate,
            verbose=verbose,
            target_duration_seconds=target_duration_seconds,
            timings=timings,
        )

    @torch.no_grad()
    def _synth_segments_with_parallel_vllm(
        self,
        segments: list[str],
        lang: str,
        semantic_features: torch.Tensor,
        style_embedding: torch.Tensor,
        reference_mel: torch.Tensor,
        temperature: float,
        top_p: float,
        top_k: int,
        num_beams: int,
        repetition_penalty: float,
        max_length: int,
        n_timesteps: int,
        inference_cfg_rate: float,
        verbose: bool,
        segment_render_batch_size: int,
        target_segment_durations: list[Optional[float]],
        timings: Optional[OrderedDict[str, float]] = None,
    ) -> tuple[list[torch.Tensor], int]:
        if self.t2s_vllm is None:
            raise RuntimeError("Parallel vLLM segment synthesis requires the vLLM backend.")

        with _StepTimer(self, timings, "text_tokenize"):
            token_ids_by_segment = [
                self._tokenize_segment(segment, lang)
                for segment in segments
            ]

        requests = [
            {
                "text_inputs": token_ids,
                "condition_vector": semantic_features,
                "max_length": max_length,
                "num_beams": num_beams,
                "do_sample": True,
                "top_p": top_p,
                "top_k": top_k,
                "temperature": temperature,
                "repetition_penalty": repetition_penalty,
                "early_stopping": True,
                "return_latent": True,
            }
            for token_ids in token_ids_by_segment
        ]

        if verbose:
            print(
                f"[ConfuciusTTS] submitting {len(requests)} segment(s) "
                "to vLLM in parallel"
            )
        with _StepTimer(self, timings, "t2s_vllm"):
            t2s_outputs = self.t2s_vllm.generate_many(requests)

        if segment_render_batch_size > 1:
            chunks = []
            semantic_token_total = 0
            for start in range(0, len(t2s_outputs), segment_render_batch_size):
                batch_outputs = t2s_outputs[start : start + segment_render_batch_size]
                if verbose:
                    end = start + len(batch_outputs)
                    print(
                        f"[ConfuciusTTS] rendering segments {start + 1}-{end}/"
                        f"{len(segments)} as a batch"
                    )
                with _StepTimer(self, timings, "s2a_prepare"):
                    prepared_conditions = [
                        self._prepare_s2a_condition(
                            t2s_out=t2s_out,
                            reference_mel=reference_mel,
                            verbose=verbose,
                            target_duration_seconds=target_segment_durations[start + offset],
                        )
                        for offset, t2s_out in enumerate(batch_outputs)
                    ]
                batch_chunks, batch_semantic_tokens = self._decode_s2a_condition_batch(
                    prepared_conditions=prepared_conditions,
                    reference_mel=reference_mel,
                    style_embedding=style_embedding,
                    n_timesteps=n_timesteps,
                    inference_cfg_rate=inference_cfg_rate,
                    timings=timings,
                )
                chunks.extend(batch_chunks)
                semantic_token_total += batch_semantic_tokens
            return chunks, semantic_token_total

        chunks = []
        semantic_token_total = 0
        for i, t2s_out in enumerate(t2s_outputs):
            if verbose:
                print(f"[ConfuciusTTS] rendering segment {i + 1}/{len(segments)}")
            audio, semantic_tokens = self._synth_audio_from_t2s(
                t2s_out=t2s_out,
                reference_mel=reference_mel,
                style_embedding=style_embedding,
                n_timesteps=n_timesteps,
                inference_cfg_rate=inference_cfg_rate,
                verbose=verbose,
                target_duration_seconds=target_segment_durations[i],
                timings=timings,
            )
            semantic_token_total += semantic_tokens
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            chunks.append(audio)

        return chunks, semantic_token_total

    @torch.no_grad()
    def generate(
        self,
        text: str,
        lang: str,
        prompt_wav: str,
        temperature: float = 0.8,
        top_p: float = 0.8,
        top_k: int = 30,
        num_beams: int = 3,
        repetition_penalty: float = 10.0,
        max_length: int = 1520,
        n_timesteps: int = 10,
        inference_cfg_rate: float = 0.7,
        max_text_tokens_per_segment: int = 80,
        segment_render_batch_size: int = 1,
        target_duration_seconds: Optional[float] = None,
        target_segment_durations: Optional[list[Optional[float]]] = None,
        cross_fade_duration: float = 0.3,
        edge_fade_duration: float = 0.1,
        edge_pad_duration: float = 0.1,
        verbose: bool = False,
        use_vllm: Optional[bool] = None,
        return_timings: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """Generate speech audio from text with voice cloning.

        Performs text normalization, segmentation, then synthesizes each segment
        independently and merges them with cross-fade.

        Args:
            text: Input text to synthesize
            lang: Language code (e.g., "zh", "en", "ja", "ko")
            prompt_wav: Path to reference audio for voice cloning
            temperature: Sampling temperature for T2S (higher = more diverse)
            top_p: Nucleus sampling probability threshold
            top_k: Top-k sampling parameter
            num_beams: Beam search width (1 = greedy)
            repetition_penalty: Penalty for repeating tokens (higher = less repetition)
            max_length: Maximum semantic token sequence length
            n_timesteps: Number of diffusion steps for S2A (more = higher quality, slower)
            inference_cfg_rate: Classifier-free guidance scale (0 = unconditional, higher = stronger guidance)
            max_text_tokens_per_segment: Maximum tokens per segment before splitting
            segment_render_batch_size: Number of vLLM segments to batch for S2A/vocoder rendering
            target_duration_seconds: Optional final output duration target in seconds
            target_segment_durations: Optional per-segment duration targets in seconds
            cross_fade_duration: Cross-fade duration between segments in seconds
            edge_fade_duration: Fade duration at start/end in seconds
            edge_pad_duration: Padding duration at edges in seconds
            verbose: Print processing info

        Returns:
            Generated audio waveform, shape (1, T_audio) at target sample rate
        """
        timings: Optional[OrderedDict[str, float]] = OrderedDict() if return_timings else None
        if timings is not None:
            self._sync_device()
        total_started = time.perf_counter()
        _, backend_name = self._select_t2s_backend(use_vllm)
        segment_render_batch_size = max(1, int(segment_render_batch_size))

        # Normalize text (punctuation, numbers, etc.)
        with _StepTimer(self, timings, "normalize_text"):
            text = self.normalizer.normalize(text, language=lang)
        if verbose:
            print(f"[ConfuciusTTS] normalized text: {text}")

        # Extract conditioning from reference audio
        with _StepTimer(self, timings, "load_prompt"):
            wav_16k, wav_tgt = self._load_prompt(prompt_wav)
        with _StepTimer(self, timings, "extract_semantic"):
            semantic_features = self._extract_semantic(wav_16k)
        with _StepTimer(self, timings, "extract_style"):
            style_embedding = self._extract_style(wav_16k)
        with _StepTimer(self, timings, "reference_mel"):
            reference_mel = self._ref_mel(wav_tgt)

        # Split long text into segments
        with _StepTimer(self, timings, "segment_text"):
            segments = self.normalizer.segment_text(
                text,
                tokenize_fn=self.tokenizer.tokenize,
                language=lang,
                max_tokens=max_text_tokens_per_segment,
            )
        if not segments:
            segments = [text]
        if verbose:
            print(f"[ConfuciusTTS] {len(segments)} segment(s)")

        if verbose:
            for i, seg in enumerate(segments):
                print(f"[ConfuciusTTS] segment {i + 1}/{len(segments)}: {seg!r}")

        segment_duration_targets = self._segment_duration_targets(
            segments=segments,
            target_duration_seconds=target_duration_seconds,
            target_segment_durations=target_segment_durations,
            cross_fade_duration=cross_fade_duration,
        )
        if verbose and any(duration is not None for duration in segment_duration_targets):
            formatted_durations = [
                None if duration is None else round(duration, 3)
                for duration in segment_duration_targets
            ]
            print(f"[ConfuciusTTS] target segment durations: {formatted_durations}")

        if backend_name == "vllm" and len(segments) > 1:
            chunks, semantic_token_total = self._synth_segments_with_parallel_vllm(
                segments=segments,
                lang=lang,
                semantic_features=semantic_features,
                style_embedding=style_embedding,
                reference_mel=reference_mel,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                max_length=max_length,
                n_timesteps=n_timesteps,
                inference_cfg_rate=inference_cfg_rate,
                verbose=verbose,
                segment_render_batch_size=segment_render_batch_size,
                target_segment_durations=segment_duration_targets,
                timings=timings,
            )
        else:
            # Synthesize each segment independently.
            chunks = []
            semantic_token_total = 0
            for seg, target_segment_duration in zip(segments, segment_duration_targets):
                audio, semantic_tokens = self._synth_segment(
                    seg, lang, semantic_features, style_embedding, reference_mel,
                    temperature, top_p, top_k, num_beams, repetition_penalty,
                    max_length, n_timesteps, inference_cfg_rate, verbose,
                    target_duration_seconds=target_segment_duration,
                    use_vllm=use_vllm,
                    timings=timings,
                )
                semantic_token_total += semantic_tokens
                if audio.dim() == 1:
                    audio = audio.unsqueeze(0)
                chunks.append(audio)

        # Merge segments with cross-fade
        with _StepTimer(self, timings, "merge_segments"):
            merged = cross_fade_concat(chunks, self.sample_rate,
                                       silence_duration=cross_fade_duration)
        if target_duration_seconds is not None and target_duration_seconds > 0:
            with _StepTimer(self, timings, "fit_target_duration"):
                merged = self._fit_audio_to_duration(merged, target_duration_seconds)

        if return_timings:
            self._sync_device()
            return merged, {
                "backend": "vLLM T2S" if backend_name == "vllm" else "Original PyTorch T2S",
                "segments": len(segments),
                "semantic_tokens": semantic_token_total,
                "target_duration_seconds": target_duration_seconds,
                "target_segment_durations": segment_duration_targets,
                "steps": timings,
                "total": time.perf_counter() - total_started,
            }
        return merged



def main():
    """CLI entry point for ConfuciusTTS inference."""
    def parse_duration_csv(value: str | None) -> list[float] | None:
        if value is None or not value.strip():
            return None
        durations = [
            float(part.strip())
            for part in value.replace("\n", ",").split(",")
            if part.strip()
        ]
        if any(duration <= 0 for duration in durations):
            raise ValueError("--target_segment_durations values must be positive seconds.")
        return durations or None

    parser = argparse.ArgumentParser(description="ConfuciusTTS zero-shot inference")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--lang", type=str, default="zh")
    parser.add_argument("--prompt_wav", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/inference_config.yaml")
    parser.add_argument("--t2s_checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_vllm", action="store_true",
                        help="Use vLLM for the autoregressive T2S semantic decoder.")
    parser.add_argument("--vllm_model_dir", type=str, default=None,
                        help="Converted T2S vLLM directory from tools/convert_t2s_vllm.py.")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.25)
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--vllm_dtype", type=str, default="float32")
    parser.add_argument("--vllm_attention_backend", type=str, default=None,
                        help="Optional vLLM attention backend override, e.g. FLASHINFER or FLASH_ATTN.")
    parser.add_argument("--cross_fade_duration", type=float, default=0.3)
    parser.add_argument("--edge_fade_duration", type=float, default=0.1)
    parser.add_argument("--edge_pad_duration", type=float, default=0.1)
    parser.add_argument("--segment_render_batch_size", type=int, default=1,
                        help="Number of vLLM text segments to batch for S2A/vocoder rendering.")
    parser.add_argument("--target_duration_seconds", type=float, default=0.0,
                        help="Optional final output duration target in seconds, sample-fitted for millisecond precision. 0 disables duration control.")
    parser.add_argument("--target_segment_durations", type=str, default="",
                        help="Comma-separated per-segment duration targets in seconds, sample-fitted for millisecond precision.")
    parser.add_argument("--compile_s2a", "--compile-s2a", "--use_torch_compile", "--use-torch-compile",
                        action=argparse.BooleanOptionalAction, default=None, dest="compile_s2a",
                        help="Compile the S2A diffusion estimator with torch.compile. Defaults to enabled on CUDA.")
    parser.add_argument("--use_bigvgan_cuda_kernel", "--use-bigvgan-cuda-kernel",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="Use BigVGAN's fused CUDA activation kernel. Defaults to enabled on CUDA.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

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
        compile_s2a=args.compile_s2a,
        use_cuda_kernel=args.use_bigvgan_cuda_kernel,
    )
    audio = model.generate(
        args.text, args.lang, args.prompt_wav,
        cross_fade_duration=args.cross_fade_duration,
        edge_fade_duration=args.edge_fade_duration,
        edge_pad_duration=args.edge_pad_duration,
        segment_render_batch_size=args.segment_render_batch_size,
        target_duration_seconds=args.target_duration_seconds,
        target_segment_durations=parse_duration_csv(args.target_segment_durations),
        verbose=args.verbose,
    )
    torchaudio.save(args.output, audio.cpu(), model.sample_rate)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
