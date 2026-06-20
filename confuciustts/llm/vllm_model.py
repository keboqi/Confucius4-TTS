"""vLLM model adapter for Confucius4-TTS Text2Semantic decoding.

This module is intentionally imported only by the optional vLLM runtime.
It registers a GPT-2-like causal LM whose prompt is supplied as precomputed
embeddings and whose decoded tokens are Confucius semantic codes.
"""

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple, Union

import torch
from torch import nn
from transformers import BatchFeature

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.gpt2 import GPT2Block
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    _merge_multimodal_embeddings,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY, ModalityData
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import (
    AudioItem,
    DictEmbeddingItems,
    ModalityDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.processing.processor import (
    MultiModalPromptUpdates,
    PlaceholderFeaturesInfo,
    apply_token_matches,
    find_mm_placeholders,
)
from vllm.sequence import IntermediateTensors

from confuciustts.llm.position_embeddings import LearnedPositionalEmbedding
from confuciustts.llm.speaker_encoder import (
    Qwen3TTSSpeakerEncoder,
    Qwen3TTSSpeakerEncoderConfig,
)
from confuciustts.llm.text_encoder import TextEmbeddingProjector


PLACEHOLDER_TOKEN = "!"
PLACEHOLDER_TOKEN_ID = 8192


def _placeholder_token_id_from_config(config: object) -> int:
    return int(getattr(config, "start_semantic_token", PLACEHOLDER_TOKEN_ID))


def _config_get(config: Any, key: str, default: Any) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


class ConfuciusTTSProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"audio": None}

    def get_data_parser(self) -> MultiModalDataParser:
        return ConfuciusTTSDataParser()


class ConfuciusTTSDummyInputsBuilder(
    BaseDummyInputsBuilder[ConfuciusTTSProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return PLACEHOLDER_TOKEN * mm_counts.get("audio", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Optional[Mapping[str, BaseDummyOptions]] = None,
    ) -> Dict[str, Any]:
        num_items = mm_counts.get("audio", 0)
        if num_items == 0:
            return {}

        config = self.info.get_hf_config()
        dummy_seq_len = min(max(seq_len, 1), 1024)
        dummy_embed = torch.rand(
            (dummy_seq_len, config.n_embd),
            dtype=torch.float16,
        )
        return {"audio": {"audio_embeds": [dummy_embed] * num_items}}


class ConfuciusTTSDataParser(MultiModalDataParser):
    def _parse_audio_data(
        self,
        data: ModalityData[AudioItem],
    ) -> Optional[ModalityDataItems[Any, Any]]:
        if isinstance(data, dict):
            if "audio_embeds" in data:
                return DictEmbeddingItems(
                    data,
                    modality="audio",
                    required_fields={"audio_embeds"},
                    fields_factory=lambda hf_inputs: dict(
                        audio_embeds=MultiModalFieldConfig.batched("audio")
                    ),
                )
            if "text_inputs" in data and "condition_vector" in data:
                return DictEmbeddingItems(
                    data,
                    modality="audio",
                    required_fields={"text_inputs", "condition_vector"},
                    fields_factory=lambda hf_inputs: dict(
                        text_inputs=MultiModalFieldConfig.batched("audio"),
                        condition_vector=MultiModalFieldConfig.batched("audio"),
                    ),
                )
            return DictEmbeddingItems(
                data,
                modality="audio",
                required_fields={"audio_embeds"},
                fields_factory=lambda hf_inputs: dict(
                    audio_embeds=MultiModalFieldConfig.batched("audio")
                ),
            )
        raise TypeError(
            "For the Confucius T2S vLLM adapter, expected audio multimodal "
            f"data shaped like {{'audio_embeds': tensor}}, got {type(data)}"
        )


class ConfuciusTTSMultiModalProcessor(
    BaseMultiModalProcessor[ConfuciusTTSProcessingInfo]
):
    def _placeholder_token_id(self) -> int:
        return _placeholder_token_id_from_config(self.info.get_hf_config())

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        return BatchFeature(
            {
                "input_ids": torch.full(
                    (1, len(prompt)),
                    self._placeholder_token_id(),
                    dtype=torch.long,
                )
            }
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            audio_embeds=MultiModalFieldConfig.batched("audio"),
            text_inputs=MultiModalFieldConfig.batched("audio"),
            condition_vector=MultiModalFieldConfig.batched("audio"),
        )

    def _get_prompt_updates(
        self,
        mm_items: "MultiModalDataItems",
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> List[PromptUpdate]:
        out_mm_data = out_mm_kwargs.get_data()
        placeholder_token_id = self._placeholder_token_id()

        def get_replacement(item_idx: int) -> PromptUpdateDetails:
            if "audio_embeds" in out_mm_data:
                embeds = out_mm_data["audio_embeds"][item_idx]
                replacement_len = embeds.shape[0]
            else:
                text_inputs = out_mm_data["text_inputs"][item_idx]
                replacement_len = int(text_inputs.numel()) + 2
            return PromptUpdateDetails.select_token_id(
                [placeholder_token_id] * replacement_len,
                placeholder_token_id,
            )

        return [
            PromptReplacement(
                modality="audio",
                target=[placeholder_token_id],
                replacement=get_replacement,
            )
        ]

    def _find_mm_placeholders(
        self,
        new_token_ids: List[int],
        mm_prompt_updates: MultiModalPromptUpdates,
    ) -> Mapping[str, List[PlaceholderFeaturesInfo]]:
        return find_mm_placeholders(new_token_ids, mm_prompt_updates, None)

    def _apply_token_matches(
        self,
        prompt: List[int],
        mm_prompt_updates: MultiModalPromptUpdates,
    ):
        return apply_token_matches(prompt, mm_prompt_updates, None)

    def _apply_prompt_updates(
        self,
        token_ids: List[int],
        mm_prompt_updates: MultiModalPromptUpdates,
    ) -> Tuple[List[int], Mapping[str, List[PlaceholderFeaturesInfo]]]:
        new_token_ids, match_result = self._apply_token_matches(
            token_ids,
            mm_prompt_updates,
        )
        if not all(
            all(update_idx is not None for update_idx in update_idxs)
            for update_idxs in match_result.values()
        ):
            raise RuntimeError(
                "Failed to apply Confucius vLLM placeholder updates using "
                "token IDs only. The adapter requires token-list prompts when "
                "skip_tokenizer_init=True."
            )

        matched_updates: Dict[str, List[List[object]]] = defaultdict(list)
        for modality, update_idxs in match_result.items():
            for item_idx, update_idx in enumerate(update_idxs):
                matched_updates[modality].append(
                    [mm_prompt_updates[modality][item_idx][update_idx]]
                )

        return new_token_ids, self._find_mm_placeholders(
            new_token_ids,
            dict(matched_updates),
        )


@support_torch_compile
class ConfuciusGPT2Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.embed_dim = config.n_embd
        self.start_layer, self.end_layer, self.h = make_layers(
            config.n_layer,
            lambda prefix: GPT2Block(
                config,
                cache_config,
                quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.h",
        )
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(["hidden_states"], config.n_embd)
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: torch.Tensor,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = inputs_embeds
        for layer in self.h[self.start_layer : self.end_layer]:
            hidden_states = layer(hidden_states)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        return self.ln_f(hidden_states)


class _SemanticPositionEmbedding(nn.Module):
    def __init__(self, seq_len: int, model_dim: int, init_std: float = 0.02):
        super().__init__()
        self.embedding = nn.Embedding(seq_len, model_dim)
        self.embedding.weight.data.normal_(mean=0.0, std=init_std)


@MULTIMODAL_REGISTRY.register_processor(
    ConfuciusTTSMultiModalProcessor,
    info=ConfuciusTTSProcessingInfo,
    dummy_inputs=ConfuciusTTSDummyInputsBuilder,
)
class ConfuciusText2SemanticForCausalLM(nn.Module, SupportsPP, SupportsMultiModal):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        t2s_config = getattr(config, "text2semantic_config", {}) or {}
        text_embedding_dim = int(
            _config_get(
                t2s_config,
                "text_embedding_dim",
                getattr(config, "text_embedding_dim", 4096),
            )
        )
        speaker_embedding_dim = int(
            _config_get(
                t2s_config,
                "speaker_embedding_dim",
                getattr(config, "speaker_embedding_dim", 1024),
            )
        )

        self.config = config
        self.quant_config = quant_config
        self.supports_worker_prefix = bool(
            getattr(config, "confucius_supports_worker_prefix", False)
        )
        if self.supports_worker_prefix:
            self.text_projector = TextEmbeddingProjector(
                vocab_size=int(
                    getattr(
                        config,
                        "text_vocab_size",
                        _config_get(t2s_config, "vocab_size", 32000),
                    )
                ),
                embed_dim=text_embedding_dim,
                output_size=config.n_embd,
            )
            self.text_position_embedding = LearnedPositionalEmbedding(
                getattr(config, "max_text_seq_lens", 520),
                config.n_embd,
            )
            speaker_config = Qwen3TTSSpeakerEncoderConfig(
                mel_dim=speaker_embedding_dim,
                enc_dim=config.n_embd,
            )
            self.speaker_encoder = Qwen3TTSSpeakerEncoder(speaker_config)
        else:
            self.text_projector = None
            self.text_position_embedding = None
            self.speaker_encoder = None
        self.transformer = ConfuciusGPT2Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "transformer"),
        )
        self.semantic_embedding = VocabParallelEmbedding(
            config.vocab_size,
            config.n_embd,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "semantic_embedding"),
        )
        semantic_positions = getattr(
            config,
            "max_semantic_seq_lens",
            getattr(config, "semantic_max_positions", config.n_positions),
        )
        self.semantic_position_embedding = _SemanticPositionEmbedding(
            semantic_positions,
            config.n_embd,
            getattr(config, "initializer_range", 0.02),
        )
        self.final_norm = nn.LayerNorm(config.n_embd)
        self.semantic_head = ParallelLMHead(
            config.vocab_size,
            config.n_embd,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "semantic_head"),
            bias=True,
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.transformer.make_empty_intermediate_tensors
        )

    def get_language_model(self) -> nn.Module:
        return self.transformer

    def embed_multimodal(self, **kwargs: object) -> Optional[MultiModalEmbeddings]:
        audio_embeds = kwargs.get("audio_embeds")
        if audio_embeds is not None:
            processed_embeds = []
            for embed in audio_embeds:
                if embed.dim() == 3 and embed.shape[0] == 1:
                    processed_embeds.append(embed.squeeze(0))
                elif embed.dim() == 2:
                    processed_embeds.append(embed)
                else:
                    raise ValueError(
                        "Expected Confucius prefix embeddings to be 2D or 3D with "
                        f"a leading batch dimension of 1, got shape {tuple(embed.shape)}"
                    )
            return processed_embeds

        text_inputs = kwargs.get("text_inputs")
        condition_vectors = kwargs.get("condition_vector")
        if text_inputs is None and condition_vectors is None:
            return None
        if text_inputs is None or condition_vectors is None:
            raise ValueError(
                "Confucius worker-side prefix construction requires both "
                "text_inputs and condition_vector."
            )
        if not self.supports_worker_prefix:
            raise RuntimeError(
                "This converted Confucius vLLM model does not contain worker-side "
                "prefix weights. Re-run tools/convert_t2s_vllm.py."
            )

        processed_embeds = []
        for text_item, condition_item in zip(text_inputs, condition_vectors):
            if text_item.dim() == 1:
                text_item = text_item.unsqueeze(0)
            if condition_item.dim() == 2:
                condition_item = condition_item.unsqueeze(0)
            if text_item.dim() != 2 or text_item.shape[0] != 1:
                raise ValueError(
                    "Expected text_inputs to be 1D token IDs or shape (1, T), "
                    f"got {tuple(text_item.shape)}"
                )
            if condition_item.dim() != 3 or condition_item.shape[0] != 1:
                raise ValueError(
                    "Expected condition_vector to be shape (T, D) or (1, T, D), "
                    f"got {tuple(condition_item.shape)}"
                )
            embed_device = self.final_norm.weight.device
            embed_dtype = self.final_norm.weight.dtype
            text_item = text_item.to(device=embed_device, dtype=torch.long)
            condition_item = condition_item.to(
                device=embed_device,
                dtype=embed_dtype,
            )
            condition_emb = self.speaker_encoder(condition_item).unsqueeze(1)
            text_emb = self.text_projector(text_item)
            text_emb = self.text_position_embedding(text_emb)
            bos = torch.full(
                (1, 1),
                _placeholder_token_id_from_config(self.config),
                dtype=torch.long,
                device=text_item.device,
            )
            bos_emb = self.semantic_embedding(bos)
            processed_embeds.append(
                torch.cat([condition_emb, text_emb, bos_emb], dim=1).squeeze(0)
            )
        return processed_embeds

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
        *,
        is_multimodal: Optional[torch.Tensor] = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        inputs_embeds = self.semantic_embedding(input_ids)
        if multimodal_embeddings is not None and len(multimodal_embeddings) != 0:
            placeholder_mask = input_ids == _placeholder_token_id_from_config(
                self.config
            )
            if is_multimodal is None:
                is_multimodal = placeholder_mask
            else:
                is_multimodal = is_multimodal | placeholder_mask
            inputs_embeds = _merge_multimodal_embeddings(
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=multimodal_embeddings,
                is_multimodal=is_multimodal,
            )
        return inputs_embeds

    def _semantic_position_embeds(
        self,
        positions: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        valid = positions >= 0
        max_pos = self.semantic_position_embedding.embedding.num_embeddings - 1
        semantic_positions = positions.clamp(min=0, max=max_pos)
        pos_embeds = self.semantic_position_embedding.embedding(
            semantic_positions
        ).to(dtype=dtype)
        return pos_embeds * valid.unsqueeze(-1).to(dtype=dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)

        inputs_embeds += self._semantic_position_embeds(
            positions,
            inputs_embeds.dtype,
        )
        transformer_output = self.transformer(
            input_ids=None,
            position_ids=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        if isinstance(transformer_output, IntermediateTensors):
            return transformer_output
        return transformer_output

    def compute_logits(self, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        hidden_states = self.final_norm(hidden_states)
        return self.logits_processor(
            self.semantic_head,
            hidden_states,
            self.semantic_head.bias,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if ".attn.bias" in name or ".attn.masked_bias" in name:
                continue
            if name not in params_dict:
                continue
            if is_pp_missing_parameter(name, self):
                continue

            param = params_dict[name]
            for conv1d_weight_name in ("c_attn", "c_proj", "c_fc"):
                if conv1d_weight_name in name and name.endswith(".weight"):
                    loaded_weight = loaded_weight.t()
                    break

            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
