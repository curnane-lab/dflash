import gc
import glob
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM

from .modeling import extract_context_feature


@dataclass
class DFlashTargetOutput:
    hidden_states: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    loss_mask: torch.Tensor


class DFlashTargetModel(ABC):
    def __init__(self):
        self.capture_layer_ids = None

    @classmethod
    @abstractmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs) -> "DFlashTargetModel":
        ...

    @abstractmethod
    def generate_dflash_data(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, loss_mask: torch.Tensor
    ) -> DFlashTargetOutput:
        ...

    def set_capture_layers(self, layer_ids: List[int]) -> None:
        self.capture_layer_ids = layer_ids


class HFDFlashTargetModel(DFlashTargetModel):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = True,
        **kwargs,
    ) -> "HFDFlashTargetModel":
        target_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            output_hidden_states=True,
            trust_remote_code=trust_remote_code,
            **kwargs,
        ).eval()
        if device:
            target_model = target_model.to(device)
        return cls(target_model)

    @torch.no_grad()
    def generate_dflash_data(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, loss_mask: torch.Tensor
    ) -> DFlashTargetOutput:
        # Ensure inputs are on the same device as the model (handles accelerate/FSDP multi-device)
        model_device = next(self.model.parameters()).device
        if input_ids.device != model_device:
            input_ids = input_ids.to(model_device)
        if attention_mask.device != model_device:
            attention_mask = attention_mask.to(model_device)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        offset = 1
        selected = []
        if self.capture_layer_ids is not None:
            for idx in self.capture_layer_ids:
                selected.append(outputs.hidden_states[idx + offset])
            hidden_states = torch.cat(selected, dim=-1)
        else:
            hidden_states = outputs.hidden_states[-1]
        return DFlashTargetOutput(
            hidden_states=hidden_states,
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
        )


class TargetEmbeddingsAndHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        embed_key: Optional[str] = None,
        lm_head_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = False,
    ) -> "TargetEmbeddingsAndHead":
        config = AutoConfig.from_pretrained(model_path, cache_dir=cache_dir, trust_remote_code=trust_remote_code)
        text_config = getattr(config, "text_config", None)
        if text_config is not None:
            config = text_config
        instance = cls(config)
        if embed_key is None:
            embed_key = "model.embed_tokens.weight"
        if lm_head_key is None:
            lm_head_key = "lm_head.weight"
        local_model_path = model_path
        if not os.path.exists(local_model_path):
            try:
                local_model_path = snapshot_download(
                    repo_id=model_path, cache_dir=cache_dir, allow_patterns=["*.json", "*.safetensors", "*.bin", "*.model"]
                )
            except Exception:
                pass
        tie_weights = getattr(config, "tie_word_embeddings", False)
        instance._load_weights(local_model_path, embed_key, lm_head_key, tie_weights)
        instance.to(device=device, dtype=dtype)
        instance.eval()
        instance.requires_grad_(False)
        return instance

    def _load_weights(self, model_path: str, embed_key: str, lm_head_key: str, tie_weights: bool):
        index_files = glob.glob(os.path.join(model_path, "*.index.json"))
        weight_map = {}
        files_to_load = {}

        # Candidate keys for embedding and lm_head, ordered by preference
        # Qwen3.5 VLM uses model.language_model.embed_tokens / model.language_model.lm_head
        embed_candidates = [
            embed_key,
            "model.language_model.embed_tokens.weight",
            "model.embed_tokens.weight",
            "embed_tokens.weight",
            "model.tok_embeddings.weight",
            "transformer.wte.weight",
        ]
        head_candidates = [
            lm_head_key,
            "model.language_model.lm_head.weight",
            "lm_head.weight",
            "model.lm_head.weight",
            "output.weight",
        ]

        def _find_key(candidates, weight_map_or_keys):
            for cand in candidates:
                if cand in weight_map_or_keys:
                    return cand
            return None

        if index_files:
            with open(index_files[0], "r") as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})

            actual_embed_key = _find_key(embed_candidates, weight_map)
            if actual_embed_key is None:
                raise ValueError(f"Embedding key not found in weight map. Tried: {embed_candidates}")
            files_to_load[actual_embed_key] = weight_map[actual_embed_key]

            if not tie_weights:
                actual_head_key = _find_key(head_candidates, weight_map)
                if actual_head_key is not None:
                    files_to_load[actual_head_key] = weight_map[actual_head_key]
        else:
            safetensors = glob.glob(os.path.join(model_path, "*.safetensors"))
            bins = glob.glob(os.path.join(model_path, "*.bin"))
            target_file = safetensors[0] if safetensors else (bins[0] if bins else None)
            if not target_file:
                raise FileNotFoundError("No checkpoint found.")
            files_to_load[embed_key] = os.path.basename(target_file)
            if not tie_weights:
                files_to_load[lm_head_key] = os.path.basename(target_file)

        loaded_keys = set()
        file_to_keys_map = {}
        for key, filename in files_to_load.items():
            full_path = os.path.join(model_path, filename)
            if full_path not in file_to_keys_map:
                file_to_keys_map[full_path] = []
            file_to_keys_map[full_path].append(key)

        # Pass the actual resolved keys to _load_file_content so it knows what to look for
        actual_embed = _find_key(embed_candidates, weight_map) if weight_map else embed_key
        actual_head = _find_key(head_candidates, weight_map) if weight_map else lm_head_key
        for file_path, keys in file_to_keys_map.items():
            self._load_file_content(file_path, keys, actual_embed or embed_key, actual_head or lm_head_key)
            loaded_keys.update(keys)

        if tie_weights:
            self.lm_head.weight = self.embed_tokens.weight

    def _load_file_content(self, file_path: str, keys_to_extract: list, target_embed_key: str, target_head_key: str):
        state_dict_part = {}
        if file_path.endswith(".safetensors"):
            with safe_open(file_path, framework="pt") as f:
                for k in keys_to_extract:
                    if k in f.keys():
                        state_dict_part[k] = f.get_tensor(k)
        else:
            full_state = torch.load(file_path, map_location="cpu")
            for k in keys_to_extract:
                if k in full_state:
                    state_dict_part[k] = full_state[k]
            del full_state
            gc.collect()

        for k, tensor in state_dict_part.items():
            if k == target_embed_key:
                self.embed_tokens.weight.data.copy_(tensor)
            elif k == target_head_key:
                if tensor.shape == self.lm_head.weight.data.shape:
                    self.lm_head.weight.data.copy_(tensor)
