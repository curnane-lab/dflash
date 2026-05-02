from typing import TYPE_CHECKING, Optional

import torch
from transformers import DataCollatorForSeq2Seq

from ...data import get_dataset, get_template_and_fix_tokenizer
from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...model import load_tokenizer
from ...model.dflash import (
    DFlashDraftModel,
    HFDFlashTargetModel,
    OnlineDFlashModel,
    TargetEmbeddingsAndHead,
    build_target_layer_ids,
)
from .trainer import DFlashTrainer


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = logging.get_logger(__name__)


def run_dflash(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="dflash", **tokenizer_module)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    logger.info_rank0("Loading DFlash target model...")
    target_model = HFDFlashTargetModel.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        torch_dtype=dtype,
        device=device,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )

    logger.info_rank0("Loading target config...")
    from transformers import AutoConfig

    target_config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        cache_dir=model_args.cache_dir,
    )

    text_config = getattr(target_config, "text_config", None)
    source_config = text_config if text_config is not None else target_config

    logger.info_rank0("Extracting target model embeddings and lm_head...")
    target_components = TargetEmbeddingsAndHead(source_config)
    # Copy weights from the already-loaded HF target model instead of re-parsing checkpoint files.
    # This bypasses issues where .index.json omits embed_tokens keys (e.g., Qwen3.5 VLM checkpoints).
    # Supports: model.embed_tokens, model.language_model.embed_tokens, model.text_model.embed_tokens
    hf_model = target_model.model
    embed_tokens_src = None
    # 1. Standard Qwen/Qwen2/Qwen3: model.model.embed_tokens
    if hasattr(hf_model, "model") and hasattr(hf_model.model, "embed_tokens"):
        embed_tokens_src = hf_model.model.embed_tokens
        logger.info_rank0("Found embed_tokens at hf_model.model.embed_tokens")
    # 2. Qwen3.5 VLM: model.language_model.embed_tokens
    elif hasattr(hf_model, "model") and hasattr(hf_model.model, "language_model"):
        lm = hf_model.model.language_model
        if hasattr(lm, "embed_tokens"):
            embed_tokens_src = lm.embed_tokens
            logger.info_rank0("Found embed_tokens at hf_model.model.language_model.embed_tokens (Qwen3.5 VLM)")
        elif hasattr(lm, "model") and hasattr(lm.model, "embed_tokens"):
            embed_tokens_src = lm.model.embed_tokens
            logger.info_rank0("Found embed_tokens at hf_model.model.language_model.model.embed_tokens")
    # 3. Qwen3.5 multimodal: model.text_model.embed_tokens
    elif hasattr(hf_model, "model") and hasattr(hf_model.model, "text_model"):
        if hasattr(hf_model.model.text_model, "embed_tokens"):
            embed_tokens_src = hf_model.model.text_model.embed_tokens
            logger.info_rank0("Found embed_tokens at hf_model.model.text_model.embed_tokens")
    if embed_tokens_src is not None:
        target_components.embed_tokens.weight.data.copy_(embed_tokens_src.weight.data)
    else:
        raise ValueError(f"Cannot find embed_tokens in target model. Type: {type(hf_model).__name__}")

    lm_head_src = None
    # 1. Standard: model.lm_head
    if hasattr(hf_model, "lm_head"):
        lm_head_src = hf_model.lm_head
        logger.info_rank0("Found lm_head at hf_model.lm_head")
    # 2. Qwen3.5 VLM: model.language_model.lm_head
    elif hasattr(hf_model, "model") and hasattr(hf_model.model, "language_model"):
        lm = hf_model.model.language_model
        if hasattr(lm, "lm_head"):
            lm_head_src = lm.lm_head
            logger.info_rank0("Found lm_head at hf_model.model.language_model.lm_head (Qwen3.5 VLM)")
    if lm_head_src is not None:
        target_components.lm_head.weight.data.copy_(lm_head_src.weight.data)
    else:
        raise ValueError(f"Cannot find lm_head in target model. Type: {type(hf_model).__name__}")

    target_components = target_components.to(device=device, dtype=dtype)
    target_components.eval()
    target_components.requires_grad_(False)

    logger.info_rank0("Creating DFlash draft model...")

    num_target_layers = getattr(target_config, "num_hidden_layers", 32)
    num_draft_layers = finetuning_args.dflash_num_draft_layers

    if finetuning_args.dflash_target_layer_ids is not None:
        target_layer_ids = finetuning_args.dflash_target_layer_ids
    else:
        target_layer_ids = build_target_layer_ids(num_target_layers, num_draft_layers)

    logger.info_rank0(f"Target layer IDs for hidden state extraction: {target_layer_ids}")

    draft_config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        cache_dir=model_args.cache_dir,
    )

    text_config = getattr(draft_config, "text_config", None)
    if text_config is not None:
        source_config = text_config
    else:
        source_config = draft_config

    # Use source_config (text_config) as the draft config, because Qwen3_5Config
    # (multimodal wrapper) does not have hidden_size / num_attention_heads etc.
    # All DFlash-specific attributes are set on source_config instead.
    source_config.num_hidden_layers = num_draft_layers
    source_config.block_size = finetuning_args.dflash_block_size
    source_config.num_target_layers = num_target_layers
    source_config.dflash_config = {
        "mask_token_id": finetuning_args.dflash_mask_token_id,
        "target_layer_ids": target_layer_ids,
    }
    source_config.layer_types = ["full_attention"] * num_draft_layers

    if hasattr(source_config, "rope_parameters") and source_config.rope_parameters is not None:
        source_config.rope_parameters = source_config.rope_parameters
    elif hasattr(source_config, "partial_rotary_factor"):
        source_config.partial_rotary_factor = source_config.partial_rotary_factor
    else:
        source_config.partial_rotary_factor = getattr(source_config, "partial_rotary_factor", 1.0)

    if hasattr(source_config, "rope_theta"):
        source_config.rope_theta = source_config.rope_theta

    if finetuning_args.dflash_pretrained_model_path is not None:
        logger.info_rank0(f"Loading pretrained DFlash draft model from {finetuning_args.dflash_pretrained_model_path}")
        draft_model = DFlashDraftModel.from_pretrained(
            finetuning_args.dflash_pretrained_model_path,
            config=source_config,
            trust_remote_code=model_args.trust_remote_code,
        )
        draft_model = draft_model.to(device=device, dtype=dtype)
    else:
        draft_model = DFlashDraftModel(source_config)
        draft_model = draft_model.to(device=device, dtype=dtype)
    logger.info_rank0(f"DFlash draft model ready with {num_draft_layers} layers, block_size={finetuning_args.dflash_block_size}")

    target_model.set_capture_layers(target_layer_ids)

    mask_token_id = finetuning_args.dflash_mask_token_id
    if mask_token_id is None:
        mask_token = getattr(tokenizer, "mask_token", None)
        if mask_token is not None:
            mask_token_id = tokenizer.convert_tokens_to_ids(mask_token)
        else:
            mask_token_id = tokenizer.vocab_size - 1
            logger.warning_rank0(f"No MASK token found in tokenizer, using token_id={mask_token_id}")

    dflash_model = OnlineDFlashModel(
        draft_model=draft_model,
        target_lm_head=target_components.lm_head,
        target_embed_tokens=target_components.embed_tokens,
        mask_token_id=mask_token_id,
        block_size=finetuning_args.dflash_block_size,
        attention_backend=finetuning_args.dflash_attention_backend,
        num_anchors=finetuning_args.dflash_num_anchors,
        loss_decay_gamma=finetuning_args.dflash_loss_decay_gamma,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=None,
        padding=True,
        pad_to_multiple_of=8 if training_args.do_train else None,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        return_tensors="pt",
    )

    trainer = DFlashTrainer(
        dflash_model=dflash_model,
        target_model=target_model,
        finetuning_args=finetuning_args,
        model_args=model_args,
        args=training_args,
        data_collator=data_collator,
        callbacks=callbacks,
        **dataset_module,
        **tokenizer_module,
    )

    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
