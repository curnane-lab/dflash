from typing import TYPE_CHECKING, Optional

import torch
from transformers import DataCollatorForSeq2Seq, Trainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...model.dflash import DFlashDraftModel, HFDFlashTargetModel, OnlineDFlashModel, TargetEmbeddingsAndHead, build_target_layer_ids


if TYPE_CHECKING:
    import torch.nn as nn
    from transformers import ProcessorMixin, Seq2SeqTrainingArguments

    from ...hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


class DFlashDataCollator(DataCollatorForSeq2Seq):

    def __call__(self, features, return_tensors=None):
        loss_masks = None
        if features and "loss_mask" in features[0]:
            loss_masks = [f.pop("loss_mask") for f in features]

        batch = super().__call__(features, return_tensors=return_tensors)

        if loss_masks is not None:
            padded_loss_masks = []
            max_len = batch["input_ids"].shape[-1]
            for lm in loss_masks:
                if isinstance(lm, list):
                    lm = torch.tensor(lm, dtype=torch.float32)
                pad_len = max_len - lm.shape[0]
                if pad_len > 0:
                    lm = torch.cat([lm, torch.zeros(pad_len, dtype=torch.float32)])
                else:
                    lm = lm[:max_len]
                padded_loss_masks.append(lm)
            batch["loss_mask"] = torch.stack(padded_loss_masks)

        return batch


class DFlashTrainer(Trainer):

    def __init__(
        self,
        dflash_model: OnlineDFlashModel,
        target_model: HFDFlashTargetModel,
        finetuning_args: "FinetuningArguments",
        model_args: Optional["ModelArguments"] = None,
        processor: Optional["ProcessorMixin"] = None,
        **kwargs,
    ) -> None:
        kwargs["processing_class"] = kwargs.pop("tokenizer", None)
        kwargs["model"] = dflash_model
        super().__init__(**kwargs)
        self.dflash_model = dflash_model
        self.target_model = target_model
        self.finetuning_args = finetuning_args
        self.model_args = model_args

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        loss_mask = inputs.get("loss_mask", None)

        if loss_mask is None:
            labels = inputs.get("labels", None)
            if labels is not None:
                loss_mask = (labels != IGNORE_INDEX).float()
            else:
                loss_mask = attention_mask.float() if attention_mask is not None else torch.ones_like(input_ids, dtype=torch.float32)

        with torch.no_grad():
            target_output = self.target_model.generate_dflash_data(
                input_ids=input_ids,
                attention_mask=attention_mask if attention_mask is not None else torch.ones_like(input_ids),
                loss_mask=loss_mask,
            )

        loss, accuracy = model(
            input_ids=target_output.input_ids,
            hidden_states=target_output.hidden_states,
            loss_mask=target_output.loss_mask,
        )

        if self.control.should_log and self.state.is_world_process_zero:
            logs = {"dflash_accuracy": accuracy.item()}
            self.log(logs)

        return loss

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        import os

        os.makedirs(output_dir, exist_ok=True)

        # Only save on the main process to avoid NPU storage pointer issues in multi-card setup
        if self.args.should_save and self.state.is_world_process_zero:
            draft_model = self.dflash_model.draft_model
            draft_model.config.save_pretrained(output_dir)

            # Use torch.save (.bin format) instead of safetensors to bypass the
            # _find_shared_tensors storage_ptr issue on NPU. Clone ensures each
            # tensor has its own independent CPU storage.
            state_dict = {k: v.detach().cpu().clone() for k, v in draft_model.state_dict().items()}
            torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))

            logger.info_rank0(f"DFlash draft model saved to {output_dir}")
