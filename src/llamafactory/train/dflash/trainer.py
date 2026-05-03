from typing import TYPE_CHECKING, Optional

import torch
from transformers import Trainer
from typing_extensions import override

from ...extras import logging
from ...model.dflash import DFlashDraftModel, HFDFlashTargetModel, OnlineDFlashModel, TargetEmbeddingsAndHead, build_target_layer_ids


if TYPE_CHECKING:
    import torch.nn as nn
    from transformers import ProcessorMixin, Seq2SeqTrainingArguments

    from ...hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


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
                loss_mask = (labels != -100).float()
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

        if self.state.is_world_process_zero and hasattr(self, "_logged_acc_once"):
            pass
        elif self.state.is_world_process_zero:
            self._logged_acc_once = True

        if self.control.should_log and self.state.is_world_process_zero:
            logs = {"dflash_accuracy": accuracy.item()}
            self.log(logs)

        return loss

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        import os
        from safetensors.torch import save_file

        os.makedirs(output_dir, exist_ok=True)

        # Only save on the main process to avoid NPU storage pointer issues in multi-card setup
        if self.args.should_save and self.state.is_world_process_zero:
            draft_model = self.dflash_model.draft_model
            draft_model.config.save_pretrained(output_dir)

            # Move all tensors to CPU before saving to avoid invalid NPU storage pointer
            state_dict = {k: v.detach().cpu().contiguous() for k, v in draft_model.state_dict().items()}
            save_file(state_dict, os.path.join(output_dir, "model.safetensors"))

            logger.info_rank0(f"DFlash draft model saved to {output_dir}")
