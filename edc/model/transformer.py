from __future__ import annotations

from typing import TYPE_CHECKING

import math

from torch.optim import AdamW
from pytorch_lightning import LightningModule
from transformers import AutoConfig, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup

from .. import utils

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import nn
    from torch.optim import Optimizer
    from transformers import PreTrainedModel

__all__ = [
    "TransformerModel"
]

class TransformerModel(LightningModule):
    CKPT_IGNORED_HPARAMS = (
        "skip_init_model",
        "grad_checkpointing"
    )

    def __init__(self, transformer_name: str, n_extra_tokens: int = 0,
        optim_factory: Callable[..., Optimizer] = AdamW, learning_rate: float = 1e-4,
        weight_decay: float = 0.01, cycle_up_duration: float = 0.4,
        skip_init_model: bool = True, grad_checkpointing: bool = True):
        super().__init__()

        self.transformer_name = transformer_name
        self.optim_factory = optim_factory
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.cycle_up_duration = cycle_up_duration

        # Create empty Transformer model from existing specifications
        if skip_init_model:
            config = AutoConfig.from_pretrained(transformer_name)
            transformer = AutoModelForSeq2SeqLM.from_config(config)
        # Initialize Transformer model from pre-trained checkpoints
        else:
            transformer = AutoModelForSeq2SeqLM.from_pretrained(transformer_name)
        
        self.transformer = transformer
        
        # Expand token embeddings
        if n_extra_tokens>0:
            self._expand_token_embed(n_extra_tokens)
        # Enable gradient checkpointing
        if grad_checkpointing:
            transformer.gradient_checkpointing_enable()
    
    def _expand_token_embed(self, n_extra_tokens: int):
        transformer = self.transformer

        # Save embedding size before expansion
        old_embed_size = transformer.get_input_embeddings().num_embeddings
        # Resize token embeddings
        new_embed = transformer.resize_token_embeddings(old_embed_size+n_extra_tokens)
        # Initialize expanded token embeddings
        utils.init_embed_weight(
            target=new_embed.weight[old_embed_size:],
            source=new_embed.weight[:old_embed_size]
        )
    
    @property
    def token_embed(self) -> nn.Embedding:
        return self.transformer.get_input_embeddings()
    
    @property
    def encoder(self) -> PreTrainedModel:
        return self.transformer.get_encoder()
    
    @property
    def decoder(self) -> PreTrainedModel:
        return self.transformer.get_decoder()

    @property
    def lm_head(self) -> nn.Module:
        return self.transformer.lm_head

    def configure_optimizers(self):
        trainer = self.trainer

        # Compute number of training steps
        trainer.reset_train_dataloader()
        epoch_steps = math.ceil(trainer.num_training_batches/trainer.accumulate_grad_batches)
        train_steps = epoch_steps*trainer.max_epochs

        warm_up_steps = int(train_steps*self.cycle_up_duration)

        # Create optimizer (learning rate controlled by scheduler)
        optimizer = self.optim_factory(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, foreach=True
        )
        # Create learning rate scheduler
        lr_sched = get_linear_schedule_with_warmup(optimizer, warm_up_steps, train_steps)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_sched,
                "interval": "step"
            }
        }
