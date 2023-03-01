from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from functools import cached_property, partial

import torch
from torch.nn import functional as f
from transformers import AutoTokenizer

from .. import utils
from ..data import get_edc_special_tokens, pred_states_tree, pred_states_one_pass
from .transformer import TransformerModel
from .patch import BART_SKIP_POS_EMBED
from .decode import greedy_decode, beam_search_decode

if TYPE_CHECKING:
    from typing import Any, Optional

    from ..data import EncoderData, DecoderData, EDCSample, TargetDataPath, PredResult
    from ..types import Tokenizer
    from .decode import DecodeStepResult

    # Past keys and values
    PastKV = tuple[tuple[torch.Tensor, ...], ...]

__all__ = [
    "EDCModel"
]

class EDCGenState(NamedTuple):
    ctx_indices: torch.Tensor
    decoder_self_attn_mask: torch.Tensor
    decoder_past_kvs: PastKV

    @classmethod
    def new(cls, ctx_index: int, device: torch.device) -> EDCGenState:
        return cls(
            ctx_indices=ctx_index.to(device),
            decoder_self_attn_mask=torch.empty(0, dtype=torch.int64, device=device),
            decoder_past_kvs=()
        )

class EDCModel(TransformerModel):
    def __init__(self, max_rounds: int, **kwargs: Any):
        # Number of special tokens
        n_special_tokens = len(get_edc_special_tokens(max_rounds))

        super().__init__(n_extra_tokens=n_special_tokens, **kwargs)

        self.max_rounds = max_rounds

        # Save hyper-parameters of model
        self.save_hyperparameters(ignore=self.CKPT_IGNORED_HPARAMS)
    
    def _decoder_embed(self, token_ids: torch.Tensor, pos_ids: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        # Get pad token ID (and fall back to 0 if it does not exist)
        pad_token_id = getattr(self.transformer.config, "pad_token_id", 0)
        # Fill padding locations
        token_ids = token_ids.where(token_ids!=-1, token_ids.new_tensor(pad_token_id))

        # Token embeddings
        embeds = self.token_embed(token_ids)
        # Position embeddings (Only applies to Bart)
        if pos_ids is not None:
            pos_embed = self.decoder.get_position_embeddings()
            pos_embeds = pos_embed.weight[pos_ids+pos_embed.offset]
            embeds = embeds+pos_embeds
        
        return embeds
    
    def _tv_step(self, sample: EDCSample, metric_prefix: str = "") -> torch.Tensor:
        encoder_data, decoder_data = sample
        # Compute loss and metrics
        loss, metrics = self(encoder_data, decoder_data)

        # Log loss and metrics
        self.log(metric_prefix+"loss", loss, batch_size=1, sync_dist=True)
        for name, value in metrics.items():
            self.log(metric_prefix+name, value, batch_size=1, sync_dist=True)

        return loss
    
    def _decoder_forward_incr(self, token_ids: torch.Tensor, states: list[EDCGenState],
        encoder_feats: torch.Tensor, decoder_cross_attn_mask: torch.Tensor,
        decoder_self_attn_mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, list[EDCGenState]]:
        batch_size, inputs_size = token_ids.shape

        # Make incremental attention mask
        if decoder_self_attn_mask is None:
            decoder_self_attn_mask = torch.ones_like(token_ids)
        
        # Coalesce generation states
        ctx_indices, batch_attn_mask, batch_past_kvs = coalesce_edc_gen_states(states)
        # Compute position IDs
        pos_id_starts = batch_attn_mask.sum(1, keepdim=True)
        pos_ids = pos_id_starts+torch.arange(inputs_size, device=token_ids.device)
        # Update attention mask
        batch_attn_mask = torch.cat((batch_attn_mask, decoder_self_attn_mask), 1)

        # Incrementally forward through decoder
        with utils.set_ctx(BART_SKIP_POS_EMBED, True):
            decoder_feats, next_past_kvs, *_ = self.decoder(
                inputs_embeds=self._decoder_embed(token_ids, pos_ids),
                attention_mask=batch_attn_mask,
                encoder_hidden_states=encoder_feats.expand(batch_size, -1, -1),
                encoder_attention_mask=decoder_cross_attn_mask[ctx_indices],
                past_key_values=batch_past_kvs or None,
                use_cache=True,
                return_dict=False
            )
        # Split and get next generation states
        next_states = split_edc_gen_state(ctx_indices, batch_attn_mask, next_past_kvs)
        
        return decoder_feats, next_states
    
    def _update_gen_state(self, token_ids: list[torch.Tensor], states: list[EDCGenState],
        encoder_feats: torch.Tensor, decoder_cross_attn_mask: torch.Tensor) -> list[EDCGenState]:
        device = self.device

        # Coalesce token IDs
        batch_token_ids = utils.pad_stack_tensors(token_ids, pad_value=-1).to(device)
        # Make decoder self attention mask
        decoder_self_attn_mask = utils.pad_stack_tensors([
            torch.ones_like(ids) for ids in token_ids
        ]).to(device)

        # Incrementally forward through decoder
        _, gen_states = self._decoder_forward_incr(
            batch_token_ids, states, encoder_feats, decoder_cross_attn_mask, decoder_self_attn_mask
        )
        return gen_states
    
    def _gen_step(self, encoder_feats: torch.Tensor, decoder_cross_attn_mask: torch.Tensor,
        token_ids: torch.Tensor, states: list[EDCGenState], _: int) -> DecodeStepResult[EDCGenState]:
        # Incrementally forward through decoder
        decoder_feats, next_states = self._decoder_forward_incr(
            token_ids.unsqueeze(-1), states, encoder_feats, decoder_cross_attn_mask
        )
        # Compute and normalize log probabilities
        logits = self.lm_head(decoder_feats.squeeze(1)).log_softmax(-1)

        return logits, next_states

    @cached_property
    def _tokenizer(self) -> Tokenizer:
        # Load pre-trained tokenizer
        tokenizer: Tokenizer = AutoTokenizer.from_pretrained(
            self.transformer_name, add_prefix_space=True
        )
        # Add special tokens
        tokenizer.add_tokens(get_edc_special_tokens(self.max_rounds), special_tokens=True)
        
        return tokenizer

    def forward(self, encoder_data: EncoderData, decoder_data: DecoderData):
        # Forward through encoder
        encoder_feats, *_ = self.encoder(
            input_ids=encoder_data.ctx_token_ids.unsqueeze(0),
            attention_mask=encoder_data.encoder_attn_mask.unsqueeze(0),
            return_dict=False
        )
        
        # Forward through decoder
        ctx_indices = decoder_data.ctx_indices
        batch_size = len(ctx_indices)
        decoder_cross_attn_mask = encoder_data.decoder_cross_attn_mask[ctx_indices]

        decoder_feats, *_ = self.decoder(
            inputs_embeds=self._decoder_embed(decoder_data.target_token_ids),
            encoder_hidden_states=encoder_feats.expand(batch_size, -1, -1),
            encoder_attention_mask=decoder_cross_attn_mask,
            use_cache=False,
            return_dict=False
        )

        # Target token mask
        target_mask = decoder_data.target_output_masks[:, 1:]
        # Logits and target token IDs
        target_logits = self.lm_head(decoder_feats[:, :-1][target_mask])
        target_ids = decoder_data.target_token_ids[:, 1:][target_mask]

        # Cross entropy losses
        l = f.cross_entropy(target_logits, target_ids)
        # Accuracy
        acc = (target_logits.argmax(-1)==target_ids).float().mean()
        
        return l, {"accuracy": acc}

    def training_step(self, sample: EDCSample, _: int) -> torch.Tensor:
        return self._tv_step(sample)

    def validation_step(self, sample: EDCSample, _: int) -> torch.Tensor:
        return self._tv_step(sample, metric_prefix="val_")

    def predict_step(self, sample: tuple[int, EDCSample], _: int):
        sample_idx, (encoder_data, _) = sample

        data_module = self.trainer.datamodule
        device = self.device
        tokenizer = self._tokenizer

        # Forward through encoder
        encoder_feats, *_ = self.encoder(
            input_ids=encoder_data.ctx_token_ids.unsqueeze(0),
            attention_mask=encoder_data.encoder_attn_mask.unsqueeze(0),
            return_dict=False
        )

        # State predictor function
        pred_states = pred_states_one_pass if data_module.one_pass else pred_states_tree
        # Number of dialog rounds
        n_rounds = 1 if data_module.standalone_ctx else len(encoder_data.decoder_cross_attn_mask)//2

        # Prediction data generator
        pred_gen = pred_states(tokenizer, n_rounds, data_module.standalone_ctx, 60)
        # Generated token IDs
        gen_token_ids = None

        while True:
            # Get prediction data from generator
            try:
                pred_data = pred_gen.send(gen_token_ids)
            except StopIteration as e:
                pred_states = e.value
                break

            # Make generation states
            gen_states = [
                EDCGenState.new(ctx_index, device) \
                for ctx_index in pred_data.ctx_indices
            ]
            # Update generation states with inputs
            gen_states = self._update_gen_state(
                token_ids=pred_data.token_ids,
                states=gen_states,
                encoder_feats=encoder_feats,
                decoder_cross_attn_mask=encoder_data.decoder_cross_attn_mask
            )

            # Generate data
            gen_outputs = greedy_decode(
                partial(self._gen_step, encoder_feats, encoder_data.decoder_cross_attn_mask),
                start_items=pred_data.start_ids.to(device),
                start_states=gen_states,
                max_len=96,
                end_items=[tokenizer.eos_token_id]
            )
            # Collect generated data
            gen_token_ids = [outputs[0].seq for outputs in gen_outputs]
        
        return sample_idx, pred_states

def coalesce_edc_gen_states(states: list[EDCGenState]) -> EDCGenState:
    sample_kv = states[0][2]
    n_layers = len(sample_kv)
    layer_kv_size = len(sample_kv[0]) if sample_kv else 0

    ctx_indices: list[torch.Tensor] = []
    attn_masks: list[torch.Tensor] = []
    past_kvs: list[list[list[torch.Tensor]]] = [
        [[] for _ in range(layer_kv_size)] for _ in range(n_layers)
    ]

    for ctx_index, attn_mask, past_kv in states:
        ctx_indices.append(ctx_index)
        attn_masks.append(attn_mask)

        if past_kvs:
            for i, layer_kv in enumerate(past_kv):
                for j, kv in enumerate(layer_kv):
                    past_kvs[i][j].append(kv)
    
    batch_ctx_indices = torch.stack(ctx_indices)
    batch_attn_mask = torch.stack(attn_masks)
    batch_past_kvs = tuple(
        tuple(torch.stack(kvs) for kvs in layer_kvs) \
        for layer_kvs in past_kvs
    )
    
    return EDCGenState(batch_ctx_indices, batch_attn_mask, batch_past_kvs)

def split_edc_gen_state(ctx_indices: torch.Tensor, batch_attn_mask: torch.Tensor,
    batch_past_kvs: PastKV) -> list[EDCGenState]:
    # Past keys and values for each sample
    past_kv_iter = (
        tuple(
            tuple(kvs[i] for kvs in layer_kvs) \
            for layer_kvs in batch_past_kvs
        ) for i in range(len(batch_past_kvs[0][0]))
    )

    return [
        EDCGenState(ctx_index, attn_mask, past_kv) \
        for ctx_index, attn_mask, past_kv \
        in zip(ctx_indices, batch_attn_mask, past_kv_iter)
    ]
