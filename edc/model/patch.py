from __future__ import annotations

from typing import TYPE_CHECKING

from contextvars import ContextVar

import torch
from transformers.models.bart import modeling_bart as bart

if TYPE_CHECKING:
    from typing import Optional

__all__ = [
    "BART_SKIP_POS_EMBED"
]

# [ Bart ]
BART_SKIP_POS_EMBED: ContextVar[bool] = ContextVar("bart_skip_pos_embed", default=False)

def bart_encoder_decoder_get_pos_embed(self) -> bart.BartLearnedPositionalEmbedding:
    return self.embed_positions

def bart_pos_embed_forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0) -> torch.Tensor:
    skip_pos_embed = BART_SKIP_POS_EMBED.get()
    # Make dummy, all-zero positional embeddings if positional embeddings are skipped
    if skip_pos_embed:
        return torch.zeros((), device=input_ids.device)
    
    # Compute range of position IDs
    batch_size, seq_len = input_ids.shape[:2]
    pos_id_start = self.offset+past_key_values_length
    pos_id_end = pos_id_start+seq_len
    # Get position embeddings
    return self.weight[pos_id_start:pos_id_end].expand(batch_size, -1, -1)

def bart_expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    if mask.ndim==2:
        # Target and source lengths are equal
        tgt_len = mask.shape[-1] if tgt_len is None else tgt_len
        # Expand mask to 3D
        mask = mask[:, None, :].expand(-1, tgt_len, -1)

    # Add extra dimension for heads
    mask = mask[:, None, ...].to(dtype)
    # Fill mask with either zero or negative infinity
    return (1-mask)*torch.finfo(dtype).min

# Patch Bart implementations
bart.BartLearnedPositionalEmbedding.forward = bart_pos_embed_forward
bart.BartEncoder.get_position_embeddings = bart_encoder_decoder_get_pos_embed
bart.BartDecoder.get_position_embeddings = bart_encoder_decoder_get_pos_embed
bart._expand_mask = bart_expand_mask
