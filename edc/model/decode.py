from __future__ import annotations

from typing import TYPE_CHECKING, Generic, Protocol, TypeVar
from dataclasses import dataclass

import numpy as np
import torch

from .. import utils

# Decoder state type
S = TypeVar("S")

__all__ = [
    "Decoder",
    "DecodeOutput",
    "DecodeStrategy",
    "SearchScorer",
    "StopCond",
    "BeamSearchStrategy",
    "DefaultScorer",
    "DefaultStopCond",
    "SearchOutput",
    "greedy_search_strategy",
    "search_decode",
    "beam_search",
    "greedy_search"
]

if TYPE_CHECKING:
    from typing import TypeVar, Optional

    # Decode step result: (scores, states)
    DecodeStepResult = tuple[torch.Tensor, list[S]]
    # Filter result: (items, scores)
    FilterResult = tuple[torch.Tensor, torch.Tensor]
    # Scorer result: (beam_indices, items, scores)
    ScorerResult = tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    __all__ += [
        "DecodeStepResult",
        "FilterResult",
        "ScorerResult"
    ]

# Dummy score and item for invalid beams
_DUMMY_ITEM = -1

class Decoder(Protocol[S]):
    def __call__(self, inputs: torch.Tensor, states: list[S], step: int) -> DecodeStepResult[S]:
        raise NotImplementedError

class Filter(Protocol):
    def __call__(self, step_scores: torch.Tensor) -> FilterResult:
        raise NotImplementedError

class Scorer(Protocol):
    def __call__(self, step_scores: torch.Tensor, beams_mask: torch.Tensor, step: int) -> ScorerResult:
        raise NotImplementedError

class StopCond(Protocol):
    def __call__(self, items: torch.Tensor, step: int) -> torch.Tensor:
        raise NotImplementedError

class GreedyScorer(Scorer):
    def __init__(self):
        self._sum_scores: Optional[torch.Tensor] = None
    
    def __call__(self, step_scores: torch.Tensor, beams_mask: torch.Tensor, step: int) -> ScorerResult:
        # Compute best items and step scores
        best_scores, best_items = step_scores.max(-1, keepdim=True)

        sum_scores = self._sum_scores
        # Initialize sum of step scores
        if sum_scores is None:
            sum_scores = self._sum_scores = torch.zeros_like(best_scores)
        # Update sum of step scores
        dummy_score = torch.finfo(best_scores.dtype).min
        sum_scores += utils.decompress(best_scores, beams_mask.squeeze(-1), dummy_score)
        
        # Synthesize beam indices
        beam_indices = torch.zeros_like(sum_scores, dtype=torch.int64)
        # Decompress best items
        best_items = utils.decompress(best_items, beams_mask.squeeze(-1), _DUMMY_ITEM)

        return beam_indices, best_items, sum_scores/(step+1)

class TopKFilter(Filter):
    def __init__(self, k: int):
        self.k = k
    
    def __call__(self, step_scores: torch.Tensor) -> FilterResult:
        return utils.topk(step_scores, self.k, sorted=False)

class TopPFilter(Filter):
    def __init__(self, p: float):
        self.p = p
    
    def __call__(self, step_scores: torch.Tensor) -> FilterResult:
        # Sort scores of current step
        sorted_step_scores, sorted_items = step_scores.sort(descending=True)
        # Compute cumulative probabilities and step mask
        cum_probs = sorted_step_scores.softmax(-1).cumsum(-1)
        step_mask = cum_probs<self.p
        step_mask[:, 0] = True

        # Compute number of top items
        n_top_items = step_mask.sum(-1).max().item()
        # Gather top items with mask
        top_step_items = torch.where(step_mask, sorted_items, _DUMMY_ITEM)
        top_step_items = top_step_items[:, :n_top_items]
        # Gather top scores with mask
        dummy_score = torch.finfo(step_scores.dtype).min
        top_step_scores = torch.where(step_mask, sorted_step_scores, dummy_score)
        top_step_scores = top_step_scores[:, :n_top_items]

        return top_step_scores, top_step_items

class BeamScorer(Scorer):
    def __init__(self, filter: Filter, max_beams: int):
        self.filter = filter
        self.max_beams = max_beams

        self._sum_scores = None
        self._flag = False
    
    def __call__(self, step_scores: torch.Tensor, beams_mask: torch.Tensor, step: int) -> ScorerResult:
        # Get top beams, items and scores of current step
        top_step_scores, top_step_items = self.filter(step_scores)
        # Number of top classes
        n_top_classes = top_step_items.shape[1]
        
        # Get or initialize sum of step scores
        prev_sum_scores = self._sum_scores
        if prev_sum_scores is None:
            prev_sum_scores = step_scores.new_zeros(len(top_step_scores), 1)
        # Update sum of step scores with mask
        dummy_score = torch.finfo(top_step_scores.dtype).min
        sum_scores = utils.decompress(top_step_scores, beams_mask, dummy_score)
        sum_scores += prev_sum_scores.unsqueeze(-1)

        # Prune beams by sum of step scores
        top_sum_scores, top_indices = utils.topk(sum_scores.flatten(start_dim=1), self.max_beams)
        # Get top beam indices and items
        top_beams = top_indices.div(n_top_classes, rounding_mode="floor")
        top_items = utils.decompress(top_step_items, beams_mask, _DUMMY_ITEM).flatten(start_dim=1)
        top_items = top_items.gather(-1, top_indices)

        self._sum_scores = top_sum_scores

        return top_beams, top_items, top_sum_scores/(step+1)

class DefaultStopCond(StopCond):
    def __init__(self, max_len: int, end_items: list[int]):
        self.max_len = max_len
        self.end_items = end_items
    
    def __call__(self, items: torch.Tensor, step: int) -> torch.Tensor:
        # Stop when reaching maximum length
        if step+2==self.max_len:
            return torch.ones_like(items, dtype=torch.bool)
        
        # Stop when encountering one of the end items
        end_items = items.new_tensor(self.end_items).unsqueeze(0)
        return (items.unsqueeze(-1)==end_items).any(-1)

@dataclass
class DecodeOutput(Generic[S]):
    seq: torch.Tensor
    score: float
    state: S

def decode(decoder: Decoder[S], scorer: Scorer, start_items: torch.Tensor, start_states: list[S],
    stop_cond: StopCond, n_outputs: int) -> list[list[DecodeOutput[S]]]:
    n_inputs = len(start_items)

    # Initial decoder inputs
    items = start_items.unsqueeze(-1)
    states = np.empty((len(start_states), 1), dtype=object)
    states[:, 0] = start_states
    step = 0
    # Initial beams mask
    beams_mask = torch.ones_like(items, dtype=torch.bool)
    beams_mask_np = beams_mask.cpu().numpy()
    # All items and previous beam indices
    all_prev_beams: list[Optional[torch.Tensor]] = [None]
    all_items = [items]
    # Decode outputs
    decode_outputs = [[] for _ in range(n_inputs)]
    done_count = 0

    while True:
        # Run decoding step
        step_scores, next_states_list = decoder(items[beams_mask], states[beams_mask_np].tolist(), step)

        next_states = np.empty(len(next_states_list), dtype=object)
        next_states[:] = next_states_list
        # Prune beams
        beams, items, beam_scores = scorer(step_scores, beams_mask, step)
        # Check stop condition for beams
        stop_flags = stop_cond(items, step)
        stop_flags_np = stop_flags.cpu().numpy()

        # Gather states for beams
        next_states = utils.decompress(next_states, beams_mask_np, None)
        next_states = np.take_along_axis(next_states, beams.cpu().numpy(), -1)
        # Update beams mask
        beams_mask = (items!=_DUMMY_ITEM)&(~stop_flags)
        beams_mask_np = beams_mask.cpu().numpy()
        # Collect active and done states
        done_states = next_states[stop_flags_np]
        states = next_states.copy()
        states[stop_flags_np] = None

        if len(done_states)>0:
            # Gather scores for completed beams
            done_beam_scores = beam_scores[stop_flags]

            # Sample indices
            sample_indices = torch.arange(n_inputs, device=beams.device)
            sample_indices = sample_indices.unsqueeze(-1).expand_as(beams)
            sample_indices = sample_indices[stop_flags]
            # Initial previous beam indices
            prev_beams = beams[stop_flags]
            
            done_items = [items[stop_flags]]
            # Gather generated items
            for past_items, past_beams in zip(reversed(all_items), reversed(all_prev_beams)):
                done_items.append(past_items[sample_indices, prev_beams])
                if past_beams is not None:
                    prev_beams = past_beams[sample_indices, prev_beams]
            
            done_items.reverse()
            done_seqs = torch.stack(done_items, -1)

            # Save completed sequences
            for sample_index, seq, beam_score, state in zip(
                sample_indices.tolist(), done_seqs, done_beam_scores.tolist(), done_states
            ):
                outputs = decode_outputs[sample_index]
                outputs.append(DecodeOutput(seq, beam_score, state))

                if len(outputs)==n_outputs:
                    done_count += 1
            # Generation is done for all inputs
            if done_count==n_inputs:
                break

        # Save beam indices and items
        all_prev_beams.append(beams)
        all_items.append(items)
        # Update step
        step += 1
    
    # Sort and prune to given number of outputs
    for outputs in decode_outputs:
        outputs.sort(key=lambda output: output.score, reverse=True)
        del outputs[n_outputs:]
    
    return decode_outputs

def beam_search_decode(decoder: Decoder[S], max_beams: int, start_items: torch.Tensor, start_states: list[S],
    max_len: int, end_items: list[int], n_outputs: int, k: Optional[int] = None, p: Optional[float] = None
    ) -> list[list[DecodeOutput[S]]]:
    if k is not None and p is not None:
        raise ValueError("cannot specify both `k` and `p` for beam search")
    elif k is not None:
        filter = TopKFilter(k)
    elif p is not None:
        filter = TopPFilter(p)
    else:
        raise ValueError("either `k` or `p` should be specified for beam search")

    return decode(
        decoder,
        scorer=BeamScorer(filter, max_beams),
        start_items=start_items,
        start_states=start_states,
        stop_cond=DefaultStopCond(max_len, end_items),
        n_outputs=n_outputs
    )

def greedy_decode(decoder: Decoder[S], start_items: torch.Tensor, start_states: list[S], max_len: int,
    end_items: list[int]) -> list[list[DecodeOutput[S]]]:
    return decode(
        decoder,
        scorer=GreedyScorer(),
        start_items=start_items,
        start_states=start_states,
        stop_cond=DefaultStopCond(max_len, end_items),
        n_outputs=1
    )
