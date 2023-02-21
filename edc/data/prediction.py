from __future__ import annotations

from typing import TYPE_CHECKING

from array import array

import torch

from .. import utils
from .types import Role, Task, SpecialToken as ST, TargetDataPath, PredData
from .preprocess import STATE_DATA_TYPES, SPECIAL_DATA_NORM_MAPPING

if TYPE_CHECKING:
    from collections.abc import Generator

    from ..types import Tokenizer, TODState
    from .types import TreeData

__all__ = [
    "pred_states_one_pass",
    "pred_states_tree"
]

SPECIAL_DATA_RESTORE_MAPPING = {
    "@ general": "@general",
    "@ none": "@none",
    "@ not care": "@notcare"
}

def pred_states_tree(tokenizer: Tokenizer, n_rounds: int, standalone_ctx: bool,
    n_pred_seqs: int) -> Generator[PredData, torch.Tensor, list[TODState]]:
    # Current data paths
    current_paths = [
        TargetDataPath(i, Role.USER, Task.DIALOG_STATE)
        for i in range(n_rounds)
    ]
    # Next and done data paths
    next_paths: list[TargetDataPath] = []
    done_paths: list[TargetDataPath] = []

    # Prediction loop
    while current_paths:
        path_token_ids: list[torch.Tensor] = []
        children_start_ids = array("q")
        ctx_indices = array("q")

        # Fetch paths for prediction
        paths = current_paths[:n_pred_seqs]
        del current_paths[:n_pred_seqs]

        for path in paths:
            # Task token
            tokens = [tokenizer.eos_token, ST.TASK(path.task)]
            # Ancestor nodes
            for ancestor, data_type in zip(path.ancestors, STATE_DATA_TYPES):
                ancestor = SPECIAL_DATA_NORM_MAPPING.get(ancestor, ancestor)
                tokens.append(ST.N_ANCESTOR(data_type))
                tokens.extend(tokenizer.tokenize(ancestor))
            
            # Save path token IDs
            path_token_ids.append(
                torch.tensor(tokenizer.convert_tokens_to_ids(tokens))
            )
            # Save children start token
            current_data_type = STATE_DATA_TYPES[path.len]
            children_start_ids.append(
                tokenizer.convert_tokens_to_ids(ST.N_START(current_data_type))
            )
            
            # Standalone mode
            if standalone_ctx:
                ctx_index = 0
            # Shared mode
            else:
                ctx_index = path.round_id*len(Role)+Role.USER.value
            # Save context index
            ctx_indices.append(ctx_index)
        
        # Yield a batch of prediction data
        children_token_ids = yield PredData(
            utils.as_tensor(ctx_indices),
            path_token_ids,
            utils.as_tensor(children_start_ids)
        )

        for token_ids, path in zip(children_token_ids, paths):
            # Get data type hierarchy and current data path
            current_data_type = STATE_DATA_TYPES[path.len]
            # Get children data
            children_seq = tokenizer.convert_tokens_to_string(
                tokenizer.convert_ids_to_tokens(token_ids[1:-2].tolist())
            )
            children_data = children_seq.split(ST.N_SEP(current_data_type))
            if not children_data[0]:
                children_data = []
            
            for data in children_data:
                # Make new path with children data
                data = data.lstrip()
                data = SPECIAL_DATA_RESTORE_MAPPING.get(data, data)
                new_path = path.with_ancestor(data)
                # Save path
                paths = done_paths if new_path.len==len(STATE_DATA_TYPES) else next_paths
                paths.append(new_path)
        
        # Replace current paths by next paths
        if not current_paths:
            current_paths = next_paths
            next_paths = []
    
    pred_states: list[TODState] = [{} for _ in range(n_rounds)]
    # Gather dialog states for rounds from paths
    for path in done_paths:
        domain, slot_name, slot_value = path.ancestors

        # Update predicted state
        pred_state = pred_states[path.round_id]
        domain_state = pred_state.setdefault(domain, {})
        domain_state[slot_name] = slot_value

    return pred_states

def parse_one_pass_seq(tokenizer: Tokenizer, seq: list[str], token_idx: int = 0, level: int = 0
    ) -> tuple[TreeData, int]:
    current_data_type = STATE_DATA_TYPES[level]
    is_last_level = level<len(STATE_DATA_TYPES)-1

    data: TreeData = None if is_last_level else {}

    while True:
        # Terminal tokens for key
        key_stop_tokens = (ST.N_SEP(current_data_type), ST.N_END(current_data_type))
        if is_last_level:
            key_stop_tokens += (ST.N_START(STATE_DATA_TYPES[level+1]))

        key_tokens: list[str] = []
        # Gather key tokens
        while True:
            token_idx += 1
            token = seq[token_idx]

            if token in key_stop_tokens:
                break
            else:
                key_tokens.append(token)
        
        # Convert key tokens back into key
        key = tokenizer.convert_tokens_to_string(key_tokens).lstrip()
        key = SPECIAL_DATA_RESTORE_MAPPING.get(key, key)

        # Save slot value
        if is_last_level:
            data = key
        # Parse and save domain and slot name data
        else:
            # Recursively parse key data
            key_data, token_idx = parse_one_pass_seq(seq, token_idx, level+1)
            data[key] = key_data
            # Advance pointer
            token_idx += 1
            token = seq[token_idx]
        
        # End of current level
        if token==ST.N_END(current_data_type):
            break
    
    return data, token_idx

def pred_states_one_pass(tokenizer: Tokenizer, n_rounds: int, standalone_ctx: bool, _: int,
    ) -> Generator[PredData, torch.Tensor, list[TODState]]:
    # Standalone mode
    if standalone_ctx:
        ctx_indices = torch.tensor([0])
    # Shared mode
    else:
        ctx_indices = torch.tensor([
            round_id*len(Role)+Role.USER.value for round_id in range(n_rounds)
        ])
    
    # Prompt token IDs
    prompt_token_ids = [torch.tensor([tokenizer.eos_token_id])]*n_rounds
    # Start IDs
    task_token_id = tokenizer.convert_tokens_to_ids(ST.TASK(Task.DIALOG_STATE))
    start_ids = torch.full((n_rounds,), task_token_id, dtype=torch.int64)

    # Yield a batch of prediction data
    children_token_ids = yield PredData(ctx_indices, prompt_token_ids, start_ids)

    pred_states: list[TODState] = [{} for _ in range(n_rounds)]
    # Parse state sequences and save predicted states
    for i, token_ids in enumerate(children_token_ids):
        one_pass_seq = tokenizer.convert_ids_to_tokens(token_ids.tolist())
        pred_state = parse_one_pass_seq(tokenizer, one_pass_seq)
        pred_states[i] = pred_state
    
    return pred_states
