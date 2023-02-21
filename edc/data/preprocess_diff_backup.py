from __future__ import annotations

from typing import TYPE_CHECKING

from array import array
from itertools import product

import torch

from .. import utils
from .types import Role, DataType, Task, SpecialToken as ST, ContextSegment, TargetDataPath, \
    TargetSeqs, EncoderData, DecoderData

if TYPE_CHECKING:
    from typing import Optional

    from ..types import Tokenizer, TODDialog
    from .types import TreeData

__all__ = [
    "ACTION_DATA_TYPES",
    "STATE_DATA_TYPES",
    "SPECIAL_DATA_NORM_MAPPING",
    "get_edc_special_tokens",
    "preprocess_data",
    "make_encoder_data",
    "make_decoder_data"
]

# Action data types
ACTION_DATA_TYPES = (
    DataType.DOMAIN,
    DataType.INTENT,
    DataType.SLOT_NAME,
    DataType.SLOT_VALUE
)

# State data types
STATE_DATA_TYPES = (
    DataType.DOMAIN,
    DataType.SLOT_NAME,
    DataType.SLOT_VALUE
)

def get_edc_special_tokens(max_rounds: int) -> list[str]:
    special_tokens: list[str] = []

    # Utterance tokens
    special_tokens.extend(utils.flatten((
        ST.U_START(role, round_id),
        ST.U_END(role, round_id)
    ) for role, round_id in product(Role, range(max_rounds))))

    # Node tokens
    special_tokens.extend(utils.flatten((
        ST.N_ANCESTOR(data_type),
        ST.N_START(data_type),
        ST.N_END(data_type)
    ) for data_type in DataType))

    special_tokens.extend(
        ST.N_SEP(x) for x in ("none", "new", "update", "del")
    )

    # Task type tokens
    special_tokens.extend(ST.TASK(task) for task in Task)
    
    return special_tokens

SPECIAL_DATA_NORM_MAPPING = {
    "@general": "@ general",
    "@none": "@ none",
    "@notcare": "@ not care"
}

def make_tree_seqs(tokenizer: Tokenizer, round_id: int, role: Role, task: Task,
    data: TreeData) -> TargetSeqs:
    # Data type hierarchy for task
    data_types = STATE_DATA_TYPES if task==Task.DIALOG_STATE else ACTION_DATA_TYPES
    # Traversal queue
    traversal_queue = [(TargetDataPath(round_id, role, task), data)]

    # Token IDs of target sequences
    target_token_ids: list[list[int]] = []
    # Lengths of target prompts
    target_prompt_lens: list[int] = []

    while traversal_queue:
        path, data = traversal_queue.pop()

        # Wrap single key as iterable
        keys = sorted(data)

        # Start and task token
        tokens = [tokenizer.eos_token, ST.TASK(task)]
        # Ancestor nodes
        for ancestor, data_type in zip(path.ancestors, data_types):
            ancestor = SPECIAL_DATA_NORM_MAPPING.get(ancestor, ancestor)
            tokens.append(ST.N_ANCESTOR(data_type))
            tokens.extend(tokenizer.tokenize(ancestor))
        # Save prompt length
        target_prompt_lens.append(len(tokens))
        
        # Children start token
        current_data_type = data_types[path.len]
        tokens.append(ST.N_START(current_data_type))
        # Children nodes
        for key in keys:
            sep = data[key][0] if task==Task.DIALOG_STATE else "none"
            key = SPECIAL_DATA_NORM_MAPPING.get(key, key)

            tokens.append(ST.N_SEP(sep))
            tokens.extend(tokenizer.tokenize(key))
        # Children end and end token
        tokens += (ST.N_END(current_data_type), tokenizer.eos_token)

        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        # Save target token IDs
        target_token_ids.append(token_ids)

        # Traverse through children
        if isinstance(data, dict):
            if task==Task.DIALOG_STATE:
                traversal_queue.extend(
                    (path.with_ancestor(key), value) \
                    for key, (_, value) in data.items() \
                    if value is not None
                )
            else:
                traversal_queue.extend(
                    (path.with_ancestor(key), value) for key, value in data.items()
                )
    
    return TargetSeqs(round_id, role, target_token_ids, target_prompt_lens)

def make_one_pass_impl(tokenizer: Tokenizer, data: TreeData, data_types: tuple[DataType, ...], level: int
    ) -> list[str]:
    # Current data type
    current_data_type = data_types[level]

    # Wrap single key as iterable
    if not isinstance(data, (dict, list)):
        data = [data]
    # Sort keys
    keys = sorted(keys)

    # Children start token
    tokens = [ST.N_START(current_data_type)]

    # Children nodes
    for key in keys:
        # Key data
        tokens.append(ST.N_SEP(current_data_type))
        tokens.extend(tokenizer.tokenize(key))
        # Traverse through children
        if isinstance(data, dict):
            make_one_pass_impl(tokenizer, data[key], data_types, level+1)
    
    # Children token
    tokens.append(ST.N_END(current_data_type))

    return tokens

def make_one_pass_seqs(tokenizer: Tokenizer, round_id: int, role: Role, task: Task, data: TreeData) -> TargetSeqs:
    # Data type hierarchy for task
    data_types = STATE_DATA_TYPES if task==Task.DIALOG_STATE else ACTION_DATA_TYPES

    # Start and task token
    tokens = [tokenizer.eos_token, ST.TASK(task)]
    # Serialize data into a single sequence
    tokens += make_one_pass_impl(tokenizer, data, data_types, level=0)
    # End token
    tokens.append(tokenizer.eos_token)

    # Convert tokens into IDs
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    # Wrap single sequence
    return TargetSeqs(round_id, role, [token_ids], [0])

def tokenize_utterance(tokenizer: Tokenizer, utterance: str, role: Role, round_id: int) -> list[int]:
    # Tokenize utterance
    tokens = tokenizer.tokenize(utterance)

    # Add start and end tokens
    tokens.insert(0, ST.U_START(role, round_id))
    tokens.append(ST.U_END(role, round_id))
    # Convert tokens to token IDs
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    return token_ids

import copy
from itertools import chain
if TYPE_CHECKING:
    from ..types import TODState

def make_ext_state(state: TODState, prev_state: TODState):
    ext_state = {}

    for domain, domain_state in state.items():
        prev_domain_state = prev_state.get(domain, {})
        domain_ext_state = {}
        
        for slot_name, slot_value in domain_state.items():
            prev_slot_value = prev_domain_state.get(slot_name)

            if prev_slot_value is None:
                slot_name_update = slot_value_update = "new"
            else:
                slot_name_update = "none"
                if prev_slot_value==slot_value:
                    slot_value_update = "none"
                else:
                    slot_value_update = "update"
            
            domain_ext_state[slot_name] = (slot_name_update, {slot_value: (slot_value_update, None)})
        
        for slot_name in prev_domain_state.keys():
            if slot_name not in domain_state:
                domain_ext_state[slot_name] = ("del", None)
        
        domain_update = "none" if prev_domain_state else "new"
        ext_state[domain] = (domain_update, domain_ext_state)

    for domain in prev_state.keys():
        if domain not in state:
            ext_state[domain] = ("del", None)
    
    return ext_state

def preprocess_data(tokenizer: Tokenizer, dialog: TODDialog, max_rounds: int, max_ctx_len: int,
    with_user_action: bool, with_sys_action: bool, one_pass: bool, standalone_round_idx: Optional[int]
    ) -> tuple[list[ContextSegment], list[TargetSeqs]]:
    ctx_segments: list[ContextSegment] = []
    target_seqs: list[TargetSeqs] = []

    # Make tree or one pass sequences
    make_seqs = make_one_pass_seqs if one_pass else make_tree_seqs

    ctx_len = 1
    prev_state = {}

    for i, round in zip(range(max_rounds), dialog["rounds"]):
        # Save target sequences when ...
        # 1) context features are shared
        # 2) sequences are for current round under standalone mode
        save_seqs = not (standalone_round_idx is not None and standalone_round_idx!=i)

        # Tokenize user input
        user_input_ids = tokenize_utterance(tokenizer, round["user_input"], Role.USER, i)
        # Do not append user input if exceeding maximum length
        if ctx_len+len(user_input_ids)<max_ctx_len:
            ctx_len += len(user_input_ids)
        else:
            user_input_ids = []
        # Make user segment
        ctx_segments.append(ContextSegment(i, Role.USER, user_input_ids))

        # Make target sequences for user action
        if with_user_action and save_seqs:
            target_seqs.append(make_seqs(
                tokenizer, i, Role.USER, Task.USER_ACTION, round["user_action"]
            ))
        # Make target sequences for dialog state
        ext_state = make_ext_state(round["state"], prev_state)
        if save_seqs:
            target_seqs.append(make_seqs(
                tokenizer, i, Role.USER, Task.DIALOG_STATE, ext_state
            ))
        prev_state = round["state"]

        # Tokenize system response
        sys_resp_ids = tokenize_utterance(tokenizer, round["sys_resp"], Role.SYSTEM, i)
        # Do not append system response if exceeding maximum length
        if ctx_len+len(sys_resp_ids)<max_ctx_len:
            ctx_len += len(sys_resp_ids)
        else:
            sys_resp_ids = []
        # Make system segment
        ctx_segments.append(ContextSegment(i, Role.SYSTEM, sys_resp_ids))

        # Make target sequences for system action
        if with_sys_action and save_seqs:
            target_seqs.append(make_seqs(
                tokenizer, i, Role.SYSTEM, Task.SYS_ACTION, round["sys_action"]
            ))
    
    return ctx_segments, target_seqs

def make_shared_encoder_data(segments: list[ContextSegment], start_token_id: int) -> EncoderData:
    ctx_token_ids = array("q", [start_token_id])

    ctx_len = 1
    ctx_lens: list[int] = []

    for segment in segments:
        # Append context IDs
        ctx_token_ids.extend(segment.token_ids)
        # Save current context length
        ctx_len += segment.len
        ctx_lens.append(ctx_len)
    
    # Make attention masks
    seq_len = ctx_lens[-1]
    encoder_attn_mask = torch.empty(seq_len, seq_len, dtype=torch.int64)
    decoder_cross_attn_mask = torch.empty(len(ctx_lens), seq_len, dtype=torch.int64)

    prev_ctx_len = 0
    for i, ctx_len in enumerate(ctx_lens):
        # Fill encoder attention mask
        encoder_segment_mask = encoder_attn_mask[prev_ctx_len:ctx_len]
        encoder_segment_mask[:, :ctx_len] = 1
        encoder_segment_mask[:, ctx_len:] = 0

        # Fill decoder cross attention mask
        cross_segment_mask = decoder_cross_attn_mask[i]
        cross_segment_mask[:ctx_len] = 1
        cross_segment_mask[ctx_len:] = 0

        prev_ctx_len = ctx_len
    
    return EncoderData(
        utils.as_tensor(ctx_token_ids),
        encoder_attn_mask,
        decoder_cross_attn_mask
    )

def make_standalone_encoder_data(segments: list[ContextSegment], start_token_id: int) -> EncoderData:
    ctx_token_ids = array("q", [start_token_id])
    ctx_len = 1

    for segment in segments:
        # Append context IDs
        ctx_token_ids.extend(segment.token_ids)
        # Save current context length
        ctx_len += len(segment.token_ids)
    
    # Make attention masks
    encoder_attn_mask = torch.ones(ctx_len, ctx_len, dtype=torch.int64)
    decoder_cross_attn_mask = torch.ones(ctx_len, dtype=torch.int64)

    return EncoderData(
        utils.as_tensor(ctx_token_ids),
        encoder_attn_mask,
        decoder_cross_attn_mask
    )

def make_decoder_data(target_seqs: list[TargetSeqs], standalone_ctx: bool = False) -> DecoderData:
    target_token_ids: list[torch.Tensor] = []
    decoder_self_attn_masks: list[torch.Tensor] = []
    target_output_masks: list[torch.Tensor] = []

    ctx_indices = array("q")

    for seqs in target_seqs:
        # Compute context index for shared mode
        if standalone_ctx:
            ctx_index = seqs.round_id*len(Role)+seqs.role.value
        # Context index is 0 for standalone mode
        else:
            ctx_index = 0

        for token_ids, target_prompt_len in zip(seqs.token_ids, seqs.target_prompt_lens):
            # Target IDs
            target_token_ids.append(torch.tensor(token_ids))

            # Decoder self attention mask
            decoder_self_attn_mask = torch.ones(len(token_ids), dtype=torch.int64)
            decoder_self_attn_masks.append(decoder_self_attn_mask)
            # Target output mask
            target_output_mask = torch.ones(len(token_ids), dtype=torch.bool)
            target_output_mask[:target_prompt_len+1] = False
            target_output_masks.append(target_output_mask)

            # Context index
            ctx_indices.append(ctx_index)
    
    # Collate decoder data
    return DecoderData(
        utils.pad_stack_tensors(target_token_ids, pad_value=-1),
        utils.pad_stack_tensors(decoder_self_attn_masks),
        utils.pad_stack_tensors(target_output_masks),
        utils.as_tensor(ctx_indices)
    )
