from __future__ import annotations

from typing import TYPE_CHECKING

from array import array
from itertools import product, repeat

import torch

from .. import utils
from .types import Role, DataType, Task, SpecialToken as ST, ExtSlotInfo, \
    ContextSegment, TargetDataPath, TargetSeqs, EncoderData, DecoderData

if TYPE_CHECKING:
    from typing import Optional

    from ..types import Tokenizer, TODDialog, TODState, TODSpans
    from .types import TreeData

    # Slot occurrence table
    OccurrenceTable = dict[str, set[tuple[str, str]]]
    # Extended domain state
    ExtDomainState = dict[str, ExtSlotInfo]
    # Extended dialog state
    ExtState = dict[str, ExtDomainState]

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
        ST.N_SEP(data_type),
        ST.N_END(data_type)
    ) for data_type in DataType))

    # Task type tokens
    special_tokens.extend(ST.TASK(task) for task in Task)
    
    return special_tokens

SPECIAL_DATA_NORM_MAPPING = {
    "@general": "@ general",
    "@none": "@ none",
    "@notcare": "@ not care"
}

def gather_occurrence(occurrence_table: OccurrenceTable, spans: TODSpans):
    for domain, domain_spans in spans.items():
        for slot_name, slot_spans in domain_spans.items():
            for slot_value, _, _ in slot_spans:
                occurrence_table.setdefault(slot_value, set()).add((domain, slot_name))

def make_ext_state(state: TODState, occurrence_table: OccurrenceTable) -> ExtState:
    ext_state: ExtState = {}

    for domain, domain_state in state.items():
        ext_domain_state: ExtDomainState = {}

        for slot_name, slot_value in domain_state.items():
            # Check whether slot value is copied from context
            slot_occurrences = occurrence_table.get(slot_value, ())
            is_copied = (domain, slot_name) in slot_occurrences
            # Save extended slot information
            ext_domain_state[slot_name] = ExtSlotInfo(slot_value, is_copied)
        
        ext_state[domain] = ext_domain_state
    
    return ext_state

def make_tree_seqs(tokenizer: Tokenizer, round_id: int, role: Role, task: Task, data: TreeData,
    slot_name_weight: float = 1., non_copied_value_weight: float = 1.) -> TargetSeqs:
    # Data type hierarchy for task
    data_types = STATE_DATA_TYPES if task==Task.DIALOG_STATE else ACTION_DATA_TYPES
    # Traversal queue
    traversal_queue = [(TargetDataPath(round_id, role, task), data)]

    # Token IDs of target sequences
    target_token_ids: list[list[int]] = []
    # target prompt lengths
    target_prompt_lens: list[int] = []
    # Target sequence weights
    target_weights: list[float] = []

    while traversal_queue:
        path, data = traversal_queue.pop()

        # Wrap single key as iterable
        if not isinstance(data, (dict, list)):
            data = [data]
        # Sort keys
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
        for i, key in enumerate(keys):
            # Separator
            if i>0:
                tokens.append(ST.N_SEP(current_data_type))
            # Key data
            key = str(key)
            norm_key = SPECIAL_DATA_NORM_MAPPING.get(key, key)
            tokens.extend(tokenizer.tokenize(norm_key))
        # Children end and end token
        tokens += (ST.N_END(current_data_type), tokenizer.eos_token)

        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        # Save target token IDs
        target_token_ids.append(token_ids)

        # Dialog state
        if task==Task.DIALOG_STATE:
            # Slot names prediction
            if current_data_type==DataType.SLOT_NAME:
                target_weight = slot_name_weight
            # Non-copied slot value prediction
            elif current_data_type==DataType.SLOT_VALUE and isinstance(data[0], ExtSlotInfo) \
                and not data[0].is_copied:
                target_weight = non_copied_value_weight
            # Other situations
            else:
                target_weight = 1.
        # User or system actions
        else:
            target_weight = 1.
        # Save target sequence weight
        target_weights.append(target_weight)

        # Traverse through children
        if isinstance(data, dict):
            traversal_queue.extend((path.with_ancestor(key), data[key]) for key in keys)
    
    return TargetSeqs(round_id, role, task, target_token_ids, target_prompt_lens, target_weights)

def make_one_pass_impl(tokenizer: Tokenizer, data: TreeData, data_types: tuple[DataType, ...],
    level: int) -> list[str]:
    # Current data type
    current_data_type = data_types[level]

    # Wrap single key as iterable
    if not isinstance(data, (dict, list)):
        data = [data]
    # Sort keys
    keys = sorted(data)

    # Children start token
    tokens = [ST.N_START(current_data_type)]
    # Children nodes
    for i, key in enumerate(keys):
        # Separator
        if i>0:
            tokens.append(ST.N_SEP(current_data_type))
        # Key data
        key = str(key)
        norm_key = SPECIAL_DATA_NORM_MAPPING.get(key, key)
        tokens.extend(tokenizer.tokenize(norm_key))
        # Traverse through children
        if isinstance(data, dict):
            tokens += make_one_pass_impl(tokenizer, data[key], data_types, level+1)
    # Children token
    tokens.append(ST.N_END(current_data_type))

    return tokens

def make_one_pass_seqs(tokenizer: Tokenizer, round_id: int, role: Role, task: Task, data: TreeData,
    _1: float = 1., _2: float = 1.) -> TargetSeqs:
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
    return TargetSeqs(round_id, role, task, [token_ids], [0], [1.])

def tokenize_utterance(tokenizer: Tokenizer, utterance: str, role: Role, round_id: int) -> list[int]:
    # Tokenize utterance
    tokens = tokenizer.tokenize(utterance)

    # Add start and end tokens
    tokens.insert(0, ST.U_START(role, round_id))
    tokens.append(ST.U_END(role, round_id))
    # Convert tokens to token IDs
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    return token_ids

def preprocess_data(tokenizer: Tokenizer, dialog: TODDialog, max_rounds: int, max_ctx_len: int,
    slot_name_weight: float, non_copied_value_weight: float, with_user_action: bool,
    with_sys_action: bool, one_pass: bool, standalone_round_idx: Optional[int] = None
    ) -> tuple[list[ContextSegment], list[TargetSeqs]]:
    ctx_segments: list[ContextSegment] = []
    target_seqs: list[TargetSeqs] = []

    occurrence_table: OccurrenceTable = {}

    # Make tree or one pass sequences
    make_seqs = make_one_pass_seqs if one_pass else make_tree_seqs
    # Context length
    ctx_len = 1

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

        # Gather slot value occurrence for user
        emphasize_non_copied_value = non_copied_value_weight!=1.
        if emphasize_non_copied_value:
            gather_occurrence(occurrence_table, round["user_spans"])
        # Make target sequences for user action
        if save_seqs and with_user_action:
            target_seqs.append(make_seqs(
                tokenizer, i, Role.USER, Task.USER_ACTION, round["user_action"]
            ))
        
        # Make target sequences for dialog state
        if save_seqs:
            state = round["state"]
            if emphasize_non_copied_value!=1.:
                state = make_ext_state(state, occurrence_table)
            
            target_seqs.append(make_seqs(
                tokenizer, i, Role.USER, Task.DIALOG_STATE, state,
                slot_name_weight, non_copied_value_weight
            ))
        
        # Exit early in standalone mode
        if standalone_round_idx is not None and i==standalone_round_idx:
            break

        # Tokenize system response
        sys_resp_ids = tokenize_utterance(tokenizer, round["sys_resp"], Role.SYSTEM, i)
        # Do not append system response if exceeding maximum length
        if ctx_len+len(sys_resp_ids)<max_ctx_len:
            ctx_len += len(sys_resp_ids)
        else:
            sys_resp_ids = []
        # Make system segment
        ctx_segments.append(ContextSegment(i, Role.SYSTEM, sys_resp_ids))

        # Gather slot value occurrence for system
        if emphasize_non_copied_value:
            gather_occurrence(occurrence_table, round["sys_spans"])
        # Make target sequences for system action
        if save_seqs and with_sys_action:
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
    decoder_cross_attn_mask = torch.ones(1, ctx_len, dtype=torch.int64)

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
    target_weights = array("f")

    for seqs in target_seqs:
        # Context index is 0 for standalone mode
        if standalone_ctx:
            ctx_index = 0
        # Compute context index for shared mode
        else:
            ctx_index = seqs.round_id*len(Role)+seqs.role.value

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
        
        n_seqs = len(seqs.token_ids)
        
        # Context index
        ctx_indices.extend(repeat(ctx_index, n_seqs))
        # Target weights
        target_weights.extend(seqs.target_weights)
    
    # Collate decoder data
    return DecoderData(
        utils.pad_stack_tensors(target_token_ids, pad_value=-1),
        utils.pad_stack_tensors(decoder_self_attn_masks),
        utils.pad_stack_tensors(target_output_masks),
        utils.as_tensor(ctx_indices),
        utils.as_tensor(target_weights)
    )
