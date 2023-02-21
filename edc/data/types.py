from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple
from enum import Enum

__all__ = [
    "Role",
    "DataType",
    "Task",
    "SpecialToken",
    "ContextSegment",
    "TargetDataPath",
    "TargetSeqs",
    "EncoderData",
    "DecoderData",
    "PredData"
]

if TYPE_CHECKING:
    from typing import Any, TypedDict, Union

    import torch

    from ..types import TODAction, TODState

    # Tree-structured data
    TreeData = Union[dict[str, "TreeData"], list[str], str]
    # Single data sample
    EDCSample = tuple["EncoderData", "DecoderData"]

    class PredResult(TypedDict):
        """ Prediction result dictionary type. """
        user_action: TODAction
        dialog_state: TODState
        sys_action: TODAction

    __all__ += [
        "TreeData",
        "EDCSample",
        "PredResult"
    ]

class Role(Enum):
    """ Role of context segment. """
    USER = 0
    SYSTEM = 1

class DataType(Enum):
    """ Data type of tree node values. """
    DOMAIN = 0
    INTENT = 1
    SLOT_NAME = 2
    SLOT_VALUE = 3

class Task(Enum):
    """ Dialog task types. """
    USER_ACTION = 0
    DIALOG_STATE = 1
    SYS_ACTION = 2

    # temp
    MLM = 3

class SpecialToken(Enum):
    """ Special token families. """
    # Utterances
    U_START = "u_start"
    U_END = "u_end"

    # Tree nodes
    N_ANCESTOR = "n_ancestor"
    N_START = "n_start"
    N_END = "n_end"

    N_SEP = "n_sep"

    # Task types
    TASK = "task"

    def __call__(self, *values: Any) -> str:
        # Prefix
        token = "["+self.value
        # Template values
        for value in values:
            if isinstance(value, Enum):
                value = value.name.lower()
            else:
                value = str(value)
            token += ":"+value
        # Suffix
        token += "]"

        return token

class ContextSegment(NamedTuple):
    """ A context segment containing user or system utterances. """
    round_id: int
    role: Role
    token_ids: list[int]

    @property
    def len(self) -> int:
        return len(self.token_ids)

class TargetDataPath(NamedTuple):
    """ Data path from target tree root to node. """
    round_id: int
    role: Role
    task: Task

    ancestors: tuple[str, ...] = ()

    @property
    def len(self) -> int:
        return len(self.ancestors)

    def with_ancestor(self, ancestor: str) -> TargetDataPath:
        return self._replace(ancestors=self.ancestors+(ancestor,))

class TargetSeqs(NamedTuple):
    round_id: int
    role: Role

    token_ids: list[list[int]]
    target_prompt_lens: list[int]

class EncoderData(NamedTuple):
    """ Data needed for the encoder part of the model. """
    ctx_token_ids: torch.Tensor

    encoder_attn_mask: torch.Tensor
    decoder_cross_attn_mask: torch.Tensor

class DecoderData(NamedTuple):
    """ Data needed for the decoder part of the model. """
    target_token_ids: torch.Tensor
    decoder_self_attn_masks: torch.Tensor
    target_output_masks: torch.Tensor

    ctx_indices: torch.Tensor

class PredData(NamedTuple):
    """ Data needed for dialog state and action prediction. """
    ctx_indices: torch.Tensor
    token_ids: list[torch.Tensor]
    start_ids: torch.Tensor
