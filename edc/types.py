from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import TypedDict, Union

    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

    __all__ = [
        "TODDomainAction",
        "TODAction",
        "TODDomainState",
        "TODState",
        "TODSpan",
        "TODSpans",
        "TODRound",
        "TODDialog",
        "TODMetadata"
        "Tokenizer"
    ]

    # [ TOD Common Types ]
    # Slot value spans
    TODSpan = tuple[str, int, int]
    TODDomainSpans = dict[str, list[TODSpan]]
    TODSpans = dict[str, TODDomainSpans]
    # Dialog action
    TODDomainAction = dict[str, dict[str, list[str]]]
    TODAction = dict[str, TODDomainAction]
    # Dialog state
    TODDomainState = dict[str, str]
    TODState = dict[str, TODDomainState]

    class TODRound(TypedDict, total=False):
        user_input: str
        user_action: TODAction
        user_spans: TODSpans

        sys_resp: str
        sys_action: TODAction
        sys_spans: TODSpans

        state: TODState
    
    class TODDialog(TypedDict):
        name: str
        rounds: list[TODRound]
    
    class TODMetadata(TypedDict):
        name: str
        subsets: dict[str, list[str]]
    
    # [ Misc Types ]
    # Transformers tokenizer type
    Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
