from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from Levenshtein import distance as lev_distance

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..types import TODAction, TODState
    from .types import PredResult

__all__ = [
    "compute_action_metrics",
    "state_matches"
]

def div(x, y):
    if y==0.:
        return float(x==y)
    else:
        return x/y

def flatten_action(action: TODAction) -> Iterator[tuple[str, ...]]:
    return (
        (domain, intent, slot_name, slot_value) \
        for domain, intents_slots in action.items() \
        for intent, slots in intents_slots.items() \
        for slot_name, slot_values in slots.items() \
        for slot_value in slot_values
    )

def compute_action_metrics(dialog_round: TODRound, pred: PredResult, role: str) -> np.ndarray:
    action = dialog_round.get(role+"_action", {})
    pred_action = pred.get(role+"_action", {})

    # Flatten ground truth and predicted actions into pairs
    action_pairs = [
        (domain, intent, slot_name, slot_value) \
        for domain, intents_slots in action.items() \
        for intent, slots in intents_slots.items() \
        for slot_name, slot_values in slots.items() \
        for slot_value in slot_values
    ]
    pred_action_pairs = [
        (domain, intent, slot_name, slot_value) \
        for domain, intents_slots in pred_action.items() \
        for intent, slots in intents_slots.items() \
        for slot_name, slot_values in slots.items() \
        for slot_value in slot_values
    ]

    metrics = []

    # Compute precision, recall and F1-score
    for level in range(4):
        partial_pairs = set(pair[:level+1] for pair in action_pairs)
        pred_partial_pairs = set(pair[:level+1] for pair in pred_action_pairs)

        tp_partial_pairs = partial_pairs.intersection(pred_partial_pairs)
        precision = div(len(tp_partial_pairs), len(pred_partial_pairs))
        recall = div(len(tp_partial_pairs), len(partial_pairs))

        if precision==0. and recall==0.:
            f1 = 0.
        else:
            f1 = (2*precision*recall)/(precision+recall)

        metrics.append((precision, recall, f1))
    
    return np.array(metrics)
    
def state_matches(pred_state: TODState, target_state: TODState, exact_match_slots: set[str],
    max_lev_dist_factor: float = 0.2) -> bool:
    # Domains should match
    if pred_state.keys()!=target_state.keys():
        return False

    for domain, pred_domain_state in pred_state.items():
        target_domain_state = target_state[domain]
        # Slots should match
        if pred_domain_state.keys()!=target_domain_state.keys():
            return False

        for slot_name, pred_value in pred_domain_state.items():
            target_value = target_domain_state[slot_name]

            # Exact match
            if pred_value==target_value:
                continue
            # Slot value must match exactly
            if slot_name in exact_match_slots:
                return False

            # Values with or without "the" are considered identical
            if "the "+pred_value==target_value or "the "+target_value==pred_value:
                continue
            # Fuzzy matching
            max_lev_dist = int(max_lev_dist_factor*len(target_value))
            if lev_distance(pred_value, target_value)>max_lev_dist:
                return False

    return True
