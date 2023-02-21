from __future__ import annotations

from typing import TYPE_CHECKING

from zipfile import ZipFile

import numpy as np
from Levenshtein import distance as lev_distance

from edc import utils
from edc.data import METADATA_PATH

if TYPE_CHECKING:
    from ..types import TODState, TODMetadata

__all__ = [
    "state_matches",
    "evaluate_preds"
]

def state_matches(pred_state: TODState, target_state: TODState, max_lev_dist_factor: float = 0.2) -> bool:
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
            # Values with or without "the" are considered identical
            if "the "+pred_value==target_value or "the "+target_value==pred_value:
                continue

            # Fuzzy matching
            max_lev_dist = int(max_lev_dist_factor*len(target_value))
            if lev_distance(pred_value, target_value)>max_lev_dist:
                return False

    return True

def evaluate_preds(dataset_path: str, preds_path: str, subset: str) -> float:
    # Number of total and correct rounds
    n_rounds = 0
    n_correct_rounds = 0

    with ZipFile(dataset_path) as f_archive_dataset, ZipFile(preds_path) as f_archive_preds:
        # Get dialog paths in the subset
        metadata: TODMetadata = utils.load_json(METADATA_PATH, root=f_archive_dataset)
        dialog_paths = metadata["subsets"][subset]

        for dialog_path in dialog_paths:
            # Load dialog and predictions
            dialog = utils.load_json(dialog_path, root=f_archive_dataset)
            preds = utils.load_json(dialog_path, root=f_archive_preds)

            # Evaluate predictions for rounds
            for round, round_pred in zip(dialog["rounds"], preds["preds"]):
                n_rounds += 1
                n_correct_rounds += int(state_matches(round["state"], round_pred["state"]))
    
    # Compute JGA
    joint_accuracy = n_correct_rounds/n_rounds

    return joint_accuracy
