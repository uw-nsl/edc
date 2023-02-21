from __future__ import annotations

from typing import TYPE_CHECKING

from zipfile import ZipFile, ZIP_DEFLATED

from torch import distributed as dist
from pytorch_lightning.callbacks import BasePredictionWriter

from .. import utils

if TYPE_CHECKING:
    from pytorch_lightning import Trainer

    from ..types import TODState

__all__ = [
    "EDCPredsWriter"
]

class EDCPredsWriter(BasePredictionWriter):
    def __init__(self, outputs_path: str, standalone_ctx: bool):
        super().__init__(write_interval="epoch")

        self.outputs_path = outputs_path
        self.standalone_ctx = standalone_ctx

    def write_on_epoch_end(self, trainer: Trainer, _1, node_pred_states: list, _2):
        all_pred_states: list[list] = [None]*dist.get_world_size()

        # Gather predictions from all nodes
        # (Use `all_gather_object` instead of `gather_object` to work around NCCL limitations)
        dist.all_gather_object(all_pred_states, node_pred_states[0])
        # Do not run on other nodes
        if trainer.global_rank!=0:
            return
        
        # Iterator of predictions
        pred_states_iter = utils.flatten(all_pred_states)
        # Gather predictions for same dialog in standalone mode
        if self.standalone_ctx:
            all_pred_states_map: dict[str, list[TODState]] = {}

            for (dialog_path, round_id), round_states in pred_states_iter:
                pred_states = all_pred_states_map.setdefault(dialog_path, [])
                # Expand predictions
                while len(pred_states)<=round_id:
                    pred_states.append({})
                # Save predicted round state
                pred_states[round_id] = round_states[0]
            
            pred_states_iter = all_pred_states_map.items()
        
        # Save prediction results to archive
        with ZipFile(self.outputs_path, "w", ZIP_DEFLATED) as f_archive_out:
            for dialog_path, pred_states in pred_states_iter:
                utils.save_json({
                    "name": dialog_path,
                    "preds": [{"state": round_state} for round_state in pred_states]
                }, dialog_path, root=f_archive_out)
