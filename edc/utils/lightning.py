from __future__ import annotations

import warnings

__all__ = [
    "ignore_pl_warnings"
]

def ignore_pl_warnings():
    # Number of data loader workers may not be sufficient
    warnings.filterwarnings("ignore", r"The dataloader, .*, does not have many workers")
    # Wrong order of optimizer and LR scheduler stepping
    warnings.filterwarnings("ignore", r"Detected call of `lr_scheduler\.step\(\)`")
