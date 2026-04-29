"""Joint (single-outcome) fine-tuning utilities for Bio+ClinicalBERT.

Trains one model per outcome by jointly optimizing MLM + a single
auxiliary BCE head. Conceptually the same architecture as the
multi-task workflow but specialized for one outcome at a time.
"""

from .joint_finetuning import (
    joint_finetune,
    get_outcome_score,
    get_pseudo_data,
)

__all__ = ["joint_finetune", "get_outcome_score", "get_pseudo_data"]
