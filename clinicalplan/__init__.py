"""
clinicalplan
============

Postoperative risk prediction from clinical notes via Bio+ClinicalBERT.

Two workflows:
    MultiTaskFinetuning — one model that predicts many outcomes jointly
    JointFinetuning     — one model per outcome (semi-supervised single-task)

Public API:
    mtl_finetune(df, text_col, outcome_cols, ...)
    get_postoperative_outcome_scores(model_name, text, ...)
    joint_finetune(df, text_col, outcome_col, ...)
    get_outcome_score(model_name, text, ...)
    get_pseudo_data()
"""

from .MultiTaskFinetuning.MultiTaskLearningPrediction import (
    mtl_finetune,
    get_postoperative_outcome_scores,
    get_pseudo_data,
)
from .JointFinetuning.joint_finetuning import (
    joint_finetune,
    get_outcome_score,
)

__version__ = "0.1.4"
__all__ = [
    "mtl_finetune",
    "get_postoperative_outcome_scores",
    "get_pseudo_data",
    "joint_finetune",
    "get_outcome_score",
]