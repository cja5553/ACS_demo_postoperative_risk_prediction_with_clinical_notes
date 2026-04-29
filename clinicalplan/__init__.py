"""
clinicalplan
============

Multi-task learning fine-tuning of Bio+ClinicalBERT for predicting
postoperative outcomes from clinical notes.

Public API:
    mtl_finetune(df, text_col, outcome_cols, ...)
    get_postoperative_outcome_scores(model_name, text, ...)
    get_pseudo_data()
"""

from .MultiTaskFinetuning.MultiTaskLearningPrediction import (
    mtl_finetune,
    get_postoperative_outcome_scores,
    get_pseudo_data,
)

__version__ = "0.1.2"
__all__ = ["mtl_finetune", "get_postoperative_outcome_scores", "get_pseudo_data"]