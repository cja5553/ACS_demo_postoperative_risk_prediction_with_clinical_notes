"""
demo_clinical_notes_risk_prediction
===================================

Multi-task learning fine-tuning of Bio+ClinicalBERT for predicting
postoperative outcomes from clinical notes.

Public API:
    mtl_finetune(df, text_col, outcome_cols, ...)
    get_postoperative_outcome_scores(model_name, text, ...)
"""

from .MultiTaskLearningPrediction import mtl_finetune, get_postoperative_outcome_scores

__version__ = "0.1.0"
__all__ = ["mtl_finetune", "get_postoperative_outcome_scores"]