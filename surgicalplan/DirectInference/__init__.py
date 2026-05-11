"""Direct inference from a pre-trained Bio+ClinicalBERT MTL model.

Provides a clinician-friendly one-call API: pass clinical text and the
outcomes of interest, get probabilities back. No training required.
"""

from .direct_inference import direct_inference_from_trained_model

__all__ = ["direct_inference_from_trained_model"]