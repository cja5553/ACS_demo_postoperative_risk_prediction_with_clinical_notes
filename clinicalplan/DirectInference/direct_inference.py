"""
Direct inference from a pre-trained MTL model on HuggingFace.

Wraps `get_postoperative_outcome_scores` with a sensible default model
identifier so that clinicians can call it without specifying a model path.
"""

from typing import Dict, List, Optional, Union

from ..MultiTaskFinetuning.MultiTaskLearningPrediction import (
    get_postoperative_outcome_scores,
)


# Default HuggingFace model — Bio+ClinicalBERT continued-pretrained on
# BJH perioperative notes and fine-tuned for postoperative risk prediction.
_DEFAULT_MODEL = "cja5553/BJH-perioperative-notes-bioClinicalBERT"


def direct_inference_from_trained_model(
    text: Union[str, List[str]],
    outcomes: Optional[List[str]] = None,
    model_name: str = _DEFAULT_MODEL,
    max_length: Optional[int] = None,
    device: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> Union[Dict[str, float], List[Dict[str, float]]]:
    """
    Score clinical text against a pre-trained multi-task model without any
    fine-tuning step. The model is downloaded from HuggingFace Hub on first
    use and cached locally thereafter.

    Parameters
    ----------
    text : str or list of str
        One clinical scenario, or a list of them.
    outcomes : list of str, optional
        Which outcomes to score. Defaults to all outcomes the model was
        trained on (recovered from `mtl_metadata.json` in the model repo).
    model_name : str, optional
        HuggingFace repo ID or local path. Defaults to
        `cja5553/BJH-perioperative-notes-bioClinicalBERT`.
    max_length : int, optional
        Token sequence length. Defaults to value used during fine-tuning.
    device : str, optional
        "cuda", "cpu", or None to auto-detect.
    hf_token : str, optional
        HuggingFace token for gated/private models. Required if the
        repo is gated.

    Returns
    -------
    dict[str, float] when text is a string, list[dict[str, float]] when
    text is a list. Each dict maps outcome name to probability in [0, 1].

    Examples
    --------
    >>> direct_inference_from_trained_model(
    ...     text="83-year-old male, ASA 4, scheduled for CABG.",
    ...     outcomes=["DVT", "PE", "death_in_30"],
    ... )
    {'DVT': 0.18, 'PE': 0.07, 'death_in_30': 0.46}
    """
    return get_postoperative_outcome_scores(
        model_name=model_name,
        text=text,
        outcomes=outcomes,
        max_length=max_length,
        device=device,
        hf_token=hf_token,
    )