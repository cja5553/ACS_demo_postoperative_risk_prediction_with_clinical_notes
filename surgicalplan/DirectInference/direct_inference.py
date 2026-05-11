"""
Direct inference from a pre-trained MTL model on HuggingFace.

Wraps `get_postoperative_outcome_scores` with a sensible default model
identifier so that clinicians can call it without specifying a model path
or outcome list. Outcome names are auto-fetched from the model's metadata
file on HuggingFace Hub.
"""

import json
from typing import Dict, List, Optional, Union

from ..MultiTaskFinetuning.MultiTaskLearningPrediction import (
    get_postoperative_outcome_scores,
)


_DEFAULT_MODEL = "cja5553/BJH-perioperative-notes-bioClinicalBERT"


def _fetch_outcomes_from_hub(model_name: str, hf_token: Optional[str]) -> Optional[List[str]]:
    """Fetch outcome names from the model's mtl_metadata.json on HF Hub.

    Returns None if metadata can't be fetched (e.g., local path, gated repo
    without token, missing file). The caller can then either pass outcomes
    explicitly or let the underlying inference function raise.
    """
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=model_name,
            filename="mtl_metadata.json",
            repo_type="model",
            token=hf_token,
        )
        with open(path) as f:
            meta = json.load(f)
        return meta.get("outcome_cols")
    except Exception:
        return None


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
    outcomes : list of str, optional
        If not provided, attempts to fetch outcome names from the model's
        `mtl_metadata.json` on HuggingFace Hub.
    model_name : str, optional
        HuggingFace repo ID or local path. Defaults to
        `cja5553/BJH-perioperative-notes-bioClinicalBERT`.
    max_length, device, hf_token : optional
        Forwarded to `get_postoperative_outcome_scores`.

    Returns
    -------
    dict[str, float] or list[dict[str, float]]
    """
    if outcomes is None:
        outcomes = _fetch_outcomes_from_hub(model_name, hf_token)

    return get_postoperative_outcome_scores(
        model_name=model_name,
        text=text,
        outcomes=outcomes,
        max_length=max_length,
        device=device,
        hf_token=hf_token,
    )