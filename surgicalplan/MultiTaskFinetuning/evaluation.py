"""Evaluation metrics for a fine-tuned multi-task model.

`evaluate_data` scores every note in an evaluation dataframe against each
outcome head and reports, per outcome, both threshold-dependent metrics
(accuracy, precision, recall, F1 -- computed at `threshold`) and
threshold-free metrics (AUROC, AUPRC). The threshold-free pair does not depend
on the cutoff and is the more reliable summary when a model's probabilities are
poorly calibrated (e.g. after only one training epoch).
"""

from typing import Dict, List, Sequence

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from .MultiTaskLearningPrediction import get_postoperative_outcome_scores


def _metrics_for_one_outcome(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: float,
) -> Dict[str, float]:
    """Compute the six metrics for a single outcome column.

    Threshold-dependent metrics (accuracy/precision/recall/F1) binarise the
    probabilities at `threshold`. Threshold-free metrics (AUROC/AUPRC) use the
    probabilities directly.

    Metrics that are undefined for the given labels are returned as ``float('nan')``
    rather than raising: AUROC/AUPRC need both classes present, and
    precision/recall/F1 are ill-defined when a class is absent from predictions
    or truth.
    """
    y_true = list(y_true)
    y_prob = list(y_prob)
    y_pred = [1 if p >= threshold else 0 for p in y_prob]

    n_pos = int(sum(y_true))
    both_classes = 0 < n_pos < len(y_true)
    nan = float("nan")

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        # zero_division=0 keeps precision/recall well-defined (0.0) when the
        # model predicts no positives, instead of emitting a warning.
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        # AUROC/AUPRC are only defined when both classes are present in y_true.
        "auroc": roc_auc_score(y_true, y_prob) if both_classes else nan,
        "auprc": average_precision_score(y_true, y_prob) if both_classes else nan,
        "support": len(y_true),
        "n_pos": n_pos,
    }


def evaluate_data(
    eval_data: pd.DataFrame,
    outcomes: List[str],
    model: str,
    threshold: float = 0.5,
    text_col: str = "clinical_note",
    max_length: int = 512,
    device: str = None,
    hf_token: str = None,
) -> pd.DataFrame:
    """Evaluate a fine-tuned multi-task model on a labelled dataframe.

    Parameters
    ----------
    eval_data : pandas.DataFrame
        Must contain `text_col` and one 0/1 column per name in `outcomes`.
        Use get_pseudo_evaluation_data() for the bundled 50-row demo set.
    outcomes : list of str
        Outcome columns to score, e.g. ["DVT", "Pneumonia", "AKI", "Delirium"].
    model : str
        Path or name passed straight to get_postoperative_outcome_scores
        (the same identifier used for scoring, e.g. "my_run").
    threshold : float, default 0.5
        Cutoff for the threshold-dependent metrics. Probabilities >= threshold
        count as a positive prediction. AUROC/AUPRC ignore this. When a model's
        scores cluster low, a lower threshold (e.g. 0.2) gives a truer picture of
        precision/recall than 0.5.
    text_col : str, default "clinical_note"
        Name of the note column in `eval_data`.
    max_length, device, hf_token
        Passed through to get_postoperative_outcome_scores.

    Returns
    -------
    pandas.DataFrame
        One row per outcome plus a final "macro avg" row, with columns:
        accuracy, precision, recall, f1, auroc, auprc, support, n_pos.
        (support/n_pos are blank on the macro-avg row.)
    """
    missing = [c for c in outcomes if c not in eval_data.columns]
    if missing:
        raise ValueError(
            f"eval_data is missing outcome column(s): {missing}. "
            f"Columns present: {list(eval_data.columns)}"
        )
    if text_col not in eval_data.columns:
        raise ValueError(
            f"eval_data has no text column '{text_col}'. "
            f"Columns present: {list(eval_data.columns)}"
        )

    # Score every note in one batched call. Passing a list returns a list of
    # per-note {outcome: prob} dicts, in the same order as the input.
    notes = eval_data[text_col].tolist()
    scored = get_postoperative_outcome_scores(
        model,
        notes,
        outcomes,
        max_length=max_length,
        device=device,
        hf_token=hf_token,
    )
    # Be robust to the single-note shortcut (a lone dict) even though a
    # dataframe of one row is unusual.
    if isinstance(scored, dict):
        scored = [scored]

    per_outcome = {}
    for outcome in outcomes:
        y_true = eval_data[outcome].astype(int).tolist()
        y_prob = [row[outcome] for row in scored]
        per_outcome[outcome] = _metrics_for_one_outcome(y_true, y_prob, threshold)

    table = pd.DataFrame.from_dict(per_outcome, orient="index")

    # Macro average over outcomes for the rate metrics; counts left blank.
    rate_cols = ["accuracy", "precision", "recall", "f1", "auroc", "auprc"]
    macro = table[rate_cols].mean(skipna=True)
    macro["support"] = ""
    macro["n_pos"] = ""
    table.loc["macro avg"] = macro

    return table
