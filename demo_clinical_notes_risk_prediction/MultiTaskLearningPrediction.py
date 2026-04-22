# importing libraries 
import torch
from torch.utils.data import ConcatDataset
from datasets import Dataset
# import packages (remember to import the local_transoformers package in your folder correctly!)
from local_transformers_bioClinical_BERT.src.transformers import TrainingArguments
from local_transformers_bioClinical_BERT.src.transformers.models.CustombioClinicalBERT import CustomBioClinicalBertForCombinedLearning # reads the local_transformers folder instead of the actual transformers package
from local_transformers_bioClinical_BERT.src.transformers.CustomComputeLoss import CustomTrainer # reads the local_transformers folder instead of the actual transformers package
import os
from local_transformers_bioClinical_BERT.src.transformers.data.data_collator import DataCollatorForLanguageModeling
from local_transformers_bioClinical_BERT.src.transformers.models.auto.tokenization_auto import AutoTokenizer
import json
from typing import Dict, List, Optional, Union
import torch


# Defaults applied when the user doesn't override them in training_configs.
# These match the paper's get_model() settings.
_DEFAULT_TRAINING_CONFIGS = {
    "num_train_epochs": 5,
    "per_device_train_batch_size": 24,
    "per_device_eval_batch_size": 24,
    "learning_rate": 1e-5,
    "warmup_ratio": 0.06,
    "weight_decay": 1e-3,
    "logging_steps": 1000,
    "save_strategy": "epoch",
    "seed": 42,
}


def _tokenize_and_prepare(batch, tokenizer, text_col, outcome_col, task_id, max_length):
    """Tokenize a batch and attach the outcome label + task id."""
    encodings = tokenizer(
        batch[text_col],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    additional_labels = torch.tensor(batch[outcome_col])
    task_ids = torch.tensor([task_id] * len(additional_labels))
    return {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "additional_labels": additional_labels,
        "labels": encodings["input_ids"],
        "task_ids": task_ids,
    }


def _prepare_data_per_task(df, tokenizer, text_col, outcome_col, task_id, max_length):
    """Build a tokenized torch-formatted Dataset for one (text, outcome, task_id) triple."""
    sub = df.dropna(subset=[outcome_col]).reset_index(drop=True)
    sub[outcome_col] = sub[outcome_col].astype(int)
    ds = Dataset.from_dict({text_col: list(sub[text_col]), outcome_col: list(sub[outcome_col])})
    ds = ds.map(
        lambda batch: _tokenize_and_prepare(
            batch, tokenizer, text_col, outcome_col, task_id, max_length
        ),
        batched=True,
        batch_size=128,
    )
    ds.set_format(
        "torch",
        columns=["input_ids", "attention_mask", "additional_labels", "labels", "task_ids"],
    )
    return ds


def _stack_data(df, tokenizer, text_col, outcome_cols, max_length):
    """Stack one tokenized Dataset per outcome into a single ConcatDataset."""
    parts = [
        _prepare_data_per_task(df, tokenizer, text_col, outcome_col, task_id, max_length)
        for task_id, outcome_col in enumerate(outcome_cols)
    ]
    return ConcatDataset(parts)


def mtl_finetune(
    df,
    text_col,
    outcome_cols,
    output_dir="mtl_finetuned",
    base_model="emilyalsentzer/Bio_ClinicalBERT",
    max_length=512,
    lambda_constant=2,
    mlm_probability=0.15,
    val_fraction=1 / 8,
    weights=None,
    training_configs=None,
):
    """
    Fine-tune a Bio+ClinicalBERT-style model on MLM + per-outcome binary classification.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain `text_col` and all of `outcome_cols`.
    text_col : str
        Name of the free-text column.
    outcome_cols : list[str]
        Names of binary outcome columns. One auxiliary head is trained per outcome.
    output_dir : str
        Where the fine-tuned model + tokenizer will be saved. Also used as the
        HuggingFace Trainer output_dir for checkpoints/logs. If `training_configs`
        contains an `output_dir` key, it is overridden by this argument.
    base_model : str
        HuggingFace model id to start from.
    max_length : int
        Sequence length for tokenization.
    lambda_constant : float
        Weight on the auxiliary (per-outcome BCE) loss relative to MLM loss.
    mlm_probability : float
        Probability of masking a token for the MLM objective.
    val_fraction : float
        Fraction of `df` to hold out as validation during fine-tuning.
    weights : list | None
        Optional per-task pos_weight for BCEWithLogitsLoss. None disables class weighting.
    training_configs : dict | None
        Any keyword arguments accepted by `transformers.TrainingArguments`.
        User-provided values override the defaults below:

            num_train_epochs=6
            per_device_train_batch_size=24
            per_device_eval_batch_size=24
            learning_rate=1e-5
            warmup_steps=1500
            weight_decay=1e-3
            logging_steps=1000
            save_strategy="epoch"
            seed=42

        Use this to pass any supported TrainingArguments field, e.g.
        `fp16=True`, `gradient_accumulation_steps=4`, `evaluation_strategy="epoch"`,
        `gradient_checkpointing=True`, etc.

    Returns
    -------
    str
        The `output_dir` where the fine-tuned model and tokenizer were saved.
    """
    # --- basic input validation --------------------------------------------
    if text_col not in df.columns:
        raise ValueError(f"text_col '{text_col}' not in dataframe columns")
    missing = [c for c in outcome_cols if c not in df.columns]
    if missing:
        raise ValueError(f"outcome_cols missing from dataframe: {missing}")
    if len(outcome_cols) == 0:
        raise ValueError("outcome_cols must contain at least one outcome")

    num_tasks = len(outcome_cols)

    # --- merge training configs: defaults <- user overrides ----------------
    cfg = dict(_DEFAULT_TRAINING_CONFIGS)
    if training_configs:
        cfg.update(training_configs)
    # The top-level `output_dir` arg always wins, so users don't have to
    # duplicate it inside training_configs.
    cfg["output_dir"] = output_dir
    cfg.setdefault("logging_dir", os.path.join(output_dir, "logs"))

    # --- train/val split (use the training seed so split is reproducible) --
    split_seed = cfg.get("seed", 42)
    train_df = df.sample(frac=1 - val_fraction, random_state=split_seed)
    val_df = df.drop(train_df.index)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    # --- model + tokenizer --------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = CustomBioClinicalBertForCombinedLearning.from_pretrained(
        base_model,
        output_hidden_states=True,
        num_tasks=num_tasks,
        lambda_constant=lambda_constant,
        weights=weights,
    )

    # --- stacked multi-task datasets ---------------------------------------
    train_dataset = _stack_data(train_df, tokenizer, text_col, outcome_cols, max_length)
    val_dataset = _stack_data(val_df, tokenizer, text_col, outcome_cols, max_length)

    # --- training ----------------------------------------------------------
    # Any TrainingArguments field is supported via training_configs; if the
    # user passes an unknown key, TrainingArguments raises TypeError.
    try:
        training_args = TrainingArguments(**cfg)
    except TypeError as e:
        raise TypeError(
            f"Invalid key in training_configs. TrainingArguments rejected: {e}"
        ) from e

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=mlm_probability
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()

    # --- save final model and tokenizer ------------------------------------
    trainer.model.save_pretrained(output_dir)
    trainer.tokenizer.save_pretrained(output_dir)

    metadata = {
        "outcome_cols": list(outcome_cols),
        "text_col": text_col,
        "max_length": max_length,
        "base_model": base_model,
        "lambda_constant": lambda_constant,
        "num_tasks": num_tasks,
    }
    with open(os.path.join(output_dir, "mtl_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return output_dir






def _load_metadata(model_name: str) -> dict:
    """Read mtl_metadata.json saved by mtl_finetune, or fall back to sane defaults."""
    meta_path = os.path.join(model_name, "mtl_metadata.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {}


def get_postoperative_outcome_scores(
    model_name: str,
    text: Union[str, List[str]],
    outcomes: Optional[List[str]] = None,
    max_length: Optional[int] = None,
    device: Optional[str] = None,
) -> Union[Dict[str, float], List[Dict[str, float]]]:
    """
    Score a text scenario (or a list of scenarios) against each outcome head
    of a fine-tuned MTL model.

    Parameters
    ----------
    model_name : str
        Path to the directory saved by `mtl_finetune` (contains the model,
        tokenizer, and `mtl_metadata.json`).
    text : str or list[str]
        A single scenario string, or a list of them.
    outcomes : list[str] | None
        Which outcomes to score. Defaults to all outcomes the model was trained on.
        Names must match those passed to `mtl_finetune`.
    max_length : int | None
        Tokenization max length. Defaults to the value used at fine-tuning time
        (recovered from metadata), else 512.
    device : str | None
        "cuda", "cpu", or None (auto-detect).

    Returns
    -------
    dict[str, float] if `text` was a single string, else list[dict[str, float]].
        Each dict maps outcome name -> probability (sigmoid of the head's
        sequence-mean logit).

    Notes
    -----
    This function uses the auxiliary heads trained jointly during MTL
    fine-tuning. The pooling matches the training-time forward pass exactly
    (mean over token logits), so scores are consistent with the signal the
    model optimized for.
    """
    # --- load metadata ------------------------------------------------------
    meta = _load_metadata(model_name)
    trained_outcomes = meta.get("outcome_cols")
    num_tasks = meta.get("num_tasks")
    if max_length is None:
        max_length = meta.get("max_length", 512)

    # --- resolve which heads to score --------------------------------------
    # If metadata is present, we can validate names. If not, we fall back to
    # positional names (Outcome_0, Outcome_1, ...) when outcomes is None.
    if outcomes is None:
        if trained_outcomes is not None:
            outcomes = list(trained_outcomes)
        elif num_tasks is not None:
            outcomes = [f"Outcome_{i}" for i in range(num_tasks)]
        else:
            raise ValueError(
                "Cannot infer outcome names: no mtl_metadata.json in "
                f"'{model_name}' and `outcomes` was not provided."
            )

    # Map requested outcome names -> head indices. If metadata lists
    # the training-time names, use that ordering; otherwise assume the
    # caller's `outcomes` list is already in head-index order.
    if trained_outcomes is not None:
        unknown = [o for o in outcomes if o not in trained_outcomes]
        if unknown:
            raise ValueError(
                f"Unknown outcome(s) {unknown}. "
                f"Model was trained on: {trained_outcomes}"
            )
        head_indices = [trained_outcomes.index(o) for o in outcomes]
    else:
        head_indices = list(range(len(outcomes)))

    # --- load model + tokenizer --------------------------------------------
    # num_tasks on from_pretrained must match what the checkpoint has; we use
    # metadata if available, else fall back to len(outcomes).
    load_num_tasks = num_tasks if num_tasks is not None else len(outcomes)
    model = CustomBioClinicalBertForCombinedLearning.from_pretrained(
        model_name, num_tasks=load_num_tasks
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # --- tokenize ----------------------------------------------------------
    was_single = isinstance(text, str)
    texts = [text] if was_single else list(text)
    inputs = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    # --- forward through BERT backbone only --------------------------------
    # We avoid the full model.forward because it expects task_ids and returns
    # MLM logits / losses rather than per-task logits.
    with torch.no_grad():
        hidden = model.bert(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        ).last_hidden_state  # [B, seq_len, hidden]

        results_per_example: List[Dict[str, float]] = [{} for _ in texts]
        for outcome_name, head_idx in zip(outcomes, head_indices):
            head = model.auxiliary[head_idx]
            logits = head(hidden)                    # [B, seq_len, 1]
            pooled = torch.mean(logits, dim=1)       # [B, 1]  (matches training-time pooling)
            probs = torch.sigmoid(pooled).squeeze(-1)  # [B]
            for i, p in enumerate(probs.cpu().tolist()):
                results_per_example[i][outcome_name] = p

    return results_per_example[0] if was_single else results_per_example