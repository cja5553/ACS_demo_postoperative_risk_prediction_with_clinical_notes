"""
Joint single-outcome fine-tuning and inference.

Exposes three functions:
    joint_finetune(df, text_col, outcome_col, ...)
        Jointly fine-tune Bio+ClinicalBERT on MLM + a single binary
        classification head, then save the model, tokenizer, and metadata.

    get_outcome_score(model_name, text, ...)
        Score a text scenario (or list of scenarios) using a single-outcome
        joint-finetuned model's auxiliary head.

"""

import json
import os
from typing import List, Optional, Union
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)

from .model import CustomBioClinicalBertForCombinedLearning
from .trainer import CustomTrainer


# =====================================================================
# Fine-tuning
# =====================================================================

# Defaults applied when the user doesn't override them in training_configs.
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
    "report_to": "none",
}


def _tokenize_and_prepare(batch, tokenizer, text_col, outcome_col, max_length):
    """Tokenize one batch and attach the single outcome's labels.

    All examples are stamped with `task_id=0` because there is only one head.
    """
    encodings = tokenizer(
        batch[text_col],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    additional_labels = torch.tensor(batch[outcome_col])
    task_ids = torch.tensor([0] * len(additional_labels))
    return {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "additional_labels": additional_labels,
        "labels": encodings["input_ids"],  # MLM uses input_ids; collator masks them
        "task_ids": task_ids,
    }


def _prepare_data(df, tokenizer, text_col, outcome_col, max_length):
    """Build a tokenized torch-formatted Dataset for one outcome."""
    sub = df.dropna(subset=[outcome_col]).reset_index(drop=True)
    sub[outcome_col] = sub[outcome_col].astype(int)
    ds = Dataset.from_dict(
        {text_col: list(sub[text_col]), outcome_col: list(sub[outcome_col])}
    )
    ds = ds.map(
        lambda batch: _tokenize_and_prepare(
            batch, tokenizer, text_col, outcome_col, max_length
        ),
        batched=True,
        batch_size=128,
    )
    ds.set_format(
        "torch",
        columns=["input_ids", "attention_mask", "additional_labels", "labels", "task_ids"],
    )
    return ds


def joint_finetune(
    df,
    text_col,
    outcome_col,
    output_dir="joint_finetuned",
    base_model="emilyalsentzer/Bio_ClinicalBERT",
    hf_token=None,
    max_length=512,
    lambda_constant=2,
    mlm_probability=0.15,
    val_fraction=1 / 8,
    weight=None,
    training_configs=None,
):
    """
    Jointly fine-tune Bio+ClinicalBERT on MLM + a single binary classification
    head for one postoperative outcome.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain `text_col` and `outcome_col`.
    text_col : str
        Name of the free-text column.
    outcome_col : str
        Name of the binary outcome column to train on. Rows with NaN in this
        column are dropped before training.
    output_dir : str
        Where the fine-tuned model, tokenizer, and joint_metadata.json are saved.
        Also the HF Trainer output_dir.
    base_model : str
        HF model id (any BERT architecture works).
    hf_token : str or None
        Optional HuggingFace token for gated/private base models. If None,
        uses cached CLI login (huggingface-cli login) when present.
    max_length : int
        Token sequence length.
    lambda_constant : float
        Weight on the auxiliary BCE loss relative to MLM loss. Total loss is
        `MLM + lambda_constant * BCE`.
    mlm_probability : float
        Token masking probability for MLM.
    val_fraction : float
        Fraction of `df` held out for validation.
    weight : torch.Tensor or None
        Optional pos_weight for BCEWithLogitsLoss to handle class imbalance.
        Useful for rare outcomes (e.g., torch.tensor([20.0]) for ~5% positive
        prevalence).
    training_configs : dict or None
        Any keyword arguments accepted by `transformers.TrainingArguments`.
        User-provided values override the defaults.

    Returns
    -------
    str
        The output_dir path.
    """
    # --- input validation --------------------------------------------------
    if text_col not in df.columns:
        raise ValueError(f"text_col '{text_col}' not in dataframe columns")
    if outcome_col not in df.columns:
        raise ValueError(f"outcome_col '{outcome_col}' not in dataframe columns")

    # --- merge training configs: defaults <- user overrides ---------------
    cfg = dict(_DEFAULT_TRAINING_CONFIGS)
    if training_configs:
        cfg.update(training_configs)
    cfg["output_dir"] = output_dir
    cfg.setdefault("logging_dir", os.path.join(output_dir, "logs"))

    # --- train/val split --------------------------------------------------
    split_seed = cfg.get("seed", 42)
    train_df = df.sample(frac=1 - val_fraction, random_state=split_seed)
    val_df = df.drop(train_df.index)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    # --- model + tokenizer -----------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token)
    weights_list = [weight] if weight is not None else None
    model = CustomBioClinicalBertForCombinedLearning.from_pretrained(
        base_model,
        output_hidden_states=True,
        num_tasks=1,                       # ← key difference from MTL: single head
        lambda_constant=lambda_constant,
        weights=weights_list,
        token=hf_token,
    )

    # --- tokenized datasets (no stacking; single outcome) -----------------
    train_dataset = _prepare_data(train_df, tokenizer, text_col, outcome_col, max_length)
    val_dataset = _prepare_data(val_df, tokenizer, text_col, outcome_col, max_length)

    # Back-compat: HF renamed evaluation_strategy -> eval_strategy in 4.41
    if "evaluation_strategy" in cfg and "eval_strategy" not in cfg:
        cfg["eval_strategy"] = cfg.pop("evaluation_strategy")

    # --- training ---------------------------------------------------------
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
        processing_class=tokenizer,
    )
    trainer.train()

    # --- save final model, tokenizer, and metadata ------------------------
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    trainer.tokenizer.save_pretrained(output_dir)

    metadata = {
        "outcome_col": outcome_col,
        "text_col": text_col,
        "max_length": max_length,
        "base_model": base_model,
        "lambda_constant": lambda_constant,
        "num_tasks": 1,
        "workflow": "joint_single_outcome",
    }
    with open(os.path.join(output_dir, "joint_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return output_dir


# =====================================================================
# Inference
# =====================================================================

def _load_metadata(model_name: str) -> dict:
    """Read joint_metadata.json saved by joint_finetune, or return {}."""
    meta_path = os.path.join(model_name, "joint_metadata.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {}


def get_outcome_score(
    model_name: str,
    text: Union[str, List[str]],
    max_length: Optional[int] = None,
    device: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> Union[float, List[float]]:
    """
    Score a text scenario (or list of scenarios) using the single auxiliary
    head of a joint-finetuned model.

    Parameters
    ----------
    model_name : str
        Path to a directory saved by joint_finetune.
    text : str or list of str
        One scenario string, or a list of them.
    max_length : int or None
        Token sequence length. Defaults to the value used during fine-tuning,
        recovered from joint_metadata.json, otherwise 512.
    device : str or None
        "cuda", "cpu", or None to auto-detect.
    hf_token : str or None
        Optional HuggingFace token for gated/private models.

    Returns
    -------
    float
        When `text` is a string: the predicted probability for the outcome.
    list of float
        When `text` is a list: one probability per input, in the same order.
    """
    # --- load metadata ----------------------------------------------------
    meta = _load_metadata(model_name)
    if max_length is None:
        max_length = meta.get("max_length", 512)

    # --- load model + tokenizer ------------------------------------------
    model = CustomBioClinicalBertForCombinedLearning.from_pretrained(
        model_name, num_tasks=1, token=hf_token
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # --- tokenize --------------------------------------------------------
    was_single = isinstance(text, str)
    texts = [text] if was_single else list(text)
    inputs = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    # --- forward through BERT backbone, then the single head -------------
    with torch.no_grad():
        hidden = model.bert(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        ).last_hidden_state                       # [B, seq_len, hidden]

        head = model.auxiliary[0]                 # only one head
        logits = head(hidden)                     # [B, seq_len, 1]
        pooled = torch.mean(logits, dim=1)        # [B, 1]
        probs = torch.sigmoid(pooled).squeeze(-1) # [B]
        results = probs.cpu().tolist()

    return results[0] if was_single else results

