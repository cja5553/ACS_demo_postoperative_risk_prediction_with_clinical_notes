"""
Multi-task fine-tuning and inference.

Exposes two functions:
    mtl_finetune(df, text_col, outcome_cols, ...)
        Fine-tune Bio+ClinicalBERT jointly on MLM + per-outcome binary
        classification, then save the model, tokenizer, and metadata.

    get_postoperative_outcome_scores(model_name, text, ...)
        Score a text scenario (or list of scenarios) against each outcome
        head of a fine-tuned MTL model.
"""

import json
import os
from typing import Dict, List, Optional, Union

import torch
from datasets import Dataset
from torch.utils.data import ConcatDataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)
import numpy as np
import pandas as pd
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
}


def _tokenize_and_prepare(batch, tokenizer, text_col, outcome_col, task_id, max_length):
    """Tokenize one batch and attach outcome labels + task ids."""
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
        "labels": encodings["input_ids"],  # MLM uses input_ids; collator masks them
        "task_ids": task_ids,
    }


def _prepare_data_per_task(df, tokenizer, text_col, outcome_col, task_id, max_length):
    """Build a tokenized torch-formatted Dataset for one (outcome, task_id) pair."""
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
    hf_token=None,
    max_length=512,
    lambda_constant=2,
    mlm_probability=0.15,
    val_fraction=1 / 8,
    weights=None,
    training_configs=None,
):
    """
    Fine-tune Bio+ClinicalBERT jointly on MLM + per-outcome binary classification.

    See README.md for full parameter documentation and examples. In short:
      df            : pandas.DataFrame containing `text_col` and all `outcome_cols`.
      text_col      : name of the free-text column.
      outcome_cols  : list of binary outcome column names (one head per outcome).
      output_dir    : where the fine-tuned model, tokenizer, and mtl_metadata.json
                      are saved. Also the HF Trainer output_dir.
      base_model    : HF model id (any BERT architecture works).
      max_length    : token sequence length.
      lambda_constant: weight on the auxiliary loss relative to MLM loss.
      mlm_probability: token masking probability for MLM.
      val_fraction  : fraction of `df` held out for validation.
      weights       : optional per-task pos_weight for BCEWithLogitsLoss.
      training_configs: any TrainingArguments kwargs to override the defaults.

    Returns
    -------
    str : output_dir
    """
    # --- input validation --------------------------------------------------
    if text_col not in df.columns:
        raise ValueError(f"text_col '{text_col}' not in dataframe columns")
    missing = [c for c in outcome_cols if c not in df.columns]
    if missing:
        raise ValueError(f"outcome_cols missing from dataframe: {missing}")
    if len(outcome_cols) == 0:
        raise ValueError("outcome_cols must contain at least one outcome")

    num_tasks = len(outcome_cols)

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
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_auth_token=hf_token)
    model = CustomBioClinicalBertForCombinedLearning.from_pretrained(
        base_model,
        output_hidden_states=True,
        num_tasks=num_tasks,
        lambda_constant=lambda_constant,
        weights=weights,
        use_auth_token=hf_token,
    )
    # --- stacked multi-task datasets --------------------------------------
    train_dataset = _stack_data(train_df, tokenizer, text_col, outcome_cols, max_length)
    val_dataset = _stack_data(val_df, tokenizer, text_col, outcome_cols, max_length)

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
        tokenizer=tokenizer,
    )
    trainer.train()

    # --- save final model, tokenizer, and metadata ------------------------
    os.makedirs(output_dir, exist_ok=True)
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


# =====================================================================
# Inference
# =====================================================================

def _load_metadata(model_name: str) -> dict:
    """Read mtl_metadata.json saved by mtl_finetune, or return {}."""
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
    hf_token: Optional[str] = None,
) -> Union[Dict[str, float], List[Dict[str, float]]]:
    """
    Score a text scenario (or list of scenarios) against each outcome head
    of a fine-tuned MTL model.

    See README.md for full documentation. In short:
      model_name  : path to a directory saved by mtl_finetune.
      text        : one scenario string, or a list of them.
      outcomes    : subset of outcomes to score (defaults to all trained ones).
      max_length  : token sequence length (defaults to training-time value).
      device      : "cuda", "cpu", or None to auto-detect.

    Returns
    -------
    dict[str, float] if `text` is a string, else list[dict[str, float]].
    """
    # --- load metadata ----------------------------------------------------
    meta = _load_metadata(model_name)
    trained_outcomes = meta.get("outcome_cols")
    num_tasks = meta.get("num_tasks")
    if max_length is None:
        max_length = meta.get("max_length", 512)

    # --- resolve which heads to score ------------------------------------
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

    # --- load model + tokenizer ------------------------------------------
    load_num_tasks = num_tasks if num_tasks is not None else len(outcomes)
    model = CustomBioClinicalBertForCombinedLearning.from_pretrained(
        model_name, num_tasks=load_num_tasks, use_auth_token=hf_token
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)

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

    # --- forward through BERT backbone only ------------------------------
    with torch.no_grad():
        hidden = model.bert(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        ).last_hidden_state  # [B, seq_len, hidden]

        results_per_example: List[Dict[str, float]] = [{} for _ in texts]
        for outcome_name, head_idx in zip(outcomes, head_indices):
            head = model.auxiliary[head_idx]
            logits = head(hidden)                       # [B, seq_len, 1]
            pooled = torch.mean(logits, dim=1)          # [B, 1]
            probs = torch.sigmoid(pooled).squeeze(-1)   # [B]
            for i, p in enumerate(probs.cpu().tolist()):
                results_per_example[i][outcome_name] = p

    return results_per_example[0] if was_single else results_per_example





def get_psuedo_data():
    rng = np.random.default_rng(0)

    # ---- building blocks ---------------------------------------------------------
    # Each procedure carries risk flags so we can build outcomes from them.
    # Flags: cardiac, thoracic, intracavitary (big abdominal/thoracic cases),
    # emergent (acute presentation).
    procedures = [
        ("appendectomy", "laparoscopic", "acute appendicitis",
         dict(cardiac=0, thoracic=0, intracavitary=1, emergent=1)),
        ("coronary artery bypass graft", "three-vessel", "severe CAD with LAD stenosis",
         dict(cardiac=1, thoracic=1, intracavitary=1, emergent=0)),
        ("total knee arthroplasty", "right", "end-stage osteoarthritis",
         dict(cardiac=0, thoracic=0, intracavitary=0, emergent=0)),
        ("cesarean section", "low transverse", "failure to progress",
         dict(cardiac=0, thoracic=0, intracavitary=1, emergent=0)),
        ("cholecystectomy", "laparoscopic", "symptomatic cholelithiasis",
         dict(cardiac=0, thoracic=0, intracavitary=1, emergent=0)),
        ("ORIF of femur", "open reduction", "comminuted femur fracture",
         dict(cardiac=0, thoracic=0, intracavitary=0, emergent=1)),
        ("carotid endarterectomy", "left", "70% carotid stenosis with TIA history",
         dict(cardiac=1, thoracic=0, intracavitary=0, emergent=0)),
        ("total hip arthroplasty", "left", "avascular necrosis",
         dict(cardiac=0, thoracic=0, intracavitary=0, emergent=0)),
        ("exploratory thoracotomy", "right", "undiagnosed pleural mass",
         dict(cardiac=0, thoracic=1, intracavitary=1, emergent=0)),
        ("partial colectomy", "sigmoid", "diverticulitis with abscess",
         dict(cardiac=0, thoracic=0, intracavitary=1, emergent=1)),
        ("unilateral mastectomy", "right", "invasive ductal carcinoma",
         dict(cardiac=0, thoracic=0, intracavitary=0, emergent=0)),
        ("craniotomy", "frontal", "resection of suspected glioma",
         dict(cardiac=0, thoracic=0, intracavitary=0, emergent=0)),
        ("inguinal hernia repair", "right, open", "reducible inguinal hernia",
         dict(cardiac=0, thoracic=0, intracavitary=0, emergent=0)),
        ("abdominal hysterectomy", "total", "symptomatic uterine fibroids",
         dict(cardiac=0, thoracic=0, intracavitary=1, emergent=0)),
        ("lumbar spinal fusion", "L4-L5", "spondylolisthesis with radiculopathy",
         dict(cardiac=0, thoracic=0, intracavitary=0, emergent=0)),
        ("aortic valve replacement", "bioprosthetic", "severe aortic stenosis",
         dict(cardiac=1, thoracic=1, intracavitary=1, emergent=0)),
        ("partial nephrectomy", "left", "renal mass suspicious for RCC",
         dict(cardiac=0, thoracic=0, intracavitary=1, emergent=0)),
        ("radical prostatectomy", "robotic", "localized prostate adenocarcinoma",
         dict(cardiac=0, thoracic=0, intracavitary=1, emergent=0)),
        ("Roux-en-Y gastric bypass", "laparoscopic", "morbid obesity, BMI 45",
         dict(cardiac=0, thoracic=0, intracavitary=1, emergent=0)),
        ("tonsillectomy and adenoidectomy", "pediatric", "recurrent tonsillitis",
         dict(cardiac=0, thoracic=0, intracavitary=0, emergent=0)),
    ]

    comorbidities_pool = [
        "hypertension", "type 2 diabetes mellitus", "hyperlipidemia",
        "chronic kidney disease stage 3", "COPD", "asthma",
        "atrial fibrillation", "coronary artery disease", "prior MI",
        "obstructive sleep apnea", "GERD", "hypothyroidism",
        "depression", "anxiety", "chronic low back pain",
        "osteoarthritis", "history of stroke",
    ]

    medications_pool = [
        "metoprolol", "lisinopril", "atorvastatin", "aspirin 81 mg",
        "metformin", "insulin glargine", "levothyroxine", "omeprazole",
        "albuterol inhaler", "warfarin", "apixaban", "furosemide",
        "sertraline", "gabapentin", "tramadol PRN", "acetaminophen PRN",
    ]

    allergies_pool = [
        "NKDA", "penicillin (rash)", "sulfa (hives)", "latex",
        "codeine (nausea)", "iodine contrast (mild reaction)",
    ]

    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def synthesize_row(rng):
        """Build a single synthetic note + its feature-driven outcomes."""
        proc, detail, indication, flags = procedures[rng.integers(0, len(procedures))]
        age = int(rng.integers(28, 86))
        sex = rng.choice(["male", "female"])
        asa = int(rng.choice([1, 2, 3, 4], p=[0.15, 0.45, 0.3, 0.10]))

        # 1-4 comorbidities; we'll build outcome features off this set.
        n_comorb = int(rng.integers(1, 5))
        comorb_list = rng.choice(comorbidities_pool, size=n_comorb, replace=False).tolist()
        has_diabetes = any("diabetes" in c for c in comorb_list)
        has_copd = any(c == "COPD" for c in comorb_list)

        # Smoking and obesity are added as separate sentences so they show up as
        # distinct lexical signals the model can learn (rather than hidden in
        # comorbidities which gets a long comma-joined list).
        is_smoker = bool(rng.binomial(1, 0.20))          # 20% current smoker
        is_obese = bool(rng.binomial(1, 0.30))           # 30% obese

        # 2-5 meds
        n_meds = int(rng.integers(2, 6))
        meds = ", ".join(rng.choice(medications_pool, size=n_meds, replace=False))
        allergies = rng.choice(allergies_pool)

        # ---- outcome logits (signal + small noise) ------------------------------
        # Outcome_1: high cardiac risk
        logit1 = -2.5
        if flags["cardiac"]:
            logit1 += 2.5
        if age >= 70:
            logit1 += 1.0
        if asa >= 3:
            logit1 += 1.0
        if is_smoker:
            logit1 += 0.7
        logit1 += rng.normal(0, 0.5)

        # Outcome_2: postoperative delirium
        logit2 = -2.0
        if age >= 75:
            logit2 += 2.0
        if asa >= 3:
            logit2 += 1.0
        if flags["intracavitary"]:
            logit2 += 0.6
        logit2 += rng.normal(0, 0.5)

        # Outcome_3: prolonged mechanical ventilation (rare by design)
        logit3 = -3.0
        if flags["thoracic"] or flags["cardiac"]:
            logit3 += 2.0
        if has_copd:
            logit3 += 1.5
        if asa >= 3:
            logit3 += 0.8
        if is_obese:
            logit3 += 0.5
        logit3 += rng.normal(0, 0.5)

        # Outcome_4: surgical site infection
        logit4 = -1.5
        if flags["emergent"]:
            logit4 += 1.5
        if has_diabetes:
            logit4 += 1.0
        if is_obese:
            logit4 += 0.8
        logit4 += rng.normal(0, 0.5)

        outcomes = [
            int(rng.binomial(1, _sigmoid(lg)))
            for lg in (logit1, logit2, logit3, logit4)
        ]

        # ---- compose the note ---------------------------------------------------
        sentences = [
            f"{age}-year-old {sex}, ASA {asa}, scheduled for {proc} ({detail}).",
            f"Indication: {indication}.",
            f"PMH: {', '.join(comorb_list)}.",
        ]
        if is_smoker:
            sentences.append("Social: current smoker, 1 pack per day.")
        if is_obese:
            sentences.append("BMI 34 (obese).")
        sentences.append(f"Home medications: {meds}.")
        sentences.append(f"Allergies: {allergies}.")
        sentences.append("Preop labs within acceptable limits. Consent obtained, plan to proceed.")

        return " ".join(sentences), outcomes

    n = 1000
    rows = [synthesize_row(rng) for _ in range(n)]
    texts = [r[0] for r in rows]
    outcomes = np.array([r[1] for r in rows])

    df_pseudo = pd.DataFrame({
        "text": texts,
        "Outcome_1": outcomes[:, 0],  # high cardiac risk
        "Outcome_2": outcomes[:, 1],  # postoperative delirium
        "Outcome_3": outcomes[:, 2],  # prolonged ventilation
        "Outcome_4": outcomes[:, 3],  # surgical site infection
    })
    return(df_pseudo)
 