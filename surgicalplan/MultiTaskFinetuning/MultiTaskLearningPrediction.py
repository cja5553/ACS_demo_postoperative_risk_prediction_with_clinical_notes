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
    "report_to": "none",   # ← add this
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
    tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token)
    model = CustomBioClinicalBertForCombinedLearning.from_pretrained(
        base_model,
        output_hidden_states=True,
        num_tasks=num_tasks,
        lambda_constant=lambda_constant,
        weights=weights,
        token=hf_token,
    )
    # --- stacked multi-task datasets --------------------------------------
    train_dataset = _stack_data(train_df, tokenizer, text_col, outcome_cols, max_length)
    val_dataset = _stack_data(val_df, tokenizer, text_col, outcome_cols, max_length)
    
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
        processing_class=tokenizer,   # was tokenizer=tokenizer
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
        model_name, num_tasks=load_num_tasks, token=hf_token
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




def get_pseudo_data() -> pd.DataFrame:
    """
    Return a fixed 500-row, manually authored synthetic tutorial cohort.

    Each concise preoperative note and its DVT, Pneumonia, AKI, and
    Delirium labels was written and paired as an individual clinical
    scenario. No random-number generator, outcome formula, or row-level
    text template is used when this function is called.

    These are synthetic educational cases, not real patient records and
    not a clinically validated cohort.
    """
    rows = [{'clinical_note': '79-year-old woman, ASA 3, for urgent hemiarthroplasty after a displaced femoral-neck fracture. She has been '
                   'largely bedbound for two days. History includes mild dementia, hypertension, and CKD stage 3a; creatinine is '
                   '1.2 mg/dL at baseline. Lungs are clear. Her daughter reports prior confusion after hospitalization.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': '42-year-old woman, ASA 1, scheduled for elective laparoscopic cholecystectomy for biliary colic. She has no '
                   'cardiopulmonary, renal, or neurologic disease, takes no daily medication, and exercises regularly. BMI is '
                   '24, room-air saturation 99%, creatinine 0.7 mg/dL, and she is fully oriented and independent.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '74-year-old man, ASA 4, undergoing three-vessel coronary artery bypass grafting. History includes ischemic '
                   'cardiomyopathy with EF 30%, insulin-treated diabetes, and CKD stage 3b with baseline creatinine 1.9 mg/dL. '
                   'He is independent in basic activities and cognitively intact. Lungs are clear without recent infection.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': '67-year-old woman, ASA 3, planned open esophagectomy for distal esophageal cancer. She has moderate COPD, '
                   'continues to smoke one pack daily, and becomes dyspneic after one flight of stairs. Room-air saturation is '
                   '92% with diminished breath sounds. Creatinine is 0.8 mg/dL and cognition is normal.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '58-year-old man, ASA 3, for pancreaticoduodenectomy for pancreatic adenocarcinoma. He had a proximal leg DVT '
                   'two years ago and takes apixaban, now held for surgery. He has lost 9 kg and is less active but remains '
                   'independent. Lungs are clear, creatinine 0.9 mg/dL, cognition intact.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '84-year-old woman, ASA 4, requires emergency open colectomy for perforated diverticulitis with sepsis. She '
                   'has moderate dementia, CKD stage 3b, poor oral intake, and a new 2-L oxygen requirement. Creatinine is 1.8 '
                   'mg/dL from 1.4. She is inattentive and relies on family for daily care.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': '36-year-old woman, ASA 1, scheduled for outpatient hemithyroidectomy for a benign thyroid nodule. She has no '
                   'chronic disease, normal exercise tolerance, and no prior anesthesia complications. Room-air saturation is '
                   '99%, creatinine 0.6 mg/dL, and neurologic examination and cognition are normal.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '69-year-old man, ASA 3, for multilevel lumbar decompression and fusion. BMI is 39, walking is limited to one '
                   'block by back pain, and he had a pulmonary embolism four years ago after surgery. Chronic apixaban has been '
                   'stopped. No COPD, creatinine 1.0 mg/dL, cognition intact.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '78-year-old woman, ASA 3, scheduled for elective total knee arthroplasty. She has mild cognitive impairment, '
                   'hearing loss, hypertension, and CKD stage 3a with creatinine 1.2 mg/dL. She lives alone but needs help '
                   'organizing medications. Lungs are clear and she walks daily with a cane.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': '63-year-old man, ASA 3, planned partial nephrectomy for a renal mass in his only functioning kidney. He has '
                   'diabetes and CKD stage 3b; baseline creatinine is 1.8 mg/dL. Blood pressure is controlled, lungs are clear, '
                   'and he remains active and fully oriented.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': '61-year-old woman, ASA 3, for right upper lobectomy by VATS for lung cancer. She has severe COPD, a '
                   '45-pack-year smoking history, and DLCO 44% predicted. She uses tiotropium and albuterol; room-air saturation '
                   'is 91%. Creatinine is 0.8 mg/dL and cognition is intact.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '82-year-old man, ASA 3, scheduled for transurethral resection of the prostate for urinary retention. He has '
                   'moderate Alzheimer disease, severe hearing impairment, and nightly lorazepam use. His daughter provides most '
                   'history. He is oriented only to person and place. Creatinine 1.0 mg/dL; lungs clear.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': '65-year-old woman, ASA 3, for open abdominal hysterectomy and staging for endometrial cancer. BMI is 43, '
                   'mobility is limited by knee pain, and she had a pulmonary embolism five years ago. Rivaroxaban has been '
                   'held. Renal function is normal; no lung disease or cognitive impairment.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '71-year-old man, ASA 4, requires emergency small-bowel resection for strangulated obstruction. He has had '
                   'vomiting and poor intake for three days. Blood pressure is 92/58 after initial fluids, lactate is elevated, '
                   'and creatinine is 1.2 mg/dL from 0.9. Normally independent and cognitively intact.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': '33-year-old woman, ASA 2, scheduled for cesarean delivery because of placenta previa. BMI is 35, but she has '
                   'remained mobile and has no personal or family history of thrombosis. Pregnancy has otherwise been '
                   'uncomplicated. Lungs are clear, creatinine 0.5 mg/dL, and she is alert.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '52-year-old man, ASA 1, for outpatient arthroscopic rotator-cuff repair. He jogs several times weekly, has '
                   'no chronic medical conditions, and takes only occasional acetaminophen. BMI is 25, room-air saturation 98%, '
                   'creatinine 0.9 mg/dL, with no history of confusion or thromboembolism.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '80-year-old woman, ASA 4, for surgical aortic valve replacement. She has severe aortic stenosis, CKD stage '
                   '3b with creatinine 1.7 mg/dL, diabetes, and frailty with slow gait. She is cognitively intact but needs help '
                   'with shopping. Lungs are clear without recent respiratory infection.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': '69-year-old man, ASA 2, scheduled for robotic prostatectomy for localized prostate cancer. Hypertension is '
                   'controlled with amlodipine. He walks four miles daily, has no lung or kidney disease, and has never had VTE. '
                   'Creatinine is 0.8 mg/dL, oxygen saturation 98%, cognition normal.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '48-year-old woman, ASA 3, for laparoscopic sleeve gastrectomy. BMI is 52, and she has severe obstructive '
                   'sleep apnea but uses CPAP consistently. She is independently mobile, has no COPD or prior VTE, recently '
                   'stopped smoking, creatinine 0.7 mg/dL, cognition intact.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '76-year-old man, ASA 3, for craniotomy and resection of a frontal meningioma. History includes remote '
                   'ischemic stroke, mild cognitive impairment, and recent dexamethasone and levetiracetam. He is independent '
                   'but repeats questions during interview. Lungs clear, creatinine 0.9 mg/dL, no VTE history.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': '57-year-old man, ASA 3, planned right hepatectomy for hepatocellular carcinoma. He has NASH cirrhosis, '
                   'albumin 2.9 g/dL, mild ascites, and baseline creatinine 1.1 mg/dL. He is oriented and ambulates '
                   'independently. No chronic lung disease or prior thromboembolism.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': '73-year-old woman, ASA 3, for open fixation of a tibial-plateau fracture. She has been non-weight-bearing in '
                   'a knee immobilizer for ten days. History includes a prior postoperative DVT; she is not on chronic '
                   'anticoagulation. Lungs are clear, creatinine 0.8 mg/dL, cognition normal.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '66-year-old man, ASA 3, for pancreaticoduodenectomy. He has moderate COPD and was treated for '
                   'community-acquired pneumonia six weeks ago; symptoms resolved, but exercise tolerance remains reduced. '
                   'Room-air saturation is 93% and cough is weak. Creatinine 0.9 mg/dL; cognition intact.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '81-year-old woman, ASA 3, scheduled for revision total hip arthroplasty. She is frail, uses a walker, and '
                   'had marked confusion for three days after her previous hip operation. CKD stage 3a is stable with creatinine '
                   '1.2 mg/dL. Lungs are clear; no previous VTE.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': '45-year-old woman, ASA 2, for unilateral mastectomy and sentinel-node biopsy for early breast cancer. She '
                   'has controlled hypothyroidism and otherwise good health. She works full time, exercises regularly, and has '
                   'no prior VTE, lung disease, kidney disease, or cognitive symptoms.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '72-year-old man, ASA 4, undergoing open abdominal aortic aneurysm repair. He has peripheral arterial '
                   'disease, hypertension, CKD stage 3a, and baseline creatinine 1.4 mg/dL. Functional capacity is poor because '
                   'of claudication. He is oriented, without lung disease or prior postoperative confusion.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': '70-year-old woman, ASA 3, planned right hemicolectomy for colon cancer. She completed chemotherapy one month '
                   'ago, has a tunneled central venous catheter, and is less active because of fatigue. She had a calf DVT '
                   'during treatment. Apixaban is held. Lungs clear; creatinine 0.8 mg/dL.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '64-year-old man, ASA 3, for total laryngectomy after chemoradiation for recurrent laryngeal cancer. He has '
                   'thick secretions, chronic aspiration with liquids, and a 50-pack-year smoking history. Room-air saturation '
                   'is 94% and cough is weak. Renal function and cognition are normal.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '86-year-old woman, ASA 4, for emergency repair of an incarcerated ventral hernia. She has severe frailty, '
                   'moderate dementia, dehydration, and CKD stage 3b. Creatinine is 2.0 mg/dL from a baseline of 1.5. She is '
                   'disoriented to date and needs assistance with all instrumental activities.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': '39-year-old man, ASA 2, scheduled for elective inguinal hernia repair. Asthma is mild and controlled with '
                   'rare albuterol use. He cycles to work, has normal kidney function, no prior thrombosis, and no cognitive or '
                   'psychiatric history. Room-air saturation is 99%.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '77-year-old man, ASA 3, for thoracic endovascular aortic repair. He has CKD stage 3b, diabetes, and '
                   'extensive atherosclerosis. Baseline creatinine is 1.8 mg/dL. He is independent and cognitively normal, with '
                   'no chronic pulmonary disease and no previous VTE.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': '68-year-old woman, ASA 3, for ovarian cancer debulking. She has malignant ascites, albumin 2.7 g/dL, and has '
                   'spent most of the last week in bed because of abdominal distention. A sister had unprovoked DVT. Lungs are '
                   'clear; creatinine 0.7 mg/dL; cognition intact.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '75-year-old man, ASA 3, planned left pneumonectomy for central lung cancer. He has moderate COPD, ongoing '
                   'tobacco use, and room-air saturation of 91%. Pulmonary function testing shows FEV1 48% predicted. Creatinine '
                   'is 1.0 mg/dL; he is oriented and lives independently.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '83-year-old woman, ASA 3, for elective sigmoid colectomy for recurrent diverticulitis. She has mild '
                   'cognitive impairment, macular degeneration, and chronic zolpidem use. Her son assists with transportation '
                   'and medications. Lungs clear, creatinine 0.9 mg/dL, and she is currently oriented.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': '55-year-old man, ASA 3, undergoing radical cystectomy with ileal conduit. He has diabetic nephropathy with '
                   'baseline creatinine 1.6 mg/dL and takes lisinopril and metformin. He is well hydrated, active, and '
                   'cognitively intact. No lung disease or history of VTE.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': '62-year-old woman, ASA 2, scheduled for laparoscopic fundoplication for refractory reflux. She has '
                   'controlled hypertension and no pulmonary, renal, thrombotic, or cognitive disease. She walks two miles '
                   'daily, room-air saturation is 98%, and creatinine is 0.7 mg/dL.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '70-year-old man, ASA 3, for revision knee arthroplasty after prosthetic loosening. He has obesity, chronic '
                   'venous insufficiency, and a previous unprovoked DVT. Warfarin is being bridged perioperatively. Mobility is '
                   'limited to household distances. Lungs clear, renal function normal, cognition intact.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '59-year-old woman, ASA 3, for open adrenalectomy for a large pheochromocytoma. She has labile hypertension '
                   'and CKD stage 3a with creatinine 1.3 mg/dL. Volume expansion has been difficult because of nausea. She is '
                   'cognitively intact with no lung disease or VTE history.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': '88-year-old man, ASA 4, for urgent debridement of an infected hip prosthesis. He has dementia, chronic '
                   'anemia, CKD stage 3a, and resides in assisted living. He is febrile, intermittently inattentive, and has '
                   'reduced oral intake. Creatinine is 1.5 mg/dL from 1.1.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': '50-year-old woman, ASA 1, undergoing laparoscopic myomectomy for symptomatic fibroids. She has no chronic '
                   'illness, normal exercise tolerance, and no prior thrombosis. Hemoglobin is 11.8 g/dL, creatinine 0.6 mg/dL, '
                   'lungs clear, cognition normal.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '73-year-old man, ASA 3, for open repair of a ruptured Achilles tendon three weeks after injury. He has '
                   'remained non-weight-bearing in a cast and previously developed DVT after knee replacement. Apixaban is not '
                   'used chronically. Lungs clear, creatinine 0.9 mg/dL, cognition intact.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '65-year-old woman, ASA 3, for VATS decortication of a persistent empyema. She recently completed antibiotics '
                   'but still has productive cough, pleuritic pain, and room-air saturation of 92%. Diabetes is controlled; '
                   'creatinine 0.8 mg/dL. She is oriented and independently mobile.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '81-year-old man, ASA 4, undergoing emergency laparotomy for ischemic bowel. He has atrial fibrillation, CKD '
                   'stage 3b, heart failure, and baseline mild cognitive impairment. He arrived hypotensive and confused; '
                   'creatinine is 2.1 mg/dL from 1.6. Oxygen saturation is 95% on 2 L.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': '46-year-old man, ASA 2, for elective cervical discectomy. He has well-controlled hypertension, normal renal '
                   'function, no pulmonary disease, and no previous thromboembolism. He works full time, exercises weekly, and '
                   'is alert with no cognitive concerns.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '72-year-old woman, ASA 3, for open gastrectomy for gastric cancer. She has severe weight loss, albumin 2.6 '
                   'g/dL, COPD, and a recent aspiration event during vomiting. Room-air saturation is 93%, cough is weak, '
                   'creatinine 0.8 mg/dL, cognition intact.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '79-year-old man, ASA 3, for carotid endarterectomy after symptomatic stenosis. He has CKD stage 3a, prior '
                   'stroke without major deficit, and mild vascular cognitive impairment. Creatinine is 1.4 mg/dL. He is '
                   'oriented but slow to answer; lungs are clear.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': '60-year-old woman, ASA 3, scheduled for nephroureterectomy for upper-tract urothelial cancer. Baseline '
                   'creatinine is 1.5 mg/dL with contralateral renal scarring. She has diabetes and hypertension but remains '
                   'active and cognitively intact. No lung disease or prior VTE.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': '67-year-old man, ASA 3, for open reduction and internal fixation of an acetabular fracture after a '
                   'motor-vehicle collision. He has been on bed rest for six days and has a history of prostate cancer. Duplex '
                   'was negative on admission. Lungs clear, renal function normal, cognition intact.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '71-year-old woman, ASA 3, undergoing right lower lobectomy. She has bronchiectasis with daily sputum '
                   'production, recurrent respiratory infections, and FEV1 55% predicted. Room-air saturation is 93%. Creatinine '
                   'is 0.7 mg/dL, she is oriented, and she has no VTE history.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '85-year-old woman, ASA 3, scheduled for operative fixation of a distal radius fracture. She lives in memory '
                   'care with moderate dementia and is taking quetiapine at night. She is calm but disoriented to place. Lungs '
                   'clear, creatinine 0.9 mg/dL, and she ambulates with supervision.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': '56-year-old man, ASA 2, for laparoscopic right colectomy for a large benign polyp. He has controlled '
                   'hypertension, walks three miles daily, and has no pulmonary, renal, thrombotic, or cognitive history. '
                   'Creatinine is 0.9 mg/dL and room-air saturation is 98%.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '76-year-old woman, ASA 4, undergoing redo coronary bypass surgery. She has CKD stage 4 with creatinine 2.3 '
                   'mg/dL, diabetes, anemia, and heart failure with EF 25%. She is frail but oriented. Mild bibasilar crackles '
                   'are present without fever or productive cough.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': '62-year-old man, ASA 3, for radical prostatectomy after recent diagnosis of high-risk prostate cancer. He '
                   'had an unprovoked proximal DVT eighteen months ago and remains on apixaban, now interrupted. BMI is 36 and '
                   'activity has declined. Lungs clear; creatinine 0.9 mg/dL.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '69-year-old woman, ASA 3, scheduled for open hiatal hernia repair. She has severe reflux with nocturnal '
                   'regurgitation, two recent aspiration episodes, and moderate restrictive lung disease. Room-air saturation is '
                   '93% and cough is weak. Kidney function and cognition are normal.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '89-year-old man, ASA 4, for urgent fixation of an intertrochanteric hip fracture. He has advanced Parkinson '
                   'disease, moderate dementia, dysphagia, and CKD stage 3b. He is bedbound since the fall, intermittently '
                   'drowsy, and requires help with all activities. Creatinine is 1.7 mg/dL.',
  'DVT': 1,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': '44-year-old woman, ASA 2, for laparoscopic salpingectomy for a benign adnexal mass. She has mild asthma '
                   'without recent symptoms and no other illness. She is fully active, takes no anticoagulant or nephrotoxic '
                   'medication, and has normal renal function and cognition.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '73-year-old man, ASA 3, planned open prostatectomy for severe benign prostatic enlargement. He has CKD stage '
                   '3b, chronic urinary obstruction, and baseline creatinine 1.9 mg/dL. He is euvolemic, independent, and '
                   'cognitively intact. No lung disease or prior thrombosis.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': '68-year-old man, ASA 3, for total knee arthroplasty. He has prior DVT after ankle surgery, obesity, and '
                   'chronic leg edema. Mobility has been poor for several months. Warfarin was stopped five days ago. Lungs are '
                   'clear, creatinine 1.0 mg/dL, cognition normal.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '77-year-old woman, ASA 3, for elective open colectomy for a large cecal polyp. She has mild Alzheimer '
                   'disease, chronic hearing impairment, and uses diphenhydramine nightly. She is oriented to person and place '
                   'but not date. Lungs and renal function are normal.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': '60-year-old man, ASA 3, undergoing VATS wedge resection for a suspicious pulmonary nodule. He has severe '
                   'emphysema, uses home oxygen with exertion, and continues to smoke. Room-air saturation is 89% with prolonged '
                   'expiration. Creatinine is 0.8 mg/dL and cognition is intact.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '51-year-old woman, ASA 2, scheduled for laparoscopic donor nephrectomy. She has no chronic illness, normal '
                   'blood pressure, creatinine 0.7 mg/dL, and excellent functional status. Pulmonary examination and cognition '
                   'are normal; there is no history of thrombosis.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '82-year-old man, ASA 4, for emergency repair of a perforated gastric ulcer. He has CKD stage 3a, vascular '
                   'dementia, chronic diuretic use, and two days of poor intake. He is hypotensive, confused, and creatinine has '
                   'risen from 1.3 to 1.9 mg/dL.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': '66-year-old woman, ASA 3, for pelvic exenteration for recurrent cervical cancer. She has a history of '
                   'iliac-vein thrombosis, lower-extremity lymphedema, and reduced mobility from pelvic pain. Therapeutic '
                   'enoxaparin is held. Lungs clear; creatinine 0.8 mg/dL; cognition intact.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '58-year-old man, ASA 3, undergoing open Ivor Lewis esophagectomy. He has severe reflux, chronic aspiration, '
                   'COPD, and a 40-pack-year smoking history. Oxygen saturation is 92% and he produces sputum each morning. '
                   'Renal function is normal and he is fully oriented.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '75-year-old woman, ASA 3, for lumbar fusion. She has CKD stage 3a, hypertension, and chronic NSAID use; '
                   'creatinine is 1.4 mg/dL. She reports good oral intake and remains independent. Lungs are clear and cognition '
                   'is normal.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': '41-year-old man, ASA 1, for outpatient excision of a wrist ganglion. He has no medical history, takes no '
                   'medications, runs regularly, and has normal pulmonary and renal function. He is alert, independent, and has '
                   'no prior VTE.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '87-year-old woman, ASA 4, for urgent open reduction of a periprosthetic femur fracture. She has severe '
                   'frailty, moderate dementia, and has been non-weight-bearing for five days. She previously had a calf DVT and '
                   'is currently receiving only prophylactic heparin. Lungs clear; creatinine 1.1 mg/dL.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': '64-year-old woman, ASA 3, for bilateral mastectomy with free-flap reconstruction. She has obesity, prior '
                   'chemotherapy, and a history of catheter-associated upper-extremity DVT. Anticoagulation was completed six '
                   'months ago. She remains active; lungs and kidneys are normal; cognition intact.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '70-year-old man, ASA 3, scheduled for open lobectomy for squamous-cell lung cancer. He has moderate COPD, '
                   'chronic productive cough, and recently completed antibiotics for bronchitis. Room-air saturation is 92%. '
                   'Creatinine is 0.9 mg/dL and cognition is normal.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '78-year-old man, ASA 4, for transcatheter aortic valve replacement. He has CKD stage 4, diabetes, anemia, '
                   'and baseline creatinine 2.4 mg/dL. He is frail, walks with a walker, and is mildly forgetful but oriented. '
                   'No active lung infection.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': '53-year-old woman, ASA 2, scheduled for robotic hysterectomy for early endometrial cancer. BMI is 31, but '
                   'she walks daily and has no prior VTE, pulmonary disease, renal impairment, or cognitive symptoms. Creatinine '
                   'is 0.6 mg/dL and oxygen saturation 99%.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '72-year-old man, ASA 3, for open cystectomy. He has muscle-invasive bladder cancer, chronic pelvic-vein '
                   'compression, and a prior pulmonary embolism during chemotherapy. Apixaban is held. He is less active because '
                   'of fatigue; lungs clear and creatinine 1.0 mg/dL.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '80-year-old woman, ASA 3, for urgent laparoscopic appendectomy for perforated appendicitis. She has mild '
                   'cognitive impairment and takes oxybutynin and clonazepam. She is febrile but hemodynamically stable; '
                   'creatinine 0.9 mg/dL and lungs are clear. Family notes increasing confusion today.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': '59-year-old man, ASA 3, for open distal pancreatectomy. He has CKD stage 3a, diabetes, and recent contrast '
                   'exposure during staging scans. Baseline creatinine is 1.4 mg/dL. He is hydrated, cognitively intact, and has '
                   'no pulmonary or thrombotic history.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': '67-year-old woman, ASA 3, for thoracic spine tumor resection. Metastatic breast cancer has caused '
                   'progressive leg weakness, and she has been wheelchair-bound for three weeks. She had a previous DVT during '
                   'chemotherapy. Lungs clear, creatinine 0.7 mg/dL, cognition normal.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '74-year-old man, ASA 3, for open repair of a large paraesophageal hernia. He has recurrent aspiration '
                   'pneumonia, dysphagia, and moderate COPD. Room-air saturation is 93% and coarse bibasilar breath sounds are '
                   'present. Renal function is normal; cognition intact.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '85-year-old man, ASA 4, undergoing urgent below-knee amputation for infected diabetic foot. He has CKD stage '
                   '4, peripheral vascular disease, dementia, and poor oral intake. Creatinine is 2.6 mg/dL from 2.1, and he is '
                   'intermittently agitated. Lungs are clear.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': '35-year-old man, ASA 1, scheduled for laparoscopic appendectomy for uncomplicated appendicitis. Symptoms '
                   'began twelve hours ago; he is hydrated and hemodynamically stable. No chronic disease, normal creatinine, '
                   'clear lungs, intact cognition, and no prior VTE.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '71-year-old woman, ASA 3, for revision shoulder arthroplasty. She has rheumatoid arthritis, obesity, and a '
                   'remote upper-extremity DVT related to a central line. She remains independently mobile and is not on '
                   'anticoagulation. Lungs and kidneys are normal; cognition intact.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '63-year-old man, ASA 3, planned total gastrectomy for proximal gastric cancer. He has emphysema, active '
                   'smoking, marked weight loss, and albumin 2.8 g/dL. Room-air saturation is 92% and cough is weak. Creatinine '
                   'is 0.8 mg/dL; cognition is normal.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '77-year-old woman, ASA 4, for open mitral valve replacement. She has CKD stage 3b, pulmonary hypertension, '
                   'diabetes, and baseline creatinine 1.8 mg/dL. She is frail and mildly forgetful but manages her own '
                   'medications. No active respiratory infection.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': '49-year-old woman, ASA 2, for elective laparoscopic colectomy for endometriosis involving the sigmoid colon. '
                   'She has iron-deficiency anemia but no other illness. She is active, cognitively normal, with clear lungs, '
                   'normal creatinine, and no VTE history.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '68-year-old man, ASA 3, for open pelvic tumor resection. He has metastatic sarcoma, chronic iliac-vein '
                   'compression, leg edema, and a previous DVT. Therapeutic anticoagulation is held for surgery. He walks only '
                   'short distances. Lungs clear; renal function normal.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '73-year-old woman, ASA 3, for VATS lobectomy. She has interstitial lung disease, uses 2 L oxygen with '
                   'exertion, and recently had worsening dry cough. Room-air saturation is 90%. Creatinine is 0.8 mg/dL; she is '
                   'independent and cognitively intact.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '82-year-old woman, ASA 3, scheduled for elective colostomy reversal. She has a history of postoperative '
                   'delirium, mild dementia, and severe visual impairment. She lives with her daughter and needs help with '
                   'medications. Lungs clear and creatinine 0.9 mg/dL.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': '61-year-old man, ASA 3, for open partial nephrectomy. He has CKD stage 3a, diabetes, and baseline creatinine '
                   '1.5 mg/dL. The tumor is central and prolonged hilar clamping is anticipated. He is active, cognitively '
                   'intact, with clear lungs and no VTE history.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': '74-year-old man, ASA 3, for ankle fusion. He has severe obesity, chronic immobility, and an unprovoked DVT '
                   'three years ago. Rivaroxaban is held. He uses a scooter outside the home. Lungs clear, creatinine 1.0 mg/dL, '
                   'cognition normal.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '65-year-old woman, ASA 3, for open thoracoabdominal aneurysm repair. She has COPD, active smoking, and '
                   'room-air saturation of 92%. She uses inhalers daily and has a weak cough. Creatinine is 1.1 mg/dL; cognition '
                   'is intact and no prior VTE is reported.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': '90-year-old man, ASA 4, for urgent cystoscopy and ureteral stent placement for obstructing stone with '
                   'infection. He has dementia, CKD stage 3b, poor intake, fever, and hypotension. Creatinine is 2.2 mg/dL from '
                   '1.5, and he is acutely disoriented.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': '38-year-old woman, ASA 1, undergoing elective breast reduction. She has no medical problems, normal BMI, '
                   'excellent functional capacity, and no history of pulmonary disease, kidney disease, thrombosis, or cognitive '
                   'symptoms. Preoperative examination and laboratory testing are normal.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '66-year-old man, ASA 3, for right hemicolectomy after neoadjuvant chemotherapy. He has a port-associated DVT '
                   'treated with apixaban and remains fatigued with reduced activity. Anticoagulation is held. Lungs are clear, '
                   'creatinine 0.9 mg/dL, cognition intact.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '70-year-old woman, ASA 3, scheduled for open nephrectomy. She has COPD, active tobacco use, and a recent '
                   'exacerbation requiring prednisone. Room-air saturation is 93% with scattered wheeze. Baseline creatinine is '
                   '0.8 mg/dL and cognition is normal.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '79-year-old woman, ASA 4, for emergency total colectomy for fulminant colitis. She has CKD stage 3b, chronic '
                   'steroid use, frailty, and baseline mild cognitive impairment. She is tachycardic, dehydrated, and confused; '
                   'creatinine is 2.0 mg/dL from 1.4.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': '54-year-old man, ASA 2, for elective robotic inguinal hernia repair. He has controlled hyperlipidemia and no '
                   'cardiopulmonary, renal, thrombotic, or neurologic disease. He exercises regularly, is fully independent, and '
                   'has normal preoperative testing.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '72-year-old woman, ASA 3, for open reduction of a pelvic fracture after a fall. She has been bedbound for '
                   'seven days, has metastatic breast cancer, and previously had a pulmonary embolism. Enoxaparin is temporarily '
                   'held. Lungs clear, creatinine 0.8 mg/dL, cognition intact.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '68-year-old man, ASA 3, undergoing pancreaticoduodenectomy. He has chronic bronchitis, continues to smoke, '
                   'and has room-air saturation of 94% with daily sputum. Albumin is 2.9 g/dL. Kidney function and cognition are '
                   'normal.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '84-year-old man, ASA 3, for elective hip arthroplasty. He has mild dementia, severe hearing loss, and '
                   'chronic nightly benzodiazepine use. He is oriented to person and place, ambulates with a cane, and has '
                   'normal lungs and renal function.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': '58-year-old woman, ASA 3, for open renal artery aneurysm repair. She has CKD stage 3a, hypertension, and '
                   'baseline creatinine 1.4 mg/dL. Recent angiography used iodinated contrast. She is hydrated, active, and '
                   'cognitively normal, without lung disease or VTE.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': '47-year-old man, ASA 2, scheduled for Achilles tendon repair two days after injury. He is otherwise healthy, '
                   'ambulates with crutches, and has no personal or family VTE history. Lungs, renal function, and cognition are '
                   'normal.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '76-year-old woman, ASA 3, for open lung biopsy. She has severe bronchiectasis, chronic colonization with '
                   'Pseudomonas, frequent exacerbations, and oxygen saturation of 91%. She uses airway-clearance therapy daily. '
                   'Creatinine is 0.7 mg/dL and cognition is intact.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '57-year-old woman, ASA 2, for laparoscopic sigmoid colectomy for recurrent diverticulitis. She has '
                   'controlled hypertension, walks daily, and has no prior VTE, lung disease, kidney disease, or cognitive '
                   'impairment. Creatinine is 0.7 mg/dL and oxygen saturation 98%.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '78-year-old man, ASA 4, for open repair of a symptomatic juxtarenal aneurysm. He has CKD stage 3b, diabetes, '
                   'heart failure, and creatinine 1.9 mg/dL. He is frail but oriented, with clear lungs and no prior '
                   'thromboembolism.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': '65-year-old woman, ASA 3, for total hip arthroplasty. She has antiphospholipid syndrome and two prior DVTs. '
                   'Warfarin is held with a perioperative bridging plan. Mobility is limited by pain. Pulmonary and renal '
                   'function are normal; cognition intact.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '69-year-old man, ASA 3, undergoing open esophagectomy. He has severe COPD, home oxygen at night, ongoing '
                   'smoking, and chronic productive cough. Room-air saturation is 90%. Baseline creatinine is 0.9 mg/dL, and he '
                   'is fully oriented.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '83-year-old woman, ASA 4, for urgent fixation of a subtrochanteric femur fracture. She has dementia, CKD '
                   'stage 3a, poor oral intake, and has been bedbound for three days. Creatinine is 1.5 mg/dL from 1.1, and she '
                   'is restless and disoriented.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': '31-year-old man, ASA 1, for laparoscopic inguinal hernia repair. He has no chronic illness, takes no regular '
                   'medications, exercises five days weekly, and has normal pulmonary, renal, and cognitive status. No personal '
                   'or family VTE history.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '62-year-old woman, ASA 3, for open nephrectomy for renal-cell carcinoma. She has CKD stage 3b in the '
                   'contralateral kidney and baseline creatinine 1.7 mg/dL. Diabetes is controlled. She is active, oriented, and '
                   'has no lung disease or VTE history.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': '74-year-old man, ASA 3, undergoing total knee replacement. He has a remote unprovoked pulmonary embolism, '
                   'BMI 41, and severe mobility restriction. Apixaban is held. Lungs are clear, creatinine 1.0 mg/dL, and '
                   'cognition is normal.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '81-year-old woman, ASA 3, for parathyroidectomy. She has mild dementia, poor vision, and a prior episode of '
                   'postoperative delirium. She lives with family and needs medication supervision. Lungs are clear and '
                   'creatinine is 0.9 mg/dL.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': '60-year-old man, ASA 3, for VATS lobectomy. Severe emphysema limits him to one block, and he uses oxygen '
                   'with activity. He stopped smoking two weeks ago. Oxygen saturation is 90% on room air. Kidney function and '
                   'cognition are normal.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '46-year-old woman, ASA 2, scheduled for laparoscopic adrenalectomy for a benign adenoma. Hypertension is '
                   'well controlled. She has normal renal function, good exercise capacity, clear lungs, no previous thrombosis, '
                   'and no cognitive concerns.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '86-year-old man, ASA 4, for emergency laparotomy after perforated appendicitis with diffuse peritonitis. He '
                   'has CKD stage 3b, dementia, dehydration, and hypotension. Creatinine is 2.1 mg/dL from 1.5. He is '
                   'inattentive and intermittently pulling at lines.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': '71-year-old woman, ASA 3, for debulking of advanced ovarian cancer. She has malignant ascites, a prior DVT, '
                   'and markedly reduced activity. Therapeutic apixaban is held. Lungs are clear, creatinine 0.8 mg/dL, and '
                   'cognition is intact.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '63-year-old man, ASA 3, planned total laryngectomy. He has chronic aspiration, thick tracheal secretions, '
                   'severe COPD, and recent treatment for bronchitis. Oxygen saturation is 92%. Creatinine is 0.9 mg/dL and '
                   'cognition is normal.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '76-year-old woman, ASA 3, for posterior lumbar fusion. She has CKD stage 3b, diabetes, chronic NSAID use, '
                   'and baseline creatinine 1.7 mg/dL. She is euvolemic and oriented, with no pulmonary disease or VTE history.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': '40-year-old man, ASA 1, for elective umbilical hernia repair. He is healthy, active, and takes no daily '
                   'medications. Pulmonary examination, creatinine, mental status, and mobility are normal; there is no '
                   'thrombosis history.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '88-year-old woman, ASA 4, for urgent hip-fracture repair. She has advanced dementia, hearing loss, frailty, '
                   'and has been immobile for four days. She developed a DVT after a prior fracture. Lungs are clear; creatinine '
                   'is 1.0 mg/dL.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': '55-year-old man, ASA 3, for extensive abdominal wall reconstruction. He has obesity, previous venous '
                   'thrombosis after bariatric surgery, and limited mobility from a large hernia. Anticoagulation was completed '
                   'last year. Lungs clear, kidneys normal, cognition intact.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '72-year-old woman, ASA 3, for open right upper lobectomy. She has severe COPD, chronic sputum production, '
                   'and FEV1 45% predicted. Room-air saturation is 91%. Baseline creatinine is 0.8 mg/dL, and she is independent '
                   'and oriented.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '79-year-old man, ASA 4, for combined aortic valve replacement and bypass grafting. He has CKD stage 4, '
                   'diabetes, heart failure, and creatinine 2.5 mg/dL. He is frail with mild cognitive impairment. No active '
                   'pulmonary infection is present.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': '52-year-old woman, ASA 2, for laparoscopic hysterectomy for fibroids. She has mild obesity but remains '
                   'active and has no history of VTE, lung disease, kidney disease, or cognitive impairment. Creatinine is 0.6 '
                   'mg/dL and saturation 99%.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '68-year-old man, ASA 3, undergoing radical cystectomy. He has bladder cancer, prior calf DVT during '
                   'chemotherapy, and reduced mobility from fatigue. Apixaban is stopped. Lungs are clear, creatinine 1.1 mg/dL, '
                   'and cognition is intact.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '82-year-old woman, ASA 3, for urgent cholecystectomy. She has mild dementia, chronic oxybutynin use, and a '
                   'history of delirium after urinary infection. She is febrile and disoriented to date. Lungs are clear; '
                   'creatinine is 1.0 mg/dL.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': '58-year-old man, ASA 3, for open Whipple procedure. He has diabetic nephropathy, baseline creatinine 1.6 '
                   'mg/dL, and recent poor intake from gastric outlet obstruction. He is oriented and has no lung disease or VTE '
                   'history.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': '66-year-old woman, ASA 3, for spinal decompression for metastatic disease. She has been wheelchair-dependent '
                   'for one month and previously had a pulmonary embolism during breast-cancer treatment. Anticoagulation is '
                   'held. Lungs clear, kidney function normal, cognition intact.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '70-year-old man, ASA 3, for open paraesophageal hernia repair. He has recurrent nocturnal aspiration, '
                   'bronchiectasis, and a recent course of antibiotics for pneumonia. Oxygen saturation is 93% and cough is '
                   'weak. Renal function and cognition are normal.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '84-year-old woman, ASA 4, for urgent above-knee amputation because of infected ischemic limb. She has CKD '
                   'stage 4, diabetes, dementia, and poor intake. Creatinine is 2.7 mg/dL from 2.2, and she is agitated and '
                   'disoriented. Lungs are clear.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': '34-year-old woman, ASA 1, for laparoscopic ovarian cystectomy. She has no chronic illness, normal exercise '
                   'tolerance, and no prior anesthesia or thrombotic complications. Creatinine, lungs, mobility, and cognition '
                   'are normal.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '69-year-old woman, ASA 3, for shoulder replacement. She has obesity and rheumatoid arthritis but remains '
                   'independently mobile. A remote line-associated DVT occurred during cancer treatment and has not recurred. '
                   'Lungs and kidneys are normal; cognition intact.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '64-year-old man, ASA 3, undergoing total gastrectomy. He has emphysema, active smoking, chronic cough, and '
                   'albumin 2.7 g/dL. Oxygen saturation is 92%. Creatinine is 0.8 mg/dL; he is alert and independent.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '75-year-old woman, ASA 4, for open mitral valve surgery. She has CKD stage 3b, pulmonary hypertension, '
                   'diabetes, and baseline creatinine 1.8 mg/dL. She is frail and mildly forgetful. Lungs are clear without '
                   'recent infection.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': '47-year-old man, ASA 2, for elective laparoscopic sigmoidectomy for diverticular stricture. He has '
                   'controlled hypertension and no pulmonary, renal, thrombotic, or neurologic disease. He works full time, '
                   'exercises regularly, and is fully independent.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '67-year-old woman, ASA 3, for pelvic sarcoma resection. Tumor compression causes unilateral leg swelling, '
                   'and she had a DVT six months ago. Apixaban is held. She walks only household distances. Lungs clear, '
                   'creatinine 0.7 mg/dL, cognition normal.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '74-year-old man, ASA 3, for VATS lobectomy. He has interstitial lung disease, exertional oxygen use, and '
                   'worsening dry cough. Room-air saturation is 89%. Renal function is normal and he is cognitively intact.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '83-year-old man, ASA 3, for elective colostomy reversal. He has a prior episode of postoperative delirium, '
                   'vascular cognitive impairment, and severe hearing loss. His wife manages medications. Lungs are clear and '
                   'creatinine is 1.0 mg/dL.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': '59-year-old woman, ASA 3, undergoing central partial nephrectomy. She has CKD stage 3a, diabetes, and '
                   'baseline creatinine 1.4 mg/dL. Prolonged warm ischemia is anticipated. She is active, oriented, and without '
                   'pulmonary disease or VTE history.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': '72-year-old man, ASA 3, for ankle arthrodesis. He has severe obesity, chronic immobility, and a prior '
                   'unprovoked DVT. Rivaroxaban is held. He uses a motorized scooter. Lungs clear, creatinine 1.0 mg/dL, '
                   'cognition intact.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '63-year-old woman, ASA 4, for thoracoabdominal aneurysm repair. She has COPD, active smoking, CKD stage 3a, '
                   'and room-air saturation of 92%. Creatinine is 1.4 mg/dL. She is cognitively normal and has no prior VTE.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': '87-year-old woman, ASA 4, for urgent ureteral stent placement for obstructed infected kidney. She has '
                   'dementia, CKD stage 3b, fever, hypotension, and poor intake. Creatinine is 2.3 mg/dL from 1.6, and she is '
                   'acutely confused.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': '43-year-old woman, ASA 1, undergoing elective abdominoplasty. She is healthy, has normal BMI, exercises '
                   'regularly, and has no pulmonary disease, renal impairment, prior thrombosis, or cognitive symptoms. '
                   'Preoperative examination is normal.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '64-year-old man, ASA 3, for open colectomy after chemotherapy. He developed a catheter-associated DVT three '
                   'months ago and remains on apixaban, now held. Fatigue has reduced his activity. Lungs clear, creatinine 0.9 '
                   'mg/dL, cognition intact.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '71-year-old woman, ASA 3, for open nephrectomy. She has moderate COPD, active smoking, and a recent '
                   'steroid-treated exacerbation. Oxygen saturation is 92% with scattered wheeze. Baseline creatinine is 0.9 '
                   'mg/dL and cognition is normal.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '80-year-old man, ASA 4, for emergency total colectomy for toxic megacolon. He has CKD stage 3b, chronic '
                   'steroid use, frailty, and mild dementia. He is dehydrated and confused; creatinine is 2.1 mg/dL from 1.5.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': '55-year-old woman, ASA 2, for robotic ventral hernia repair. She has controlled hyperlipidemia and no '
                   'pulmonary, renal, thrombotic, or neurologic disease. She walks daily and is fully independent with normal '
                   'testing.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '73-year-old man, ASA 3, for pelvic-fracture fixation. He has been bedbound for eight days, has metastatic '
                   'prostate cancer, and previously had a pulmonary embolism. Enoxaparin is held. Lungs clear, creatinine 0.9 '
                   'mg/dL, cognition intact.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '67-year-old woman, ASA 3, for pancreaticoduodenectomy. She has chronic bronchitis, continues to smoke, and '
                   'has daily sputum production with oxygen saturation 93%. Albumin is 2.9 g/dL. Kidney function and cognition '
                   'are normal.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '86-year-old woman, ASA 3, for elective hip replacement. She has mild dementia, severe hearing impairment, '
                   'and chronic benzodiazepine use. She is oriented to person and place and ambulates with a cane. Lungs and '
                   'renal function are normal.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': '60-year-old man, ASA 3, for open renal-artery reconstruction. He has CKD stage 3a, hypertension, and recent '
                   'iodinated contrast exposure. Baseline creatinine is 1.5 mg/dL. He is hydrated, active, and cognitively '
                   'normal without pulmonary disease.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': '49-year-old woman, ASA 2, for Achilles tendon repair three days after injury. She is otherwise healthy, '
                   'mobile with crutches, and has no personal or family history of VTE. Lungs, kidneys, and cognition are '
                   'normal.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '77-year-old man, ASA 3, for open lung biopsy. He has severe bronchiectasis, chronic Pseudomonas '
                   'colonization, frequent exacerbations, and oxygen saturation of 91%. He performs airway clearance twice '
                   'daily. Creatinine is 0.8 mg/dL and cognition is intact.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '58-year-old man, ASA 2, for elective laparoscopic splenectomy for a benign cyst. He has controlled '
                   'hypertension, good exercise tolerance, normal renal function, clear lungs, no VTE history, and no cognitive '
                   'concerns.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '73-year-old woman, ASA 4, for open repair of a descending thoracic aneurysm. She has CKD stage 3b, diabetes, '
                   'and baseline creatinine 1.8 mg/dL. Moderate COPD and oxygen saturation of 93% are noted. She is oriented and '
                   'independent.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': '66-year-old man, ASA 3, for revision total hip arthroplasty. He has a history of two postoperative DVTs, '
                   'obesity, and limited mobility. Warfarin is held with bridging. Lungs are clear, creatinine 1.0 mg/dL, '
                   'cognition normal.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '75-year-old woman, ASA 3, for open diaphragmatic hernia repair. She has chronic aspiration from dysphagia, '
                   'moderate COPD, and a weak cough. Room-air saturation is 92%. Kidney function is normal and she is '
                   'cognitively intact.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '88-year-old man, ASA 4, for urgent fixation of a displaced humeral fracture. He has advanced dementia, '
                   'Parkinson disease, dysphagia, and resides in a nursing facility. He is disoriented and intermittently '
                   'somnolent. Lungs have coarse bases; creatinine is 1.1 mg/dL.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': '37-year-old woman, ASA 1, scheduled for laparoscopic endometriosis excision. She has no chronic disease, '
                   'normal exercise capacity, and no prior thrombotic or anesthesia complications. Pulmonary, renal, and '
                   'cognitive assessments are normal.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '61-year-old man, ASA 3, for ureteral reconstruction. He has a solitary kidney, CKD stage 3a, and recurrent '
                   'obstruction; creatinine is 1.5 mg/dL. He is well hydrated, active, and oriented, with no pulmonary disease '
                   'or VTE history.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': '70-year-old woman, ASA 3, for ankle replacement. She has Factor V Leiden, a prior DVT, obesity, and six '
                   'weeks of limited weight-bearing. Rivaroxaban is held. Lungs clear, kidney function normal, cognition intact.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '80-year-old man, ASA 3, for elective inguinal hernia repair. He has mild cognitive impairment, severe '
                   'macular degeneration, and nightly diphenhydramine use. He lives independently with family oversight. Lungs '
                   'and renal function are normal.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': '65-year-old woman, ASA 3, undergoing left upper lobectomy. She has severe asthma-COPD overlap, three '
                   'exacerbations this year, and oxygen saturation of 92%. She completed prednisone one week ago. Creatinine 0.7 '
                   'mg/dL; cognition intact.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '45-year-old man, ASA 2, for elective laparoscopic appendectomy after resolved appendiceal abscess. He is '
                   'healthy apart from controlled reflux, exercises regularly, and has normal lungs, renal function, mobility, '
                   'and cognition.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '84-year-old woman, ASA 4, for emergency laparotomy for perforated colon cancer. She has CKD stage 3b, heart '
                   'failure, baseline dementia, and poor intake. She is hypotensive and confused; creatinine is 2.0 mg/dL from '
                   '1.4. Oxygen saturation is 94% on 2 L.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': '69-year-old man, ASA 3, for radical prostatectomy. Metastatic pelvic lymphadenopathy compresses the iliac '
                   'vein, and he had a DVT eight months ago. Apixaban is stopped. He remains ambulatory; lungs clear, creatinine '
                   '0.9 mg/dL, cognition normal.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '60-year-old woman, ASA 3, for open esophageal diverticulectomy. She has recurrent aspiration, '
                   'bronchiectasis, and chronic productive cough. Room-air saturation is 93%. Renal function is normal and she '
                   'is fully oriented.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '77-year-old man, ASA 3, for cervical fusion. He has CKD stage 3b, diabetes, and chronic ibuprofen use; '
                   'creatinine is 1.8 mg/dL. He is euvolemic and cognitively intact, with no lung disease or previous VTE.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': '29-year-old woman, ASA 1, for excision of a benign neck mass. She has no medical history, takes no '
                   'medications, and has normal exercise capacity, renal function, pulmonary status, and cognition.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '86-year-old man, ASA 4, for urgent fixation of a femoral neck fracture. He has moderate dementia, prior DVT, '
                   'and has been bedbound for three days. He is disoriented and dehydrated. Lungs clear; creatinine is 1.3 mg/dL '
                   'from 1.0.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': '63-year-old woman, ASA 3, for bilateral breast reconstruction. She has obesity, active malignancy, and a '
                   'previous catheter-associated DVT. Anticoagulation was recently completed. She remains active; lungs and '
                   'kidneys are normal; cognition intact.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '71-year-old man, ASA 3, for open segmentectomy for lung cancer. He has severe COPD, frequent sputum, and a '
                   'recent respiratory infection. Room-air saturation is 91%. Creatinine is 0.9 mg/dL and cognition is normal.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '81-year-old woman, ASA 4, for open tricuspid valve repair. She has CKD stage 3b, right-heart failure, '
                   'diabetes, frailty, and baseline creatinine 1.9 mg/dL. She is mildly forgetful and needs help with '
                   'medications. No active lung infection.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': '50-year-old man, ASA 2, for robotic partial colectomy for a benign tumor. He has controlled hypertension and '
                   'otherwise good health. He runs weekly, has normal renal and pulmonary function, no VTE history, and intact '
                   'cognition.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '67-year-old woman, ASA 3, for open hysterectomy for uterine sarcoma. She has a prior pulmonary embolism, '
                   'pelvic venous compression, and decreased mobility from pain. Apixaban is held. Lungs clear; creatinine 0.7 '
                   'mg/dL; cognition normal.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '79-year-old man, ASA 3, for urgent TURP because of hematuria and clot retention. He has vascular cognitive '
                   'impairment, hearing loss, and takes nightly zolpidem. He is oriented to person and place but not date. Lungs '
                   'clear; creatinine 1.1 mg/dL.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': '56-year-old woman, ASA 3, for open pancreatic resection. She has CKD stage 3a, diabetes, and recent vomiting '
                   'with creatinine 1.6 mg/dL from 1.3. She is oriented and has no lung disease or VTE history.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': '68-year-old man, ASA 3, for decompression of metastatic spinal cord compression. He has been nearly immobile '
                   'for two weeks and had a DVT during chemotherapy. Anticoagulation is held. Lungs clear, renal function '
                   'normal, cognition intact.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '72-year-old woman, ASA 3, for open repair of a giant hiatal hernia. She has nocturnal aspiration, chronic '
                   'cough, and moderate restrictive lung disease. Oxygen saturation is 92% and cough is weak. Kidney function '
                   'and cognition are normal.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '87-year-old man, ASA 4, for urgent transmetatarsal amputation for diabetic infection. He has CKD stage 4, '
                   'dementia, dehydration, and fever. Creatinine is 2.8 mg/dL from 2.2, and he is inattentive and agitated. '
                   'Lungs are clear.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': '41-year-old woman, ASA 1, for outpatient laparoscopic tubal ligation. She has no chronic illness, normal '
                   'BMI, excellent exercise tolerance, and no pulmonary, renal, thrombotic, or cognitive history.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '70-year-old man, ASA 3, for reverse shoulder arthroplasty. He has obesity and rheumatoid arthritis but '
                   'remains mobile and has no personal VTE history. Lungs clear, creatinine 1.0 mg/dL, and cognition is intact.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '62-year-old woman, ASA 3, for total gastrectomy. She has COPD, active smoking, chronic sputum, and severe '
                   'malnutrition with albumin 2.6 g/dL. Room-air saturation is 92%. Creatinine is normal and cognition intact.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '78-year-old man, ASA 4, for open aortic valve replacement. He has CKD stage 3b, diabetes, anemia, and '
                   'baseline creatinine 1.8 mg/dL. He is frail with mild cognitive impairment. Lungs are clear without '
                   'infection.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': '48-year-old woman, ASA 2, for laparoscopic bowel resection for Crohn stricture. Disease is controlled and '
                   'she is nutritionally replete. She has no pulmonary or renal disease, prior thrombosis, or cognitive '
                   'impairment and remains active.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '65-year-old man, ASA 3, for pelvic tumor debulking. He has iliac-vein compression, chronic leg edema, active '
                   'cancer, and a DVT nine months ago. Apixaban is held. Mobility is limited; lungs and renal function are '
                   'normal.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '76-year-old woman, ASA 3, for VATS wedge resection. She has pulmonary fibrosis, exertional oxygen use, and '
                   'room-air saturation of 90%. A recent cough has worsened. Creatinine is 0.7 mg/dL and cognition is intact.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '84-year-old woman, ASA 3, for elective hernia repair. She has a prior postoperative delirium episode, mild '
                   'Alzheimer disease, and severe hearing impairment. Her daughter manages medications. Lungs and renal function '
                   'are normal.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': '60-year-old man, ASA 3, for partial nephrectomy of a hilar mass. He has CKD stage 3a and baseline creatinine '
                   '1.5 mg/dL; prolonged clamping is anticipated. He is active and oriented, with no pulmonary disease or VTE '
                   'history.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': '71-year-old woman, ASA 3, for hindfoot fusion. She has Factor V Leiden, previous DVT, obesity, and prolonged '
                   'limited weight-bearing. Rivaroxaban is held. Lungs clear, kidneys normal, cognition intact.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '64-year-old man, ASA 4, for open thoracic aneurysm repair. He has severe COPD, active smoking, CKD stage 3a, '
                   'oxygen saturation 91%, and creatinine 1.4 mg/dL. He is cognitively normal with no prior VTE.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': '89-year-old woman, ASA 4, for urgent nephrostomy and stone extraction for urosepsis. She has dementia, CKD '
                   'stage 3b, hypotension, and poor intake. Creatinine is 2.4 mg/dL from 1.6, and she is acutely disoriented.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': '36-year-old man, ASA 1, for elective gynecomastia excision. He is healthy, active, and takes no regular '
                   'medications. Pulmonary and renal function, mobility, cognition, and thrombosis history are unremarkable.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '63-year-old woman, ASA 3, for hemicolectomy during active chemotherapy. She had a port-related DVT four '
                   'months ago and is on apixaban, now held. She remains fatigued and less mobile. Lungs clear, creatinine 0.8 '
                   'mg/dL, cognition intact.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '69-year-old man, ASA 3, for open nephrectomy. He has severe COPD, ongoing smoking, and completed prednisone '
                   'for an exacerbation five days ago. Room-air saturation is 92% with wheeze. Creatinine is 0.9 mg/dL and '
                   'cognition normal.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '82-year-old woman, ASA 4, for emergency colectomy for bowel necrosis. She has CKD stage 3b, chronic '
                   'steroids, frailty, and dementia. She is hypotensive, dehydrated, and confused; creatinine is 2.2 mg/dL from '
                   '1.5.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': '51-year-old man, ASA 2, for elective laparoscopic ventral hernia repair. He has controlled dyslipidemia and '
                   'otherwise good health. He exercises regularly and has normal lungs, kidneys, mobility, and cognition with no '
                   'VTE history.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '74-year-old woman, ASA 3, for fixation of a pelvic ring fracture. She has been bedbound for nine days, has '
                   'active lymphoma, and previously had a pulmonary embolism. Enoxaparin is held. Lungs clear, creatinine 0.8 '
                   'mg/dL, cognition intact.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '66-year-old man, ASA 3, for open Whipple procedure. He has chronic bronchitis, daily sputum, active smoking, '
                   'and room-air saturation 93%. Albumin is 2.8 g/dL. Kidney function and cognition are normal.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '85-year-old man, ASA 3, for elective knee replacement. He has mild dementia, severe hearing loss, and '
                   'chronic clonazepam use. He is oriented to person and place and walks with a cane. Lungs and renal function '
                   'are normal.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': '57-year-old woman, ASA 3, for open renal artery repair. She has CKD stage 3a, hypertension, and recent '
                   'contrast angiography; creatinine is 1.5 mg/dL. She is hydrated, active, oriented, and without lung disease '
                   'or VTE history.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': '45-year-old man, ASA 2, for acute Achilles repair. He is otherwise healthy, mobile with crutches, and has no '
                   'personal or family VTE history. Pulmonary and renal function and cognition are normal.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '75-year-old woman, ASA 3, for surgical lung biopsy. She has severe bronchiectasis, chronic '
                   'resistant-organism colonization, frequent exacerbations, and oxygen saturation of 91%. Creatinine is 0.8 '
                   'mg/dL and cognition is intact.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '54-year-old woman, ASA 2, presents for elective laparoscopic gastric bypass. BMI is 46, obstructive sleep '
                   'apnea is treated consistently with CPAP, and she walks thirty minutes daily. She has no prior VTE, kidney '
                   'disease, cognitive impairment, or current respiratory symptoms.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '81-year-old man, ASA 4, for urgent open cholecystectomy after failed percutaneous drainage. He has COPD, '
                   'mild dementia, CKD stage 3b, and persistent sepsis. Oxygen saturation is 92% on 2 L; creatinine is 1.9 mg/dL '
                   'from 1.5, and attention fluctuates.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': '59-year-old woman, ASA 3, scheduled for open resection of a retroperitoneal sarcoma. The tumor compresses '
                   'the inferior vena cava, and she developed a proximal DVT three months ago. Apixaban is held. She remains '
                   'ambulatory; lungs and kidneys are normal; cognition intact.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '72-year-old man, ASA 3, for maxillectomy with free-flap reconstruction after radiation. He has severe '
                   'dysphagia, recurrent aspiration, thick secretions, and a 60-pack-year smoking history. Oxygen saturation is '
                   '93%. Creatinine is normal and he is cognitively intact.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '77-year-old woman, ASA 4, for open descending aortic repair. She has CKD stage 4, diabetes, and baseline '
                   'creatinine 2.2 mg/dL. She is frail with mild cognitive impairment but has no active pulmonary disease. '
                   'Family assists with medications.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': '28-year-old man, ASA 1, for outpatient ACL reconstruction. He is healthy, athletic, and was weight-bearing '
                   'until surgery. There is no personal or family history of VTE, and pulmonary, renal, and cognitive '
                   'assessments are normal.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '68-year-old woman, ASA 3, for total knee replacement. She has obesity, varicose veins, and limited mobility '
                   'but no prior VTE or thrombophilia. She remains independent and takes daily walks. Lungs clear, creatinine '
                   '0.8 mg/dL, cognition normal.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '83-year-old man, ASA 3, for transurethral resection of a bladder tumor. He has Parkinson disease, mild '
                   'dementia, visual impairment, and prior delirium during hospitalization. He is oriented to person and place. '
                   'Renal function and lungs are stable.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': '64-year-old woman, ASA 3, undergoing right pneumonectomy. She has moderate COPD, active smoking, and DLCO '
                   '41% predicted. Oxygen saturation is 90%, with chronic productive cough. Creatinine is 0.7 mg/dL and '
                   'cognition is normal.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '57-year-old man, ASA 3, for radical nephrectomy. He has a solitary contralateral kidney with CKD stage 3a, '
                   'diabetes, and creatinine 1.5 mg/dL. He is euvolemic, active, and oriented. Lungs clear; no VTE history.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': '43-year-old woman, ASA 2, for laparoscopic hysterectomy for endometriosis. She has controlled migraines but '
                   'no cardiopulmonary, renal, thrombotic, or cognitive disease. She exercises regularly and has normal '
                   'laboratory testing.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '85-year-old woman, ASA 4, for urgent fixation of a hip fracture. She has advanced dementia, chronic '
                   'aspiration, and has been bedbound for four days. She is coughing with meals and oxygen saturation is 92%. '
                   'Creatinine is 1.0 mg/dL.',
  'DVT': 1,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': '66-year-old man, ASA 3, for open colectomy for colon cancer. He had a pulmonary embolism during '
                   'chemotherapy, has a central venous port, and reports declining activity. Apixaban is held. Lungs clear, '
                   'creatinine 0.9 mg/dL, cognition intact.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '70-year-old woman, ASA 3, for open pancreatic resection. She has emphysema, ongoing tobacco use, and a '
                   'recent COPD exacerbation. Oxygen saturation is 92% with wheezing. Creatinine is 0.8 mg/dL, and she is fully '
                   'oriented.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '79-year-old man, ASA 4, for coronary bypass surgery. He has CKD stage 3b, diabetes, low EF, and creatinine '
                   '1.9 mg/dL. Baseline cognition is normal, and he lives independently. Lungs are clear without infection.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': '52-year-old woman, ASA 2, for lumpectomy and sentinel-node biopsy. She has controlled hypothyroidism, normal '
                   'BMI, and excellent functional capacity. No pulmonary, renal, thrombotic, or cognitive history is present.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '74-year-old man, ASA 3, for ankle-fracture fixation after ten days in an external fixator. He has active '
                   'pancreatic cancer and a previous DVT. Anticoagulation is held. He is non-weight-bearing; lungs clear, renal '
                   'function normal, cognition intact.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '68-year-old woman, ASA 3, for open esophagectomy. She has scleroderma with severe reflux, recurrent '
                   'aspiration, and restrictive lung disease. Oxygen saturation is 92% and cough is weak. Creatinine is normal; '
                   'cognition is intact.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '87-year-old man, ASA 4, for emergency repair of a perforated duodenal ulcer. He has CKD stage 3b, moderate '
                   'dementia, and two days of vomiting. He is hypotensive and confused; creatinine is 2.0 mg/dL from 1.4.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': '49-year-old man, ASA 2, for laparoscopic adrenalectomy. He has controlled hypertension and otherwise good '
                   'health. He remains active with normal pulmonary and renal function, no prior VTE, and intact cognition.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '69-year-old woman, ASA 3, for open pelvic exenteration. She has recurrent cervical cancer, prior iliac DVT, '
                   'unilateral leg edema, and reduced mobility. Therapeutic enoxaparin is withheld. Lungs clear, creatinine 0.8 '
                   'mg/dL, cognition normal.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '73-year-old man, ASA 3, for lower lobectomy. He has COPD, bronchiectasis, daily sputum, and two respiratory '
                   'infections in the past year. Oxygen saturation is 92%. Renal function is normal; cognition intact.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '82-year-old woman, ASA 3, for elective mastectomy. She has mild Alzheimer disease, severe hearing loss, and '
                   'chronic anticholinergic bladder medication. She is oriented to person and place. Lungs and kidneys are '
                   'normal.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': '60-year-old man, ASA 3, for partial nephrectomy. He has diabetic CKD stage 3b with creatinine 1.7 mg/dL and '
                   'recent contrast exposure. He is hydrated, oriented, and has no pulmonary disease or VTE history.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': '71-year-old woman, ASA 3, for multilevel lumbar fusion. She has obesity, limited mobility, and a prior '
                   'postoperative DVT. Apixaban is held. Lungs clear, creatinine 0.9 mg/dL, cognition intact.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '65-year-old man, ASA 4, for open thoracic aneurysm repair. He has severe COPD, CKD stage 3a, active smoking, '
                   'oxygen saturation 91%, and creatinine 1.4 mg/dL. Cognition is normal and there is no VTE history.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': '90-year-old woman, ASA 4, for urgent debridement of an infected pressure ulcer. She has advanced dementia, '
                   'CKD stage 3b, dehydration, and fever. Creatinine is 2.1 mg/dL from 1.5, and she is somnolent and '
                   'inattentive.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': '32-year-old man, ASA 1, for outpatient repair of an umbilical hernia. He has no chronic conditions, takes no '
                   'regular medication, exercises frequently, and has normal lungs, renal function, cognition, and mobility.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '64-year-old woman, ASA 3, for colectomy during treatment of metastatic colon cancer. She has a recent '
                   'port-associated DVT and is on apixaban, now held. Fatigue limits activity. Lungs clear, creatinine 0.8 '
                   'mg/dL, cognition intact.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '72-year-old man, ASA 3, for open partial hepatectomy. He has COPD, active smoking, chronic sputum, and '
                   'oxygen saturation 93%. Albumin is 3.0 g/dL. Renal function is normal and cognition intact.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '80-year-old woman, ASA 4, for urgent bowel resection for mesenteric ischemia. She has CKD stage 3b, heart '
                   'failure, frailty, and mild dementia. She arrived hypotensive and confused; creatinine is 2.2 mg/dL from 1.6.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': '55-year-old man, ASA 2, for elective robotic prostatectomy. He has controlled hypertension, exercises '
                   'regularly, and has no pulmonary disease, renal impairment, VTE history, or cognitive concerns. Creatinine is '
                   '0.9 mg/dL.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '75-year-old woman, ASA 3, for pelvic-fracture fixation. She has been non-weight-bearing for twelve days, has '
                   'active lymphoma, and previously developed PE after surgery. Anticoagulation is held. Lungs clear, kidneys '
                   'normal, cognition intact.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '67-year-old man, ASA 3, for open gastrectomy. He has chronic bronchitis, active smoking, weak cough, and '
                   'room-air saturation 92%. Albumin is 2.8 g/dL. Creatinine is normal and cognition intact.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '86-year-old man, ASA 3, for elective shoulder replacement. He has mild dementia, severe hearing loss, and '
                   'nightly lorazepam use. He is oriented to person and place and lives with family. Lungs and renal function '
                   'are normal.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': '58-year-old woman, ASA 3, for open repair of renal artery stenosis. She has CKD stage 3a, diabetes, and '
                   'recent contrast angiography; creatinine is 1.5 mg/dL. She is euvolemic and cognitively intact, without lung '
                   'disease.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': '46-year-old woman, ASA 2, for repair of an acute Achilles rupture. She is healthy, ambulates with crutches, '
                   'and has no prior VTE or family thrombophilia. Pulmonary, renal, and cognitive status are normal.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '78-year-old man, ASA 3, for VATS decortication after recurrent empyema. He has bronchiectasis, chronic '
                   'sputum, and oxygen saturation of 91%. He recently completed antibiotics but remains deconditioned. '
                   'Creatinine is 0.9 mg/dL and cognition intact.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '74-year-old woman, ASA 3, for elective total hip arthroplasty. She has obesity and chronic venous '
                   'insufficiency but no prior thrombosis. She remains active with a cane and has normal lungs, renal function, '
                   'and cognition.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '82-year-old man, ASA 4, for open aortic arch repair. He has CKD stage 4, diabetes, prior stroke, and mild '
                   'vascular cognitive impairment. Creatinine is 2.4 mg/dL. Lungs are clear and he is oriented at baseline.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': '62-year-old woman, ASA 3, for pancreaticoduodenectomy. She has previous DVT during ovarian-cancer treatment, '
                   'obesity, and reduced mobility from abdominal pain. Apixaban is held. Lungs clear, creatinine 0.8 mg/dL, '
                   'cognition intact.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '69-year-old man, ASA 3, for open esophagectomy. He has severe reflux with aspiration, COPD, active smoking, '
                   'and oxygen saturation 92%. A productive cough has worsened over the last month. Kidney function and '
                   'cognition are normal.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '88-year-old woman, ASA 4, for emergency surgery for strangulated hernia. She has CKD stage 3b, dementia, '
                   'vomiting, and hypotension. Creatinine is 2.0 mg/dL from 1.4, and she is disoriented and pulling at '
                   'monitoring leads.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': '39-year-old man, ASA 1, for elective arthroscopic meniscus repair. He is healthy, exercises regularly, and '
                   'has normal pulmonary and renal function, intact cognition, and no personal or family history of thrombosis.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '70-year-old woman, ASA 3, for open ovarian-cancer debulking. She has malignant ascites, prior DVT, obesity, '
                   'and limited mobility. Enoxaparin is withheld. Lungs are clear, creatinine 0.8 mg/dL, and cognition is '
                   'normal.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '73-year-old man, ASA 3, for left upper lobectomy. He has severe COPD, chronic productive cough, and FEV1 43% '
                   'predicted. Oxygen saturation is 91%. Creatinine is normal and cognition intact.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '84-year-old woman, ASA 3, for elective bowel resection. She has mild Alzheimer disease, poor vision, and '
                   'prior postoperative confusion. Her son manages medications. Lungs and renal function are normal; she is '
                   'oriented to person and place.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': '61-year-old man, ASA 3, for nephroureterectomy. He has CKD stage 3b, diabetes, and baseline creatinine 1.8 '
                   'mg/dL. The remaining kidney has reduced function. He is active and oriented, with clear lungs and no VTE '
                   'history.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': '72-year-old woman, ASA 3, for revision knee replacement. She has a prior unprovoked DVT, obesity, and '
                   'limited mobility. Rivaroxaban is held. Lungs clear, creatinine 0.9 mg/dL, cognition intact.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '66-year-old man, ASA 4, for open thoracoabdominal aneurysm repair. He has severe COPD, CKD stage 3a, active '
                   'smoking, oxygen saturation 91%, and creatinine 1.4 mg/dL. He is cognitively intact.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': '89-year-old man, ASA 4, for urgent ureteral stenting for obstructive pyelonephritis. He has dementia, CKD '
                   'stage 3b, hypotension, and poor intake. Creatinine is 2.5 mg/dL from 1.7, and he is acutely agitated.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': '55-year-old woman, ASA 2, for elective laparoscopic liver cyst fenestration. She has controlled '
                   'hypertension, good functional capacity, normal creatinine, clear lungs, no VTE history, and intact '
                   'cognition.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '76-year-old man, ASA 4, for open renal-artery bypass. He has CKD stage 4, diabetes, diffuse vascular '
                   'disease, and creatinine 2.3 mg/dL. He is frail but oriented, with no active pulmonary disease.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': '68-year-old woman, ASA 3, for femoral-popliteal bypass. She has a prior DVT, active smoking, obesity, and '
                   'limited walking from ischemic pain. Apixaban is held. Lungs clear, creatinine 1.0 mg/dL, cognition normal.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '71-year-old man, ASA 3, for open splenectomy after recurrent lymphoma. He has severe COPD, daily sputum, and '
                   'oxygen saturation 92%. He recently completed antibiotics for bronchitis. Renal function and cognition are '
                   'normal.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '85-year-old woman, ASA 4, for urgent fixation of an ankle fracture. She has dementia, frailty, and has been '
                   'non-weight-bearing for five days. She is disoriented but hemodynamically stable. Lungs clear; creatinine 1.0 '
                   'mg/dL.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': '40-year-old woman, ASA 1, for elective laparoscopic cholecystectomy. She has no chronic conditions, takes no '
                   'regular medication, and has normal mobility, lungs, renal function, and cognition.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '63-year-old man, ASA 3, for open pyeloplasty in a solitary kidney. He has CKD stage 3b and baseline '
                   'creatinine 1.8 mg/dL. He is hydrated and oriented, with clear lungs and no VTE history.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': '74-year-old woman, ASA 3, for total hip replacement. She has a prior unprovoked DVT, obesity, and severe '
                   'preoperative immobility. Rivaroxaban is held. Lungs clear, renal function normal, cognition intact.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '82-year-old man, ASA 3, for elective cataract surgery under sedation. He has mild dementia, severe hearing '
                   'impairment, and prior delirium after hospitalization. He is oriented to person and place. Lungs and kidneys '
                   'are stable.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': '66-year-old woman, ASA 3, for open lobectomy. She has emphysema, active smoking, chronic cough, and oxygen '
                   'saturation 91%. Creatinine is 0.7 mg/dL and cognition is normal.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '48-year-old man, ASA 2, for robotic partial nephrectomy. He has controlled hypertension and otherwise good '
                   'health. Creatinine is 0.9 mg/dL, lungs are clear, and there is no VTE or cognitive history.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '83-year-old woman, ASA 4, for emergency small-bowel resection. She has CKD stage 3b, dementia, vomiting, and '
                   'sepsis. Creatinine is 2.1 mg/dL from 1.5; she is hypotensive and acutely confused.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': '70-year-old man, ASA 3, for radical cystectomy. He has a prior PE, active cancer, and reduced activity. '
                   'Apixaban is held. Lungs clear, creatinine 1.1 mg/dL, cognition intact.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '62-year-old woman, ASA 3, for open esophagectomy. She has recurrent aspiration, bronchiectasis, and oxygen '
                   'saturation 92%. She produces sputum daily. Renal function and cognition are normal.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '75-year-old man, ASA 3, for posterior spinal fusion. He has CKD stage 3b, diabetes, and chronic NSAID '
                   'exposure; creatinine is 1.8 mg/dL. Lungs clear and cognition intact.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': '35-year-old woman, ASA 1, for outpatient breast biopsy. She is healthy, active, and has normal pulmonary, '
                   'renal, thrombotic, and cognitive status.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '87-year-old man, ASA 4, for urgent hemiarthroplasty. He has dementia, prior DVT, and three days of '
                   'immobility. He is disoriented and needs full assistance. Lungs clear; creatinine 1.1 mg/dL.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': '60-year-old woman, ASA 3, for abdominal cancer debulking. She has obesity, a previous '
                   'chemotherapy-associated DVT, and low activity. Anticoagulation is held. Lungs and kidneys are normal; '
                   'cognition intact.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '73-year-old man, ASA 3, for left lower lobectomy. He has COPD, chronic sputum, and recent pneumonia. Oxygen '
                   'saturation is 91%. Creatinine is normal and cognition intact.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '80-year-old woman, ASA 4, for open valve replacement. She has CKD stage 4, diabetes, frailty, and creatinine '
                   '2.3 mg/dL. Mild cognitive impairment is documented. No active lung infection.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': '51-year-old man, ASA 2, for laparoscopic colectomy. He has controlled hypertension, exercises regularly, and '
                   'has normal lungs, kidneys, mobility, and cognition with no VTE history.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '69-year-old woman, ASA 3, for open hysterectomy for malignancy. She has pelvic-vein compression, prior DVT, '
                   'and limited mobility. Apixaban is held. Lungs clear, creatinine normal, cognition intact.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '81-year-old man, ASA 3, for cystoscopy and tumor resection. He has vascular dementia, severe hearing loss, '
                   'and nightly zolpidem use. He is oriented to person and place. Lungs and kidneys are stable.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': '58-year-old woman, ASA 3, for distal pancreatectomy. She has CKD stage 3a, diabetes, and recent poor intake; '
                   'creatinine is 1.6 mg/dL from 1.3. She is oriented with clear lungs.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': '67-year-old man, ASA 3, for spinal tumor decompression. He has been wheelchair-bound for three weeks and '
                   'previously had a DVT during chemotherapy. Anticoagulation is held. Lungs and renal function are normal.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '72-year-old woman, ASA 3, for open hiatal hernia repair. She has recurrent aspiration, chronic cough, and '
                   'restrictive lung disease. Oxygen saturation is 92%. Kidney function and cognition are normal.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '86-year-old man, ASA 4, for urgent foot amputation. He has CKD stage 4, diabetes, dementia, dehydration, and '
                   'infection. Creatinine is 2.7 mg/dL from 2.1; he is agitated and inattentive.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': '38-year-old man, ASA 1, for outpatient hydrocelectomy. He is healthy and active, with normal pulmonary and '
                   'renal function, no VTE history, and intact cognition.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '71-year-old woman, ASA 3, for shoulder arthroplasty. She has obesity and rheumatoid arthritis but remains '
                   'mobile. No prior thrombosis, lung disease, kidney disease, or cognitive impairment is present.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '64-year-old man, ASA 3, for total gastrectomy. He has COPD, active smoking, weak cough, and albumin 2.7 '
                   'g/dL. Oxygen saturation is 92%. Renal function and cognition are normal.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '77-year-old woman, ASA 4, for mitral valve surgery. She has CKD stage 3b, diabetes, frailty, and creatinine '
                   '1.9 mg/dL. She is mildly forgetful. Lungs are clear without infection.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': '47-year-old woman, ASA 2, for laparoscopic bowel resection. She has controlled inflammatory bowel disease, '
                   'good nutrition, and no pulmonary, renal, thrombotic, or cognitive disease.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '66-year-old man, ASA 3, for pelvic tumor resection. He has active cancer, iliac-vein compression, chronic '
                   'leg edema, and a prior DVT. Apixaban is held. Lungs and kidneys are normal.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '75-year-old woman, ASA 3, for VATS wedge resection. She has pulmonary fibrosis, exertional oxygen use, and '
                   'room-air saturation 90%. A dry cough has worsened. Creatinine and cognition are normal.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '84-year-old man, ASA 3, for elective hernia repair. He has mild Alzheimer disease, severe hearing '
                   'impairment, and prior postoperative delirium. His spouse manages medications. Lungs and kidneys are normal.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': '59-year-old woman, ASA 3, for partial nephrectomy. She has CKD stage 3a, diabetes, and creatinine 1.5 mg/dL. '
                   'Prolonged hilar clamping is expected. She is active and oriented.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': '73-year-old man, ASA 3, for hindfoot fusion. He has Factor V Leiden, prior DVT, obesity, and prolonged '
                   'limited weight-bearing. Rivaroxaban is held. Lungs and kidneys are normal.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '65-year-old woman, ASA 4, for thoracoabdominal aneurysm repair. She has severe COPD, active smoking, CKD '
                   'stage 3a, oxygen saturation 91%, and creatinine 1.4 mg/dL. Cognition is intact.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': '89-year-old woman, ASA 4, for urgent ureteral stenting for urosepsis. She has dementia, CKD stage 3b, '
                   'hypotension, and poor intake. Creatinine is 2.4 mg/dL from 1.6; she is acutely confused.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': '36-year-old woman, ASA 1, for elective breast augmentation. She is healthy, active, and has normal '
                   'pulmonary, renal, thrombotic, and cognitive status.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '64-year-old man, ASA 3, for hemicolectomy during chemotherapy. He had a port-associated DVT four months ago '
                   'and takes apixaban, now held. Fatigue limits activity. Lungs and kidneys are normal.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '70-year-old woman, ASA 3, for open nephrectomy. She has COPD, active smoking, and a recent steroid-treated '
                   'exacerbation. Oxygen saturation is 92% with wheeze. Creatinine is 0.9 mg/dL.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '81-year-old man, ASA 4, for emergency colectomy. He has CKD stage 3b, chronic steroids, frailty, and mild '
                   'dementia. He is hypotensive, dehydrated, and confused; creatinine is 2.2 mg/dL from 1.5.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': '54-year-old woman, ASA 2, for robotic ventral hernia repair. She has controlled hyperlipidemia and otherwise '
                   'good health. She walks daily and has normal lungs, kidneys, mobility, and cognition.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': '74-year-old man, ASA 3, for pelvic-fracture fixation. He has been bedbound for nine days, has active '
                   'lymphoma, and previously had a pulmonary embolism. Enoxaparin is held. Lungs and renal function are normal.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 73-year-old woman is having VATS resection of a pleural tumor. She has poorly controlled asthma, chronic '
                   'mucus plugging, and an admission for pneumonia last month. Room-air saturation is 92%, with reduced breath '
                   'sounds. Renal function and baseline cognition are normal.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'An 86-year-old retired teacher is scheduled for transurethral bladder surgery. She has moderate hearing '
                   'loss, early Alzheimer disease, and takes nightly temazepam. Her son reports that unfamiliar settings make '
                   'her confused. Lungs are clear and creatinine is 0.8 mg/dL.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': 'A 62-year-old man will undergo open repair of a renal-artery stenosis. He has diabetic CKD stage 3b, '
                   'creatinine 1.9 mg/dL, and received contrast twice this week. He is euvolemic, alert, and has no respiratory '
                   'symptoms.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': 'A 45-year-old woman presents for elective excision of a benign breast mass. She has no chronic illness, '
                   'takes no daily medication, and maintains excellent exercise tolerance. Examination, creatinine, oxygenation, '
                   'cognition, and VTE history are unremarkable.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 57-year-old man with ASA class 2 is scheduled for laparoscopic repair of a recurrent inguinal hernia. '
                   'Aside from treated hypertension, he is healthy, walks several miles without limitation, and has normal renal '
                   'function, oxygenation, cognition, and no thrombosis history.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'Preoperative evaluation of a 79-year-old woman for redo sternotomy and aortic-root replacement. She has '
                   'diabetic CKD stage 4, creatinine 2.2 mg/dL, EF 35%, and frailty. Her memory is mildly impaired, although she '
                   'remains oriented. Chest examination is clear.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': 'A 63-year-old man will undergo open resection of a pelvic chondrosarcoma. The mass compresses the left iliac '
                   'vein; he had ipsilateral DVT six months ago and remains on apixaban, now stopped. He walks with difficulty. '
                   'Kidney and lung function are preserved.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 68-year-old woman is planned for composite resection of an oral cancer with free-flap reconstruction. '
                   'Prior radiation left severe dysphagia and recurrent aspiration. She has COPD, copious secretions, and '
                   'room-air saturation of 92%. Creatinine and baseline cognition are normal.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'An 86-year-old man presents for urgent drainage of an intra-abdominal abscess and bowel repair. He has '
                   'dementia, CKD stage 3b, poor intake, and sepsis-associated hypotension. Creatinine rose from 1.5 to 2.2 '
                   'mg/dL. He is inattentive and does not recognize the hospital.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': 'A healthy 34-year-old woman, ASA 1, is scheduled for outpatient arthroscopic shoulder stabilization. She '
                   'takes no regular medication, is physically active, and has no pulmonary, renal, thrombotic, or cognitive '
                   'risk factors.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 72-year-old woman is undergoing revision total knee arthroplasty. She had an unprovoked DVT two years ago, '
                   'has BMI 38, and walks only within the home. Rivaroxaban is withheld for surgery. Lungs are clear and '
                   'creatinine is 0.9 mg/dL.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 75-year-old man with severe COPD is scheduled for right upper lobectomy. He uses oxygen during exertion, '
                   'continues to smoke, and has FEV1 42% predicted with daily sputum. Room-air saturation is 90%. Renal function '
                   'and cognition are preserved.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'An 82-year-old woman is scheduled for elective parotidectomy. She has mild Alzheimer disease, profound '
                   'hearing loss, and takes amitriptyline for neuropathy. Her daughter manages medications. She is oriented to '
                   'person and place; lungs and kidneys are normal.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': 'A 61-year-old man is planned for partial nephrectomy of a central tumor. Diabetic CKD stage 3b is present '
                   'with creatinine 1.8 mg/dL, and prolonged renal ischemia is expected. He is euvolemic, alert, and free of '
                   'respiratory disease.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': 'A 43-year-old man, ASA 2, is having laparoscopic sleeve gastrectomy. BMI is 47 and OSA is treated with CPAP. '
                   'He remains independently mobile, has no prior VTE, and has normal creatinine, lungs, and cognition.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'An 84-year-old woman requires emergency fixation of an intertrochanteric fracture. She has dementia, chronic '
                   'dysphagia, and has lain in bed for three days. She coughs after drinking and needs 2 L oxygen. Creatinine is '
                   'stable at 1.1 mg/dL.',
  'DVT': 1,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': 'A 67-year-old man is scheduled for pancreatic cancer resection. He developed a proximal DVT during '
                   'chemotherapy and has lost weight with reduced activity. Apixaban is stopped. Lungs are clear, creatinine is '
                   '0.9 mg/dL, and mental status is normal.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 70-year-old woman is undergoing open repair of a paraesophageal hernia. She reports frequent nocturnal '
                   'regurgitation and two admissions for aspiration pneumonia. She has bronchiectasis, weak cough, and '
                   'saturation 92%. Kidney function and cognition are intact.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 78-year-old man is scheduled for open abdominal aortic aneurysm repair. CKD stage 3b, diabetes, and '
                   'diffuse vascular disease are present; creatinine is 1.8 mg/dL. He remains cognitively intact and has no '
                   'chronic lung disease.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': 'A 46-year-old woman, ASA 1, will undergo laparoscopic ovarian cystectomy. She is healthy, exercises '
                   'regularly, takes no daily medication, and has normal mobility, lungs, kidneys, and cognition without prior '
                   'thrombosis.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'An 80-year-old man is planned for elective lumbar decompression. He has mild vascular dementia, severe '
                   'hearing impairment, and previously became confused after anesthesia. He is oriented today and lives with his '
                   'wife. Renal and pulmonary status are stable.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': 'A 69-year-old woman is scheduled for open colectomy for metastatic cancer. She has an indwelling port, a '
                   'prior pulmonary embolism, unilateral leg edema, and declining mobility. Anticoagulation is held. Creatinine '
                   'and pulmonary examination are normal.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 64-year-old man with COPD is undergoing esophagectomy. He continues to smoke, has chronic sputum and '
                   'documented aspiration, and saturates 91% on room air. Baseline creatinine is 0.8 mg/dL and he is cognitively '
                   'intact.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'An 87-year-old woman requires urgent laparotomy for an incarcerated hernia. She has CKD stage 3b, dementia, '
                   'vomiting, and dehydration. Creatinine is 2.0 mg/dL from 1.4, and she alternates between somnolence and '
                   'agitation.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': 'A 52-year-old man is scheduled for elective robotic prostatectomy. He has well-controlled hypertension, runs '
                   'three times weekly, and has no renal, pulmonary, thrombotic, or cognitive disease. Creatinine is 0.9 mg/dL.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 74-year-old woman will undergo open reduction of a pelvic fracture. She has been bedbound for eleven days, '
                   'has active ovarian cancer, and previously had DVT. Enoxaparin is held. Lungs and kidneys are normal; '
                   'cognition is intact.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 71-year-old man is scheduled for VATS lobectomy. He has bronchiectasis, recurrent infections, daily '
                   'purulent sputum, and oxygen saturation 92%. He completed antibiotics two weeks ago. Creatinine and cognition '
                   'are normal.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'An 85-year-old woman presents for elective total shoulder replacement. Mild dementia, poor vision, and '
                   'chronic lorazepam use are documented. She needs help with medications but is calm and oriented to person and '
                   'place. Lungs and kidneys are normal.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': 'A 60-year-old man is planned for nephroureterectomy. The remaining kidney has CKD stage 3b with creatinine '
                   '1.7 mg/dL. Diabetes and hypertension are controlled. He is alert, ambulatory, and has no lung disease or VTE '
                   'history.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': 'A 49-year-old woman is scheduled for outpatient bunion correction. She is healthy, weight-bearing, and has '
                   'no history of thrombosis, pulmonary disease, kidney impairment, or cognitive symptoms. Preoperative '
                   'examination is normal.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 66-year-old man will undergo open liver resection for metastatic colon cancer. He has prior '
                   'chemotherapy-associated DVT, obesity, and reduced activity. Apixaban is stopped. Lungs are clear and '
                   'creatinine is 0.9 mg/dL.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 73-year-old woman is planned for open lung resection. She has severe emphysema, home oxygen at night, '
                   'continued smoking, and a weak productive cough. Oxygen saturation is 89%. Kidney function and cognition are '
                   'normal.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'An 81-year-old man is undergoing coronary bypass surgery. He has CKD stage 4, diabetes, anemia, and '
                   'creatinine 2.4 mg/dL. He is frail and mildly forgetful but oriented. Chest examination shows no infection.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': 'A 38-year-old woman, ASA 1, is scheduled for laparoscopic salpingectomy. She has no chronic disease, normal '
                   'exercise tolerance, and no prior anesthesia, thrombotic, renal, pulmonary, or cognitive problems.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 70-year-old man is planned for revision hip arthroplasty. He has Factor V Leiden, two previous DVTs, '
                   'obesity, and very limited walking. Warfarin is interrupted with bridging. Lungs and renal function are '
                   'stable.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 67-year-old woman is scheduled for gastrectomy. Severe COPD, active tobacco use, chronic sputum, '
                   'malnutrition, and oxygen saturation of 92% are noted. Creatinine is 0.8 mg/dL and baseline cognition is '
                   'normal.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'An 88-year-old man requires emergency bowel resection for volvulus. He has dementia, CKD stage 3b, and two '
                   'days of vomiting. Blood pressure is low; creatinine increased from 1.5 to 2.2 mg/dL. He is acutely '
                   'disoriented.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': 'A 56-year-old woman, ASA 2, is scheduled for laparoscopic fundoplication. She has controlled reflux without '
                   'aspiration, normal lungs and kidneys, good exercise tolerance, no VTE history, and intact cognition.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 76-year-old man is scheduled for tibial-plateau fixation. He has been non-weight-bearing for two weeks, '
                   'has a prior PE, and is receiving chemotherapy for prostate cancer. Anticoagulation is held. Lungs and '
                   'kidneys are normal.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 69-year-old woman will undergo open pancreatic surgery. She has moderate COPD, a recent exacerbation, '
                   'daily sputum, and oxygen saturation 93%. She stopped smoking one week ago. Renal function and cognition are '
                   'normal.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'An 83-year-old woman is planned for elective colectomy. She has mild Alzheimer disease, severe hearing loss, '
                   'and a prior delirium episode after pneumonia. Her daughter assists with medications. Lungs and kidneys are '
                   'currently stable.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': 'A 62-year-old man is scheduled for open renal-artery aneurysm repair. He has CKD stage 3a, diabetes, and '
                   'recent contrast angiography; creatinine is 1.5 mg/dL. He is euvolemic and cognitively normal.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': 'A 44-year-old man, ASA 2, presents for acute Achilles repair. He is otherwise healthy and mobile with '
                   'crutches. There is no personal or family VTE history, and pulmonary, renal, and cognitive status are normal.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 77-year-old woman is scheduled for VATS decortication. She has chronic bronchiectasis, persistent '
                   'productive cough, recent empyema, and oxygen saturation of 91%. Creatinine is 0.8 mg/dL and she is oriented.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 71-year-old man with ASA 3 is undergoing total knee arthroplasty. Obesity and chronic venous insufficiency '
                   'are present, but he has no prior VTE and remains active with a cane. Lungs, kidneys, and cognition are '
                   'normal.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'An 80-year-old woman is scheduled for aortic arch replacement. She has CKD stage 4, diabetes, prior stroke, '
                   'and mild vascular cognitive impairment. Creatinine is 2.3 mg/dL. No active pulmonary disease is present.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': 'A 63-year-old man will undergo pancreaticoduodenectomy. He had DVT during prior chemotherapy, has obesity, '
                   'and is less mobile from abdominal pain. Apixaban is held. Lungs and renal function are normal.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 70-year-old woman is planned for open esophagectomy. She has severe reflux with aspiration, COPD, current '
                   'smoking, and oxygen saturation 92%. Her productive cough has recently increased. Kidney function and '
                   'cognition are normal.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'An 89-year-old man requires emergency repair of a strangulated ventral hernia. He has CKD stage 3b, '
                   'dementia, vomiting, and hypotension. Creatinine is 2.1 mg/dL from 1.5, and he is agitated and disoriented.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': 'A 41-year-old woman, ASA 1, is scheduled for arthroscopic meniscus repair. She is healthy, exercises '
                   'regularly, and has normal mobility, pulmonary and renal function, cognition, and no VTE history.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 69-year-old woman is scheduled for open ovarian-cancer debulking. She has malignant ascites, prior DVT, '
                   'obesity, and limited mobility from abdominal distention. Enoxaparin is held. Lungs are clear, kidney '
                   'function is normal, and cognition is intact.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 74-year-old man is undergoing left upper lobectomy. Severe COPD, chronic productive cough, and FEV1 43% '
                   'predicted are documented. Oxygen saturation is 91%. Creatinine is normal and he is cognitively intact.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'An 85-year-old woman is scheduled for elective bowel resection. She has mild Alzheimer disease, poor vision, '
                   'and prior postoperative confusion. Her son manages medications. Lungs and renal function are normal.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': 'A 60-year-old man will undergo nephroureterectomy. CKD stage 3b, diabetes, and creatinine 1.8 mg/dL are '
                   'present; the remaining kidney has reduced function. He is active, oriented, and has clear lungs.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': 'A 53-year-old woman, ASA 2, is scheduled for laparoscopic repair of a small incisional hernia. She has '
                   'treated hypothyroidism, walks daily, and has no pulmonary disease, renal impairment, VTE history, or '
                   'cognitive concerns.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 78-year-old man will undergo open repair of a suprarenal aneurysm. He has diabetic CKD stage 4, creatinine '
                   '2.2 mg/dL, and diffuse vascular disease. He is independent and cognitively intact, with no active '
                   'respiratory illness.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': 'A 65-year-old woman is planned for revision hip arthroplasty. She has antiphospholipid syndrome, two prior '
                   'DVTs, obesity, and poor mobility. Warfarin is held with bridging. Pulmonary and renal status are stable.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 73-year-old man is scheduled for open esophagectomy. He has severe COPD, chronic aspiration, active '
                   'smoking, and room-air saturation of 90%. He uses inhalers daily and has a weak productive cough. Renal '
                   'function and cognition are normal.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'An 87-year-old woman requires urgent repair of a femur fracture. She has advanced dementia, CKD stage 3a, '
                   'dehydration, and four days of immobility. Creatinine is 1.4 mg/dL from 1.0, and she is restless and '
                   'disoriented.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': 'A 30-year-old man, ASA 1, is scheduled for elective laparoscopic hernia repair. He has no medical '
                   'conditions, takes no regular drugs, and has normal functional capacity, lungs, kidneys, cognition, and no '
                   'VTE history.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 64-year-old woman is undergoing ureteral reimplantation. She has a solitary kidney with CKD stage 3b and '
                   'creatinine 1.7 mg/dL. Diabetes is controlled. She is euvolemic, active, and cognitively intact with clear '
                   'lungs.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': 'A 72-year-old man is scheduled for total knee arthroplasty. Prior unprovoked DVT, obesity, and severe '
                   'mobility limitation are present. Apixaban is held. He has normal pulmonary and renal function and intact '
                   'cognition.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'An 81-year-old woman is planned for elective thyroidectomy. Mild Alzheimer disease, hearing loss, and '
                   'chronic diphenhydramine use are documented. She is oriented to person and place. Lungs and renal function '
                   'are normal.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': 'A 67-year-old man is scheduled for right lower lobectomy. He has severe emphysema, home oxygen with '
                   'exertion, active smoking, and oxygen saturation 89%. Creatinine is 0.9 mg/dL and cognition is intact.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 45-year-old woman, ASA 2, is having robotic partial nephrectomy. Hypertension is well controlled, '
                   'creatinine is 0.8 mg/dL, and she has normal pulmonary function, mobility, cognition, and no VTE history.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'An 84-year-old man requires emergency laparotomy for perforated diverticulitis. He has CKD stage 3b, '
                   'dementia, poor intake, and sepsis. Creatinine is 2.0 mg/dL from 1.4, and he is hypotensive and acutely '
                   'confused.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': 'A 69-year-old woman is planned for pelvic cancer resection. She has prior PE, iliac-vein compression, and '
                   'reduced mobility. Rivaroxaban is held. Lungs are clear, renal function is normal, and cognition is intact.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 61-year-old man is scheduled for total laryngectomy. He has chronic aspiration, thick secretions, COPD, '
                   'and recent bronchitis. Oxygen saturation is 92%. Kidney function and cognition are normal.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 76-year-old woman is undergoing lumbar fusion. She has CKD stage 3b, diabetes, chronic NSAID exposure, and '
                   'creatinine 1.8 mg/dL. She is hydrated and oriented, with clear lungs.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': 'A 39-year-old man, ASA 1, is scheduled for outpatient excision of a benign back mass. He is healthy, active, '
                   'and has normal pulmonary, renal, thrombotic, and cognitive status.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'An 89-year-old woman requires operative repair of a displaced hip fracture. She has vascular dementia, a '
                   'previous postoperative DVT, and has remained on bed rest since falling four days ago. She cannot state the '
                   'year and needs full assistance. Lungs are clear; creatinine is 1.0 mg/dL.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': 'A 62-year-old man is planned for abdominal wall reconstruction. Obesity, active cancer, previous DVT, and '
                   'poor mobility are present. Anticoagulation is stopped. Lungs and renal function are normal; cognition is '
                   'intact.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 68-year-old man is scheduled for pneumonectomy after induction chemotherapy. He has severe COPD, chronic '
                   'green sputum, and uses oxygen at night. Saturation is 90% on room air and FEV1 is 39% predicted. Renal '
                   'function and cognition are preserved.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'An 82-year-old woman presents for emergency drainage of an infected obstructed kidney. Baseline CKD stage '
                   '3b, dementia, fever, and hypotension are present. Creatinine increased from 1.6 to 2.4 mg/dL. She is '
                   'hallucinating and repeatedly tries to leave the bed.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': 'A 50-year-old woman, ASA 2, is scheduled for laparoscopic hysterectomy. She has controlled hypertension, '
                   'good exercise tolerance, normal lungs and kidneys, no prior thrombosis, and intact cognition.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 68-year-old man is undergoing radical cystectomy. He has active bladder cancer, prior DVT during '
                   'chemotherapy, and reduced mobility. Apixaban is held. Lungs clear, creatinine 1.0 mg/dL, cognition normal.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'An 80-year-old man is planned for elective carotid endarterectomy. He has mild vascular cognitive '
                   'impairment, profound hearing loss, and a prior delirium episode after pneumonia. His wife manages '
                   'medications. Pulmonary and renal examinations are stable.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': 'A 58-year-old woman will undergo open radical nephrectomy. She has a solitary contralateral kidney with '
                   'diabetic CKD stage 3a, creatinine 1.5 mg/dL, and recent dehydration from nausea. She is oriented and has no '
                   'chronic pulmonary disease.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': 'A 70-year-old man is scheduled for complex acetabular reconstruction. He has metastatic prostate cancer, a '
                   'previous pulmonary embolism, and has been wheelchair-bound for three weeks. Therapeutic enoxaparin is held. '
                   'Lungs are clear and creatinine is 0.9 mg/dL.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 69-year-old woman is undergoing open repair of a large diaphragmatic hernia. She has recurrent aspiration, '
                   'weak cough, and restrictive lung disease after prior thoracic radiation. Oxygen saturation is 91%. Renal '
                   'function and cognition are intact.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'An 83-year-old man requires urgent bowel resection for necrotic intestine. He has CKD stage 3b, heart '
                   'failure, mild dementia, and shock requiring vasopressors. Creatinine is 2.3 mg/dL from 1.5. He is '
                   'inattentive and unable to follow commands.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': 'A 33-year-old woman, ASA 1, is scheduled for laparoscopic treatment of endometriosis. She has no chronic '
                   'disease, normal exercise tolerance, and no pulmonary, renal, thrombotic, or cognitive concerns.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A healthy 37-year-old man, ASA 1, is scheduled for outpatient arthroscopic elbow surgery. He takes no '
                   'regular medication, plays recreational soccer, and has normal cardiopulmonary, renal, cognitive, and '
                   'thrombotic history.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 71-year-old woman is planned for open pancreatic resection. She has severe COPD, persistent wheezing, and '
                   'a hospitalization for an exacerbation three weeks ago. Room-air saturation is 92%. Kidney function and '
                   'mental status are normal.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'An 81-year-old woman is undergoing open aortic-valve replacement. She has CKD stage 4, diabetes, frailty, '
                   'and baseline creatinine 2.3 mg/dL. Mild cognitive impairment affects medication management. No current '
                   'respiratory infection is present.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': 'A 48-year-old woman, ASA 2, is planned for laparoscopic bowel resection for endometriosis. She is active, '
                   'has normal lungs and renal function, no prior thrombosis, and intact cognition.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 65-year-old man will undergo resection of a retroperitoneal liposarcoma. The mass narrows the inferior '
                   'vena cava, and he had a proximal DVT while receiving chemotherapy. Apixaban is stopped. He is ambulatory '
                   'with normal lungs and renal function.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 72-year-old woman is scheduled for surgical lung biopsy. She has fibrotic interstitial lung disease, uses '
                   '3 L oxygen when walking, and reports a recent increase in dry cough. Resting saturation is 89%. Creatinine '
                   'and cognition are normal.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'An 84-year-old man is planned for elective colostomy reversal. He has mild Alzheimer disease, severe hearing '
                   'loss, and a previous postoperative delirium episode. His wife manages medications. Lungs and renal function '
                   'are stable.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': 'A 57-year-old man is planned for partial nephrectomy of a hilar lesion. He has CKD stage 3a, proteinuria, '
                   'and baseline creatinine 1.6 mg/dL; prolonged arterial clamping is anticipated. He remains active and fully '
                   'oriented.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': 'A 74-year-old woman is having ankle fusion after months in a walking boot. She has protein S deficiency, two '
                   'previous DVTs, obesity, and very limited weight-bearing. Rivaroxaban is withheld. Pulmonary and renal status '
                   'are stable.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 63-year-old man is scheduled for open descending aortic repair. Severe emphysema, active smoking, CKD '
                   'stage 3a, saturation 90%, and creatinine 1.5 mg/dL are documented. He remains cognitively intact.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': 'An 88-year-old woman requires urgent debridement for necrotizing soft-tissue infection. She has dementia, '
                   'CKD stage 3b, poor intake, and septic hypotension. Creatinine has risen from 1.5 to 2.4 mg/dL. She is '
                   'lethargic and disoriented.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': 'A 37-year-old woman, ASA 1, is scheduled for elective breast reduction. She is healthy and active, with '
                   'normal lungs, renal function, cognition, mobility, and no VTE history.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 67-year-old woman will undergo colectomy for recurrent colon cancer. She developed an upper-extremity DVT '
                   'around her port six weeks ago and remains on apixaban, now interrupted. Cancer-related fatigue has reduced '
                   'activity. Lungs and kidneys are normal.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 66-year-old man is planned for open adrenalectomy. He has severe COPD, uses oxygen overnight, and recently '
                   'completed antibiotics and prednisone for an exacerbation. Oxygen saturation is 91% with diffuse wheeze. '
                   'Creatinine is 0.9 mg/dL.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'An 80-year-old woman presents for emergency laparotomy after bowel perforation. She has chronic steroid '
                   'exposure, CKD stage 3b, frailty, and mild dementia. Blood pressure is low and creatinine is 2.1 mg/dL from '
                   '1.4. She is acutely confused.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': 'A 48-year-old man is scheduled for robotic repair of a small ventral hernia. His only condition is '
                   'controlled hyperlipidemia. He cycles regularly and has normal lungs, kidney function, cognition, and no '
                   'previous thrombosis.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 76-year-old woman is having fixation of a sacral fracture. She has active lymphoma, a prior PE, and has '
                   'been confined to bed for twelve days. Enoxaparin is held for surgery. Pulmonary and renal function are '
                   'normal.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 64-year-old man will undergo total gastrectomy. He has chronic bronchitis, current tobacco use, daily '
                   'sputum, weak cough, and oxygen saturation of 92%. Albumin is 2.8 g/dL. Kidney function and cognition are '
                   'normal.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'An 84-year-old former accountant is scheduled for knee arthroplasty. He has mild Alzheimer disease, macular '
                   'degeneration, and takes clonazepam nightly. His daughter organizes medications. He is oriented to person and '
                   'place; lungs and kidneys are stable.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': 'A 59-year-old woman is planned for open reconstruction of the renal artery. CKD stage 3a, hypertension, and '
                   'two recent angiograms are noted; creatinine is 1.5 mg/dL. She is well hydrated and cognitively intact.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': 'A 43-year-old man is having repair of an acute Achilles rupture. He is otherwise healthy, has remained '
                   'mobile on crutches, and has no VTE history or abnormalities of lungs, kidneys, or cognition.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 77-year-old woman is scheduled for thoracoscopic decortication of a chronic empyema. She has '
                   'bronchiectasis, daily purulent sputum, and oxygen saturation of 91% despite recent antibiotics. Creatinine '
                   'is normal and she is oriented.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 56-year-old woman, ASA 2, is scheduled for elective laparoscopic colectomy for a benign polyp. She has '
                   'controlled hypertension, walks daily, and has normal pulmonary and renal function, intact cognition, and no '
                   'VTE history.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 79-year-old man is planned for open repair of a pararenal aneurysm. He has CKD stage 4, diabetes, and '
                   'extensive vascular disease; creatinine is 2.3 mg/dL. He is oriented and independent, with no current '
                   'pulmonary infection.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': 'A 66-year-old woman is undergoing revision knee arthroplasty. She has antiphospholipid syndrome, two prior '
                   'DVTs, obesity, and restricted walking. Warfarin is held with bridging. Lungs and kidneys are stable.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 74-year-old man is scheduled for open esophagectomy. Severe COPD, chronic aspiration, continued smoking, '
                   'and oxygen saturation of 90% are documented. He has daily sputum. Renal function and cognition are normal.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'An 88-year-old woman requires urgent repair of a periprosthetic femur fracture. She has dementia, CKD stage '
                   '3a, dehydration, and five days of immobility. Creatinine is 1.5 mg/dL from 1.1, and she is restless and '
                   'disoriented.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': 'A 32-year-old man, ASA 1, is scheduled for elective laparoscopic hernia repair. He has no chronic '
                   'conditions, takes no regular medication, and has normal mobility, lungs, kidney function, cognition, and no '
                   'thrombosis history.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 65-year-old woman is planned for ureteral reconstruction in a solitary kidney. CKD stage 3b is present '
                   'with creatinine 1.8 mg/dL. She is hydrated, active, and oriented, with no pulmonary disease or VTE history.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': 'A 73-year-old man is scheduled for total hip arthroplasty. Prior unprovoked DVT, BMI 39, and severe mobility '
                   'limitation are present. Apixaban is held. Pulmonary and renal function are normal; cognition is intact.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'An 82-year-old woman is scheduled for elective parathyroid surgery. She has mild Alzheimer disease, severe '
                   'hearing loss, and chronic anticholinergic medication use. She is oriented to person and place. Lungs and '
                   'kidneys are normal.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': 'A 65-year-old woman is planned for right pneumonectomy. She has severe emphysema, chronic productive cough, '
                   'and uses oxygen with activity. Pulmonary rehabilitation was limited by dyspnea; resting saturation is 90%. '
                   'Kidney function and cognition are intact.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 47-year-old woman, ASA 2, is scheduled for robotic partial nephrectomy. Hypertension is controlled, '
                   'creatinine is 0.8 mg/dL, and pulmonary function, mobility, cognition, and thrombosis history are '
                   'unremarkable.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'An 85-year-old man requires emergency repair of a perforated gastric ulcer. CKD stage 3b, dementia, '
                   'dehydration, and septic hypotension are present. Creatinine is 2.2 mg/dL from 1.5. He is combative and does '
                   'not recognize family.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': 'A 68-year-old woman will undergo pelvic exenteration for recurrent cervical cancer. She has iliac-vein '
                   'encasement, a prior DVT, unilateral edema, and limited walking. Apixaban is held. Pulmonary and renal '
                   'function are normal.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 62-year-old man is scheduled for laryngectomy. Chronic aspiration, thick secretions, COPD, and recent '
                   'bronchitis are present. Oxygen saturation is 92%. Kidney function and cognition are normal.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 75-year-old woman is undergoing posterior lumbar fusion. She has CKD stage 3b, diabetes, chronic NSAID '
                   'use, and creatinine 1.8 mg/dL. She is hydrated and oriented, with clear lungs.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': 'A 40-year-old man, ASA 1, is scheduled for outpatient excision of a benign thigh mass. He is healthy and '
                   'active, with normal pulmonary, renal, cognitive, and thrombotic history.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'An 87-year-old woman is scheduled for fixation of a displaced proximal femur fracture. She has dementia, a '
                   'remote DVT, and has been immobile for three days. She is disoriented and needs complete assistance. Lungs '
                   'are clear; creatinine is 1.0 mg/dL.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': 'A 61-year-old man is having abdominal wall reconstruction after cancer surgery. He has severe obesity, a '
                   'previous PE, and spends most of the day seated because of pain. Anticoagulation is stopped. Lungs and '
                   'kidneys are stable.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 73-year-old woman is planned for right lower lobectomy. She has bronchiectasis, chronic Pseudomonas '
                   'infection, daily sputum, and oxygen saturation 91%. She completed antibiotics last week. Renal function and '
                   'cognition are normal.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'An 82-year-old man will undergo combined mitral-valve replacement and bypass surgery. He has CKD stage 4, '
                   'diabetes, frailty, creatinine 2.5 mg/dL, and mild cognitive impairment. There is no active pulmonary '
                   'infection.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': 'A 51-year-old woman, ASA 2, is scheduled for laparoscopic hysterectomy. She has controlled hypertension, '
                   'good exercise tolerance, normal lungs and kidneys, no prior thrombosis, and intact cognition.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 69-year-old woman is scheduled for radical cystectomy. She had a proximal DVT during neoadjuvant '
                   'chemotherapy and now walks only short distances. Apixaban is withheld. Lungs are clear, creatinine 1.0 '
                   'mg/dL, and cognition is normal.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'An 84-year-old man presents for urgent gallbladder surgery. He has mild dementia, severe hearing loss, and '
                   'prior delirium during infection. Fever and disorientation are present, but lungs are clear and creatinine is '
                   '1.0 mg/dL.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': 'A 58-year-old woman will undergo distal pancreatectomy. Diabetic CKD stage 3a and several days of vomiting '
                   'are noted; creatinine is 1.7 mg/dL from 1.3. She is alert and has no chronic respiratory illness.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': 'A 66-year-old man is scheduled for decompression of metastatic spinal disease. Progressive weakness has left '
                   'him wheelchair-bound for five weeks, and he previously had PE during chemotherapy. Anticoagulation is held. '
                   'Renal and pulmonary function are stable.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 74-year-old woman is planned for repair of a giant paraesophageal hernia. She has repeated nocturnal '
                   'aspiration, bronchiectasis, and two recent pneumonias. Oxygen saturation is 92% and cough is weak. Kidney '
                   'function and cognition are normal.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'An 88-year-old man requires below-knee amputation for a septic diabetic foot. He has CKD stage 4, dementia, '
                   'and poor intake. Creatinine is 2.8 mg/dL from 2.2. He is inattentive, agitated, and unable to answer '
                   'questions.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': 'A 34-year-old man, ASA 1, is scheduled for laparoscopic treatment of an umbilical hernia. He has no chronic '
                   'disease, normal exercise tolerance, and no pulmonary, renal, thrombotic, or cognitive concerns.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 70-year-old man is having elective reverse shoulder arthroplasty. Obesity and rheumatoid arthritis are '
                   'present, but he remains independently mobile and has no previous VTE. Lungs, kidneys, and cognitive testing '
                   'are normal.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 64-year-old man is scheduled for total gastrectomy. COPD, active smoking, weak cough, and albumin 2.7 g/dL '
                   'are present. Oxygen saturation is 92%. Kidney function and cognition are normal.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 78-year-old woman is undergoing open mitral valve surgery. She has CKD stage 3b, diabetes, frailty, and '
                   'creatinine 1.9 mg/dL. Mild forgetfulness is noted. Lungs are clear.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': 'A 49-year-old man, ASA 2, is planned for laparoscopic bowel resection for Crohn disease. He is active, has '
                   'normal lungs and renal function, no prior thrombosis, and intact cognition.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 67-year-old woman will undergo en-bloc resection of a pelvic sarcoma. The tumor compresses the iliac '
                   'vessels; she has chronic leg swelling and a prior DVT. Apixaban is stopped. Pulmonary and renal function are '
                   'normal.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 76-year-old man is scheduled for VATS wedge biopsy. He has idiopathic pulmonary fibrosis, requires oxygen '
                   'for routine walking, and reports worsening cough. Resting saturation is 90%. Creatinine and cognition are '
                   'normal.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'An 85-year-old woman is planned for elective colostomy reversal. She has mild Alzheimer disease, severe '
                   'hearing loss, and a previous postoperative delirium episode. Her daughter manages medications. Lungs and '
                   'renal function are stable.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': 'A 61-year-old man is undergoing partial nephrectomy. He has CKD stage 3a, diabetes, and creatinine 1.5 '
                   'mg/dL; prolonged hilar clamping is anticipated. He is active and oriented, with clear lungs.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': 'A 73-year-old woman will undergo hindfoot arthrodesis. She has Factor V Leiden, prior DVT, BMI 38, and has '
                   'been minimally weight-bearing for six weeks. Rivaroxaban is held. Lungs and kidneys are stable.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 66-year-old man is planned for thoracoabdominal aneurysm repair. Severe COPD, active smoking, CKD stage '
                   '3a, oxygen saturation 91%, and creatinine 1.4 mg/dL are documented. Cognition is intact.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': 'A 90-year-old woman requires urgent ureteral stenting for urosepsis. She has dementia, CKD stage 3b, '
                   'hypotension, and poor intake. Creatinine is 2.5 mg/dL from 1.7, and she is acutely agitated.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': 'A 38-year-old woman, ASA 1, is scheduled for elective breast reduction. She is healthy and active, with '
                   'normal lungs, renal function, cognition, mobility, and no VTE history.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 65-year-old woman is scheduled for colectomy while receiving chemotherapy. A port-associated DVT occurred '
                   'two months ago, and apixaban is now interrupted. She has marked fatigue and reduced activity. Pulmonary and '
                   'renal function are normal.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 71-year-old man will undergo open nephrectomy. He has severe COPD, continues to smoke, and was treated '
                   'with prednisone for an exacerbation last week. Saturation is 91% with wheeze. Creatinine is 0.9 mg/dL.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'An 83-year-old woman requires emergency colectomy for ischemic bowel. She has CKD stage 3b, chronic '
                   'steroids, frailty, and mild dementia. She is hypotensive and confused; creatinine is 2.3 mg/dL from 1.5.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': 'A 50-year-old woman is scheduled for robotic ventral hernia repair. She has controlled dyslipidemia, walks '
                   'daily, and has normal pulmonary and renal function, intact cognition, and no VTE history.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 75-year-old man is having fixation of a complex pelvic fracture. He has active lymphoma, a prior PE, and '
                   'has been bedbound for ten days. Enoxaparin is held. Lungs and kidneys are normal.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 68-year-old woman is planned for pancreaticoduodenectomy. She has chronic bronchitis, current smoking, '
                   'daily sputum, and saturation 93%. Albumin is 2.9 g/dL. Renal function and cognition are normal.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'An 86-year-old man is scheduled for elective hip arthroplasty. Mild dementia, severe hearing impairment, and '
                   'chronic benzodiazepine use are present. He is oriented to person and place. Lungs and kidneys are stable.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': 'A 60-year-old woman will undergo open renal-artery repair. CKD stage 3a, hypertension, and recent contrast '
                   'exposure are present; creatinine is 1.6 mg/dL. She is hydrated and fully oriented.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': 'A 47-year-old man is having Achilles repair after an acute sports injury. He has remained mobile with '
                   'crutches and has no personal or family VTE history. Pulmonary, renal, and cognitive assessments are normal.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 78-year-old woman is scheduled for VATS decortication after unresolved empyema. Bronchiectasis, persistent '
                   'purulent sputum, and oxygen saturation of 91% remain despite antibiotics. Creatinine is normal and she is '
                   'cognitively intact.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 52-year-old man, ASA 2, is scheduled for elective laparoscopic appendectomy after interval resolution of '
                   'an abscess. He is active, has normal lungs and renal function, no prior VTE, and intact cognition.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 78-year-old woman is undergoing open repair of a complex renal artery aneurysm. She has CKD stage 4, '
                   'diabetes, and creatinine 2.2 mg/dL. She remains oriented and has no active pulmonary disease.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': 'A 67-year-old man is scheduled for revision total hip arthroplasty. He has two previous postoperative DVTs, '
                   'obesity, and marked mobility restriction. Warfarin is held with bridging. Lungs and renal function are '
                   'stable.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 72-year-old woman is planned for open esophagectomy. She has severe COPD, recurrent aspiration, continued '
                   'smoking, and oxygen saturation 90%. A weak productive cough is present. Kidney function and cognition are '
                   'normal.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'An 89-year-old man requires urgent fixation of a periprosthetic femur fracture. He has dementia, CKD stage '
                   '3a, dehydration, prior DVT, and five days of immobility. Creatinine is 1.5 mg/dL from 1.1; he is '
                   'disoriented.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': 'A 31-year-old woman, ASA 1, is scheduled for elective laparoscopic umbilical hernia repair. She has no '
                   'chronic illness, takes no daily medication, and has normal pulmonary, renal, thrombotic, and cognitive '
                   'status.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 66-year-old man is planned for ureteral reconstruction in a solitary kidney. CKD stage 3b is present with '
                   'creatinine 1.8 mg/dL. He is hydrated, active, and oriented, with clear lungs and no VTE history.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': 'A 74-year-old woman is scheduled for total knee arthroplasty. She has a prior unprovoked DVT, BMI 40, and '
                   'severe mobility limitation. Apixaban is held. Pulmonary and renal function are normal; cognition is intact.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'An 83-year-old man is scheduled for elective parathyroidectomy. He has mild Alzheimer disease, severe '
                   'hearing impairment, and chronic oxybutynin use. He is oriented to person and place. Lungs and kidneys are '
                   'normal.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': 'A 69-year-old man is planned for right upper lobectomy. He has severe COPD, home oxygen with exertion, '
                   'current smoking, and resting saturation of 89%. Creatinine is 0.9 mg/dL and baseline cognition is normal.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 44-year-old man, ASA 2, is scheduled for robotic partial nephrectomy. Hypertension is controlled, '
                   'creatinine is 0.9 mg/dL, and pulmonary function, mobility, cognition, and thrombosis history are '
                   'unremarkable.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'An 86-year-old woman requires emergency surgery for perforated diverticulitis. She has CKD stage 3b, '
                   'dementia, sepsis, and poor intake. Creatinine is 2.2 mg/dL from 1.5. She is hypotensive and unable to '
                   'maintain attention.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': 'A 71-year-old man is scheduled for resection of a pelvic malignancy. Prior PE, iliac-vein compression, and '
                   'reduced mobility are documented. Rivaroxaban is held. Pulmonary and renal function are normal; cognition is '
                   'intact.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 60-year-old woman is scheduled for laryngectomy. Chronic aspiration, thick secretions, COPD, and recent '
                   'bronchitis are present. Oxygen saturation is 92%. Kidney function and cognition are normal.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 77-year-old man is undergoing posterior lumbar fusion. He has CKD stage 3b, diabetes, chronic NSAID use, '
                   'and creatinine 1.8 mg/dL. He is hydrated and oriented, with clear lungs.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': 'A 42-year-old woman, ASA 1, is scheduled for outpatient excision of a benign forearm mass. She is healthy '
                   'and active, with normal pulmonary, renal, cognitive, and thrombotic history.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'An 88-year-old man presents for urgent hip-fracture fixation. Moderate dementia, previous DVT, frailty, and '
                   'four days of immobility are present. He is disoriented and dependent for care. Lungs are clear; creatinine '
                   'is 1.1 mg/dL.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': 'A 64-year-old woman will undergo abdominal wall reconstruction. Active malignancy, obesity, prior DVT, and '
                   'poor mobility are present. Anticoagulation is interrupted. Lungs and kidneys are stable, and cognition is '
                   'normal.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 72-year-old man is planned for upper lobectomy. Severe COPD, recurrent bronchitis, daily sputum, and '
                   'oxygen saturation of 91% are documented. Creatinine is normal and cognition is intact.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'An 82-year-old woman will undergo combined valve replacement and coronary bypass. CKD stage 4, diabetes, '
                   'frailty, creatinine 2.4 mg/dL, and mild cognitive impairment are present. No active respiratory infection is '
                   'noted.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': 'A 52-year-old woman, ASA 2, is scheduled for laparoscopic hysterectomy for symptomatic fibroids. She has '
                   'controlled hypertension, good exercise tolerance, normal lungs and kidneys, no prior thrombosis, and intact '
                   'cognition.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 70-year-old woman is scheduled for radical cystectomy. She had DVT during chemotherapy and remains less '
                   'active than baseline. Apixaban is held. Lungs are clear, creatinine is 1.0 mg/dL, and cognition is normal.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'An 85-year-old man is scheduled for urgent cholecystectomy. He has mild dementia, severe hearing impairment, '
                   'and prior delirium with infection. He is febrile and disoriented. Lungs are clear and creatinine is 1.0 '
                   'mg/dL.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': 'A 59-year-old woman is planned for distal pancreatectomy. She has diabetic CKD stage 3a, recent vomiting, '
                   'and creatinine 1.6 mg/dL from 1.3. She is alert with no pulmonary disease or VTE history.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': 'A 68-year-old man is undergoing decompression for metastatic spinal cord compression. He has been '
                   'wheelchair-dependent for one month and previously had PE during treatment. Anticoagulation is held. Renal '
                   'and pulmonary function are stable.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 76-year-old woman is planned for open repair of a paraesophageal hernia. She has recurrent aspiration, '
                   'bronchiectasis, recent pneumonia, saturation 92%, and a weak cough. Kidney function and cognition are '
                   'normal.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'An 88-year-old man requires urgent below-knee amputation for infected ischemic foot. He has CKD stage 4, '
                   'diabetes, dementia, and poor intake. Creatinine is 2.7 mg/dL from 2.2, and he is agitated and inattentive.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': 'A 35-year-old woman, ASA 1, is scheduled for laparoscopic treatment of endometriosis. She has no chronic '
                   'disease, normal exercise tolerance, and no pulmonary, renal, thrombotic, or cognitive concerns.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 71-year-old man is scheduled for shoulder arthroplasty. Obesity and rheumatoid arthritis limit endurance, '
                   'but he remains independently mobile and has no prior VTE. Lungs, kidneys, and cognition are normal.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 65-year-old woman will undergo total gastrectomy. COPD, active smoking, weak cough, low albumin, and '
                   'room-air saturation of 92% are present. Renal function and baseline cognition are normal.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 79-year-old man is planned for open mitral-valve surgery. He has CKD stage 3b, diabetes, frailty, '
                   'creatinine 1.9 mg/dL, and mild forgetfulness. Lungs are clear without infection.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': 'A 50-year-old woman, ASA 2, is planned for laparoscopic bowel resection for endometriosis. She is active, '
                   'has normal lungs and renal function, no prior thrombosis, and intact cognition.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 69-year-old man is scheduled for pelvic sarcoma resection. Active cancer, iliac-vein compression, chronic '
                   'leg edema, and prior DVT are documented. Apixaban is held. Pulmonary and renal function are normal.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 77-year-old woman is undergoing VATS wedge resection. Pulmonary fibrosis, exertional oxygen use, resting '
                   'saturation 90%, and worsening dry cough are present. Creatinine and cognition are normal.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'An 86-year-old man is planned for elective colostomy reversal. He has mild Alzheimer disease, severe hearing '
                   'loss, and a previous postoperative delirium episode. His spouse manages medications. Lungs and renal '
                   'function are stable.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': 'A 62-year-old woman will undergo partial nephrectomy. CKD stage 3a, diabetes, creatinine 1.5 mg/dL, and '
                   'expected prolonged hilar clamping are noted. She is active, oriented, and has clear lungs.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': 'A 74-year-old man is scheduled for hindfoot fusion. Factor V Leiden, prior DVT, obesity, and prolonged '
                   'restricted weight-bearing are present. Rivaroxaban is held. Pulmonary and renal function are normal.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 67-year-old woman is planned for thoracoabdominal aneurysm repair. Severe COPD, current smoking, CKD stage '
                   '3a, saturation 91%, and creatinine 1.4 mg/dL are documented. Cognition is intact.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': 'A 91-year-old man requires urgent ureteral stenting for urosepsis. Dementia, CKD stage 3b, hypotension, and '
                   'poor intake are present. Creatinine is 2.5 mg/dL from 1.7, and he is acutely agitated.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': 'A 39-year-old woman, ASA 1, is scheduled for elective breast reduction. She is healthy and active, with '
                   'normal lungs, renal function, cognition, mobility, and no VTE history.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 66-year-old man is undergoing right hemicolectomy during chemotherapy. He had a port-related DVT four '
                   'months ago and apixaban is now held. Fatigue limits activity. Lungs and kidneys are normal.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 72-year-old woman is scheduled for open nephrectomy. COPD, active smoking, a recent steroid-treated '
                   'exacerbation, saturation 92%, and wheeze are present. Creatinine is 0.9 mg/dL.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'An 84-year-old man requires emergency colectomy for bowel ischemia. CKD stage 3b, chronic steroids, frailty, '
                   'and mild dementia are present. He is hypotensive and confused; creatinine is 2.2 mg/dL from 1.5.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 1},
 {'clinical_note': 'A 56-year-old woman is planned for robotic ventral hernia repair. She has controlled hyperlipidemia, walks '
                   'daily, and has normal pulmonary and renal function, intact cognition, and no prior VTE.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 76-year-old man is scheduled for pelvic-fracture fixation. He has been bedbound for nine days, has active '
                   'lymphoma, and previously had PE. Enoxaparin is held. Lungs and kidneys are normal.',
  'DVT': 1,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 70-year-old woman will undergo pancreaticoduodenectomy. Chronic bronchitis, current smoking, daily sputum, '
                   'saturation 93%, and albumin 2.9 g/dL are present. Kidney function and cognition are normal.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'An 87-year-old man is scheduled for elective hip replacement. Mild dementia, severe hearing loss, and '
                   'chronic benzodiazepine use are present. He is oriented to person and place. Lungs and kidneys are normal.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 1},
 {'clinical_note': 'A 60-year-old woman will undergo open renal-artery repair. CKD stage 3a, hypertension, recent contrast '
                   'exposure, and creatinine 1.5 mg/dL are documented. She is hydrated and oriented.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 1,
  'Delirium': 0},
 {'clinical_note': 'A 47-year-old man is scheduled for Achilles tendon repair. He is otherwise healthy, remains mobile with '
                   'crutches, and has no personal or family VTE history. Lungs, kidneys, and cognition are normal.',
  'DVT': 0,
  'Pneumonia': 0,
  'AKI': 0,
  'Delirium': 0},
 {'clinical_note': 'A 78-year-old woman is planned for VATS decortication after empyema. Chronic bronchiectasis, persistent '
                   'productive cough, recent infection, and saturation 91% are present. Creatinine is 0.9 mg/dL and cognition is '
                   'intact.',
  'DVT': 0,
  'Pneumonia': 1,
  'AKI': 0,
  'Delirium': 0}]

    df = pd.DataFrame.from_records(
        rows,
        columns=["clinical_note", "DVT", "Pneumonia", "AKI", "Delirium"],
    )

    if len(df) != 500:
        raise RuntimeError(f"Expected 500 curated rows, found {len(df)}.")

    outcome_columns = ["DVT", "Pneumonia", "AKI", "Delirium"]
    if df["clinical_note"].duplicated().any():
        raise RuntimeError("Curated cohort contains duplicate clinical notes.")
    if not all(set(df[column].unique()).issubset({0, 1}) for column in outcome_columns):
        raise RuntimeError("Outcome columns must contain only 0 and 1.")

    return df



