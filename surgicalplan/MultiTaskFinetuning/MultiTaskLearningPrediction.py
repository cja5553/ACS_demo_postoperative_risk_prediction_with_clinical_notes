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




import pandas as pd


def get_pseudo_data():
    rows = [
        {'clinical_note': '79-year-old woman, ASA 3, for urgent hemiarthroplasty after a displaced femoral-neck fracture. She has been largely bedbound for two days. History includes mild dementia, hypertension, and CKD stage 3a; creatinine is 1.2 mg/dL at baseline. She is afebrile with clear lungs and normally ambulates with a walker. Her daughter reports prior confusion after hospitalization.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '42-year-old woman, ASA 1, scheduled for elective laparoscopic cholecystectomy for biliary colic. She has no cardiopulmonary, renal, or neurologic disease, takes no daily medication, and exercises regularly. BMI is 24. Room-air saturation is 99%, creatinine 0.7 mg/dL, and she is fully oriented and independent.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '74-year-old man, ASA 4, undergoing three-vessel coronary artery bypass grafting. History includes ischemic cardiomyopathy with EF 30%, insulin-treated diabetes, and CKD stage 3b with baseline creatinine 1.9 mg/dL. He is independent in basic activities and cognitively intact. Lungs are clear, and there is no recent infection.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '67-year-old woman, ASA 3, planned open esophagectomy for distal esophageal cancer. She has moderate COPD, continues to smoke one pack daily, and becomes dyspneic after one flight of stairs. Room-air saturation is 92%, with diminished breath sounds but no active infection. Creatinine is 0.8 mg/dL, and cognition is normal.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '58-year-old man, ASA 3, for pancreaticoduodenectomy for pancreatic adenocarcinoma. He had a proximal leg DVT two years ago and takes apixaban, now held for surgery. He has lost 9 kg and is less active but remains independent. Lungs are clear, creatinine is 0.9 mg/dL, and cognition is intact.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '84-year-old woman, ASA 4, requires emergency open colectomy for perforated diverticulitis with sepsis. She has moderate dementia, CKD stage 3b, poor oral intake, and a new 2 L oxygen requirement. Creatinine is 1.8 mg/dL from a baseline of 1.4. She is inattentive and relies on family for daily care.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '36-year-old woman, ASA 1, scheduled for outpatient hemithyroidectomy for a benign thyroid nodule. She has no chronic disease, normal exercise tolerance, and no prior anesthesia complications. Room-air saturation is 99%, creatinine is 0.6 mg/dL, and neurologic examination and cognition are normal.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '69-year-old man, ASA 3, for multilevel lumbar decompression and fusion. BMI is 39, walking is limited to one block by back pain, and he had a pulmonary embolism four years ago after surgery. Chronic apixaban has been stopped preoperatively. He has no COPD, creatinine is 1.0 mg/dL, and cognition is intact.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '78-year-old woman, ASA 3, scheduled for elective total knee arthroplasty. She has mild cognitive impairment, hearing loss, hypertension, and CKD stage 3a with creatinine 1.2 mg/dL. She lives alone but needs help organizing medications. Lungs are clear, oxygen saturation is 97%, and she walks daily with a cane.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '63-year-old man, ASA 3, planned partial nephrectomy for a renal mass in his only functioning kidney. He has diabetes and CKD stage 3b; baseline creatinine is 1.8 mg/dL. Blood pressure is controlled, lungs are clear, and he remains active and fully oriented. No prior thromboembolism is reported.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '61-year-old woman, ASA 3, for right upper lobectomy by VATS for lung cancer. She has severe COPD, a 45-pack-year smoking history, and DLCO of 44% predicted. She uses tiotropium and albuterol and has room-air saturation of 91%. Creatinine is 0.8 mg/dL, and she is cognitively intact.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '82-year-old man, ASA 3, scheduled for transurethral resection of the prostate for urinary retention. He has moderate Alzheimer disease, severe hearing impairment, and nightly lorazepam use. His daughter provides most history. He is calm but oriented only to person and place. Creatinine is 1.0 mg/dL, and lungs are clear.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '65-year-old woman, ASA 3, for open abdominal hysterectomy and staging for endometrial cancer. BMI is 43, mobility is limited by knee pain, and she had a pulmonary embolism five years ago. Rivaroxaban has been held. She has normal renal function, no lung disease, and no cognitive impairment.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '71-year-old man, ASA 4, requires emergency small-bowel resection for strangulated obstruction. He has had persistent vomiting and poor intake for three days. Blood pressure is 92/58 mmHg after initial fluids, lactate is elevated, and creatinine is 1.2 mg/dL from a baseline of 0.9. He is normally independent and cognitively intact.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '33-year-old woman, ASA 2, scheduled for cesarean delivery because of placenta previa. BMI is 35, but she has remained mobile and has no personal or family history of thrombosis. Pregnancy has otherwise been uncomplicated. Lungs are clear, creatinine is 0.5 mg/dL, and she is alert without neurologic or psychiatric disease.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '52-year-old man, ASA 1, for outpatient arthroscopic rotator-cuff repair. He jogs several times weekly, has no chronic medical conditions, and takes only occasional acetaminophen. BMI is 25. Room-air saturation is 98%, creatinine is 0.9 mg/dL, and he has no history of confusion or thromboembolism.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '80-year-old woman, ASA 4, for surgical aortic valve replacement. She has severe aortic stenosis, CKD stage 3b with creatinine 1.7 mg/dL, diabetes, and frailty with slow gait. She is cognitively intact but needs help with shopping. Lungs are clear, and there has been no recent respiratory infection.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '69-year-old man, ASA 2, scheduled for robotic prostatectomy for localized prostate cancer. Hypertension is well controlled with amlodipine. He walks four miles daily, has no lung or kidney disease, and has never had VTE. Creatinine is 0.8 mg/dL, oxygen saturation is 98%, and cognition is normal.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '48-year-old woman, ASA 3, for laparoscopic sleeve gastrectomy. BMI is 52, and she has severe obstructive sleep apnea but uses CPAP consistently. She is independently mobile, has no COPD or prior VTE, and recently stopped smoking. Creatinine is 0.7 mg/dL, and cognition is intact.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '76-year-old man, ASA 3, for craniotomy and resection of a frontal meningioma. History includes a remote ischemic stroke, mild cognitive impairment, and recent initiation of dexamethasone and levetiracetam. He is independent but repeats questions during the interview. Lungs are clear, creatinine is 0.9 mg/dL, and there is no VTE history.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '57-year-old man, ASA 3, planned right hepatectomy for hepatocellular carcinoma. He has NASH cirrhosis, albumin 2.9 g/dL, mild ascites, and baseline creatinine 1.1 mg/dL. He is oriented and ambulates independently. There is no chronic lung disease or prior thromboembolism.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '73-year-old woman, ASA 3, for open fixation of a tibial-plateau fracture. She has been non-weight-bearing in a knee immobilizer for ten days. History includes a prior postoperative DVT; she is not on chronic anticoagulation. Lungs are clear, creatinine is 0.8 mg/dL, and she is cognitively normal.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '66-year-old man, ASA 3, for pancreaticoduodenectomy. He has moderate COPD and was treated for community-acquired pneumonia six weeks ago; symptoms resolved, but exercise tolerance remains reduced. Room-air saturation is 93%, and he has a weak cough. Creatinine is 0.9 mg/dL, and cognition is intact.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '81-year-old woman, ASA 3, scheduled for revision total hip arthroplasty. She is frail, uses a walker, and had marked confusion for three days after her previous hip operation. She has CKD stage 3a with stable creatinine of 1.2 mg/dL. Lungs are clear, and there is no previous VTE.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '45-year-old woman, ASA 2, for unilateral mastectomy and sentinel-node biopsy for early breast cancer. She has controlled hypothyroidism and otherwise good health. She works full time, exercises regularly, and has no prior VTE, lung disease, kidney disease, or cognitive symptoms. Creatinine is 0.7 mg/dL and oxygen saturation is 99%.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '72-year-old man, ASA 3, for elective open abdominal aortic aneurysm repair. He has peripheral arterial disease, CKD stage 3a, and hypertension; creatinine is 1.4 mg/dL. He remains independent but becomes fatigued after two blocks. Cognition is intact, lungs are clear, and he has no prior VTE.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '68-year-old woman, ASA 3, for total knee arthroplasty. She had a calf DVT after her last knee replacement and completed anticoagulation. BMI is 37, and she walks only short distances because of pain. Renal function and pulmonary examination are normal, and she is fully oriented.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '77-year-old man, ASA 4, for urgent repair of a perforated gastric ulcer. He has COPD on home oxygen, chronic prednisone use, a productive cough, mild cognitive impairment, and prior confusion after hospitalization. Oxygen saturation is 89% on room air; creatinine is 1.0 mg/dL.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '55-year-old woman, ASA 2, for laparoscopic adrenalectomy for a nonfunctioning adrenal mass. Hypertension is controlled, renal function is normal, and she hikes on weekends. No lung disease, prior thrombosis, or cognitive concerns are present.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '83-year-old man, ASA 4, for urgent decompression of cervical spinal cord compression. He has Parkinson disease, mild dementia, dysphagia, and recurrent coughing with meals. He walks with assistance. Creatinine is 1.1 mg/dL, and oxygen saturation is 95% without active infection.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '62-year-old woman, ASA 3, scheduled for radical cystectomy with ileal conduit for bladder cancer. She has CKD stage 3b, diabetes, and hypertension; creatinine is 1.8 mg/dL. She is independent and cognitively intact, with clear lungs and no VTE history.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '70-year-old man, ASA 3, for open colectomy for colon cancer. He received neoadjuvant chemotherapy, has lost 12 kg, and has a remote unprovoked DVT. He is less active but still independent. Lungs are clear, creatinine is 0.9 mg/dL, and cognition is normal.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '46-year-old man, ASA 2, for laparoscopic inguinal hernia repair. He has well-controlled asthma but has not used albuterol in six months. He runs three times weekly, has normal renal function, and denies prior VTE or cognitive symptoms.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '75-year-old woman, ASA 4, for mitral valve replacement. She has pulmonary hypertension, atrial fibrillation, CKD stage 3b, and prior delirium after ICU admission. Creatinine is 1.9 mg/dL. She is independent in basic activities but needs help with finances.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '64-year-old man, ASA 3, for esophagectomy. He has severe reflux with nocturnal regurgitation, chronic aspiration changes on CT, and a weak cough after prior stroke. Room-air saturation is 94%. Creatinine is normal and he is cognitively intact.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '59-year-old woman, ASA 3, for ankle fusion. She has rheumatoid arthritis treated with prednisone, BMI 34, and has been using a knee scooter for six weeks. A sister had an unprovoked PE, but the patient has no personal VTE history. Lungs and renal function are normal.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '39-year-old man, ASA 1, for elective appendectomy after interval treatment of appendicitis. He is asymptomatic, physically active, and has no chronic medical conditions. Creatinine is 0.8 mg/dL, oxygen saturation 99%, and cognition is normal.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '86-year-old woman, ASA 4, for emergency laparotomy for ischemic bowel. She has vascular dementia, heart failure, CKD stage 4, and hypotension requiring norepinephrine. Creatinine is 2.8 mg/dL from 2.1. She had witnessed aspiration during vomiting, with new bibasilar infiltrates, and is disoriented.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '73-year-old man, ASA 3, for elective carotid endarterectomy. He has prior TIA, hypertension, and diabetes but normal renal function. He walks independently, has no lung disease, and is fully oriented. No VTE history is present.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '66-year-old woman, ASA 3, for nephroureterectomy for upper-tract urothelial carcinoma. She has a solitary kidney, CKD stage 3b, and diabetes. Baseline creatinine is 1.7 mg/dL. She is active, cognitively intact, and has clear lungs.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '71-year-old woman, ASA 3, for VATS wedge resection of a pulmonary nodule. She has mild COPD, stopped smoking 15 years ago, and walks two miles daily. Oxygen saturation is 97%, spirometry shows only mild obstruction, and she has no recent respiratory illness.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '60-year-old man, ASA 3, for total hip arthroplasty. He has obesity, obstructive sleep apnea, and a history of unprovoked DVT on chronic apixaban, which is held. He is limited to household walking. Creatinine is normal, lungs are clear, and cognition is intact.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '79-year-old man, ASA 3, for transurethral resection of a bladder tumor. He has mild cognitive impairment, severe visual loss, and uses diphenhydramine nightly. He is independent with dressing but needs help with medications. Lungs and kidney function are stable.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '51-year-old woman, ASA 2, for laparoscopic hysterectomy for fibroids. BMI is 31, but she remains active and has no cardiopulmonary, renal, thrombotic, or cognitive history. Creatinine and oxygen saturation are normal.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '68-year-old man, ASA 4, for emergency debridement of necrotizing soft-tissue infection. He has poorly controlled diabetes, CKD stage 3b, fever, and septic shock requiring vasopressors. Creatinine is 2.3 mg/dL from a baseline of 1.6. He is confused and tachypneic.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '74-year-old woman, ASA 3, for open reduction of a distal femur fracture. She has metastatic breast cancer, has been bedbound for five days, and previously developed a DVT during chemotherapy. Anticoagulation is held for surgery. Lungs and renal function are stable.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '65-year-old man, ASA 3, for open right hemicolectomy. He has bronchiectasis with daily sputum production, recurrent respiratory infections, and room-air saturation of 93%. He is not currently febrile. Creatinine is 1.0 mg/dL and cognition is intact.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '88-year-old woman, ASA 3, for repair of an incarcerated inguinal hernia. She lives independently but has mild cognitive impairment and severe hearing loss. She takes zolpidem nightly and had transient confusion after a prior operation. Renal function and lungs are normal.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '57-year-old man, ASA 2, for elective shoulder arthroplasty. He has controlled hypertension and no other chronic illness. He cycles regularly, has normal renal and pulmonary function, and no prior VTE or cognitive symptoms.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '70-year-old woman, ASA 4, for open thoracoabdominal aneurysm repair. She has CKD stage 3b, diabetes, and baseline creatinine 1.8 mg/dL. She is cognitively intact and has no chronic lung disease. Functional capacity is limited by claudication.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '76-year-old man, ASA 3, for radical prostatectomy. He has a remote DVT after a long flight, obesity, and limited activity because of lumbar stenosis. He is not anticoagulated. Lungs are clear, creatinine is 0.9 mg/dL, and cognition is normal.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '44-year-old woman, ASA 2, scheduled for laparoscopic ovarian cystectomy. She has mild asthma controlled with an inhaled steroid, normal exercise tolerance, and no kidney disease, VTE history, or cognitive concerns.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '81-year-old man, ASA 4, for urgent fixation of an intertrochanteric hip fracture. He has moderate dementia, chronic benzodiazepine use, and CKD stage 3a. He was found on the floor after an unknown period and is now mildly dehydrated. Lungs are clear.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '63-year-old woman, ASA 3, for open lobectomy for lung cancer. She has moderate COPD, continues to smoke, and has chronic productive cough. FEV1 is 48% predicted and oxygen saturation is 92%. Renal function and cognition are normal.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '69-year-old man, ASA 3, for laparoscopic colectomy for colon cancer. He has stage 3 CKD but stable creatinine of 1.4 mg/dL, good exercise tolerance, and no diabetes. Lungs are clear and cognition is intact.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '72-year-old woman, ASA 3, for reverse shoulder arthroplasty. She has Parkinson disease, mild cognitive impairment, and a prior episode of postoperative delirium. She remains ambulatory with a cane. Renal and pulmonary function are normal.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '54-year-old man, ASA 3, for liver transplantation. He has decompensated cirrhosis, hepatorenal physiology, creatinine 1.6 mg/dL, recurrent ascites, and two prior admissions for hepatic encephalopathy treated with lactulose. He is oriented today but fatigued; lungs are clear.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '58-year-old woman, ASA 3, for pelvic exenteration for recurrent cervical cancer. She has a recent proximal DVT treated with apixaban, which is held, and spends most of the day in a chair because of pain. Creatinine and lung examination are normal.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '31-year-old man, ASA 1, for outpatient excision of a benign hand mass. He is healthy, active, and takes no medication. No lung, kidney, thrombotic, or cognitive history is present.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '77-year-old woman, ASA 4, for urgent open repair of a ruptured abdominal aortic aneurysm. She is hypotensive, oliguric, and has CKD stage 3b. Creatinine is 2.2 mg/dL from a baseline of 1.5. She is intermittently confused but has no chronic lung disease.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '64-year-old man, ASA 3, for esophagectomy. He has obesity, severe obstructive sleep apnea with poor CPAP adherence, and recurrent nocturnal aspiration. Oxygen saturation is 93%. Renal function and cognition are normal.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '70-year-old woman, ASA 3, for total knee arthroplasty. She has a family history of VTE and BMI 36 but no personal thrombosis, remains active, and has no cancer. Lungs, kidneys, and cognition are normal.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '84-year-old man, ASA 3, for elective inguinal hernia repair. He has mild dementia, macular degeneration, and takes oxybutynin for overactive bladder. He is independent in basic activities but needs assistance with medications. Renal and pulmonary function are normal.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '67-year-old woman, ASA 4, for CABG. She has CKD stage 4, diabetes, and heart failure with EF 25%. Creatinine is 2.4 mg/dL. She is cognitively intact and has no active pulmonary infection.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '49-year-old man, ASA 2, for elective laparoscopic sigmoid colectomy for recurrent diverticulitis. He has controlled hypertension and no lung, kidney, thrombotic, or cognitive disease. He walks several miles daily.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '68-year-old woman, ASA 3, for VATS decortication after a recent empyema. She has COPD, persistent productive cough, and oxygen saturation of 91% on room air. She completed antibiotics but remains deconditioned. Renal function and cognition are normal.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '75-year-old man, ASA 3, for lumbar fusion. He has chronic immobility from severe spinal stenosis, obesity, and prior PE after knee replacement. Warfarin is stopped and bridged preoperatively. Lungs and renal function are stable.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '82-year-old woman, ASA 4, for emergency surgery for perforated appendicitis. She has frailty, mild dementia, poor oral intake, and creatinine 1.5 mg/dL from a baseline of 1.0. She is tachypneic but chest radiograph is clear.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '56-year-old man, ASA 2, for partial colectomy for a localized neuroendocrine tumor. He has no prior VTE, remains active, and has normal renal and pulmonary function. Cognition is intact.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '73-year-old woman, ASA 3, for robotic hysterectomy for endometrial cancer. She has obesity and prior DVT during pregnancy, but is currently active and not anticoagulated. Lungs and kidneys are normal, and cognition is intact.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '65-year-old man, ASA 3, for open pancreatic surgery. He has chronic pancreatitis, malnutrition, albumin 2.6 g/dL, and CKD stage 3a. Creatinine is 1.5 mg/dL. Lungs are clear and he is fully oriented.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '79-year-old woman, ASA 3, for elective colectomy. She has mild cognitive impairment, hearing aids, and takes amitriptyline for neuropathic pain. She lives with her daughter but is independent in basic activities. Renal and pulmonary function are stable.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '62-year-old man, ASA 3, for right pneumonectomy. He has COPD, ongoing tobacco use, reduced DLCO, and oxygen saturation of 90%. He has no kidney disease or cognitive impairment.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '37-year-old woman, ASA 1, for laparoscopic salpingectomy for ectopic pregnancy. She is hemodynamically stable, has normal renal function, clear lungs, and no thrombotic or cognitive history.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '71-year-old man, ASA 4, for open repair of infected aortic graft. He has sepsis, CKD stage 3b, diabetes, and baseline creatinine 1.8 mg/dL now increased to 2.3. He is lethargic but follows commands; lungs are clear.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '60-year-old woman, ASA 3, for total hip arthroplasty. She has rheumatoid arthritis, BMI 35, chronic prednisone use, and a prior calf DVT. She is limited to household walking. Renal and pulmonary function are normal.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '47-year-old man, ASA 2, for laparoscopic fundoplication. He has well-controlled reflux and no aspiration history, lung disease, kidney disease, prior VTE, or cognitive symptoms. Exercise tolerance is excellent.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '85-year-old woman, ASA 4, for urgent colectomy after lower GI bleeding and hypotension. She has CKD stage 3b, mild dementia, and heart failure. Creatinine is 2.0 mg/dL from a baseline of 1.4. She is disoriented and requires 2 L oxygen for bibasilar atelectatic changes.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '67-year-old man, ASA 3, for nephrectomy for renal-cell carcinoma. He has CKD stage 3a, diabetes, and baseline creatinine 1.5 mg/dL. He is active, cognitively intact, and has no lung disease.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '74-year-old woman, ASA 3, for VATS lobectomy. She stopped smoking ten years ago but has moderate COPD and uses home oxygen at night. Room-air saturation is 91%, and she has a weak cough. Renal function is normal.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '66-year-old man, ASA 3, for radical cystectomy. He has obesity, pelvic malignancy, chronic venous insufficiency, and a prior unprovoked DVT. Apixaban is held. He remains ambulatory but tires easily.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '53-year-old woman, ASA 2, for laparoscopic partial nephrectomy. She has controlled hypertension but normal baseline creatinine, no diabetes, and good exercise tolerance. Cognition and pulmonary function are normal.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '80-year-old man, ASA 3, for total shoulder arthroplasty. He has moderate dementia, hearing loss, and prior postoperative agitation. He is dependent for medication management. Renal and pulmonary function are stable.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '59-year-old woman, ASA 3, for open colectomy. She has severe asthma with two hospitalizations in the past year, chronic oral steroid use, and a current productive cough. Oxygen saturation is 94%. Kidney function and cognition are normal.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '72-year-old man, ASA 4, for transcatheter aortic valve replacement. He has CKD stage 4, diabetes, and heart failure. Creatinine is 2.5 mg/dL. He is cognitively intact and has no active lung disease.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '64-year-old woman, ASA 3, for open fixation of an ankle fracture. She has been non-weight-bearing for three weeks, has active ovarian cancer, and developed a prior DVT during chemotherapy. Anticoagulation is held. Lungs and renal function are normal.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '40-year-old man, ASA 1, for outpatient vasectomy reversal. He is healthy, takes no medications, and has no relevant pulmonary, renal, thrombotic, or cognitive history.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '78-year-old woman, ASA 4, for emergency cholecystectomy for gangrenous cholecystitis. She has mild dementia, CKD stage 3a, and poor intake. Creatinine is 1.6 mg/dL from a baseline of 1.1. She is inattentive and febrile but has clear lungs.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '68-year-old man, ASA 3, for esophagectomy. He has COPD, ongoing smoking, and recurrent aspiration from dysphagia. Oxygen saturation is 92% and he has coarse bibasilar breath sounds. Renal function and cognition are normal.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '71-year-old woman, ASA 3, for pelvic sarcoma resection. She has obesity, limited mobility, and prior PE after chemotherapy. Rivaroxaban is held for surgery. Kidney and lung function are stable.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '61-year-old man, ASA 3, for robotic prostatectomy. He has CKD stage 3a and diabetes but stable creatinine of 1.3 mg/dL, good blood pressure control, and excellent exercise tolerance. Cognition and lungs are normal.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '83-year-old woman, ASA 3, for elective mastectomy. She has mild cognitive impairment, severe hearing loss, and takes gabapentin and zolpidem. She is independent in dressing but needs help with transportation. Renal and pulmonary function are normal.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '55-year-old man, ASA 3, for Whipple procedure. He has CKD stage 3b, diabetes, and recent dehydration from poor oral intake. Creatinine is 1.9 mg/dL from a baseline of 1.5. Lungs are clear and cognition is normal.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '69-year-old woman, ASA 3, for open thoracotomy and decortication. She has bronchiectasis, chronic sputum, and recurrent pneumonias. Oxygen saturation is 92%. Renal function and cognition are normal.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '76-year-old man, ASA 3, for revision knee arthroplasty. He has a prior DVT, obesity, and markedly reduced mobility because of prosthetic loosening. Apixaban is held. Lungs and renal function are stable.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '34-year-old woman, ASA 1, for outpatient excision of a wrist ganglion. She is healthy, active, and has normal renal, pulmonary, cognitive, and thrombotic history.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '82-year-old man, ASA 4, for emergency laparotomy for perforated colon. He has COPD on home oxygen, CKD stage 3b, and vascular dementia. He is febrile, hypotensive, and confused; creatinine is 2.1 mg/dL from 1.5 and chest radiograph shows a new right lower-lobe infiltrate.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '57-year-old woman, ASA 2, for laparoscopic donor nephrectomy. She has normal renal function, no diabetes, excellent exercise capacity, and no pulmonary or cognitive disease. No VTE history is present.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '73-year-old man, ASA 3, for open aortic valve replacement. He has CKD stage 3a, diabetes, and prior transient postoperative confusion. Creatinine is 1.5 mg/dL. He remains independent and has clear lungs.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '66-year-old woman, ASA 3, for open colectomy for ovarian-cancer recurrence. She has active malignancy, a prior unprovoked PE, BMI 38, and limited walking due to ascites. Anticoagulation is held. Renal and pulmonary function are stable.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '69-year-old man, ASA 3, scheduled for open distal pancreatectomy. He has CKD stage 3a, insulin-treated diabetes, and recurrent vomiting with mild volume depletion. Creatinine is 1.6 mg/dL from a baseline of 1.3. He is oriented and has clear lungs.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '74-year-old woman, ASA 3, for thoracic spine fusion. She has severe COPD, uses nocturnal oxygen, and was treated for a COPD exacerbation three weeks ago. Room-air saturation is 90% with a weak cough. Renal function and cognition are normal.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '63-year-old man, ASA 3, for open prostatectomy. He has metastatic prostate cancer, a prior DVT, and reduced mobility from painful bone metastases. Enoxaparin is held for surgery. Kidney and lung function are stable.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '41-year-old woman, ASA 1, for laparoscopic myomectomy. She is healthy, exercises regularly, and has normal renal, pulmonary, and cognitive function with no VTE history.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '87-year-old man, ASA 3, for repair of a strangulated femoral hernia. He has moderate dementia, severe hearing loss, and takes clonazepam nightly. He is dehydrated but creatinine remains near baseline. Lungs are clear.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '58-year-old woman, ASA 4, for bilateral lung transplantation. She has end-stage interstitial lung disease, requires 6 L oxygen, and has frequent respiratory infections. Creatinine is 0.8 mg/dL and cognition is normal.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '71-year-old man, ASA 4, for open repair of a renal artery aneurysm. He has CKD stage 3b, renovascular hypertension, and diabetes. Baseline creatinine is 1.9 mg/dL. He is cognitively intact and has no active pulmonary disease.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '67-year-old woman, ASA 3, for revision total hip arthroplasty. She has chronic venous insufficiency and obesity but no prior VTE, remains independently mobile, and takes aspirin. Renal and pulmonary function are normal.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '80-year-old man, ASA 4, for emergency craniotomy after subdural hematoma. He has baseline mild cognitive impairment, atrial fibrillation, and CKD stage 3a. He is drowsy and disoriented. Lungs are clear and oxygen saturation is 96%.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '62-year-old woman, ASA 3, for open colectomy. She has severe obesity, obstructive sleep apnea without CPAP use, and recurrent aspiration from achalasia. Oxygen saturation is 93% and cough is weak. Renal function and cognition are normal.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '50-year-old man, ASA 2, for robotic partial nephrectomy. He has one small renal mass, normal contralateral kidney, creatinine 0.9 mg/dL, and no diabetes. He is active and cognitively intact.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '75-year-old woman, ASA 3, for distal femoral replacement after pathologic fracture. She has metastatic lung cancer, has been bedbound for one week, and previously had a PE. Anticoagulation is held. Lungs are stable on room air and creatinine is normal.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '65-year-old man, ASA 3, for CABG. He has diabetes and CKD stage 3a but creatinine is stable at 1.3 mg/dL, EF is 55%, and he is physically active. Cognition and pulmonary status are normal.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '84-year-old woman, ASA 4, for urgent hip-fracture repair. She has severe frailty, moderate dementia, and swallowing difficulty after a prior stroke. She has a wet cough and oxygen saturation of 92%. Creatinine is 1.0 mg/dL.', 'DVT': 1, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '56-year-old man, ASA 3, for radical nephrectomy. He has CKD stage 3b in the contralateral kidney and poorly controlled hypertension. Creatinine is 1.8 mg/dL. Lungs are clear and cognition is normal.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '72-year-old woman, ASA 3, for total knee arthroplasty. She has a prior unprovoked DVT and wears compression stockings for chronic leg swelling. Apixaban is held. She remains independent but walks slowly.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '45-year-old man, ASA 2, for laparoscopic colectomy. He has mild well-controlled asthma, no smoking history, normal renal function, and excellent exercise tolerance. Cognition is intact.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '78-year-old man, ASA 4, for urgent CABG after NSTEMI. He has CKD stage 3b, heart failure with EF 28%, diabetes, and moderate COPD with daily sputum. Creatinine is 2.0 mg/dL. He has mild cognitive impairment and became delirious during a prior ICU admission.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '70-year-old woman, ASA 3, for VATS lobectomy. She has mild COPD but quit smoking 20 years ago, walks two miles daily, and has oxygen saturation of 97%. Renal function and cognition are normal.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '68-year-old man, ASA 3, for open cystectomy. He has active bladder cancer, obesity, prior PE, and reduced mobility after chemotherapy-induced neuropathy. Rivaroxaban is held. Renal and pulmonary function are stable.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '82-year-old woman, ASA 3, for elective laparoscopic cholecystectomy. She has mild cognitive impairment, severe hearing loss, and takes amitriptyline. She is otherwise independent with normal renal and pulmonary function.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '59-year-old man, ASA 3, for open aortic aneurysm repair. He has CKD stage 3a, diabetes, and baseline creatinine 1.5 mg/dL. He is active and cognitively intact; lungs are clear.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '64-year-old woman, ASA 3, for esophagectomy. She has severe COPD, active smoking, and recent hospitalization for pneumonia. Oxygen saturation is 91% and cough remains productive. Kidney function and cognition are normal.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '38-year-old man, ASA 1, for laparoscopic appendectomy. He is otherwise healthy, has normal renal and pulmonary function, and no prior VTE or cognitive history.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '77-year-old woman, ASA 4, for emergency open colectomy for ischemic colitis. She has CKD stage 4, moderate dementia, and hypotension requiring vasopressors. Creatinine is 3.0 mg/dL from 2.2. She is confused and requires 3 L oxygen after witnessed emesis.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '73-year-old man, ASA 3, for open left hemicolectomy. He has prior postoperative DVT, active colon cancer, BMI 33, and limited mobility from knee arthritis. Apixaban is held. Lungs and renal function are normal.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '58-year-old woman, ASA 2, for laparoscopic sacrocolpopexy. She has controlled hypertension, normal kidney function, no lung disease, no previous VTE, and remains physically active.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '81-year-old man, ASA 3, for elective TURP. He has Parkinson disease, mild dementia, and takes clonazepam and oxybutynin. His wife manages medications. Renal function is normal and lungs are clear.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '66-year-old woman, ASA 4, for lung-volume-reduction surgery. She has severe emphysema, uses 3 L home oxygen, and has frequent COPD exacerbations. Oxygen saturation is 88% on room air. Creatinine is 0.8 mg/dL and cognition is normal.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '69-year-old man, ASA 3, for open partial nephrectomy. He has a solitary kidney, CKD stage 3b, hypertension, and diabetes. Creatinine is 1.9 mg/dL. He is active and cognitively intact.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '52-year-old woman, ASA 3, for bariatric surgery. BMI is 48 and she has OSA, but she uses CPAP nightly, walks daily, has no prior VTE, and has normal renal and pulmonary testing.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '78-year-old man, ASA 4, for emergency repair of a ruptured iliac aneurysm. He is hypotensive and has CKD stage 3b; creatinine is 2.4 mg/dL from 1.7. He is confused but has no chronic cognitive disorder. Lungs are clear.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '62-year-old woman, ASA 3, for open ovarian-cancer debulking. She has a recent proximal DVT, extensive pelvic disease, ascites, and reduced mobility. Therapeutic enoxaparin is held. Renal and pulmonary function are stable.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '71-year-old man, ASA 3, for esophagectomy. He has prior stroke with dysphagia, chronic cough after meals, and recurrent aspiration pneumonia. Oxygen saturation is 94%. Creatinine is normal and cognition is intact.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '43-year-old woman, ASA 1, for arthroscopic meniscal repair. She is healthy, fully mobile, and has no previous thrombosis, lung disease, kidney disease, or cognitive concerns.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '85-year-old woman, ASA 4, for urgent fixation of a humeral fracture after a fall. She has moderate dementia, severe visual impairment, and chronic opioid use. She is oriented only to person. Renal function and lungs are stable.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '64-year-old man, ASA 4, for CABG. He has CKD stage 4, insulin-treated diabetes, peripheral vascular disease, and creatinine 2.6 mg/dL. He is cognitively intact and has no active pulmonary infection.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '57-year-old woman, ASA 3, for open lobectomy. She has bronchiectasis, daily purulent sputum, and two pneumonias in the past year. Oxygen saturation is 93%. Renal function and cognition are normal.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '70-year-old man, ASA 3, for revision ankle arthrodesis. He has been non-weight-bearing for eight weeks, has obesity, and had a PE after prior foot surgery. Warfarin is held. Lungs and kidneys are stable.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '61-year-old woman, ASA 2, for laparoscopic adrenalectomy. She has controlled diabetes but normal creatinine, good exercise tolerance, clear lungs, and no cognitive or VTE history.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '76-year-old man, ASA 4, for open repair of an infected femoral graft. He has sepsis, CKD stage 3b, and creatinine 2.2 mg/dL from 1.6. He is inattentive and febrile, with clear lungs.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '68-year-old woman, ASA 3, for robotic hysterectomy. She has obesity and a remote pregnancy-associated DVT but remains active and has no current cancer beyond early-stage uterine disease. Renal and pulmonary function are normal.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '72-year-old man, ASA 3, for partial laryngectomy. He has COPD, active smoking, thick secretions, and a weak cough after prior radiation. Oxygen saturation is 92%. Kidney function and cognition are normal.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '59-year-old man, ASA 3, for open nephroureterectomy. He has CKD stage 3b and diabetes; creatinine is 1.8 mg/dL. He is active and has normal lungs and cognition.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '82-year-old woman, ASA 3, for elective breast lumpectomy. She has mild cognitive impairment and hearing loss but lives independently and takes no sedatives. Renal and pulmonary function are normal.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '65-year-old woman, ASA 3, for open resection of pelvic sarcoma. She has active malignancy, chronic leg edema, a prior unprovoked DVT, and reduced walking tolerance. Apixaban is held.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '48-year-old man, ASA 2, for laparoscopic cholecystectomy. He has mild OSA treated with CPAP and otherwise normal renal, pulmonary, thrombotic, and cognitive history.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '79-year-old man, ASA 4, for emergency bowel resection after mesenteric ischemia. He has heart failure, CKD stage 3b, hypotension, and mild dementia. Creatinine is 2.0 mg/dL from 1.5. He vomited during transport and now has a new right-basilar infiltrate with oxygen requirement.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '63-year-old woman, ASA 3, for right upper lobectomy. She has mild COPD but quit smoking 18 years ago, has FEV1 75% predicted, and walks three miles daily. Oxygen saturation is 97%.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '74-year-old man, ASA 3, for open prostatectomy. He has CKD stage 3a, diabetes, and creatinine 1.4 mg/dL but good volume status and controlled blood pressure. Cognition and lungs are normal.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '67-year-old woman, ASA 3, for total hip arthroplasty. She has prior unprovoked PE, BMI 40, and limited mobility from severe arthritis. Rivaroxaban is held. Renal and pulmonary function are stable.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '53-year-old man, ASA 2, for laparoscopic inguinal hernia repair. He has controlled hypertension, normal renal function, no lung disease, and excellent exercise tolerance.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '83-year-old woman, ASA 4, for urgent hip-fracture repair. She has Alzheimer disease, severe hearing impairment, and nightly zolpidem. She is disoriented in the emergency department. Lungs are clear and creatinine is near baseline.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '61-year-old man, ASA 3, for esophagectomy. He has severe COPD, ongoing smoking, chronic sputum, and oxygen saturation of 91%. Renal function and cognition are normal.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '71-year-old woman, ASA 4, for aortic valve replacement. She has CKD stage 3b, diabetes, heart failure, and creatinine 1.9 mg/dL. She is cognitively intact and has no recent respiratory illness.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '39-year-old woman, ASA 1, for outpatient breast reduction. She is healthy, active, and has no relevant renal, pulmonary, thrombotic, or cognitive history.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '76-year-old man, ASA 3, for open colectomy. He has moderate dementia, prior delirium, hearing loss, and chronic gabapentin use. Renal function and lungs are stable.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '58-year-old woman, ASA 3, for distal pancreatectomy. She has CKD stage 3b, diabetes, and baseline creatinine 1.7 mg/dL. She is euvolemic and active; lungs and cognition are normal.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '69-year-old man, ASA 3, for VATS lobectomy. He has severe emphysema, uses oxygen with exertion, and had pneumonia two months ago. Oxygen saturation is 90%. Renal function is normal.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '62-year-old woman, ASA 3, for open ovarian-cancer debulking. She has a prior DVT, large pelvic mass compressing iliac veins, and limited mobility. Enoxaparin is held. Renal and pulmonary function are stable.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '55-year-old man, ASA 2, for laparoscopic nephrectomy as a living donor. He has normal kidney function, no diabetes, good exercise tolerance, and no lung, VTE, or cognitive history.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '86-year-old woman, ASA 4, for emergency laparotomy after perforated ulcer. She has CKD stage 4, moderate dementia, hypotension, and aspiration during vomiting. Creatinine is 2.9 mg/dL from 2.1, and she is confused on 3 L oxygen.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '64-year-old man, ASA 3, for lumbar fusion. He has obesity and limited mobility but no prior VTE, cancer, or thrombophilia. He remains ambulatory with a cane. Renal and pulmonary function are normal.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '72-year-old woman, ASA 3, for partial laryngectomy. She has prior radiation, dysphagia, weak cough, and recurrent aspiration bronchitis. Oxygen saturation is 94%. Kidney function and cognition are normal.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '68-year-old man, ASA 4, for open thoracoabdominal aneurysm repair. He has CKD stage 4, diabetes, and creatinine 2.3 mg/dL. He is cognitively intact and has clear lungs.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '80-year-old woman, ASA 3, for elective cataract surgery under monitored anesthesia. She has mild cognitive impairment and uses a hearing aid but lives independently, takes no sedatives, and has stable organ function.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '70-year-old man, ASA 3, for radical cystectomy. He has prior unprovoked DVT, obesity, active bladder cancer, and reduced walking after chemotherapy. Apixaban is held.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '46-year-old woman, ASA 2, for laparoscopic hysterectomy. She has controlled asthma, normal renal function, no prior VTE, and no cognitive concerns.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '78-year-old man, ASA 4, for urgent CABG after acute coronary syndrome. He has CKD stage 3b, diabetes, EF 30%, moderate COPD with chronic sputum, and creatinine 1.9 mg/dL. He is frail, hard of hearing, and had delirium during a prior ICU stay.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '60-year-old woman, ASA 3, for open colectomy. She has bronchiectasis with daily sputum and recurrent antibiotic courses. Oxygen saturation is 93%. Renal function and cognition are normal.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '73-year-old man, ASA 3, for total knee arthroplasty. He had a PE after prior surgery, has BMI 38, and is limited to household walking. Warfarin is held. Lungs and kidneys are stable.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '66-year-old woman, ASA 3, for partial nephrectomy. She has CKD stage 3a but normal contralateral kidney function, controlled blood pressure, and no diabetes. Creatinine is stable at 1.3 mg/dL.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '82-year-old man, ASA 3, for elective hernia repair. He has moderate dementia, visual impairment, and takes diphenhydramine nightly. He needs help with all medications. Lungs and renal function are normal.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '56-year-old man, ASA 3, for hepatectomy. He has cirrhosis, ascites, albumin 2.7 g/dL, and creatinine 1.4 mg/dL. He is oriented and has clear lungs.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '75-year-old woman, ASA 3, for VATS lobectomy. She has mild COPD, quit smoking 25 years ago, oxygen saturation 97%, and good exercise tolerance. Renal function and cognition are normal.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '69-year-old man, ASA 3, for revision hip arthroplasty. He has a prior unprovoked DVT, obesity, and poor mobility because of prosthetic pain. Apixaban is held. Renal and pulmonary function are normal.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '50-year-old woman, ASA 2, for laparoscopic cholecystectomy. She has controlled hypothyroidism and no pulmonary, renal, thrombotic, or cognitive disease. She remains active.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '84-year-old man, ASA 4, for urgent decompression after cauda equina syndrome. He has mild dementia, chronic opioid use, and CKD stage 3a. He is dehydrated after poor intake and intermittently disoriented. Lungs are clear.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '63-year-old woman, ASA 3, for esophagectomy. She has COPD, active smoking, chronic cough, and oxygen saturation of 92%. She had aspiration pneumonia three months ago. Renal function and cognition are normal.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '72-year-old man, ASA 3, for open nephrectomy. He has CKD stage 3b, diabetes, and baseline creatinine 1.8 mg/dL. He is active and cognitively intact.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '59-year-old woman, ASA 3, for pelvic exenteration. She has recurrent cervical cancer, prior PE, leg edema, and limited mobility. Therapeutic anticoagulation is held. Lungs and kidneys are stable.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '41-year-old man, ASA 1, for outpatient umbilical hernia repair. He has no chronic disease and normal exercise tolerance, renal function, pulmonary status, and cognition.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '80-year-old woman, ASA 3, for elective colectomy. She has mild dementia, hearing loss, and prior postoperative confusion. She takes tramadol and gabapentin. Renal and pulmonary function are stable.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '67-year-old man, ASA 4, for urgent aortic valve replacement. He has CKD stage 4, heart failure, diabetes, moderate COPD, and a weak cough after prior stroke. Creatinine is 2.5 mg/dL. He previously became confused during a heart-failure admission.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '58-year-old woman, ASA 3, for VATS lobectomy. She has severe asthma with recent steroid burst and persistent wheeze. Oxygen saturation is 93%. Kidney function and cognition are normal.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '74-year-old man, ASA 3, for total knee arthroplasty. He has prior DVT, obesity, and chronic venous insufficiency. Apixaban is held. He walks only short distances due to pain.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '52-year-old woman, ASA 2, for laparoscopic colectomy. She has controlled diabetes but normal creatinine, good exercise tolerance, and no lung disease or VTE history.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '86-year-old man, ASA 4, for emergency surgery after bowel perforation. He has vascular dementia, CKD stage 4, hypotension, and witnessed aspiration. Creatinine is 3.1 mg/dL from 2.3; he is confused and hypoxemic.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '65-year-old woman, ASA 3, for radical nephrectomy. She has CKD stage 3a and hypertension but no diabetes; creatinine is 1.4 mg/dL. She remains active with clear lungs and normal cognition.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '71-year-old man, ASA 3, for laryngectomy. He has COPD, active smoking, thick secretions, and poor cough from prior radiation. Oxygen saturation is 91%.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '66-year-old woman, ASA 3, for revision ankle surgery. She has active breast cancer, prior DVT, and has been non-weight-bearing for six weeks. Enoxaparin is held. Renal and pulmonary function are normal.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '38-year-old woman, ASA 1, for laparoscopic ovarian cystectomy. She is healthy, physically active, and has no relevant medical history.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '79-year-old man, ASA 3, for elective TURP. He has moderate dementia, severe hearing loss, and nightly lorazepam use. His daughter manages medications. Renal and pulmonary function are normal.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '62-year-old man, ASA 4, for CABG. He has CKD stage 3b, diabetes, and EF 30%; creatinine is 1.9 mg/dL. He is cognitively intact and has no active respiratory infection.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '69-year-old woman, ASA 3, for VATS decortication. She has bronchiectasis, chronic sputum, and recent empyema. Oxygen saturation is 92%. Renal function and cognition are normal.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '73-year-old man, ASA 3, for open colectomy. He has prior PE, active colon cancer, obesity, and limited walking. Warfarin is held. Kidney and lung function are stable.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '55-year-old woman, ASA 2, for laparoscopic adrenalectomy. She has controlled hypertension and normal kidney, lung, cognitive, and thrombotic status.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '82-year-old woman, ASA 4, for urgent hip-fracture repair. She has mild dementia, CKD stage 3a, dehydration, and prior postoperative confusion. She is afebrile with clear lungs.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '64-year-old man, ASA 3, for partial nephrectomy. He has normal contralateral kidney function, creatinine 0.9 mg/dL, no diabetes, and good functional status.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '76-year-old woman, ASA 3, for open esophagectomy. She has severe COPD on home oxygen, recurrent aspiration, a weak cough, mild cognitive impairment, and prior postoperative confusion. Oxygen saturation is 89% on room air.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '68-year-old man, ASA 3, for total hip arthroplasty. He had an unprovoked DVT five years ago, has BMI 37, and walks only indoors. Apixaban is held. Renal and pulmonary function are normal.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '49-year-old woman, ASA 2, for laparoscopic hysterectomy. She has no chronic cardiopulmonary, renal, thrombotic, or cognitive disease and remains active.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '85-year-old man, ASA 4, for emergency colectomy for perforation. He has moderate dementia, CKD stage 3b, hypotension, and new oxygen requirement after vomiting. Creatinine is 2.3 mg/dL from 1.6 and he is confused.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '60-year-old woman, ASA 3, for right upper lobectomy. She has COPD, active smoking, chronic productive cough, and FEV1 45% predicted. Oxygen saturation is 91%.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '72-year-old man, ASA 3, for open nephroureterectomy. He has CKD stage 3b, diabetes, and baseline creatinine 1.8 mg/dL. Cognition and lungs are normal.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '65-year-old woman, ASA 3, for pelvic tumor resection. She has prior PE, active malignancy, obesity, and limited mobility due to pain. Rivaroxaban is held.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '35-year-old man, ASA 1, for outpatient arthroscopic shoulder repair. He is healthy, athletic, and has no relevant medical history.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '79-year-old woman, ASA 3, for elective colectomy. She has mild cognitive impairment, severe hearing loss, and takes amitriptyline. She remains independent with help for medications. Organ function is stable.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '67-year-old man, ASA 4, for open aortic aneurysm repair. He has CKD stage 4, diabetes, and creatinine 2.4 mg/dL. He is cognitively intact with clear lungs.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '61-year-old woman, ASA 3, for esophagectomy. She has dysphagia, recurrent aspiration bronchitis, COPD, and oxygen saturation of 93%. Kidney function and cognition are normal.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '74-year-old man, ASA 3, for revision knee arthroplasty. He has prior DVT, chronic leg swelling, obesity, and poor mobility. Apixaban is held.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '58-year-old woman, ASA 2, for laparoscopic colectomy. She has controlled hypertension and normal kidney, lung, cognitive, and thrombotic status.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '83-year-old man, ASA 4, for urgent bowel resection after obstruction. He has Parkinson disease, mild dementia, dysphagia, and poor intake. Creatinine is 1.7 mg/dL from 1.2; he is inattentive and coughing after sips.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '63-year-old man, ASA 3, for radical nephrectomy. He has CKD stage 3a and diabetes; creatinine is 1.5 mg/dL. He remains active and cognitively intact.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '70-year-old woman, ASA 3, for VATS lobectomy. She has moderate COPD, home oxygen at night, and chronic sputum. Oxygen saturation is 91%.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '66-year-old man, ASA 3, for open prostatectomy. He has a remote provoked DVT, obesity, and limited mobility but no active cancer beyond localized disease. Renal and pulmonary function are normal.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '42-year-old woman, ASA 1, for laparoscopic cholecystectomy. She is healthy, active, and takes no daily medications.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '81-year-old woman, ASA 3, for elective mastectomy. She has moderate dementia, visual impairment, and nightly zolpidem. Her daughter manages all medications. Lungs and kidney function are stable.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '59-year-old man, ASA 4, for CABG. He has CKD stage 3b, diabetes, and EF 25%; creatinine is 1.9 mg/dL. Lungs are clear and cognition is intact.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '68-year-old woman, ASA 3, for open thoracotomy. She has bronchiectasis, recurrent pneumonias, and chronic purulent sputum. Oxygen saturation is 92%.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '72-year-old man, ASA 3, for pelvic sarcoma resection. He has prior PE, active malignancy, obesity, and reduced mobility. Anticoagulation is held.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '54-year-old woman, ASA 2, for laparoscopic adrenalectomy. She has normal renal and pulmonary function, no VTE history, and intact cognition.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '87-year-old man, ASA 4, for emergency hip-fracture repair. He has moderate dementia, CKD stage 3b, dehydration, and prior delirium. He is bedbound and disoriented; lungs are clear.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '62-year-old woman, ASA 3, for partial nephrectomy. She has normal baseline creatinine, no diabetes, good exercise capacity, and preserved contralateral kidney function.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '75-year-old man, ASA 3, for esophagectomy. He has COPD, active smoking, recurrent aspiration, hearing loss, and a history of confusion after ICU admission. Oxygen saturation is 91%.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '69-year-old woman, ASA 3, for total knee arthroplasty. She has prior unprovoked DVT, BMI 36, and limited walking due to pain. Apixaban is held. Renal and pulmonary function are normal.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '45-year-old man, ASA 2, for laparoscopic appendectomy. He has controlled hypertension and otherwise normal renal, pulmonary, cognitive, and thrombotic history.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '84-year-old woman, ASA 4, for emergency colectomy. She has CKD stage 4, moderate dementia, hypotension, and witnessed aspiration. Creatinine is 3.0 mg/dL from 2.2; she is confused and requires oxygen.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '62-year-old man, ASA 3, for lobectomy. He has severe COPD, active smoking, chronic productive cough, and oxygen saturation of 90%. Renal function and cognition are normal.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '71-year-old woman, ASA 3, for nephroureterectomy. She has CKD stage 3b, diabetes, and creatinine 1.8 mg/dL. Lungs are clear and cognition is intact.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '64-year-old man, ASA 3, for open colectomy for cancer. He has prior PE, obesity, and limited mobility after chemotherapy. Warfarin is held. Renal and pulmonary function are stable.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '37-year-old woman, ASA 1, for laparoscopic ovarian cystectomy. She is healthy, active, and has no relevant medical history.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '80-year-old man, ASA 3, for TURP. He has Alzheimer disease, hearing loss, and chronic benzodiazepine use. He is oriented only to person and place. Renal and pulmonary function are stable.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '66-year-old woman, ASA 4, for CABG. She has CKD stage 4, diabetes, and EF 30%; creatinine is 2.5 mg/dL. She is cognitively intact with clear lungs.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '59-year-old man, ASA 3, for esophagectomy. He has COPD, active smoking, recurrent aspiration, and a weak cough. Oxygen saturation is 92%.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '73-year-old woman, ASA 3, for revision hip arthroplasty. She has prior DVT, obesity, and poor mobility. Apixaban is held. Kidneys and lungs are stable.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '52-year-old man, ASA 2, for robotic sigmoid colectomy after recurrent diverticulitis. Diabetes is controlled with metformin, creatinine is 0.8 mg/dL, and he walks four miles daily. He has no chronic lung disease, prior VTE, or cognitive concerns.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '86-year-old woman, ASA 4, for urgent bowel resection. She has Parkinson disease, moderate dementia, dysphagia, poor intake, and CKD stage 3a. She is confused, coughing after liquids, and creatinine is 1.7 mg/dL from 1.2.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '63-year-old man, ASA 3, for open radical nephrectomy of a 7-cm renal mass. Hypertensive nephropathy has reduced eGFR to 48 mL/min with creatinine 1.5 mg/dL. He has no diabetes, remains active, and is cognitively intact.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '70-year-old woman, ASA 3, for VATS lobectomy. She has severe COPD and uses nocturnal oxygen. Oxygen saturation is 91% with chronic sputum.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '67-year-old man, ASA 3, for prostatectomy. He has obesity and a remote provoked DVT but walks three miles daily and is not currently anticoagulated. Organ function is normal.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '40-year-old woman, ASA 1, for laparoscopic cholecystectomy. She is healthy and physically active with normal renal and pulmonary function.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '82-year-old woman, ASA 3, for mastectomy. She has mild dementia, severe hearing loss, and takes zolpidem nightly. Her daughter manages medications. Lungs and kidneys are stable.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '58-year-old man, ASA 4, for open aortic aneurysm repair. He has CKD stage 3b, diabetes, and creatinine 1.9 mg/dL. Cognition and lungs are normal.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '65-year-old woman, ASA 3, for thoracotomy. She has bronchiectasis, recurrent pneumonias, and daily purulent sputum. Oxygen saturation is 92%.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '74-year-old man, ASA 3, for pelvic cancer resection. He has prior PE, obesity, and limited mobility. Anticoagulation is held.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '55-year-old woman, ASA 2, for adrenalectomy. She has controlled hypertension, normal organ function, and no VTE or cognitive history.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '88-year-old man, ASA 4, for emergency hip-fracture repair. He has dementia, CKD stage 3b, dehydration, and prior delirium. He is bedbound and disoriented.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '60-year-old woman, ASA 3, for partial nephrectomy. She has normal creatinine, no diabetes, and preserved contralateral kidney function.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '76-year-old man, ASA 3, for esophagectomy. He has COPD, active smoking, recurrent aspiration, a weak cough, mild cognitive impairment, and nightly zolpidem use. Oxygen saturation is 91%.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '65-year-old woman, ASA 3, for total hip arthroplasty. She has a prior unprovoked DVT, BMI 39, and walks only with a walker. Apixaban is held. Renal and pulmonary function are stable.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '48-year-old man, ASA 2, for laparoscopic hernia repair. He has no chronic disease, exercises regularly, and has normal organ function.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '83-year-old woman, ASA 4, for emergency colectomy. She has CKD stage 3b, dementia, sepsis, hypotension, and a new right-lower-lobe infiltrate after emesis. Creatinine is 2.2 mg/dL from 1.5 and she is confused.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '61-year-old man, ASA 3, for lobectomy. He has COPD, active smoking, chronic sputum, and FEV1 47% predicted. Oxygen saturation is 91%.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '70-year-old woman, ASA 3, for nephroureterectomy. She has CKD stage 3b, diabetes, and creatinine 1.8 mg/dL. She is active and cognitively intact.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '63-year-old man, ASA 3, for open colon-cancer resection. He has a prior PE, obesity, active cancer, and limited mobility. Warfarin is held.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '36-year-old woman, ASA 1, for laparoscopic salpingectomy. She is stable, healthy, and has normal kidney and lung function with no VTE history.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '81-year-old man, ASA 3, for TURP. He has moderate dementia, hearing loss, and takes lorazepam nightly. He is dependent for medications.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '67-year-old woman, ASA 4, for CABG. She has CKD stage 4, diabetes, EF 28%, and creatinine 2.4 mg/dL. Lungs are clear and cognition is intact.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '60-year-old man, ASA 3, for esophagectomy. He has COPD, ongoing smoking, recurrent aspiration, and weak cough. Oxygen saturation is 92%.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '72-year-old woman, ASA 3, for revision knee arthroplasty. She has prior DVT, obesity, and household-only mobility. Apixaban is held.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '51-year-old man, ASA 2, for laparoscopic colectomy. He has controlled hypertension and normal organ function with good exercise tolerance.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '85-year-old woman, ASA 4, for urgent bowel resection. She has Parkinson disease, dementia, dysphagia, CKD stage 3a, and dehydration. She is confused and coughing with liquids.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '64-year-old man, ASA 3, for radical nephrectomy. He has CKD stage 3a and diabetes; creatinine is 1.5 mg/dL.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '71-year-old woman, ASA 3, for VATS lobectomy. She has severe COPD, nocturnal oxygen use, and chronic sputum. Oxygen saturation is 90%.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '66-year-old man, ASA 3, for prostatectomy. He has obesity and a remote provoked DVT but remains very active and has no current anticoagulation.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '43-year-old woman, ASA 1, for laparoscopic cholecystectomy. She is healthy and takes no daily medication.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '80-year-old woman, ASA 3, for mastectomy. She has mild cognitive impairment, visual loss, and nightly zolpidem. Her family manages medications.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '59-year-old man, ASA 4, for open aneurysm repair. He has CKD stage 3b, diabetes, and creatinine 1.9 mg/dL.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '67-year-old woman, ASA 3, for thoracotomy. She has bronchiectasis, recurrent respiratory infections, and daily sputum. Oxygen saturation is 92%.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '73-year-old man, ASA 3, for pelvic sarcoma resection. He has prior PE, obesity, active malignancy, and limited mobility. Anticoagulation is held.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '54-year-old woman, ASA 2, for adrenalectomy. She has normal organ function and no VTE or cognitive history.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '87-year-old man, ASA 4, for emergency hip-fracture repair. He has dementia, CKD stage 3b, dehydration, and prior postoperative confusion. He is bedbound and disoriented.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '61-year-old woman, ASA 3, for partial nephrectomy. She has normal creatinine and good contralateral kidney function without diabetes.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '75-year-old man, ASA 3, for esophagectomy. He has COPD, active smoking, prior aspiration pneumonia, severe hearing loss, and previous postoperative agitation. Oxygen saturation is 91%.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': 'Preoperative medicine consult for a 57-year-old man undergoing open resection of a retroperitoneal sarcoma. He has a chronic left-leg DVT diagnosed four months ago, tumor compression of the iliac vein, and marked reduction in walking. Therapeutic apixaban is stopped. Creatinine is 0.9 mg/dL; lungs and cognition are normal.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': 'Anesthesia assessment: 32-year-old woman, ASA 1, for outpatient laparoscopic sterilization. She takes no medications, runs half-marathons, and has no cardiopulmonary, renal, thrombotic, or neurologic history.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': 'A 79-year-old man with severe aortic stenosis is scheduled for valve replacement. He has CKD stage 3b, insulin-treated diabetes, and EF 35%; creatinine is 1.9 mg/dL. He remains oriented but is frail and requires help with shopping. Chest examination is clear.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': 'Pulmonary pre-op note for a 66-year-old woman before right lower lobectomy. She has moderate COPD, a 50-pack-year history, persistent morning sputum, and DLCO 42% predicted. Oxygen saturation is 91% on room air. Kidney function and cognition are normal.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '68-year-old man, ASA 3, for open repair of a tibial nonunion. He has been mostly wheelchair-bound for two months and had a DVT after the initial fracture. Warfarin is held. He has no lung disease, creatinine is 0.8 mg/dL, and cognition is intact.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': 'A 73-year-old woman is planned for laparoscopic right colectomy. She has CKD stage 3a with stable creatinine 1.3 mg/dL and well-controlled hypertension, but no diabetes or recent illness. She walks daily, is cognitively intact, and has clear lungs.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': 'Emergency surgery evaluation for an 88-year-old woman with incarcerated hernia and small-bowel ischemia. She has Alzheimer disease, CKD stage 3b, two days of vomiting, and systolic pressure in the 80s. Creatinine is 2.2 mg/dL from 1.4; she is disoriented but lungs remain clear.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '63-year-old man, ASA 3, scheduled for total laryngectomy after chemoradiation. He has copious secretions, impaired swallowing, chronic aspiration, and COPD. Oxygen saturation is 92%, with coarse breath sounds. Renal function and cognition are normal.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '44-year-old woman, ASA 2, for elective laparoscopic nephrectomy of a nonfunctioning duplicated kidney. Her contralateral renal function is normal, creatinine is 0.7 mg/dL, and she has no diabetes, lung disease, prior VTE, or cognitive concerns.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '76-year-old man, ASA 3, for transurethral bladder tumor resection. He has mild cognitive impairment, macular degeneration, and takes cyclobenzaprine and diphenhydramine at night. He lives alone but his son organizes medications. Lungs and renal function are stable.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': 'A 61-year-old woman with ovarian cancer is undergoing open cytoreductive surgery. She has a prior PE, ascites causing reduced mobility, BMI 35, and chronic apixaban therapy that is held. Creatinine is 0.8 mg/dL and pulmonary examination is normal.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '39-year-old man, ASA 1, for elective laparoscopic inguinal hernia repair. He exercises five days weekly and has no chronic illness, medication use, or prior surgical complications.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': 'A 70-year-old man is scheduled for open pancreaticoduodenectomy. He has diabetes, CKD stage 3b, poor oral intake, and creatinine 1.9 mg/dL from a baseline of 1.5. He is cognitively intact and has no active pulmonary disease.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '81-year-old woman, ASA 4, for urgent hip-fracture fixation. She has moderate dementia, prior postoperative delirium, severe hearing loss, and has been bedbound for three days. Creatinine is 1.0 mg/dL and lungs are clear.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': 'Thoracic surgery consult for a 58-year-old man undergoing pneumonectomy. He has severe emphysema, continues to smoke, uses oxygen with exertion, and has FEV1 39% predicted. He reports daily sputum but no current fever. Renal function and cognition are normal.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '65-year-old woman, ASA 3, for elective lumbar fusion. She has obesity and reduced mobility from pain but no personal or family VTE history, no cancer, and remains ambulatory with a cane. Kidney and lung function are normal.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': 'A 74-year-old man with ischemic cardiomyopathy is scheduled for CABG. He has CKD stage 4, diabetes, EF 25%, mild cognitive impairment, and prior confusion after cardiac catheterization. Creatinine is 2.5 mg/dL; lungs are clear after diuresis.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '52-year-old woman, ASA 2, for robotic hysterectomy. She has controlled hypothyroidism, normal renal function, no lung disease or VTE history, and walks several miles daily.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '77-year-old man, ASA 3, for esophagectomy. Prior stroke left dysphagia and a weak cough; he has had two aspiration pneumonias. He also has mild cognitive impairment and prior postoperative confusion. Oxygen saturation is 94%.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '69-year-old woman, ASA 3, for radical cystectomy. She has a history of unprovoked DVT, chronic leg edema, and reduced mobility after chemotherapy. Rivaroxaban is held. Pulmonary and renal function are stable.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': 'A 56-year-old man is scheduled for laparoscopic partial nephrectomy. Creatinine is 0.9 mg/dL, the opposite kidney is normal, and he has no diabetes or vascular disease. He remains active with normal cognition and lungs.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '83-year-old woman, ASA 4, requires emergency laparotomy for perforated diverticulitis. She has vascular dementia, CKD stage 3b, septic hypotension, and a new oxygen requirement after vomiting. Creatinine is 2.4 mg/dL from 1.6; she is agitated and inattentive.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '62-year-old man, ASA 3, for right upper lobectomy. He has mild COPD but stopped smoking 22 years ago, FEV1 is 78% predicted, oxygen saturation is 97%, and he walks three miles daily. Kidney function and cognition are normal.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '71-year-old woman, ASA 3, for revision total knee arthroplasty. She had a prior postoperative PE, is obese, and walks only with a walker. Apixaban is held. Lungs and renal function are stable.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '45-year-old man, ASA 2, for outpatient shoulder arthroscopy. He has controlled asthma without recent symptoms, normal exercise tolerance, and no renal, thrombotic, or cognitive disease.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '80-year-old man, ASA 3, for elective colectomy. He has mild Alzheimer disease, hearing impairment, and a history of confusion after hospitalization. He takes gabapentin for neuropathy. Renal function and lungs are stable.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': 'A 64-year-old woman with renal-cell carcinoma is planned for radical nephrectomy. She has CKD stage 3b, diabetes, and baseline creatinine 1.8 mg/dL. She is active, oriented, and has no lung disease.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': 'Pulmonary evaluation for a 68-year-old man before open esophagectomy notes COPD, ongoing smoking, chronic sputum, and recurrent aspiration from tumor-related dysphagia. Oxygen saturation is 91%. Renal function and cognition are normal.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '73-year-old woman, ASA 3, for open fixation of a periprosthetic femur fracture. She has active breast cancer, prior DVT, and has been non-weight-bearing for four days. Enoxaparin is held. Kidney and lung function are stable.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '35-year-old woman, ASA 1, for elective laparoscopic cholecystectomy. She has no chronic illness, exercises regularly, and takes no medication.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '78-year-old man, ASA 4, for urgent mitral valve replacement. He has CKD stage 3b, diabetes, chronic bronchitis with daily sputum, and baseline creatinine 1.9 mg/dL. He is frail, mildly forgetful, and uses hearing aids.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '59-year-old woman, ASA 3, for VATS lobectomy. She has bronchiectasis, daily sputum, and three respiratory infections in the last year. Oxygen saturation is 93%. Creatinine and cognition are normal.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '67-year-old man, ASA 3, for open prostatectomy. He has obesity, prior DVT, and limited mobility from spinal stenosis. Warfarin is held. Renal and pulmonary function are normal.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '54-year-old woman, ASA 2, for laparoscopic colectomy. She has normal kidney and lung function, no VTE history, intact cognition, and good exercise tolerance.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '86-year-old woman, ASA 4, for emergency repair of a strangulated hernia. She has moderate dementia, CKD stage 3a, poor intake, and systolic pressure in the 90s. Creatinine is 1.7 mg/dL from 1.1; she is disoriented but lungs are clear.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '66-year-old man, ASA 3, for right hepatectomy. He has cirrhosis, albumin 2.8 g/dL, ascites, and creatinine 1.3 mg/dL. He is oriented and has no chronic lung disease.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '72-year-old woman, ASA 3, for total hip arthroplasty. She has prior PE, obesity, and markedly reduced mobility. Apixaban is held. Lungs and kidneys are stable.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '47-year-old man, ASA 2, for laparoscopic fundoplication. He has reflux without aspiration, normal lung and kidney function, and no VTE or cognitive history.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '82-year-old man, ASA 3, for TURP. He has Parkinson disease, mild dementia, visual impairment, and chronic clonazepam use. His daughter manages medications.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '60-year-old woman, ASA 3, for open thoracotomy. She has severe COPD, active smoking, chronic sputum, and oxygen saturation of 90%. Renal function and cognition are normal.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '69-year-old man, ASA 4, for open aortic aneurysm repair. He has CKD stage 4, diabetes, and creatinine 2.4 mg/dL. He is cognitively intact and has clear lungs.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '64-year-old woman, ASA 3, for pelvic cancer resection. She has prior unprovoked DVT, active cancer, chronic leg edema, and limited mobility. Rivaroxaban is held.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '42-year-old man, ASA 1, for outpatient hand surgery. He is healthy, active, and takes no daily medication.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '84-year-old woman, ASA 4, for urgent hip-fracture repair. She has Alzheimer disease, dysphagia after stroke, CKD stage 3a, and has been bedbound for two days. She is confused and coughs after liquids.', 'DVT': 1, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '57-year-old man, ASA 3, for partial nephrectomy. He has CKD stage 3a but stable creatinine 1.3 mg/dL, no diabetes, and a normal contralateral kidney. He exercises regularly.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '75-year-old woman, ASA 3, for esophagectomy. She has COPD, recurrent aspiration, a weak cough, severe hearing loss, and prior delirium after abdominal surgery. Oxygen saturation is 92%.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '63-year-old man, ASA 3, for revision ankle fusion. He has prior DVT, obesity, and has been non-weight-bearing for seven weeks. Anticoagulation is held.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '50-year-old woman, ASA 2, for robotic hysterectomy. She has controlled hypertension, normal organ function, and no VTE or cognitive history.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '79-year-old man, ASA 4, for emergency bowel resection. He has CKD stage 3b, sepsis, hypotension, and new confusion. Creatinine is 2.3 mg/dL from 1.6; chest examination is clear.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '62-year-old woman, ASA 3, for VATS lobectomy. She has mild COPD, stopped smoking 20 years ago, oxygen saturation 97%, and good exercise tolerance.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '70-year-old man, ASA 3, for open colectomy for colon cancer. He has a prior PE, obesity, and reduced mobility after chemotherapy-induced neuropathy. Warfarin is held. Renal and pulmonary function are stable.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '46-year-old woman, ASA 2, for laparoscopic appendectomy. She has no chronic disease and normal functional capacity, kidney function, pulmonary status, and cognition.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '85-year-old man, ASA 4, for emergency laparotomy for bowel ischemia. He has vascular dementia, CKD stage 4, hypotension, and witnessed aspiration. Creatinine is 3.0 mg/dL from 2.2; he is disoriented and hypoxemic.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '61-year-old woman, ASA 3, for open lobectomy. She has severe COPD, active smoking, daily sputum, and oxygen saturation of 90%. Renal function and cognition are normal.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '68-year-old man, ASA 3, for radical nephrectomy. He has CKD stage 3b, diabetes, and creatinine 1.8 mg/dL. He is active with normal lungs and cognition.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '65-year-old woman, ASA 3, for pelvic sarcoma resection. She has prior DVT, active cancer, chronic venous edema, and limited mobility. Apixaban is held.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '38-year-old man, ASA 1, for outpatient inguinal hernia repair. He is healthy, active, and takes no medication.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '82-year-old woman, ASA 3, for elective colectomy. She has moderate dementia, severe hearing loss, and chronic zolpidem use. Her daughter provides medication support. Renal and pulmonary function are stable.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '59-year-old man, ASA 4, for CABG. He has CKD stage 4, diabetes, EF 30%, and creatinine 2.4 mg/dL. He is oriented and has clear lungs.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '67-year-old woman, ASA 3, for esophagectomy. She has COPD, recurrent aspiration, chronic cough, and oxygen saturation of 92%. Kidney function and cognition are normal.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '73-year-old man, ASA 3, for revision knee arthroplasty. He has prior PE, obesity, and limited walking. Warfarin is held.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '52-year-old woman, ASA 2, for laparoscopic colectomy. She has controlled hypertension, normal organ function, and no VTE or cognitive history.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '87-year-old woman, ASA 4, for urgent bowel resection. She has Parkinson disease, dementia, dysphagia, CKD stage 3a, dehydration, and confusion. She coughs after sips of water.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '64-year-old man, ASA 3, for nephroureterectomy. He has CKD stage 3b and diabetes; creatinine is 1.8 mg/dL.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '70-year-old woman, ASA 3, for VATS lobectomy. She has severe COPD, nocturnal oxygen use, and chronic productive cough. Oxygen saturation is 91%.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '66-year-old man, ASA 3, for prostatectomy. He has obesity and a remote travel-associated DVT but remains active and has no current cancer spread or anticoagulation.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '41-year-old woman, ASA 1, for laparoscopic cholecystectomy. She is healthy and physically active.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '80-year-old man, ASA 3, for TURP. He has mild dementia, visual impairment, and takes diphenhydramine nightly. His spouse manages medications.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '60-year-old woman, ASA 4, for open aneurysm repair. She has CKD stage 3b, diabetes, and creatinine 1.9 mg/dL.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '68-year-old man, ASA 3, for thoracotomy. He has bronchiectasis, recurrent pneumonias, and daily purulent sputum. Oxygen saturation is 92%.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '72-year-old woman, ASA 3, for pelvic cancer resection. She has prior DVT, obesity, and limited mobility. Apixaban is held.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '55-year-old man, ASA 2, for adrenalectomy. He has normal organ function and no relevant history.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '88-year-old woman, ASA 4, for urgent fixation of a displaced intertrochanteric fracture. She has Lewy body dementia, CKD stage 3b, three days of poor intake, and has not left bed since the fall. Creatinine is 1.9 mg/dL from 1.4; she is hallucinating and disoriented.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '61-year-old man, ASA 3, for partial nephrectomy. He has normal creatinine, no diabetes, and preserved contralateral renal function.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '76-year-old woman, ASA 3, for esophagectomy. She has COPD, active smoking, prior aspiration pneumonia, mild cognitive impairment, and chronic lorazepam use. Oxygen saturation is 91%.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': 'Pre-op clinic note for a 62-year-old man before open pelvic exenteration for recurrent rectal cancer. He had an unprovoked DVT last year, has left-leg edema, and spends most of the day seated because of pelvic pain. Apixaban is held. Creatinine is 0.9 mg/dL; lungs and cognition are normal.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': 'A 43-year-old woman is scheduled for laparoscopic endometriosis excision. She has no chronic medical conditions, takes no daily medications, and remains physically active. Kidney and lung function are normal.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '79-year-old man, ASA 4, for urgent CABG after myocardial infarction. He has CKD stage 3b, diabetes, EF 25%, and severe COPD requiring nocturnal oxygen. Creatinine is 2.0 mg/dL. He has mild cognitive impairment and prior postoperative confusion.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': 'Thoracic pre-op assessment for a 65-year-old woman scheduled for right pneumonectomy. She has severe COPD, ongoing tobacco use, chronic sputum, and DLCO 38% predicted. Oxygen saturation is 90% on room air.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '70-year-old man, ASA 3, for open partial nephrectomy. He has a solitary kidney, CKD stage 3b, diabetes, and baseline creatinine 1.9 mg/dL. He is oriented and has clear lungs.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '67-year-old woman, ASA 3, for revision total hip arthroplasty. She has prior PE, BMI 41, and household-only mobility. Rivaroxaban is held. Renal and pulmonary function are stable.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '50-year-old man, ASA 2, for laparoscopic cholecystectomy. He has controlled hypertension, normal organ function, and exercises regularly.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '84-year-old woman, ASA 3, for elective mastectomy. She has moderate dementia, severe hearing loss, and takes lorazepam at bedtime. Family manages medications. Lungs and kidneys are stable.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '63-year-old man, ASA 4, for open thoracoabdominal aneurysm repair. He has CKD stage 4, diabetes, and creatinine 2.5 mg/dL. Cognition is intact and lungs are clear.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '69-year-old woman, ASA 3, for transhiatal esophagectomy. Achalasia has caused nightly regurgitation and two aspiration events; she also has moderate COPD and a weak cough. Oxygen saturation is 93%.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '74-year-old man, ASA 3, for open colectomy for cancer. He has prior DVT, obesity, and limited mobility. Apixaban is held.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '56-year-old woman, ASA 2, for laparoscopic adrenalectomy. She has controlled diabetes with normal creatinine and no lung, VTE, or cognitive history.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '86-year-old man, ASA 4, for emergency bowel resection. He has dementia, CKD stage 3b, septic hypotension, and witnessed aspiration. Creatinine is 2.5 mg/dL from 1.7; he is confused and hypoxemic.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '65-year-old woman, ASA 3, for radical nephrectomy. She has CKD stage 3a and hypertension; creatinine is 1.5 mg/dL. She remains active and cognitively intact.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '71-year-old man, ASA 3, for right middle and lower bilobectomy. He has emphysema requiring oxygen with exertion, FEV1 42% predicted, and a recent COPD flare treated with prednisone. Room-air saturation is 91%.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '68-year-old woman, ASA 3, for robotic hysterectomy. She has obesity and a remote pregnancy-associated DVT but remains active with no current anticoagulation. Renal and pulmonary function are normal.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '39-year-old man, ASA 1, for outpatient shoulder arthroscopy. He is healthy and takes no medication.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '81-year-old woman, ASA 3, for colectomy. She has mild cognitive impairment, visual loss, and takes amitriptyline. She needs help with medications.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '58-year-old man, ASA 4, for CABG. He has CKD stage 3b, diabetes, EF 30%, and creatinine 1.9 mg/dL.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '66-year-old woman, ASA 3, for open pleural decortication after recurrent empyema. She has cystic bronchiectasis, daily green sputum, reduced breath sounds on the left, and room-air saturation of 93%.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '73-year-old man, ASA 3, for pelvic sarcoma resection. He has prior PE, obesity, active malignancy, and poor mobility. Warfarin is held.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '53-year-old woman, ASA 2, for laparoscopic colectomy. She has normal organ function and no VTE or cognitive history.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '87-year-old woman, ASA 4, for urgent hip-fracture repair. She has dementia, CKD stage 3a, dehydration, and prior confusion. She is bedbound and disoriented.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '60-year-old man, ASA 3, for partial nephrectomy. He has normal creatinine and preserved contralateral kidney function without diabetes.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '75-year-old woman, ASA 3, for esophagectomy. She has COPD, aspiration history, a weak cough, hearing loss, and previous postoperative confusion. Oxygen saturation is 91%.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': 'A 58-year-old man with locally advanced bladder cancer is scheduled for radical cystectomy. He has a proximal DVT diagnosed three months ago, persistent leg swelling, and reduced walking after chemotherapy. Therapeutic apixaban is held. Creatinine is 1.0 mg/dL; lungs and cognition are normal.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': 'Pre-op anesthesia note: 34-year-old woman, ASA 1, for laparoscopic ovarian cystectomy. She is healthy, takes no daily medication, and has normal exercise capacity, kidney function, pulmonary status, and cognition.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': 'An 82-year-old man is admitted for urgent fixation of an intertrochanteric fracture. He has Alzheimer disease, CKD stage 3a, poor intake since the fall, and has been bedbound for 48 hours. Creatinine is 1.6 mg/dL from 1.1; he is disoriented but lungs are clear.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': 'Pulmonary clearance for a 64-year-old woman before esophagectomy documents severe COPD, continued smoking, chronic sputum, and recurrent aspiration from dysphagia. Room-air saturation is 90%. Renal function and cognition are normal.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '70-year-old man, ASA 3, planned for radical nephrectomy. He has CKD stage 3b, insulin-treated diabetes, and baseline creatinine 1.9 mg/dL. He is euvolemic, oriented, and has clear lungs.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '67-year-old woman, ASA 3, for revision total knee arthroplasty. She has a prior postoperative PE, BMI 39, and walks only with a walker. Rivaroxaban is held. Renal and pulmonary function are stable.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '49-year-old man, ASA 2, for laparoscopic sigmoid colectomy. Hypertension is controlled, renal and pulmonary function are normal, and he exercises regularly. No prior VTE or cognitive concerns are reported.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': 'An 85-year-old woman is scheduled for elective mastectomy. She has moderate dementia, severe hearing loss, and takes temazepam nightly. Her son manages medications. Lungs and renal function are stable.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '62-year-old man, ASA 4, for open abdominal aortic aneurysm repair. He has CKD stage 4, diabetes, peripheral vascular disease, and creatinine 2.6 mg/dL. Cognition is intact and lungs are clear.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': 'A 69-year-old woman is scheduled for right upper lobectomy. She has bronchiectasis, daily mucopurulent sputum, and two pneumonias in the past year. Oxygen saturation is 92%. Kidney function and cognition are normal.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '74-year-old man, ASA 3, for open resection of rectal cancer. He has prior unprovoked DVT, obesity, and limited mobility from neuropathy. Warfarin is held. Renal and pulmonary function are stable.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '56-year-old woman, ASA 2, for laparoscopic adrenalectomy. She has controlled hypertension, normal renal function, no lung disease, no VTE history, and intact cognition.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': 'Emergency general-surgery note for an 87-year-old man with perforated diverticulitis. He has vascular dementia, CKD stage 3b, septic hypotension, and aspiration during vomiting. Creatinine is 2.5 mg/dL from 1.7; he is confused and needs 4 L oxygen.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '65-year-old woman, ASA 3, for nephroureterectomy. She has CKD stage 3a and hypertension; creatinine is 1.5 mg/dL. She is active with normal lungs and cognition.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '71-year-old man, ASA 3, for VATS lobectomy. He has severe COPD, home oxygen at night, and chronic productive cough. Oxygen saturation is 90%.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '68-year-old woman, ASA 3, for robotic hysterectomy. She has BMI 36 and a remote pregnancy-associated DVT but remains active and has no ongoing anticoagulation. Renal and pulmonary function are normal.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '40-year-old man, ASA 1, for outpatient excision of a foot neuroma. He is healthy, fully mobile, and takes no medication.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '80-year-old woman, ASA 3, for elective colectomy. She has mild cognitive impairment, macular degeneration, and takes oxybutynin. She needs help organizing medications. Lungs and kidneys are stable.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '59-year-old man, ASA 4, for CABG. He has CKD stage 3b, diabetes, EF 28%, and creatinine 1.9 mg/dL. He is oriented and has no active lung infection.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '67-year-old woman, ASA 3, for open thoracotomy. She has bronchiectasis, recurrent pneumonias, chronic sputum, and oxygen saturation of 92%.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '72-year-old man, ASA 3, for pelvic sarcoma resection. He has a prior PE, active malignancy, obesity, and poor mobility. Apixaban is held.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '53-year-old woman, ASA 2, for laparoscopic colectomy. She has controlled diabetes with creatinine 0.8 mg/dL, good exercise tolerance, and no lung disease or VTE history.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': 'An 88-year-old woman requires urgent hip-fracture repair. She has dementia, CKD stage 3b, dehydration, prior delirium, and has been bedbound for two days. She is disoriented; lungs are clear.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '61-year-old man, ASA 3, for laparoscopic partial nephrectomy. Creatinine is 0.9 mg/dL, the opposite kidney is normal, and he has no diabetes or vascular disease.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '76-year-old woman, ASA 3, for three-field esophagectomy. She has emphysema requiring oxygen with exertion, recurrent regurgitation, and a prior aspiration admission. She is frail, uses hearing aids, and previously became delirious after bowel surgery. Resting saturation is 92%.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '66-year-old man, ASA 3, for open colectomy. He has an unprovoked DVT history, active colon cancer, obesity, and walks only short distances. Warfarin is held. Kidney and lung function are stable.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '44-year-old woman, ASA 2, for laparoscopic hysterectomy. She is healthy apart from controlled hypothyroidism, exercises regularly, and has no renal, pulmonary, thrombotic, or cognitive disease.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '83-year-old man, ASA 4, for emergency bowel resection. He has Parkinson disease, moderate dementia, dysphagia, CKD stage 3a, and poor intake. He is confused, coughs with liquids, and creatinine is 1.8 mg/dL from 1.2.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '63-year-old woman, ASA 3, for left lower lobectomy. She has steroid-dependent asthma, recent hospitalization for status asthmaticus, persistent wheeze, and room-air saturation of 93%.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '69-year-old man, ASA 3, for radical nephrectomy. He has CKD stage 3b, diabetes, and creatinine 1.8 mg/dL. He is active and oriented.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '65-year-old woman, ASA 3, for pelvic cancer surgery. She has prior PE, chronic leg edema, obesity, and poor mobility. Rivaroxaban is held.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '36-year-old man, ASA 1, for outpatient arthroscopic knee surgery. He is healthy, fully mobile, and has no previous VTE or organ disease.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '81-year-old woman, ASA 3, for mastectomy. She has moderate dementia, hearing loss, and nightly lorazepam use. Family manages medications.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '60-year-old man, ASA 4, for open aneurysm repair. He has CKD stage 4, diabetes, and creatinine 2.5 mg/dL. Cognition is intact and lungs are clear.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '68-year-old woman, ASA 3, for esophagectomy. She has COPD, recurrent aspiration, chronic cough, and oxygen saturation of 92%.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '73-year-old man, ASA 3, for revision knee arthroplasty. He has prior DVT, obesity, and poor mobility. Apixaban is held.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '51-year-old woman, ASA 2, for laparoscopic colectomy. She has normal renal and pulmonary function, good exercise tolerance, and no VTE or cognitive history.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '86-year-old man, ASA 4, for emergency right hemicolectomy after cecal perforation. He has Alzheimer disease, CKD stage 3b, norepinephrine-dependent sepsis, and a new left-basilar infiltrate after vomiting. Creatinine is 2.5 mg/dL from 1.6; he is combative.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '64-year-old woman, ASA 3, for nephroureterectomy. She has CKD stage 3b and diabetes; creatinine is 1.8 mg/dL.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '70-year-old man, ASA 3, for VATS resection of a right lower-lobe cancer. He has chronic bronchitis, uses 2 L oxygen overnight, and reports yellow sputum most mornings. FEV1 is 46% predicted.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '67-year-old woman, ASA 3, for robotic hysterectomy. She has obesity and a remote provoked DVT but remains active and has no current anticoagulation.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '42-year-old man, ASA 1, for outpatient hand surgery. He is healthy and takes no medications.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '80-year-old woman, ASA 3, for open right colectomy. She has mild cognitive impairment, glaucoma with poor vision, and takes nortriptyline for neuropathy. She cooks independently but her son fills pill boxes.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '61-year-old man, ASA 4, for CABG after unstable angina. He has CKD stage 3b, insulin-treated diabetes, EF 32%, and creatinine 1.8 mg/dL. He is oriented and has clear lungs after diuresis.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '66-year-old woman, ASA 3, for posterolateral thoracotomy and pleural peel. She has non-CF bronchiectasis, recurrent pseudomonal infections, copious morning sputum, and oxygen saturation of 93%.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '72-year-old man, ASA 3, for pelvic sarcoma resection. He has prior PE, active cancer, obesity, and poor mobility. Warfarin is held.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '54-year-old woman, ASA 2, for laparoscopic adrenalectomy. She has normal organ function and no relevant history.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '87-year-old woman, ASA 4, for urgent cephalomedullary nailing of a hip fracture. She has vascular dementia, CKD stage 3a, two days of minimal intake, and prior hallucinations after anesthesia. She is bedbound, dry-mouthed, and disoriented.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '61-year-old man, ASA 3, for partial nephrectomy. He has normal creatinine and preserved contralateral kidney function.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '75-year-old woman, ASA 3, for Ivor Lewis esophagectomy. She has emphysema, nocturnal regurgitation, prior aspiration pneumonia, a weak cough after an old stroke, hearing loss, and prior postoperative delirium. Resting oxygen saturation is 92%.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': 'Preoperative evaluation for a 57-year-old woman undergoing open ovarian-cancer debulking. She has a proximal DVT diagnosed six months ago, persistent pelvic venous compression, and now walks only within her home. Apixaban is held. Creatinine is 0.8 mg/dL; pulmonary examination and cognition are normal.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': 'A 46-year-old man, ASA 2, is scheduled for laparoscopic cholecystectomy. He has controlled hypertension, jogs regularly, and has normal renal function, clear lungs, no VTE history, and intact cognition.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': 'An 84-year-old woman requires emergency laparotomy for perforated bowel. She has Alzheimer disease, CKD stage 3b, septic shock, and aspiration during repeated emesis. Creatinine is 2.6 mg/dL from 1.7; she is agitated and needs 4 L oxygen.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': 'Thoracic surgery note for a 62-year-old man before right upper lobectomy. He has severe COPD, continues to smoke, produces sputum each morning, and has DLCO 40% predicted. Oxygen saturation is 90% on room air.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '70-year-old woman, ASA 3, for open nephrectomy. She has CKD stage 3b, diabetes, and baseline creatinine 1.9 mg/dL. She remains independent, oriented, and free of active lung disease.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '65-year-old man, ASA 3, for revision total hip arthroplasty. He has a prior unprovoked PE, BMI 38, and household-only mobility. Rivaroxaban is held. Renal and pulmonary function are stable.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '38-year-old woman, ASA 1, for outpatient breast biopsy. She is healthy, takes no daily medication, and has no cardiopulmonary, renal, thrombotic, or cognitive history.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': 'An 82-year-old man is scheduled for TURP. He has Parkinson disease, mild dementia, poor vision, and takes clonazepam at night. His spouse manages all medications. Kidney and lung function are stable.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '60-year-old woman, ASA 4, for open repair of a thoracoabdominal aneurysm. She has CKD stage 4, diabetes, and creatinine 2.5 mg/dL. She is cognitively intact and has no active pulmonary process.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': 'A 68-year-old man is planned for esophagectomy. Prior stroke left dysphagia and a weak cough; he has COPD and two episodes of aspiration pneumonia. Oxygen saturation is 92%. Renal function and cognition are normal.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '73-year-old woman, ASA 3, for open resection of endometrial-cancer recurrence. She has prior DVT, chronic leg edema, obesity, and reduced mobility. Apixaban is held.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '55-year-old man, ASA 2, for laparoscopic colectomy. He has controlled hypertension, normal kidney and lung function, and exercises regularly. No prior VTE or cognitive symptoms are reported.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': 'Emergency evaluation of an 86-year-old man with strangulated bowel obstruction. He has vascular dementia, CKD stage 3b, two days of vomiting, and systolic pressure of 84 mmHg. Creatinine is 2.3 mg/dL from 1.5; he is disoriented but lungs are clear.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '64-year-old woman, ASA 3, for nephroureterectomy. She has CKD stage 3b, hypertension, and diabetes; creatinine is 1.8 mg/dL. She is active and oriented.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '71-year-old man, ASA 3, for VATS lobectomy. He has bronchiectasis, chronic purulent sputum, and recurrent respiratory infections. Oxygen saturation is 92%.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '67-year-old woman, ASA 3, for robotic hysterectomy. She has BMI 35 and a remote pregnancy-associated DVT but is independently mobile and has no current anticoagulation. Organ function is normal.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '40-year-old man, ASA 1, for outpatient carpal-tunnel release. He is healthy, active, and takes no medications.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '80-year-old woman, ASA 3, for elective colectomy. She has mild cognitive impairment, hearing loss, and takes oxybutynin and gabapentin. Her daughter organizes medications. Lungs and kidneys are stable.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '59-year-old man, ASA 4, for CABG. He has CKD stage 3b, diabetes, EF 30%, and creatinine 1.9 mg/dL. He is cognitively intact and has clear lungs.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '66-year-old woman, ASA 3, for open thoracotomy. She has severe COPD, active smoking, daily sputum, and oxygen saturation of 90%.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '72-year-old man, ASA 3, for pelvic sarcoma resection. He has prior PE, active malignancy, obesity, and limited mobility. Warfarin is held.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '53-year-old woman, ASA 2, for laparoscopic adrenalectomy. She has controlled diabetes with normal creatinine and no pulmonary, thrombotic, or cognitive disease.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': 'An 88-year-old woman is scheduled for urgent hip-fracture repair. She has moderate dementia, CKD stage 3a, poor intake, prior delirium, and has been bedbound for three days. She is disoriented; lungs are clear.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '61-year-old man, ASA 3, for robotic partial nephrectomy. His creatinine is 0.9 mg/dL, the opposite kidney is normal, and he has no diabetes or vascular disease.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '76-year-old woman, ASA 3, for esophagectomy. She has COPD on nocturnal oxygen, recurrent aspiration, a weak cough, severe hearing loss, and prior postoperative delirium. Oxygen saturation is 91%.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': 'A 63-year-old man with colon cancer is scheduled for open colectomy. He has a previous unprovoked DVT, BMI 36, and reduced mobility from chemotherapy neuropathy. Apixaban is held. Kidney and lung function are stable.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '47-year-old woman, ASA 2, for laparoscopic myomectomy. She has no chronic medical conditions, good exercise tolerance, and normal renal and pulmonary function.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': 'An 83-year-old man requires emergency small-bowel resection. He has Parkinson disease, dementia, dysphagia, CKD stage 3a, and poor intake. He is confused, coughs with water, and creatinine is 1.8 mg/dL from 1.2.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '62-year-old woman, ASA 3, for lobectomy. She has severe COPD, active smoking, chronic productive cough, and FEV1 44% predicted. Oxygen saturation is 90%.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '69-year-old man, ASA 3, for radical nephrectomy. He has CKD stage 3b, diabetes, and baseline creatinine 1.8 mg/dL. Cognition and lungs are normal.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '65-year-old woman, ASA 3, for pelvic-cancer resection. She has prior PE, chronic leg edema, obesity, and household-only mobility. Rivaroxaban is held.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '36-year-old man, ASA 1, for outpatient knee arthroscopy. He is healthy, fully mobile, and has no VTE, renal, pulmonary, or cognitive history.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '81-year-old woman, ASA 3, for mastectomy. She has moderate dementia, macular degeneration, and nightly lorazepam use. Family manages medications. Organ function is stable.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '60-year-old man, ASA 4, for open aneurysm repair. He has CKD stage 4, diabetes, peripheral vascular disease, and creatinine 2.5 mg/dL.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '68-year-old woman, ASA 3, for esophagectomy. She has COPD, chronic aspiration, weak cough, and oxygen saturation of 92%.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '75-year-old man, ASA 3, for revision total knee arthroplasty after prosthetic loosening. He had a DVT after the original operation, has BMI 34, and is limited to transfers. Apixaban is held.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '51-year-old woman, ASA 2, for robotic right colectomy for a large benign polyp. She has no chronic disease, creatinine 0.7 mg/dL, oxygen saturation 99%, and walks five miles on weekends.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '86-year-old man, ASA 4, for emergency colectomy. He has dementia, CKD stage 3b, septic hypotension, and aspiration during vomiting. Creatinine is 2.4 mg/dL from 1.6; he is confused and hypoxemic.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '64-year-old woman, ASA 3, for nephroureterectomy. She has CKD stage 3b, diabetes, and creatinine 1.8 mg/dL.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '70-year-old man, ASA 3, for left upper lobectomy after induction chemotherapy. He has severe centrilobular emphysema, desaturates to 86% with walking, and recently completed antibiotics for bronchitis.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '67-year-old woman, ASA 3, for robotic hysterectomy. She has obesity and a remote provoked DVT but remains active and is not anticoagulated.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '42-year-old man, ASA 1, for outpatient trigger-finger release under regional anesthesia. He is a recreational cyclist, takes no medication, and has no renal, pulmonary, thrombotic, or cognitive history.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '80-year-old woman, ASA 3, for laparoscopic left colectomy. She has early Alzheimer disease, cataracts, and chronic amitriptyline use for postherpetic neuralgia. She needs reminders for medications but performs basic activities independently.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '56-year-old man, ASA 4, for four-vessel CABG. He has diabetic nephropathy with creatinine 2.0 mg/dL and ischemic cardiomyopathy with EF 29%. Pulmonary examination is clear and cognition is intact.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '68-year-old woman, ASA 3, for left thoracotomy and decortication after recurrent empyema. She has bronchiectasis, daily green sputum, oxygen saturation of 93%, and two antibiotic courses in the last six months.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '71-year-old man, ASA 3, for resection of a pelvic chordoma. He has a prior PE, active malignancy, BMI 36, and walks with a walker because of sacral pain. Warfarin is held.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '57-year-old woman, ASA 2, for laparoscopic adrenalectomy for a small aldosteronoma. Blood pressure is controlled on two agents, creatinine is 0.7 mg/dL, and she hikes weekly. She has no lung disease, prior VTE, or cognitive symptoms.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '86-year-old woman, ASA 4, for urgent fixation of a subtrochanteric hip fracture. She has moderate dementia, CKD stage 3a, poor intake, and prior postoperative confusion. She has been bedbound for four days and is disoriented.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '63-year-old man, ASA 3, for robotic partial nephrectomy of a 2-cm renal mass. Creatinine is 0.8 mg/dL, the contralateral kidney is normal, and he has no diabetes or vascular disease. He cycles regularly.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '77-year-old woman, ASA 3, for open esophagectomy. She has COPD on nocturnal oxygen, recurrent aspiration from dysphagia, a weak cough, mild cognitive impairment, and prior postoperative confusion. Room-air saturation is 90%.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': 'Preoperative consult for a 64-year-old man undergoing open rectal-cancer resection. He has a prior unprovoked DVT, BMI 37, chronic leg edema, and walks only short distances. Apixaban is held. Renal and pulmonary function are stable.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': 'A 45-year-old woman, ASA 2, is scheduled for laparoscopic cholecystectomy. She has controlled hypothyroidism, exercises regularly, and has normal renal, pulmonary, thrombotic, and cognitive status.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': 'An 85-year-old man requires emergency bowel resection for perforation. He has vascular dementia, CKD stage 3b, septic hypotension, and aspiration after vomiting. Creatinine is 2.5 mg/dL from 1.7; he is confused and needs supplemental oxygen.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': 'Thoracic evaluation for a 63-year-old woman before lobectomy. She has severe COPD, continues to smoke, has daily sputum, and FEV1 43% predicted. Oxygen saturation is 90% on room air.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '70-year-old man, ASA 3, for radical nephrectomy. He has CKD stage 3b, diabetes, and baseline creatinine 1.9 mg/dL. He is oriented and has clear lungs.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '66-year-old woman, ASA 3, for pelvic-cancer debulking. She has prior PE, chronic leg edema, obesity, and household-only mobility. Rivaroxaban is held.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '37-year-old man, ASA 1, for outpatient shoulder arthroscopy. He is healthy, athletic, and has no relevant medical history.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '82-year-old woman, ASA 3, for mastectomy. She has moderate dementia, severe hearing loss, and nightly temazepam use. Her family manages medications. Organ function is stable.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '60-year-old man, ASA 4, for open thoracoabdominal aneurysm repair. He has CKD stage 4, diabetes, and creatinine 2.5 mg/dL.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '69-year-old woman, ASA 3, for esophagectomy. She has COPD, recurrent aspiration, a weak cough, and oxygen saturation of 92%.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '74-year-old man, ASA 3, for revision knee arthroplasty. He has prior DVT, BMI 38, and poor mobility. Apixaban is held.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '52-year-old woman, ASA 2, for laparoscopic colectomy. She has normal renal and pulmonary function, intact cognition, and no VTE history.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '87-year-old man, ASA 4, for damage-control laparotomy after perforated sigmoid colon. He has vascular dementia, CKD stage 3b, septic shock, and copious emesis before intubation. Creatinine is 2.3 mg/dL from 1.5; he is lethargic and needs oxygen.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '65-year-old woman, ASA 3, for nephroureterectomy. She has CKD stage 3b, diabetes, and creatinine 1.8 mg/dL.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '71-year-old man, ASA 3, for VATS right lower lobectomy. He has alpha-1 antitrypsin deficiency with severe obstruction, uses nocturnal oxygen, and has a weak, nonproductive cough. Resting saturation is 90%.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '68-year-old woman, ASA 3, for robotic hysterectomy. She has obesity and a remote pregnancy-associated DVT but remains active and is not anticoagulated.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '41-year-old man, ASA 1, for excision of a dorsal wrist ganglion under monitored anesthesia. He is healthy, works in construction, and has no chronic medication use or relevant medical history.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '81-year-old woman, ASA 3, for colectomy. She has mild cognitive impairment, macular degeneration, and takes amitriptyline. She needs help with medications.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 1},
        {'clinical_note': '59-year-old man, ASA 4, for urgent CABG after NSTEMI. Diabetic nephropathy has produced creatinine 2.0 mg/dL, and ischemic cardiomyopathy has reduced EF to 27%. He is euvolemic after diuresis.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 0},
        {'clinical_note': '67-year-old woman, ASA 3, for open thoracotomy with drainage of chronic empyema. She has traction bronchiectasis, frequent antibiotic courses, coarse left-sided crackles, and resting saturation of 92%.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '73-year-old man, ASA 3, for en bloc resection of a pelvic chondrosarcoma. He has a remote PE, current iliac-vein compression, BMI 35, and uses a wheelchair for pelvic pain. Warfarin is held.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '49-year-old woman, ASA 2, scheduled for robotic adrenalectomy for a benign cortisol-producing adenoma. Diabetes is diet controlled, renal function is normal, and she has excellent exercise tolerance without pulmonary, thrombotic, or cognitive history.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '88-year-old woman, ASA 4, for hemiarthroplasty after a displaced femoral-neck fracture. She has Alzheimer disease, CKD stage 3a, four days of poor intake, and previous postoperative confusion. She is bedbound and oriented only to person.', 'DVT': 1, 'Pneumonia': 0, 'AKI': 1, 'Delirium': 1},
        {'clinical_note': '62-year-old man, ASA 3, for robotic partial nephrectomy of a 1.8-cm renal mass. Baseline creatinine is 0.8 mg/dL, the other kidney is normal, and he has no diabetes or vascular disease. He swims three times weekly.', 'DVT': 0, 'Pneumonia': 0, 'AKI': 0, 'Delirium': 0},
        {'clinical_note': '76-year-old woman, ASA 3, for transhiatal esophagectomy. She uses 2 L oxygen at night for COPD, has recurrent choking with meals, prior aspiration pneumonia, severe hearing loss, and previous postoperative agitation.', 'DVT': 0, 'Pneumonia': 1, 'AKI': 0, 'Delirium': 1},
    ]
    return pd.DataFrame(rows, columns=["clinical_note", "DVT", "Pneumonia", "AKI", "Delirium"])
