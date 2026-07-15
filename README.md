

## Overview

SurgicalPLAN (*Surgical **P**ostoperative Risk Prediction with **L**anguage Models **A**dapting to Clinical **N**otes*) is a Python package for predicting postoperative risks from clinical notes using language models. It provides flexible and clinically oriented workflows that support a range of perioperative use cases, enabling clinicians, researchers, and healthcare institutions to train and fine-tune models using preoperative or intraoperative clinical text.

The package is designed to be accessible to a broad range of users, including clinicians, surgeons, and researchers with limited programming experience. It minimizes the need to interact with lower-level machine learning frameworks such as PyTorch. With just a few lines of high-level functions, users can begin training and fine-tuning their own models.

SurgicalPLAN supports multiple modeling strategies, including:

1. **Direct inference** with fine-tuned language models  
2. **(Joint) Semi-supervised learning** approaches for leveraging partially labeled data  
3. A **multi-task learning framework** that enables simultaneous prediction of multiple postoperative outcomes  

The package was developed for the American College of Surgeons (ACS) workshop, *AI for Clinicians and Surgeons: A Hands-On Introduction Across the Care Continuum*.

The accompanying work is:  
[*The foundational capabilities of large language models in predicting postoperative risks using clinical notes*](https://www.nature.com/articles/s41746-025-01489-2)  
Alba, Xue, Abraham, Kannampallil, and Lu (2025), *npj Digital Medicine*

---

## Installation

```bash
pip install surgicalplan
```

Because `torch` CUDA wheels aren't hosted on PyPI, install PyTorch first matching your GPU's CUDA version, then install this package. For example, on a machine with CUDA 11.8 drivers:

```bash
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install surgicalplan
```

**Python version**: 3.9–3.12 (tested on 3.12).

---

## Quick example

```python
import pandas as pd
from surgicalplan import mtl_finetune, get_postoperative_outcome_scores

df = pd.read_csv("my_clinical_data.csv")
# df columns: "clinical_note", "Outcome_1", "Outcome_2", "Outcome_3", "Outcome_4"

# 1. Fine-tune
mtl_finetune(
    df,
    text_col="clinical_note",
    outcome_cols=["Outcome_1", "Outcome_2", "Outcome_3", "Outcome_4"],
    output_dir="my_finetuned_model",
)




# 2. Score a new scenario

note_1 = (
    "83-year-old male, ASA 4, scheduled for coronary artery bypass graft (emergent three-vessel). "
    "Indication: severe CAD with LAD stenosis, presenting with unstable angina. "
    "PMH: COPD, type 2 diabetes mellitus, coronary artery disease, prior MI, chronic kidney disease stage 3. "
    "Social: current smoker, 1 pack per day. "
    "BMI 34 (obese). "
    "Home medications: metoprolol, aspirin 81 mg, atorvastatin, insulin glargine, furosemide. "
    "Allergies: NKDA. "
    "Preop labs within acceptable limits. Consent obtained, plan to proceed."
)


scores = get_postoperative_outcome_scores(
    "my_finetuned_model",
    note_1
)
# {'Outcome_1': 0.12, 'Outcome_2': 0.28, 'Outcome_3': 0.04, 'Outcome_4': 0.39}
```

---

## API reference

### Direct inference 

Allows users to use out-of-the-box models that have already been trained on clinical data and its associated post-operative outcomes. Unlike the later ones, this is a direct inference function that loads a pre-trained, ready-to-use model from HuggingFace Hub and therefore requires no model training. 

The default model is `cja5553/BJH-perioperative-notes-bioClinicalBERT`, which is our a Bio+ClinicalBERT model variant that was multi-task fine-tuned across 6 postoperative outcomes: (1) death in 30, (2) DVT, (3) PE, (4) AKI, (5) delirium and (6) Pneumonoia. This model was used in our [accompanying *npj Digital Medicine* paper](https://www.nature.com/articles/s41746-025-01489-2).

#### `direct_inference_from_trained_model`

Score clinical text against a pre-trained multi-task model without any fine-tuning step. The model is downloaded from HuggingFace Hub on first use and cached locally thereafter.

**Example**

```python
from surgicalplan import direct_inference_from_trained_model

note = (
    "Redo coronary artery bypass graft with aortic valve replacement "
    "bioprosthetic. Indication: severe ischemic cardiomyopathy, "
    "ejection fraction 25 percent, prior MI, ventricular arrhythmia "
    "status post AICD placement, stage 3 chronic kidney disease, COPD."
)

scores = direct_inference_from_trained_model(text=note)
# {'DVT': 0.17, 'PE': 0.06, 'PNA': 0.28, 'postop_del': 0.81,
#  'death_in_30': 0.46, 'post_aki_status': 0.93}
```

**Parameters**

- `text` (`str | list[str]`, *required*): One clinical scenario, or a list of them. Determines the shape of the return value.
- `outcomes` (`list[str] | None`, default: `None`): Which outcomes to score. Defaults to all outcomes the default model was trained on (`DVT`, `PE`, `PNA`, `postop_del`, `death_in_30`, `post_aki_status`), recovered from the model's `mtl_metadata.json`. Pass a subset to score only some.
- `model_name` (`str`, default: `"cja5553/BJH-perioperative-notes-bioClinicalBERT"`): HuggingFace repo ID or local path. Override to use your own fine-tuned model.
- `max_length` (`int | None`, default: `None`): Token sequence length. Defaults to the value used during fine-tuning, recovered from metadata.
- `device` (`str | None`, default: `None`): `"cuda"`, `"cpu"`, or `None` to auto-detect.
- `hf_token` (`str | None`, default: `None`): Optional HuggingFace token, required only if the model repo is gated/private.

**Returns**

- `dict[str, float]` when `text` is a string — maps each outcome name to a probability in `[0, 1]`.
- `list[dict[str, float]]` when `text` is a list — one dict per input, in the same order.

**Notes**

- First call downloads the model (~440 MB) from HuggingFace and caches it locally; subsequent calls use the cache.
- Inference runs on CPU in ~5 seconds per note, or ~0.5 seconds with a GPU.
- For users who want to fine-tune their own model, see `mtl_finetune` (multi-outcome) or `joint_finetune` (single-outcome).

---

### Joint or semi-supervised finetuning 

Joint Single-Outcome Finetuning trains a separate model for each postoperative outcome of interest. The jointly learns the structure of your clinical notes whilst learns to predict the outcome, ensuring the model captures both the linguistic patterns of your institution's documentation style and the clinical features that drive your specific outcomes. Unlike the below `MultiTaskLearningPrediction`, this is catered to a single specific outcome as opposed to multiple outcomes. 

![description of joint JointFinetuning](https://raw.githubusercontent.com/cja5553/ACS_demo_postoperative_risk_prediction_with_clinical_notes/main/joint_supervision_visualization.jpg)

#### `JointFinetuning`

Perform Joint (or semi-supervised) finetuning. 



**Example**

```python
joint_finetune(
    df,
    text_col="clinical_note",
    outcome_col="DVT",
    output_dir="DVT_model",
    training_configs={
        "num_train_epochs": 3,
        "per_device_train_batch_size": 16,
        "evaluation_strategy": "steps",
        "eval_steps": 100,
        "logging_steps": 100,
        "learning_rate": 2e-5,
    },
)
```

Fine-tune Bio+ClinicalBERT on MLM jointly with a single binary classification head for one outcome.

**Parameters**

- `df` (`pandas.DataFrame`, *required*): Must contain `text_col` and `outcome_col`.
- `text_col` (`str`, *required*): Name of the free-text column.
- `outcome_col` (`str`, *required*): Name of a single binary (0/1) outcome column. Rows with NaN in this column are dropped before training.
- `output_dir` (`str`, default `"joint_finetuned"`): Directory to save the fine-tuned model, tokenizer, and metadata. Also used as the HuggingFace Trainer `output_dir` for checkpoints and logs.
- `base_model` (`str`, default `"emilyalsentzer/Bio_ClinicalBERT"`): HuggingFace model id to start from. Any BERT-architecture model should work.
- `hf_token` (`str | None`, default `None`): Optional HuggingFace token for gated/private base models. If `None`, uses the cached CLI login when present.
- `max_length` (`int`, default `512`): Token sequence length for tokenization.
- `lambda_constant` (`float`, default `2`): Weight on the auxiliary (BCE) loss relative to MLM loss. Total loss = MLM + λ · BCE.
- `mlm_probability` (`float`, default `0.15`): Token masking probability for MLM.
- `val_fraction` (`float`, default `1/8`): Fraction of `df` held out for validation during training.
- `weight` (`torch.Tensor | None`, default `None`): Optional `pos_weight` for `BCEWithLogitsLoss` to handle class imbalance. Useful for rare outcomes (e.g., `torch.tensor([20.0])` for ~5% positive prevalence).
- `training_configs` (`dict | None`, default `None`): Any keyword arguments accepted by `transformers.TrainingArguments`. User-provided values override the defaults below. Default `training_configs` is `{"num_train_epochs": 5, "per_device_train_batch_size": 24, "per_device_eval_batch_size": 24, "learning_rate": 1e-5, "warmup_ratio": 0.06, "weight_decay": 1e-3, "logging_steps": 1000, "save_strategy": "epoch", "seed": 42, "report_to": "none"}`.

**Returns**

`str` — the `output_dir` path. After training, this directory contains:

- `pytorch_model.bin` (or `model.safetensors`) — model weights
- `config.json` — model architecture config
- `tokenizer.json`, `vocab.txt`, `tokenizer_config.json`, `special_tokens_map.json` — tokenizer
- `joint_metadata.json` — records `outcome_col`, `text_col`, `max_length`, `base_model`, `lambda_constant`, `num_tasks` (always 1), and `workflow` so inference can recover them automatically
- `checkpoint-*` — per-epoch training checkpoints (can be deleted after training)
- `logs/` — TensorBoard-compatible training logs

---

#### `get_outcome_score`

Score a text scenario (or list of scenarios) against the single auxiliary head of a joint-finetuned model.

**Example**

```python
get_outcome_score(
    model_name="DVT_model",
    text="83-year-old male, ASA 4, scheduled for CABG. PMH: COPD, diabetes.",
)
```

**Parameters**

- `model_name` (`str`, *required*): Path to a directory saved by `joint_finetune`.
- `text` (`str | list[str]`, *required*): One scenario string, or a list of them. Determines the shape of the return value.
- `max_length` (`int | None`, default: `None`): Token sequence length. Defaults to the value used during fine-tuning, recovered from `joint_metadata.json`, otherwise `512`.
- `device` (`str | None`, default: `None`): `"cuda"`, `"cpu"`, or `None` to auto-detect.
- `hf_token` (`str | None`, default: `None`): Optional HuggingFace token for gated/private models.

**Returns**

- `float` when `text` is a string — the predicted probability for the trained outcome, in `[0, 1]`.
- `list[float]` when `text` is a list — one probability per input, in the same order.


---


### Multi-task finetuning  

Multi-Task Learning (MTL) allows you to train a single versatile model capable of predicting multiple postoperative outcomes from the same clinical notes. Unlike traditional finetuning strategies — where you'd need to train a single model for each outcome — MTL allows you to create a model capable of simultaneously predicting multiple risks — analogous to foundation models. 

![description of MTL](https://raw.githubusercontent.com/cja5553/ACS_demo_postoperative_risk_prediction_with_clinical_notes/main/MTL_illustration.jpg)

#### `MultiTaskLearningPrediction`



Performs MTL finetuning. 


**Example**  

```python
mtl_finetune(
    df,
    text_col="clinical_note",
    outcome_cols=["death_30d", "dvt", "pneumonia", "aki", "AUR", "PE"],
    output_dir="my_run",
    training_configs={
        "num_train_epochs": 3,
        "per_device_train_batch_size": 16,
        "evaluation_strategy": "steps",
        "eval_steps": 100,
        "logging_steps": 100,     
        "learning_rate": 2e-5
        }
)
```

Fine-tune Bio+ClinicalBERT on MLM jointly with one binary classification head per outcome.

**Parameters**

- `df` (`pandas.DataFrame`, *required*): Must contain `text_col` and all `outcome_cols`.  
- `text_col` (`str`, *required*): Name of the free-text column.  
- `outcome_cols` (`list[str]`, *required*): Names of binary (0/1) outcome columns. One auxiliary head is trained per outcome. Rows with NaN in a given outcome are dropped for that outcome's task but used for the others.   
- `output_dir` (`str`, default `"mtl_finetuned"`): Directory to save the fine-tuned model, tokenizer, and metadata. Also used as the HuggingFace Trainer `output_dir` for checkpoints and logs.  
- `base_model` (`str`, default `"emilyalsentzer/Bio_ClinicalBERT"`): HuggingFace model id to start from. Any BERT-architecture model should work.  
- `max_length` (`int`, default `512`): Token sequence length for tokenization.  
- `lambda_constant` (`float`, default `2`): Weight on the auxiliary (per-outcome BCE) loss relative to MLM loss. Total loss = MLM + λ · mean(per-task BCE).   
- `val_fraction` (`float`, default `1/8`): Fraction of `df` held out for validation during training.  
- `training_configs` (`dict | None`, default `None`): Any keyword arguments accepted by `transformers.TrainingArguments`. User-provided values override the defaults below.  Default `training_configs` is `{"num_train_epochs": 5, "per_device_train_batch_size": 24, "per_device_eval_batch_size": 24, "learning_rate": 1e-5, "warmup_ratio": 0.06, "weight_decay": 1e-3, "logging_steps": 1000 "save_strategy": "epoch", "seed": 42,}`




**Returns**

`str` — the `output_dir` path. After training, this directory contains:

- `pytorch_model.bin` (or `model.safetensors`) — model weights
- `config.json` — model architecture config
- `tokenizer.json`, `vocab.txt`, `tokenizer_config.json`, `special_tokens_map.json` — tokenizer
- `mtl_metadata.json` — records `outcome_cols`, `text_col`, `max_length`, `base_model`, `lambda_constant`, `num_tasks` so inference can recover them automatically
- `checkpoint-*` — per-epoch training checkpoints (can be deleted after training)
- `logs/` — TensorBoard-compatible training logs



---

#### `get_postoperative_outcome_scores`



Score a text scenario (or list of scenarios) against each auxiliary head of a fine-tuned MTL model.

**Example**

```python
get_postoperative_outcome_scores(
    model_name,
    text,
    outcomes=["death_30d", "dvt", "pneumonia", "aki", "AUR", "PE"],
)
```


**Parameters**

- `model_name` (`str`, required): Path to a directory saved by `mtl_finetune`.
- `text` (`str | list[str]`, required): One scenario string, or a list of them. Determines the shape of the return value.
- `outcomes` (`list[str] | None`, default: `None`): Which outcomes to score. Defaults to all outcomes the model was trained on, recovered from `mtl_metadata.json`. Pass a subset to score only some. Names must match those used in `mtl_finetune`.
- `max_length` (`int | None`, default: `None`): Token sequence length. Defaults to the value used during fine-tuning, recovered from metadata, otherwise `512`.
- `device` (`str | None`, default: `None`): `"cuda"`, `"cpu"`, or `None` to auto-detect.


**Returns**

- `dict[str, float]` when `text` is a string — maps each outcome name to a probability in `[0, 1]`.
- `list[dict[str, float]]` when `text` is a list — one dict per input, in the same order.


---

### Pseudo data

#### `get_pseudo_data`

Returns a fixed dataset of 500 hand-written preoperative clinical notes with
hand-assigned binary outcomes, for testing and demonstration. The notes and
labels are curated rather than generated: each note was written by hand, and
each label assigned by reading that note. Labels correlate with clinical
content, and include deliberately discordant cases, so a fine-tuned model
learns probabilistic rather than deterministic associations.

**Outcome prevalence is deliberately inflated** above real-world incidence
(true postoperative DVT/pneumonia/AKI run ~1-2%) so that a model has
recoverable signal at n=500. These are not epidemiological estimates.

**Example**

```python
df = get_pseudo_data()
print(df.shape)             # (500, 5)
print(df.columns.tolist())  # ['clinical_note', 'DVT', 'Pneumonia', 'AKI', 'Delirium']
```

**Parameters**

None.

**Returns**

`pandas.DataFrame` with 500 rows and 5 columns:

- `clinical_note` (`str`) — hand-written preoperative note.
- `DVT`, `Pneumonia`, `AKI`, `Delirium` (`int`, 0/1) — hand-assigned outcomes.

---

