---
title: 'SurgicalPLAN: A Python package for postoperative risk prediction and training from clinical notes'
tags:
  - Python
  - clinical natural language processing
  - perioperative care
  - multi-task learning
  - language models
  - risk prediction
authors:
  - name: Charles Alba
    orcid: 0000-0001-7711-360X
    affiliation: 1
  - name: Chenyang Lu
    orcid: 0000-0003-1709-6769
    affiliation: 1
affiliations:
  - index: 1
    name: Washington University in St. Louis, United States
    ror: 01yc7t268
date: 20 May 2026
bibliography: paper.bib
---

# Summary

Before surgery, clinicians document notes describing the patient
and the planned procedure. These narratives carry rich information that is not
easily captured by the structured, tabular fields of an electronic health
record. `SurgicalPLAN` is a Python package that can be trained from such preoperative
notes to predict a patient's risk of postoperative complications, such as
30-day mortality, pneumonia, or delirium.

The highlight of the package is its multi-task learning (MTL) fine-tuning
capability: rather than building a separate model for each complication,
`SurgicalPLAN` fine-tunes a single language model once that can predict multiple
risks simultaneously. The package also supports inference with already-trained
models and joint fine-tuning on text and a single outcome together. It is primarily designed for
clinicians, surgeons, and researchers who are not machine-learning programmers,
so that training and prediction are carried out through a few high-level
commands rather than direct use of deep-learning packages and frameworks (e.g., `transformers`). Models run locally
on the user's own notes, so that sensitive clinical text never needs to leave
the user's environment. The package supports three workflows: scoring notes
with a ready-made model that requires no training, fine-tuning a model for a
single outcome of interest, and fine-tuning one unified model across multiple
outcomes. Beyond its use as a predictive tool, `SurgicalPLAN` also serves as an
educational resource that lets clinical users learn modern language-model
techniques by training on their own data. The underlying methods were
evaluated in a peer-reviewed study [@alba2025].

# Statement of need

More than 10% of surgical patients experiences a major postoperative
complication, many of which are preventable when at-risk patients are
identified early [@xue2021]. Preoperative clinical notes serves as a
information-rich source for such early identification, but turning them into
working predictive models presents three practical obstacles that
`SurgicalPLAN` is designed to remove.

First, predicting several complications at once is operationally difficult. The
standard workflow adaptively pre-trains a model on clinical notes, tunes it to
each outcome separately, and then stores and maintains a separate model per
outcome. Clinical notes, however, are often low-resource in nature, such that a separate
adaptive pre-training stage may offer limited exposure to the textual
characteristics of a given institution's notes. 
Multi-task learning (MTL) — fine-tuning a single model that predicts many outcomes 
simultaneously — is a more efficient and often more robust alternative, 
particularly for rare outcomes, but implementing and
deploying an MTL fine-tuning pipeline from scratch requires substantial
machine-learning expertise. `SurgicalPLAN` packages this capability so that
multi-task fine-tuning becomes accessible in a few lines of code. Even for a
single outcome, it fine-tunes jointly on notes and outcome together rather than
as a separate two-stage process, which is better suited to the low-resource
nature of clinical text. 

Second, the clinicians who would benefit most — surgeons, anesthesiologists,
and perioperative trainees — may not be proficient programmers, or technically
trained at all, and so generally do not write the PyTorch code needed to
fine-tune transformer-based language models. Without an accessible interface,
the capability remains out of practical reach. `SurgicalPLAN` exposes its
workflows through high-level functions that accept a data frame of notes and
labels, minimizing the need to interact with lower-level frameworks. This also
makes the package well suited to teaching: in workshop settings, the workflows
can be demonstrated directly to a clinical audience rather than requiring
attendees to first learn deep-learning frameworks.

Third, clinical notes typically contain protected health information. Tools
that rely on transmitting text to cloud or web-hosted language-model services
are therefore often unsuitable, or outright unusable, in clinical settings.
`SurgicalPLAN` runs local, self-contained models for both fine-tuning and
inference, so that notes can be processed entirely within the user's own
environment.

The intended audience is thus twofold: clinical practitioners who wish to train
and apply models on their own institutional notes, and researchers and
educators who need an accessible, reproducible entry point to language-model
methods for perioperative risk. The methodological foundations underlying the
package were validated on roughly 85,000 preoperative notes in a peer-reviewed
study [@alba2025]; this paper concerns the software that makes those methods
reusable.

# State of the field

`SurgicalPLAN` is built on the Hugging Face `transformers` library [@wolf2020]
and its `Trainer` infrastructure. It defaults to Bio_ClinicalBERT
[@alsentzer2019], a BERT-architecture model [@devlin2019] pretrained on clinical
text, but accepts any BERT-based or encoder-only model available on HuggingFace or 
locally within the users system as its backbone. These
provide the model architecture and training scaffolding; `SurgicalPLAN`
contributes the multi-task fine-tuning model, its combined training objective,
and a clinician-facing interface built on top of them.

Two categories of existing tooling fall short for the target use case. General
clinical natural-language-processing pipelines and fine-tuning scripts assume
machine-learning fluency and are typically organized around predicting a single
outcome at a time, leaving the user to assemble and maintain a separate model
per complication. Cloud- or API-based large-language-model services, meanwhile,
are frequently unsuitable for clinical notes because the text must be
transmitted off-site.

To our knowledge, no existing package offers a clinician-accessible
implementation of multi-task fine-tuning for postoperative risk prediction that
runs locally on the user's own notes. The contribution is therefore less one of
algorithmic novelty than of consolidation and access: `SurgicalPLAN` turns a
workflow that previously demanded a machine-learning engineer into a small,
reproducible set of function calls, with the unified multi-task model as its
centerpiece.

# Software design

The central design decision is how supervised outcome prediction is combined
with language-model adaptation. Rather than the conventional two-stage recipe
— fine-tuning a language model on text, then training a separate downstream
classifier — `SurgicalPLAN` optimizes a single joint objective that adds a
supervised classification loss to the model's self-supervised masked-language
objective. The total loss is the masked-language-modeling loss plus a
weighted classification loss, $\mathcal{L} = \mathcal{L}_{\text{MLM}} +
\lambda \cdot \mathcal{L}_{\text{BCE}}$. This lets the model adapt to the
documentation style of the user's notes and learn the outcome signal at the
same time, and the weighting term `lambda_constant` controls the balance
between the two; its value was reasoned about by examining loss convergence
during development.

The multi-task model extends this design. A single shared language-model
backbone is paired with one lightweight binary classification head per outcome;
each training example is routed to the head for its outcome by a task
identifier, the per-outcome losses are averaged, and that average is combined
with the language objective. Sharing a backbone across outcomes allows
knowledge learned for common complications to support predictions for rare ones
— an important property given that some postoperative outcomes have very low
event rates — and yields a single deployable model rather than one model per
outcome. The single-outcome workflow is the same architecture instantiated with
one head, which keeps the codebase unified.

![Standard fine-tuning versus multi-task learning. In the conventional workflow
(left), each outcome requires its own fine-tuning run and produces a separate
model, so predicting several complications means training and maintaining
several models. `SurgicalPLAN`'s multi-task approach (right) fine-tunes a single
shared model once and predicts all outcomes together.\label{fig:mtl}](MTL_illustration.jpg)

The contrast between this design and the conventional approach is shown in
\autoref{fig:mtl}: where standard fine-tuning trains one model per outcome,
`SurgicalPLAN` fine-tunes once and predicts many.

A second deliberate trade-off concerns the interface. `SurgicalPLAN` hides the
underlying framework behind data-frame-level functions, accepting reduced
flexibility for advanced users in exchange for accessibility to clinical users
who are not programmers. The package exposes three entry points matched to
different data situations: direct inference with a ready-made model when no
labeled data is available, single-outcome fine-tuning when one outcome is of
interest, and multi-task fine-tuning across several outcomes. All workflows
execute locally, which is both an accessibility and a privacy design property.

# Research impact

The methods implemented in `SurgicalPLAN` were developed and evaluated in a
peer-reviewed study published in *npj Digital Medicine* [@alba2025], which
provides external, reviewed evidence that the underlying approach is sound. The
package operationalizes those methods as reusable software so that other groups
can apply them to their own notes.

The package was initially developed for and presented to surgeons at an American
College of Surgeons (ACS) workshop on artificial intelligence in surgical care.
Owing to demand following that workshop, it is being retained for use in future
conferences and workshops supporting both clinical and educational applications.
To support adoption and reproducibility without access to protected clinical
data, the package includes a synthetic-data generator that produces realistic
example notes with associated outcomes, enabling a complete end-to-end
fine-tuning and inference demonstration out of the box. Its local execution
model is intended to lower the barrier to adoption at institutions where
protected health information cannot be sent to external services.

# AI usage disclosure

Generative AI tools were used to assist in the writing of this submission,
including copy-editing this paper, and/or assisting with code and documentation. 
The authors reviewed, edited, and validated all AI-assisted output
and take full responsibility for contents of this paper. 

# Acknowledgements

[PLACEHOLDER: acknowledge any financial or grant support, and any non-author
contributors.]

# References
