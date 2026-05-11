"""
Multi-task Bio+ClinicalBERT model.

This is a subclass of `transformers.BertForMaskedLM` with a set of per-outcome
binary classification heads. At training time it jointly optimizes the MLM
loss and a (lambda-scaled) mean of per-outcome BCE losses.

Unlike the original vendored-fork implementation, this file uses stock
`transformers` without any patching. The model's forward pass returns a
dataclass that extends `MaskedLMOutput` with two extra fields (`main_loss`
and `additional_loss`) so downstream code can access the loss components.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import BertForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput


@dataclass
class MTLMaskedLMOutput(MaskedLMOutput):
    """Extends HuggingFace's MaskedLMOutput with the two component losses.

    `loss` (inherited) is the total loss actually optimized.
    `main_loss` is the MLM component.
    `additional_loss` is the lambda-scaled mean of per-outcome BCE losses.
    """
    main_loss: Optional[torch.FloatTensor] = None
    additional_loss: Optional[torch.FloatTensor] = None


class CustomBioClinicalBertForCombinedLearning(BertForMaskedLM):
    """
    BERT-for-MLM with `num_tasks` additional per-outcome binary classification
    heads. Each head is a single `nn.Linear(hidden_size, 1)`. For a batch with
    `task_ids` indicating which outcome each example belongs to, the forward
    pass routes each example to its corresponding head, computes a BCE loss
    per task, averages across tasks, and adds a lambda-scaled version of that
    to the MLM loss.
    """

    def __init__(self, config, num_tasks=6, lambda_constant=10, weights=None):
        super().__init__(config)
        self.lambda_constant = lambda_constant
        self.weights = weights if weights is not None else [None] * num_tasks
        self.auxiliary = nn.ModuleList(
            [nn.Linear(config.hidden_size, 1) for _ in range(num_tasks)]
        )

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        additional_labels=None,
        task_ids=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        # --- Per-task auxiliary heads ------------------------------------
        additional_losses = []
        if task_ids is not None:
            for task_id in range(len(self.auxiliary)):
                task_specific_indices = torch.where(task_ids == task_id)[0]

                # Skip tasks with zero examples in this batch. Without this,
                # torch.mean over an empty tensor returns nan, BCEWithLogitsLoss
                # on nan returns nan, and the averaged additional_loss becomes
                # nan, which poisons total_loss. This happens in eval batches
                # or rare-outcome scenarios with small batch sizes.
                if task_specific_indices.numel() == 0:
                    continue

                head = self.auxiliary[task_id]
                task_hidden = sequence_output[task_specific_indices]
                aux_output = head(task_hidden)                    # [n, seq_len, 1]
                aux_output_pooled = torch.mean(aux_output, dim=1)  # [n, 1]

                if additional_labels is not None:
                    task_labels = additional_labels[task_specific_indices]
                    if self.weights[task_id] is not None:
                        loss_fct = torch.nn.BCEWithLogitsLoss(
                            pos_weight=self.weights[task_id]
                        )
                    else:
                        loss_fct = torch.nn.BCEWithLogitsLoss()
                    additional_loss = loss_fct(
                        aux_output_pooled.view(-1),
                        task_labels.view(-1).float(),
                    )
                    additional_losses.append(additional_loss)

            additional_loss = (
                torch.stack(additional_losses).mean() if additional_losses else 0
            )
        else:
            additional_loss = 0

        # --- Combine with MLM loss ---------------------------------------
        total_loss, masked_lm_loss = None, None
        if labels is not None:
            mlm_fct = torch.nn.CrossEntropyLoss()
            masked_lm_loss = mlm_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1),
            )
            additional_loss = self.lambda_constant * additional_loss
            total_loss = masked_lm_loss + additional_loss

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (total_loss, additional_loss, masked_lm_loss) + output

        return MTLMaskedLMOutput(
            loss=total_loss,
            main_loss=masked_lm_loss,
            additional_loss=additional_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )