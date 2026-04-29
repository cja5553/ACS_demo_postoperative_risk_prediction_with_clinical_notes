"""
Custom HuggingFace Trainer for joint MLM + auxiliary fine-tuning.

Note: this file is intentionally a duplicate of
`clinicalplan/MultiTaskFinetuning/trainer.py`. The trainer's only role is to
log the MLM and auxiliary loss components separately, which is
workflow-agnostic. If you change one, keep them in sync.

The total loss returned matches stock HF's contract (a single scalar), so
every part of `transformers.Trainer` works unchanged.
"""

from transformers import Trainer


class CustomTrainer(Trainer):
    """
    Subclass of HF Trainer that logs `main_loss` and `additional_loss`
    as extra metrics alongside the total loss.

    The model is expected to return either:
    - a dict-like object (e.g. our MTLMaskedLMOutput) with `loss`,
      `main_loss`, and `additional_loss` attributes, or
    - a tuple `(loss, additional_loss, main_loss, ...)` (when
      `return_dict=False`).
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute the joint loss.

        Note: `**kwargs` absorbs any extra keyword arguments that HF's
        Trainer may pass in newer versions (e.g. `num_items_in_batch`
        introduced in transformers 4.41). We ignore them because our
        loss is already pre-aggregated in the model's forward pass.
        """
        outputs = model(**inputs)

        # Dict-like output (return_dict=True, the default in HF)
        if isinstance(outputs, dict) or hasattr(outputs, "loss"):
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
            main_loss = (
                outputs.get("main_loss")
                if isinstance(outputs, dict)
                else getattr(outputs, "main_loss", None)
            )
            additional_loss = (
                outputs.get("additional_loss")
                if isinstance(outputs, dict)
                else getattr(outputs, "additional_loss", None)
            )
        else:
            # Tuple output (return_dict=False); order matches our model.
            loss = outputs[0]
            additional_loss = outputs[1] if len(outputs) > 1 else None
            main_loss = outputs[2] if len(outputs) > 2 else None

        # Log the components so they show up in the Trainer's logs /
        # TensorBoard alongside the total loss. Only log when in training
        # mode to avoid noisy per-step eval output.
        if self.model.training:
            metrics_to_log = {}
            if main_loss is not None and hasattr(main_loss, "item"):
                metrics_to_log["main_loss"] = main_loss.item()
            if additional_loss is not None and hasattr(additional_loss, "item"):
                metrics_to_log["additional_loss"] = additional_loss.item()
            if metrics_to_log:
                # self.log routes to every enabled integration (stdout,
                # TensorBoard, W&B, etc.). Swallow errors in case the
                # Trainer isn't fully initialized yet.
                try:
                    self.log(metrics_to_log)
                except Exception:
                    pass

        return (loss, outputs) if return_outputs else loss
