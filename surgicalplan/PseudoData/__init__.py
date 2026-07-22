"""Hand-curated synthetic preoperative notes for demonstration and testing.

Two datasets, both hand-written and hand-labelled by the same process and baked
into `curated_data.py`:

- get_pseudo_training_data() -> 500 rows, for fine-tuning.
- get_pseudo_evaluation_data() -> 50 rows, disjoint from training, for scoring.

Neither is generated at call time. See `curated_data.py` for provenance and for
the caveat on inflated outcome prevalence.

`get_pseudo_data` is kept as a backwards-compatible alias for
get_pseudo_training_data.
"""

from .curated_data import (
    get_pseudo_training_data,
    get_pseudo_evaluation_data,
)

# Backwards-compatible alias: older code / notebooks called get_pseudo_data.
get_pseudo_data = get_pseudo_training_data

__all__ = [
    "get_pseudo_training_data",
    "get_pseudo_evaluation_data",
    "get_pseudo_data",
]
