"""Hand-curated synthetic preoperative notes for demonstration and testing.

The dataset is not generated at call time: it is a fixed table of 500
hand-written notes with hand-assigned outcome labels, baked into
`curated_data.py`. See that module's docstring for provenance and for the
caveat on inflated outcome prevalence.
"""

from .curated_data import get_pseudo_data

__all__ = ["get_pseudo_data"]