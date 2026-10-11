"""Higher-order co-activity analysis for longitudinal MEA recordings."""

from .encoding import bin_spikes
from .simplicial import persistence_summary_from_supports, simplex_supports

__all__ = [
    "bin_spikes",
    "simplex_supports",
    "persistence_summary_from_supports",
]

