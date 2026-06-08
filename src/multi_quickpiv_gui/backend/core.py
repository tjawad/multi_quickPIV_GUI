"""Compatibility exports for GUI-independent post-processing helpers.

New post-processing code lives under multi_quickpiv_gui.postprocessing.
This module is kept so existing imports continue to work.
"""

from __future__ import annotations

from multi_quickpiv_gui.postprocessing.pipeline import (
    PostProcessResult,
    apply_postprocessing,
    apply_spatiotemporal_average,
)

from multi_quickpiv_gui.postprocessing.spatial import (
    median_despike,
    median_despike_vector_field,
    sn_threshold_filter,
)

from multi_quickpiv_gui.backend.julia_bridge import backend_vector_magnitudes

__all__ = [
    "PostProcessResult",
    "apply_postprocessing",
    "apply_spatiotemporal_average",
    "median_despike",
    "median_despike_vector_field",
    "sn_threshold_filter",
    "backend_vector_magnitudes",
]
