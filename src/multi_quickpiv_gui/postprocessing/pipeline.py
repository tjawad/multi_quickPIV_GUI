"""Post-processing pipeline orchestration for computed PIV vector fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from multi_quickpiv_gui.backend.julia_bridge import backend_average_vector_field
from multi_quickpiv_gui.postprocessing.spatial import (
    median_despike_vector_field,
    sn_threshold_filter,
)
from multi_quickpiv_gui.workflow.params import WorkflowParams


@dataclass(slots=True)
class PostProcessResult:
    """Processed vector field plus metadata about the applied filters."""

    u: np.ndarray
    v: np.ndarray
    w: np.ndarray | None = None
    sn: np.ndarray | None = None
    sn_replaced: int = 0


def apply_spatiotemporal_average(
    u: np.ndarray,
    v: np.ndarray,
    *,
    params: WorkflowParams,
    w: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Apply backend spatio-temporal averaging using multi_quickPIV.average.

    This is intentionally batch/multi-field only. The goal is to smooth
    vector-field sequences for animation/export, not to spatially smooth
    a single vector field.
    """
    average = params.postprocess.spatiotemporal_average
    average.validate()

    u_out = np.asarray(u, dtype=np.float64).copy()
    v_out = np.asarray(v, dtype=np.float64).copy()
    w_out = None if w is None else np.asarray(w, dtype=np.float64).copy()

    if not average.enabled:
        return u_out, v_out, w_out

    is_2d_multifield = w_out is None and u_out.ndim == 3
    is_3d_multifield = w_out is not None and u_out.ndim == 4

    if not (is_2d_multifield or is_3d_multifield):
        raise ValueError(
            "Spatio-temporal averaging is only supported for batch or loaded "
            "multi-field PIV results."
        )

    if is_3d_multifield and average.temporal_radius > 0:
        raise ValueError(
            "Temporal averaging is not supported yet for 3D PIV. "
            "Use temporal radius 0 for 3D batch spatial smoothing."
        )

    return backend_average_vector_field(
        u_out,
        v_out,
        w=w_out,
        spatial_radius=average.spatial_radius,
        temporal_radius=average.temporal_radius,
    )


def apply_postprocessing(
    u: np.ndarray,
    v: np.ndarray,
    *,
    params: WorkflowParams,
    sn: np.ndarray | None = None,
    w: np.ndarray | None = None,
) -> PostProcessResult:
    """
    Apply all configured post-processing steps to a computed vector field.
    """
    params.validate()

    if u.shape != v.shape:
        raise ValueError("u and v must have the same shape.")

    if w is not None and w.shape != u.shape:
        raise ValueError("w must have the same shape as u and v.")

    u_out = np.asarray(u, dtype=np.float64).copy()
    v_out = np.asarray(v, dtype=np.float64).copy()
    w_out = None if w is None else np.asarray(w, dtype=np.float64).copy()
    sn_out = None if sn is None else np.asarray(sn, dtype=np.float64).copy()

    post = params.postprocess

    if post.median_despike.enabled:
        u_out, v_out, w_out = median_despike_vector_field(
            u_out,
            v_out,
            w=w_out,
            ksize=post.median_despike.ksize,
            threshold=post.median_despike.threshold,
            use_magnitude=True,
        )

    sn_replaced = 0
    if post.sn_filter.enabled:
        if w_out is not None:
            raise ValueError("SN filtering is not implemented for 3D vector fields yet.")

        if sn_out is None:
            raise ValueError(
                "SN filtering is enabled, but no SN array was provided."
            )

        u_out, v_out, sn_replaced = sn_threshold_filter(
            u_out,
            v_out,
            sn_out,
            sn_min=post.sn_filter.minimum,
            ksize=post.median_despike.ksize,
        )

    return PostProcessResult(
        u=u_out,
        v=v_out,
        w=w_out,
        sn=sn_out,
        sn_replaced=sn_replaced,
    )
