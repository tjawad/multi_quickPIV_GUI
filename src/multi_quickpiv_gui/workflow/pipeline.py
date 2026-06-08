"""Single-run and batch workflow orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from multi_quickpiv_gui.backend.core import apply_postprocessing
from multi_quickpiv_gui.backend.julia_bridge import (
    resolve_background_filter,
    run_piv as run_piv_2d_julia,
    run_piv_3d as run_piv_3d_julia,
)
from multi_quickpiv_gui.workflow.params import WorkflowParams


def _downsample_pair(
    img1: np.ndarray,
    img2: np.ndarray,
    factor: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply per-axis pre-PIV downsampling to one frame pair."""
    factors = tuple(int(value) for value in factor)

    if img1.ndim == 2:
        if len(factors) != 2:
            raise ValueError("2D downsampling requires Y and X factors.")

        factor_y, factor_x = factors

        if factor_y < 1 or factor_x < 1:
            raise ValueError("Downsampling factors must be at least 1.")

        if factor_y == 1 and factor_x == 1:
            return img1, img2

        return img1[::factor_y, ::factor_x], img2[::factor_y, ::factor_x]

    if img1.ndim == 3:
        if len(factors) != 3:
            raise ValueError("3D downsampling requires Z, Y, and X factors.")

        factor_z, factor_y, factor_x = factors

        if factor_z < 1 or factor_y < 1 or factor_x < 1:
            raise ValueError("Downsampling factors must be at least 1.")

        if factor_z == 1 and factor_y == 1 and factor_x == 1:
            return img1, img2

        return (
            img1[::factor_z, ::factor_y, ::factor_x],
            img2[::factor_z, ::factor_y, ::factor_x],
        )

    raise ValueError(
        f"Downsampling supports only 2D or 3D frame pairs, got shape {img1.shape}."
    )

def _compute_valid_interrogation_mask_3d(
    volume: np.ndarray,
    *,
    vf_shape: tuple[int, int, int],
    inter_size: tuple[int, int, int],
    step: tuple[int, int, int],
    background_filter: str,
) -> np.ndarray:
    """
    Compute a per-vector validity mask from 3D interrogation volumes.

    For the current Background filter implementation:
      valid if max(interrogation_volume) >= threshold
    """
    valid = np.ones(vf_shape, dtype=np.uint8)

    filter_name, threshold = resolve_background_filter(background_filter)
    if filter_name == "none" or threshold < 0:
        return valid

    if filter_name != "maximum":
        raise ValueError(f"Unsupported Background filter backend: {filter_name!r}")

    if len(inter_size) != 3 or len(step) != 3:
        raise ValueError("3D valid-interrogation export requires 3D inter_size and step.")

    if any(value <= 0 for value in inter_size):
        raise ValueError("inter_size values must be greater than 0.")
    if any(value <= 0 for value in step):
        raise ValueError("step values must be greater than 0.")

    iz, iy, ix = inter_size
    sz, sy, sx = step

    for z in range(vf_shape[0]):
        z0 = z * sz
        z1 = z0 + iz

        for y in range(vf_shape[1]):
            y0 = y * sy
            y1 = y0 + iy

            for x in range(vf_shape[2]):
                x0 = x * sx
                x1 = x0 + ix

                window = volume[z0:z1, y0:y1, x0:x1]

                if window.size == 0:
                    valid[z, y, x] = 0
                    continue

                finite_window = np.asarray(window)[np.isfinite(window)]
                if finite_window.size == 0 or float(np.max(finite_window)) < threshold:
                    valid[z, y, x] = 0

    return valid

@dataclass(slots=True)
class PIVPairResult:
    """Complete result of one frame-pair PIV workflow."""

    img1: np.ndarray
    img2: np.ndarray
    u: np.ndarray
    v: np.ndarray
    xg: np.ndarray
    yg: np.ndarray
    sn: np.ndarray | None = None
    sn_replaced: int = 0

    # 3D-only fields. These stay None for 2D PIV.
    w: np.ndarray | None = None
    zg: np.ndarray | None = None
    valid_interrogation: np.ndarray | None = None


@dataclass(slots=True)
class BatchPIVResult:
    """Complete result of a batch PIV workflow over a frame stack."""

    pair_results: list[PIVPairResult]

    @property
    def u_list(self) -> list[np.ndarray]:
        """Return the list of U fields for export or preview."""
        return [result.u for result in self.pair_results]

    @property
    def v_list(self) -> list[np.ndarray]:
        """Return the list of V fields for export or preview."""
        return [result.v for result in self.pair_results]

    @property
    def w_list(self) -> list[np.ndarray] | None:
        """Return the list of W fields if available for all results."""
        if not self.pair_results:
            return []
        if any(result.w is None for result in self.pair_results):
            return None
        return [result.w for result in self.pair_results if result.w is not None]

    @property
    def sn_list(self) -> list[np.ndarray] | None:
        """Return the list of SN fields if available for all results."""
        if not self.pair_results:
            return []
        if any(result.sn is None for result in self.pair_results):
            return None
        return [result.sn for result in self.pair_results if result.sn is not None]

    @property
    def xg(self) -> np.ndarray | None:
        """Return the grid from the first result, if present."""
        return self.pair_results[0].xg if self.pair_results else None

    @property
    def yg(self) -> np.ndarray | None:
        """Return the grid from the first result, if present."""
        return self.pair_results[0].yg if self.pair_results else None

    @property
    def zg(self) -> np.ndarray | None:
        """Return the 3D z-grid from the first result, if present."""
        return self.pair_results[0].zg if self.pair_results else None


def run_piv_pair(
    img1: np.ndarray,
    img2: np.ndarray,
    *,
    params: WorkflowParams,
) -> PIVPairResult:
    """
    Run the complete PIV workflow for one frame pair.
    """
    params.validate()

    img1_arr = np.asarray(img1)
    img2_arr = np.asarray(img2)

    img1_arr, img2_arr = _downsample_pair(
        img1_arr,
        img2_arr,
        params.run.downsample_factor,
    )

    if img1_arr.shape != img2_arr.shape:
        raise ValueError("img1 and img2 must have the same shape.")
    
    valid_interrogation: np.ndarray | None = None

    if img1_arr.ndim == 2:
        raw = run_piv_2d_julia(
            img1_arr,
            img2_arr,
            inter_size=params.run.inter_size,
            search_margin=params.run.search_margin,
            step=params.run.step,
            compute_sn=params.run.compute_sn,
            corr_alg=params.run.corr_alg,
            background_filter=params.run.background_filter,
        )
    elif img1_arr.ndim == 3:
        post = params.postprocess
        if post.sn_filter.enabled:
            raise ValueError(
                "3D SN filtering is not implemented yet because 3D computeSN "
                "is currently blocked by a multi_quickPIV backend error."
            )

        raw = run_piv_3d_julia(
            img1_arr,
            img2_arr,
            inter_size=params.run.inter_size,
            search_margin=params.run.search_margin,
            step=params.run.step,
            compute_sn=params.run.compute_sn,
            corr_alg=params.run.corr_alg,
            background_filter=params.run.background_filter,
        )
        valid_interrogation = _compute_valid_interrogation_mask_3d(
            img1_arr,
            vf_shape=raw.u.shape,
            inter_size=params.run.inter_size,  # Z,Y,X
            step=params.run.step,              # Z,Y,X
            background_filter=params.run.background_filter,
        )
    else:
        raise ValueError(
            f"PIV frame pairs must be 2D or 3D, got shape {img1_arr.shape}."
        )

    processed = apply_postprocessing(
        raw.u,
        raw.v,
        params=params,
        sn=raw.sn,
        w=raw.w,
    )

    return PIVPairResult(
        img1=img1_arr,
        img2=img2_arr,
        u=processed.u,
        v=processed.v,
        xg=raw.xg,
        yg=raw.yg,
        sn=processed.sn,
        sn_replaced=processed.sn_replaced,
        w=processed.w,
        zg=raw.zg,
        valid_interrogation=valid_interrogation,
    )

def run_batch_piv(
    stack: np.ndarray,
    *,
    params: WorkflowParams,
    progress_callback: Callable[[int, int, PIVPairResult], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> BatchPIVResult:
    """
    Run the PIV workflow over all consecutive frame pairs in a stack.

    The expected stack shape is either (T, H, W) for 2D PIV
    or (T, Z, Y, X) for 3D PIV.
    """
    params.validate()

    stack_arr = np.asarray(stack)
    if stack_arr.ndim not in {3, 4}:
        raise ValueError("Batch PIV expects a stack with shape (T, H, W) for 2D "
            "or (T, Z, Y, X) for 3D.")
    if stack_arr.shape[0] < 2:
        raise ValueError("At least 2 frames are required for batch PIV.")

    pair_results: list[PIVPairResult] = []
    total_pairs = stack_arr.shape[0] - 1

    for pair_index in range(total_pairs):
        if should_stop is not None and should_stop():
            break

        result = run_piv_pair(
            stack_arr[pair_index],
            stack_arr[pair_index + 1],
            params=params,
        )
        pair_results.append(result)

        if progress_callback is not None:
            progress_callback(pair_index + 1, total_pairs, result)

    return BatchPIVResult(pair_results=pair_results)