"""Input/output helpers for image stacks and saved PIV results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence

import h5py
import numpy as np
import tifffile


@dataclass(slots=True)
class LoadedStack:
    """Loaded image stack plus minimal metadata."""

    data: np.ndarray
    source_path: Path
    dataset_name: str | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the stack shape."""
        return self.data.shape

    @property
    def num_frames(self) -> int:
        """Return the number of frames in the stack."""
        return int(self.data.shape[0])

    @property
    def spatial_ndim(self) -> int:
        """Return the number of spatial dimensions per time point."""
        return self.data.ndim - 1

    @property
    def is_2d(self) -> bool:
        """Return True if the stack contains 2D frames."""
        return self.spatial_ndim == 2

    @property
    def is_3d(self) -> bool:
        """Return True if the stack contains 3D volumes."""
        return self.spatial_ndim == 3


@dataclass(slots=True)
class LoadedPIVResult:
    """Loaded saved PIV arrays from NPZ or HDF5."""

    u: np.ndarray
    v: np.ndarray
    xg: np.ndarray
    yg: np.ndarray
    sn: np.ndarray | None = None
    source_path: Path | None = None

    # 3D-only fields. These stay None for 2D results.
    w: np.ndarray | None = None
    zg: np.ndarray | None = None

    # Optional 3D/VTK validity mask.
    valid_interrogation: np.ndarray | None = None

    @property
    def spatial_ndim(self) -> int:
        """Return 2 for U/V results and 3 for U/V/W results."""
        return 3 if self.w is not None else 2

    @property
    def is_2d(self) -> bool:
        """Return True if this is a 2D PIV result."""
        return self.spatial_ndim == 2

    @property
    def is_3d(self) -> bool:
        """Return True if this is a 3D PIV result."""
        return self.spatial_ndim == 3

    @property
    def field_count(self) -> int:
        """Return the number of vector fields stored in this result."""
        if self.is_2d:
            if self.u.ndim == 2:
                return 1
            if self.u.ndim == 3:
                return int(self.u.shape[0])

        if self.is_3d:
            if self.u.ndim == 3:
                return 1
            if self.u.ndim == 4:
                return int(self.u.shape[0])

        raise ValueError(f"Unsupported loaded PIV result shape: {self.u.shape}")


def robust_read_h5(path: str | Path) -> tuple[np.ndarray, str]:
    """
    Read an HDF5 file and choose a reasonable default dataset name.

    Mirrors the current GUI behavior:
    - prefer 'tiff_data' if present
    - otherwise use the first dataset key
    """
    path_obj = Path(path)

    with h5py.File(path_obj, "r") as hf:
        if "tiff_data" in hf:
            key = "tiff_data"
        else:
            key = next(iter(hf.keys()))
        data = np.array(hf[key])

    return data, key


def normalize_stack(data: np.ndarray) -> np.ndarray:
    """
    Normalize loaded image data into a time stack.

    Supported shapes:
    - 2D image: converted from (H, W) to (1, H, W)
    - 2D time series: (T, H, W)
    - 3D time series: (T, Z, Y, X)

    A single 3D array is treated as a 2D time series, following the legacy GUI.
    """
    arr = np.asarray(data)

    if arr.ndim == 2:
        arr = arr[None, :, :]

    if arr.ndim not in {3, 4}:
        raise ValueError(
            "Expected image stack with shape (T, H, W) for 2D PIV "
            f"or (T, Z, Y, X) for 3D PIV, got shape {arr.shape}."
        )

    if arr.shape[0] < 2:
        raise ValueError("At least 2 frames are required for PIV.")

    return arr


def load_stack(path: str | Path) -> LoadedStack:
    """
    Load a TIFF or HDF5 stack and normalize it for PIV use.
    """
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()

    if suffix in {".tif", ".tiff"}:
        data = tifffile.imread(path_obj)
        dataset_name = None
    elif suffix == ".h5":
        data, dataset_name = robust_read_h5(path_obj)
    else:
        raise ValueError("Invalid file type. Supported: .tif, .tiff, .h5")

    normalized = normalize_stack(data)
    return LoadedStack(
        data=normalized,
        source_path=path_obj,
        dataset_name=dataset_name,
    )

def load_3d_tiff_sequence(paths: Sequence[str | Path]) -> LoadedStack:
    """
    Load separate 3D TIFF volumes as a 3D PIV time series.

    Each input TIFF is expected to contain one 3D volume with shape (Z, Y, X).
    The returned stack has shape (T, Z, Y, X), where T is the number of
    selected TIFF files.
    """
    path_objs = [Path(path) for path in paths]

    if len(path_objs) < 2:
        raise ValueError(
            "3D PIV from separate TIFF volumes requires at least two files."
        )

    path_objs = sorted(path_objs, key=lambda path: path.name)

    volumes: list[np.ndarray] = []
    expected_shape: tuple[int, ...] | None = None

    for path_obj in path_objs:
        suffix = path_obj.suffix.lower()
        if suffix not in {".tif", ".tiff"}:
            raise ValueError(
                "3D TIFF sequence loading only supports .tif and .tiff files."
            )

        volume = np.asarray(tifffile.imread(path_obj))

        if volume.ndim != 3:
            raise ValueError(
                "Each selected TIFF must contain one 3D volume with shape "
                f"(Z, Y, X). File {path_obj.name} had shape {volume.shape}."
            )

        if expected_shape is None:
            expected_shape = volume.shape
        elif volume.shape != expected_shape:
            raise ValueError(
                "All selected 3D TIFF volumes must have the same shape. "
                f"Expected {expected_shape}, but {path_obj.name} had "
                f"shape {volume.shape}."
            )

        volumes.append(volume)

    stack = np.stack(volumes, axis=0)

    return LoadedStack(
        data=stack,
        source_path=path_objs[0],
        dataset_name=f"3D TIFF sequence ({len(path_objs)} files)",
    )

def load_saved_piv_result(path: str | Path) -> LoadedPIVResult:
    """
    Load a previously saved PIV result from NPZ or HDF5.
    """
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()
    valid_interrogation = None

    if suffix == ".npz":
        with np.load(path_obj) as data:
            u = np.array(data["U"])
            v = np.array(data["V"])
            w = np.array(data["W"]) if "W" in data else None
            xg = np.array(data["xgrid"])
            yg = np.array(data["ygrid"])
            zg = np.array(data["zgrid"]) if "zgrid" in data else None
            sn = np.array(data["SN"]) if "SN" in data else None
            valid_interrogation = (
                np.array(data["valid_interrogation"])
                if "valid_interrogation" in data
                else None
            )

    elif suffix == ".h5":
        with h5py.File(path_obj, "r") as hf:
            u = np.array(hf["U"])
            v = np.array(hf["V"])
            w = np.array(hf["W"]) if "W" in hf else None
            xg = np.array(hf["xgrid"])
            yg = np.array(hf["ygrid"])
            zg = np.array(hf["zgrid"]) if "zgrid" in hf else None
            sn = np.array(hf["SN"]) if "SN" in hf else None
            valid_interrogation = (
                np.array(hf["valid_interrogation"])
                if "valid_interrogation" in hf
                else None
            )
    else:
        raise ValueError("Invalid result file type. Supported: .npz, .h5")

    if u.shape != v.shape:
        raise ValueError("Saved PIV result has mismatched U and V shapes.")

    if w is not None and w.shape != u.shape:
        raise ValueError("Saved 3D PIV result has mismatched U, V, and W shapes.")

    if xg.shape != yg.shape:
        raise ValueError("Saved PIV result has mismatched xgrid and ygrid shapes.")

    if w is not None:
        if zg is None:
            raise ValueError("Saved 3D PIV result contains W but no zgrid.")
        if zg.shape != xg.shape:
            raise ValueError("Saved 3D PIV result has mismatched grid shapes.")

    if valid_interrogation is not None and valid_interrogation.shape != u.shape:
        raise ValueError(
            "Saved valid_interrogation shape does not match U/V/W shape."
        )

    return LoadedPIVResult(
        u=u,
        v=v,
        w=w,
        xg=xg,
        yg=yg,
        zg=zg,
        sn=sn,
        valid_interrogation=valid_interrogation,
        source_path=path_obj,
    )
