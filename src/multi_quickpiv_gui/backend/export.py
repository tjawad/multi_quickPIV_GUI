"""Export helpers for NPZ/HDF5 and animated PIV outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

import h5py
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation

try:
    import imageio_ffmpeg
except Exception:
    imageio_ffmpeg = None

from multi_quickpiv_gui.workflow.pipeline import BatchPIVResult, PIVPairResult

from multi_quickpiv_gui.backend.io import LoadedPIVResult


@dataclass(slots=True)
class ExportPath:
    """Path written by one export operation."""

    path: Path


def _normalize_export_path(path: str | Path) -> Path:
    """
    Normalize the requested export path.

    Supported suffixes:
    - .npz
    - .h5
    - .mp4
    - .gif

    If no suffix is provided, default to .npz.
    """
    out_path = Path(path)

    if out_path.suffix == "":
        out_path = out_path.with_suffix(".npz")

    suffix = out_path.suffix.lower()
    if suffix not in {".npz", ".h5", ".mp4", ".gif"}:
        raise ValueError("Unsupported export format. Use .npz, .h5, .mp4, or .gif")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def save_piv_arrays(
    out_path: str | Path,
    *,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray | None = None,
    xg: np.ndarray,
    yg: np.ndarray,
    zg: np.ndarray | None = None,
    sn: np.ndarray | None = None,
    valid_interrogation: np.ndarray | None = None,
) -> ExportPath:
    
    """
    Save one PIV result payload in the format selected by the file suffix.

    Supported output formats:
    - .npz
    - .h5
    """
    out_path = _normalize_export_path(out_path)

    u_arr = np.asarray(u)
    v_arr = np.asarray(v)
    xg_arr = np.asarray(xg)
    yg_arr = np.asarray(yg)
    w_arr = None if w is None else np.asarray(w)
    zg_arr = None if zg is None else np.asarray(zg)

    if u_arr.shape != v_arr.shape:
        raise ValueError("u and v must have the same shape.")
    
    if xg_arr.shape != yg_arr.shape:
        raise ValueError("xg and yg must have the same shape.")

    if w_arr is not None or zg_arr is not None:
        if w_arr is None or zg_arr is None:
            raise ValueError("3D exports require both w and zg.")

        if w_arr.shape != u_arr.shape:
            raise ValueError("w must have the same shape as u and v.")

        if zg_arr.shape != xg_arr.shape:
            raise ValueError("zg must have the same shape as xg and yg.")

    sn_arr: np.ndarray | None = None
    if sn is not None:
        sn_arr = np.asarray(sn)
        if sn_arr.shape != u_arr.shape:
            raise ValueError("sn must have the same shape as u and v.")

    valid_interrogation_arr: np.ndarray | None = None
    if valid_interrogation is not None:
        valid_interrogation_arr = np.asarray(valid_interrogation)
        if valid_interrogation_arr.shape != u_arr.shape:
            raise ValueError(
                "valid_interrogation must have the same shape as U and V."
            )

    suffix = out_path.suffix.lower()

    if suffix == ".npz":
        save_kwargs = {
            "U": u_arr,
            "V": v_arr,
        }
        if w_arr is not None:
            save_kwargs["W"] = w_arr

        save_kwargs["xgrid"] = xg_arr
        save_kwargs["ygrid"] = yg_arr
        if zg_arr is not None:
            save_kwargs["zgrid"] = zg_arr

        if sn_arr is not None:
            save_kwargs["SN"] = sn_arr

        if valid_interrogation_arr is not None:
            save_kwargs["valid_interrogation"] = valid_interrogation_arr

        np.savez(out_path, **save_kwargs)

    elif suffix == ".h5":
        with h5py.File(out_path, "w") as hf:
            hf.create_dataset("U", data=u_arr)
            hf.create_dataset("V", data=v_arr)
            if w_arr is not None:
                hf.create_dataset("W", data=w_arr)

            hf.create_dataset("xgrid", data=xg_arr)
            hf.create_dataset("ygrid", data=yg_arr)
            if zg_arr is not None:
                hf.create_dataset("zgrid", data=zg_arr)

            if sn_arr is not None:
                hf.create_dataset("SN", data=sn_arr)

            if valid_interrogation_arr is not None:
                hf.create_dataset(
                    "valid_interrogation",
                    data=valid_interrogation_arr,
                )
    else:
        raise ValueError("save_piv_arrays only supports .npz and .h5 outputs.")

    return ExportPath(path=out_path)


def save_pair_result(
    out_path: str | Path,
    result: PIVPairResult,
) -> ExportPath:
    """Save one frame-pair result in the selected format."""
    return save_piv_arrays(
        out_path,
        u=result.u,
        v=result.v,
        w=result.w,
        xg=result.xg,
        yg=result.yg,
        zg=result.zg,
        sn=result.sn,
        valid_interrogation=result.valid_interrogation,
    )

def save_batch_result(
    out_path: str | Path,
    result: BatchPIVResult,
) -> ExportPath:
    """Save one batch result in the selected format."""
    if not result.pair_results:
        raise ValueError("Cannot export an empty batch result.")

    u = np.stack(result.u_list)
    v = np.stack(result.v_list)
    w_list = result.w_list
    w = np.stack(w_list) if w_list is not None else None

    sn_list = result.sn_list
    sn = np.stack(sn_list) if sn_list is not None else None

    xg = result.xg
    yg = result.yg
    zg = result.zg
    if xg is None or yg is None:
        raise ValueError("Batch result is missing grid coordinates.")

    valid_items = [result.valid_interrogation for result in result.pair_results]
    valid_interrogation = (
        np.stack(valid_items)
        if valid_items and all(item is not None for item in valid_items)
        else None
    )
    return save_piv_arrays(
        out_path,
        u=u,
        v=v,
        w=w,
        xg=xg,
        yg=yg,
        zg=zg,
        sn=sn,
        valid_interrogation=valid_interrogation,
    )

_NUMPY_TO_VTK_TYPE = {
    np.dtype(np.uint8): "unsigned_char",
    np.dtype(np.int8): "char",
    np.dtype(np.uint16): "unsigned_short",
    np.dtype(np.int16): "short",
    np.dtype(np.uint32): "unsigned_int",
    np.dtype(np.int32): "int",
    np.dtype(np.uint64): "unsigned_long",
    np.dtype(np.int64): "long",
    np.dtype(np.float32): "float",
    np.dtype(np.float64): "double",
}

def _vtk_data_type(array: np.ndarray) -> str:
    """Return the legacy VTK scalar type name for a NumPy array."""
    dtype = np.asarray(array).dtype
    data_type = _NUMPY_TO_VTK_TYPE.get(dtype)
    if data_type is None:
        raise ValueError(f"Unrecognized data type for VTK export: {dtype}")
    return data_type

def _parse_vtk_filename(
    filename: str | Path,
    *,
    path: str | Path = "",
    suffix: str = ".vtk",
) -> Path:
    """
    Mirror quickPIV's parseFilename behavior for VTK output.

    If no suffix is present, append .vtk.
    If path is given and filename has no directory, prepend path.
    """
    out_path = Path(filename)

    if path and not out_path.is_absolute() and out_path.parent == Path("."):
        out_path = Path(path) / out_path

    if not out_path.name.lower().endswith(suffix.lower()):
        out_path = out_path.with_name(out_path.name + suffix)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path

def _strip_literal_suffix(path: str | Path, suffix: str) -> Path:
    """Strip a specific file suffix without treating internal dots as suffixes."""
    path_obj = Path(path)
    if path_obj.name.lower().endswith(suffix.lower()):
        return path_obj.with_name(path_obj.name[: -len(suffix)])
    return path_obj

def _append_literal_suffix(path: str | Path, suffix: str) -> Path:
    """Append a suffix without removing text after internal dots."""
    path_obj = Path(path)
    if path_obj.name.lower().endswith(suffix.lower()):
        return path_obj
    return path_obj.with_name(path_obj.name + suffix)

def _vtk_sequence_path(base_path: str | Path, index: int, total: int) -> Path:
    """Build one VTK path for a single file or indexed VTK sequence."""
    base = _strip_literal_suffix(base_path, ".vtk")

    if total == 1:
        return _append_literal_suffix(base, ".vtk")

    return base.with_name(f"{base.name}_{index:03d}.vtk")

def vector_field_2d_to_vtk(
    filename: str | Path,
    u: np.ndarray,
    v: np.ndarray,
    *,
    xg: np.ndarray | None = None,
    yg: np.ndarray | None = None,
    path: str | Path = "",
    mode: str = "w",
) -> ExportPath:
    """
    Store one 2D vector field as a legacy ASCII VTK file.

    The field is written as a structured grid with one Z plane.

    VTK arrays written:
    - finite_mask
    - directions
    - direction_mag
    """
    u_arr = np.asarray(u, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)

    if u_arr.ndim != 2:
        raise ValueError("2D VTK export requires 2D U and V arrays.")
    if v_arr.shape != u_arr.shape:
        raise ValueError("U and V must have the same shape for 2D VTK export.")

    if xg is None:
        xg_arr = None
    else:
        xg_arr = np.asarray(xg, dtype=np.float64)
        if xg_arr.shape != u_arr.shape:
            raise ValueError("xgrid must have the same shape as U and V.")

    if yg is None:
        yg_arr = None
    else:
        yg_arr = np.asarray(yg, dtype=np.float64)
        if yg_arr.shape != u_arr.shape:
            raise ValueError("ygrid must have the same shape as U and V.")

    finite_arr = np.isfinite(u_arr) & np.isfinite(v_arr)

    u_write = np.where(finite_arr, u_arr, 0.0)
    v_write = np.where(finite_arr, v_arr, 0.0)
    direction_mag = np.sqrt(u_write ** 2 + v_write ** 2)

    ny, nx = u_arr.shape
    npoints = int(u_arr.size)

    out_path = _parse_vtk_filename(filename, path=path)

    with open(out_path, mode, encoding="utf-8") as io:
        io.write("# vtk DataFile Version 2.0\n")
        io.write("multi_quickPIV GUI 2D vector field\n")
        io.write("ASCII\n")
        io.write("DATASET STRUCTURED_GRID\n")
        io.write(f"DIMENSIONS {nx} {ny} 1\n")
        io.write(f"POINTS {npoints} double\n")

        for y in range(ny):
            for x in range(nx):
                if xg_arr is not None and yg_arr is not None:
                    px = xg_arr[y, x]
                    py = yg_arr[y, x]
                else:
                    px = x + 1
                    py = y + 1

                io.write(f"{float(px)} {float(py)} 0.0\n")

        io.write(f"POINT_DATA {npoints}\n")

        io.write("SCALARS finite_mask unsigned_char 1\n")
        io.write("LOOKUP_TABLE default\n")
        for y in range(ny):
            for x in range(nx):
                io.write(f"{1 if finite_arr[y, x] else 0}\n")

        io.write("VECTORS directions double\n")
        for y in range(ny):
            for x in range(nx):
                io.write(
                    f"{u_write[y, x].item()} "
                    f"{v_write[y, x].item()} "
                    "0.0\n"
                )

        io.write("SCALARS direction_mag double 1\n")
        io.write("LOOKUP_TABLE default\n")
        for y in range(ny):
            for x in range(nx):
                io.write(f"{direction_mag[y, x].item()}\n")

    return ExportPath(path=out_path)

def vector_field_to_vtk(
    filename: str | Path,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    *,
    path: str | Path = "",
    mode: str = "w",
    valid_interrogation: np.ndarray | None = None,
) -> ExportPath:
    """
    Store one 3D vector field as a legacy ASCII VTK file.

    The GUI stores 3D arrays in (Z, Y, X) order.

    VTK arrays written:
    - finite_mask
    - valid_interrogation
    - directions
    - direction_mag
    """
    u_arr = np.asarray(u, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)
    w_arr = np.asarray(w, dtype=np.float64)

    if u_arr.ndim != 3:
        raise ValueError("VTK vector-field export requires 3D U, V, W arrays.")
    if v_arr.shape != u_arr.shape or w_arr.shape != u_arr.shape:
        raise ValueError("U, V, and W must have the same shape for VTK export.")

    finite_arr = (
        np.isfinite(u_arr)
        & np.isfinite(v_arr)
        & np.isfinite(w_arr)
    )

    if valid_interrogation is None:
        valid_interrogation_arr = np.ones_like(finite_arr, dtype=np.uint8)
    else:
        valid_interrogation_arr = np.asarray(valid_interrogation, dtype=np.uint8)
        if valid_interrogation_arr.shape != u_arr.shape:
            raise ValueError(
                "valid_interrogation must have the same shape as U, V, and W."
            )

    u_write = np.where(finite_arr, u_arr, 0.0)
    v_write = np.where(finite_arr, v_arr, 0.0)
    w_write = np.where(finite_arr, w_arr, 0.0)

    direction_mag = np.sqrt(
        u_write ** 2 + v_write ** 2 + w_write ** 2
    )

    data_type = "double"

    nz, ny, nx = u_arr.shape
    npoints = int(u_arr.size)

    out_path = _parse_vtk_filename(filename, path=path)

    with open(out_path, mode, encoding="utf-8") as io:
        io.write("# vtk DataFile Version 2.0\n")
        io.write("multi_quickPIV GUI 3D vector field\n")
        io.write("ASCII\n")
        io.write("DATASET STRUCTURED_GRID\n")
        io.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        io.write(f"POINTS {npoints} int\n")

        # 1-based lattice coordinates, matching the previous quickPIV-style export.
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    io.write(f"{x + 1} {y + 1} {z + 1}\n")

        io.write(f"POINT_DATA {npoints}\n")

        io.write("SCALARS finite_mask unsigned_char 1\n")
        io.write("LOOKUP_TABLE default\n")
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    io.write(f"{1 if finite_arr[z, y, x] else 0}\n")

        io.write("SCALARS valid_interrogation unsigned_char 1\n")
        io.write("LOOKUP_TABLE default\n")
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    io.write(f"{int(valid_interrogation_arr[z, y, x])}\n")

        io.write(f"VECTORS directions {data_type}\n")
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    # Image-axis convention: X=U, Y=V, Z=W.
                    io.write(
                        f"{u_write[z, y, x].item()} "
                        f"{v_write[z, y, x].item()} "
                        f"{w_write[z, y, x].item()}\n"
                    )

        io.write("SCALARS direction_mag double 1\n")
        io.write("LOOKUP_TABLE default\n")
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    io.write(f"{direction_mag[z, y, x].item()}\n")

    return ExportPath(path=out_path)

def save_batch_vector_fields_to_vtk(
    out_path: str | Path,
    result: BatchPIVResult,
) -> list[ExportPath]:
    """
    Save every 3D frame-pair vector field in a batch result as quickPIV-style VTK.

    For one pair:
        result.npz -> result.vtk

    For multiple pairs:
        result.npz -> result_000.vtk, result_001.vtk, ...
    """
    if not result.pair_results:
        raise ValueError("Cannot export an empty batch result to VTK.")

    base_path = _strip_literal_suffix(out_path, ".vtk")

    written: list[ExportPath] = []
    total = len(result.pair_results)

    for index, pair_result in enumerate(result.pair_results):
        if pair_result.w is None:
            raise ValueError("VTK export requires 3D PIV results with W components.")

        vtk_path = _vtk_sequence_path(base_path, index, total)

        written.append(
            vector_field_to_vtk(
                vtk_path,
                pair_result.u,
                pair_result.v,
                pair_result.w,
                valid_interrogation=pair_result.valid_interrogation,
            )
        )

    return written

def save_batch_2d_vector_fields_to_vtk(
    out_path: str | Path,
    result: BatchPIVResult,
) -> list[ExportPath]:
    """
    Save every 2D frame-pair vector field in a batch result as VTK.

    For one pair:
        result.npz -> result.vtk

    For multiple pairs:
        result.npz -> result_000.vtk, result_001.vtk, ...
    """
    if not result.pair_results:
        raise ValueError("Cannot export an empty batch result to VTK.")

    base_path = _strip_literal_suffix(out_path, ".vtk")

    written: list[ExportPath] = []
    total = len(result.pair_results)

    for index, pair_result in enumerate(result.pair_results):
        if pair_result.w is not None:
            raise ValueError("2D VTK export received a 3D PIV result.")

        vtk_path = _vtk_sequence_path(base_path, index, total)

        written.append(
            vector_field_2d_to_vtk(
                vtk_path,
                pair_result.u,
                pair_result.v,
                xg=pair_result.xg,
                yg=pair_result.yg,
            )
        )

    return written

def save_loaded_piv_result(
    out_path: str | Path,
    result: LoadedPIVResult,
) -> ExportPath | list[ExportPath]:
    """Save a loaded PIV result after optional post-processing."""
    out_path = Path(out_path)
    suffix = out_path.suffix.lower()

    if suffix == ".vtk":
        base_path = _strip_literal_suffix(out_path, ".vtk")

        if result.is_2d:
            if result.u.ndim == 2:
                return vector_field_2d_to_vtk(
                    out_path,
                    result.u,
                    result.v,
                    xg=result.xg,
                    yg=result.yg,
                )

            written: list[ExportPath] = []
            total = result.field_count

            for index in range(total):
                vtk_path = _vtk_sequence_path(base_path, index, total)

                written.append(
                    vector_field_2d_to_vtk(
                        vtk_path,
                        result.u[index],
                        result.v[index],
                        xg=result.xg,
                        yg=result.yg,
                    )
                )
            return written

        if result.is_3d:
            if result.w is None:
                raise ValueError("3D VTK export requires W components.")

            if result.u.ndim == 3:
                return vector_field_to_vtk(
                    out_path,
                    result.u,
                    result.v,
                    result.w,
                    valid_interrogation=result.valid_interrogation,
                )

            written = []
            total = result.field_count

            for index in range(total):
                vtk_path = _vtk_sequence_path(base_path, index, total)

                valid = None
                if result.valid_interrogation is not None:
                    valid = result.valid_interrogation[index]

                written.append(
                    vector_field_to_vtk(
                        vtk_path,
                        result.u[index],
                        result.v[index],
                        result.w[index],
                        valid_interrogation=valid,
                    )
                )
            return written

    return save_piv_arrays(
        out_path,
        u=result.u,
        v=result.v,
        w=result.w,
        xg=result.xg,
        yg=result.yg,
        zg=result.zg,
        sn=result.sn,
        valid_interrogation=result.valid_interrogation,
    )

def save_piv_animation(
    out_path: str | Path,
    *,
    u: np.ndarray,
    v: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
) -> ExportPath:
    """
    Save a quiver-only PIV animation as MP4 or GIF.

    Expected shapes:
    - u, v: (T, H, W)
    - xg, yg: (H, W)
    """
    out_path = _normalize_export_path(out_path)

    u_arr = np.asarray(u)
    v_arr = np.asarray(v)
    xg_arr = np.asarray(xg)
    yg_arr = np.asarray(yg)

    if u_arr.ndim != 3 or v_arr.ndim != 3:
        raise ValueError("Animation export requires u and v with shape (T, H, W).")
    if u_arr.shape != v_arr.shape:
        raise ValueError("u and v must have the same shape.")
    if xg_arr.shape != yg_arr.shape:
        raise ValueError("xg and yg must have the same shape.")
    if xg_arr.shape != u_arr.shape[1:]:
        raise ValueError("xg/yg shape must match the spatial shape of u/v.")

    fig, ax = plt.subplots(figsize=(6, 6))

    try:
        ax.set_aspect("equal")
        ax.set_xlim(float(np.min(xg_arr)), float(np.max(xg_arr)))
        ax.set_ylim(float(np.min(yg_arr)), float(np.max(yg_arr)))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("PIV: Field 0")

        quiv = ax.quiver(
            xg_arr,
            yg_arr,
            v_arr[0],
            u_arr[0],
            scale=87,
            width=0.004,
        )

        def update(frame_index: int):
            quiv.set_UVC(v_arr[frame_index], u_arr[frame_index])
            ax.set_title(f"PIV: Field {frame_index}")
            return (quiv,)

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=u_arr.shape[0],
            interval=100,
            blit=True,
        )

        suffix = out_path.suffix.lower()

        if suffix == ".mp4":
            ffmpeg_exe = None

            if imageio_ffmpeg is not None:
                try:
                    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                except Exception:
                    ffmpeg_exe = None

            if ffmpeg_exe is None:
                ffmpeg_exe = shutil.which("ffmpeg")

            if ffmpeg_exe is None:
                gif_path = out_path.with_suffix(".gif")
                writer = animation.PillowWriter(fps=10)
                ani.save(str(gif_path), writer=writer)
                return ExportPath(path=gif_path)

            matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg_exe
            writer = animation.FFMpegWriter(
                fps=10,
                codec="libx264",
                extra_args=["-pix_fmt", "yuv420p"],
            )
            ani.save(str(out_path), writer=writer, dpi=150)

        elif suffix == ".gif":
            writer = animation.PillowWriter(fps=10)
            ani.save(str(out_path), writer=writer)

        else:
            raise ValueError("Animation export only supports .mp4 and .gif outputs.")

    finally:
        plt.close(fig)

    return ExportPath(path=out_path)
