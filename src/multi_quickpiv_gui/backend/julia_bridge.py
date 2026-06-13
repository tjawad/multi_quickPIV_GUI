"""Bridge to the Julia multi_quickPIV backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import subprocess

import numpy as np

_JL = None
_J = None

_BACKGROUND_FILTER_THRESHOLDS = {
    "Off": ("none", -1.0),
    "Low": ("maximum", 25.0),
    "Medium": ("maximum", 50.0),
    "High": ("maximum", 100.0),
    "Very High": ("maximum", 200.0),
}


def resolve_background_filter(level: str) -> tuple[str, float]:
    """Resolve the GUI Background filter level to Julia backend arguments."""
    try:
        return _BACKGROUND_FILTER_THRESHOLDS[level]
    except KeyError as exc:
        allowed = ", ".join(_BACKGROUND_FILTER_THRESHOLDS)
        raise ValueError(
            f"Unknown Background filter level {level!r}. "
            f"Expected one of: {allowed}."
        ) from exc
    

@dataclass(slots=True)
class JuliaPIVResult:
    """Result returned from one Julia-backed PIV computation."""

    u: np.ndarray
    v: np.ndarray
    xg: np.ndarray
    yg: np.ndarray
    sn: np.ndarray | None = None

    # 3D-only fields. These stay None for normal 2D PIV.
    w: np.ndarray | None = None
    zg: np.ndarray | None = None


def _repo_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[3]


def _julia_env_dir() -> Path:
    """Return the local Julia environment directory."""
    return _repo_root() / "julia_env"


def _ensure_julia_bindir_on_path() -> None:
    """
    Ensure Julia's real bin directory is on PATH.
    """
    julia_exe_override = os.environ.get("JULIA_EXE")
    if julia_exe_override and os.path.exists(julia_exe_override):
        julia_bin = os.path.dirname(julia_exe_override)
        os.environ["PATH"] = julia_bin + os.pathsep + os.environ.get("PATH", "")
        return

    julia_cmd = shutil.which("julia")
    if not julia_cmd:
        return

    try:
        bindir = subprocess.check_output(
            [julia_cmd, "-e", "print(Sys.BINDIR)"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        if bindir and os.path.isdir(bindir):
            os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
            return
    except Exception:
        pass

    os.environ["PATH"] = (
        os.path.dirname(julia_cmd) + os.pathsep + os.environ.get("PATH", "")
    )


def ensure_julia_initialized() -> None:
    """Initialize the embedded Julia runtime and load multi_quickPIV."""
    global _JL, _J

    if _JL is not None and _J is not None:
        return

    _ensure_julia_bindir_on_path()

    import julia
    from julia import Main as J

    try:
        JL = julia.Julia(compiled_modules=True)
    except Exception:
        JL = julia.Julia(compiled_modules=False)

    J.JULIA_ENV_DIR = str(_julia_env_dir())

    J.eval(
        """
using Pkg
Pkg.activate(JULIA_ENV_DIR)
try
    Pkg.instantiate()
catch err
    @warn "Pkg.instantiate() failed (continuing)" exception=(err, catch_backtrace())
end

try
    using multi_quickPIV
catch
    @warn "multi_quickPIV missing, installing..."
    using Pkg
    Pkg.add(url="https://github.com/Marc-3d/multi_quickPIV.git")
    using multi_quickPIV
end

function run_piv(img1::Array{Float64,2}, img2::Array{Float64,2};
                 corr_alg="nsqecc", interSize=(64,64), searchMargin=(128,128),
                 step=(32,32), computeSN=true,
                 backgroundFilter="none", backgroundThreshold=-1.0)

    filtFun = maximum
    threshold_value = backgroundThreshold

    if backgroundFilter == "none"
        threshold_value = -1.0
    elseif backgroundFilter == "maximum"
        filtFun = maximum
    else
        error("Unsupported backgroundFilter: $backgroundFilter")
    end

    pivparams = multi_quickPIV.setPIVParameters(
        corr_alg=corr_alg,
        interSize=interSize,
        searchMargin=searchMargin,
        step=step,
        computeSN=computeSN,
        filtFun=filtFun,
        threshold=threshold_value,
    )

    VF, SN = multi_quickPIV.PIV(img1, img2, pivparams)
    U = VF[1, :, :]
    V = VF[2, :, :]
    vfsize = size(U)
    stepv = multi_quickPIV._step(pivparams)[1:2]
    isize = multi_quickPIV._isize(pivparams)[1:2]
    xgrid = [(x - 1) * stepv[2] + div(isize[2], 2) for y in 1:vfsize[1], x in 1:vfsize[2]]
    ygrid = [(y - 1) * stepv[1] + div(isize[1], 2) for y in 1:vfsize[1], x in 1:vfsize[2]]
    return -U, V, xgrid, ygrid, SN
end

function run_piv_3d(img1::Array{Float64,3}, img2::Array{Float64,3};
                    corr_alg="nsqecc",
                    interSize=(32,32,32),
                    searchMargin=(64,64,64),
                    step=(16,16,16),
                    computeSN=true,
                    backgroundFilter="none",
                    backgroundThreshold=-1.0)

    filtFun = maximum
    threshold_value = backgroundThreshold

    if backgroundFilter == "none"
        threshold_value = -1.0
    elseif backgroundFilter == "maximum"
        filtFun = maximum
    else
        error("Unsupported backgroundFilter: $backgroundFilter")
    end

    pivparams = multi_quickPIV.setPIVParameters(
        corr_alg=corr_alg,
        interSize=interSize,
        searchMargin=searchMargin,
        step=step,
        computeSN=computeSN,
        filtFun=filtFun,
        threshold=threshold_value,
    )

    VF, SN = multi_quickPIV.PIV(img1, img2, pivparams)

    U = VF[1, :, :, :]
    V = VF[2, :, :, :]
    W = VF[3, :, :, :]

    vfsize = size(U)
    stepv = multi_quickPIV._step(pivparams)[1:3]
    isize = multi_quickPIV._isize(pivparams)[1:3]

    zgrid = [
        (z - 1) * stepv[1] + div(isize[1], 2)
        for z in 1:vfsize[1], y in 1:vfsize[2], x in 1:vfsize[3]
    ]

    ygrid = [
        (y - 1) * stepv[2] + div(isize[2], 2)
        for z in 1:vfsize[1], y in 1:vfsize[2], x in 1:vfsize[3]
    ]

    xgrid = [
        (x - 1) * stepv[3] + div(isize[3], 2)
        for z in 1:vfsize[1], y in 1:vfsize[2], x in 1:vfsize[3]
    ]

    return U, V, W, xgrid, ygrid, zgrid, SN
end

"""
    )

    _JL = JL
    _J = J


def run_piv(
    img1: np.ndarray,
    img2: np.ndarray,
    *,
    inter_size: tuple[int, int] = (64, 64),
    search_margin: tuple[int, int] = (128, 128),
    step: tuple[int, int] = (32, 32),
    compute_sn: bool = True,
    corr_alg: str = "nsqecc",
    background_filter: str = "Off",
) -> JuliaPIVResult:
    """Run one PIV computation through the embedded Julia backend."""
    ensure_julia_initialized()

    assert _J is not None

    _J.img1 = np.asarray(img1, dtype=np.float64)
    _J.img2 = np.asarray(img2, dtype=np.float64)
    _J.corr_alg = corr_alg

    background_filter_name, background_filter_threshold = (
        resolve_background_filter(background_filter)
    )
    _J.background_filter_name = background_filter_name
    _J.background_filter_threshold = background_filter_threshold

    _J.eval(
        f"U_, V_, xg_, yg_, SN_ = run_piv("
        f"img1, img2; "
        f"corr_alg=corr_alg, "
        f"interSize=({inter_size[0]}, {inter_size[1]}), "
        f"searchMargin=({search_margin[0]}, {search_margin[1]}), "
        f"step=({step[0]}, {step[1]}), "
        f"computeSN={'true' if compute_sn else 'false'}, "
        f"backgroundFilter=background_filter_name, "
        f"backgroundThreshold=background_filter_threshold)"
    )

    u = np.array(_J.eval("U_"))
    v = np.array(_J.eval("V_"))
    xg = np.array(_J.eval("xg_"))
    yg = np.array(_J.eval("yg_"))

    sn = None
    if compute_sn:
        sn = np.array(_J.eval("SN_"))
        if sn.size == 0:
            raise RuntimeError(
                "SN_ came back empty from Julia while computeSN is enabled."
            )

    return JuliaPIVResult(u=u, v=v, xg=xg, yg=yg, sn=sn)

def run_piv_3d(
    img1: np.ndarray,
    img2: np.ndarray,
    *,
    inter_size: tuple[int, int, int] = (32, 32, 32),
    search_margin: tuple[int, int, int] = (64, 64, 64),
    step: tuple[int, int, int] = (16, 16, 16),
    compute_sn: bool = True,
    corr_alg: str = "nsqecc",
    background_filter: str = "Off",
) -> JuliaPIVResult:
    """Run one 3D PIV computation through the embedded Julia backend."""
    ensure_julia_initialized()

    if img1.ndim != 3 or img2.ndim != 3:
        raise ValueError("3D PIV requires two 3D arrays shaped like (z, y, x).")
    if img1.shape != img2.shape:
        raise ValueError("img1 and img2 must have the same shape for 3D PIV.")

    assert _J is not None

    _J.img1 = np.asarray(img1, dtype=np.float64)
    _J.img2 = np.asarray(img2, dtype=np.float64)
    _J.corr_alg = corr_alg

    background_filter_name, background_filter_threshold = (
        resolve_background_filter(background_filter)
    )
    _J.background_filter_name = background_filter_name
    _J.background_filter_threshold = background_filter_threshold

    _J.eval(
        f"U_, V_, W_, xg_, yg_, zg_, SN_ = run_piv_3d("
        f"img1, img2; "
        f"corr_alg=corr_alg, "
        f"interSize=({inter_size[0]}, {inter_size[1]}, {inter_size[2]}), "
        f"searchMargin=({search_margin[0]}, {search_margin[1]}, {search_margin[2]}), "
        f"step=({step[0]}, {step[1]}, {step[2]}), "
        f"computeSN={'true' if compute_sn else 'false'}, "
        f"backgroundFilter=background_filter_name, "
        f"backgroundThreshold=background_filter_threshold)"
    )

    u = np.array(_J.eval("U_"))
    v = np.array(_J.eval("V_"))
    w = np.array(_J.eval("W_"))
    xg = np.array(_J.eval("xg_"))
    yg = np.array(_J.eval("yg_"))
    zg = np.array(_J.eval("zg_"))

    sn = None
    if compute_sn:
        sn = np.array(_J.eval("SN_"))
        if sn.size == 0:
            raise RuntimeError(
                "SN_ came back empty from Julia while computeSN is enabled."
            )

    return JuliaPIVResult(u=u, v=v, xg=xg, yg=yg, sn=sn, w=w, zg=zg)

def _radius_tuple_literal(radius: tuple[int, ...]) -> str:
    """Return a Julia tuple literal for a validated non-negative radius tuple."""
    if not radius:
        raise ValueError("Backend average radius must not be empty.")

    values = tuple(int(value) for value in radius)
    if any(value < 0 for value in values):
        raise ValueError("Backend average radii must be at least 0.")

    return "(" + ", ".join(str(value) for value in values) + ")"


def _run_backend_average_vf(
    vf: np.ndarray,
    radius: tuple[int, ...],
) -> np.ndarray:
    """Run multi_quickPIV.average on one combined backend vector field."""
    if len(radius) not in {2, 3}:
        raise ValueError(
            "multi_quickPIV.average supports 2D or 3D backend radii in this GUI."
        )

    ensure_julia_initialized()

    assert _J is not None

    _J.backend_average_vf = np.asarray(vf, dtype=np.float64)
    radius_literal = _radius_tuple_literal(radius)

    _J.eval(
        "backend_average_out = "
        f"multi_quickPIV.average({radius_literal}, backend_average_vf)"
    )

    return np.array(_J.eval("backend_average_out"))


def _run_backend_magnitudes_vf(vf: np.ndarray) -> np.ndarray:
    """Run multi_quickPIV.magnitudes on one combined backend vector field."""
    ensure_julia_initialized()

    assert _J is not None

    _J.backend_magnitudes_vf = np.asarray(vf, dtype=np.float64)
    _J.eval(
        "backend_magnitudes_out = "
        "multi_quickPIV.magnitudes(backend_magnitudes_vf)"
    )

    return np.array(_J.eval("backend_magnitudes_out"))


def backend_vector_magnitudes(
    u: np.ndarray,
    v: np.ndarray,
    *,
    w: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute vector magnitude using multi_quickPIV.magnitudes.

    Supported shapes:
      2D single field:
        U,V = (H, W) -> magnitude = (H, W)

      2D multi-field:
        U,V = (T, H, W) -> magnitude = (T, H, W)

      3D single field:
        U,V,W = (Z, Y, X) -> magnitude = (Z, Y, X)

      3D multi-field:
        U,V,W = (T, Z, Y, X) -> magnitude = (T, Z, Y, X)
    """
    u_arr = np.asarray(u, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)

    if u_arr.shape != v_arr.shape:
        raise ValueError("u and v must have the same shape for magnitude calculation.")

    if w is None:
        if u_arr.ndim == 2:
            vf = np.stack((u_arr, v_arr), axis=0)
            return _run_backend_magnitudes_vf(vf)

        if u_arr.ndim == 3:
            magnitudes = []
            for index in range(u_arr.shape[0]):
                vf = np.stack((u_arr[index], v_arr[index]), axis=0)
                magnitudes.append(_run_backend_magnitudes_vf(vf))
            return np.stack(magnitudes)

        raise ValueError(
            "2D magnitude expects U,V shaped as (H, W) or (T, H, W)."
        )

    w_arr = np.asarray(w, dtype=np.float64)

    if w_arr.shape != u_arr.shape:
        raise ValueError(
            "w must have the same shape as u and v for magnitude calculation."
        )

    if u_arr.ndim == 3:
        vf = np.stack((u_arr, v_arr, w_arr), axis=0)
        return _run_backend_magnitudes_vf(vf)

    if u_arr.ndim == 4:
        magnitudes = []
        for index in range(u_arr.shape[0]):
            vf = np.stack((u_arr[index], v_arr[index], w_arr[index]), axis=0)
            magnitudes.append(_run_backend_magnitudes_vf(vf))
        return np.stack(magnitudes)

    raise ValueError(
        "3D magnitude expects U,V,W shaped as (Z, Y, X) or (T, Z, Y, X)."
    )


def backend_average_vector_field(
    u: np.ndarray,
    v: np.ndarray,
    *,
    w: np.ndarray | None = None,
    spatial_radius: int = 1,
    temporal_radius: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Smooth vector fields using multi_quickPIV.average.

    Supported shapes:
      2D single field:
        U,V = (H, W)

      2D multi-field:
        U,V = (T, H, W)

      3D single field:
        U,V,W = (Z, Y, X)

      3D multi-field:
        U,V,W = (T, Z, Y, X), spatial smoothing only.

    For 2D multi-field data, temporal_radius is included as the final
    backend averaging dimension. This averages over a full local space-time
    block, not quickPIV's original spatial-block-plus-temporal-line algorithm.
    """
    spatial_radius = int(spatial_radius)
    temporal_radius = int(temporal_radius)

    if spatial_radius < 0:
        raise ValueError("Spatio-temporal averaging spatial radius must be at least 0.")
    if temporal_radius < 0:
        raise ValueError("Spatio-temporal averaging temporal radius must be at least 0.")

    u_arr = np.asarray(u, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)

    if u_arr.shape != v_arr.shape:
        raise ValueError("u and v must have the same shape for backend averaging.")

    if spatial_radius == 0 and temporal_radius == 0:
        w_copy = None if w is None else np.asarray(w, dtype=np.float64).copy()
        return u_arr.copy(), v_arr.copy(), w_copy

    if w is None:
        if u_arr.ndim == 2:
            if temporal_radius > 0:
                raise ValueError(
                    "Temporal averaging requires a multi-field 2D result. "
                    "Use temporal radius 0 for a single 2D vector field."
                )

            vf = np.stack((u_arr, v_arr), axis=0)
            averaged = _run_backend_average_vf(
                vf,
                (spatial_radius, spatial_radius),
            )
            return averaged[0], averaged[1], None

        if u_arr.ndim == 3:
            # GUI/export shape: (T, H, W)
            # backend combined VF shape: (component, H, W, T)
            vf = np.stack(
                (
                    np.moveaxis(u_arr, 0, -1),
                    np.moveaxis(v_arr, 0, -1),
                ),
                axis=0,
            )
            averaged = _run_backend_average_vf(
                vf,
                (spatial_radius, spatial_radius, temporal_radius),
            )

            u_out = np.moveaxis(averaged[0], -1, 0)
            v_out = np.moveaxis(averaged[1], -1, 0)
            return u_out, v_out, None

        raise ValueError(
            "2D backend averaging expects U,V shaped as (H, W) or (T, H, W)."
        )

    w_arr = np.asarray(w, dtype=np.float64)

    if w_arr.shape != u_arr.shape:
        raise ValueError("w must have the same shape as u and v for backend averaging.")

    if temporal_radius > 0:
        raise ValueError(
            "Temporal averaging is not supported for 3D PIV with the current "
            "multi_quickPIV.average backend. Use temporal radius 0 for 3D spatial smoothing."
        )

    if u_arr.ndim == 3:
        # Single 3D vector field: (Z, Y, X)
        vf = np.stack((u_arr, v_arr, w_arr), axis=0)
        averaged = _run_backend_average_vf(
            vf,
            (spatial_radius, spatial_radius, spatial_radius),
        )
        return averaged[0], averaged[1], averaged[2]

    if u_arr.ndim == 4:
        # Multi-field 3D result: (T, Z, Y, X)
        # Apply spatial-only backend averaging independently to each field.
        u_fields: list[np.ndarray] = []
        v_fields: list[np.ndarray] = []
        w_fields: list[np.ndarray] = []

        for index in range(u_arr.shape[0]):
            vf = np.stack(
                (
                    u_arr[index],
                    v_arr[index],
                    w_arr[index],
                ),
                axis=0,
            )
            averaged = _run_backend_average_vf(
                vf,
                (spatial_radius, spatial_radius, spatial_radius),
            )

            u_fields.append(averaged[0])
            v_fields.append(averaged[1])
            w_fields.append(averaged[2])

        return np.stack(u_fields), np.stack(v_fields), np.stack(w_fields)

    raise ValueError(
        "3D backend averaging expects U,V,W shaped as (Z, Y, X) or (T, Z, Y, X)."
    )
