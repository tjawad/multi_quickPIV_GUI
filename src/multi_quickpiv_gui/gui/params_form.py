"""Parameter form state and builders for the multi_quickPIV GUI."""

from __future__ import annotations

from dataclasses import dataclass, field
import tkinter as tk
from tkinter import messagebox, ttk

from multi_quickpiv_gui.workflow.params import (
    MedianDespikeParams,
    PIVRunParams,
    PostProcessParams,
    SNFilterParams,
    SpatioTemporalAverageParams,
    WorkflowParams,
)


@dataclass(slots=True)
class ParamsFormState:
    """Tk variable bundle for the full parameter form."""

    intersize_x: tk.StringVar
    intersize_y: tk.StringVar
    intersize_z: tk.StringVar
    search_x: tk.StringVar
    search_y: tk.StringVar
    search_z: tk.StringVar
    step_x: tk.StringVar
    step_y: tk.StringVar
    step_z: tk.StringVar
    mode_note: tk.StringVar

    compute_sn: tk.BooleanVar
    corr_alg: tk.StringVar
    background_filter: tk.StringVar
    downsample_x: tk.StringVar
    downsample_y: tk.StringVar
    downsample_z: tk.StringVar

    despike: tk.BooleanVar
    despike_ksize: tk.StringVar
    despike_thr: tk.StringVar

    sn_filter: tk.BooleanVar
    sn_min: tk.StringVar
    average_enabled: tk.BooleanVar
    average_spatial_radius: tk.StringVar
    average_temporal_radius: tk.StringVar
    
    compute_sn_widget: ttk.Checkbutton | None = field(default=None, init=False)
    sn_filter_widget: ttk.Checkbutton | None = field(default=None, init=False)
    average_temporal_widget: ttk.Entry | None = field(default=None, init=False)

def _auto_fill_following_fields(
    primary: tk.StringVar,
    followers: tuple[tk.StringVar, ...],
) -> None:
    """
    Auto-fill later fields from the primary field.

    Whenever the primary field changes, all follower fields are updated to match.
    The follower fields remain editable afterward.
    """
    def _on_change(*_args) -> None:
        new_value = primary.get()

        for follower in followers:
            follower.set(new_value)

    primary.trace_add("write", _on_change)

def create_params_form_state(master: tk.Misc) -> ParamsFormState:
    """Create the Tk variables used by the parameter form."""
    form = ParamsFormState(
        intersize_x=tk.StringVar(master=master, value="64"),
        intersize_y=tk.StringVar(master=master, value="64"),
        intersize_z=tk.StringVar(master=master, value="64"),
        search_x=tk.StringVar(master=master, value="128"),
        search_y=tk.StringVar(master=master, value="128"),
        search_z=tk.StringVar(master=master, value="128"),
        step_x=tk.StringVar(master=master, value="32"),
        step_y=tk.StringVar(master=master, value="32"),
        step_z=tk.StringVar(master=master, value="32"),
        mode_note=tk.StringVar(
            master=master,
            value="2D mode: X and Y are used. Z is ignored.",
        ),
        compute_sn=tk.BooleanVar(master=master, value=True),
        corr_alg=tk.StringVar(master=master, value="nsqecc"),
        background_filter=tk.StringVar(master=master, value="Off"),
        downsample_x=tk.StringVar(master=master, value="1"),
        downsample_y=tk.StringVar(master=master, value="1"),
        downsample_z=tk.StringVar(master=master, value="1"),
        despike=tk.BooleanVar(master=master, value=False),
        despike_ksize=tk.StringVar(master=master, value="3"),
        despike_thr=tk.StringVar(master=master, value="3.5"),
        sn_filter=tk.BooleanVar(master=master, value=False),
        sn_min=tk.StringVar(master=master, value="1.0"),
                average_enabled=tk.BooleanVar(master=master, value=False),
        average_spatial_radius=tk.StringVar(master=master, value="1"),
        average_temporal_radius=tk.StringVar(master=master, value="0"),
    )

    _auto_fill_following_fields(
        form.intersize_x,
        (form.intersize_y, form.intersize_z),
    )
    _auto_fill_following_fields(
        form.search_x,
        (form.search_y, form.search_z),
    )
    _auto_fill_following_fields(
        form.step_x,
        (form.step_y, form.step_z),
    )
    _auto_fill_following_fields(
        form.downsample_x,
        (form.downsample_y, form.downsample_z),
    )

    return form

def _show_background_filter_info() -> None:
    """Show a short explanation of the Background filter control."""
    messagebox.showinfo(
        "Background filter",
        "Skips low-signal interrogation regions before PIV; High is recommended for 3D data.",
    )

CORR_ALG_HELP = {
    "nsqecc": (
        "NSQECC: normalized squared error cross-correlation; the recommended "
        "default for robust matching in biological image data."
    ),
    "zncc": (
        "ZNCC: zero-normalized cross-correlation; useful for reducing the dot-product "
        "bias toward high-intensity regions."
    ),
    "fft": (
        "FFT: frequency-domain cross-correlation; computes cross-correlation "
        "efficiently in the frequency domain - fast."
    ),
}

def _show_corr_alg_info(form: ParamsFormState) -> None:
    """Show information about the available cross-correlation algorithms."""
    selected = str(form.corr_alg.get()).strip().lower()

    lines = []
    for name in ("nsqecc", "zncc", "fft"):
        prefix = "▶ " if name == selected else "  "
        lines.append(f"{prefix}{CORR_ALG_HELP[name]}")

    messagebox.showinfo(
        "Cross-correlation algorithms",
        "\n\n".join(lines),
    )

def _show_spatiotemporal_average_info() -> None:
    """Show a short explanation of the spatio-temporal averaging control."""
    messagebox.showinfo(
        "Spatio-temporal averaging",
        "Smooths batch or loaded multi-field vector-field sequences. \n\n"
        "For 3D PIV, temporal averaging is not supported yet, so temporal radius is fixed to 0.",
    )

def build_params_panel(parent: ttk.Frame, form: ParamsFormState) -> None:
    """Build the full parameter panel into the given parent frame."""
    piv_frame = ttk.LabelFrame(parent, text="PIV Parameters", padding=8)
    piv_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))

    ttk.Label(piv_frame, text="").grid(row=0, column=0, sticky="w")
    ttk.Label(piv_frame, text="X").grid(row=0, column=1)
    ttk.Label(piv_frame, text="Y").grid(row=0, column=2)
    ttk.Label(piv_frame, text="Z").grid(row=0, column=3)

    ttk.Label(piv_frame, text="interSize").grid(row=1, column=0, sticky="w")
    ttk.Entry(
        piv_frame, width=8, textvariable=form.intersize_x
    ).grid(row=1, column=1, padx=4)
    ttk.Entry(
        piv_frame, width=8, textvariable=form.intersize_y
    ).grid(row=1, column=2, padx=4)
    ttk.Entry(
        piv_frame, width=8, textvariable=form.intersize_z
    ).grid(row=1, column=3, padx=4)

    ttk.Label(piv_frame, text="searchMargin").grid(
        row=2, column=0, sticky="w"
    )
    ttk.Entry(
        piv_frame, width=8, textvariable=form.search_x
    ).grid(row=2, column=1, padx=4)
    ttk.Entry(
        piv_frame, width=8, textvariable=form.search_y
    ).grid(row=2, column=2, padx=4)
    ttk.Entry(
        piv_frame, width=8, textvariable=form.search_z
    ).grid(row=2, column=3, padx=4)

    ttk.Label(piv_frame, text="step").grid(row=3, column=0, sticky="w")
    ttk.Entry(
        piv_frame, width=8, textvariable=form.step_x
    ).grid(row=3, column=1, padx=4)
    ttk.Entry(
        piv_frame, width=8, textvariable=form.step_y
    ).grid(row=3, column=2, padx=4)
    ttk.Entry(
        piv_frame, width=8, textvariable=form.step_z
    ).grid(row=3, column=3, padx=4)

    ttk.Label(
        piv_frame,
        textvariable=form.mode_note,
    ).grid(row=4, column=0, columnspan=4, sticky="w", pady=(6, 0))

    ttk.Label(piv_frame, text="Downsampling").grid(row=5, column=0, sticky="w")
    ttk.Entry(
        piv_frame, width=8, textvariable=form.downsample_x
    ).grid(row=5, column=1, padx=4, pady=4)
    ttk.Entry(
        piv_frame, width=8, textvariable=form.downsample_y
    ).grid(row=5, column=2, padx=4, pady=4)
    ttk.Entry(
        piv_frame, width=8, textvariable=form.downsample_z
    ).grid(row=5, column=3, padx=4, pady=4)

    ttk.Label(piv_frame, text="Background filter").grid(row=6, column=0, sticky="w")
    background_combo = ttk.Combobox(
        piv_frame,
        textvariable=form.background_filter,
        values=("Off", "Low", "Medium", "High", "Very High"),
        width=12,
        state="readonly",
    )
    background_combo.grid(row=6, column=1, columnspan=2, sticky="ew", pady=4)

    ttk.Button(
        piv_frame,
        text="?",
        width=3,
        command=_show_background_filter_info,
    ).grid(row=6, column=3, sticky="e", pady=4)

    form.compute_sn_widget = ttk.Checkbutton(
        piv_frame,
        text="computeSN",
        variable=form.compute_sn,
    )
    form.compute_sn_widget.grid(row=7, column=0, sticky="w", pady=(8, 0))
    
    ttk.Label(piv_frame, text="corr_alg").grid(row=8, column=0, sticky="w")
    corr_alg_combo = ttk.Combobox(
        piv_frame,
        textvariable=form.corr_alg,
        values=("nsqecc", "zncc", "fft"),
        width=12,
        state="normal",
    )
    corr_alg_combo.grid(row=8, column=1, columnspan=2, sticky="ew", pady=4)

    ttk.Button(
        piv_frame,
        text="ⓘ",
        width=2,
        command=lambda: _show_corr_alg_info(form),
        takefocus=False,
    ).grid(row=8, column=3, sticky="e", pady=4)

    filt_frame = ttk.LabelFrame(parent, text="Median Filter", padding=8)
    filt_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))

    ttk.Checkbutton(
        filt_frame,
        text="Enable median despike",
        variable=form.despike,
    ).grid(row=0, column=0, columnspan=3, sticky="w")

    ttk.Label(filt_frame, text="Window size").grid(row=1, column=0, sticky="w")
    ttk.Entry(
        filt_frame, width=8, textvariable=form.despike_ksize
    ).grid(row=1, column=1, padx=4, pady=4)

    ttk.Label(filt_frame, text="Threshold (MAD ×)").grid(
        row=2, column=0, sticky="w"
    )
    ttk.Entry(
        filt_frame, width=8, textvariable=form.despike_thr
    ).grid(row=2, column=1, padx=4, pady=4)

    avg_frame = ttk.LabelFrame(parent, text="Spatio-temporal averaging", padding=8)
    avg_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))

    ttk.Checkbutton(
        avg_frame,
        text="Enable averaging",
        variable=form.average_enabled,
    ).grid(row=0, column=0, columnspan=3, sticky="w")

    ttk.Label(avg_frame, text="Spatial radius").grid(row=1, column=0, sticky="w")
    ttk.Entry(
        avg_frame, width=8, textvariable=form.average_spatial_radius
    ).grid(row=1, column=1, padx=4, pady=4)

    ttk.Label(avg_frame, text="Temporal radius").grid(row=2, column=0, sticky="w")
    form.average_temporal_widget = ttk.Entry(
        avg_frame, width=8, textvariable=form.average_temporal_radius
    )
    form.average_temporal_widget.grid(row=2, column=1, padx=4, pady=4)

    ttk.Button(
        avg_frame,
        text="?",
        width=3,
        command=_show_spatiotemporal_average_info,
    ).grid(row=2, column=2, sticky="e", pady=4)

    sn_frame = ttk.LabelFrame(parent, text="SN Filter", padding=8)
    sn_frame.grid(row=3, column=0, sticky="ew")

    form.sn_filter_widget = ttk.Checkbutton(
        sn_frame,
        text="Enable SN filtering",
        variable=form.sn_filter,
    )
    form.sn_filter_widget.grid(row=0, column=0, columnspan=2, sticky="w")

    ttk.Label(sn_frame, text="SN minimum").grid(row=1, column=0, sticky="w")
    ttk.Entry(
        sn_frame, width=8, textvariable=form.sn_min
    ).grid(row=1, column=1, padx=4, pady=4)

def _read_int(var: tk.Variable, field_name: str) -> int:
    """Read an integer value from a Tk variable."""
    try:
        return int(var.get())
    except Exception as exc:
        raise ValueError(f"Invalid integer for {field_name}.") from exc


def _read_float(var: tk.Variable, field_name: str) -> float:
    """Read a float value from a Tk variable."""
    try:
        return float(var.get())
    except Exception as exc:
        raise ValueError(f"Invalid float for {field_name}.") from exc

def set_sn_controls_enabled(form: ParamsFormState, *, enabled: bool) -> None:
    """Enable or disable SN-related controls."""
    state = "normal" if enabled else "disabled"

    if form.compute_sn_widget is not None:
        form.compute_sn_widget.config(state=state)

    if form.sn_filter_widget is not None:
        form.sn_filter_widget.config(state=state)

def set_spatiotemporal_controls_for_mode(
    form: ParamsFormState,
    *,
    spatial_ndim: int,
) -> None:
    """Enable or restrict spatio-temporal averaging controls by analysis mode."""
    if spatial_ndim == 3:
        form.average_temporal_radius.set("0")
        if form.average_temporal_widget is not None:
            form.average_temporal_widget.config(state="disabled")
    elif spatial_ndim == 2:
        if form.average_temporal_widget is not None:
            form.average_temporal_widget.config(state="normal")
    else:
        raise ValueError("spatial_ndim must be 2 or 3.")
    
def set_parameter_mode_note(
    form: ParamsFormState,
    *,
    spatial_ndim: int,
) -> None:
    """Update the parameter-panel note for 2D or 3D mode."""
    if spatial_ndim == 2:
        form.mode_note.set("2D mode: X and Y are used. Z is ignored.")
    elif spatial_ndim == 3:
        form.mode_note.set("3D mode: X, Y, and Z/depth are used.")
    else:
        raise ValueError("spatial_ndim must be 2 or 3.")

def build_workflow_params(
    form: ParamsFormState,
    *,
    spatial_ndim: int = 2,
) -> WorkflowParams:
    """Build and validate WorkflowParams from the parameter form state."""
    if spatial_ndim not in {2, 3}:
        raise ValueError("spatial_ndim must be 2 or 3.")

    inter_x = _read_int(form.intersize_x, "interSize X")
    inter_y = _read_int(form.intersize_y, "interSize Y")
    inter_z = _read_int(form.intersize_z, "interSize Z")

    search_x = _read_int(form.search_x, "searchMargin X")
    search_y = _read_int(form.search_y, "searchMargin Y")
    search_z = _read_int(form.search_z, "searchMargin Z")

    step_x = _read_int(form.step_x, "step X")
    step_y = _read_int(form.step_y, "step Y")
    step_z = _read_int(form.step_z, "step Z")

    downsample_x = _read_int(form.downsample_x, "downsampling X")
    downsample_y = _read_int(form.downsample_y, "downsampling Y")
    downsample_z = _read_int(form.downsample_z, "downsampling Z")

    average_spatial_radius = max(
        0,
        _read_int(form.average_spatial_radius, "spatio-temporal spatial radius"),
    )
    average_temporal_radius = max(
        0,
        _read_int(form.average_temporal_radius, "spatio-temporal temporal radius"),
    )

    if spatial_ndim == 3:
        average_temporal_radius = 0

    if spatial_ndim == 2:
        inter_size = (inter_y, inter_x)
        search_margin = (search_y, search_x)
        step = (step_y, step_x)
        downsample_factor = (downsample_y, downsample_x)
    else:
        inter_size = (inter_z, inter_y, inter_x)
        search_margin = (search_z, search_y, search_x)
        step = (step_z, step_y, step_x)
        downsample_factor = (downsample_z, downsample_y, downsample_x)

    params = WorkflowParams(
        run=PIVRunParams(
            inter_size=inter_size,
            search_margin=search_margin,
            step=step,
            compute_sn=bool(form.compute_sn.get()),
            corr_alg=str(form.corr_alg.get()).strip() or "nsqecc",
            background_filter=str(form.background_filter.get()).strip() or "Off",
            downsample_factor=downsample_factor,
        ),
        postprocess=PostProcessParams(
            median_despike=MedianDespikeParams(
                enabled=bool(form.despike.get()),
                ksize=max(3, _read_int(form.despike_ksize, "median ksize")),
                threshold=_read_float(form.despike_thr, "median threshold"),
            ),
            sn_filter=SNFilterParams(
                enabled=bool(form.sn_filter.get()),
                minimum=(
                    _read_float(form.sn_min, "SN minimum")
                    if form.sn_filter.get()
                    else 1.0
                ),
            ),
            spatiotemporal_average=SpatioTemporalAverageParams(
                enabled=bool(form.average_enabled.get()),
                spatial_radius=average_spatial_radius,
                temporal_radius=average_temporal_radius,
            ),
        ),
    )
    params.validate()
    return params