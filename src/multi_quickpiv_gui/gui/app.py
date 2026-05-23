"""Minimal Tkinter app shell for the extracted multi_quickPIV GUI pipeline."""

from pathlib import Path
import time
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import tkinter.ttk as ttk
import numpy as np

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from multi_quickpiv_gui.backend.export import (
    save_batch_2d_vector_fields_to_vtk,
    save_batch_result,
    save_batch_vector_fields_to_vtk,
    save_loaded_piv_result,
    save_pair_result,
    save_piv_animation,
)

from multi_quickpiv_gui.backend.io import (
    LoadedPIVResult,
    LoadedStack,
    load_3d_tiff_sequence,
    load_saved_piv_result,
    load_stack,
)

from multi_quickpiv_gui.gui.preview import (
    PreviewState,
    draw_loaded_frame,
    draw_vector_field_only,
    ensure_preview_artists,
    reset_preview_state,
)
from multi_quickpiv_gui.gui.dialogs import (
    BatchExportDialog,
    BatchRunDialog,
    BatchRunOptions,
    ThreeDFileOrderDialog,
    ThreeDLoadInfoDialog,
)
from multi_quickpiv_gui.runtime.batch import BatchRuntimeState
from multi_quickpiv_gui.backend.core import apply_postprocessing

from multi_quickpiv_gui.gui.params_form import (
    ParamsFormState,
    build_params_panel,
    build_workflow_params,
    create_params_form_state,
    set_parameter_mode_note,
    set_sn_controls_enabled,
)

from multi_quickpiv_gui.workflow.pipeline import (
    BatchPIVResult,
    PIVPairResult,
    run_piv_pair,
)

class MultiQuickPIVApp:
    """Minimal application shell around the extracted pipeline."""
  
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("multi_quickPIV GUI")
        self.root.geometry("1500x850")

        try:
            self.root.state("zoomed")  # Windows: start maximized
        except tk.TclError:
            self.root.attributes("-zoomed", True)  # Linux fallback

        self.loaded_stack: LoadedStack | None = None
        self.loaded_piv_result: LoadedPIVResult | None = None
        self.current_result: PIVPairResult | BatchPIVResult | None = None
        self.current_export_name_hint = "piv_result"
        self.current_single_pair_indices: tuple[int, int] | None = None
        self.analysis_mode: str = "2d"
        self._show_3d_load_info: bool = True
        self.preview_state = PreviewState()
        self._status_after_id: str | None = None
        self.params_form: ParamsFormState = create_params_form_state(self.root)

        self.batch = BatchRuntimeState()
        self._batch_started_at: float | None = None
        self._last_pair_elapsed_seconds: float | None = None
        self._last_3d_batch_elapsed_seconds: float | None = None
        self._last_3d_export_summary: str | None = None

        self._build_variables()
        self._build_layout()
        self._set_idle_state()
        self.root.after(300, self.show_piv_info)

    def _build_variables(self) -> None:
        self.var_file_name = tk.StringVar(value="File: (none)")
        self.var_result = tk.StringVar(value="Pick a file to start")
        self.var_status = tk.StringVar(value="")
        self.var_frame = tk.IntVar(value=0)

    def _build_layout(self) -> None:
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(1, weight=1)

        top = ttk.Frame(self.root, padding=8)
        top.grid(row=0, column=0, columnspan=3, sticky="ew")
        top.columnconfigure(1, weight=1)

        ttk.Label(top, textvariable=self.var_file_name).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(top, textvariable=self.var_result).grid(
            row=0, column=1, sticky="ew", padx=12
        )

        left = ttk.Frame(self.root, padding=8)
        left.grid(row=1, column=0, sticky="nsw")

        center = ttk.Frame(self.root, padding=8)
        center.grid(row=1, column=1, sticky="nsew")
        center.rowconfigure(0, weight=1)
        center.columnconfigure(0, weight=1)

        right = ttk.Frame(self.root, padding=8)
        right.grid(row=1, column=2, sticky="nse")

        bottom = ttk.Frame(self.root, padding=8)
        bottom.grid(row=2, column=0, columnspan=3, sticky="ew")
        bottom.columnconfigure(0, weight=1)

        self._build_parameter_panel(left)
        self._build_preview_panel(center)
        self._build_action_panel(right)

        ttk.Label(bottom, textvariable=self.var_status, anchor="w").grid(
            row=0, column=0, sticky="ew"
        )

    def _build_parameter_panel(self, parent: ttk.Frame) -> None:
        """Build the parameter panel from the shared form state."""
        build_params_panel(parent, self.params_form)

    def _build_preview_panel(self, parent: ttk.Frame) -> None:
        self.figure = Figure(figsize=(7, 7), dpi=100)
        self.preview_ax = self.figure.add_subplot(111)
        self.preview_ax.set_title("No file loaded")
        self.preview_ax.axis("off")

        self.preview_canvas = FigureCanvasTkAgg(self.figure, master=parent)
        widget = self.preview_canvas.get_tk_widget()
        widget.grid(row=0, column=0, sticky="nsew")

        slider_frame = ttk.Frame(parent)
        slider_frame.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        slider_frame.columnconfigure(1, weight=1)

        ttk.Label(slider_frame, text="Frame").grid(row=0, column=0, sticky="w")
        self.slider_frame = tk.Scale(
            slider_frame,
            from_=0,
            to=0,
            orient="horizontal",
            variable=self.var_frame,
            command=self._on_frame_slider,
        )
        self.slider_frame.grid(row=0, column=1, sticky="ew")

    def _build_action_panel(self, parent: ttk.Frame) -> None:
        self.btn_load = ttk.Button(
            parent,
            text="Load file for 2D PIV",
            command=self.on_load_file,
        )
        self.btn_load.grid(row=0, column=0, sticky="ew", pady=4)

        self.btn_load_3d = ttk.Button(
            parent,
            text="Load file(s) for 3D PIV",
            command=self.on_load_3d_file,
        )
        self.btn_load_3d.grid(row=1, column=0, sticky="ew", pady=4)

        self.btn_load_piv_result = ttk.Button(
            parent,
            text="Load PIV result",
            command=self.on_load_piv_result,
        )
        self.btn_load_piv_result.grid(row=2, column=0, sticky="ew", pady=4)

        self.btn_single = ttk.Button(
            parent, text="Run single PIV", command=self.on_run_single
        )
        self.btn_single.grid(row=3, column=0, sticky="ew", pady=4)

        self.btn_batch = ttk.Button(
            parent, text="Run batch PIV", command=self.on_run_batch
        )
        self.btn_batch.grid(row=4, column=0, sticky="ew", pady=4)

        self.btn_pause = ttk.Button(
            parent, text="Pause", command=self.on_pause_batch
        )
        self.btn_pause.grid(row=5, column=0, sticky="ew", pady=4)
        self.btn_pause.config(state="disabled")

        self.btn_continue = ttk.Button(
            parent, text="Continue", command=self.on_continue_batch
        )
        self.btn_continue.grid(row=6, column=0, sticky="ew", pady=4)
        self.btn_continue.config(state="disabled")

        self.btn_abort = ttk.Button(
            parent, text="Abort", command=self.on_abort_batch
        )
        self.btn_abort.grid(row=7, column=0, sticky="ew", pady=4)
        self.btn_abort.config(state="disabled")

        self.btn_apply_postprocess = ttk.Button(
            parent,
            text="Apply post-processing",
            command=self.on_apply_postprocessing,
        )
        self.btn_apply_postprocess.grid(row=8, column=0, sticky="ew", pady=4)

        self.btn_export = ttk.Button(
            parent, text="Export current result", command=self.on_export_current
        )
        self.btn_export.grid(row=9, column=0, sticky="ew", pady=4)

        self.btn_export_video = ttk.Button(
            parent, text="Export video / GIF", command=self.on_export_animation
        )
        self.btn_export_video.grid(row=10, column=0, sticky="ew", pady=4)

        ttk.Separator(parent, orient="horizontal").grid(
            row=11, column=0, sticky="ew", pady=(12, 8)
        )

        ttk.Label(parent, text="Batch progress").grid(
            row=12, column=0, sticky="w", pady=(0, 4)
        )
        self.progress = ttk.Progressbar(parent, mode="determinate", length=220)
        self.progress.grid(row=13, column=0, sticky="ew")
        
    def _set_batch_running_state(self) -> None:
        """Disable normal actions and enable batch controls during a running batch."""
        self.btn_load.config(state="disabled")
        self.btn_load_3d.config(state="disabled")
        self.btn_load_piv_result.config(state="disabled")
        self.btn_single.config(state="disabled")
        self.btn_batch.config(state="disabled")
        self.btn_export.config(state="disabled")
        self.btn_export_video.config(state="disabled")
        self.btn_pause.config(state="normal")
        self.btn_continue.config(state="disabled")
        self.btn_abort.config(state="normal")
        self.btn_apply_postprocess.config(state="disabled")

    def _set_batch_idle_state(self) -> None:
        """Restore normal actions after a batch stops or finishes."""
        self.btn_load.config(state="normal")
        self.btn_load_3d.config(state="normal")
        self.btn_load_piv_result.config(state="normal")
        if self.loaded_stack is None:
            self._set_idle_state()
        else:
            self._set_loaded_state()
            self.btn_export.config(
                state="normal" if self.current_result is not None else "disabled"
            )
        self.btn_export_video.config(
            state="normal"
            if self.analysis_mode == "2d"
            and isinstance(self.current_result, BatchPIVResult)
            else "disabled"
        )
        self.btn_pause.config(state="disabled")
        self.btn_continue.config(state="disabled")
        self.btn_abort.config(state="disabled")
    
    def _set_idle_state(self) -> None:
        self.btn_single.config(state="disabled")
        self.btn_batch.config(state="disabled")
        self.btn_export.config(state="disabled")
        self.btn_export_video.config(state="disabled")
        self.btn_apply_postprocess.config(state="disabled")

    def _set_loaded_state(self) -> None:
        self.btn_apply_postprocess.config(state="disabled")
        
        if self.analysis_mode == "3d":
            self.btn_single.config(state="disabled")
            self.btn_batch.config(state="normal")
            self.btn_export_video.config(state="disabled")
            return

        self.btn_single.config(state="normal")
        self.btn_batch.config(state="normal")
        self.btn_export_video.config(state="disabled")

    def _apply_2d_parameter_defaults(self) -> None:
        """Apply safe default parameters for 2D PIV."""
        self.params_form.intersize_x.set("64")
        self.params_form.intersize_y.set("64")
        self.params_form.intersize_z.set("64")

        self.params_form.search_x.set("128")
        self.params_form.search_y.set("128")
        self.params_form.search_z.set("128")

        self.params_form.step_x.set("32")
        self.params_form.step_y.set("32")
        self.params_form.step_z.set("32")

        self.params_form.background_filter.set("Off")
        self.params_form.downsample_factor.set("1×")

    def _apply_3d_parameter_defaults(self) -> None:
        """Apply practical default parameters for 3D PIV."""
        self.params_form.intersize_x.set("20")
        self.params_form.intersize_y.set("20")
        self.params_form.intersize_z.set("20")

        self.params_form.search_x.set("8")
        self.params_form.search_y.set("8")
        self.params_form.search_z.set("8")

        self.params_form.step_x.set("10")
        self.params_form.step_y.set("10")
        self.params_form.step_z.set("10")

        self.params_form.background_filter.set("High")
        self.params_form.downsample_factor.set("3×")

    def _show_loaded_frame(self, frame_index: int) -> None:
        if self.loaded_stack is None:
            return

        if self.analysis_mode == "3d":
            self._show_3d_summary(title="3D stack loaded")
            return

        frame = self.loaded_stack.data[frame_index]
        draw_loaded_frame(
            self.preview_ax,
            self.preview_canvas,
            self.preview_state,
            frame,
            title=f"Frame {frame_index}",
        )
    
    def _show_3d_summary(self, *, title: str = "3D PIV") -> None:
        """Show a text summary for 3D mode instead of an image/vector preview."""
        self.preview_ax.clear()
        self.preview_ax.axis("off")
        self.preview_ax.set_title(title)

        if self.loaded_stack is None:
            summary = "No 3D stack loaded."
        else:
            stack_label = self.loaded_stack.dataset_name or self.loaded_stack.source_path.name
            num_pairs = max(self.loaded_stack.num_frames - 1, 0)

            median_state = (
                "enabled" if bool(self.params_form.despike.get()) else "disabled"
            )

            summary_lines = [
                "3D PIV results are saved for external visualization and analysis.",
                "",
                f"Input: {stack_label}",
                f"Time points (T): {self.loaded_stack.num_frames}",
                f"Frame pairs: {num_pairs}",
                f"Shape (T, Z, Y, X): {self.loaded_stack.shape}",
                "",
                "Available:",
                "  - batch 3D PIV",
                "  - export to HDF5 and/or NPZ",
                "  - export VTK for ParaView",
                "  - U, V, W vector components",
                "  - xgrid, ygrid, zgrid coordinates",
                f"  - 3D median despike: currently {median_state}",
            ]

            if (
                self._last_pair_elapsed_seconds is not None
                or self._last_3d_batch_elapsed_seconds is not None
                or self._last_3d_export_summary is not None
            ):
                summary_lines.extend(["", "Last 3D run:"])

                if self._last_pair_elapsed_seconds is not None:
                    summary_lines.append(
                        f"  - last pair time: {self._last_pair_elapsed_seconds:.1f} s"
                    )

                if self._last_3d_batch_elapsed_seconds is not None:
                    summary_lines.append(
                        f"  - total batch time: {self._last_3d_batch_elapsed_seconds:.1f} s"
                    )

                if self._last_3d_export_summary is not None:
                    summary_lines.append(f"  - exported: {self._last_3d_export_summary}")

            summary_lines.extend(
                [
                    "",
                    "3D visualization:",
                    "  - open exported VTK files manually in ParaView",
                    "",
                    "Not provided for 3D mode:",
                    "  - in-GUI 3D image/vector preview",
                    "  - video/GIF export",
                    "  - computeSN",
                    "  - SN filtering",
                ]
            )

            summary = "\n".join(summary_lines)

        self.preview_ax.text(
            0.03,
            0.97,
            summary,
            transform=self.preview_ax.transAxes,
            va="top",
            ha="left",
            fontsize=11,
            family="monospace",
        )
        self.preview_canvas.draw()

    def _show_loaded_3d_piv_result_summary(
        self,
        field_index: int = 0,
        *,
        title: str = "Loaded 3D PIV result",
    ) -> None:
        """Show a summary for loaded 3D PIV results instead of visualizing them."""
        self.preview_ax.clear()
        self.preview_ax.axis("off")
        self.preview_ax.set_title(title)

        if self.loaded_piv_result is None:
            summary = "No loaded PIV result."
        else:
            result = self.loaded_piv_result
            field_count = result.field_count
            idx = max(0, min(field_index, field_count - 1))

            median_state = (
                "enabled" if bool(self.params_form.despike.get()) else "disabled"
            )

            summary_lines = [
                "Loaded 3D PIV result",
                "",
                f"File: {result.source_path.name if result.source_path else '(unknown)'}",
                f"Fields: {field_count}",
                f"Current field index: {idx}",
                f"U shape: {result.u.shape}",
                f"V shape: {result.v.shape}",
                f"W shape: {result.w.shape if result.w is not None else None}",
                f"xgrid shape: {result.xg.shape}",
                f"ygrid shape: {result.yg.shape}",
                f"zgrid shape: {result.zg.shape if result.zg is not None else None}",
                f"valid_interrogation: {result.valid_interrogation is not None}",
                "",
                "Visualization:",
                "  - 3D vector-field preview is not supported inside the GUI.",
                "  - Export VTK and open it in ParaView for 3D visualization.",
                "",
                "Post-processing:",
                f"  - 3D median despike: currently {median_state}",
                "  - SN filtering is not supported for 3D vector fields.",
                "  - Apply post-processing, then export the updated result.",
            ]

            summary = "\n".join(summary_lines)

        self.preview_ax.text(
            0.03,
            0.97,
            summary,
            transform=self.preview_ax.transAxes,
            va="top",
            ha="left",
            fontsize=11,
            family="monospace",
        )
        self.preview_canvas.draw()

    def _show_pair_result(self, result: PIVPairResult, *, title: str) -> None:
        ensure_preview_artists(
            self.preview_ax,
            self.preview_canvas,
            self.preview_state,
            result.img1,
            result.img2,
            result.xg,
            result.yg,
            result.u,
            result.v,
            title=title,
        )

    def _set_result_view_state(self) -> None:
        """UI state for inspecting a saved PIV result without an image stack."""
        self.btn_single.config(state="disabled")
        self.btn_batch.config(state="disabled")
        self.btn_apply_postprocess.config(state="normal")
        self.btn_export.config(state="normal")
        self.btn_pause.config(state="disabled")
        self.btn_continue.config(state="disabled")
        self.btn_abort.config(state="disabled")
        self.btn_export_video.config(
            state="normal"
            if (
                self.loaded_piv_result is not None
                and self.loaded_piv_result.is_2d
                and self.loaded_piv_result.u.ndim == 3
            )
            else "disabled"
        )

    def _loaded_piv_field_count(self) -> int:
        """Return the number of vector fields in the loaded saved result."""
        if self.loaded_piv_result is None:
            return 0
        return self.loaded_piv_result.field_count

    def _show_loaded_piv_result(self, field_index: int) -> None:
        """Display a saved vector field without requiring an image stack."""
        if self.loaded_piv_result is None:
            return

        result = self.loaded_piv_result

        if result.is_3d:
            self._show_loaded_3d_piv_result_summary(field_index)
            return

        if result.u.ndim == 2:
            u = result.u
            v = result.v
            title = f"Loaded 2D PIV result: {result.source_path.name}"
        elif result.u.ndim == 3:
            idx = max(0, min(field_index, result.u.shape[0] - 1))
            u = result.u[idx]
            v = result.v[idx]
            title = f"Loaded 2D PIV result: Field {idx}"
        else:
            raise ValueError(
                f"Unsupported saved 2D U shape: {result.u.shape}"
            )

        draw_vector_field_only(
            self.preview_ax,
            self.preview_canvas,
            self.preview_state,
            result.xg,
            result.yg,
            u,
            v,
            title=title,
        )
    
    def _show_result_for_frame_index(self, frame_index: int) -> None:
        """Show the appropriate result view for the current slider position."""
        if self.analysis_mode == "3d" and self.loaded_piv_result is None:
            self._show_3d_summary(title="3D PIV result")
            return

        if self.loaded_piv_result is not None:
            self._show_loaded_piv_result(frame_index)
            return

        if self.current_result is None:
            self._show_loaded_frame(frame_index)
            return

        if isinstance(self.current_result, BatchPIVResult):
            if not self.current_result.pair_results:
                self._show_loaded_frame(frame_index)
                return

            pair_index = max(
                0,
                min(frame_index, len(self.current_result.pair_results) - 1),
            )
            pair_result = self.current_result.pair_results[pair_index]
            self._show_pair_result(
                pair_result,
                title=f"Batch PIV: Frame {pair_index} → {pair_index + 1}",
            )
            return

        if self.current_single_pair_indices is not None:
            t1, t2 = self.current_single_pair_indices
            title = f"Single PIV: Frame {t1} → {t2}"
        else:
            title = "Single PIV result"

        self._show_pair_result(self.current_result, title=title)

    def _ask_batch_run_options(self) -> BatchRunOptions | None:
        """Show the batch-run options dialog for the current analysis mode."""
        if self.analysis_mode == "3d":
            dialog = BatchExportDialog(self.root)
        else:
            dialog = BatchRunDialog(self.root)

        return dialog.result

    def _ask_batch_export_path(self) -> Path | None:
        """Ask for the base export path before starting the batch run."""
        save_path = filedialog.asksaveasfilename(
            title="Choose base name for batch PIV export",
            initialfile=self._build_export_name_hint(mode="batch"),
            defaultextension="",
            filetypes=[("All files", "*.*")],
        )
        if not save_path:
            return None

        path = Path(save_path)
        return self._strip_known_export_suffix(path)

    def _strip_known_export_suffix(self, path: Path) -> Path:
        """Strip only known export suffixes without damaging internal dots."""
        for suffix in (".npz", ".h5", ".vtk"):
            if path.name.lower().endswith(suffix):
                return path.with_name(path.name[: -len(suffix)])
        return path

    def _with_export_suffix(self, base_path: Path, suffix: str) -> Path:
        """Append an export suffix without removing text after internal dots."""
        base = self._strip_known_export_suffix(base_path)
        return base.with_name(base.name + suffix)

    def _export_batch_result_direct(
        self,
        result: BatchPIVResult,
        out_path: Path,
        *,
        export_npz: bool = False,
        export_h5: bool = False,
        export_vtk: bool = False,
        batch_elapsed_seconds: float | None = None,
        last_pair_elapsed_seconds: float | None = None,
    ) -> None:
        """Export a finished batch result directly to the chosen path."""
        written_names: list[str] = []

        if export_npz:
            npz_path = save_batch_result(self._with_export_suffix(out_path, ".npz"), result)
            written_names.append(npz_path.path.name)

        if export_h5:
            h5_path = save_batch_result(self._with_export_suffix(out_path, ".h5"), result)
            written_names.append(h5_path.path.name)

        if export_vtk:
            vtk_out_path = self._with_export_suffix(out_path, ".vtk")

            if self.analysis_mode == "3d":
                vtk_paths = save_batch_vector_fields_to_vtk(vtk_out_path, result)
            else:
                vtk_paths = save_batch_2d_vector_fields_to_vtk(vtk_out_path, result)

            if len(vtk_paths) == 1:
                written_names.append(vtk_paths[0].path.name)
            else:
                written_names.append(
                    f"{len(vtk_paths)} VTK files "
                    f"({vtk_paths[0].path.name} ... {vtk_paths[-1].path.name})"
                )

        if self.analysis_mode == "3d":
            self._last_3d_export_summary = ", ".join(written_names)

        message = "Saved: " + ", ".join(written_names)

        if self.analysis_mode == "3d":
            timing_parts = []

            if last_pair_elapsed_seconds is not None:
                timing_parts.append(f"last pair {last_pair_elapsed_seconds:.1f} s")

            if batch_elapsed_seconds is not None:
                timing_parts.append(f"total {batch_elapsed_seconds:.1f} s")

            if timing_parts:
                message += " | 3D PIV time: " + ", ".join(timing_parts)

        self._set_status("Export complete", 3000)
        self.var_result.set(message)

    def _on_frame_slider(self, _value: str) -> None:
        if self.loaded_stack is None and self.loaded_piv_result is None:
            return
        self._show_result_for_frame_index(int(self.var_frame.get()))
        
    def on_load_file(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.tif *.tiff *.h5")]
        )
        if not path:
            return

        try:
            loaded = load_stack(path)
            if not loaded.is_2d:
                raise ValueError(
                    "Use 'Load file for 3D PIV' for stacks with shape (T, Z, Y, X)."
                )

            self.analysis_mode = "2d"
            self._apply_2d_parameter_defaults()
            set_sn_controls_enabled(self.params_form, enabled=True)
            set_parameter_mode_note(self.params_form, spatial_ndim=2)
            self.loaded_stack = loaded
            self.loaded_piv_result = None
            self.current_result = None
            self.current_single_pair_indices = None
            self.current_export_name_hint = loaded.source_path.stem
            reset_preview_state(self.preview_state)

            self.var_file_name.set(f"2D PIV file: {loaded.source_path.name}")
            self.var_result.set(
                f"Loaded for 2D PIV: {loaded.source_path.name} – "
                f"Shape (T, H, W): {loaded.shape}"
            )
            self._set_status("2D file loaded", 3000)

            self.slider_frame.config(to=max(loaded.num_frames - 1, 0))
            self.var_frame.set(0)
            self._show_loaded_frame(0)
            self._set_loaded_state()
            self.btn_export.config(state="disabled")
            self.btn_export_video.config(state="disabled")
            self.progress["value"] = 0
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))
            self.var_result.set(f"Loading failed: {exc}")
            self._set_status("Load failed")

    def on_load_3d_file(self) -> None:
        """Load a 3D time-series stack for 3D PIV."""
        if self._show_3d_load_info:
            info_dialog = ThreeDLoadInfoDialog(self.root)
            if not info_dialog.result:
                return
            if bool(info_dialog.dont_show_again.get()):
                self._show_3d_load_info = False

        paths = filedialog.askopenfilenames(
            title="Load file(s) for 3D PIV",
            filetypes=[
                ("3D PIV input files", "*.h5 *.tif *.tiff"),
                ("HDF5 files", "*.h5"),
                ("TIFF files", "*.tif *.tiff"),
            ],
        )
        if not paths:
            return

        if len(paths) > 1:
            sorted_paths = sorted((Path(path) for path in paths), key=lambda path: path.name)
            order_dialog = ThreeDFileOrderDialog(self.root, sorted_paths)
            if not order_dialog.result:
                return
            paths = [str(path) for path in sorted_paths]

        try:
            if len(paths) == 1:
                loaded = load_stack(paths[0])

                if not loaded.is_3d:
                    raise ValueError(
                        "3D PIV requires either a single 4D stack with shape "
                        "(T, Z, Y, X), or multiple 3D TIFF time-point files. "
                        f"Loaded shape was {loaded.shape}."
                    )
            else:
                loaded = load_3d_tiff_sequence(paths)

            if not loaded.is_3d:
                raise ValueError(
                    "3D PIV requires data with shape (T, Z, Y, X). "
                    f"Loaded shape was {loaded.shape}."
                )

            self.analysis_mode = "3d"
            self._apply_3d_parameter_defaults()
            self.loaded_stack = loaded
            self.loaded_piv_result = None
            self.current_result = None
            self.current_single_pair_indices = None
            self.current_export_name_hint = loaded.source_path.stem

            self._batch_started_at = None
            self._last_pair_elapsed_seconds = None
            self._last_3d_batch_elapsed_seconds = None
            self._last_3d_export_summary = None

            # 3D computeSN and SN filtering are currently unavailable because
            # multi_quickPIV.compute_SN fails for 3D inputs. Median despike remains usable.
            self.params_form.compute_sn.set(False)
            self.params_form.sn_filter.set(False)
            set_sn_controls_enabled(self.params_form, enabled=False)
            set_parameter_mode_note(self.params_form, spatial_ndim=3)

            reset_preview_state(self.preview_state)

            self.var_file_name.set(f"3D PIV file: {loaded.source_path.name}")
            self.var_result.set(
                f"Loaded for 3D PIV: {loaded.dataset_name or loaded.source_path.name} – "
                f"Shape (T, Z, Y, X): {loaded.shape} | "
                "computeSN/SN filtering disabled for 3D"
            )
            self._set_status("3D file loaded", 3000)

            self.slider_frame.config(to=0)
            self.var_frame.set(0)

            self._show_3d_summary(title="3D stack loaded")

            self._set_loaded_state()
            self.btn_export.config(state="disabled")
            self.btn_export_video.config(state="disabled")
            self.progress["value"] = 0

        except Exception as exc:
            messagebox.showerror("Load 3D error", str(exc))
            self.var_result.set(f"Loading 3D file failed: {exc}")
            self._set_status("Load 3D failed")

    def on_load_piv_result(self) -> None:
        """Load a saved PIV result for vector-field-only inspection."""
        path = filedialog.askopenfilename(
            filetypes=[("PIV result files", "*.npz *.h5")]
        )
        if not path:
            return

        try:
            loaded = load_saved_piv_result(path)
            
            self.analysis_mode = "3d" if loaded.is_3d else "2d"

            if loaded.is_3d:
                self._apply_3d_parameter_defaults()
                self.params_form.compute_sn.set(False)
                self.params_form.sn_filter.set(False)
                set_sn_controls_enabled(self.params_form, enabled=False)
                set_parameter_mode_note(self.params_form, spatial_ndim=3)
            else:
                self._apply_2d_parameter_defaults()
                if loaded.sn is None:
                    self.params_form.compute_sn.set(False)
                    self.params_form.sn_filter.set(False)
                set_sn_controls_enabled(self.params_form, enabled=loaded.sn is not None)
                set_parameter_mode_note(self.params_form, spatial_ndim=2)
            self.loaded_stack = None
            self.loaded_piv_result = loaded
            self.current_result = None
            self.current_single_pair_indices = None
            self.current_export_name_hint = loaded.source_path.stem
            reset_preview_state(self.preview_state)

            self.var_file_name.set(f"PIV result: {loaded.source_path.name}")
            self.var_result.set(
                f"Loaded {'3D' if loaded.is_3d else '2D'} PIV result: "
                f"{loaded.source_path.name} – U shape: {loaded.u.shape}"
            )
            self._set_status("PIV result loaded", 3000)

            field_count = self._loaded_piv_field_count()
            self.slider_frame.config(to=max(field_count - 1, 0))
            self.var_frame.set(0)
            self._show_loaded_piv_result(0)
            self._set_result_view_state()
            self.btn_export_video.config(
                state="normal" if loaded.is_2d and loaded.u.ndim == 3 else "disabled"
            )
            self.progress["value"] = 0

        except Exception as exc:
            messagebox.showerror("Load PIV result error", str(exc))
            self.var_result.set(f"Loading PIV result failed: {exc}")
            self._set_status("Load PIV result failed")

    def on_run_single(self) -> None:
        if self.loaded_stack is None:
            messagebox.showerror("Error", "No file loaded.")
            return
        
        if self.analysis_mode == "3d":
            messagebox.showerror(
                "Single PIV error",
                "Single PIV preview is not available for 3D mode. Use batch export instead.",
            )
            return

        num_frames = self.loaded_stack.num_frames
        t1 = simpledialog.askinteger(
            "Single PIV",
            f"Start frame (0-{num_frames - 2}):",
            minvalue=0,
            maxvalue=num_frames - 2,
            parent=self.root,
        )
        if t1 is None:
            return

        try:
            params = build_workflow_params(
                self.params_form,
                spatial_ndim=self._current_spatial_ndim(),
            )
            result = run_piv_pair(
                self.loaded_stack.data[t1],
                self.loaded_stack.data[t1 + 1],
                params=params,
            )
            self.current_result = result
            self.current_single_pair_indices = (t1, t1 + 1)
            self.current_export_name_hint = self._build_export_name_hint(
                mode="single",
                frame1_idx=t1,
                frame2_idx=t1 + 1,
            )
            self._show_pair_result(
                result,
                title=f"Single PIV: Frame {t1} → {t1 + 1}",
            )
            self.var_result.set(
                f"Single PIV complete: {t1} → {t1 + 1} | "
                f"grid={result.xg.shape} | SN={result.sn is not None}"
            )
            self._set_status("Single PIV complete", 3000)
            self.btn_export.config(state="normal")
            self.btn_export_video.config(state="disabled")
        except Exception as exc:
            messagebox.showerror("Run error", str(exc))
            self.var_result.set(f"Single PIV failed: {exc}")
            self._set_status("Single PIV failed")

    def on_run_batch(self) -> None:
        if self.loaded_stack is None:
            messagebox.showerror("Error", "No file loaded.")
            return

        options = self._ask_batch_run_options()
        if options is None:
            return

        export_path: Path | None = None
        if options.export_after_run:
            export_path = self._ask_batch_export_path()
            if export_path is None:
                return

        try:
            params = build_workflow_params(
                self.params_form,
                spatial_ndim=self._current_spatial_ndim(),
            )

            if self.analysis_mode == "3d":
                # 3D computeSN currently fails inside multi_quickPIV.compute_SN
                # with a BoundsError from 4D indexing on a 3D correlation matrix.
                params.run.compute_sn = False
                params.postprocess.median_despike.enabled = False
                params.postprocess.sn_filter.enabled = False

        except Exception as exc:
            messagebox.showerror("Batch error", str(exc))
            self.var_result.set(f"Batch PIV failed: {exc}")
            self._set_status("Batch PIV failed")
            return

        total_pairs = self.loaded_stack.num_frames - 1
        self.batch.start(
            options=options,
            export_path=export_path,
            total_pairs=total_pairs,
            params=params,
        )

        self._batch_started_at = time.perf_counter()
        self._last_pair_elapsed_seconds = None

        self.progress["maximum"] = max(self.batch.total_pairs, 1)
        self.progress["value"] = 0

        self._set_batch_running_state()

        if self.analysis_mode == "3d":
            self._set_status("3D batch started, computeSN off")
        else:
            self._set_status("Batch started")

        self.root.after(1, self._run_next_batch_step)

    def _run_next_batch_step(self) -> None:
        """Run one batch step and schedule the next one."""
        if not self.batch.running:
            return

        if self.batch.abort_requested:
            self.batch.reset()
            self._batch_started_at = None
            self._last_pair_elapsed_seconds = None
            self.current_result = None
            self.current_single_pair_indices = None
            self._set_batch_idle_state()
            self._set_status("Batch aborted", 3000)
            self.var_result.set("Batch aborted")
            if self.loaded_stack is not None:
                self._show_loaded_frame(int(self.var_frame.get()))
            return

        if self.batch.paused:
            self._set_status("Batch paused")
            return

        if self.loaded_stack is None or self.batch.params is None:
            self.batch.reset()
            self._set_batch_idle_state()
            self._set_status("Batch stopped")
            return

        if self.batch.is_finished():
            result = self.batch.build_batch_result()
            export_path = self.batch.export_path
            options = self.batch.options

            self.current_result = result
            self.current_single_pair_indices = None
            self.current_export_name_hint = self._build_export_name_hint(mode="batch")
            self.batch.reset()
            self._set_batch_idle_state()
            self.btn_export_video.config(
                state="normal" if self.analysis_mode == "2d" else "disabled"
            )

            if (
                options is not None
                and options.export_after_run
                and export_path is not None
            ):
                try:
                    batch_elapsed = None
                    if self._batch_started_at is not None:
                        batch_elapsed = time.perf_counter() - self._batch_started_at

                    if self.analysis_mode == "3d":
                        self._last_3d_batch_elapsed_seconds = batch_elapsed

                    self._export_batch_result_direct(
                        result,
                        export_path,
                        export_npz=(options is not None and options.export_npz),
                        export_h5=(options is not None and options.export_h5),
                        export_vtk=(options is not None and options.export_vtk),
                        batch_elapsed_seconds=batch_elapsed,
                        last_pair_elapsed_seconds=self._last_pair_elapsed_seconds,
                    )

                    if self.analysis_mode == "3d":
                        self._show_3d_summary(title="3D batch complete")

                except Exception as exc:
                    messagebox.showerror("Export error", str(exc))
                    self.var_result.set(f"Batch export failed: {exc}")
                    self._set_status("Batch finished, export failed")
                return

            if self.analysis_mode == "3d":
                total_elapsed = None
                if self._batch_started_at is not None:
                    total_elapsed = time.perf_counter() - self._batch_started_at

                elapsed_text = (
                    f" | total time={total_elapsed:.1f} s"
                    if total_elapsed is not None
                    else ""
                )

                self._last_3d_batch_elapsed_seconds = total_elapsed
                self._show_3d_summary(title="3D batch complete")

                self.var_result.set(
                    f"3D batch PIV complete: {len(result.pair_results)} pairs | "
                    f"grid={result.xg.shape if result.xg is not None else None} | "
                    f"SN=False{elapsed_text}"
                )
                self._set_status(
                    f"3D batch PIV complete{elapsed_text}",
                    3000,
                )
            else:
                self.var_result.set(
                    f"Batch PIV complete: {len(result.pair_results)} pairs | "
                    f"grid={result.xg.shape if result.xg is not None else None} | "
                    f"SN={result.sn_list is not None}"
                )
                self._set_status("Batch PIV complete", 3000)

            return

        t = self.batch.next_pair_index

        try:
            pair_start = time.perf_counter()

            if self.analysis_mode == "3d":
                self.progress["value"] = t + 0.5
                self.var_result.set(
                    f"Processing 3D pair {t + 1}/{self.batch.total_pairs}..."
                )
                self._set_status(
                    f"Processing 3D pair {t + 1}/{self.batch.total_pairs}; "
                    "this may take time for large volumes."
                )
                self.root.update_idletasks()

            pair_result = run_piv_pair(
                self.loaded_stack.data[t],
                self.loaded_stack.data[t + 1],
                params=self.batch.params,
            )

            pair_elapsed = time.perf_counter() - pair_start
            self._last_pair_elapsed_seconds = pair_elapsed

            self.batch.append_result(pair_result)
            self.progress["value"] = self.batch.next_pair_index

            if (
                self.analysis_mode == "2d"
                and self.batch.options is not None
                and self.batch.options.preview_mode == "live"
            ):
                self._show_pair_result(
                    pair_result,
                    title=f"Batch PIV: Frame {t} → {t + 1}",
                )

            if self.analysis_mode == "3d":
                self.var_result.set(
                    f"Processed 3D pair {self.batch.next_pair_index}/"
                    f"{self.batch.total_pairs} in {pair_elapsed:.1f} s"
                )
                self._set_status(
                    f"Processed 3D pair {self.batch.next_pair_index}/"
                    f"{self.batch.total_pairs} in {pair_elapsed:.1f} s"
                )
            else:
                self._set_status(
                    f"Processed pair {self.batch.next_pair_index}/{self.batch.total_pairs}"
                )
            self.root.after(1, self._run_next_batch_step)

        except Exception as exc:
            self.batch.reset()
            self._set_batch_idle_state()
            messagebox.showerror("Batch error", str(exc))
            self.var_result.set(f"Batch PIV failed: {exc}")
            self._set_status("Batch PIV failed")

    def on_abort_batch(self) -> None:
        """Request that the running batch stop after the current step."""
        if not self.batch.running:
            return

        was_paused = self.batch.paused
        self.batch.abort_requested = True
        self.batch.paused = False
        self.btn_pause.config(state="disabled")
        self.btn_continue.config(state="disabled")
        self._set_status("Batch abort requested...")

        if was_paused:
            self.root.after(1, self._run_next_batch_step)

    def on_pause_batch(self) -> None:
        """Pause the running batch after the current step."""
        if not self.batch.running:
            return
        if self.batch.paused:
            return

        self.batch.paused = True
        self.btn_pause.config(state="disabled")
        self.btn_continue.config(state="normal")
        self._set_status("Batch paused")

    def on_continue_batch(self) -> None:
        """Resume a paused batch."""
        if not self.batch.paused:
            return

        self.batch.paused = False
        self.btn_pause.config(state="normal")
        self.btn_continue.config(state="disabled")
        self._set_status("Batch resuming...")
        self.root.after(1, self._run_next_batch_step)

    def _postprocess_loaded_piv_result(
        self,
        result: LoadedPIVResult,
    ) -> LoadedPIVResult:
        """Apply configured post-processing to a loaded 2D or 3D PIV result."""
        params = build_workflow_params(
            self.params_form,
            spatial_ndim=result.spatial_ndim,
        )

        if result.is_3d:
            params.run.compute_sn = False
            params.postprocess.sn_filter.enabled = False

        if result.is_2d:
            if result.u.ndim == 2:
                processed = apply_postprocessing(
                    result.u,
                    result.v,
                    params=params,
                    sn=result.sn,
                )
                return LoadedPIVResult(
                    u=processed.u,
                    v=processed.v,
                    xg=result.xg,
                    yg=result.yg,
                    sn=processed.sn,
                    source_path=result.source_path,
                    valid_interrogation=result.valid_interrogation,
                )

            u_fields = []
            v_fields = []
            sn_fields = [] if result.sn is not None else None

            for index in range(result.u.shape[0]):
                sn_i = result.sn[index] if result.sn is not None else None

                processed = apply_postprocessing(
                    result.u[index],
                    result.v[index],
                    params=params,
                    sn=sn_i,
                )

                u_fields.append(processed.u)
                v_fields.append(processed.v)

                if sn_fields is not None and processed.sn is not None:
                    sn_fields.append(processed.sn)

            return LoadedPIVResult(
                u=np.stack(u_fields),
                v=np.stack(v_fields),
                xg=result.xg,
                yg=result.yg,
                sn=np.stack(sn_fields) if sn_fields is not None else None,
                source_path=result.source_path,
                valid_interrogation=result.valid_interrogation,
            )

        if result.w is None:
            raise ValueError("Loaded 3D PIV result is missing W components.")

        if result.u.ndim == 3:
            processed = apply_postprocessing(
                result.u,
                result.v,
                w=result.w,
                params=params,
            )
            return LoadedPIVResult(
                u=processed.u,
                v=processed.v,
                w=processed.w,
                xg=result.xg,
                yg=result.yg,
                zg=result.zg,
                sn=None,
                source_path=result.source_path,
                valid_interrogation=result.valid_interrogation,
            )

        u_fields = []
        v_fields = []
        w_fields = []

        for index in range(result.u.shape[0]):
            processed = apply_postprocessing(
                result.u[index],
                result.v[index],
                w=result.w[index],
                params=params,
            )

            u_fields.append(processed.u)
            v_fields.append(processed.v)

            if processed.w is None:
                raise ValueError("3D post-processing unexpectedly returned no W field.")
            w_fields.append(processed.w)

        return LoadedPIVResult(
            u=np.stack(u_fields),
            v=np.stack(v_fields),
            w=np.stack(w_fields),
            xg=result.xg,
            yg=result.yg,
            zg=result.zg,
            sn=None,
            source_path=result.source_path,
            valid_interrogation=result.valid_interrogation,
        )

    def on_apply_postprocessing(self) -> None:
        """Apply the current post-processing settings to a loaded PIV result."""
        if self.loaded_piv_result is None:
            messagebox.showerror(
                "Post-processing error",
                "Load a saved PIV result before applying post-processing.",
            )
            return

        try:
            self.loaded_piv_result = self._postprocess_loaded_piv_result(
                self.loaded_piv_result
            )

            field_index = int(self.var_frame.get())
            self._show_loaded_piv_result(field_index)

            self.btn_export.config(state="normal")
            self._set_status("Post-processing applied", 3000)
            self.var_result.set(
                f"Post-processing applied to loaded "
                f"{'3D' if self.loaded_piv_result.is_3d else '2D'} PIV result."
            )

        except Exception as exc:
            messagebox.showerror("Post-processing error", str(exc))
            self.var_result.set(f"Post-processing failed: {exc}")
            self._set_status("Post-processing failed")

    def on_export_current(self) -> None:
        if self.current_result is None and self.loaded_piv_result is None:
            messagebox.showerror("Export error", "No result available for export.")
            return

        out_path = filedialog.asksaveasfilename(
            title="Export PIV result",
            initialfile=self.current_export_name_hint,
            defaultextension=".npz",
            filetypes=[
                ("NumPy zipped", "*.npz"),
                ("HDF5", "*.h5"),
                ("VTK", "*.vtk"),
            ],
        )
        if not out_path:
            return

        try:
            out_path_obj = Path(out_path)
            suffix = out_path_obj.suffix.lower()

            if self.loaded_piv_result is not None:
                export_result = save_loaded_piv_result(
                    out_path_obj,
                    self.loaded_piv_result,
                )

                if isinstance(export_result, list):
                    saved_text = f"{len(export_result)} VTK file(s)"
                else:
                    saved_text = export_result.path.name

            elif isinstance(self.current_result, PIVPairResult):
                if suffix == ".vtk":
                    batch_wrapper = BatchPIVResult(pair_results=[self.current_result])
                    if self.current_result.w is None:
                        vtk_paths = save_batch_2d_vector_fields_to_vtk(
                            out_path_obj,
                            batch_wrapper,
                        )
                    else:
                        vtk_paths = save_batch_vector_fields_to_vtk(
                            out_path_obj,
                            batch_wrapper,
                        )
                    saved_text = f"{len(vtk_paths)} VTK file(s)"
                else:
                    export_path = save_pair_result(out_path_obj, self.current_result)
                    saved_text = export_path.path.name

            else:
                if suffix == ".vtk":
                    if self.analysis_mode == "3d":
                        vtk_paths = save_batch_vector_fields_to_vtk(
                            out_path_obj,
                            self.current_result,
                        )
                    else:
                        vtk_paths = save_batch_2d_vector_fields_to_vtk(
                            out_path_obj,
                            self.current_result,
                        )
                    saved_text = f"{len(vtk_paths)} VTK file(s)"
                else:
                    export_path = save_batch_result(out_path_obj, self.current_result)
                    saved_text = export_path.path.name

            self._set_status("Export complete", 3000)
            self.var_result.set(f"Saved: {saved_text}")

        except Exception as exc:
            messagebox.showerror("Export error", str(exc))
            self.var_result.set(f"Export failed: {exc}")
            self._set_status("Export failed")

    def on_export_animation(self) -> None:
        """Export the current batch-like vector field sequence as MP4 or GIF."""
        u = None
        v = None
        xg = None
        yg = None

        if isinstance(self.current_result, BatchPIVResult):
            if not self.current_result.pair_results:
                messagebox.showerror("Export video error", "No batch result available.")
                return
            u = self.current_result.u_list
            v = self.current_result.v_list
            xg = self.current_result.xg
            yg = self.current_result.yg

        elif (
            self.loaded_piv_result is not None
            and self.loaded_piv_result.is_2d
            and self.loaded_piv_result.u.ndim == 3
        ):
            u = self.loaded_piv_result.u
            v = self.loaded_piv_result.v
            xg = self.loaded_piv_result.xg
            yg = self.loaded_piv_result.yg

        else:
            messagebox.showerror(
                "Export video error",
                "Animation export requires a batch result or a loaded multi-field PIV result.",
            )
            return

        out_path = filedialog.asksaveasfilename(
            title="Export PIV animation",
            initialfile=self.current_export_name_hint,
            defaultextension=".mp4",
            filetypes=[
                ("MP4 Video", "*.mp4"),
                ("GIF Animation", "*.gif"),
            ],
        )
        if not out_path:
            return

        try:
            u_arr = u if isinstance(u, tuple) else u
            v_arr = v if isinstance(v, tuple) else v

            if isinstance(u_arr, list):
                import numpy as _np
                u_arr = _np.stack(u_arr)
            if isinstance(v_arr, list):
                import numpy as _np
                v_arr = _np.stack(v_arr)

            export_path = save_piv_animation(
                out_path,
                u=u_arr,
                v=v_arr,
                xg=xg,
                yg=yg,
            )
            self._set_status("Video export complete", 3000)
            self.var_result.set(f"Saved animation: {export_path.path.name}")

        except Exception as exc:
            messagebox.showerror("Export video error", str(exc))
            self.var_result.set(f"Video export failed: {exc}")
            self._set_status("Video export failed")

    def _build_export_name_hint(
        self,
        *,
        mode: str,
        frame1_idx: int | None = None,
        frame2_idx: int | None = None,
    ) -> str:
        """Build the default export filename (without extension)."""
        if self.loaded_stack is not None:
            stem = self.loaded_stack.source_path.stem
        elif (
            self.loaded_piv_result is not None
            and self.loaded_piv_result.source_path is not None
        ):
            stem = self.loaded_piv_result.source_path.stem
        else:
            stem = "piv_result"

        if mode == "single" and frame1_idx is not None and frame2_idx is not None:
            return f"{stem}_piv_{frame1_idx:04d}_to_{frame2_idx:04d}"
        if mode == "batch":
            return f"{stem}_batch_piv"
        return stem
    
    def show_piv_info(self) -> None:
        """Show an informational pop-up about the current PIV capabilities."""
        msg = (
            "This program supports 2D PIV with preview and export.\n\n"
            "3D PIV is available for batch processing of time-series stacks shaped "
            "as (T, Z, Y, X). The computed 3D vector fields can be saved as HDF5, "
            "NPZ, and VTK files. VTK files can be opened manually in ParaView for "
            "3D visualization.\n\n"
            "The current implementation uses an FFT-based windowed "
            "cross-correlation algorithm via the multi_quickPIV library."
        )
        messagebox.showinfo("PIV Configuration Information", msg)

    def _set_status(self, text: str, timeout_ms: int | None = None) -> None:
        """Set the status-bar text, optionally clearing it after a delay."""
        if self._status_after_id is not None:
            try:
                self.root.after_cancel(self._status_after_id)
            except Exception:
                pass
            self._status_after_id = None

        self.var_status.set(text)

        if timeout_ms is not None and timeout_ms > 0:
            def _clear_status() -> None:
                self.var_status.set("")
                self._status_after_id = None

            self._status_after_id = self.root.after(timeout_ms, _clear_status)

    def _current_spatial_ndim(self) -> int:
        """Return the active PIV spatial dimensionality."""
        return 3 if self.analysis_mode == "3d" else 2


def main() -> None:
    """Entry point for the minimal GUI shell."""
    root = tk.Tk()
    app = MultiQuickPIVApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()