"""Dialog windows for the multi_quickPIV GUI."""

from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk


@dataclass(slots=True)
class BatchRunOptions:
    """User choices for a batch PIV run."""

    preview_mode: str = "live"
    export_after_run: bool = False
    export_npz: bool = False
    export_h5: bool = False
    export_vtk: bool = False


class BatchRunDialog(simpledialog.Dialog):
    """Modal dialog for configuring a batch run."""

    def __init__(self, parent) -> None:
        self.result: BatchRunOptions | None = None
        self.var_preview_mode = tk.StringVar(value="live")
        self.var_export_after = tk.BooleanVar(value=False)
        self.var_export_npz = tk.BooleanVar(value=False)
        self.var_export_h5 = tk.BooleanVar(value=False)
        self.var_export_vtk = tk.BooleanVar(value=False)
        self.export_checkbuttons: list[ttk.Checkbutton] = []
        super().__init__(parent, title="Batch PIV Options")

    def body(self, master):
        ttk.Label(
            master,
            text="Choose how this batch run should behave.",
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        preview_frame = ttk.LabelFrame(master, text="Preview", padding=8)
        preview_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        ttk.Radiobutton(
            preview_frame,
            text="Live preview",
            value="live",
            variable=self.var_preview_mode,
        ).grid(row=0, column=0, sticky="w")

        ttk.Radiobutton(
            preview_frame,
            text="No preview",
            value="off",
            variable=self.var_preview_mode,
        ).grid(row=1, column=0, sticky="w")

        export_frame = ttk.LabelFrame(master, text="Export", padding=8)
        export_frame.grid(row=2, column=0, columnspan=2, sticky="ew")

        ttk.Checkbutton(
            export_frame,
            text="Export automatically after batch run",
            variable=self.var_export_after,
            command=self._toggle_export_widgets,
        ).grid(row=0, column=0, columnspan=2, sticky="w")

        npz_check = ttk.Checkbutton(
            export_frame,
            text="NumPy zipped (.npz)",
            variable=self.var_export_npz,
        )
        npz_check.grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))

        h5_check = ttk.Checkbutton(
            export_frame,
            text="HDF5 (.h5)",
            variable=self.var_export_h5,
        )
        h5_check.grid(row=2, column=0, columnspan=2, sticky="w")

        vtk_check = ttk.Checkbutton(
            export_frame,
            text="VTK (.vtk)",
            variable=self.var_export_vtk,
        )
        vtk_check.grid(row=3, column=0, columnspan=2, sticky="w")

        self.export_checkbuttons = [npz_check, h5_check, vtk_check]

        self._toggle_export_widgets()
        return preview_frame

    def buttonbox(self):
        box = ttk.Frame(self)

        ttk.Button(box, text="Run", command=self.ok).pack(side="left", padx=5, pady=5)
        ttk.Button(box, text="Cancel", command=self.cancel).pack(side="left", padx=5, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

        box.pack()

    def _toggle_export_widgets(self) -> None:
        state = "normal" if self.var_export_after.get() else "disabled"
        for widget in self.export_checkbuttons:
            widget.config(state=state)

    def apply(self) -> None:
        export_after = bool(self.var_export_after.get())

        self.result = BatchRunOptions(
            preview_mode=self.var_preview_mode.get(),
            export_after_run=export_after,
            export_npz=export_after and bool(self.var_export_npz.get()),
            export_h5=export_after and bool(self.var_export_h5.get()),
            export_vtk=export_after and bool(self.var_export_vtk.get()),
        )

    def validate(self) -> bool:
        if self.var_export_after.get() and not (
            self.var_export_npz.get()
            or self.var_export_h5.get()
            or self.var_export_vtk.get()
        ):
            messagebox.showerror(
                "Export selection required",
                "Please select at least one export option.",
                parent=self,
            )
            return False

        return True

class BatchExportDialog(simpledialog.Dialog):
    """Modal dialog for configuring 3D batch export outputs."""

    def __init__(self, parent) -> None:
        self.result: BatchRunOptions | None = None
        self.var_export_npz = tk.BooleanVar(value=False)
        self.var_export_h5 = tk.BooleanVar(value=False)
        self.var_export_vtk = tk.BooleanVar(value=True)
        super().__init__(parent, title="3D Batch PIV Export")

    def body(self, master):
        ttk.Label(
            master,
            text="Choose which outputs to create for the 3D batch result.",
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        store_frame = ttk.LabelFrame(
            master,
            text="Store vector data",
            padding=8,
        )
        store_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        ttk.Checkbutton(
            store_frame,
            text="HDF5 (.h5)",
            variable=self.var_export_h5,
        ).grid(row=0, column=0, sticky="w")

        ttk.Checkbutton(
            store_frame,
            text="NumPy zipped (.npz)",
            variable=self.var_export_npz,
        ).grid(row=1, column=0, sticky="w")

        paraview_frame = ttk.LabelFrame(
            master,
            text="Visualize with ParaView",
            padding=8,
        )
        paraview_frame.grid(row=2, column=0, columnspan=2, sticky="ew")

        ttk.Checkbutton(
            paraview_frame,
            text="VTK (.vtk)",
            variable=self.var_export_vtk,
        ).grid(row=0, column=0, sticky="w")

        return store_frame

    def buttonbox(self):
        box = ttk.Frame(self)

        ttk.Button(box, text="Run and export", command=self.ok).pack(
            side="left", padx=5, pady=5
        )
        ttk.Button(box, text="Cancel", command=self.cancel).pack(
            side="left", padx=5, pady=5
        )

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

        box.pack()

    def apply(self) -> None:
        self.result = BatchRunOptions(
            preview_mode="off",
            export_after_run=True,
            export_npz=bool(self.var_export_npz.get()),
            export_h5=bool(self.var_export_h5.get()),
            export_vtk=bool(self.var_export_vtk.get()),
        )

    def validate(self) -> bool:
        if not (
            self.var_export_npz.get()
            or self.var_export_h5.get()
            or self.var_export_vtk.get()
        ):
            messagebox.showerror(
                "Export selection required",
                "Please select at least one export option.",
                parent=self,
            )
            return False
        return True

class ThreeDLoadInfoDialog(simpledialog.Dialog):
    """Explain accepted 3D input formats before opening the file picker."""

    def __init__(self, parent) -> None:
        self.result: bool = False
        self.dont_show_again = tk.BooleanVar(value=False)
        super().__init__(parent, title="3D PIV input files")

    def body(self, master):
        message = (
            "For 3D PIV, you can load either:\n\n"
            "1. One 4D HDF5/TIFF stack shaped as (T, Z, Y, X), or\n"
            "2. Multiple 3D TIFF files, where each file is one time point "
            "shaped as (Z, Y, X).\n\n"
            "When multiple TIFF files are selected, they are ordered by filename. "
            "Use zero-padded timepoint names such as:\n\n"
            "  object_t000.tif\n"
            "  object_t001.tif\n"
            "  object_t002.tif\n\n"
            "Avoid names such as t1, t2, t10 because filename sorting can place "
            "t10 before t2."
        )

        ttk.Label(
            master,
            text=message,
            justify="left",
            wraplength=520,
        ).grid(row=0, column=0, sticky="w", pady=(0, 10))

        ttk.Checkbutton(
            master,
            text="Don't show again during this session",
            variable=self.dont_show_again,
        ).grid(row=1, column=0, sticky="w")

        return master

    def buttonbox(self):
        box = ttk.Frame(self)

        ttk.Button(box, text="Continue", command=self.ok).pack(
            side="left", padx=5, pady=5
        )
        ttk.Button(box, text="Cancel", command=self.cancel).pack(
            side="left", padx=5, pady=5
        )

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

        box.pack()

    def apply(self) -> None:
        self.result = True

class ThreeDFileOrderDialog(simpledialog.Dialog):
    """Confirm the filename-sorted order for multiple 3D TIFF time points."""

    def __init__(self, parent, sorted_paths) -> None:
        self.sorted_paths = list(sorted_paths)
        self.result: bool = False
        super().__init__(parent, title="Confirm 3D TIFF time order")

    def body(self, master):
        ttk.Label(
            master,
            text=(
                "The selected 3D TIFF files will be loaded in filename order.\n"
                "This order defines the time axis used for 3D PIV.\n\n"
                "Please confirm that this is the correct time order:"
            ),
            justify="left",
            wraplength=620,
        ).grid(row=0, column=0, sticky="w", pady=(0, 10))

        frame = ttk.Frame(master)
        frame.grid(row=1, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(frame, orient="vertical")
        listbox = tk.Listbox(
            frame,
            width=90,
            height=min(15, len(self.sorted_paths)),
        )

        listbox.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=listbox.yview)

        for index, path in enumerate(self.sorted_paths):
            listbox.insert("end", f"{index:04d}: {path.name}")

        ttk.Label(
            master,
            text=(
                "If this order is wrong, cancel and rename the files using "
                "zero-padded timepoint numbers."
            ),
            justify="left",
            wraplength=620,
        ).grid(row=2, column=0, sticky="w", pady=(10, 0))

        return listbox

    def buttonbox(self):
        box = ttk.Frame(self)

        ttk.Button(box, text="Use this order", command=self.ok).pack(
            side="left", padx=5, pady=5
        )
        ttk.Button(box, text="Cancel", command=self.cancel).pack(
            side="left", padx=5, pady=5
        )

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

        box.pack()

    def apply(self) -> None:
        self.result = True


