# multi_quickPIV GUI - 2D and 3D PIV with Python and Julia

This project provides a graphical user interface for processing Particle Image Velocimetry (PIV) data using the Julia [`multi_quickPIV`](https://github.com/Marc-3d/multi_quickPIV) backend.

The GUI supports:

- **2D PIV** with image/vector preview, single-pair analysis, batch analysis, post-processing, and export.
- **3D PIV** for batch processing of time-series volume data, with export to analysis formats and ParaView-compatible VTK files.

The interface is built with Python and Tkinter. The actual PIV computation is carried out in Julia using the [`multi_quickPIV`](https://github.com/Marc-3d/multi_quickPIV) backend.

The goal of this project is to make `multi_quickPIV` easier to use for users who prefer a visual workflow instead of directly calling the Julia package from code.

For 3D PIV, the GUI computes vector fields from time-series volume data and saves the results for downstream analysis and visualization. The GUI does not render 3D volumes or 3D vector fields internally; 3D vector-field visualization is done manually by opening the exported VTK files in ParaView.

## Features

The GUI can load PIV input data in TIFF format (`.tif`, `.tiff`) and HDF5 format (`.h5`).

Supported input formats are:

- **2D PIV time series**: a stack shaped as `(T, H, W)`, where `T` is time/frame index.
- **3D PIV time series**: a stack shaped as `(T, Z, Y, X)`.
- **Separate 3D TIFF time points**: multiple TIFF files where each file is one 3D volume shaped as `(Z, Y, X)`. These files are sorted by filename, shown to the user for confirmation, and internally stacked as `(T, Z, Y, X)`.

Two main 2D evaluation modes are available:

- **Single PIV**: performs the PIV calculation between one selected frame and the following frame.
- **Batch PIV**: automatically processes all consecutive frame pairs in the loaded image stack.

For 3D PIV, the GUI supports:

- loading a single 4D HDF5/TIFF stack
- loading multiple 3D TIFF time-point files
- running batch 3D PIV
- factor-of downsampling as a pre-PIV step
- a pre-PIV Background filter for skipping low-signal interrogation volumes
- exporting vector components and grid coordinates
- exporting ParaView-compatible VTK files for manual 3D visualization

The workflow also includes optional post-processing tools for improving computed vector fields:

- median despiking for 2D and 3D vector fields
- signal-to-noise filtering for 2D PIV only
- spatio-temporal averaging for batch or loaded multi-field vector-field sequences

During 2D processing, the GUI provides an interactive preview. Individual frames can be viewed, and computed velocity vectors are displayed as a quiver plot over the image data.

For 3D processing, computed vector fields are saved to disk. Visualization of 3D vector fields is handled outside the GUI by opening the exported VTK files in ParaView.

The results can be exported as:

- NumPy archive (`.npz`) for storing vector components and grid coordinates
- HDF5 file (`.h5`) for storing vector components and grid coordinates
- VTK files (`.vtk`) for visualizing 2D or 3D vector fields in ParaView
- optional video or GIF for 2D batch vector-field evolution

## Requirements

To use the GUI, you need:

- Git
- Conda or Miniconda
- Python 3.10
- Julia
- a working terminal or command prompt

Julia must be available from the terminal. You can check this with:

```bash
julia -v
```

For MP4 video export, a working FFmpeg installation is useful. If FFmpeg is not available, GIF export can be used instead.

## Installation

Clone the repository:

```bash
git clone https://github.com/tjawad/multi_quickPIV_GUI.git
cd multi_quickPIV_GUI
```

Create the Python environment:

```bash
conda env create -f environment.yml
conda activate quickpiv
```

Install the GUI package in editable mode:

```bash
pip install -e .
```

Bind PyJulia/PyCall to the active Python environment:

```bash
python -c "import julia; julia.install()"
```

This step is important because the Python GUI communicates with the Julia backend through PyJulia.

## Launching the GUI

After installation, the GUI can be started with:

```bash
quickpiv-gui
```

On first launch, Julia may instantiate the local Julia environment and install `multi_quickPIV` if needed. This can take some time. Later launches should be faster.

## Exact Windows environment

The default `environment.yml` is intended to be portable and should be used by most users.

For reproducing the original tested Windows setup more exactly, use:

```bash
conda env create -f environment-windows-lock.yml
conda activate quickpiv
pip install -e .
python -c "import julia; julia.install()"
quickpiv-gui
```

## Processing workflow

The GUI follows a three-stage processing workflow:

1. **Pre-processing / input conditioning**  
   Options such as downsampling and background filtering are applied before PIV computation.

2. **PIV computation**  
   The PIV settings define the interrogation size, search margin, step size, correlation algorithm, and whether signal-to-noise values are computed.

3. **Post-processing**  
   Post-processing options modify the resulting vector fields after PIV computation. These options can be selected before running PIV so they are applied automatically to the output, or applied later using **Apply post-processing** on the current computed result or on a loaded saved PIV result.

### Post-processing options

The post-processing controls operate on vector fields after they have been computed.

- **Median despike** removes local outlier vectors using a neighbourhood-based median filter.
- **SN filtering** removes or replaces vectors based on signal-to-noise values when SN values are available.
- **Spatio-temporal averaging** smooths vector-field sequences using `multi_quickPIV.average`.

Post-processing can be configured before running PIV, in which case it is applied to the generated result, or applied afterward using **Apply post-processing**. The same button can be used for loaded saved PIV results and for the current result displayed in the GUI.

For single vector fields, median despiking and SN filtering can be applied, but spatio-temporal averaging requires a batch or loaded multi-field result.

### Spatio-temporal averaging

Spatio-temporal averaging is intended for smoothing vector-field sequences, such as batch PIV results or loaded multi-field PIV results.

For 2D PIV, the spatial radius defines a square neighbourhood around each vector, and the temporal radius defines how many neighbouring vector fields are included before and after the current field. The operation uses `multi_quickPIV.average`, so it averages over the full local space-time neighbourhood.

For 3D PIV, temporal averaging is not currently supported. The temporal radius is fixed to `0`, and the spatial radius defines a cubic neighbourhood applied to each 3D vector field independently.

Because this implementation uses `multi_quickPIV.average`, it may smooth more strongly than the original quickPIV space-time averaging routine, which combines spatial neighbourhood information with a temporal line through the same location. Use smaller radii when preserving local flow structure is important.

## Basic workflow

### 2D PIV workflow

A typical 2D workflow is:

1. Start the GUI with `quickpiv-gui`.
2. Select **Load file for 2D PIV**.
3. Load a TIFF or HDF5 image stack shaped as `(T, H, W)`.
4. Adjust the pre-processing, PIV, and post-processing parameters if needed.
5. Run a single PIV calculation first to check the result.
6. If the result looks reasonable, run batch PIV.
7. Apply additional post-processing to the current result if needed.
8. Export the computed vector fields as NPZ/HDF5, or as VTK files for ParaView visualization.
9. Optionally export a video or GIF of the 2D vector fields.

### 3D PIV workflow

A typical 3D workflow is:

1. Start the GUI with `quickpiv-gui`.
2. Select **Load file for 3D PIV**.
3. Load either:
   - one 4D HDF5/TIFF stack shaped as `(T, Z, Y, X)`, or
   - multiple 3D TIFF time-point files, each shaped as `(Z, Y, X)`.

   When multiple 3D TIFF time-point files are selected, the GUI orders them by filename and shows the sorted order for confirmation before loading.

   Use zero-padded time-point names so that filename sorting matches the real acquisition order, for example `object_t000.tif`, `object_t001.tif`, and `object_t002.tif`.

   Avoid non-padded names such as `object_t1.tif`, `object_t2.tif`, and `object_t10.tif`, because filename sorting can place `t10` before `t2`.

4. Adjust the pre-processing, PIV, and post-processing parameters if needed.
5. Run batch PIV.
6. Apply additional post-processing to the current result if needed.
7. Choose one or more output options:
   - HDF5 (`.h5`) for storing vector data
   - NumPy zipped (`.npz`) for storing vector data
   - VTK (`.vtk`) for ParaView visualization
8. Export the 3D vector fields.
9. Open the exported VTK file(s) manually in ParaView to visualize the 3D vector field.

## Loading saved PIV results

Saved PIV result files can be loaded back into the GUI using **Load PIV result**.

Supported saved result formats are:

- HDF5 (`.h5`)
- NumPy zipped (`.npz`)

For saved 2D PIV results, the GUI displays the vector fields directly. If the result contains multiple frame-pair fields, the frame slider can be used to inspect each vector field.

For saved 3D PIV results, the GUI shows a summary panel instead of rendering the 3D vector field internally. The summary includes the number of fields, vector-component shapes, grid shapes, and whether a `valid_interrogation` mask is present. To visualize 3D vector fields, export the loaded result as VTK and open it in ParaView.

Loaded PIV results can be post-processed using the current post-processing settings and then exported again as `.h5`, `.npz`, or `.vtk`. This provides a way to adjust post-processing settings without rerunning the original PIV computation.

For loaded 2D multi-field results, median despiking, SN filtering, and spatio-temporal averaging are available when the required data are present.

For loaded 3D results, median despiking and spatial averaging are available, but signal-to-noise filtering remains disabled and temporal averaging is fixed to `0`.

## Visualizing 2D PIV results in ParaView

For 2D batch PIV results, VTK export writes one `.vtk` file per processed frame pair. The files are written as an indexed sequence, for example:

```text
sample_000.vtk
sample_001.vtk
sample_002.vtk
```

The 2D VTK files contain a flat vector field with Z = 0. In ParaView, apply Glyph, set Orientation Array to directions, and set Scale Array to direction_mag.

## Visualizing 3D PIV results in ParaView

The GUI exports 3D vector fields as legacy ASCII VTK files (`.vtk`). These files are intended to be opened manually in ParaView.

A typical ParaView workflow is:

1. Open ParaView.
2. Select **File > Open**.
3. Choose the exported `.vtk` file.
4. Click **Apply** in the Properties panel.
5. Apply **Threshold**.
6. Set the threshold array to `valid_interrogation`.
7. Keep only the range `1` to `1`.
8. Click **Apply**.
9. Apply **Glyph** to the thresholded data.
10. Set **Orientation Array** to `directions`. This tells ParaView which way each PIV vector points.
11. Set **Scale Array** to `direction_mag`. This scales arrow length by the displacement magnitude instead of drawing all arrows at the same size.
12. Adjust the glyph scale factor as needed.
13. Optionally color by `direction_mag`.

The VTK export includes these arrays:

- `finite_mask`: marks vectors with finite numeric components
- `valid_interrogation`: marks interrogation volumes that passed the Background filter
- `directions`: vector array used by ParaView Glyph to orient arrows
- `direction_mag`: displacement magnitude of each vector, used to scale or color glyphs

`direction_mag` represents displacement magnitude in voxel units per frame pair; physical velocity requires voxel-size and time-interval calibration.

For batch 3D PIV results, the GUI writes one VTK file per processed frame pair. Open the desired time-pair file in ParaView, or load multiple files if you want to inspect several computed vector fields.

## PIV parameters

The main PIV parameters are:

- **interSize**: size of the interrogation window in pixels.
- **searchMargin**: search area around the interrogation window.
- **step**: spacing between neighboring vectors.
- **Downsampling**: factor-of pre-PIV downsampling. `1×` means no downsampling.
- **Background filter**: skips low-signal interrogation regions before PIV. Available levels are `Off`, `Low`, `Medium`, `High`, and `Very High`.
- **computeSN**: enables signal-to-noise computation in the Julia backend.

The GUI displays spatial parameters in user-facing order:

```text
X, Y, Z
```

Internally, these are converted to the array order expected by the backend:

```text
2D PIV:
  backend order = (Y, X)

3D PIV:
  backend order = (Z, Y, X)
```

For 2D PIV, the `Z` parameter field is ignored.

Smaller steps create denser vector fields but increase computation time. Larger interrogation windows can make the correlation more stable but reduce spatial resolution.

Default settings depend on the selected workflow:

```text
2D PIV defaults:
  Downsampling = 1×
  Background filter = Off

3D PIV defaults:
  Downsampling = 3×
  Background filter = High
```

For 3D data, High is the recommended starting point for the Background filter.

### 3D signal-to-noise limitation

At present, `computeSN` and signal-to-noise filtering are disabled in 3D mode.

This is because 3D `computeSN=true` currently triggers a backend error inside `multi_quickPIV.compute_SN`. The GUI therefore forces `computeSN=False` for 3D PIV until the backend issue is resolved.

Median despiking remains available for 3D vector fields.

## Smoke test

A command-line smoke test is available, but it requires an input image stack.

The repository does not include large test image stacks. To run the smoke test, place your own `.tif`, `.tiff`, or `.h5` stack somewhere locally, for example in a local `test_data/` folder.

The `test_data/` folder is ignored by Git so that large microscopy files are not accidentally committed.

For a single frame pair:

```bash
python scripts/smoke_test_pipeline.py test_data/example_stack.h5 --mode single --frame-index 0 --out test_outputs/example_single_result.npz
```

For a full batch run:

```bash
python scripts/smoke_test_pipeline.py test_data/example_stack.h5 --mode batch --out test_outputs/example_batch_result.npz
```

The output format is selected by the file extension passed to `--out`. Use `.npz` for a NumPy archive or `.h5` for HDF5 output. VTK export is available through the GUI batch export workflow for both 2D and 3D results.

Additional smoke tests are available for the 3D workflow:

```bash
python scripts/smoke_test_3d_bridge.py
python scripts/smoke_test_3d_batch_export.py
python scripts/smoke_test_3d_median_despike.py
python scripts/smoke_test_params_mapping.py
python scripts/smoke_test_3d_tiff_sequence_loading.py
```

A smoke-test runner is also available:

```bash
python scripts/run_smoke_tests.py
```

By default, this runs the lightweight Python-only smoke tests.

To also run the Julia-backed smoke tests, use:

```bash
python scripts/run_smoke_tests.py --include-julia
```

The Julia-backed tests initialize the Julia runtime and may take longer to run.

These tests check:

- Python-to-Julia 3D PIV bridge behavior
- 3D batch export/reload behavior
- 3D median despiking
- GUI parameter mapping from `X, Y, Z` to backend tuple order
- loading separate 3D TIFF volumes as a time series

## 3D real-data validation

The 3D workflow has been locally validated using cropped time-point volumes from the example dataset associated with Pereyra et al. (2021), the original quickPIV publication.

The validation input was a cropped HDF5 stack with shape:

```text
(T, Z, Y, X) = (2, 128, 256, 256)
```

The validation checked the following path:

```text
→ load 3D stack
→ run 3D batch PIV
→ apply 3D median despike
→ export NPZ
→ reload NPZ
→ export HDF5
→ reload HDF5
→ export VTK
```

The resulting vector-field shapes were:

```text
Per frame pair:
  U, V, W: (4, 8, 8)

Batch export:
  U, V, W: (1, 4, 8, 8)

Grids:
  xgrid, ygrid, zgrid: (4, 8, 8)

SN:
  None, as expected for 3D mode
```

Recent 3D VTK exports also include `finite_mask`, `valid_interrogation`, `directions`, and `direction_mag`, allowing ParaView users to threshold out skipped/background interrogation volumes before applying Glyph.

The cropped validation data are not included in the repository because microscopy datasets are large and the `test_data/` folder is intentionally ignored by Git.

## Project structure

The repository is organized as a Python package:

```text
multi_quickPIV_GUI/
|-- src/multi_quickpiv_gui/   # GUI, backend bridge, runtime, and workflow code
|-- julia_env/                # local Julia environment for multi_quickPIV
|-- scripts/                  # helper scripts and smoke tests
|-- requirements.txt          # Python dependencies
|-- environment.yml           # portable conda environment
|-- environment-windows-lock.yml
|-- pyproject.toml            # Python package and launcher configuration
`-- README.md
```

The Julia backend is managed through the local `julia_env/` directory. This keeps the Julia dependency setup separate from the user's global Julia environment.

## Notes

This GUI is a frontend for `multi_quickPIV`. The Julia backend itself remains a separate Julia-only project.

The GUI supports interactive 2D PIV and batch-based 3D PIV. For 3D datasets, the GUI computes and exports vector fields; 3D visualization is performed manually in ParaView using the exported VTK files.

Current 3D design choices and limitations:

- 3D image volumes are not previewed inside the GUI
- 3D vector fields are visualized in ParaView, not inside the GUI
- saved 3D results show a summary panel inside the GUI, but 3D vector-field visualization is done in ParaView
- 3D video/GIF export is not provided
- 3D signal-to-noise computation and SN filtering are disabled until the backend issue is resolved

This project is under active development.
