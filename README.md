# npx-pipeline

A Neuropixels 1.0 processing pipeline for Windows. It takes raw Open Ephys recordings through SpikeInterface preprocessing, GPU Kilosort4 spike sorting, and manual curation in Phy. After curation it computes quality metrics, checks drift, and assigns putative cell types. Everything runs on a single Windows machine.

```
Open Ephys raw data
  → SpikeInterface: load and check (bad channels, phase shift)
  → Kilosort4 (called via SpikeInterface run_sorter, on GPU)
  → Phy manual curation
  → SpikeInterface: quality metrics, drift check, cell-type classification
```

## Repository contents

| File / folder | Purpose |
|---|---|
| `pipeline.py` | Main entry point |
| `npxpipe/session.py` | Load an Open Ephys session, list streams and probes |
| `npxpipe/qc.py` | Signal and bad-channel checks |
| `npxpipe/sorting.py` | Build the sorting input and run Kilosort4 |
| `npxpipe/curation.py` | Create the Phy working copy and read back curated results |
| `npxpipe/postprocess.py` | SortingAnalyzer and quality metrics |
| `npxpipe/drift.py` | Drift plots |
| `npxpipe/celltype.py` | Putative cell-type classification (Python implementation of CellExplorer criteria) |
| `npxpipe/config.py` | Parameters |
| `npxpipe/utils.py` | Shared helpers |
| `setup_env.ps1` | Sets up both Python environments |
| `requirements_lock_analysis.txt` | Pinned versions for the analysis environment |
| `requirements_lock_phy.txt` | Pinned versions for the Phy environment |
| `env_gpu.txt` | Record of the tested GPU, driver and CUDA versions |
| `.vscode/tasks.json` | Ready-to-run VS Code tasks |

Raw data and sorting outputs are **not** included in this repository. You need to supply your own data.

## Requirements

- **OS**: Windows 10/11
- **GPU**: NVIDIA, 12 GB VRAM or more recommended (developed on an RTX 4070 12 GB), with a recent driver. See `env_gpu.txt` for tested versions
- **RAM**: 64 GB recommended
- **Disk**: NVMe SSD. Per probe-hour of recording, plan on about 90 GB for raw data plus about 83 GB for the sorting input `recording.dat`, plus other outputs. Leave free space of at least 2.5× the raw data size
- **Software**: [Git](https://git-scm.com/download/win), [uv](https://docs.astral.sh/uv/), VS Code (optional)

## Installation

### 1. Clone the repository

```powershell
cd D:\
git clone https://github.com/ViviiXing/npx-pipeline.git
cd npx-pipeline
```

### 2. Set up the environments

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
```

The script uses the lock files to create two separate environments:

- **Analysis environment** (`.venv`, Python 3.11) with SpikeInterface, Kilosort4, CUDA PyTorch and related packages
- **Phy environment**, kept separate from the analysis environment to avoid PyTorch/Qt conflicts

### 3. Check the GPU

```powershell
.\.venv\Scripts\python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The output should show `True` followed by your GPU name. If it shows `False`, see Troubleshooting.

## Usage

### Data

Keep the folder structure that Open Ephys creates and **do not rename anything**. For example, the space in `Record Node 101` must stay. Sorting outputs are written inside the session folder.

### Running the pipeline

Open this folder in VS Code, press `Ctrl+Shift+P` → **Tasks: Run Task**, and choose a task.

Or from the command line:

```powershell
.\.venv\Scripts\python pipeline.py --help
```

to see the available steps and options. Recommended order:

1. **Check**: load the data and confirm probe info, channel count and sampling rate; review the bad-channel detection
2. **Sort**: run the full pipeline on a ~5-minute test segment first, then run Kilosort4 on the full recording. Process probes one at a time, with only one GPU job running at once
3. **Curate**: open the working copy in Phy, label clusters good / mua / noise, and merge or split clusters where needed
4. **Post-curation analysis**: read the Phy results back and compute quality metrics, drift plots and putative cell types

### Important notes

- **Keep `recording.dat`.** Phy needs it to display waveforms. The pipeline sets `delete_recording_dat=False`, so do not delete this file by hand.
- **Always run Kilosort4 through this pipeline.** Do not load raw data directly into the Kilosort4 GUI for production sorting. If you do, phase-shift correction and bad-channel removal are not applied.
- **Channel numbers in Phy** are column indices in `recording.dat`, which has bad channels removed. They are not the original channel IDs.
- **Multi-probe sync:** do not use the `timestamps.npy` files from Open Ephys online sync. With a 50 Hz sync square wave, the timestamps of non-main probes have been found to jump by tens to hundreds of milliseconds. Use sample numbers and TTL rising edges, with an offline linear mapping between probes.

## Troubleshooting

| Problem | Cause and fix |
|---|---|
| `import torch` fails with `WinError 1114` | You are running Anaconda's Python. Use uv's Python 3.11 and run from `.venv` |
| `torch.cuda.is_available()` is `False` | Either a CPU build of PyTorch was installed or the driver is too old. Update the driver and rerun `setup_env.ps1` |
| Phy views are blank | Windows long-path support is off. As administrator, run `New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force`, then restart |
| Phy cannot find the data file | Check that `dat_path` in the working copy's `params.py` points to an existing `recording.dat` |
| Script hangs or keeps spawning processes | This is a Windows multiprocessing issue. Run from a `.py` script; to debug, set `n_jobs=1` |
| Out of GPU memory or RAM during sorting | Close other GPU programs and set `clear_cache=True`. If it still fails, increase `cluster_downsampling` moderately |
| Errors with output paths | Use short output paths with no spaces or non-ASCII characters, outside OneDrive or other synced folders |

## Software versions

Production analysis uses the pinned versions in the lock files. Do not upgrade packages casually. To upgrade, create a new environment and re-validate it on the test segment.

Main dependencies:
[SpikeInterface](https://spikeinterface.readthedocs.io/) ·
[Kilosort4](https://github.com/MouseLand/Kilosort) ·
[Phy](https://github.com/cortex-lab/phy) ·
[PyTorch](https://pytorch.org/)

## Author

Vivi ([@ViviiXing](https://github.com/ViviiXing))
