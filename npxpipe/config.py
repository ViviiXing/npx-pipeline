"""All user-adjustable settings of the pipeline.

Change values here instead of editing the step code. Every value is copied into the
run metadata (run_meta.json) so that results can always be traced back to settings.
"""
import os

# ---------------------------------------------------------------- parallel processing
# Number of worker processes SpikeInterface may use when it reads/writes data in chunks.
# Leave two CPU cores free so the computer stays responsive.
N_JOBS = max(1, min(8, (os.cpu_count() or 4) - 2))
CHUNK_DURATION = "1s"          # size of each chunk processed by one worker

# ---------------------------------------------------------------- inspection / QC
EXPECTED_N_CHANNELS = 384      # Neuropixels 1.0 records 384 channels at a time
EXPECTED_FS = 30000.0          # AP-band sampling rate (Hz)
HIGHPASS_HZ = 300.0            # high-pass used ONLY for bad-channel detection and QC plots
BAD_CHANNEL_METHOD = "coherence+psd"   # IBL method; gives good / noise / dead / out labels
QC_WINDOW_S = 120              # length of the start / middle / end windows for detection
QC_SNIPPETS = 20               # number of 1-s snippets used for RMS / activity-vs-depth
# A channel is proposed for removal if it is not "good" in the whole-recording detection
# OR in at least this many of the three time windows.
BAD_IN_N_WINDOWS = 2

# ---------------------------------------------------------------- sorting
KS_FIXED_PARAMS = dict(
    torch_device="cuda",        # fail loudly if the GPU is not available (no silent CPU)
    delete_recording_dat=False, # KEEP recording.dat: Phy needs it to show waveforms
)
DEFAULT_NBLOCKS = 1            # Kilosort4 default: rigid drift correction
ALT_NBLOCKS = 5                # Kilosort docs: 5 can be a good choice for single-shank NP probes
DISK_MARGIN_GB = 10            # extra free space required on top of the estimated need

# ---------------------------------------------------------------- drift check
DRIFT_TOL_UM = 10.0            # half of the 20-um NP1.0 site pitch
DRIFT_WIN_S = 60.0             # time window for per-unit position medians
DRIFT_MIN_SPIKES = 30          # spikes a unit needs in a window for that window to count
DRIFT_MIN_COVER = 0.8          # a unit must be measurable in >= 80 % of windows
DRIFT_BIN_UM = 400.0           # depth bins used to summarise units

# ---------------------------------------------------------------- file names
STATE_FILE = "npx_pipeline_state.json"   # written in the session root folder
LOG_FILE = "npx_pipeline_log.txt"        # written in the session root folder
# Files Open Ephys writes into every continuous-stream folder. They are NEVER moved,
# renamed, overwritten or deleted by this pipeline.
RAW_FILES = {"continuous.dat", "sample_numbers.npy", "timestamps.npy"}
# Files/folders this pipeline writes into the AP folder that are NOT sorting results.
QC_PREFIXES = ("qc_",)
KEEP_DIRS = {"_previous_runs"}

# ---------------------------------------------------------------- waveforms / post-processing
# Run only after manual curation in Phy (step 'waveforms').
UNIT_LABELS = ["good"]          # which manual labels to analyse; e.g. ["good", "mua"]
ANALYSIS_FREQ_MIN = 300.0       # analysis chain band-pass (Hz) - same idea as Phy's display
ANALYSIS_FREQ_MAX = 6000.0
MAX_SPIKES_PER_UNIT = 500       # random spikes per unit used for waveforms / PCA
WF_MS_BEFORE = 1.0              # waveform window before the spike peak (ms)
WF_MS_AFTER = 2.0               # waveform window after the spike peak (ms)
CCG_WINDOW_MS = 100.0           # auto-correlogram window (total width, ms)
CCG_BIN_MS = 1.0
QUALITY_METRICS = ["num_spikes", "firing_rate", "presence_ratio", "snr", "isi_violation",
                   "rp_violation", "amplitude_cutoff", "amplitude_median",
                   "mahalanobis", "d_prime"]
BURST_ISI_MS = 6.0              # burst index = fraction of ISIs shorter than this

# Putative cell-type rules: CellExplorer 'standard' schema (Petersen et al., Neuron 2021;
# defaults from CellExplorer preferences_ProcessCellMetrics.m). Firing rate is NOT used.
# Applied ONLY to probes whose region is listed here, because the thresholds were set on
# hippocampal / cortical data and do not transfer to other regions (e.g. LS).
CELLTYPE_REGIONS = ["dCA1", "CA1"]
NARROW_TROUGH_TO_PEAK_MS = 0.425        # trough-to-peak <= this: narrow interneuron
WIDE_INTERNEURON_MIN_TAU_RISE_MS = 6.0  # trough-to-peak > 0.425 ms and ACG tau_rise > this: wide interneuron
# everything else: putative pyramidal cell
# tau_rise comes from the triple-exponential ACG fit of CellExplorer fit_ACG.m
# (0.5 ms bins, +-50 ms), computed in npxpipe/celltype.py independently of CCG_* above.

