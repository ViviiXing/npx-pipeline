"""Find probes and stream folders in an Open Ephys session; keep the pipeline state.

Layout written by this pipeline (same style as the earlier 'checkdata' sessions):

  <session>/
    npx_pipeline_state.json      <- what has been done, user decisions (bad channels, regions)
    npx_pipeline_log.txt         <- full log of every pipeline run
    Record Node 101/experiment1/recording1/continuous/
      Neuropix-PXI-100.ProbeA-AP/
        continuous.dat, sample_numbers.npy, timestamps.npy   <- raw Open Ephys files (untouched)
        qc_*.png / qc_*.csv / qc_*.json                      <- inspection & bad-channel files
        recording.dat                                        <- sorting input (bad channels removed + phase_shift)
        params.py, spike_times.npy, ... kilosort4.log         <- Kilosort4 / Phy files
        run_meta.json, drift_*.png, qc_drift_check.json
        ks_auto_backup/      <- automatic Kilosort labels, saved before any manual curation
        spikeinterface_run/  <- SpikeInterface run logs
        _previous_runs/      <- older results moved aside when a probe is re-sorted
"""
import re
from pathlib import Path

import numpy as np

from . import config
from .utils import read_json, write_json, now


class Session:
    """One Open Ephys recording session (the folder that contains 'Record Node ...')."""

    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Session folder not found: {self.root}")
        if not list(self.root.glob("Record Node*")):
            raise FileNotFoundError(f"No 'Record Node ...' folder inside {self.root}. "
                                    "Give the session folder (the one containing 'Record Node 101').")
        self.state_path = self.root / config.STATE_FILE
        self.log_path = self.root / config.LOG_FILE
        self.state = read_json(self.state_path, default={"session": str(self.root), "probes": {}, "history": []})

    # ------------------------------------------------------------------ state handling
    def save(self):
        """Write the state dict back to npx_pipeline_state.json."""
        self.state["updated"] = now()
        write_json(self.state_path, self.state)

    def probe_state(self, probe):
        """Return (and create if needed) the state sub-dict of one probe."""
        return self.state["probes"].setdefault(probe, {})

    def mark(self, probe, step, **info):
        """Record that `step` finished for `probe` (probe=None for session-level steps)."""
        entry = {"step": step, "probe": probe, "time": now(), **info}
        self.state["history"].append(entry)
        target = self.state if probe is None else self.probe_state(probe)
        target.setdefault("done", {})[step] = now()
        self.save()

    def is_done(self, probe, step):
        target = self.state if probe is None else self.state["probes"].get(probe, {})
        return step in target.get("done", {})

    # ------------------------------------------------------------------ discovery
    def discover(self):
        """Read the session with Neo and list every Neuropixels AP stream (one per probe).

        Returns a dict {probe_name: {"stream": neo_stream_name, "ap_dir": folder}}.
        Raises an error if the session has several blocks/segments (not handled here).
        """
        import neo  # imported here so that 'status' works even without heavy packages

        io = neo.rawio.OpenEphysBinaryRawIO(dirname=str(self.root))
        io.parse_header()
        n_blocks = io.block_count()
        n_segments = io.segment_count(0)
        names = list(io.header["signal_streams"]["name"])

        probes = {}
        for name in names:
            # e.g. "Record Node 101#Neuropix-PXI-100.ProbeA-AP"
            if not name.endswith("-AP") or "#" not in name:
                continue
            node, folder = name.split("#", 1)
            probe = folder.split(".")[-1][:-3]          # "ProbeA-AP" -> "ProbeA"
            ap_dirs = list(self.root.glob(f"{node}/experiment*/recording*/continuous/{folder}"))
            probes[probe] = {"stream": name, "ap_dir": str(ap_dirs[0]) if len(ap_dirs) == 1 else None,
                             "n_ap_dirs_found": len(ap_dirs)}

        info = {"streams": names, "n_blocks": n_blocks, "n_segments": n_segments, "probes": probes}
        return info

    def ap_dir(self, probe):
        """Folder of the AP stream of `probe` (where all results are written)."""
        p = self.state["probes"].get(probe, {}).get("ap_dir")
        if not p:
            raise RuntimeError(f"{probe}: AP folder unknown - run the 'inspect' step first.")
        return Path(p)

    def stream(self, probe):
        return self.state["probes"][probe]["stream"]

    def probes(self, selected=None):
        """Probe names known to the state, optionally filtered by a user selection."""
        names = sorted(self.state["probes"])
        if selected:
            unknown = [p for p in selected if p not in names]
            if unknown:
                raise ValueError(f"Unknown probe(s) {unknown}; known: {names}")
            names = [p for p in names if p in selected]
        return names


def first_sample_number(ap_dir):
    """First Open Ephys sample number of a stream (needed later for time alignment)."""
    sn = np.load(Path(ap_dir) / "sample_numbers.npy", mmap_mode="r")
    return int(sn[0])


def read_params_py(ap_dir):
    """Parse params.py (Phy's settings file) into a dict of strings."""
    txt = (Path(ap_dir) / "params.py").read_text()
    out = {}
    for key in ("n_channels_dat", "dtype", "sample_rate", "dat_path"):
        m = re.search(rf"^{key}\s*=\s*(.*)$", txt, flags=re.M)
        out[key] = m.group(1).strip() if m else None
    out["dat_file"] = None
    if out["dat_path"]:
        q = re.search(r"['\"](.+?)['\"]", out["dat_path"])   # text between the first pair of quotes
        out["dat_file"] = q.group(1) if q else None
    return out
