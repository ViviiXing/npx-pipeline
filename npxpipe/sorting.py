"""Step 'sort': build the sorting input and run Kilosort4 on the GPU.

Sorting input chain:
    raw AP  ->  remove bad channels  ->  phase_shift (stays int16)  ->  Kilosort4
No filtering / CAR / whitening here: Kilosort4 does those internally.

Kilosort4 results are written DIRECTLY into the probe's AP folder (checkdata layout).
"""
import platform
import shutil
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .qc import load_raw
from .session import first_sample_number, read_params_py
from .utils import log, ok, fail, write_json, gb, stamp, now


# ============================================================================ input chain
def load_sorting_input(session, probe):
    """Raw AP -> remove the confirmed bad channels -> phase_shift (int16)."""
    import spikeinterface.preprocessing as spre

    ps = session.state["probes"][probe]
    if "bad_channel_ids" not in ps:
        raise RuntimeError(f"{probe}: bad channels not confirmed yet - run the 'badch' step first.")
    rec = load_raw(session, probe)
    rec = rec.remove_channels(ps["bad_channel_ids"])
    rec = spre.phase_shift(rec, dtype="int16")
    return rec


def _set_job_kwargs():
    import spikeinterface as si
    si.set_global_job_kwargs(n_jobs=config.N_JOBS, chunk_duration=config.CHUNK_DURATION, progress_bar=True)


def _versions():
    return {p: version(p) for p in ["kilosort", "spikeinterface", "probeinterface", "neo", "torch", "numpy"]}


# ============================================================================ checks
def check_sorter_output(folder, n_samples, n_channels):
    """Check a Kilosort4 output folder (here: the AP folder). Returns (summary, problems)."""
    folder = Path(folder)
    problems = []

    def check(cond, good_msg, bad_msg):
        (ok if cond else fail)(good_msg if cond else bad_msg)
        if not cond:
            problems.append(bad_msg)

    dat = folder / "recording.dat"
    expect = n_samples * n_channels * 2
    check(dat.exists() and dat.stat().st_size == expect,
          f"recording.dat present, size correct ({gb(expect)} GB)", "recording.dat missing or wrong size")

    prm = read_params_py(folder)
    check(prm["n_channels_dat"] == str(n_channels), f"params.py n_channels_dat = {n_channels}",
          f"params.py n_channels_dat = {prm['n_channels_dat']} (expected {n_channels})")
    check("int16" in (prm["dtype"] or ""), "params.py dtype int16", f"params.py dtype {prm['dtype']}")
    dat_in_params = Path(prm["dat_file"] or "")
    check(prm["dat_file"] is not None and dat_in_params.exists(), "params.py dat_path exists",
          f"params.py dat_path not found: {prm['dat_path']}")

    st = np.load(folder / "spike_times.npy").ravel()
    sc = np.load(folder / "spike_clusters.npy").ravel() if (folder / "spike_clusters.npy").exists() \
        else np.load(folder / "spike_templates.npy").ravel()
    cm = np.load(folder / "channel_map.npy").ravel()
    check(len(st) == len(sc) and len(st) > 0, f"{len(st):,} spikes, {len(np.unique(sc))} clusters",
          "spike_times / spike_clusters length mismatch or empty")
    check(st.min() >= 0 and st.max() < n_samples, f"spike indices within 0..{n_samples - 1}",
          f"spike indices out of range ({st.min()}..{st.max()})")
    check(len(cm) == n_channels, f"channel_map has {len(cm)} channels", f"channel_map has {len(cm)} channels")

    logtxt = (folder / "kilosort4.log").read_text(errors="ignore") if (folder / "kilosort4.log").exists() else ""
    check("GPU memory" in logtxt or "CUDA memory" in logtxt, "kilosort4.log shows GPU use",
          "kilosort4.log does not show GPU use")

    kl = {}
    if (folder / "cluster_KSLabel.tsv").exists():
        kl = pd.read_csv(folder / "cluster_KSLabel.tsv", sep="\t").KSLabel.value_counts().to_dict()
    log(f"  KSLabel: {kl}")
    summary = dict(n_spikes=int(len(st)), n_units=int(len(np.unique(sc))), KSLabel=kl)
    return summary, problems



# ============================================================================ full run
def existing_results(ap_dir):
    """Names of everything in the AP folder that is a sorting result (not raw, not QC)."""
    names = []
    for p in Path(ap_dir).iterdir():
        if p.name in config.RAW_FILES or p.name in config.KEEP_DIRS:
            continue
        if p.is_file() and p.name.startswith(config.QC_PREFIXES):
            continue
        names.append(p.name)
    return sorted(names)


def archive_results(ap_dir, label):
    """Move existing sorting results to <AP>/_previous_runs/<time>_<label>/.

    recording.dat (large, can be regenerated) is deleted instead of moved.
    Returns the archive folder.
    """
    ap_dir = Path(ap_dir)
    names = existing_results(ap_dir)
    dest = ap_dir / "_previous_runs" / f"{stamp()}_{label}"
    dest.mkdir(parents=True)
    for name in names:
        src = ap_dir / name
        if name == "recording.dat":
            src.unlink()
            continue
        shutil.move(str(src), str(dest / name))
    return dest


def run_full(session, probe, nblocks):
    """Sort the whole recording of one probe and place the results in the AP folder.

    Steps: build input -> run Kilosort4 in a temporary folder -> move results into the
    AP folder -> point params.py to recording.dat -> back up automatic labels -> check.
    """
    import torch
    import spikeinterface.sorters as ss

    assert torch.cuda.is_available(), "CUDA not available - stopping."
    _set_job_kwargs()
    ap_dir = session.ap_dir(probe)
    if existing_results(ap_dir):
        raise RuntimeError(f"{probe}: AP folder already holds sorting results; archive them first.")

    rec = load_sorting_input(session, probe)
    n, nch = rec.get_num_samples(), rec.get_num_channels()
    need = n * nch * 2 + 5e9
    free = shutil.disk_usage(ap_dir).free
    if free < need + config.DISK_MARGIN_GB * 1e9:
        raise RuntimeError(f"Not enough disk space: need ~{gb(need)} GB + margin, free {gb(free)} GB")
    log(f"  input: {nch} channels, {n/rec.get_sampling_frequency():.1f} s, nblocks={nblocks}; "
        f"free disk {gb(free)} GB")

    tmp = ap_dir / "_ks_tmp"
    params = dict(config.KS_FIXED_PARAMS, nblocks=nblocks)
    t0 = time.time()
    ss.run_sorter("kilosort4", recording=rec, folder=tmp, remove_existing_folder=True, verbose=True, **params)
    runtime = (time.time() - t0) / 60

    # ---- move results into the AP folder (raw files are never touched)
    so = tmp / "sorter_output"
    clash = [p.name for p in so.iterdir() if (ap_dir / p.name).exists()]
    if clash:
        raise RuntimeError(f"Name clash with existing files in the AP folder: {clash}")
    for p in so.iterdir():
        shutil.move(str(p), str(ap_dir / p.name))
    si_dir = ap_dir / "spikeinterface_run"
    si_dir.mkdir(exist_ok=True)
    for p in tmp.iterdir():
        if p.is_file():
            shutil.move(str(p), str(si_dir / p.name))
    shutil.rmtree(tmp, ignore_errors=True)

    # ---- params.py must point to recording.dat at its new place
    prm = ap_dir / "params.py"
    lines = prm.read_text().splitlines()
    lines = [f"dat_path = r'{(ap_dir / 'recording.dat').as_posix()}'" if l.startswith("dat_path") else l
             for l in lines]
    prm.write_text("\n".join(lines) + "\n")

    # ---- keep the automatic Kilosort labels before anyone edits them in Phy
    bk = ap_dir / "ks_auto_backup"
    bk.mkdir(exist_ok=True)
    for name in ["spike_clusters.npy", "cluster_group.tsv", "cluster_KSLabel.tsv"]:
        if (ap_dir / name).exists():
            shutil.copy2(ap_dir / name, bk / name)

    log(f"  Kilosort4 finished in {runtime:.1f} min; checking results ...")
    summary, problems = check_sorter_output(ap_dir, n, nch)

    ps = session.probe_state(probe)
    meta = dict(probe=probe, region=ps.get("region", ""), finished=now(), nblocks=nblocks,
                ks_params=params, n_channels=nch, n_samples=int(n),
                first_sample_number=first_sample_number(ap_dir),
                bad_channel_orig_index=ps.get("bad_channel_orig_index"),
                runtime_min=round(runtime, 2), gpu=torch.cuda.get_device_name(0),
                python=platform.python_version(), versions=_versions(), **summary)
    write_json(ap_dir / "run_meta.json", meta)
    return meta, problems
