"""Step 'phy', 'check-curation' and 'fix-paths': everything around manual curation in Phy.

Phy edits files in place in the AP folder (spike_clusters.npy, cluster_group.tsv, ...).
The automatic Kilosort result stays available in two places:
  spike_templates.npy  (never changed by Phy)  and  ks_auto_backup/ (copied after sorting).
"""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .session import read_params_py
from .utils import log, ok, warn, fail


def fix_dat_path(ap_dir):
    """Point params.py to the data file that actually lies in this AP folder.

    Needed when a session was moved to another drive/folder (Phy stores an absolute path).
    Prefers recording.dat (our sorting input); falls back to continuous.dat (checkdata style).
    """
    ap_dir = Path(ap_dir)
    prm = read_params_py(ap_dir)
    old = prm["dat_file"]
    old_name = Path(old).name if old else "recording.dat"
    target = ap_dir / old_name
    if not target.exists():
        target = ap_dir / "recording.dat" if (ap_dir / "recording.dat").exists() else ap_dir / "continuous.dat"
    if old and Path(old).resolve() == target.resolve() and target.exists():
        ok(f"params.py already points to {target.name}")
        return False
    p = ap_dir / "params.py"
    shutil.copy2(p, ap_dir / "params.py.bak")
    lines = [f"dat_path = r'{target.as_posix()}'" if l.startswith("dat_path") else l
             for l in p.read_text().splitlines()]
    p.write_text("\n".join(lines) + "\n")
    ok(f"params.py dat_path: {old}  ->  {target.as_posix()}  (backup: params.py.bak)")
    return True


# Phy keeps a cache in <AP>/.phy/ with sub-paths of up to ~140 characters. Together with the
# deep Open Ephys folder this exceeds Windows' classic 260-character path limit, and then the
# waveform / amplitude / correlogram views stay EMPTY (joblib cannot write its cache files).
PHY_CACHE_EXTRA_CHARS = 150
WINDOWS_MAX_PATH = 260


def long_paths_enabled():
    """True/False if Windows long paths are enabled; None when not on Windows or unknown."""
    if not sys.platform.startswith("win"):
        return None
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem")
        val, _ = winreg.QueryValueEx(key, "LongPathsEnabled")
        return bool(val)
    except OSError:
        return False


def launch_phy(ap_dir):
    """Open Phy on this AP folder and wait until the Phy window is closed.

    Returns False (without starting Phy) when Phy would fail because of the path-length limit.
    """
    ap_dir = Path(ap_dir)
    phy = shutil.which("phy")
    if phy is None:
        fail("'phy' command not found. Install with: uv tool install --python 3.12 phy")
        return False
    too_long = len(str(ap_dir)) + PHY_CACHE_EXTRA_CHARS > WINDOWS_MAX_PATH
    if too_long and long_paths_enabled() is False:
        fail(f"The AP folder path is {len(str(ap_dir))} characters long; Phy's cache would exceed the "
             f"Windows {WINDOWS_MAX_PATH}-character limit and its views would stay empty.")
        log("  Enable Windows long paths once (PowerShell as administrator), then restart the computer:")
        log('    New-ItemProperty -Path "HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem" '
            '-Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force')
        return False
    # a cache written while the limit was hit is incomplete -> remove it, Phy rebuilds it
    cache = ap_dir / ".phy"
    if cache.exists() and (ap_dir / ".phy_cache_ok").exists() is False:
        shutil.rmtree(cache, ignore_errors=True)
        log("  removed old/incomplete Phy cache (.phy); Phy rebuilds it")
    fix_dat_path(ap_dir)
    log(f"  starting Phy in {ap_dir} (close the Phy window to continue) ...")
    subprocess.run([phy, "template-gui", "params.py"], cwd=str(ap_dir))
    (ap_dir / ".phy_cache_ok").write_text("Phy cache created with long paths available\n")
    return True


def check_curation(ap_dir):
    """Compare Phy's saved files with the automatic result and with SpikeInterface's read-back.

    Returns (summary, problems).
    """
    import spikeinterface.extractors as se

    ap_dir = Path(ap_dir)
    problems = []
    auto = np.load(ap_dir / "spike_templates.npy").ravel()
    now_ = np.load(ap_dir / "spike_clusters.npy").ravel()
    grp = pd.read_csv(ap_dir / "cluster_group.tsv", sep="\t")
    ids_auto, ids_now = set(np.unique(auto).tolist()), set(np.unique(now_).tolist())

    log(f"  spikes: automatic {len(auto):,} | now {len(now_):,}")
    log(f"  clusters: automatic {len(ids_auto)} | now {len(ids_now)}")
    log(f"  removed by merge/split: {len(ids_auto - ids_now)} | new: {len(ids_now - ids_auto)}")
    if len(auto) != len(now_):
        problems.append("spike count changed")

    s = se.read_phy(ap_dir, exclude_cluster_groups=None, load_all_cluster_properties=True)
    key = "quality" if "quality" in s.get_property_keys() else "group"
    labels = pd.Series(s.get_property(key), index=s.get_unit_ids()).astype(object)
    labels = labels.where(labels.notna(), "unlabeled")
    counts = labels.value_counts().to_dict()
    log(f"  manual labels ({key}): {counts}")

    count_phy = pd.Series(now_).value_counts()
    mism = [int(u) for u in s.get_unit_ids() if len(s.get_unit_spike_train(u)) != count_phy.get(u, -1)]
    lab_phy = grp.set_index("cluster_id")["group"]
    lab_mism = [int(u) for u in s.get_unit_ids() if u in lab_phy.index and labels[u] != lab_phy[u]]
    (ok if not mism else fail)(f"spike counts per unit match Phy" if not mism else f"spike count mismatch: {mism[:10]}")
    (ok if not lab_mism else fail)("labels match Phy" if not lab_mism else f"label mismatch: {lab_mism[:10]}")
    problems += (["spike count mismatch"] if mism else []) + (["label mismatch"] if lab_mism else [])

    unlabeled = [int(u) for u in labels.index if labels[u] in ("unlabeled", "unsorted", "")]
    if unlabeled:
        warn(f"{len(unlabeled)} clusters have no manual label (e.g. {unlabeled[:10]}) - curation not finished")
    else:
        ok("every cluster has a manual label")
    summary = dict(n_units=int(s.get_num_units()), labels=counts, n_unlabeled=len(unlabeled),
                   n_removed=len(ids_auto - ids_now), n_new=len(ids_now - ids_auto))
    return summary, problems
