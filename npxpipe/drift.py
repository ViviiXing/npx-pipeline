"""Step 'drift': drift figure and an objective check of residual drift after Kilosort4.

Why per unit? Spikes of ONE unit come from ONE neuron, so the unit's median position can
only change if the probe moved relative to the tissue. Mixing all spikes of a depth range
would also change when different neurons fire (e.g. place cells), which is not drift.
"""
from pathlib import Path

import numpy as np

from . import config
from .utils import log, write_json, setup_plot_fonts


def _load(ap_dir):
    ap_dir = Path(ap_dir)
    fs = float(np.load(ap_dir / "ops.npy", allow_pickle=True).item().get("fs", config.EXPECTED_FS))
    st = np.load(ap_dir / "spike_times.npy").ravel() / fs
    amp = np.load(ap_dir / "amplitudes.npy").ravel()
    y = np.load(ap_dir / "spike_positions.npy")[:, 1]
    # use the AUTOMATIC cluster labels (unaffected by later manual curation)
    auto = ap_dir / "ks_auto_backup" / "spike_clusters.npy"
    cl = np.load(auto if auto.exists() else ap_dir / "spike_templates.npy").ravel()
    ops = np.load(ap_dir / "ops.npy", allow_pickle=True).item()
    return st, amp, y, cl, ops, fs


def plot_drift(ap_dir, title):
    """Save drift_raster.png: spike depth vs time (left) and Kilosort's drift estimate (right)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap_dir = Path(ap_dir)
    st, amp, y, cl, ops, fs = _load(ap_dir)
    dshift = np.asarray(ops["dshift"])
    dshift = dshift[:, None] if dshift.ndim == 1 else dshift

    setup_plot_fonts(plt)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(st), size=min(400_000, len(st)), replace=False)
    a = np.clip(amp[idx], np.percentile(amp, 5), np.percentile(amp, 95))
    fig, axs = plt.subplots(1, 2, figsize=(17, 8), gridspec_kw={"width_ratios": [3.2, 1]})
    axs[0].scatter(st[idx], y[idx], c=a, s=0.3, cmap="gray_r", alpha=0.6, rasterized=True)
    axs[0].set_xlabel("time (s)"); axs[0].set_ylabel("spike position y (um, after correction)")
    axs[0].set_title(f"{title}: spike depth over time (darker = larger amplitude)")
    t = np.arange(dshift.shape[0]) * ops.get("batch_size", 60000) / fs
    for b in range(dshift.shape[1]):
        axs[1].plot(dshift[:, b], t, lw=1, label=f"block {b}")
    axs[1].invert_yaxis(); axs[1].set_xlabel("estimated drift (um)"); axs[1].set_ylabel("time (s)")
    axs[1].set_title("Kilosort drift estimate"); axs[1].legend(fontsize=7)
    out = ap_dir / "drift_raster.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    log(f"  drift estimate {dshift.shape}: min {dshift.min():.2f}, max {dshift.max():.2f}, sd {dshift.std():.2f} um")
    return out


def check_drift(ap_dir):
    """Per-unit residual drift. Returns a result dict incl. 'need_more_blocks' (bool or None)."""
    ap_dir = Path(ap_dir)
    st, amp, y, cl, ops, fs = _load(ap_dir)
    dshift = np.asarray(ops["dshift"])
    rigid_range = float(np.percentile(dshift, 95) - np.percentile(dshift, 5))

    tedges = np.arange(0, st.max() + config.DRIFT_WIN_S, config.DRIFT_WIN_S)
    twin = np.digitize(st, tedges) - 1
    nwin = len(tedges) - 1

    units = []          # (depth, residual, trajectory)
    for u in np.unique(cl):
        m = cl == u
        yu, tu = y[m], twin[m]
        med = np.full(nwin, np.nan)
        for w in range(nwin):
            mm = tu == w
            if mm.sum() >= config.DRIFT_MIN_SPIKES:
                med[w] = np.median(yu[mm])
        if np.mean(~np.isnan(med)) >= config.DRIFT_MIN_COVER:
            traj = med - np.nanmedian(med)
            units.append((float(np.median(yu)), float(np.nanpercentile(traj, 95) - np.nanpercentile(traj, 5)), traj))

    res = dict(rigid_range_um=rigid_range, n_units_total=int(len(np.unique(cl))), n_units_used=len(units),
               tol_um=config.DRIFT_TOL_UM)
    log(f"  Kilosort rigid drift range (5-95 %): {rigid_range:.2f} um")
    log(f"  units firing stably enough to be tracked: {len(units)} of {res['n_units_total']}")
    if len(units) < 5:
        res.update(verdict="too few stable units - judge from drift_raster.png", need_more_blocks=None)
        log("  " + res["verdict"])
        write_json(ap_dir / "drift_check.json", res)
        return res

    depths = np.array([u[0] for u in units])
    resid = np.array([u[1] for u in units])
    T = np.vstack([u[2] for u in units])
    edges = np.arange(np.floor(depths.min() / config.DRIFT_BIN_UM) * config.DRIFT_BIN_UM,
                      depths.max() + config.DRIFT_BIN_UM, config.DRIFT_BIN_UM)
    bins, bin_traj = {}, []
    for d0, d1 in zip(edges[:-1], edges[1:]):
        m = (depths >= d0) & (depths < d1)
        if m.sum() >= 2:
            bins[f"{d0:.0f}-{d1:.0f}"] = {"n_units": int(m.sum()), "median_residual_um": float(np.median(resid[m]))}
            bin_traj.append(np.nanmedian(T[m], axis=0))
    for k, v in bins.items():
        flag = "  <-- above tolerance" if v["median_residual_um"] >= config.DRIFT_TOL_UM else ""
        log(f"    {k:>11s} um  units {v['n_units']:3d}  median residual {v['median_residual_um']:5.2f} um{flag}")

    worst = max(v["median_residual_um"] for v in bins.values())
    B = np.vstack(bin_traj)
    nonrigid = float(np.nanmax(np.nanpercentile(B, 95, axis=0) - np.nanpercentile(B, 5, axis=0))) if len(B) > 1 else 0.0
    need = bool(worst >= config.DRIFT_TOL_UM or nonrigid >= config.DRIFT_TOL_UM)
    res.update(bins=bins, unit_residual_median_um=float(np.median(resid)),
               unit_residual_p90_um=float(np.percentile(resid, 90)),
               max_bin_residual_um=worst, nonrigid_um=nonrigid, need_more_blocks=need,
               verdict=("re-sort with more drift blocks recommended" if need else "stable - keep current nblocks"))
    log(f"  unit residual drift: median {res['unit_residual_median_um']:.2f} um, "
        f"90th pct {res['unit_residual_p90_um']:.2f} um")
    log(f"  worst depth bin {worst:.2f} um | top-vs-bottom difference {nonrigid:.2f} um | "
        f"tolerance {config.DRIFT_TOL_UM} um")
    log(f"  verdict: {res['verdict']}")
    write_json(ap_dir / "drift_check.json", res)
    return res
