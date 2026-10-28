"""Step 'inspect' and step 'badch': check what was recorded and find bad channels.

Nothing here modifies raw data. Figures and tables are written into the AP folder
with the prefix 'qc_'.
"""
import numpy as np
import pandas as pd

from . import config
from .utils import log, ok, fail, write_json, setup_plot_fonts


def load_raw(session, probe):
    """Open the raw AP stream of one probe as a SpikeInterface recording (lazy, no copy)."""
    import spikeinterface.extractors as se
    return se.read_openephys(session.root, stream_name=session.stream(probe))


# ============================================================================ inspect
def inspect_probe(session, probe):
    """Check basic properties of one probe's AP stream and save a probe figure.

    Returns (summary_dict, list_of_problems). An empty problem list means all checks passed.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from probeinterface import write_probeinterface
    from probeinterface.plotting import plot_probe

    ap_dir = session.ap_dir(probe)
    rec = load_raw(session, probe)
    problems = []

    fs = rec.get_sampling_frequency()
    n = rec.get_num_samples()
    pr = rec.get_probe()
    pos = pr.contact_positions
    keys = rec.get_property_keys()
    summary = dict(
        probe=probe, stream=session.stream(probe), ap_dir=str(ap_dir),
        fs=fs, n_channels=rec.get_num_channels(), n_samples=int(n), duration_s=round(n / fs, 2),
        dtype=str(rec.get_dtype()), n_segments=rec.get_num_segments(),
        gain_to_uV=sorted(set(np.round(rec.get_channel_gains(), 4).tolist())),
        probe_model=pr.model_name, serial_number=pr.serial_number,
        y_range_um=[float(pos[:, 1].min()), float(pos[:, 1].max())],
        has_inter_sample_shift="inter_sample_shift" in keys,
    )

    # ---- checks: each failed check is added to `problems`
    def check(cond, good_msg, bad_msg):
        (ok if cond else fail)(good_msg if cond else bad_msg)
        if not cond:
            problems.append(bad_msg)

    check(fs == config.EXPECTED_FS, f"sampling rate {fs:.0f} Hz", f"unexpected sampling rate {fs}")
    check(summary["n_channels"] == config.EXPECTED_N_CHANNELS, f"{summary['n_channels']} channels",
          f"unexpected channel count {summary['n_channels']}")
    check(summary["dtype"] == "int16", "dtype int16", f"unexpected dtype {summary['dtype']}")
    check(summary["n_segments"] == 1, "1 segment", f"{summary['n_segments']} segments (not supported)")
    check(summary["has_inter_sample_shift"], "inter_sample_shift present (phase_shift possible)",
          "inter_sample_shift missing (phase_shift impossible)")

    # sample numbers must increase by exactly 1 (no dropped / repeated samples)
    sn = np.load(ap_dir / "sample_numbers.npy", mmap_mode="r")
    d = np.diff(np.asarray(sn))
    n_gaps = int(np.sum(d != 1))
    summary["first_sample_number"] = int(sn[0])
    summary["sample_number_gaps"] = n_gaps
    check(n_gaps == 0, f"sample numbers continuous ({len(sn):,} samples, first {int(sn[0])})",
          f"{n_gaps} gaps/repeats in sample_numbers.npy")

    # continuous.dat size must match samples x channels x 2 bytes
    size = (ap_dir / "continuous.dat").stat().st_size
    expect = n * summary["n_channels"] * 2
    check(size == expect, f"continuous.dat size matches ({size/1e9:.1f} GB)",
          f"continuous.dat size {size} != expected {expect}")

    log(f"  duration {summary['duration_s']:.1f} s | model {pr.model_name} | serial {pr.serial_number} | "
        f"y {summary['y_range_um'][0]:.0f}-{summary['y_range_um'][1]:.0f} um | gain {summary['gain_to_uV']} uV/bit")

    # ---- probe figure and probe json
    write_probeinterface(ap_dir / "qc_probe.json", pr)
    setup_plot_fonts(plt)
    fig, ax = plt.subplots(figsize=(4, 12))
    plot_probe(pr, ax=ax, with_contact_id=False)
    ax.set_ylim(-100, pos[:, 1].max() + 200)
    ax.set_title(f"{probe} ({pr.serial_number})")
    fig.savefig(ap_dir / "qc_probe.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    write_json(ap_dir / "qc_inspect.json", summary)
    return summary, problems


# ============================================================================ bad channels
def _rms_and_activity(rec, gains, starts, fs, dur_s=1.0):
    """Per-channel RMS (uV) and negative threshold-crossing rate (Hz) on 1-s snippets.

    Each snippet is median-referenced across channels (simple CAR) first.
    The threshold is 5 x robust noise (MAD / 0.6745) of each channel.
    Returns the median over snippets for both numbers.
    """
    n = int(dur_s * fs)
    rms_all, rate_all = [], []
    for s in starts:
        tr = rec.get_traces(start_frame=int(s), end_frame=int(s) + n).astype(np.float32) * gains
        tr = tr - np.median(tr, axis=1, keepdims=True)
        mad = np.median(np.abs(tr - np.median(tr, axis=0)), axis=0) / 0.6745
        rms_all.append(np.sqrt(np.mean(tr ** 2, axis=0)))
        below = tr < -5 * mad
        rate_all.append(np.sum(below[1:] & ~below[:-1], axis=0) / dur_s)
    return np.median(rms_all, axis=0), np.median(rate_all, axis=0)


def detect_bad_channels(session, probe):
    """Run bad-channel detection on the whole recording and on 3 time windows.

    Writes qc_bad_channel_detection.csv and two figures, and returns
    (table, proposed_removal) where proposed_removal is {orig_index: reason}.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import spikeinterface.preprocessing as spre

    ap_dir = session.ap_dir(probe)
    rec = load_raw(session, probe)
    fs = rec.get_sampling_frequency()
    # check chain: high-pass -> phase_shift. Used ONLY here, never for sorting.
    rec_chk = spre.phase_shift(spre.highpass_filter(rec, freq_min=config.HIGHPASS_HZ))
    ntot = rec.get_num_samples()
    ids = rec.get_channel_ids()
    pos = rec.get_channel_locations()
    gains = rec.get_channel_gains().astype(np.float32)

    log("  detection on the whole recording ...")
    _, lab_all = spre.detect_bad_channels(rec_chk, method=config.BAD_CHANNEL_METHOD, seed=0)
    table = pd.DataFrame({"channel_id": ids, "orig_index": np.arange(len(ids)),
                          "x_um": pos[:, 0], "y_um": pos[:, 1], "label_all": lab_all})

    win = int(min(config.QC_WINDOW_S * fs, ntot // 4))
    margin = int(min(30 * fs, ntot // 20))
    windows = {"start": margin, "middle": ntot // 2 - win // 2, "end": ntot - win - margin}
    for name, s0 in windows.items():
        log(f"  detection on the {name} window ...")
        sub = rec_chk.frame_slice(start_frame=s0, end_frame=s0 + win)
        _, lab = spre.detect_bad_channels(sub, method=config.BAD_CHANNEL_METHOD, seed=0, num_random_chunks=50)
        table[f"label_{name}"] = lab

    log("  RMS and activity versus depth ...")
    starts = np.linspace(margin, ntot - margin - fs, config.QC_SNIPPETS).astype(int)
    rms, rate = _rms_and_activity(rec_chk, gains, starts, fs)
    table["rms_uV"] = np.round(rms, 2)
    table["crossing_rate_Hz"] = np.round(rate, 2)

    # ---- proposal: bad in the whole recording, or bad in >= BAD_IN_N_WINDOWS windows
    win_cols = ["label_start", "label_middle", "label_end"]
    proposed = {}
    for _, r in table.iterrows():
        n_bad_win = sum(r[c] != "good" for c in win_cols)
        if r.label_all != "good" or n_bad_win >= config.BAD_IN_N_WINDOWS:
            labels = [r.label_all] + [r[c] for c in win_cols]
            main = max(set(l for l in labels if l != "good"), key=labels.count)
            proposed[int(r.orig_index)] = f"{main} (all={r.label_all}, windows={n_bad_win}/3 bad)"
    table["proposed_remove"] = table.orig_index.isin(proposed)
    table.to_csv(ap_dir / "qc_bad_channel_detection.csv", index=False, encoding="utf-8-sig")

    _plot_depth(plt, table, probe, ap_dir / "qc_bad_channels_depth.png")
    _plot_traces(plt, rec_chk, gains, pos, probe, ap_dir / "qc_traces_middle.png")

    for c in ["label_all"] + win_cols:
        log(f"  {c:13s} {dict(pd.Series(table[c]).value_counts())}")
    return table, proposed


def _plot_depth(plt, table, probe, out_png):
    """Figure: detection labels, RMS and activity along the probe depth."""
    setup_plot_fonts(plt)
    colors = {"good": "tab:green", "noise": "tab:orange", "dead": "black", "out": "tab:blue"}
    c = [colors.get(l, "red") for l in table.label_all]
    fig, axs = plt.subplots(1, 3, figsize=(13, 10), sharey=True)
    for lab, col in colors.items():
        m = table.label_all == lab
        axs[0].scatter(table.x_um[m], table.y_um[m], c=col, s=10, label=f"{lab} ({m.sum()})")
    m = table.proposed_remove
    axs[0].scatter(table.x_um[m], table.y_um[m], facecolors="none", edgecolors="red", s=40,
                   label=f"proposed removal ({m.sum()})")
    axs[0].set_xlabel("x (um)"); axs[0].set_ylabel("distance from tip, y (um)")
    axs[0].legend(loc="upper left", fontsize=8); axs[0].set_title("Detection (whole recording)")
    axs[1].scatter(table.rms_uV, table.y_um, s=6, c=c)
    axs[1].set_xlabel("RMS (uV, 300 Hz HP + CAR)"); axs[1].set_title("Noise vs depth")
    axs[2].scatter(table.crossing_rate_Hz, table.y_um, s=6, c=c)
    axs[2].set_xlabel("threshold crossings (Hz)"); axs[2].set_title("Spiking activity vs depth")
    for ax in axs:
        ax.grid(alpha=0.3)
    fig.suptitle(probe)
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _plot_traces(plt, rec_chk, gains, pos, probe, out_png):
    """Figure: 0.5 s of high-passed, phase-shifted, median-referenced data, sorted by depth."""
    setup_plot_fonts(plt)
    fs = rec_chk.get_sampling_frequency()
    s = rec_chk.get_num_samples() // 2
    tr = rec_chk.get_traces(start_frame=s, end_frame=s + int(fs // 2)).astype(np.float32) * gains
    tr = tr - np.median(tr, axis=1, keepdims=True)
    order = np.argsort(pos[:, 1])
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(tr[:, order].T, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-60, vmax=60,
              extent=[0, 500, pos[order[0], 1], pos[order[-1], 1]])
    ax.set_xlabel("time (ms)"); ax.set_ylabel("distance from tip, y (um)")
    ax.set_title(f"{probe}: middle 0.5 s (300 Hz HP + phase_shift + CAR, +/-60 uV)")
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


def save_channel_decision(session, probe, table, remove):
    """Store the final removal list (decided by the user) in the state and in qc_channel_table.csv.

    `remove` is {orig_index: reason}. The channel table maps original channel index to the
    column index inside recording.dat (= the channel numbering Phy shows).
    """
    ap_dir = session.ap_dir(probe)
    t = table.copy()
    t["removed"] = t.orig_index.isin(remove)
    t["reason"] = t.orig_index.map(remove).fillna("")
    kept_idx = t.index[~t.removed]
    t["sorting_col_index"] = -1
    t.loc[kept_idx, "sorting_col_index"] = np.arange(len(kept_idx))
    t.to_csv(ap_dir / "qc_channel_table.csv", index=False, encoding="utf-8-sig")

    ps = session.probe_state(probe)
    ps["bad_channel_ids"] = t.loc[t.removed, "channel_id"].tolist()
    ps["bad_channel_orig_index"] = sorted(int(i) for i in remove)
    ps["bad_channel_reasons"] = {str(k): v for k, v in remove.items()}
    ps["n_channels_sorted"] = int((~t.removed).sum())
    session.save()
    return ps["n_channels_sorted"]
