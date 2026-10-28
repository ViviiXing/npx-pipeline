"""Step 'waveforms': waveform extraction and unit characterisation AFTER manual curation.

Analysis chain (µV, not whitened, same channels as the sorting input):
    raw AP -> remove the confirmed bad channels -> phase_shift -> band-pass -> median CAR
Spike times come from the CURATED Phy result (spike_clusters.npy + manual labels), so merges
and splits made in Phy are respected.

Everything is written to <AP>/postprocess/ (or <AP>/postprocess_preview/ when run on
uncurated automatic labels):
    sorting_analyzer/      full SpikeInterface SortingAnalyzer (can be re-loaded later)
    units_summary.csv      one row per unit: labels, depth, rates, quality + waveform metrics, type
    units_waveforms.npz    mean waveforms (all channels + main channel), channel positions, etc.
    waveforms_png/         one figure per unit (waveforms + auto-correlogram)
    summary_units.png      depth / firing-rate / waveform-width overview
"""
import shutil

import numpy as np
import pandas as pd

from . import config
from .celltype import trough_to_peak_ms, acg_narrow, fit_acg, classify
from .qc import load_raw
from .utils import log, ok, warn, fail, write_json, setup_plot_fonts, now


# ============================================================================ inputs
def load_analysis_recording(session, probe):
    """Raw AP -> remove bad channels -> phase_shift -> band-pass -> global median reference."""
    import spikeinterface.preprocessing as spre
    ps = session.state["probes"][probe]
    rec = load_raw(session, probe).remove_channels(ps["bad_channel_ids"])
    rec = spre.phase_shift(rec)
    rec = spre.bandpass_filter(rec, freq_min=config.ANALYSIS_FREQ_MIN, freq_max=config.ANALYSIS_FREQ_MAX)
    rec = spre.common_reference(rec, reference="global", operator="median")
    return rec


def load_curated_sorting(ap_dir, curated):
    """Read the Phy folder and keep only the units whose label is in config.UNIT_LABELS.

    curated=True  -> use the manual labels (Phy 'group', SpikeInterface property 'quality')
    curated=False -> use Kilosort's automatic KSLabel (preview only)
    """
    import spikeinterface.extractors as se
    s = se.read_phy(ap_dir, exclude_cluster_groups=None, load_all_cluster_properties=True)
    keys = s.get_property_keys()
    if curated:
        key = "quality" if "quality" in keys else "group"
    else:
        key = "KSLabel"
    labels = pd.Series(s.get_property(key), index=s.get_unit_ids()).astype(str)
    keep = [u for u in s.get_unit_ids() if labels[u] in config.UNIT_LABELS]
    log(f"  labels ({key}): {labels.value_counts().to_dict()} -> keeping {len(keep)} units {config.UNIT_LABELS}")
    return s.select_units(keep), labels, key


# ============================================================================ helpers
def _available(names, getter_paths):
    """Keep only metric names that this SpikeInterface version knows (names change between versions)."""
    import importlib
    known = set()
    for mod, fn in getter_paths:
        try:
            known |= set(getattr(importlib.import_module(mod), fn)())
        except Exception:
            pass
    if not known:
        return list(names)
    missing = [n for n in names if n not in known]
    if missing:
        warn(f"metrics not available in this SpikeInterface version, skipped: {missing}")
    return [n for n in names if n in known]


def _find_col(df, *candidates):
    """Return the first column of df whose name contains one of the candidate strings."""
    for c in candidates:
        for col in df.columns:
            if c in col:
                return col
    return None


def _to_ms(values):
    """Template metrics come in seconds in SpikeInterface; convert to ms (robust to either)."""
    v = np.asarray(values, dtype=float)
    return v * 1000.0 if np.nanmedian(np.abs(v)) < 0.05 else v


def _depth_center_of_mass(templates, main_ch, chan_pos, radius_um=60.0):
    """Unit depth as the amplitude-weighted mean y of the channels within radius_um of the main
    channel (same idea as Kilosort/Phy: centre of mass of the template amplitude)."""
    ptp = np.ptp(templates, axis=1)                                   # (units, channels)
    depth = np.full(len(main_ch), np.nan)
    for i, c in enumerate(main_ch):
        near = np.linalg.norm(chan_pos - chan_pos[c], axis=1) <= radius_um
        w = ptp[i, near] ** 2
        if w.sum() > 0:
            depth[i] = float(np.sum(w * chan_pos[near, 1]) / w.sum())
    return depth


def _read_phy_cluster_info(ap_dir):
    """Phy's own 'ch' and 'depth' per cluster, from cluster_info.tsv (written by Phy on save)."""
    f = ap_dir / "cluster_info.tsv"
    if not f.exists():
        return {}
    t = pd.read_csv(f, sep="\t")
    idc = "cluster_id" if "cluster_id" in t.columns else "id"
    cols = [c for c in ("ch", "depth") if c in t.columns]
    return {int(r[idc]): {c: float(r[c]) for c in cols} for _, r in t.iterrows()}


def _burst_index(sorting, unit_id, fs):
    """Fraction of inter-spike intervals shorter than config.BURST_ISI_MS."""
    st = sorting.get_unit_spike_train(unit_id) / fs
    if len(st) < 3:
        return np.nan
    isi = np.diff(st) * 1000.0
    return float(np.mean(isi < config.BURST_ISI_MS))


NARROW_TTP_MS = getattr(config, "NARROW_TROUGH_TO_PEAK_MS", 0.425)
WIDE_TAU_RISE_MS = getattr(config, "WIDE_INTERNEURON_MIN_TAU_RISE_MS", 6.0)


def _cell_type(region, ttp_ms, tau_rise_ms):
    """Putative type, CellExplorer standard schema (waveform width + ACG tau_rise).
    Firing rate is not used. Only applied to regions listed in config.CELLTYPE_REGIONS."""
    if region not in config.CELLTYPE_REGIONS:
        return "n/a (no rule for this region)"
    return classify(ttp_ms, tau_rise_ms, NARROW_TTP_MS, WIDE_TAU_RISE_MS)


# ============================================================================ main step
def run_waveforms(session, probe, curated):
    """Build the SortingAnalyzer, compute all extensions, write tables and figures.

    Returns (summary_dict, problems).
    """
    import spikeinterface as si

    ap_dir = session.ap_dir(probe)
    ps = session.state["probes"][probe]
    region = ps.get("region", "")
    out = ap_dir / ("postprocess" if curated else "postprocess_preview")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    problems = []
    si.set_global_job_kwargs(n_jobs=config.N_JOBS, chunk_duration=config.CHUNK_DURATION, progress_bar=True)

    # ---- inputs
    sorting, labels, label_key = load_curated_sorting(ap_dir, curated)
    if sorting.get_num_units() == 0:
        fail("no units with the selected labels - nothing to do")
        return {}, ["no units"]
    rec = load_analysis_recording(session, probe)
    fs = rec.get_sampling_frequency()
    if rec.get_num_channels() != ps["n_channels_sorted"]:
        problems.append("analysis recording channel count differs from sorting input")
        fail(f"channel count {rec.get_num_channels()} != {ps['n_channels_sorted']}")

    # ---- SortingAnalyzer (sparse: each unit keeps only channels near it)
    log("  creating SortingAnalyzer ...")
    an = si.create_sorting_analyzer(sorting, rec, format="binary_folder", folder=out / "sorting_analyzer",
                                    sparse=True, overwrite=True)
    steps = [
        ("random_spikes", dict(method="uniform", max_spikes_per_unit=config.MAX_SPIKES_PER_UNIT, seed=0)),
        ("waveforms", dict(ms_before=config.WF_MS_BEFORE, ms_after=config.WF_MS_AFTER)),
        ("templates", dict(operators=["average", "std"])),
        ("noise_levels", {}),
        ("spike_amplitudes", {}),
        ("unit_locations", dict(method="monopolar_triangulation")),
        ("correlograms", dict(window_ms=config.CCG_WINDOW_MS, bin_ms=config.CCG_BIN_MS)),
        ("template_similarity", {}),
        ("principal_components", dict(n_components=3, mode="by_channel_local")),
    ]
    for name, kw in steps:
        log(f"  computing {name} ...")
        an.compute(name, **kw)

    log("  computing template_metrics ...")
    try:
        an.compute("template_metrics", include_multi_channel_metrics=True)
    except Exception as e:  # parameter name differs between SpikeInterface versions
        warn(f"multi-channel template metrics not available ({e}); single-channel metrics only")
        an.compute("template_metrics")
    tmet = an.get_extension("template_metrics").get_data()

    log("  computing quality_metrics ...")
    qm_names = _available(config.QUALITY_METRICS,
                          [("spikeinterface.qualitymetrics", "get_quality_metric_list"),
                           ("spikeinterface.qualitymetrics", "get_quality_pca_metric_list")])
    try:
        an.compute("quality_metrics", metric_names=qm_names)
    except Exception as e:
        warn(f"quality metrics with PCA failed ({e}); retrying without PCA-based metrics")
        qm_names = [n for n in qm_names if n not in ("isolation_distance", "l_ratio", "d_prime")]
        an.compute("quality_metrics", metric_names=qm_names)
    qmet = an.get_extension("quality_metrics").get_data()
    ok(f"SortingAnalyzer saved: {out / 'sorting_analyzer'}")

    # ---- unit table
    unit_ids = list(an.unit_ids)
    templates = an.get_extension("templates").get_templates(operator="average")  # (units, samples, channels)
    tstd = an.get_extension("templates").get_templates(operator="std")
    locs = an.get_extension("unit_locations").get_data()
    main_ch = np.argmax(np.ptp(templates, axis=1), axis=1)                     # column in recording.dat
    chan_table = pd.read_csv(ap_dir / "qc_channel_table.csv")
    col2orig = dict(zip(chan_table.sorting_col_index, chan_table.orig_index))

    # trough-to-peak (CellExplorer definition) from the mean waveform on the main channel;
    # SpikeInterface's own template metrics are still kept in the tm_* columns
    ttp_ms = np.array([trough_to_peak_ms(templates[i][:, main_ch[i]], fs) for i in range(len(unit_ids))])
    hw_col = _find_col(tmet, "half_width")
    hw_ms = _to_ms(tmet[hw_col]) if hw_col else np.full(len(unit_ids), np.nan)
    n_pos = int(np.isnan(ttp_ms).sum())
    if n_pos:
        warn(f"{n_pos} units have a positive-dominant waveform -> trough-to-peak NaN, type 'unclassified'")

    # ACG triple-exponential fit (CellExplorer fit_ACG.m) on ALL spikes of each unit
    log("  fitting autocorrelograms (CellExplorer fit_ACG) ...")
    acg_fits = pd.DataFrame([fit_acg(*acg_narrow(sorting.get_unit_spike_train(u) / fs)) for u in unit_ids])

    df = pd.DataFrame({"unit_id": unit_ids})
    df["manual_label" if curated else "auto_label"] = [labels[u] for u in unit_ids]
    ks = sorting.get_property("KSLabel") if "KSLabel" in sorting.get_property_keys() else None
    ks_map = dict(zip(sorting.get_unit_ids(), ks)) if ks is not None else {}
    df["KSLabel"] = [ks_map.get(u, "") for u in unit_ids]
    # channel numbering: main_channel_col = column in recording.dat = the "ch" numbering used by Phy;
    # main_channel_orig = original Open Ephys channel (differs once bad channels were removed)
    df["main_channel_col"] = main_ch
    df["main_channel_orig"] = [col2orig.get(int(c), -1) for c in main_ch]
    df["depth_um"] = _depth_center_of_mass(templates, main_ch, rec.get_channel_locations())
    df["x_um"] = locs[:, 0]
    df["depth_triang_um"] = locs[:, 1]
    phy = _read_phy_cluster_info(ap_dir)
    df["phy_ch"] = [phy.get(u, {}).get("ch", np.nan) for u in unit_ids]
    df["phy_depth_um"] = [phy.get(u, {}).get("depth", np.nan) for u in unit_ids]
    if phy:
        dch = (df["main_channel_col"] - df["phy_ch"]).abs()
        ddep = (df["depth_um"] - df["phy_depth_um"]).abs()
        log(f"  vs Phy: main channel identical for {int((dch == 0).sum())}/{len(df)} units, "
            f"within 2 channels for {int((dch <= 2).sum())}; depth difference median {np.nanmedian(ddep):.1f} um, "
            f"max {np.nanmax(ddep):.1f} um")
    else:
        warn("cluster_info.tsv not found (Phy writes it when you save) - phy_ch / phy_depth_um left empty")
    df["trough_to_peak_ms"] = np.round(ttp_ms, 4)
    df["half_width_ms"] = np.round(hw_ms, 4)
    df["peak_amplitude_uV"] = [float(np.ptp(templates[i][:, main_ch[i]])) for i in range(len(unit_ids))]
    df["burst_index"] = [_burst_index(sorting, u, fs) for u in unit_ids]
    for col in acg_fits.columns:
        df[col] = np.round(acg_fits[col].to_numpy(dtype=float), 4)
    df = df.merge(qmet, left_on="unit_id", right_index=True, how="left")
    df = df.merge(tmet.add_prefix("tm_"), left_on="unit_id", right_index=True, how="left")
    df["putative_type"] = [_cell_type(region, t, tr) for t, tr in zip(ttp_ms, acg_fits["acg_tau_rise"])]
    df.insert(1, "probe", probe)
    df.insert(2, "region", region)
    df["curated"] = curated
    df.to_csv(out / "units_summary.csv", index=False, encoding="utf-8-sig")

    np.savez_compressed(out / "units_waveforms.npz", unit_ids=np.array(unit_ids),
                        templates_mean=templates.astype(np.float32), templates_std=tstd.astype(np.float32),
                        main_channel_col=main_ch, channel_positions=rec.get_channel_locations(),
                        sampling_frequency=fs, ms_before=config.WF_MS_BEFORE, ms_after=config.WF_MS_AFTER,
                        main_waveform=np.stack([templates[i][:, main_ch[i]] for i in range(len(unit_ids))]))

    _plot_units(an, df, templates, main_ch, out, probe, region)
    _plot_summary(df, out / "summary_units.png", probe, region)

    # ---- checks
    def check(cond, good_msg, bad_msg):
        (ok if cond else fail)(good_msg if cond else bad_msg)
        if not cond:
            problems.append(bad_msg)
    check(len(df) == sorting.get_num_units(), f"{len(df)} units in units_summary.csv", "unit count mismatch")
    check(df["depth_um"].notna().all(), "all units have a location", "some units have no location")
    check(np.isfinite(df["peak_amplitude_uV"]).all() and (df["peak_amplitude_uV"] > 0).all(),
          "all mean waveforms non-empty", "empty mean waveform(s)")
    n_png = len(list((out / "waveforms_png").glob("*.png")))
    check(n_png == len(df), f"{n_png} unit figures", f"only {n_png} unit figures")

    types = df.putative_type.value_counts().to_dict()
    log(f"  putative types: {types}")
    summary = dict(curated=curated, label_source=label_key, units=len(df), putative_types=types,
                   output=str(out), finished=now())
    write_json(out / "postprocess_meta.json", dict(summary, unit_labels=config.UNIT_LABELS,
               analysis_band_hz=[config.ANALYSIS_FREQ_MIN, config.ANALYSIS_FREQ_MAX],
               quality_metrics=qm_names, celltype_rules=dict(
                   schema="CellExplorer standard (Petersen et al. 2021); firing rate not used",
                   regions=config.CELLTYPE_REGIONS, narrow_ttp_ms=NARROW_TTP_MS,
                   wide_in_acg_tau_rise_ms=WIDE_TAU_RISE_MS,
                   ttp_source="mean waveform on main channel, analysis chain, cubic-spline x10",
                   acg="0.5 ms bins, +-50 ms, triple-exponential fit as fit_ACG.m")))
    if not curated:
        (out / "PREVIEW_NOT_CURATED.txt").write_text(
            "These results use Kilosort's automatic labels, not manual curation. Do not use for final analysis.\n")
    return summary, problems


# ============================================================================ figures
def _plot_units(an, df, templates, main_ch, out, probe, region):
    """One figure per unit: 50 single waveforms + mean on the main channel, and the ACG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_plot_fonts(plt)
    d = out / "waveforms_png"
    d.mkdir(exist_ok=True)
    wf_ext = an.get_extension("waveforms")
    ccgs, bins = an.get_extension("correlograms").get_data()
    fs = an.sampling_frequency
    unit_ids = list(an.unit_ids)
    for i, u in enumerate(unit_ids):
        wfs = wf_ext.get_waveforms_one_unit(u)                     # (spikes, samples, sparse channels)
        sp_chans = an.sparsity.unit_id_to_channel_indices[u] if an.sparsity is not None else np.arange(templates.shape[2])
        j = int(np.where(sp_chans == main_ch[i])[0][0]) if main_ch[i] in sp_chans else 0
        t = (np.arange(wfs.shape[1]) / fs - config.WF_MS_BEFORE / 1000) * 1000
        r = df.iloc[i]
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
        for w in wfs[:50, :, j]:
            axs[0].plot(t, w, color="0.75", lw=0.5)
        axs[0].plot(t, templates[i][:, main_ch[i]], color="tab:red", lw=2)
        axs[0].set_xlabel("time (ms)"); axs[0].set_ylabel("uV")
        axs[0].set_title(f"unit {u} | ch {int(r.main_channel_col)} (orig {int(r.main_channel_orig)}) | "
                         f"depth {r.depth_um:.0f} um", fontsize=9)
        acg = ccgs[i, i].astype(float)
        centers = (bins[:-1] + bins[1:]) / 2
        axs[1].bar(centers, acg, width=config.CCG_BIN_MS, color="0.3")
        axs[1].set_xlabel("lag (ms)"); axs[1].set_title(
            f"ACG | {r.get('firing_rate', np.nan):.1f} Hz | TTP {r.trough_to_peak_ms:.3f} ms | "
            f"tau_rise {r.acg_tau_rise:.1f} ms | {r.putative_type}", fontsize=8)
        fig.suptitle(f"{probe} ({region})", fontsize=9)
        fig.tight_layout()
        fig.savefig(d / f"unit_{u:04d}.png", dpi=90)
        plt.close(fig)


def _plot_summary(df, out_png, probe, region):
    """Overview: depth vs firing rate (colour = type), and the trough-to-peak distribution."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_plot_fonts(plt)
    fig, axs = plt.subplots(1, 3, figsize=(18, 7))
    for typ, g in df.groupby("putative_type"):
        axs[0].scatter(g.get("firing_rate", np.nan), g.depth_um, s=18, label=f"{typ} ({len(g)})")
        axs[2].scatter(g.trough_to_peak_ms, g.acg_tau_rise, s=18, label=typ)
    axs[0].set_xscale("log"); axs[0].set_xlabel("firing rate (Hz, log)"); axs[0].set_ylabel("depth from tip (um)")
    axs[0].legend(fontsize=8); axs[0].set_title(f"{probe} ({region}): units by depth")
    axs[1].hist(df.trough_to_peak_ms.dropna(), bins=40, color="0.4")
    axs[1].axvline(NARROW_TTP_MS, color="tab:red", ls="--", label=f"narrow / wide {NARROW_TTP_MS} ms")
    axs[1].set_xlabel("trough-to-peak (ms)"); axs[1].set_ylabel("units"); axs[1].legend(fontsize=8)
    axs[1].set_title("waveform width")
    axs[2].axvline(NARROW_TTP_MS, color="tab:red", ls="--")
    axs[2].axhline(WIDE_TAU_RISE_MS, color="tab:red", ls="--")
    axs[2].set_yscale("log"); axs[2].set_xlabel("trough-to-peak (ms)"); axs[2].set_ylabel("ACG tau_rise (ms, log)")
    axs[2].legend(fontsize=8); axs[2].set_title("CellExplorer classification")
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
