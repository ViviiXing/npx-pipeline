r"""Neuropixels 1.0 pipeline: Open Ephys -> SpikeInterface -> Kilosort4 (GPU) -> Phy.

Run from the Code folder (VS Code terminal or the VS Code task "NPX pipeline"):

    .\.venv\Scripts\python.exe pipeline.py "D:\...\RHA1-000385_2026-10-21_14-09-44"
    .\.venv\Scripts\python.exe pipeline.py "<session>" --step drift --probes ProbeB

Steps ('all' runs env -> inspect -> badch -> region -> sort -> drift, i.e. up to finished sorting;
manual curation in Phy is started separately with --step phy):
    status          show what has been done for this session
    env             check Python packages, GPU and the 'phy' command
    inspect         find probes, check every AP stream, save qc_probe.png
    badch           detect bad channels, show figures, ask you to confirm/edit the list
    region          type the brain region of each probe (stored in the state file)
    sort            full Kilosort4 sort; results go straight into each AP folder
    drift           drift figure + per-unit residual-drift check; offers a re-sort if needed
    phy             open Phy for one probe after another (run separately, after sorting)
    check-curation  after Phy: read the curated result back and check it
    waveforms       after curation: waveforms, waveform/quality metrics, unit locations,
                    auto-correlograms, putative cell types, tables and figures (in <AP>/postprocess)
    fix-paths       repair params.py dat_path after a session was moved

Every step checks its own results, writes them to npx_pipeline_log.txt in the session
folder and stops to ask you at the points that need a human decision.
"""
import argparse
import sys
import traceback

from npxpipe import config
from npxpipe.session import Session
from npxpipe.utils import (header, log, ok, warn, fail, ask_yes_no, ask_choice, ask_text,
                           parse_index_list, compress_ranges, open_file, set_log_file)

STEPS = ["status", "env", "inspect", "badch", "region", "sort", "drift", "phy",
         "check-curation", "waveforms", "fix-paths"]
ALL_SEQUENCE = ["env", "inspect", "badch", "region", "sort", "drift"]


class StopPipeline(Exception):
    """Raised when a check fails or the user chooses to stop."""


def stop_if(problems, what):
    """Ask whether to stop when a step reported problems."""
    if problems:
        fail(f"{what}: {len(problems)} problem(s): " + "; ".join(problems))
        if not ask_yes_no("Continue anyway?", default=False):
            raise StopPipeline(what)


# ============================================================================ steps
def step_status(s, probes):
    header("STATUS")
    log(f"  session: {s.root}")
    log(f"  session steps done: {list(s.state.get('done', {}))}")
    for p in s.probes():
        ps = s.state["probes"][p]
        log(f"  {p}: region={ps.get('region', '?')} | channels sorted={ps.get('n_channels_sorted', '?')} | "
            f"steps done={list(ps.get('done', {}))}")
    if not s.state["probes"]:
        log("  no probes known yet - start with the 'inspect' step (or 'all').")


def step_env(s, probes):
    header("ENV: Python packages, GPU, Phy")
    import platform
    import shutil
    from importlib.metadata import version
    problems = []
    log(f"  Python {platform.python_version()} at {sys.executable}")
    if "anaconda" in sys.base_prefix.lower() or "conda" in sys.base_prefix.lower():
        warn("This Python comes from Anaconda; its old DLLs broke torch before. Use the uv-managed .venv.")
    for pkg in ["torch", "kilosort", "spikeinterface", "probeinterface", "neo", "numpy", "scipy", "pandas", "matplotlib"]:
        try:
            log(f"  {pkg:15s} {version(pkg)}")
        except Exception:
            fail(f"{pkg} not installed"); problems.append(f"{pkg} missing")
    import torch
    if torch.cuda.is_available():
        a = torch.rand(2000, 2000, device="cuda")
        _ = (a @ a).sum().item()
        ok(f"CUDA works on {torch.cuda.get_device_name(0)} (torch CUDA build {torch.version.cuda})")
    else:
        fail("CUDA not available"); problems.append("no CUDA")
    if shutil.which("phy"):
        ok(f"phy found: {shutil.which('phy')}")
    else:
        warn("'phy' command not found (needed only for manual curation)")
    from npxpipe.curation import long_paths_enabled
    lp = long_paths_enabled()
    if lp is True:
        ok("Windows long paths enabled (needed by Phy's cache inside deep Open Ephys folders)")
    elif lp is False:
        warn("Windows long paths are DISABLED: Phy views stay empty for deep session folders. "
             "Fix once (admin PowerShell): see the manual, section 'Windows long paths'.")
    free = shutil.disk_usage(s.root).free / 1e9
    log(f"  free space on the session drive: {free:.0f} GB")
    stop_if(problems, "environment check")
    s.mark(None, "env")


def step_inspect(s, probes):
    header("INSPECT: find probes and check the recorded data")
    from npxpipe.qc import inspect_probe
    info = s.discover()
    log("  streams in this session:")
    for n in info["streams"]:
        log(f"    {n}")
    if info["n_blocks"] != 1 or info["n_segments"] != 1:
        fail(f"{info['n_blocks']} blocks / {info['n_segments']} segments: recording was stopped and "
             "restarted. This pipeline handles single-segment sessions only.")
        raise StopPipeline("multi-segment session")
    found = info["probes"]
    if not found:
        raise StopPipeline("no Neuropixels AP streams found")
    log(f"  probes found: {sorted(found)}")
    for p, d in found.items():
        if d["ap_dir"] is None:
            raise StopPipeline(f"{p}: AP folder not found uniquely ({d['n_ap_dirs_found']} matches)")
        ps = s.probe_state(p)
        ps.update(stream=d["stream"], ap_dir=d["ap_dir"])
    s.save()

    all_problems = []
    for p in s.probes(probes):
        log(f"\n  --- {p} ---")
        summary, problems = inspect_probe(s, p)
        s.probe_state(p)["inspect"] = summary
        all_problems += [f"{p}: {x}" for x in problems]
        if not problems:
            s.mark(p, "inspect")
    stop_if(all_problems, "inspection")


def step_badch(s, probes):
    header("BADCH: bad-channel detection (you confirm the final list)")
    from npxpipe.qc import detect_bad_channels, save_channel_decision
    for p in s.probes(probes):
        ap = s.ap_dir(p)
        log(f"\n  --- {p} ({s.probe_state(p).get('region', 'region not set')}) ---")
        table, proposed = detect_bad_channels(s, p)
        log(f"  proposed removal ({len(proposed)} channels): {compress_ranges(proposed)}")
        for i, why in proposed.items():
            r = table.loc[table.orig_index == i].iloc[0]
            log(f"    ch {i:3d}  y={r.y_um:6.0f} um  rms={r.rms_uV:6.2f} uV  rate={r.crossing_rate_Hz:5.1f} Hz  {why}")
        log(f"  figures: {ap / 'qc_bad_channels_depth.png'}\n           {ap / 'qc_traces_middle.png'}")
        open_file(ap / "qc_bad_channels_depth.png")
        open_file(ap / "qc_traces_middle.png")

        remove = dict(proposed)
        while True:
            c = ask_choice(f"{p}: remove these {len(remove)} channels ({compress_ranges(remove)})?",
                           {"y": "accept", "e": "edit the list", "q": "stop the pipeline"})
            if c == "q":
                raise StopPipeline("stopped at bad-channel confirmation")
            if c == "y":
                break
            add = parse_index_list(ask_text("  channel indices to ADD (e.g. 10,20-25; Enter = none)", ""))
            keep = parse_index_list(ask_text("  channel indices to KEEP, i.e. take out of the list (Enter = none)", ""))
            for i in add:
                remove[i] = "added manually"
            for i in keep:
                remove.pop(i, None)
        n_sorted = save_channel_decision(s, p, table, remove)
        ok(f"{p}: {len(remove)} channels removed, {n_sorted} channels go into sorting "
           f"(saved in the state file and qc_channel_table.csv)")
        s.mark(p, "badch", removed=sorted(remove))


def step_region(s, probes):
    header("REGION: brain region of each probe")
    for p in s.probes(probes):
        ps = s.probe_state(p)
        serial = ps.get("inspect", {}).get("serial_number", "?")
        ps["region"] = ask_text(f"  {p} (serial {serial}) region", ps.get("region", ""))
    s.save()
    s.mark(None, "region")



def _sort_one(s, p, nblocks):
    from npxpipe.sorting import run_full, existing_results, archive_results
    ap = s.ap_dir(p)
    old = existing_results(ap)
    if old:
        warn(f"{p}: the AP folder already contains sorting results ({len(old)} items, e.g. {old[:4]}).")
        log("  They will be moved to _previous_runs/ (recording.dat is deleted; it is regenerated).")
        if not ask_yes_no(f"  Archive the old results of {p} and sort again?", default=False):
            log(f"  {p}: kept the existing results, not sorted again")
            return None
        dest = archive_results(ap, f"before_nblocks{nblocks}")
        ok(f"old results moved to {dest}")
    meta, problems = run_full(s, p, nblocks)
    stop_if(problems, f"{p} sorting")
    s.probe_state(p)["sort"] = {k: meta[k] for k in ("nblocks", "n_units", "n_spikes", "KSLabel", "runtime_min")}
    s.mark(p, "sort", nblocks=nblocks, n_units=meta["n_units"])
    return meta


def step_sort(s, probes):
    header("SORT: Kilosort4 on the whole recording (results go into each AP folder)")
    todo = s.probes(probes)
    for p in todo:
        if "bad_channel_ids" not in s.probe_state(p):
            raise StopPipeline(f"{p}: bad channels not confirmed - run 'badch' first")
    est = sum(s.probe_state(p)["inspect"]["duration_s"] for p in todo) / 60 * 0.8
    log(f"  probes: {todo} | rough time: {est:.0f}-{est*1.5:.0f} min in total")
    if not ask_yes_no("Start sorting now?", default=True):
        raise StopPipeline("sorting not started")
    for p in todo:
        log(f"\n  --- {p} ({s.probe_state(p).get('region', '')}) ---")
        _sort_one(s, p, config.DEFAULT_NBLOCKS)


def step_drift(s, probes):
    header("DRIFT: drift figure and residual-drift check")
    from npxpipe.drift import plot_drift, check_drift
    for p in s.probes(probes):
        ap = s.ap_dir(p)
        log(f"\n  --- {p} ({s.probe_state(p).get('region', '')}) ---")
        png = plot_drift(ap, f"{p} ({s.probe_state(p).get('region', '')})")
        res = check_drift(ap)
        open_file(png)
        s.probe_state(p)["drift"] = {k: res.get(k) for k in ("verdict", "unit_residual_median_um",
                                                             "max_bin_residual_um", "nonrigid_um")}
        s.mark(p, "drift", verdict=res["verdict"])
        current = s.probe_state(p).get("sort", {}).get("nblocks", config.DEFAULT_NBLOCKS)
        if res["need_more_blocks"] is None or res["need_more_blocks"]:
            log(f"  look at {png.name}: are the horizontal bands tilted or bent?")
            if current < config.ALT_NBLOCKS and ask_yes_no(
                    f"  Re-sort {p} with nblocks={config.ALT_NBLOCKS}? (old result goes to _previous_runs)", default=False):
                _sort_one(s, p, config.ALT_NBLOCKS)
                plot_drift(ap, f"{p} nblocks={config.ALT_NBLOCKS}")
                check_drift(ap)
        else:
            ok(f"{p}: drift is small, no re-sort needed")


def step_phy(s, probes):
    header("PHY: manual curation")
    from npxpipe.curation import launch_phy
    log("  Phy edits the files in the AP folder directly; automatic labels are kept in ks_auto_backup/.")
    log("  Shortcuts: G merge | K split (Ctrl+click polygon, Ctrl+right-click clears) | "
        "Alt+G/M/N good/mua/noise | Ctrl+S save | Ctrl+Z undo")
    for p in s.probes(probes):
        if "sort" not in s.probe_state(p).get("done", {}):
            warn(f"{p}: not sorted yet - skipped")
            continue
        if ask_yes_no(f"Open Phy for {p} ({s.probe_state(p).get('region', '')}) now?", default=True):
            if not launch_phy(s.ap_dir(p)):
                raise StopPipeline("Phy could not be started safely (see message above)")
            if ask_yes_no(f"  Check the curated result of {p} now?", default=True):
                _check_curation_one(s, p)


def _check_curation_one(s, p):
    from npxpipe.curation import check_curation
    log(f"\n  --- {p} ---")
    summary, problems = check_curation(s.ap_dir(p))
    s.probe_state(p)["curation"] = summary
    if not problems and summary["n_unlabeled"] == 0:
        s.mark(p, "curation_checked", **summary)
        ok(f"{p}: curation read back correctly and complete")
    elif not problems:
        warn(f"{p}: read-back OK, but {summary['n_unlabeled']} clusters still unlabeled")
        s.save()
    else:
        fail(f"{p}: {problems}")
        s.save()


def step_check_curation(s, probes):
    header("CHECK-CURATION: read Phy results back")
    for p in s.probes(probes):
        _check_curation_one(s, p)


def step_waveforms(s, probes):
    header("WAVEFORMS: waveform extraction and unit characterisation (after curation)")
    from npxpipe.postprocess import run_waveforms
    log(f"  units analysed: labels {config.UNIT_LABELS} (change UNIT_LABELS in config.py)")
    for p in s.probes(probes):
        ps = s.probe_state(p)
        log(f"\n  --- {p} ({ps.get('region', '')}) ---")
        if "sort" not in ps.get("done", {}):
            warn(f"{p}: not sorted yet - skipped")
            continue
        curated = "curation_checked" in ps.get("done", {})
        if not curated:
            warn(f"{p}: manual curation not finished/checked (no 'curation_checked' in status).")
            log("  Final results must use curated units. A PREVIEW with Kilosort's automatic labels can be")
            log("  made now (written to postprocess_preview/, marked as not curated).")
            if not ask_yes_no(f"  Make a preview for {p} with automatic labels?", default=False):
                log(f"  {p}: skipped - curate first: --step phy --probes {p}")
                continue
        summary, problems = run_waveforms(s, p, curated)
        stop_if(problems, f"{p} waveforms")
        if summary:
            ps["waveforms"] = summary
            if curated:
                s.mark(p, "waveforms", units=summary["units"])
            else:
                s.save()
            out = s.ap_dir(p) / ("postprocess" if curated else "postprocess_preview")
            ok(f"{p}: results in {out}")
            open_file(out / "summary_units.png")


def step_fix_paths(s, probes):
    header("FIX-PATHS: repair params.py dat_path")
    from npxpipe.curation import fix_dat_path
    for p in s.probes(probes):
        if (s.ap_dir(p) / "params.py").exists():
            fix_dat_path(s.ap_dir(p))
        else:
            warn(f"{p}: no params.py (not sorted yet)")


STEP_FUNCS = {"status": step_status, "env": step_env, "inspect": step_inspect, "badch": step_badch,
              "region": step_region, "sort": step_sort, "drift": step_drift,
              "phy": step_phy, "check-curation": step_check_curation, "waveforms": step_waveforms,
              "fix-paths": step_fix_paths}


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", nargs="?", help="session folder (contains 'Record Node ...')")
    ap.add_argument("--step", default="all", choices=["all"] + STEPS)
    ap.add_argument("--probes", default="", help="comma-separated probe names, e.g. ProbeA,ProbeB (default: all)")
    ap.add_argument("--redo", action="store_true", help="in 'all', also repeat steps already done")
    args = ap.parse_args()

    root = args.session or input("Session folder (paste the path, quotes are fine): ").strip().strip('"')
    s = Session(root)
    set_log_file(s.log_path)
    probes = [x.strip() for x in args.probes.split(",") if x.strip()] or None
    log(f"\n##### pipeline started | session {s.root} | step {args.step} | probes {probes or 'all'}")

    try:
        if args.step != "all":
            STEP_FUNCS[args.step](s, probes)
        else:
            for step in ALL_SEQUENCE:
                done = s.is_done(None, step) if step in ("env", "region") else \
                    (bool(s.state["probes"]) and all(s.is_done(p, step) for p in s.probes(probes)))
                if done and not args.redo and step not in ("phy",):
                    log(f"\n  [skip] '{step}' already done (use --redo or --step {step} to repeat)")
                    continue
                STEP_FUNCS[step](s, probes)
            header("SORTING FINISHED")
            step_status(s, probes)
            log("\n  Next: manual curation in Phy, one probe at a time, e.g.")
            log(f'    .\\.venv\\Scripts\\python.exe pipeline.py "{s.root}" --step phy --probes {s.probes(probes)[0]}')
    except StopPipeline as e:
        fail(f"pipeline stopped: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        fail("interrupted by the user (Ctrl+C)")
        sys.exit(1)
    except Exception:
        log(traceback.format_exc())
        fail("unexpected error - the full message is above and in npx_pipeline_log.txt")
        sys.exit(1)
    log(f"##### pipeline ended normally")


if __name__ == "__main__":
    main()
