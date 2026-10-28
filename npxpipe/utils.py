"""Small helpers shared by all steps: logging, console prompts, file helpers."""
import json
import os
import sys
import datetime as _dt
from pathlib import Path

_LOG_PATH = None  # set by set_log_file(); every log() call is also appended there


def set_log_file(path):
    """Remember where log() should append its messages (the session log file)."""
    global _LOG_PATH
    _LOG_PATH = Path(path)


def now():
    """Current local time as a readable string, e.g. 2026-10-25 14:03:12."""
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def stamp():
    """Compact timestamp for folder names, e.g. 20261025_140312."""
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def log(msg=""):
    """Print a message and append it (with a timestamp) to the session log file."""
    print(msg, flush=True)
    if _LOG_PATH is not None:
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            for line in str(msg).splitlines() or [""]:
                f.write(f"[{now()}] {line}\n")


def header(title):
    """Print a clearly visible section header."""
    bar = "=" * 72
    log(f"\n{bar}\n  {title}\n{bar}")


def ok(msg):
    log(f"  [OK]   {msg}")


def warn(msg):
    log(f"  [WARN] {msg}")


def fail(msg):
    log(f"  [FAIL] {msg}")


def ask_yes_no(question, default=None):
    """Ask a y/n question in the terminal and return True/False.

    default=None means the user must type y or n explicitly (Enter alone is not accepted).
    """
    suffix = {True: " [Y/n] ", False: " [y/N] ", None: " [y/n] "}[default]
    while True:
        ans = input(question + suffix).strip().lower()
        if not ans and default is not None:
            log(f"  > {question} -> {'yes' if default else 'no'} (default)")
            return default
        if ans in ("y", "yes"):
            log(f"  > {question} -> yes")
            return True
        if ans in ("n", "no"):
            log(f"  > {question} -> no")
            return False
        print("  Please type y or n.")


def ask_choice(question, choices):
    """Ask the user to pick one of several single-letter choices, e.g. {'y': ..., 'e': ...}."""
    keys = "/".join(choices)
    while True:
        print(question)
        for k, v in choices.items():
            print(f"    {k} = {v}")
        ans = input(f"  Your choice [{keys}]: ").strip().lower()
        if ans in choices:
            log(f"  > {question} -> {ans} ({choices[ans]})")
            return ans
        print("  Not a valid choice.")


def ask_text(question, default=""):
    """Ask for free text; Enter keeps the default."""
    ans = input(f"{question} [{default}]: ").strip()
    ans = ans or default
    log(f"  > {question} -> {ans}")
    return ans


def parse_index_list(text):
    """Turn '10, 20-23,40' into [10, 20, 21, 22, 23, 40]. Empty text gives []."""
    out = set()
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def compress_ranges(values):
    """Turn [1,2,3,7,9,10] into '1-3, 7, 9-10' for compact printing."""
    values = sorted(int(v) for v in values)
    if not values:
        return "(none)"
    parts, start, prev = [], values[0], values[0]
    for v in values[1:] + [None]:
        if v is not None and v == prev + 1:
            prev = v
            continue
        parts.append(f"{start}" if start == prev else f"{start}-{prev}")
        if v is not None:
            start = prev = v
    return ", ".join(parts)


def open_file(path):
    """Open a file (e.g. a PNG figure) with the default Windows program, if possible."""
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # noqa: S606 - opening a local figure for the user
        else:
            print(f"  (open manually: {path})")
    except Exception as e:  # never stop the pipeline because a viewer failed
        warn(f"Could not open {path}: {e}")


def write_json(path, obj):
    """Write a dict to a UTF-8 JSON file (readable, keeps Chinese/µ characters)."""
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def read_json(path, default=None):
    """Read a JSON file; return `default` if the file does not exist."""
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def gb(nbytes):
    """Bytes -> gigabytes (decimal), rounded to 1 decimal."""
    return round(nbytes / 1e9, 1)


def setup_plot_fonts(plt):
    """Use fonts that exist on Windows; call right before drawing each figure."""
    plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
