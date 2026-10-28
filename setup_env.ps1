# One-time environment setup on a NEW Windows computer (run in PowerShell inside the Code folder).
# Requires: NVIDIA driver installed, uv installed (https://docs.astral.sh/uv/), internet access.
# Uses uv's own Python (NOT Anaconda): Anaconda's old vcruntime140.dll breaks torch (WinError 1114).

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

uv python install 3.11
uv venv .venv --python 3.11 --python-preference only-managed

if (Test-Path "requirements_lock_analysis.txt") {
    # exact versions that were validated before (torch CUDA wheels come from the PyTorch index)
    uv pip install -r requirements_lock_analysis.txt --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
} else {
    uv pip install torch --index-url https://download.pytorch.org/whl/cu128
    uv pip install kilosort "spikeinterface[full]"
}

uv tool install --python 3.12 --python-preference only-managed phy

.\.venv\Scripts\python.exe pipeline.py --help | Select-Object -First 5
Write-Host "Done. Next: .\.venv\Scripts\python.exe pipeline.py <session folder> --step env"
