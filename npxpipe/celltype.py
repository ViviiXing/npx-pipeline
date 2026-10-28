"""Putative cell type following the CellExplorer 'standard' schema.

Petersen et al., Neuron 2021; https://cellexplorer.org/pipeline/cell-type-classification/
Thresholds: CellExplorer preferences_ProcessCellMetrics.m

    trough-to-peak <= 0.425 ms                          -> narrow_interneuron
    trough-to-peak >  0.425 ms and acg_tau_rise > 6 ms  -> wide_interneuron
    everything else                                     -> pyramidal
Firing rate is NOT part of the rule.

acg_tau_rise comes from the triple-exponential ACG fit of CellExplorer fit_ACG.m:
    narrow ACG, 0.5 ms bins centred on 0, -50..+50 ms, in Hz;
    bins at -0.5, 0, +0.5 ms set to 0; fit on lags 0.5..50 ms with
    max(c*(exp(-(x-f)/a) - d*exp(-(x-f)/b)) + h*exp(-(x-f)/g) + e, 0)
    a=tau_decay b=tau_rise c d e=asymptote f=refrac g=tau_burst h
    start = [20, 1, 30, 2, 0.5, 5, 1.5, 2]
    lower = [1, 0.1, 0, 0, -30, 0, 0.1, 0]
    upper = [500, 50, 500, 15, 50, 20, 5, 100]
"""
import numpy as np

ACG_BIN_MS = 0.5
ACG_HALF_WIDTH_MS = 50.0

FIT_P0 = [20, 1, 30, 2, 0.5, 5, 1.5, 2]
FIT_LB = [1, 0.1, 0, 0, -30, 0, 0.1, 0]
FIT_UB = [500, 50, 500, 15, 50, 20, 5, 100]
FIT_NAMES = ["acg_tau_decay", "acg_tau_rise", "acg_c", "acg_d",
             "acg_asymptote", "acg_refrac", "acg_tau_burst", "acg_h"]


# ---------------------------------------------------------------- waveform width
def trough_to_peak_ms(waveform, fs, upsample=10):
    """Trough -> following peak (ms) of the mean waveform on the main channel.
    Cubic-spline upsampling: at 30 kHz one sample is 0.033 ms, and 0.425 ms lies
    between two samples. Positive-dominant (inverted / axonal) waveforms -> NaN."""
    from scipy.interpolate import CubicSpline
    w = np.asarray(waveform, dtype=float)
    if w.ndim != 1 or w.size < 4 or not np.all(np.isfinite(w)):
        return np.nan
    if abs(w.max()) > abs(w.min()):
        return np.nan
    t = np.arange(w.size) / fs
    tf = np.linspace(t[0], t[-1], (w.size - 1) * upsample + 1)
    wf = CubicSpline(t, w)(tf)
    i_trough = int(np.argmin(wf))
    if i_trough >= wf.size - 1:
        return np.nan
    i_peak = i_trough + int(np.argmax(wf[i_trough:]))
    return float((tf[i_peak] - tf[i_trough]) * 1e3)


# ---------------------------------------------------------------- autocorrelogram
def acg_narrow(spike_times_s):
    """CellExplorer narrow ACG: 201 bins of 0.5 ms centred on 0 (-50..+50 ms), in Hz.
    Vectorised over spike-pair offsets, fast even for units with >100k spikes."""
    st = np.sort(np.asarray(spike_times_s, dtype=float))
    bin_s = ACG_BIN_MS / 1e3
    n_half = int(round(ACG_HALF_WIDTH_MS / ACG_BIN_MS))
    lags_ms = np.arange(-n_half, n_half + 1) * ACG_BIN_MS
    counts = np.zeros(lags_ms.size)
    if st.size < 2:
        return lags_ms, counts
    max_lag = (n_half + 0.5) * bin_s
    k = 1
    while k < st.size:
        d = st[k:] - st[:-k]
        d = d[d < max_lag]
        if d.size == 0:
            break
        idx = np.floor(d / bin_s + 0.5).astype(int)          # bin index on the positive side
        idx = idx[idx <= n_half]
        np.add.at(counts, n_half + idx, 1)
        np.add.at(counts, n_half - idx, 1)
        k += 1
    return lags_ms, counts / (st.size * bin_s)


def _acg_model(x, a, b, c, d, e, f, g, h):
    return np.maximum(c * (np.exp(-(x - f) / a) - d * np.exp(-(x - f) / b))
                      + h * np.exp(-(x - f) / g) + e, 0)


def fit_acg(lags_ms, acg_hz):
    """Triple-exponential fit of the positive half of the narrow ACG (fit_ACG.m)."""
    from scipy.optimize import curve_fit
    acg = np.asarray(acg_hz, dtype=float).copy()
    zero = int(np.argmin(np.abs(lags_ms)))
    acg[zero - 1:zero + 2] = 0
    sel = (lags_ms > 0) & (lags_ms <= ACG_HALF_WIDTH_MS)
    x, y = lags_ms[sel], acg[sel]
    out = {k: np.nan for k in FIT_NAMES}
    out["acg_fit_rsquare"] = np.nan
    if not np.any(y > 0):
        return out
    try:
        p, _ = curve_fit(_acg_model, x, y, p0=FIT_P0, bounds=(FIT_LB, FIT_UB), maxfev=20000)
    except (RuntimeError, ValueError):
        return out
    yhat = _acg_model(x, *p)
    ss_tot = np.sum((y - y.mean()) ** 2)
    out.update(dict(zip(FIT_NAMES, (float(v) for v in p))))
    out["acg_fit_rsquare"] = float(1 - np.sum((y - yhat) ** 2) / ss_tot) if ss_tot > 0 else np.nan
    return out


# ---------------------------------------------------------------- rule
def classify(ttp_ms, tau_rise_ms, narrow_ttp_ms=0.425, wide_tau_rise_ms=6.0):
    """CellExplorer standard schema. No firing-rate criterion."""
    if not np.isfinite(ttp_ms):
        return "unclassified"
    if ttp_ms <= narrow_ttp_ms:
        return "narrow_interneuron"
    if not np.isfinite(tau_rise_ms):
        return "unclassified"            # wide waveform but the ACG fit failed
    if tau_rise_ms > wide_tau_rise_ms:
        return "wide_interneuron"
    return "pyramidal"
