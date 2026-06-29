"""
qus_pipeline.py
===============
Quantitative Ultrasound (QUS) pipeline for FUJIFILM VisualSonics Vevo .iq data.

A Python port of Elizabeth Berndl's MATLAB QUS_batch pipeline (TMU Kolios/Tsai labs),
validated against MATLAB to ~1% on all parameters across a 4-concentration series
(see validation notes at bottom). Extends the original by computing per-frame QUS
*trajectories* and bubble-destruction kinetics, which the frame-1-only MATLAB
pipeline could not produce.

Pipeline stages (each a direct translation of the corresponding .m file):
    parse_vsi_xml          <- VsiParseXml.m
    read_file_header       <- ReadFileHeader.m
    read_bmode_iq          <- VsiBModeIQ.m
    reconstruct_rf         <- US_VevoLAZR_ReconstructRF.m   (interp -> interp_matlab)
    compute_sa_lci         <- compute_SA_LCI.m
    sig_fft2_2d            <- sig_fft2.m   (D==2 path)
    matlab_smooth          <- MATLAB smooth(y, span)
    fit_qus                <- compute_SS_MBF_YINT.m

High-level entry points:
    run_qus_frame()        -> SS, MBF, YINT, us_core for one frame (matches MATLAB)
    run_qus_trajectory()   -> arrays of SS, MBF, YINT, us_core over all frames
    quantify_destruction() -> exponential-decay kinetics (Porter et al. 2006 method)

Author: ported with assistance, June 2026.
Reference for destruction kinetics: Porter, Smith & Holland, J Ultrasound Med
25(12):1519-1529 (2006), doi:10.7863/jum.2006.25.12.1519 — fit A*exp(-k*t)+N to
ROI backscatter, k = decay constant, normalize to peak for cross-pressure comparison.
"""

from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.signal import hilbert, resample_poly, firwin
from scipy.optimize import curve_fit

try:
    from scipy.io import loadmat
    import mat73
except ImportError:
    loadmat = None
    mat73 = None


# ----------------------------------------------------------------------------- #
#  .mat loading (handles both v7 via scipy and v7.3/HDF5 via mat73)
# ----------------------------------------------------------------------------- #
def load_mat_any(path):
    """Load a .mat file regardless of version. Returns a dict-like object."""
    try:
        return loadmat(str(path), squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        return mat73.loadmat(str(path))


# ----------------------------------------------------------------------------- #
#  Stage 1 — XML parameter parsing  (VsiParseXml.m)
# ----------------------------------------------------------------------------- #
def parse_vsi_xml(xml_path):
    """Read B-mode scan parameters from a Vevo .iq.xml file.

    Handles the Vevo 2100 node naming: Quad-2x is absent (-> False, IntFac=16),
    focal zones stored as 'Focal-Zones-Count'.
    """
    root = ET.parse(str(xml_path)).getroot()
    raw = {p.get("name"): p.get("value")
           for p in root.iter("parameter") if p.get("name")}
    f = lambda k: float(raw[k]) if k in raw else None

    quad = raw.get("B-Mode/Quad-2x")
    quad2x = (quad == "true") if quad is not None else False
    fz = f("B-Mode/Focal-Zones-Count") or f("B-Mode/Focal-Zones") or 1.0

    return {
        "BmodeNumSamples":    f("B-Mode/Samples"),
        "BmodeNumLines":      f("B-Mode/Lines"),
        "BmodeDepthOffset":   f("B-Mode/Depth-Offset"),
        "BmodeDepth":         f("B-Mode/Depth"),
        "BmodeWidth":         f("B-Mode/Width"),
        "BmodeTxFrequency":   f("B-Mode/TX-Frequency"),
        "BmodeRxFrequency":   f("B-Mode/RX-Frequency"),
        "BmodeQuad2x":        quad2x,
        "BmodeNumFocalZones": fz,
    }


# ----------------------------------------------------------------------------- #
#  Stage 2 — file header  (ReadFileHeader.m)
# ----------------------------------------------------------------------------- #
def read_file_header(bmode_path):
    """Return (n_frames, frame_size_bytes)."""
    file_h, frame_h = 40, 56
    with open(bmode_path, "rb") as fid:
        fid.seek(4, 0)
        fmax = int(np.fromfile(fid, "<i4", 1)[0])
        fid.seek(60, 0)
        frame_data = int(np.fromfile(fid, "<i4", 1)[0])
    return fmax, frame_data + frame_h


def read_frame_times(bmode_path, divide_by=400.0, to_seconds=True):
    """Per-frame timestamps. MATLAB divides raw clock by 400 (-> milliseconds on
    Vevo 2100); to_seconds divides by a further 1000 to give seconds."""
    file_h, frame_h = 40, 56
    with open(bmode_path, "rb") as fid:
        fid.seek(4, 0)
        fmax = int(np.fromfile(fid, "<i4", 1)[0])
        fid.seek(60, 0)
        frame_size = int(np.fromfile(fid, "<i4", 1)[0]) + frame_h
        t = np.zeros(fmax)
        for k in range(fmax):
            fid.seek(file_h + k * frame_size, 0)
            t[k] = np.fromfile(fid, "<i4", 1)[0] / divide_by
    return t / 1000.0 if to_seconds else t


# ----------------------------------------------------------------------------- #
#  Stage 3 — IQ binary read  (VsiBModeIQ.m)
# ----------------------------------------------------------------------------- #
def read_bmode_iq(bmode_path, param, iframe):
    """Read interleaved Q,I int16 samples for one frame (1-based iframe).
    On-disk layout per line: Q0 I0 Q1 I1 ... (2 bytes each)."""
    file_h, frame_h, line_h, sb = 40, 56, 4, 2
    NS = int(param["BmodeNumSamples"])
    NL = int(param["BmodeNumLines"])
    NF = int(param["BmodeNumFocalZones"])
    Nl = NF * NL

    I = np.zeros((NS, Nl))
    Q = np.zeros((NS, Nl))
    header = file_h + frame_h * iframe + (sb * NS * Nl * 2 + Nl * line_h) * (iframe - 1)
    stride = sb * NS * 2 + line_h
    with open(bmode_path, "rb") as fid:
        for i in range(Nl):
            fid.seek(header + stride * i + line_h, 0)
            r = np.fromfile(fid, "<i2", NS * 2)
            Q[:, i] = r[0::2]
            I[:, i] = r[1::2]
    return I, Q


# ----------------------------------------------------------------------------- #
#  Stage 4 — RF reconstruction  (US_VevoLAZR_ReconstructRF.m)
# ----------------------------------------------------------------------------- #
def interp_matlab(x, r, l=4):
    """Approximation of MATLAB interp(x, r): unity-gain lowpass FIR of length
    2*l*r+1 at cutoff 1/r, applied via polyphase resampling. Matches MATLAB RF
    to correlation ~0.9998."""
    h = firwin(2 * l * r + 1, 1.0 / r)
    return resample_poly(x, r, 1, window=h)


def reconstruct_rf(I, Q, param):
    """Reconstruct RF from IQ: upsample by IntFac, remodulate to f_rf."""
    fs = param["BmodeRxFrequency"]
    quad = param["BmodeQuad2x"]
    IntFac = 8 if quad else 16
    if quad:
        fs *= 2
    f_rf = param["BmodeRxFrequency"]
    fs_int = fs * IntFac

    NS, NL = I.shape
    n_up = NS * IntFac
    ce = np.exp(1j * 2 * np.pi * f_rf * (np.arange(n_up) / fs_int))

    RF = np.zeros((n_up, NL))
    for i in range(NL):
        RF[:, i] = np.real(
            (interp_matlab(I[:, i], IntFac) + 1j * interp_matlab(Q[:, i], IntFac)) * ce
        )
    if quad:
        RF = -RF

    AxD = np.linspace(param["BmodeDepthOffset"], param["BmodeDepth"], RF.shape[0])
    LatD = np.linspace(0, param["BmodeWidth"], NL)
    return RF, LatD, AxD


# ----------------------------------------------------------------------------- #
#  Stage 5 — envelope / log-compression  (compute_SA_LCI.m)
# ----------------------------------------------------------------------------- #
def compute_sa_lci(RF):
    """Signal amplitude (envelope) and log-compressed image."""
    SA = np.abs(hilbert(RF, axis=0))
    MV = SA.max()
    with np.errstate(divide="ignore"):
        LCI = 20 * np.log10(SA / MV)
    LCI[np.isinf(LCI)] = np.nan
    return SA, LCI


# ----------------------------------------------------------------------------- #
#  Stage 6 — power spectrum  (sig_fft2.m, D==2)
# ----------------------------------------------------------------------------- #
def sig_fft2_2d(z, fs_Hz):
    """2-D power spectrum of a masked ROI: drop all-zero rows/cols, Hamming-window
    the collapsed region, FFT down columns, |.|^2, 10*log10. Returns (freq_MHz, Z)."""
    z = z[~np.all(z == 0, 1), :][:, ~np.all(z == 0, 0)]
    z = z * np.hamming(z.shape[0])[:, None]
    NFFT = int(2 ** np.ceil(np.log2(10640)))   # 16384
    freq = (fs_Hz / 2) * np.linspace(0, 1, NFFT // 2 + 1) / 1e6
    Z = np.abs(np.fft.fft(z, NFFT, axis=0)[:len(freq), :] ** 2)
    with np.errstate(divide="ignore"):
        Z = 10 * np.log10(Z)
    Z[np.isinf(Z)] = np.nan
    return freq, Z


def matlab_smooth(y, span):
    """MATLAB smooth(y, span): centered moving average, span forced odd,
    window shrinks symmetrically at the edges."""
    y = np.asarray(y, float)
    if span % 2 == 0:
        span -= 1
    half = span // 2
    n = len(y)
    out = np.empty(n)
    for i in range(n):
        k = min(i, n - 1 - i, half)
        out[i] = np.mean(y[i - k:i + k + 1])
    return out


# ----------------------------------------------------------------------------- #
#  Stage 7 — line fit  (compute_SS_MBF_YINT.m)
# ----------------------------------------------------------------------------- #
def fit_qus(freq, nor_PS, LF, UF, MF):
    """Linear fit of the normalized spectrum over the -6 dB band [LF, UF].
    Returns (spectral_slope, midband_fit, y_intercept)."""
    df = freq[2] - freq[1]
    Lin = len(np.arange(0, LF + 1e-9, df))
    Uin = len(np.arange(0, UF + 1e-9, df))
    MDin = len(np.arange(0, MF + 1e-9, df))
    rng = np.arange(Lin - 1, Uin)
    x, y = freq[rng], nor_PS[rng]
    fit = np.polyfit(x, y, 1)
    lin = fit[0] * x + fit[1]
    return fit[0], lin[MDin - Lin], fit[1]


# ----------------------------------------------------------------------------- #
#  High-level: reference handling
# ----------------------------------------------------------------------------- #
def load_reference(ref_path):
    """Load a Zus*_ref.mat reference struct -> dict with Z, F, LowF, UpF, MidF, ..."""
    d = load_mat_any(ref_path)
    z = d["Zus_ref"]
    if hasattr(z, "Z"):                      # scipy object
        return {k: getattr(z, k) for k in
                ["F", "Z", "LowF", "UpF", "MidF", "Start", "End", "Mid", "Max", "Transducer"]}
    return {k: z[k] for k in z}              # mat73 dict


def load_mask(mask_path):
    """Load an inputMask from a B_*_mask.mat file."""
    return np.asarray(load_mat_any(mask_path)["inputMask"])


# ----------------------------------------------------------------------------- #
#  High-level: single frame  (matches MATLAB QUS_batch frame-1 output)
# ----------------------------------------------------------------------------- #
def run_qus_frame(bmode_path, mask, ref, iframe=1):
    """Full QUS for one frame. Returns dict with SS, MBF, YINT, us_core."""
    param = parse_vsi_xml(str(bmode_path).replace(".iq.bmode", ".iq.xml"))
    I, Q = read_bmode_iq(bmode_path, param, iframe)
    RF, _, _ = reconstruct_rf(I, Q, param)
    m = mask > 0

    SA = np.abs(hilbert(RF, axis=0))
    us_core = SA[m].mean()

    fs_Hz = param["BmodeRxFrequency"] * 16
    Z_ref = np.asarray(ref["Z"]).ravel()
    freq, Z2d = sig_fft2_2d(RF * m, fs_Hz)
    nor = matlab_smooth(np.nanmean(Z2d, axis=1), 70) - Z_ref
    SS, MBF, YINT = fit_qus(freq, nor, float(ref["LowF"]), float(ref["UpF"]), float(ref["MidF"]))
    return dict(frame=iframe, SS=SS, MBF=MBF, YINT=YINT, us_core=us_core)


# ----------------------------------------------------------------------------- #
#  High-level: per-frame trajectory  (NEW — not in MATLAB)
# ----------------------------------------------------------------------------- #
def run_qus_trajectory(bmode_path, mask, ref, n_frames=None, progress=True):
    """Compute QUS for every frame. Returns dict of arrays:
    frames, SS, MBF, YINT, us_core, plus the param dict."""
    param = parse_vsi_xml(str(bmode_path).replace(".iq.bmode", ".iq.xml"))
    fmax, _ = read_file_header(bmode_path)
    if n_frames:
        fmax = min(fmax, n_frames)
    fs_Hz = param["BmodeRxFrequency"] * 16
    Z_ref = np.asarray(ref["Z"]).ravel()
    LF, UF, MF = float(ref["LowF"]), float(ref["UpF"]), float(ref["MidF"])
    m = mask > 0

    SS = np.zeros(fmax); MBF = np.zeros(fmax); YINT = np.zeros(fmax); core = np.zeros(fmax)
    for fr in range(1, fmax + 1):
        I, Q = read_bmode_iq(bmode_path, param, fr)
        RF, _, _ = reconstruct_rf(I, Q, param)
        SA = np.abs(hilbert(RF, axis=0))
        core[fr - 1] = SA[m].mean()
        freq, Z2d = sig_fft2_2d(RF * m, fs_Hz)
        nor = matlab_smooth(np.nanmean(Z2d, axis=1), 70) - Z_ref
        SS[fr - 1], MBF[fr - 1], YINT[fr - 1] = fit_qus(freq, nor, LF, UF, MF)
        if progress and fr % 50 == 0:
            print(f"    frame {fr}/{fmax}")
    return dict(frames=np.arange(1, fmax + 1), SS=SS, MBF=MBF, YINT=YINT,
                us_core=core, param=param)


# ----------------------------------------------------------------------------- #
#  High-level: destruction kinetics  (Porter et al. 2006)
# ----------------------------------------------------------------------------- #
def _exp_decay(t, A, k, N):
    return A * np.exp(-k * t) + N


def quantify_destruction(core, times, label="", min_drop_pct=15.0):
    """Quantify bubble destruction from an ROI backscatter trajectory.

    Method (Porter, Smith & Holland 2006): locate the signal peak, normalize the
    post-peak decay to the peak value, fit A*exp(-k*t)+N. k (1/s) is the decay
    constant; %drop is the peak-to-end loss. The exponential is fit only when the
    decay exceeds min_drop_pct, since flat (below-threshold) curves give
    meaningless k (the 'below threshold / static diffusion' regime).

    `times` must be in seconds for k to be in 1/s (use read_frame_times).
    """
    core = np.asarray(core, float)
    times = np.asarray(times, float)
    peak_i = int(np.argmax(core))
    t_decay = times[peak_i:] - times[peak_i]
    y_decay = core[peak_i:] / core[peak_i]
    pct_drop = (core[peak_i] - core[-1]) / core[peak_i] * 100.0

    result = dict(label=label, peak_frame=peak_i + 1, peak_val=core[peak_i],
                  pct_drop=pct_drop, t_decay=t_decay, y_decay=y_decay,
                  k=np.nan, k_err=np.nan, A=np.nan, N=np.nan)

    if pct_drop < min_drop_pct:
        result["note"] = "below threshold (no fit)"
        return result

    A0, N0 = y_decay[0] - y_decay[-1], y_decay[-1]
    k0 = 2.0 / max(t_decay[-1], 1e-6)
    try:
        popt, pcov = curve_fit(_exp_decay, t_decay, y_decay,
                               p0=[max(A0, 0.01), max(k0, 1e-4), N0],
                               bounds=([0, 0, 0], [2, np.inf, 2]), maxfev=20000)
        result["A"], result["k"], result["N"] = popt
        result["k_err"] = float(np.sqrt(np.diag(pcov))[1])
        result["note"] = "fit"
    except (RuntimeError, ValueError):
        result["note"] = "fit failed"
    return result


# ----------------------------------------------------------------------------- #
#  Convenience: run a whole concentration/power series
# ----------------------------------------------------------------------------- #
def run_series(data_root, roi_dir, ref_path, tag_map, n_frames=None):
    """Run trajectories + destruction kinetics for a labelled set of acquisitions.

    tag_map : dict like {"4p": "04-30", "10p": "10-48", ...} mapping a label to a
              substring that uniquely identifies each file's basename.
    Returns dict[label] -> {traj, times, destruction}.
    """
    data_root, roi_dir = Path(data_root), Path(roi_dir)
    ref = load_reference(ref_path)
    out = {}
    for label, tag in tag_map.items():
        bmode = next(data_root.glob(f"*{tag}*.iq.bmode"))
        mask = load_mask(next(roi_dir.glob(f"B_*{tag}*_mask.mat")))
        print(f"{label}: {bmode.name}")
        traj = run_qus_trajectory(bmode, mask, ref, n_frames=n_frames)
        times = read_frame_times(bmode)[:len(traj["us_core"])]
        dest = quantify_destruction(traj["us_core"], times, label=label)
        out[label] = dict(traj=traj, times=times, destruction=dest)
    return out


# =========================================================================== #
#  VALIDATION NOTES
#  ----------------
#  Validated against fresh MATLAB runs on a 4-concentration noson series
#  (4p/10p/50p/100p, MS250 transducer, 30 dB B-mode reference), frame 1:
#
#       conc   Python SS / MATLAB SS    Python MBF / MATLAB MBF
#       4p     2.1423 / 2.1507          -26.73 / -26.83
#       10p    1.9944 / 2.0027          -28.88 / -28.98
#       50p    1.0822 / 1.0931          -28.43 / -28.52
#       100p   1.0099 / 1.0215          -29.46 / -29.55
#
#  Agreement ~1% across all params/concentrations; residual is the
#  interp() vs interp_matlab() filter difference (RF correlation 0.9998),
#  which is below the measurement's run-to-run noise.
#
#  Per-frame trajectory validated against MATLAB at frames 1/75/150/260
#  (SS, MBF, YINT, us_core) to ~1% / ~0.3% respectively.
# =========================================================================== #
