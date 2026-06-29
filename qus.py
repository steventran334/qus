"""
qus_app.py — Streamlit app for Vevo 2100 QUS + nanobubble destruction analysis.

Run with:   streamlit run qus_app.py

Backend is qus_pipeline.py (must be in the same folder). The app:
  1. Browse a data folder, list acquisitions (one per .iq.bmode).
  2. Show B-mode and contrast frames; draw a polygon ROI on either.
  3. Apply each ROI per-acquisition, OR draw once and apply to all powers.
  4. Run per-frame QUS trajectories + Porter-2006 destruction kinetics.
  5. View trajectory overlays, destruction summary, and a results table.

QUS results are validated for B-MODE only (matched to MATLAB to ~1%). The
contrast view is for visualization / ROI placement; contrast QUS is not yet
validated and is not computed here.
"""

import json
from pathlib import Path

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath

import qus_pipeline as qp

try:
    from streamlit_drawable_canvas import st_canvas
    HAVE_CANVAS = True
except ImportError:
    HAVE_CANVAS = False

# --------------------------------------------------------------------------- #
#  Page setup + light, deliberate styling (lab instrument, not a landing page)
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="QUS · Nanobubble Destruction", layout="wide")

st.markdown("""
<style>
  .stApp { background: #0e1116; }
  h1, h2, h3 { color: #e8edf4; font-family: 'IBM Plex Mono', monospace; letter-spacing:-0.5px; }
  .metric-card { background:#161b22; border:1px solid #2a323d; border-radius:8px;
                 padding:14px 16px; }
  .accent { color:#4ec9b0; }
  .muted  { color:#7a8694; font-size:0.85rem; }
  .stDataFrame { font-family:'IBM Plex Mono', monospace; }
</style>
""", unsafe_allow_html=True)

st.title("QUS · Nanobubble Destruction")
st.markdown("<span class='muted'>Vevo 2100 quantitative ultrasound — per-frame "
            "trajectories and acoustic destruction kinetics. "
            "<span class='accent'>B-mode QUS validated against MATLAB to ~1%.</span></span>",
            unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
#  Session state
# --------------------------------------------------------------------------- #
ss = st.session_state
ss.setdefault("rois", {})          # {label: vertices list}
ss.setdefault("results", {})       # {label: result dict}
ss.setdefault("shared_roi", None)  # vertices used for all powers


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def list_acquisitions(folder):
    """Return list of (label, bmode_path) for every .iq.bmode in folder."""
    folder = Path(folder)
    out = []
    for b in sorted(folder.glob("*.iq.bmode")):
        out.append((b.stem.replace(".iq", ""), str(b)))
    return out


@st.cache_data(show_spinner=False)
def bmode_image(bmode_path, iframe=1):
    """Reconstruct one frame -> (LCI dB image, contrast LCI or None, extent)."""
    param = qp.parse_vsi_xml(str(bmode_path).replace(".iq.bmode", ".iq.xml"))
    I, Q = qp.read_bmode_iq(bmode_path, param, iframe)
    RF, LatD, AxD = qp.reconstruct_rf(I, Q, param)
    _, LCI = qp.compute_sa_lci(RF)
    extent = [LatD[0], LatD[-1], AxD[-1], AxD[0]]

    # contrast (if present) — display only
    cpath = str(bmode_path).replace(".iq.bmode", ".iq.contrast")
    contrast = None
    if Path(cpath).exists():
        try:
            # contrast uses the same line geometry; reconstruct frame 1 for display
            Ic, Qc = qp.read_bmode_iq(cpath, param, iframe)  # same layout
            RFc, _, _ = qp.reconstruct_rf(Ic, Qc, param)
            _, contrast = qp.compute_sa_lci(RFc)
        except Exception:
            contrast = None
    return LCI, contrast, extent, RF.shape


def lci_to_uint8(LCI, vmin=-60, vmax=0):
    """Map a dB image to 0-255 grayscale for canvas background."""
    img = np.clip(LCI, vmin, vmax)
    img = (img - vmin) / (vmax - vmin) * 255
    return np.nan_to_num(img).astype(np.uint8)


def polygon_to_mask(vertices, shape, canvas_w, canvas_h):
    """Rasterize polygon (canvas pixel coords) onto the RF grid (shape = rows,cols).
    Matches MATLAB roipoly: point-in-polygon fill on the image grid."""
    rows, cols = shape
    # canvas coords are in display pixels; scale to RF grid
    sx = cols / canvas_w
    sy = rows / canvas_h
    verts = np.array([[v[0] * sx, v[1] * sy] for v in vertices])
    path = MplPath(verts)
    yy, xx = np.mgrid[0:rows, 0:cols]
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    mask = path.contains_points(pts).reshape(rows, cols)
    return mask


# --------------------------------------------------------------------------- #
#  Sidebar — data + reference selection
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Data")
    data_folder = st.text_input("Experiment folder",
                                value=str(Path.home() / "Documents" / "QUS_python" /
                                          "data" / "2026-06-26 - test"))
    roi_folder = st.text_input("ROI / reference folder",
                               value=str(Path(data_folder) / "260626_ROI"))
    ref_file = st.text_input("Reference .mat",
                             value="Zus_MS250_bmode_30dB_ref.mat")

    st.divider()
    st.header("ROI mode")
    roi_mode = st.radio("Apply ROI", ["Per-acquisition", "One ROI for all powers"],
                        help="Per-acquisition: draw a separate ROI on each well "
                             "(use when gels differ per power). Shared: draw once, "
                             "apply to all.")

    st.divider()
    n_frames = st.number_input("Frames to process (0 = all)", 0, 1000, 0)
    n_frames = None if n_frames == 0 else int(n_frames)

# load acquisitions + reference
try:
    acqs = list_acquisitions(data_folder)
    ref = qp.load_reference(Path(roi_folder) / ref_file)
    ref_ok = True
except Exception as e:
    st.error(f"Could not load data/reference: {e}")
    acqs, ref_ok = [], False

if not HAVE_CANVAS:
    st.warning("`streamlit-drawable-canvas` not installed. Run: "
               "`pip install streamlit-drawable-canvas`  — then restart the app. "
               "Polygon drawing is disabled until then.")

# --------------------------------------------------------------------------- #
#  Tabs
# --------------------------------------------------------------------------- #
tab_roi, tab_run, tab_results = st.tabs(["1 · Draw ROIs", "2 · Run QUS", "3 · Results"])

# ---- Tab 1: ROI drawing -------------------------------------------------- #
with tab_roi:
    if not acqs:
        st.info("Point the sidebar at a folder containing .iq.bmode files.")
    else:
        labels = [a[0] for a in acqs]
        # friendly short labels (e.g. the power token if present)
        sel = st.selectbox("Acquisition", labels,
                           format_func=lambda s: s[:60] + ("…" if len(s) > 60 else ""))
        bmode_path = dict(acqs)[sel]

        LCI, contrast, extent, rf_shape = bmode_image(bmode_path)
        bg = lci_to_uint8(LCI)

        view = st.radio("Draw on", ["B-mode", "Contrast"] if contrast is not None else ["B-mode"],
                        horizontal=True)
        img = bg if view == "B-mode" else lci_to_uint8(contrast)

        # show both side by side for reference
        c1, c2 = st.columns(2)
        with c1:
            st.caption("B-mode (QUS computed here)")
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.imshow(LCI, cmap="gray", vmin=-60, vmax=0, aspect="auto", extent=extent)
            ax.set_xlabel("lateral (mm)"); ax.set_ylabel("axial (mm)")
            st.pyplot(fig); plt.close(fig)
        with c2:
            if contrast is not None:
                st.caption("Contrast (display only)")
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.imshow(contrast, cmap="inferno", vmin=-60, vmax=0, aspect="auto", extent=extent)
                ax.set_xlabel("lateral (mm)"); ax.set_ylabel("axial (mm)")
                st.pyplot(fig); plt.close(fig)
            else:
                st.caption("No contrast file for this acquisition.")

        st.markdown("**Draw polygon** — click vertices, double-click / close to finish.")
        if HAVE_CANVAS:
            from PIL import Image
            disp_h, disp_w = 360, 540
            bg_img = Image.fromarray(img).resize((disp_w, disp_h))
            canvas = st_canvas(
                fill_color="rgba(78,201,176,0.25)",
                stroke_color="#4ec9b0", stroke_width=2,
                background_image=bg_img, height=disp_h, width=disp_w,
                drawing_mode="polygon", key=f"canvas_{sel}_{view}",
            )
            if canvas.json_data and canvas.json_data["objects"]:
                obj = canvas.json_data["objects"][-1]
                if obj["type"] == "path":
                    pts = [(p[1], p[2]) for p in obj["path"] if len(p) >= 3]
                elif "points" in obj:
                    pts = [(p["x"], p["y"]) for p in obj["points"]]
                else:
                    pts = []
                if len(pts) >= 3:
                    if st.button("Save this ROI"):
                        mask = polygon_to_mask(pts, rf_shape, disp_w, disp_h)
                        ss.rois[sel] = {"vertices": pts, "shape": rf_shape,
                                        "disp": (disp_w, disp_h), "npix": int(mask.sum())}
                        if roi_mode == "One ROI for all powers":
                            ss.shared_roi = ss.rois[sel]
                        st.success(f"ROI saved for {sel[:40]} — {int(mask.sum())} pixels"
                                   + (" (shared with all powers)" if roi_mode.startswith("One") else ""))

        # show saved ROIs
        if ss.rois:
            st.divider()
            st.caption("Saved ROIs")
            for lbl, r in ss.rois.items():
                st.write(f"• {lbl[:50]} — {r['npix']} px")


# ---- Tab 2: run -------------------------------------------------------- #
with tab_run:
    if not acqs or not ref_ok:
        st.info("Load data and a reference first.")
    elif not ss.rois and ss.shared_roi is None:
        st.info("Draw at least one ROI in tab 1.")
    else:
        st.write("Acquisitions ready to process:")
        to_run = []
        for lbl, path in acqs:
            roi = ss.shared_roi if (roi_mode.startswith("One") and ss.shared_roi) else ss.rois.get(lbl)
            status = "✓ ROI" if roi else "— no ROI"
            st.write(f"• {lbl[:55]} … {status}")
            if roi:
                to_run.append((lbl, path, roi))

        if st.button("Run QUS on all with ROIs", type="primary"):
            prog = st.progress(0.0)
            for i, (lbl, path, roi) in enumerate(to_run):
                mask = polygon_to_mask(roi["vertices"], roi["shape"], *roi["disp"])
                traj = qp.run_qus_trajectory(path, mask, ref, n_frames=n_frames, progress=False)
                times = qp.read_frame_times(path)[:len(traj["us_core"])]
                dest = qp.quantify_destruction(traj["us_core"], times, label=lbl)
                ss.results[lbl] = {"traj": traj, "times": times, "destruction": dest}
                prog.progress((i + 1) / len(to_run))
            st.success(f"Processed {len(to_run)} acquisitions.")


# ---- Tab 3: results ---------------------------------------------------- #
with tab_results:
    if not ss.results:
        st.info("Run QUS in tab 2 to see results.")
    else:
        labels = list(ss.results.keys())

        # summary metrics row
        st.subheader("Destruction summary")
        cols = st.columns(len(labels))
        for col, lbl in zip(cols, labels):
            d = ss.results[lbl]["destruction"]
            kstr = f"{d['k']:.3f} s⁻¹" if not np.isnan(d["k"]) else "—"
            col.markdown(f"<div class='metric-card'><b>{lbl[:18]}</b><br>"
                         f"<span class='muted'>peak→end loss</span><br>"
                         f"<span style='font-size:1.6rem' class='accent'>{d['pct_drop']:.1f}%</span><br>"
                         f"<span class='muted'>k = {kstr}</span></div>",
                         unsafe_allow_html=True)

        # trajectory overlay
        st.subheader("QUS trajectories")
        keys = [("us_core", "ROI backscatter"), ("MBF", "Midband fit (dB)"),
                ("SS", "Spectral slope (dB/MHz)"), ("YINT", "Y-intercept (dB)")]
        fig, ax = plt.subplots(2, 2, figsize=(12, 7))
        for lbl in labels:
            t = ss.results[lbl]["traj"]
            for j, (k, ttl) in enumerate(keys):
                ax.flat[j].plot(t["frames"], t[k], alpha=0.8, label=lbl[:14])
        for j, (k, ttl) in enumerate(keys):
            ax.flat[j].set_title(ttl); ax.flat[j].set_xlabel("frame")
            ax.flat[j].grid(alpha=0.3); ax.flat[j].legend(fontsize=7)
        fig.patch.set_facecolor("#0e1116")
        for a in ax.flat:
            a.set_facecolor("#161b22")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        # destruction fits
        st.subheader("Destruction kinetics")
        c1, c2 = st.columns(2)
        with c1:
            fig, axx = plt.subplots(figsize=(6, 4.5))
            for lbl in labels:
                d = ss.results[lbl]["destruction"]
                axx.plot(d["t_decay"], d["y_decay"], ".", ms=3, alpha=0.35)
                if not np.isnan(d["k"]):
                    tt = np.linspace(0, d["t_decay"][-1], 200)
                    axx.plot(tt, qp._exp_decay(tt, d["A"], d["k"], d["N"]),
                             "-", lw=2, label=f"{lbl[:12]} k={d['k']:.3f}")
            axx.set_xlabel("time since peak (s)"); axx.set_ylabel("normalized backscatter")
            axx.legend(fontsize=7); axx.grid(alpha=0.3)
            st.pyplot(fig); plt.close(fig)
        with c2:
            fig, axx = plt.subplots(figsize=(6, 4.5))
            drops = [ss.results[l]["destruction"]["pct_drop"] for l in labels]
            axx.bar(range(len(labels)), drops, color="#4ec9b0")
            axx.set_xticks(range(len(labels)))
            axx.set_xticklabels([l[:10] for l in labels], rotation=30, ha="right", fontsize=7)
            axx.set_ylabel("% backscatter lost"); axx.set_title("Destruction extent")
            axx.grid(alpha=0.3, axis="y")
            st.pyplot(fig); plt.close(fig)

        # table + download
        st.subheader("Results table")
        import pandas as pd
        rows = []
        for lbl in labels:
            d = ss.results[lbl]["destruction"]
            rows.append({"acquisition": lbl, "peak_frame": d["peak_frame"],
                         "peak_val": round(d["peak_val"], 1),
                         "pct_drop": round(d["pct_drop"], 1),
                         "k_per_s": round(d["k"], 4) if not np.isnan(d["k"]) else None,
                         "k_err": round(d["k_err"], 4) if not np.isnan(d["k_err"]) else None,
                         "note": d["note"]})
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        st.download_button("Download results CSV", df.to_csv(index=False),
                           "qus_destruction_results.csv", "text/csv")
