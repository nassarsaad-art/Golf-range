import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.patches import Ellipse

# ============================
# Page setup
# ============================
st.set_page_config(page_title="Virtual Range", page_icon="⛳", layout="wide")

# IMPORTANT: Do NOT hide Streamlit header on mobile; it contains the sidebar toggle.
# We style, but keep it functional.

st.markdown(
    """
<style>
/* Page background */
.stApp { background: #E9EDF2; }

/* Sidebar (dark like mockup) */
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, #1F2937 0%, #111827 100%);
  border-right: 1px solid rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] * { color: rgba(255,255,255,0.92) !important; }
section[data-testid="stSidebar"] label { color: rgba(255,255,255,0.80) !important; }
section[data-testid="stSidebar"] .stMarkdown { color: rgba(255,255,255,0.88) !important; }
section[data-testid="stSidebar"] [data-baseweb="select"] > div{
  background: rgba(255,255,255,0.08) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
}
section[data-testid="stSidebar"] input, section[data-testid="stSidebar"] textarea{
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  color: rgba(255,255,255,0.92) !important;
}

/* Main shell (card) */
.vr-shell{
  background: #F7F8FA;
  border: 1px solid rgba(0,0,0,0.10);
  border-radius: 16px;
  overflow: hidden;
  box-shadow: 0 18px 36px rgba(0,0,0,0.16);
}
.vr-topbar{
  height: 52px;
  background: linear-gradient(180deg, #2B3646 0%, #1F2937 100%);
  display:flex; align-items:center; justify-content:space-between;
  padding: 0 16px;
  color: rgba(255,255,255,0.92);
}
.vr-topbar .left{
  display:flex; align-items:center; gap:10px;
  font-weight: 900; letter-spacing: -0.02em; font-size: 18px;
}
.vr-topbar .right{
  display:flex; align-items:center; gap:14px;
  color: rgba(255,255,255,0.78);
  font-size: 16px;
}
.vr-sessionline{
  height: 38px;
  display:flex; align-items:center; justify-content:center;
  background: #F7F8FA;
  border-bottom: 1px solid rgba(0,0,0,0.08);
  color: rgba(0,0,0,0.62);
  font-size: 12px;
}
.vr-plotpad{ padding: 8px 14px 0 14px; }

/* KPI bar */
.vr-kpibar{
  background: #F1F3F6;
  border-top: 1px solid rgba(0,0,0,0.10);
  display:flex;
  flex-wrap: wrap;
}
.vr-kpi{
  flex: 1;
  min-width: 160px;
  padding: 12px 14px;
  border-right: 1px solid rgba(0,0,0,0.08);
  border-bottom: 1px solid rgba(0,0,0,0.08);
}
.vr-kpi:last-child{ border-right:none; }
.vr-kpi .label{ font-size: 11px; color: rgba(0,0,0,0.58); }
.vr-kpi .value{ font-size: 18px; font-weight: 900; color: rgba(0,0,0,0.86); letter-spacing: -0.01em; }
.vr-kpi .value.small{ font-size: 15px; font-weight: 900; }
.vr-kpi .mono{ font-variant-numeric: tabular-nums; font-feature-settings:"tnum"; }

/* Make uploader / inputs feel native */
div[data-testid="stFileUploader"]{
  background: rgba(255,255,255,0.85);
  border: 1px solid rgba(0,0,0,0.10);
  border-radius: 14px;
  padding: 10px 12px;
}
textarea{
  border-radius: 12px !important;
}

/* Container size */
.block-container{ padding-top: 0.8rem; padding-bottom: 1.0rem; max-width: 1200px; }
</style>
""",
    unsafe_allow_html=True
)

# ============================
# Helpers
# ============================
def parse_direction(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, str):
        v = val.strip().upper()
        if v.startswith("L"):
            try: return -float(v[1:])
            except: return np.nan
        if v.startswith("R"):
            try: return float(v[1:])
            except: return np.nan
        try: return float(v)
        except: return np.nan
    try:
        return float(val)
    except:
        return np.nan

def detect_datetime_column(df: pd.DataFrame):
    preferred = ["Time", "Date", "Datetime", "Timestamp"]
    for c in preferred:
        if c in df.columns:
            return c
    for c in df.columns:
        lc = str(c).lower()
        if "time" in lc or "date" in lc:
            return c
    return None

def mahalanobis_filter(df: pd.DataFrame, keep_pct: float) -> pd.DataFrame:
    def filt(g):
        if len(g) < 6:
            return g
        X = g[["Dir_signed", "Carry[yd]"]].values.astype(float)
        mu = X.mean(axis=0)
        cov = np.cov(X.T)
        inv = np.linalg.pinv(cov)
        d2 = np.einsum("ij,jk,ik->i", X - mu, inv, X - mu)
        cutoff = np.quantile(d2, keep_pct)
        return g[d2 <= cutoff]
    return df.groupby("Type", group_keys=False).apply(filt)

# Tight enclosing ellipse (MVEE)
def mvee(P: np.ndarray, tol: float = 1e-4, max_iter: int = 5000):
    P = P.astype(float)
    n, d = P.shape
    if n < 2:
        return P.mean(axis=0), np.eye(d)
    Q = np.column_stack([P, np.ones(n)]).T
    u = np.ones(n) / n
    for _ in range(max_iter):
        X = Q @ np.diag(u) @ Q.T
        X_inv = np.linalg.pinv(X)
        M = np.einsum("ij,jk,ki->i", Q.T, X_inv, Q)
        j = int(np.argmax(M))
        max_M = float(M[j])
        step = (max_M - (d + 1)) / ((d + 1) * (max_M - 1))
        new_u = (1 - step) * u
        new_u[j] += step
        if np.linalg.norm(new_u - u) < tol:
            u = new_u
            break
        u = new_u
    c = P.T @ u
    S = (P - c).T @ np.diag(u) @ (P - c)
    A = np.linalg.pinv(S) / d
    return c, A

def ellipse_params_from_points(xy: np.ndarray):
    if xy.shape[0] < 3:
        return None
    c, A = mvee(xy)
    shape = np.linalg.pinv(A)
    eigvals, eigvecs = np.linalg.eigh(shape)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    a = float(np.sqrt(eigvals[0]))
    b = float(np.sqrt(eigvals[1]))
    angle = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))
    return c, 2*a, 2*b, angle

def rotation_matrix(deg):
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s],[s, c]])

def point_inside_ellipse(pt, center, width, height, angle_deg, pad=0.0):
    R = rotation_matrix(angle_deg)
    v = np.array(pt) - np.array(center)
    u = R.T @ v
    a = width/2.0 + pad
    b = height/2.0 + pad
    if a <= 0 or b <= 0:
        return False
    return (u[0]**2)/(a*a) + (u[1]**2)/(b*b) <= 1.0

def pick_label_position(center, width, height, angle_deg, text, all_points_xy, other_ellipses, other_labels, xlim, ylim):
    pad_x, pad_y = 2.0, 2.0
    candidates = [0, 30, -30, 90, -90, 150, -150, 180]
    R = rotation_matrix(angle_deg)
    label_r = 3.3 + 0.20 * max(0, len(text) - 6)
    a = width/2.0
    b = height/2.0

    def in_bounds(p):
        return (xlim[0] + 1 <= p[0] <= xlim[1] - 1) and (ylim[0] + 1 <= p[1] <= ylim[1] - 1)

    def far_from_points(p):
        if all_points_xy.size == 0:
            return True
        d2 = np.sum((all_points_xy - p)**2, axis=1)
        return float(np.min(d2)) >= (label_r**2)

    def far_from_labels(p):
        for q in other_labels:
            if (p[0]-q[0])**2 + (p[1]-q[1])**2 < (label_r**2):
                return False
        return True

    def not_inside_other_ellipses(p):
        for (c2, w2, h2, ang2) in other_ellipses:
            if point_inside_ellipse(p, c2, w2, h2, ang2, pad=1.2):
                return False
        return True

    for deg in candidates:
        t = np.deg2rad(deg)
        local = np.array([(a + pad_x) * np.cos(t), (b + pad_y) * np.sin(t)])
        p = (np.array(center) + R @ local).astype(float)

        if not in_bounds(p): continue
        if not far_from_points(p): continue
        if not far_from_labels(p): continue
        if not not_inside_other_ellipses(p): continue

        ha = "left" if deg in (0, 30, -30) else ("right" if deg in (180, 150, -150) else "center")
        va = "bottom" if deg in (90, 30, 150) else ("top" if deg in (-90, -30, -150) else "center")
        return p, ha, va

    p = np.array([center[0] + a + pad_x, center[1]])
    p[0] = min(max(p[0], xlim[0] + 1), xlim[1] - 1)
    p[1] = min(max(p[1], ylim[0] + 1), ylim[1] - 1)
    return p, "left", "center"

# ============================
# Plot styling
# ============================
def add_background(ax, xlim, ylim):
    W, H = 900, 520
    img = np.zeros((H, W, 3), dtype=float)

    top = np.array([0.96, 0.97, 0.98])
    mid = np.array([0.93, 0.94, 0.96])
    grass = np.array([0.78, 0.86, 0.78])

    split = int(H * 0.58)
    for y in range(H):
        if y < split:
            tt = y / max(1, split - 1)
            col = top * (1 - tt) + mid * tt
        else:
            tt = (y - split) / max(1, (H - split - 1))
            col = mid * (1 - tt) + grass * tt
        img[y, :, :] = col

    ax.imshow(img, extent=[xlim[0], xlim[1], ylim[0], ylim[1]], aspect="auto", zorder=0)

    for i in range(-8, 9):
        ax.plot([i*5.0, i*1.8], [ylim[0], ylim[0] + (ylim[1]-ylim[0])*0.48],
                color=(0.60, 0.72, 0.62, 0.20), lw=1.0, zorder=1)

    for y in np.linspace(ylim[0] + 12, ylim[0] + (ylim[1]-ylim[0])*0.50, 7):
        ax.plot([xlim[0], xlim[1]], [y, y], color=(0.20,0.22,0.25,0.08), lw=1.0, zorder=1)

def plot_virtual_range(df, clubs, session_label: str):
    fig = plt.figure(figsize=(10.5, 5.7))
    ax = plt.gca()

    xlim = (-40, 40)
    max_y = int(df["Carry[yd]"].max()//20*20 + 20) if len(df) else 220
    ylim = (0, max_y + 10)

    add_background(ax, xlim, ylim)

    theta = np.linspace(-np.pi/2, np.pi/2, 400)
    for r in range(20, max_y + 1, 20):
        ax.plot(r*np.sin(theta), r*np.cos(theta), color=(0.20,0.22,0.25,0.12), lw=1.0, zorder=2)

    ax.axvline(0, color=(0.12,0.13,0.15,0.35), lw=1.1, zorder=3)

    ax.text(0, ylim[1] - 6, session_label, fontsize=9, color=(0.10,0.11,0.12,0.55),
            ha="center", va="top", zorder=4)

    markers = {"3W":"o","5H":"s","6I":"P","7I":"^","8I":"X","9I":"D","PW":"v","AW":"v","GW":"v","SW":"x","LW":"x"}
    colors  = {"3W":"#2563EB","5H":"#16A34A","6I":"#0F766E","7I":"#2563EB","8I":"#06B6D4","9I":"#EF4444",
               "PW":"#111827","AW":"#111827","GW":"#111827","SW":"#7C2D12","LW":"#7C2D12"}

    all_points_xy = df[["Dir_signed","Carry[yd]"]].values.astype(float) if len(df) else np.zeros((0,2))
    ellipses_drawn, labels_placed = [], []

    for c in clubs:
        sub = df[df["Type"] == c]
        if sub.empty:
            continue

        col = colors.get(c, "#111827")
        mk = markers.get(c, "o")

        ax.scatter(sub["Dir_signed"], sub["Carry[yd]"],
                   s=28, marker=mk, color=col, alpha=0.35,
                   edgecolors=(1,1,1,0.35), linewidths=0.5, zorder=5)

        if len(sub) >= 3:
            params = ellipse_params_from_points(sub[["Dir_signed","Carry[yd]"]].values.astype(float))
            if params:
                center, w, h, ang = params
                ax.add_patch(Ellipse(center, w, h, angle=ang, fill=False, linewidth=2.0,
                                     edgecolor=col, alpha=0.85, zorder=6))

                avg_carry = float(sub["Carry[yd]"].mean())
                txt = f"{int(round(avg_carry))} yd"
                p, ha, va = pick_label_position(center, w, h, ang, txt, all_points_xy, ellipses_drawn, labels_placed, xlim, ylim)
                ax.text(float(p[0]), float(p[1]), txt,
                        fontsize=10, ha=ha, va=va, color=(0.08,0.09,0.10,0.90),
                        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="none", alpha=0.95),
                        zorder=7)
                ellipses_drawn.append((center, w, h, ang))
                labels_placed.append((float(p[0]), float(p[1])))

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.patch.set_facecolor("#E9EDF2")
    return fig

# ============================
# Sidebar controls (optional)
# ============================
with st.sidebar:
    st.markdown("### 🔎 Clubs")
    compare = st.toggle("Compare", value=False)

    st.markdown("### CORE SHOTS")
    keep_pct = st.slider("", 0.50, 0.95, 0.70, 0.05)
    st.markdown("<div style='font-size:11px; color:rgba(255,255,255,0.70); margin-top:6px;'>Mostrando golpes más consistentes</div>", unsafe_allow_html=True)

# ============================
# MAIN: Always show working input
# ============================
st.markdown('<div class="vr-shell">', unsafe_allow_html=True)
st.markdown(
    """
    <div class="vr-topbar">
      <div class="left">🛡️ Virtual Range</div>
      <div class="right">＋ &nbsp; ◯ &nbsp; 👤</div>
    </div>
    """,
    unsafe_allow_html=True
)

# Inputs (always visible; not inside HTML overlays)
tabs = st.tabs(["Subir CSV", "Pegar CSV"])

df = None
with tabs[0]:
    f = st.file_uploader("Selecciona el CSV", type="csv", accept_multiple_files=False)
    if f is not None:
        df = pd.read_csv(f)

with tabs[1]:
    txt = st.text_area("Pega el CSV completo", height=220, placeholder="Copia desde Golfboy y pega aquí…")
    if df is None and txt and len(txt.strip()) >= 50:
        df = pd.read_csv(io.StringIO(txt.strip().lstrip("\ufeff")))

if df is None:
    st.markdown('<div class="vr-sessionline">Sube un CSV para ver el range</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

required = {"Carry[yd]", "Launch Direction", "Type"}
if not required.issubset(df.columns):
    st.error("CSV inválido: faltan columnas requeridas (Carry[yd], Launch Direction, Type).")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# Date slider (main, always works)
dates_used = []
dt_col = detect_datetime_column(df)
dates = []
if dt_col:
    df["_dt"] = pd.to_datetime(df[dt_col], errors="coerce")
    df["_date"] = df["_dt"].dt.date
    dates = sorted(df["_date"].dropna().unique())

if len(dates) > 1:
    n_dates = st.slider("Últimas fechas", 1, len(dates), min(len(dates), 3), 1)
    dates_used = list(dates[-n_dates:])
    df = df[df["_date"].isin(dates_used)].copy()
elif len(dates) == 1:
    dates_used = [dates[0]]
    df = df[df["_date"] == dates[0]].copy()

# Clean + core filter
df["Dir_signed"] = df["Launch Direction"].apply(parse_direction)
df = df.dropna(subset=["Carry[yd]", "Dir_signed", "Type"]).copy()
if df.empty:
    st.error("No quedaron golpes válidos.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

df = mahalanobis_filter(df, keep_pct)

clubs_all = sorted(df["Type"].unique())
if not clubs_all:
    st.error("No hay datos luego del filtro.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# Club selection (main)
if compare:
    clubs_plot = st.multiselect("Palos (comparar)", clubs_all, default=clubs_all[:min(len(clubs_all), 4)])
    if not clubs_plot:
        clubs_plot = clubs_all[:1]
    focus_club = clubs_plot[0]
else:
    focus_club = st.selectbox("Palo", clubs_all, index=0)
    clubs_plot = [focus_club]

# KPIs
dff = df[df["Type"] == focus_club].copy()
shots = int(len(dff)) if len(dff) else 0
carry_avg = float(dff["Carry[yd]"].mean()) if shots else float("nan")
carry_med = float(np.median(dff["Carry[yd]"].values)) if shots else float("nan")
lr_p84 = float(np.quantile(np.abs(dff["Dir_signed"].values), 0.84)) if shots else float("nan")
depth_p84 = float(np.quantile(np.abs(dff["Carry[yd]"].values - carry_med), 0.84)) if shots else float("nan")

disp = "Tight"
if not np.isnan(lr_p84) and not np.isnan(depth_p84):
    score = lr_p84 + 0.6 * depth_p84
    if score > 14: disp = "Wide"
    elif score > 10: disp = "OK"

n_dates_show = len(dates_used) if dates_used else 1
session_label = f"Session: Last {n_dates_show} dates  ·  Core: {int(round(keep_pct*100))}%  ·  Cali"
st.markdown(f'<div class="vr-sessionline">{session_label}</div>', unsafe_allow_html=True)

# Plot
st.markdown('<div class="vr-plotpad">', unsafe_allow_html=True)
fig = plot_virtual_range(df[df["Type"].isin(clubs_plot)], clubs_plot, session_label=session_label)
st.pyplot(fig, clear_figure=True, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# KPI bar
st.markdown(
    f"""
    <div class="vr-kpibar">
      <div class="vr-kpi">
        <div class="label">{focus_club}</div>
        <div class="value mono">{int(round(carry_avg)) if not np.isnan(carry_avg) else "-"} yd</div>
      </div>
      <div class="vr-kpi">
        <div class="label">Carry Avg</div>
        <div class="value mono">{int(round(carry_avg)) if not np.isnan(carry_avg) else "-"}</div>
      </div>
      <div class="vr-kpi">
        <div class="label">Dispersion</div>
        <div class="value small">{disp}</div>
      </div>
      <div class="vr-kpi">
        <div class="label">L/R</div>
        <div class="value mono">±{int(round(lr_p84)) if not np.isnan(lr_p84) else "-"}</div>
      </div>
      <div class="vr-kpi">
        <div class="label">Depth</div>
        <div class="value mono">±{int(round(depth_p84)) if not np.isnan(depth_p84) else "-"}</div>
      </div>
      <div class="vr-kpi">
        <div class="label">Shots</div>
        <div class="value mono">{shots}</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("</div>", unsafe_allow_html=True)
