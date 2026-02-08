import io
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.patches import Ellipse

# ---------------- Page / Style ----------------
st.set_page_config(page_title="Virtual Range", page_icon="⛳", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 1.2rem;}
      h1, h2, h3 {letter-spacing: -0.02em;}
      .vr-card {
        background: #F7F8FA;
        border: 1px solid rgba(0,0,0,0.06);
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow: 0 1px 0 rgba(0,0,0,0.03);
      }
      .vr-kpi {
        font-size: 13px;
        color: rgba(0,0,0,0.65);
        margin-bottom: 2px;
      }
      .vr-kpi strong {
        font-size: 20px;
        color: rgba(0,0,0,0.92);
        font-weight: 700;
        letter-spacing: -0.01em;
      }
      .vr-pill {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        border: 1px solid rgba(0,0,0,0.08);
        background: white;
        font-size: 12px;
        color: rgba(0,0,0,0.70);
        margin-right: 8px;
      }
      .vr-subtle {color: rgba(0,0,0,0.60); font-size: 12px;}
      .stRadio > div {gap: 6px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Helpers ----------------
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
        try:
            return float(v)
        except:
            return np.nan
    try:
        return float(val)
    except:
        return np.nan

def detect_datetime_column(df):
    preferred = ["Time", "Date", "Datetime", "Timestamp"]
    for c in preferred:
        if c in df.columns:
            return c
    for c in df.columns:
        lc = str(c).lower()
        if "time" in lc or "date" in lc:
            return c
    return None

def mahalanobis_filter(df, keep_pct):
    """Keep closest keep_pct per club using Mahalanobis distance in (Dir, Carry)."""
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

def mvee(P, tol=1e-4, max_iter=5000):
    """Minimum Volume Enclosing Ellipse (MVEE) via Khachiyan algorithm."""
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

def ellipse_params_from_points(xy):
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
    pad_x = 2.2
    pad_y = 2.2
    candidates = [0, 35, -35, 90, -90, 145, -145, 180]
    R = rotation_matrix(angle_deg)
    label_r = 3.4 + 0.22 * max(0, len(text) - 6)
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

        if not in_bounds(p):
            continue
        if not far_from_points(p):
            continue
        if not far_from_labels(p):
            continue
        if not not_inside_other_ellipses(p):
            continue

        ha = "left" if deg in (0, 35, -35) else ("right" if deg in (180, 145, -145) else "center")
        va = "bottom" if deg in (90, 35, 145) else ("top" if deg in (-90, -35, -145) else "center")
        return p, ha, va

    p = np.array([center[0] + a + pad_x, center[1]])
    p[0] = min(max(p[0], xlim[0] + 1), xlim[1] - 1)
    p[1] = min(max(p[1], ylim[0] + 1), ylim[1] - 1)
    return p, "left", "center"

def plot_virtual_range(df, clubs, title, show_ellipses):
    fig = plt.figure(figsize=(8.2, 10.0))
    ax = plt.gca()

    max_y = int(df["Carry[yd]"].max()//20*20 + 20) if len(df) else 200
    xlim = (-40, 40)
    ylim = (0, max_y + 10)

    # Soft arcs
    theta = np.linspace(-np.pi/2, np.pi/2, 400)
    for r in range(20, max_y + 1, 20):
        ax.plot(r*np.sin(theta), r*np.cos(theta), color=(0.82,0.83,0.86,0.85), lw=0.8)
        ax.text(0, r+1, f"{r} yd", ha="center", fontsize=8, color=(0.35,0.36,0.38,0.9))

    # Axis line
    ax.axvline(0, color=(0.1,0.1,0.1,0.85), lw=1.0)

    markers = {"3W":"o","5H":"s","6I":"P","7I":"^","8I":"X","9I":"D","AW":"v","PW":"v","GW":"v","SW":"x","LW":"x"}
    colors  = {"3W":"#2563EB","5H":"#16A34A","6I":"#0F766E","7I":"#F59E0B","8I":"#06B6D4","9I":"#EF4444","AW":"#7C3AED","PW":"#7C3AED","GW":"#7C3AED","SW":"#7C2D12","LW":"#7C2D12"}

    all_points_xy = df[["Dir_signed","Carry[yd]"]].values.astype(float) if len(df) else np.zeros((0,2))
    ellipses_drawn = []
    labels_placed = []

    for c in clubs:
        sub = df[df["Type"] == c]
        if sub.empty:
            continue

        col = colors.get(c, "#111827")
        mk = markers.get(c, "o")

        ax.scatter(sub["Dir_signed"], sub["Carry[yd]"], marker=mk, color=col, alpha=0.82, s=36, edgecolors="none")

        avg_carry = float(sub["Carry[yd]"].mean())
        avg_txt = f"{int(round(avg_carry))} yd"

        if show_ellipses and len(sub) >= 3:
            params = ellipse_params_from_points(sub[["Dir_signed","Carry[yd]"]].values.astype(float))
            if params:
                center, w, h, ang = params
                ax.add_patch(Ellipse(center, w, h, angle=ang, fill=False, linewidth=2, edgecolor=col, alpha=0.95))

                p, ha, va = pick_label_position(
                    center=center, width=w, height=h, angle_deg=ang, text=avg_txt,
                    all_points_xy=all_points_xy,
                    other_ellipses=ellipses_drawn,
                    other_labels=labels_placed,
                    xlim=xlim, ylim=ylim
                )
                ax.text(
                    float(p[0]), float(p[1]), avg_txt,
                    fontsize=10, ha=ha, va=va, color=col, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.80),
                )
                ellipses_drawn.append((center, w, h, ang))
                labels_placed.append((float(p[0]), float(p[1])))

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Dirección (yd)  ← Izq | Der →")
    ax.set_ylabel("Carry (yd)")
    ax.set_title(title)
    ax.set_facecolor("#F7F8FA")
    fig.patch.set_facecolor("white")
    ax.grid(False)
    return fig

def summarize_session(df, keep_pct, dates_used):
    parts = []
    if dates_used:
        parts.append(f"Last {len(dates_used)} date(s)")
    parts.append(f"Core {int(round(keep_pct*100))}%")
    return " • ".join(parts)

# ---------------- Sidebar Controls ----------------
st.sidebar.markdown("### Controles")

# Core filter always on (as you requested earlier)
keep_pct = st.sidebar.slider("Core shots", 0.50, 0.95, 0.70, 0.05)

show_ellipses = st.sidebar.checkbox("Óvalos", True)

view_mode = st.sidebar.radio("Vista", ["Un palo", "Comparar"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("### Datos")
input_mode = st.sidebar.radio("Entrada", ["Pegar CSV", "Subir archivo"], index=0)

# ---------------- Data load ----------------
df = None
if input_mode == "Pegar CSV":
    txt = st.sidebar.text_area("Pega el CSV completo", height=180)
    if txt and len(txt.strip()) >= 50:
        df = pd.read_csv(io.StringIO(txt.strip().lstrip("\ufeff")))
else:
    f = st.sidebar.file_uploader("Sube CSV", type="csv")
    if f:
        df = pd.read_csv(f)

if df is None:
    st.markdown("## ⛳ Virtual Range")
    st.markdown('<span class="vr-subtle">Pega el CSV en la barra lateral o súbelo para ver el range.</span>', unsafe_allow_html=True)
    st.stop()

required = {"Carry[yd]", "Launch Direction", "Type"}
if not required.issubset(df.columns):
    st.error("CSV inválido: faltan columnas requeridas (Carry[yd], Launch Direction, Type).")
    st.stop()

# Date filtering (last N dates)
dates_used = None
dt_col = detect_datetime_column(df)
if dt_col:
    df["_dt"] = pd.to_datetime(df[dt_col], errors="coerce")
    df["_date"] = df["_dt"].dt.date
    dates = sorted(df["_date"].dropna().unique())
    if len(dates) >= 2:
        n = st.sidebar.slider("Últimas fechas", 1, len(dates), min(len(dates), 5), 1)
        dates_used = list(dates[-n:])
        df = df[df["_date"].isin(dates_used)].copy()
    elif len(dates) == 1:
        dates_used = [dates[0]]
        df = df[df["_date"] == dates[0]].copy()

# Clean + filter
df["Dir_signed"] = df["Launch Direction"].apply(parse_direction)
df = df.dropna(subset=["Carry[yd]", "Dir_signed", "Type"]).copy()
if df.empty:
    st.error("No quedaron golpes válidos con los filtros actuales.")
    st.stop()

df = mahalanobis_filter(df, keep_pct)

clubs_all = sorted(df["Type"].unique())
if not clubs_all:
    st.error("No hay datos luego de filtrar el core.")
    st.stop()

# Club selection
default_focus = clubs_all[0]
if view_mode == "Un palo":
    club_focus = st.sidebar.selectbox("Palo", clubs_all, index=0)
    clubs_plot = [club_focus]
else:
    clubs_plot = st.sidebar.multiselect("Palos", clubs_all, default=clubs_all)

# ---------------- Header ----------------
session_line = summarize_session(df, keep_pct, dates_used)
st.markdown(
    f"""
    <div style="display:flex; align-items:flex-end; justify-content:space-between; gap:14px;">
      <div>
        <h1 style="margin-bottom:0.1rem;">⛳ Virtual Range</h1>
        <div class="vr-subtle">{session_line}</div>
      </div>
      <div>
        <span class="vr-pill">Shots: {len(df)}</span>
        <span class="vr-pill">Clubs: {df['Type'].nunique()}</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("")
tab1, tab2 = st.tabs(["Range View", "Gapping"])

# ---------------- Tab 1: Range View ----------------
with tab1:
    left, right = st.columns([1.35, 1.0], gap="large")

    with left:
        title = "Virtual Range"
        fig = plot_virtual_range(df[df["Type"].isin(clubs_plot)], clubs_plot, title, show_ellipses)
        st.pyplot(fig, clear_figure=True)

    with right:
        # KPIs for focus club (or aggregate if compare)
        if view_mode == "Un palo":
            dff = df[df["Type"] == clubs_plot[0]].copy()
            label = clubs_plot[0]
        else:
            dff = df[df["Type"].isin(clubs_plot)].copy()
            label = "Selected"

        carry_avg = float(dff["Carry[yd]"].mean())
        carry_p50 = float(dff["Carry[yd]"].median())
        lr_p84 = float(np.quantile(np.abs(dff["Dir_signed"].values), 0.84)) if len(dff) else float("nan")
        depth_p84 = float(np.quantile(np.abs(dff["Carry[yd]"].values - carry_p50), 0.84)) if len(dff) else float("nan")

        st.markdown('<div class="vr-card">', unsafe_allow_html=True)
        st.markdown(f"### {label}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="vr-kpi">Carry Avg<br><strong>{int(round(carry_avg))} yd</strong></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="vr-kpi">Carry Median<br><strong>{int(round(carry_p50))} yd</strong></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="vr-kpi">L/R (p84)±<br><strong>{int(round(lr_p84))} yd</strong></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="vr-kpi">Depth (p84)±<br><strong>{int(round(depth_p84))} yd</strong></div>', unsafe_allow_html=True)

        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

        # Small per-club mini table
        by_club = (df[df["Type"].isin(clubs_plot)]
                   .groupby("Type")
                   .agg(Shots=("Carry[yd]", "size"),
                        CarryAvg=("Carry[yd]", "mean"),
                        DirAvg=("Dir_signed", "mean"),
                        DirAbsP84=("Dir_signed", lambda s: np.quantile(np.abs(s), 0.84)))
                   .reset_index())
        by_club["CarryAvg"] = by_club["CarryAvg"].round(0).astype(int)
        by_club["DirAvg"] = by_club["DirAvg"].round(1)
        by_club["DirAbsP84"] = by_club["DirAbsP84"].round(0).astype(int)
        st.dataframe(by_club, use_container_width=True, hide_index=True)

        st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Tab 2: Gapping ----------------
with tab2:
    st.markdown("### Gapping (core shots)")
    gap = (df.groupby("Type")
             .agg(Shots=("Carry[yd]", "size"),
                  CarryAvg=("Carry[yd]", "mean"),
                  CarryP25=("Carry[yd]", lambda s: np.quantile(s, 0.25)),
                  CarryP75=("Carry[yd]", lambda s: np.quantile(s, 0.75)),
                  DirAbsP84=("Dir_signed", lambda s: np.quantile(np.abs(s), 0.84)))
             .reset_index())
    gap["CarryAvg"] = gap["CarryAvg"].round(0).astype(int)
    gap["IQR"] = (gap["CarryP75"] - gap["CarryP25"]).round(0).astype(int)
    gap["DirAbsP84"] = gap["DirAbsP84"].round(0).astype(int)
    gap = gap.drop(columns=["CarryP25","CarryP75"])
    gap = gap.sort_values("CarryAvg", ascending=False)

    st.dataframe(gap, use_container_width=True, hide_index=True)

    # Optional simple gapping deltas
    g2 = gap[["Type","CarryAvg"]].copy().reset_index(drop=True)
    g2["Δ vs next"] = (g2["CarryAvg"].diff(-1).abs()).fillna(np.nan).round(0)
    st.markdown('<div class="vr-subtle">Δ vs next: diferencia de carry promedio con el palo inmediatamente inferior (según CarryAvg).</div>', unsafe_allow_html=True)
    st.dataframe(g2, use_container_width=True, hide_index=True)
