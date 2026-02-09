import io
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.patches import Ellipse

# ---------------- Page ----------------
st.set_page_config(page_title="Virtual Range", layout="wide")

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
        try: return float(v)
        except: return np.nan
    try: return float(val)
    except: return np.nan

def rotation_matrix(angle_deg):
    t = math.radians(angle_deg)
    return np.array([[math.cos(t), -math.sin(t)],
                     [math.sin(t),  math.cos(t)]])

def ellipse_params_from_points(X):
    if len(X) < 3:
        return None
    mu = X.mean(axis=0)
    cov = np.cov(X.T)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    width, height = 2*np.sqrt(vals[0])*2.2, 2*np.sqrt(vals[1])*2.2
    angle = math.degrees(math.atan2(vecs[1,0], vecs[0,0]))
    return mu, width, height, angle

def blend_with_white(rgb, a):
    return tuple(a + (1-a)*c for c in rgb)

def darken(rgb, f):
    return tuple(max(0, c*f) for c in rgb)

def point_inside_ellipse(p, center, w, h, ang, pad=1.0):
    R = rotation_matrix(-ang)
    q = R @ (np.array(p) - np.array(center))
    return (q[0]**2)/((w/2*pad)**2) + (q[1]**2)/((h/2*pad)**2) <= 1

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Controles")
    up = st.file_uploader("Sube tu CSV de Golfboy", type=["csv"])

    core_pct = st.slider("Core shots (%)", min_value=10, max_value=100, value=70, step=5)

# ---------------- Main ----------------
tabs = st.tabs(["Virtual Range", "Trayectoria", "Métricas"])

if not up:
    with tabs[0]:
        st.info("Sube un CSV para ver el Virtual Range")
    st.stop()

# ---------------- Load data ----------------
df = pd.read_csv(up)

# Expected columns (best-effort)
# Try common names; adjust if missing
col_map = {
    "Club": ["Club", "Type"],
    "Carry": ["Carry", "Carry[yd]", "Carry (yd)"],
    "Direction": ["Direction", "Launch Direction", "Dir"],
    "Date": ["Date", "Session", "Fecha"]
}

def pick_col(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None

club_col = pick_col(df, col_map["Club"])
carry_col = pick_col(df, col_map["Carry"])
dir_col   = pick_col(df, col_map["Direction"])
date_col  = pick_col(df, col_map["Date"])

if club_col is None or carry_col is None or dir_col is None:
    st.error("No se encontraron columnas necesarias (Club, Carry, Direction).")
    st.stop()

df = df.copy()
df["Type"] = df[club_col].astype(str)
df["Carry[yd]"] = pd.to_numeric(df[carry_col], errors="coerce")
df["Dir_signed"] = df[dir_col].apply(parse_direction)

if date_col:
    df["_date"] = pd.to_datetime(df[date_col], errors="coerce")
else:
    df["_date"] = pd.NaT

df = df.dropna(subset=["Carry[yd]", "Dir_signed"])

# ---------------- Date slider (SAFE) ----------------
dates = df["_date"].dropna().sort_values().unique()
with st.sidebar:
    if len(dates) == 0:
        n_dates = 1
        st.caption("Sin fechas detectables; usando todos los golpes")
    else:
        max_n = len(dates)
        n_dates = st.slider("¿Cuántas fechas?", min_value=1, max_value=max_n, value=min(3, max_n), step=1)

if len(dates) > 0:
    keep = set(dates[-n_dates:])
    df = df[df["_date"].isin(keep)]

# ---------------- Core shots by distance clustering ----------------
def core_by_distance(df, pct):
    keep_idx = []
    for c, g in df.groupby("Type"):
        if len(g) < 3:
            keep_idx.extend(g.index.tolist())
            continue
        mu = g[["Dir_signed","Carry[yd]"]].mean().values
        d = np.linalg.norm(g[["Dir_signed","Carry[yd]"]].values - mu, axis=1)
        k = max(1, int(len(g)*pct/100))
        keep = g.iloc[np.argsort(d)[:k]].index
        keep_idx.extend(keep.tolist())
    return df.loc[keep_idx]

dfc = core_by_distance(df, core_pct)

# ---------------- Colors ----------------
palette = [
    (0.20,0.45,0.85),(0.20,0.70,0.35),(0.85,0.35,0.35),
    (0.45,0.25,0.65),(0.85,0.60,0.20),(0.10,0.65,0.65)
]
clubs = sorted(dfc["Type"].unique(), key=lambda x: dfc[dfc["Type"]==x]["Carry[yd]"].mean())
color_map = {c: palette[i%len(palette)] for i,c in enumerate(clubs)}
marker_map = ["o","s","^","D","v","x"]

# ---------------- Virtual Range ----------------
with tabs[0]:
    fig, ax = plt.subplots(figsize=(6,10))
    ax.axvline(0, color=(0.8,0.8,0.8), lw=1)
    ellipses = []
    labels = []
    prev_y = -1e9

    for i,c in enumerate(clubs):
        g = dfc[dfc["Type"]==c]
        col = color_map[c]
        ax.scatter(g["Dir_signed"], g["Carry[yd]"], s=30, alpha=0.35, color=col)
        p = ellipse_params_from_points(g[["Dir_signed","Carry[yd]"]].values)
        if not p: continue
        center,w,h,ang = p
        ax.add_patch(Ellipse(center, w, h, angle=ang, fill=False, lw=2, ec=col))
        ellipses.append((center,w,h,ang))

        avg = g["Carry[yd]"].mean()
        # label close to ellipse, left/right as needed
        rx = w/2; ry = h/2
        margin = 0.8
        candidates = [(center[0]+rx+margin, center[1]), (center[0]-rx-margin, center[1])]
        x,y = candidates[0]
        if y < prev_y + 2:
            y = prev_y + 2
        prev_y = y

        face = blend_with_white(col, 0.82)
        txtc = darken(col, 0.72)
        ax.text(x, y, f"{int(round(avg))} yd", ha="left", va="center",
                fontsize=10, color=txtc,
                bbox=dict(boxstyle="round,pad=0.35", facecolor=face, edgecolor="none", alpha=0.9))

    ax.set_xlabel("Dirección (yd)  ← Izq | Der →")
    ax.set_ylabel("Carry (yd)")
    ax.set_title("Virtual Range")
    ax.set_ylim(bottom=0)
    st.pyplot(fig)

# ---------------- Trajectory ----------------
with tabs[1]:
    fig, ax = plt.subplots(figsize=(6,10))
    placed = []
    for c in clubs:
        g = dfc[dfc["Type"]==c]
        d = g["Carry[yd]"].mean()
        h = min(30, max(8, 0.12*d))
        x = np.linspace(0, d, 80)
        y = 4*h*(x/d)*(1-x/d)
        col = color_map[c]
        ax.plot(x, y, lw=2, color=col)
        lx, ly = d*0.5, h+0.8
        for _ in range(10):
            if all(abs(lx-px)>10 or abs(ly-py)>1.2 for px,py in placed):
                break
            ly += 1.0
        placed.append((lx,ly))
        face = blend_with_white(col, 0.84)
        txtc = darken(col, 0.70)
        ax.text(lx, ly, f"{c}  {int(round(d))}yd / {int(round(h))}m",
                ha="center", va="bottom", fontsize=9, color=txtc,
                bbox=dict(boxstyle="round,pad=0.30", facecolor=face, edgecolor="none", alpha=0.9))
    ax.set_xlabel("Carry (yd)")
    ax.set_ylabel("Altura (m)")
    ax.set_title("Trayectoria promedio")
    st.pyplot(fig)

# ---------------- Metrics ----------------
with tabs[2]:
    rows = []
    for c in clubs:
        g = dfc[dfc["Type"]==c]
        rows.append({
            "Palo": c,
            "Carry Avg (yd)": round(g["Carry[yd]"].mean(),1),
            "Shots": len(g)
        })
    st.dataframe(pd.DataFrame(rows))
