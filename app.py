import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.patches import Ellipse

st.set_page_config(page_title="Virtual Range", layout="wide")
st.title("Virtual Range (Carry vs Dirección)")

# ---------------- Helpers ----------------
def parse_direction(val):
    """Convert Launch Direction like 'L12'/'R8' to signed float (-12/+8)."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, str):
        v = val.strip().upper()
        if v.startswith("L"):
            try:
                return -float(v[1:])
            except:
                return np.nan
        if v.startswith("R"):
            try:
                return float(v[1:])
            except:
                return np.nan
        try:
            return float(v)
        except:
            return np.nan
    try:
        return float(val)
    except:
        return np.nan

def remove_outliers_per_club(df: pd.DataFrame, sigma: float) -> pd.DataFrame:
    """Outliers por palo usando ±sigma desviaciones estándar en Carry y Smash."""
    def filt(g):
        if len(g) < 6:
            return g

        c_mean = g["Carry[yd]"].mean()
        c_std  = g["Carry[yd]"].std(ddof=0)

        s_mean = g["Smash Factor"].mean()
        s_std  = g["Smash Factor"].std(ddof=0)

        if c_std == 0:
            c_ok = np.ones(len(g), dtype=bool)
        else:
            c_ok = (g["Carry[yd]"] >= c_mean - sigma * c_std) & (g["Carry[yd]"] <= c_mean + sigma * c_std)

        if s_std == 0:
            s_ok = np.ones(len(g), dtype=bool)
        else:
            s_ok = (g["Smash Factor"] >= s_mean - sigma * s_std) & (g["Smash Factor"] <= s_mean + sigma * s_std)

        return g[c_ok & s_ok]

    return df.groupby("Type", group_keys=False).apply(filt)

def _k_from_mode(mode: str) -> float:
    # sqrt(chi2.ppf(p, df=2)) hardcode (sin scipy)
    if mode.startswith("68%"):
        return 1.51   # ~68%
    if mode.startswith("90%"):
        return 2.15   # ~90%
    if mode.startswith("95%"):
        return 2.45   # ~95%
    return np.nan

def add_ellipse(ax, x, y, mode: str, color: str):
    """Confidence ellipse centrada (68/90/95%) o 'encierra todo'."""
    xy = np.column_stack([x, y]).astype(float)
    if xy.shape[0] < 3:
        return

    mu = xy.mean(axis=0)
    cov = np.cov(xy.T)

    if np.linalg.det(cov) <= 1e-12:
        return

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

    if mode == "Encierra todo":
        inv_cov = np.linalg.inv(cov)
        dif = xy - mu
        d2 = np.einsum("...i,ij,...j->...", dif, inv_cov, dif)
        k = float(np.sqrt(np.max(d2)))
    else:
        k = _k_from_mode(mode)

    a = k * np.sqrt(eigvals[0])
    b = k * np.sqrt(eigvals[1])

    e = Ellipse(
        xy=mu,
        width=2 * a,
        height=2 * b,
        angle=angle,
        fill=False,
        linewidth=2,
        edgecolor=color
    )
    ax.add_patch(e)

def plot_virtual_range(df: pd.DataFrame, clubs: list[str], title: str, draw_oval: bool, oval_mode: str):
    fig = plt.figure(figsize=(7, 10))
    ax = plt.gca()

    # Range arcs (every 20 yd)
    max_y = int(df["Carry[yd]"].max() // 20 * 20 + 20) if len(df) else 200
    theta = np.linspace(-np.pi/2, np.pi/2, 400)
    for r in range(20, max_y + 1, 20):
        x_arc = r * np.sin(theta)
        y_arc = r * np.cos(theta)
        ax.plot(x_arc, y_arc, color="lightgray", linewidth=0.8)
        ax.text(0, r + 1, f"{r} yd", color="gray", ha="center", fontsize=8)

    markers = {"3W":"o","5H":"s","6I":"P","7I":"^","8I":"X","9I":"D","AW":"v","LW":"x"}
    colors  = {"3W":"blue","5H":"green","6I":"gray","7I":"orange","8I":"teal","9I":"red","AW":"purple","LW":"brown"}

    for c in clubs:
        sub = df[df["Type"] == c]
        if len(sub) == 0:
            continue

        col = colors.get(c, "black")
        ax.scatter(
            sub["Dir_signed"], sub["Carry[yd]"],
            marker=markers.get(c, "o"),
            color=col,
            alpha=0.8,
            label=f"{c} (n={len(sub)})"
        )

        if draw_oval:
            add_ellipse(ax, sub["Dir_signed"].values, sub["Carry[yd]"].values, oval_mode, col)

    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlim(-40, 40)
    ax.set_ylim(0, max_y + 10)
    ax.set_xlabel("Dirección (yd)  ← Izq | Der →")
    ax.set_ylabel("Carry (yd)")
    ax.set_title(title)
    ax.legend()
    ax.grid(False)

    st.pyplot(fig, clear_figure=True)

# ---------------- UI ----------------
st.sidebar.header("Controles")

include_outliers = st.sidebar.checkbox("Incluir outliers", value=True)
sigma = st.sidebar.slider("Filtro outliers: ±σ", 0.5, 2.5, 1.0, 0.1)

draw_oval = st.sidebar.checkbox("Mostrar óvalo", value=True)
oval_mode = st.sidebar.selectbox("Tipo de óvalo", ["68% (centrado)", "90%", "95%", "Encierra todo"], index=0)

st.header("Carga de datos")

modo = st.radio("Entrada", ["Pegar CSV (texto)", "Subir archivo"], index=0)

if modo == "Pegar CSV (texto)":
    csv_text = st.text_area(
        "Pega aquí el CSV completo (incluye headers).",
        height=260,
        placeholder="Time,Total[yd],Carry[yd],Height[m],Smash Factor,Club Speed[km/h],Ball Speed[km/h],Launch Angle,Launch Direction,Type"
    )
    if not csv_text or len(csv_text.strip()) < 50:
        st.info("Pega el CSV para continuar.")
        st.stop()
    csv_text = csv_text.strip().lstrip("\ufeff")
    df = pd.read_csv(io.StringIO(csv_text))
else:
    uploaded = st.file_uploader("Sube tu CSV de Golfboy", type=["csv"])
    if not uploaded:
        st.info("Sube un CSV para continuar.")
        st.stop()
    df = pd.read_csv(uploaded)

required_cols = {"Carry[yd]", "Launch Direction", "Smash Factor", "Type"}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"El CSV no tiene estas columnas requeridas: {sorted(list(missing))}")
    st.stop()

df["Dir_signed"] = df["Launch Direction"].apply(parse_direction)
df = df.dropna(subset=["Carry[yd]", "Dir_signed", "Type"])

available_clubs = sorted(df["Type"].dropna().unique().tolist())

mode = st.sidebar.radio("Qué mostrar", ["Un palo", "Varios palos"], index=0)
if mode == "Un palo":
    club = st.sidebar.selectbox("Palo", available_clubs, index=0)
    clubs_to_plot = [club]
else:
    default_pick = [c for c in ["3W","5H","7I","9I","AW","LW"] if c in available_clubs]
    clubs_to_plot = st.sidebar.multiselect("Palos", available_clubs, default=default_pick or available_clubs)

df_plot = df.copy()
if not include_outliers:
    df_plot = remove_outliers_per_club(df_plot, sigma)

title = "Virtual Range (incluye outliers)" if include_outliers else f"Virtual Range (outliers excluidos ±{sigma:.1f}σ)"
plot_virtual_range(df_plot, clubs_to_plot, title, draw_oval, oval_mode)

st.caption("Dirección firmada: L negativo, R positivo. Eje X usa 'Launch Direction' convertido a número.")
