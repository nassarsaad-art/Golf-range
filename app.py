import io
import math
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

def mvee(P: np.ndarray, tol: float = 1e-4, max_iter: int = 5000):
    """Minimum Volume Enclosing Ellipse (MVEE) via Khachiyan algorithm."""
    P = P.astype(float)
    n, d = P.shape

    if n < 2:
        c = P.mean(axis=0)
        A = np.eye(d)
        return c, A

    Q = np.column_stack([P, np.ones(n)]).T  # (d+1) x n
    u = np.ones(n) / n

    for _ in range(max_iter):
        X = Q @ np.diag(u) @ Q.T
        try:
            X_inv = np.linalg.inv(X)
        except np.linalg.LinAlgError:
            X_inv = np.linalg.inv(X + 1e-9 * np.eye(X.shape[0]))

        M = np.einsum("ij,jk,ki->i", Q.T, X_inv, Q)  # diag(Q^T X^-1 Q)
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
    try:
        A = np.linalg.inv(S) / d
    except np.linalg.LinAlgError:
        A = np.linalg.pinv(S) / d
    return c, A

def add_tight_enclosing_ellipse(ax, x, y, color: str):
    """Draw MVEE (tight ellipse that encloses all points)."""
    xy = np.column_stack([x, y]).astype(float)
    if xy.shape[0] < 3:
        return

    c, A = mvee(xy)

    try:
        shape = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        shape = np.linalg.pinv(A)

    eigvals, eigvecs = np.linalg.eigh(shape)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    a = float(np.sqrt(eigvals[0]))
    b = float(np.sqrt(eigvals[1]))
    angle = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))

    e = Ellipse(
        xy=c,
        width=2 * a,
        height=2 * b,
        angle=angle,
        fill=False,
        linewidth=2,
        edgecolor=color
    )
    ax.add_patch(e)

def plot_virtual_range(df: pd.DataFrame, clubs: list[str], title: str, draw_oval: bool):
    fig = plt.figure(figsize=(7, 10))
    ax = plt.gca()

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
            add_tight_enclosing_ellipse(ax, sub["Dir_signed"].values, sub["Carry[yd]"].values, col)

    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlim(-40, 40)
    ax.set_ylim(0, max_y + 10)
    ax.set_xlabel("Dirección (yd)  ← Izq | Der →")
    ax.set_ylabel("Carry (yd)")
    ax.set_title(title)
    ax.legend()
    ax.grid(False)

    st.pyplot(fig, clear_figure=True)

def detect_datetime_column(df: pd.DataFrame) -> str | None:
    # Prefer explicit names if present
    preferred = ["Time", "Date", "Datetime", "Timestamp"]
    for c in preferred:
        if c in df.columns:
            return c

    # Fallback: any column containing date/time keywords
    for c in df.columns:
        lc = str(c).lower()
        if "time" in lc or "date" in lc:
            return c
    return None

# ---------------- UI ----------------
st.sidebar.header("Controles")

include_outliers = st.sidebar.checkbox("Incluir outliers", value=True)
sigma = st.sidebar.slider("Filtro outliers: ±σ", 0.5, 2.5, 1.0, 0.1)
draw_oval = st.sidebar.checkbox("Mostrar óvalo (ajustado)", value=True)

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

# ---- Date filter (by most recent unique dates) ----
dt_col = detect_datetime_column(df)
if dt_col:
    dt = pd.to_datetime(df[dt_col], errors="coerce")
    df["_dt"] = dt
    df["_date"] = df["_dt"].dt.date

    unique_dates = [d for d in sorted(df["_date"].dropna().unique())]
    if len(unique_dates) >= 1:
        n = st.sidebar.slider("¿Cuántas fechas incluir (más recientes)?", 1, len(unique_dates), len(unique_dates), 1)
        selected_dates = set(unique_dates[-n:])
        df = df[df["_date"].isin(selected_dates)].copy()
        with st.expander("Fechas incluidas", expanded=False):
            st.write(list(unique_dates[-n:]))
else:
    st.sidebar.caption("No se detectó columna de fecha/hora en el CSV.")

# ---- Parse direction + clean ----
df["Dir_signed"] = df["Launch Direction"].apply(parse_direction)
df = df.dropna(subset=["Carry[yd]", "Dir_signed", "Type"])

available_clubs = sorted(df["Type"].dropna().unique().tolist())
if len(available_clubs) == 0:
    st.error("Con el filtro actual no quedaron golpes para graficar.")
    st.stop()

view_mode = st.sidebar.radio("Qué mostrar", ["Un palo", "Varios palos"], index=0)
if view_mode == "Un palo":
    club = st.sidebar.selectbox("Palo", available_clubs, index=0)
    clubs_to_plot = [club]
else:
    default_pick = [c for c in ["3W","5H","7I","9I","AW","LW"] if c in available_clubs]
    clubs_to_plot = st.sidebar.multiselect("Palos", available_clubs, default=default_pick or available_clubs)

df_plot = df.copy()
if not include_outliers:
    df_plot = remove_outliers_per_club(df_plot, sigma)

title = "Virtual Range (incluye outliers)" if include_outliers else f"Virtual Range (outliers excluidos ±{sigma:.1f}σ)"
plot_virtual_range(df_plot, clubs_to_plot, title, draw_oval)

st.caption("Dirección firmada: L negativo, R positivo. Eje X usa 'Launch Direction' convertido a número.")
