import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Virtual Range", layout="wide")
st.title("Virtual Range (Carry vs Dirección)")

# ---------- Helpers ----------
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

def remove_outliers_per_club(df: pd.DataFrame) -> pd.DataFrame:
    """IQR outliers per club on Carry + Smash."""
    def filt(g):
        # If too few points, don't filter
        if len(g) < 8:
            return g

        q1c, q3c = g["Carry[yd]"].quantile(0.25), g["Carry[yd]"].quantile(0.75)
        iqrc = q3c - q1c
        q1s, q3s = g["Smash Factor"].quantile(0.25), g["Smash Factor"].quantile(0.75)
        iqrs = q3s - q1s

        m = (
            (g["Carry[yd]"] >= q1c - 1.5 * iqrc) &
            (g["Carry[yd]"] <= q3c + 1.5 * iqrc) &
            (g["Smash Factor"] >= q1s - 1.5 * iqrs) &
            (g["Smash Factor"] <= q3s + 1.5 * iqrs)
        )
        return g[m]

    return df.groupby("Type", group_keys=False).apply(filt)

def plot_virtual_range(df: pd.DataFrame, clubs: list[str], title: str):
    fig = plt.figure(figsize=(7, 10))

    # Arcos (cada 20 yd)
    max_y = int(df["Carry[yd]"].max() // 20 * 20 + 20) if len(df) else 200
    theta = np.linspace(-np.pi/2, np.pi/2, 400)
    for r in range(20, max_y + 1, 20):
        x_arc = r * np.sin(theta)
        y_arc = r * np.cos(theta)
        plt.plot(x_arc, y_arc, color="lightgray", linewidth=0.8)
        plt.text(0, r + 1, f"{r} yd", color="gray", ha="center", fontsize=8)

    # Estilos por palo
    markers = {"3W":"o","5H":"s","6I":"P","7I":"^","8I":"X","9I":"D","AW":"v","LW":"x"}
    colors  = {"3W":"blue","5H":"green","6I":"gray","7I":"orange","8I":"teal","9I":"red","AW":"purple","LW":"brown"}

    for c in clubs:
        sub = df[df["Type"] == c]
        if len(sub) == 0:
            continue
        plt.scatter(
            sub["Dir_signed"],
            sub["Carry[yd]"],
            marker=markers.get(c, "o"),
            color=colors.get(c, "black"),
            alpha=0.8,
            label=f"{c} (n={len(sub)})"
        )

    plt.axvline(0, color="black", linewidth=1)
    plt.xlim(-40, 40)
    plt.ylim(0, max_y + 10)
    plt.xlabel("Dirección (yd)  ← Izq | Der →")
    plt.ylabel("Carry (yd)")
    plt.title(title)
    plt.legend()
    plt.grid(False)

    st.pyplot(fig, clear_figure=True)

# ---------- UI ----------
st.sidebar.header("Carga y filtros")

uploaded = st.sidebar.file_uploader("Sube tu CSV de Golfboy", type=["csv"])

if not uploaded:
    st.info("Sube un CSV para ver el range.")
    st.stop()

df = pd.read_csv(uploaded)

required_cols = {"Carry[yd]", "Launch Direction", "Smash Factor", "Type"}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"El CSV no tiene estas columnas requeridas: {sorted(list(missing))}")
    st.stop()

df["Dir_signed"] = df["Launch Direction"].apply(parse_direction)
df = df.dropna(subset=["Carry[yd]", "Dir_signed", "Type"])

include_outliers = st.sidebar.checkbox("Incluir outliers", value=True)

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
    df_plot = remove_outliers_per_club(df_plot)

title = "Virtual Range (incluye outliers)" if include_outliers else "Virtual Range (outliers excluidos)"
plot_virtual_range(df_plot, clubs_to_plot, title)

st.caption("Dirección firmada: L negativo, R positivo. Eje X es 'Launch Direction' convertido a número.")
