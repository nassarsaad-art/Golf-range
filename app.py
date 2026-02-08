import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.patches import Ellipse

st.set_page_config(page_title="Virtual Range", layout="wide")
st.title("Virtual Range (Carry vs Dirección)")

def parse_direction(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, str):
        v = val.strip().upper()
        if v.startswith("L"):
            return -float(v[1:]) if v[1:].replace('.', '', 1).isdigit() else np.nan
        if v.startswith("R"):
            return float(v[1:]) if v[1:].replace('.', '', 1).isdigit() else np.nan
        try:
            return float(v)
        except:
            return np.nan
    try:
        return float(val)
    except:
        return np.nan

def mahalanobis_filter(df, keep_pct):
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

def add_ellipse(ax, x, y, color):
    xy = np.column_stack([x, y]).astype(float)
    if xy.shape[0] < 3:
        return
    c, A = mvee(xy)
    shape = np.linalg.pinv(A)
    eigvals, eigvecs = np.linalg.eigh(shape)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    a = np.sqrt(eigvals[0])
    b = np.sqrt(eigvals[1])
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    ax.add_patch(Ellipse(c, 2*a, 2*b, angle=angle, fill=False, linewidth=2, edgecolor=color))

def plot_range(df, clubs, title, draw_oval):
    fig = plt.figure(figsize=(7,10))
    ax = plt.gca()
    max_y = int(df["Carry[yd]"].max()//20*20 + 20) if len(df) else 200
    theta = np.linspace(-np.pi/2, np.pi/2, 400)
    for r in range(20, max_y+1, 20):
        ax.plot(r*np.sin(theta), r*np.cos(theta), color="lightgray", lw=0.8)
        ax.text(0, r+1, f"{r} yd", ha="center", fontsize=8, color="gray")
    markers = {"3W":"o","5H":"s","7I":"^","9I":"D","AW":"v","LW":"x"}
    colors = {"3W":"blue","5H":"green","7I":"orange","9I":"red","AW":"purple","LW":"brown"}
    for c in clubs:
        sub = df[df["Type"]==c]
        if sub.empty: continue
        col = colors.get(c,"black")
        ax.scatter(sub["Dir_signed"], sub["Carry[yd]"], marker=markers.get(c,"o"),
                   color=col, alpha=0.8, label=f"{c} (n={len(sub)})")
        if draw_oval:
            add_ellipse(ax, sub["Dir_signed"].values, sub["Carry[yd]"].values, col)
    ax.axvline(0,color="black")
    ax.set_xlim(-40,40)
    ax.set_ylim(0,max_y+10)
    ax.set_xlabel("Dirección (yd)")
    ax.set_ylabel("Carry (yd)")
    ax.set_title(title)
    ax.legend()
    ax.grid(False)
    st.pyplot(fig, clear_figure=True)

def detect_datetime_column(df):
    for c in df.columns:
        if any(k in c.lower() for k in ["time","date"]):
            return c
    return None

st.sidebar.header("Controles")
use_filter = st.sidebar.checkbox("Mostrar solo golpes más agrupados", True)
keep_pct = st.sidebar.slider("Porcentaje a conservar", 0.5, 0.95, 0.7, 0.05)
draw_oval = st.sidebar.checkbox("Mostrar óvalo", True)

st.header("Carga de datos")
modo = st.radio("Entrada", ["Pegar CSV (texto)", "Subir archivo"], index=0)
if modo=="Pegar CSV (texto)":
    txt = st.text_area("Pega el CSV completo", height=260)
    if not txt or len(txt.strip())<50: st.stop()
    df = pd.read_csv(io.StringIO(txt.strip().lstrip("\ufeff")))
else:
    f = st.file_uploader("Sube CSV", type="csv")
    if not f: st.stop()
    df = pd.read_csv(f)

required = {"Carry[yd]","Launch Direction","Type"}
if not required.issubset(df.columns):
    st.error("CSV inválido")
    st.stop()

dt_col = detect_datetime_column(df)
if dt_col:
    df["_dt"] = pd.to_datetime(df[dt_col], errors="coerce")
    df["_date"] = df["_dt"].dt.date
    dates = sorted(df["_date"].dropna().unique())
    if len(dates)>1:
        n = st.sidebar.slider("Últimas fechas", 1, len(dates), len(dates), 1)
        df = df[df["_date"].isin(dates[-n:])]

df["Dir_signed"] = df["Launch Direction"].apply(parse_direction)
df = df.dropna(subset=["Carry[yd]","Dir_signed","Type"])

if use_filter:
    df = mahalanobis_filter(df, keep_pct)

clubs = sorted(df["Type"].unique())
if not clubs:
    st.error("No hay datos")
    st.stop()

mode = st.sidebar.radio("Vista", ["Un palo","Varios palos"], index=0)
if mode=="Un palo":
    clubs_plot = [st.sidebar.selectbox("Palo", clubs)]
else:
    clubs_plot = st.sidebar.multiselect("Palos", clubs, default=clubs)

title = f"Virtual Range – core {int(keep_pct*100)}%"
plot_range(df, clubs_plot, title, draw_oval)
