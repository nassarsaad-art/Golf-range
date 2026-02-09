import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.patches import Ellipse

# ============================
# Page setup (mobile-first)
# ============================
st.set_page_config(page_title="Virtual Range", page_icon="⛳", layout="wide")

# Mobile-first CSS: full-width, remove extra padding, harmonious palette, readable contrasts.
# Keep Streamlit header (iOS needs it for sidebar toggle).
st.markdown(
    """
<style>
:root{
  --bg: #E9EDF2;
  --card: #F7F8FA;
  --top1: #2B3646;
  --top2: #1F2937;
  --text: rgba(0,0,0,0.82);
  --muted: rgba(0,0,0,0.55);
  --line: rgba(0,0,0,0.10);
  --shadow: 0 18px 36px rgba(0,0,0,0.16);
  --side1: #1F2937;
  --side2: #111827;
  --sideText: rgba(255,255,255,0.92);
  --sideMuted: rgba(255,255,255,0.70);
  --accent: #E25555;
}

/* App background */
.stApp{ background: var(--bg); }

/* Reduce page padding so plot can go bigger */
.block-container{
  padding-top: 0.35rem;
  padding-bottom: 0.35rem;
  padding-left: 0.55rem;
  padding-right: 0.55rem;
  max-width: 1400px;
}

/* Sidebar */
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, var(--side1) 0%, var(--side2) 100%);
  border-right: 1px solid rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] * { color: var(--sideText) !important; }
section[data-testid="stSidebar"] label { color: rgba(255,255,255,0.82) !important; }
section[data-testid="stSidebar"] .stMarkdown { color: var(--sideText) !important; }
section[data-testid="stSidebar"] div[data-testid="stFileUploader"]{
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.14);
  border-radius: 14px;
  padding: 10px 10px;
}
section[data-testid="stSidebar"] input, section[data-testid="stSidebar"] textarea{
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
  border-radius: 14px !important;
}

/* Shell */
.vr-shell{
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 18px;
  overflow: hidden;
  box-shadow: var(--shadow);
}
.vr-topbar{
  height: 56px;
  background: linear-gradient(180deg, var(--top1) 0%, var(--top2) 100%);
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
.vr-session{
  height: 38px;
  display:flex; align-items:center; justify-content:center;
  background: var(--card);
  border-bottom: 1px solid rgba(0,0,0,0.08);
  color: var(--muted);
  font-size: 12px;
}
.vr-plotpad{
  padding: 6px 8px 8px 8px;
}

/* Tabs styling: closer to your mockups */
div[data-baseweb="tab-list"]{
  gap: 10px;
  background: transparent;
}
div[data-baseweb="tab-list"] button{
  font-weight: 900;
  color: rgba(0,0,0,0.55);
}
div[data-baseweb="tab-list"] button[aria-selected="true"]{
  color: rgba(0,0,0,0.85);
}
/* Streamlit draws a red underline sometimes; harmonize it */
div[data-baseweb="tab-highlight"]{ background: var(--accent) !important; }

/* Make the plot take as much vertical room as possible on phones */
@media (max-width: 768px){
  .block-container{ padding-left: 0.35rem; padding-right: 0.35rem; }
  .vr-plotpad{ padding: 4px 6px 6px 6px; }
}

/* Remove extra blank space under pyplot (Streamlit adds some margins) */
div[data-testid="stVerticalBlock"]{ gap: 0.4rem; }
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
    """
    Pick a label position that is:
    - just OUTSIDE its own ellipse (always outside)
    - close to the ellipse
    - not overlapping points, other labels, or other ellipses
    """
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

    def outside_own_ellipse(p):
        # Must be strictly outside (with a small pad) so the full chip isn't on top of the ellipse
        return not point_inside_ellipse(p, center, width, height, angle_deg, pad=0.8)

    for deg in candidates:
        t = np.deg2rad(deg)
        local = np.array([(a + pad_x) * np.cos(t), (b + pad_y) * np.sin(t)])
        p = (np.array(center) + R @ local).astype(float)

        if not in_bounds(p): continue
        if not outside_own_ellipse(p): continue
        if not far_from_points(p): continue
        if not far_from_labels(p): continue
        if not not_inside_other_ellipses(p): continue

        ha = "left" if deg in (0, 30, -30) else ("right" if deg in (180, 150, -150) else "center")
        va = "bottom" if deg in (90, 30, 150) else ("top" if deg in (-90, -30, -150) else "center")
        return p, ha, va

    # Fallback: right side outside
    p = np.array([center[0] + a + pad_x, center[1]])
    # ensure outside
    if not outside_own_ellipse(p):
        p = np.array([center[0] - a - pad_x, center[1]])
    p[0] = min(max(p[0], xlim[0] + 1), xlim[1] - 1)
    p[1] = min(max(p[1], ylim[0] + 1), ylim[1] - 1)
    return p, "left", "center"


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4))

def rgb_to_hex(rgb):
    return "#{:02X}{:02X}{:02X}".format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

def blend_with_white(hex_color, alpha):
    """alpha=0 -> original, alpha=1 -> white"""
    r,g,b = hex_to_rgb(hex_color)
    r2 = r*(1-alpha) + 1.0*alpha
    g2 = g*(1-alpha) + 1.0*alpha
    b2 = b*(1-alpha) + 1.0*alpha
    return (r2,g2,b2)

def darken(hex_color, factor):
    """factor in (0,1): 0.8 means 20% darker"""
    r,g,b = hex_to_rgb(hex_color)
    return (r*factor, g*factor, b*factor)

def build_style_maps(clubs):
    """Deterministic unique color/marker per club (for consistent visuals across tabs)."""
    base_palette = [
        "#2563EB", "#16A34A", "#EF4444", "#06B6D4", "#8B5CF6", "#F59E0B",
        "#0F766E", "#EC4899", "#10B981", "#F97316", "#84CC16", "#A855F7",
        "#14B8A6", "#DC2626", "#1D4ED8", "#65A30D"
    ]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "8"]

    clubs_unique = list(dict.fromkeys(list(clubs)))
    color_map = {c: base_palette[i % len(base_palette)] for i, c in enumerate(clubs_unique)}
    marker_map = {c: markers[i % len(markers)] for i, c in enumerate(clubs_unique)}
    return color_map, marker_map

def detect_height_column(df: pd.DataFrame):
    """Try common apex/height column names."""
    candidates = [
        "Height[m]", "Height (m)", "Height", "Apex Height", "Apex Height[m]",
        "Peak Height", "Peak Height[m]", "Max Height", "Max Height[m]",
        "Apex", "Apex[m]", "Max Height (m)"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        lc = str(c).lower()
        if ("height" in lc) or ("apex" in lc) or ("peak" in lc):
            return c
    return None

def plot_flight_profiles(df_core, clubs, color_map, marker_map, height_col, session_label: str, portrait: bool = True):
    """
    Side view: X=carry (yd), Y=height (m).
    We approximate the mean flight with a parabola that peaks at mid-carry:
        y(x) = 4*h*(x/d)*(1-x/d), with y(0)=0, y(d)=0, peak=h at x=d/2.
    """
    fig = plt.figure(figsize=(7.2, 10.0) if portrait else (12.0, 6.2))
    ax = plt.gca()

    ax.set_facecolor("#F7F8FA")
    arc_gray = (0.20, 0.22, 0.25, 0.42)   # same family as arc labels
    grid_gray = (0.20, 0.22, 0.25, 0.12)  # very light grid
    for spine in ax.spines.values():
        spine.set_color(grid_gray)

    # Build curves
    max_d = 0.0
    max_h = 0.0
    placed = []  # (x,y) for apex label anchor

    for c in clubs:
        sub = df_core[df_core["Type"] == c].copy()
        if sub.empty:
            continue
        d = float(sub["Carry[yd]"].mean())
        h = float(pd.to_numeric(sub[height_col], errors="coerce").dropna().mean())

        if not np.isfinite(d) or not np.isfinite(h) or d <= 0 or h <= 0:
            continue

        max_d = max(max_d, d)
        max_h = max(max_h, h)

        x = np.linspace(0, d, 140)
        y = 4.0*h*(x/d)*(1.0 - x/d)

        col = color_map.get(c, "#111827")
        ax.plot(x, y, lw=2.6, color=col, alpha=0.90, zorder=3)
        # Apex marker + label
        ax.scatter([d/2], [h], s=42, color=col, zorder=4, edgecolors=(1,1,1,0.6), linewidths=0.6)
        # Tinted chip label (same logic as Virtual Range), avoid overlaps
        face = blend_with_white(col, 0.84)
        text_col = darken(col, 0.70)

        lx = float(d/2)
        ly = float(h + max(0.6, 0.04*h))

        # If labels are too close, bump this one up a bit (repeat if needed)
        for _ in range(12):
            collide = False
            for (px, py) in placed:
                if abs(lx - px) < 18 and abs(ly - py) < 1.6:
                    collide = True
                    break
            if not collide:
                break
            ly += 1.0

        placed.append((lx, ly))

        ax.text(lx, ly, f"{c}  {int(round(d))}yd / {int(round(h))}m",
                fontsize=9, color=text_col,
                ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.30", facecolor=face, edgecolor="none", alpha=0.90),
                zorder=5)

    if max_d <= 0:
        ax.text(0.5, 0.5, "No hay datos suficientes para dibujar trayectorias.",
                transform=ax.transAxes, ha="center", va="center", color=(0,0,0,0.55))
        return fig

    # Axes limits + grid
    ax.set_xlim(0, max_d*1.08)
    ax.set_ylim(0, max_h*1.25)
    # Match the subtle gray used by arc labels in the Virtual Range
    ax.grid(False)
    ax.tick_params(axis="both", colors=arc_gray)
    ax.xaxis.label.set_color(arc_gray)
    ax.yaxis.label.set_color(arc_gray)
    ax.set_xlabel("Carry (yd)")
    ax.set_ylabel("Altura (m)")
    ax.set_title("Trayectoria promedio (Carry vs Altura)", fontsize=12, color=(0,0,0,0.60), pad=10)
    # Put session label inside the plot area (top-left) to avoid overlapping with the title
    ax.text(0.01, 0.99, session_label, transform=ax.transAxes,
            ha="left", va="top", fontsize=9, color=(0,0,0,0.42))

    return fig

# ============================
# Plot styling (portrait-friendly)
# ============================
def add_background(ax, xlim, ylim):
    W, H = 900, 620
    img = np.zeros((H, W, 3), dtype=float)

    top = np.array([0.96, 0.97, 0.98])
    mid = np.array([0.93, 0.94, 0.96])
    grass = np.array([0.78, 0.86, 0.78])

    split = int(H * 0.56)
    for y in range(H):
        if y < split:
            tt = y / max(1, split - 1)
            col = top * (1 - tt) + mid * tt
        else:
            tt = (y - split) / max(1, (H - split - 1))
            col = mid * (1 - tt) + grass * tt
        img[y, :, :] = col

    ax.imshow(img, extent=[xlim[0], xlim[1], ylim[0], ylim[1]], aspect="auto", zorder=0)

    # converging turf lines
    for i in range(-8, 9):
        ax.plot([i*5.0, i*1.8], [ylim[0], ylim[0] + (ylim[1]-ylim[0])*0.52],
                color=(0.60, 0.72, 0.62, 0.20), lw=1.0, zorder=1)

    # subtle horizontals
    for y in np.linspace(ylim[0] + 12, ylim[0] + (ylim[1]-ylim[0])*0.55, 7):
        ax.plot([xlim[0], xlim[1]], [y, y], color=(0.20,0.22,0.25,0.08), lw=1.0, zorder=1)

def plot_virtual_range(df, clubs, session_label: str, portrait: bool = True):
    # Portrait-ish figure: bigger height than width so on iPhone it fills vertical space nicely
    fig = plt.figure(figsize=(7.2, 10.0) if portrait else (12.0, 6.2))
    ax = plt.gca()

    xlim = (-40, 40)
    max_y = int(df["Carry[yd]"].max()//20*20 + 20) if len(df) else 220
    ylim = (0, max_y + 10)

    add_background(ax, xlim, ylim)

    # Arcs (like launch monitors)
    theta = np.linspace(-np.pi/2, np.pi/2, 400)
    arc_color = (0.20, 0.22, 0.25, 0.14)
    for r in range(20, max_y + 1, 20):
        xs = r*np.sin(theta)
        ys = r*np.cos(theta)
        ax.plot(xs, ys, color=arc_color, lw=1.0, zorder=2)

        # Label each arc on the LEFT edge, just above the arc
        x_lab = xlim[0] + 1.5
        # y on arc at that x (approx)
        # arc equation: y = sqrt(r^2 - x^2) for |x|<=r
        if abs(x_lab) < r:
            y_lab = float(np.sqrt(max(r*r - x_lab*x_lab, 0.0))) + 1.4
            if ylim[0] <= y_lab <= ylim[1]:
                ax.text(x_lab, y_lab, f"{r}",
                        fontsize=8, color=(0.20,0.22,0.25,0.42),
                        ha="left", va="bottom", zorder=4)

    # Center line
    ax.axvline(0, color=(0.12, 0.13, 0.15, 0.35), lw=1.1, zorder=3)

    # Session label
    ax.text(0, ylim[1] - 6, session_label, fontsize=9, color=(0.10,0.11,0.12,0.55),
            ha="center", va="top", zorder=4)

        # Ensure unique colors per club (shared palette)
    clubs_unique = list(dict.fromkeys(clubs))  # preserve order
    color_map, marker_map = build_style_maps(clubs_unique)

    all_points_xy = df[["Dir_signed","Carry[yd]"]].values.astype(float) if len(df) else np.zeros((0,2))

    # Pass 1: draw points + ellipses, and collect label intents
    ellipses_drawn = []
    label_intents = []  # each: dict with avg_carry, txt, color, ellipse params
    for c in clubs_unique:
        sub = df[df["Type"] == c]
        if sub.empty:
            continue

        col = color_map[c]
        mk = marker_map[c]

        ax.scatter(sub["Dir_signed"], sub["Carry[yd]"],
                   s=28, marker=mk, color=col, alpha=0.35,
                   edgecolors=(1,1,1,0.35), linewidths=0.5, zorder=5)

        if len(sub) >= 3:
            params = ellipse_params_from_points(sub[["Dir_signed","Carry[yd]"]].values.astype(float))
            if params:
                center, w, h, ang = params
                ax.add_patch(Ellipse(center, w, h, angle=ang, fill=False, linewidth=2.2,
                                     edgecolor=col, alpha=0.90, zorder=6))
                ellipses_drawn.append((center, w, h, ang))

                avg_carry = float(sub["Carry[yd]"].mean())
                txt = f"{int(round(avg_carry))} yd"
                label_intents.append(dict(
                    club=c, avg=avg_carry, txt=txt, color=col,
                    center=center, w=w, h=h, ang=ang
                ))

    # Pass 2: place labels OUTSIDE and in ASCENDING order (by carry),
    # aligned as a clean right-side column (like launch monitors).
    labels_placed = []
    label_intents.sort(key=lambda d: d["avg"])  # ascending carry

    # Right column X position (inside plot bounds, but outside every ellipse in practice)
    col_x = xlim[1] - 1.2
    y_step = 3.2  # minimum vertical separation between labels (yd)

    prev_y = -1e9
    for d in label_intents:
        center, w, h, ang = d["center"], d["w"], d["h"], d["ang"]
        txt, col = d["txt"], d["color"]
        cx, cy = float(center[0]), float(center[1])
        rx = float(w/2.0)

        # Place at the same "height" as the ellipse, but keep carry hierarchy and avoid overlaps
        ly = cy
        ly = max(ly, prev_y + y_step)

        # Clamp into bounds
        ly = float(min(max(ly, ylim[0] + 2.0), ylim[1] - 2.0))

        # Ensure label is outside its own ellipse (very conservative check)
        # If somehow inside (extremely wide ellipse), push the label further right by a small margin.
        lx = float(col_x)
        if point_inside_ellipse((lx, ly), center, w, h, ang, pad=1.0):
            lx = float(min(xlim[1] - 0.6, cx + rx + 3.0))

        # Tinted chip label (no border)
        face = blend_with_white(col, 0.82)
        text_col = darken(col, 0.72)

        ax.text(lx, ly, txt,
                fontsize=10, ha="right", va="center", color=text_col,
                bbox=dict(boxstyle="round,pad=0.38", facecolor=face, edgecolor="none", alpha=0.90),
                zorder=7)

        labels_placed.append((lx, ly))
        prev_y = ly

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.patch.set_facecolor("#E9EDF2")
    return fig

# ============================
# Sidebar controls (ALL controls live here)
# ============================
with st.sidebar:
    st.markdown("## 🛡️ Virtual Range")

    st.markdown("### Datos")
    input_mode = st.radio("Entrada", ["Subir archivo", "Pegar CSV"], index=0)

    uploaded = None
    pasted = ""
    if input_mode == "Subir archivo":
        uploaded = st.file_uploader("CSV", type="csv", accept_multiple_files=False)
    else:
        pasted = st.text_area("Pega el CSV", height=180, placeholder="Copia desde Golfboy y pega aquí…")

    st.markdown("---")
    st.markdown("### Clubs")
    compare = st.toggle("Compare", value=False)

    st.markdown("### CORE SHOTS")
    keep_pct = st.slider("", 0.50, 0.95, 0.70, 0.05)
    st.caption("Golpes más agrupados (Mahalanobis 2D por palo).")

    st.markdown("---")
    st.markdown("### Session")
    session_placeholder = st.empty()

# ============================
# Load data
# ============================
df = None
if input_mode == "Subir archivo":
    if uploaded is not None:
        df = pd.read_csv(uploaded)
else:
    if pasted and len(pasted.strip()) >= 50:
        df = pd.read_csv(io.StringIO(pasted.strip().lstrip("\ufeff")))

# Main shell (only viewer + tabs, no extra messages)
st.markdown('<div class="vr-shell">', unsafe_allow_html=True)
st.markdown(
    """
    <div class="vr-topbar">
      <div class="left">🛡️ Virtual Range</div>
      <div class="right">＋ &nbsp; 🔍 &nbsp; 👤</div>
    </div>
    """,
    unsafe_allow_html=True
)

tabs = st.tabs(["Virtual Range", "Trayectoria", "Métricas"])

if df is None:
    # Minimal empty state: keep UI clean; no big callouts.
    st.markdown('<div class="vr-session">Sube o pega un CSV desde el sidebar</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

required = {"Carry[yd]", "Launch Direction", "Type"}
if not required.issubset(df.columns):
    st.error("CSV inválido: faltan columnas requeridas (Carry[yd], Launch Direction, Type).")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# Date filter (sidebar)
dates_used = []
dt_col = detect_datetime_column(df)
if dt_col:
    df["_dt"] = pd.to_datetime(df[dt_col], errors="coerce")
    df["_date"] = df["_dt"].dt.date
    dates = sorted(df["_date"].dropna().unique())
    if len(dates) > 1:
        with session_placeholder.container():
            n_dates = st.slider("Últimas fechas", 1, len(dates), min(len(dates), 3), 1, key="n_dates")
        dates_used = list(dates[-n_dates:])
        df = df[df["_date"].isin(dates_used)].copy()
    elif len(dates) == 1:
        with session_placeholder.container():
            st.write(f"Fecha: {dates[0]}")
        dates_used = [dates[0]]
        df = df[df["_date"] == dates[0]].copy()
else:
    with session_placeholder.container():
        st.write("Sin fecha en el CSV")

# Clean + core filter
df["Dir_signed"] = df["Launch Direction"].apply(parse_direction)
df = df.dropna(subset=["Carry[yd]", "Dir_signed", "Type"]).copy()
if df.empty:
    st.error("No quedaron golpes válidos.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

df_core = mahalanobis_filter(df, keep_pct)

clubs_all = sorted(df_core["Type"].unique())
if not clubs_all:
    st.error("No hay datos luego del filtro CORE.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# Club selection (sidebar only)
with st.sidebar:
    st.markdown("### Palos")

    # Initialize checkbox states for each club
    for c in clubs_all:
        k = f"club__{c}"
        if k not in st.session_state:
            st.session_state[k] = False

    # Track previous state to detect toggles
    if "_sel_all_prev" not in st.session_state:
        st.session_state["_sel_all_prev"] = False
    if "_sel_none_prev" not in st.session_state:
        st.session_state["_sel_none_prev"] = False

    sel_all = st.checkbox("Seleccionar todos", key="_sel_all")
    sel_none = st.checkbox("Seleccionar ninguno", key="_sel_none")

    # If user toggles Select All ON -> set all clubs ON and turn off Select None
    if sel_all and not st.session_state["_sel_all_prev"]:
        for c in clubs_all:
            st.session_state[f"club__{c}"] = True
        st.session_state["_sel_none"] = False

    # If user toggles Select None ON -> set all clubs OFF and turn off Select All
    if sel_none and not st.session_state["_sel_none_prev"]:
        for c in clubs_all:
            st.session_state[f"club__{c}"] = False
        st.session_state["_sel_all"] = False

    st.session_state["_sel_all_prev"] = bool(st.session_state.get("_sel_all", False))
    st.session_state["_sel_none_prev"] = bool(st.session_state.get("_sel_none", False))

    st.markdown("")

    # One checkbox per club (always shown)
    for c in clubs_all:
        st.checkbox(c, key=f"club__{c}")

    clubs_plot = [c for c in clubs_all if st.session_state.get(f"club__{c}", False)]
    # Permitir selección vacía: si no hay palos marcados, el Virtual Range se muestra vacío.

# Session label


n_dates_show = len(dates_used) if dates_used else 1
session_label = f"Session: Last {n_dates_show} dates  ·  Core: {int(round(keep_pct*100))}%  ·  Cali"
st.markdown(f'<div class="vr-session">{session_label}</div>', unsafe_allow_html=True)

# Portrait on phones
is_mobile = st.session_state.get("_is_mobile", None)
# Heuristic: Streamlit doesn't give UA; so default to portrait (works well on phones and acceptable on desktop).
portrait = True

# ============================
# Tab 1: Virtual Range (maximize space)
# ============================
with tabs[0]:
    st.markdown('<div class="vr-plotpad">', unsafe_allow_html=True)
    fig = plot_virtual_range(df_core[df_core["Type"].isin(clubs_plot)], clubs_plot, session_label=session_label, portrait=portrait)
    st.pyplot(fig, clear_figure=True, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ============================
# Tab 2: Trayectoria (Carry vs Altura)
# ============================
with tabs[1]:
    if not clubs_plot:
        st.markdown('<div class="vr-session">Sin palos seleccionados</div>', unsafe_allow_html=True)
    else:

        height_col = detect_height_column(df_core)
        if height_col is None:
            st.warning("Este CSV no trae una columna de altura/apex (Height/Apex).")
        else:
            # Consistent styles
            color_map, marker_map = build_style_maps(clubs_plot)
            fig2 = plot_flight_profiles(
                df_core[df_core["Type"].isin(clubs_plot)],
                clubs_plot,
                color_map,
                marker_map,
                height_col=height_col,
                session_label=session_label,
                portrait=portrait
            )
            st.pyplot(fig2, clear_figure=True, use_container_width=True)

    # ============================
    # Tab 2: Metrics (table)
    # ============================with tabs[2]:
    rows = []
    for c in clubs_all:
        sub = df_core[df_core["Type"] == c].copy()
        if sub.empty:
            continue
        shots = int(len(sub))
        carry_avg = float(sub["Carry[yd]"].mean())
        carry_med = float(np.median(sub["Carry[yd]"].values))
        lr_p84 = float(np.quantile(np.abs(sub["Dir_signed"].values), 0.84)) if shots else np.nan
        depth_p84 = float(np.quantile(np.abs(sub["Carry[yd]"].values - carry_med), 0.84)) if shots else np.nan

        disp = "Tight"
        score = lr_p84 + 0.6 * depth_p84
        if score > 14: disp = "Wide"
        elif score > 10: disp = "OK"

        rows.append({
            "Palo": c,
            "Carry Avg (yd)": int(round(carry_avg)),
            "Shots (core)": shots,
            "L/R ± (yd) (p84)": int(round(lr_p84)) if not np.isnan(lr_p84) else None,
            "Depth ± (yd) (p84)": int(round(depth_p84)) if not np.isnan(depth_p84) else None,
            "Dispersion": disp,
        })
    t = pd.DataFrame(rows).sort_values(by="Carry Avg (yd)", ascending=False)
    st.dataframe(t, use_container_width=True, hide_index=True)

st.markdown("</div>", unsafe_allow_html=True)
