"""Genera las figuras del reporte LaTeX a partir de los resultados del sitio.

Uso (desde la raíz del repo):
    python reporte/generar_figuras.py
Escribe PNG en reporte/images/.
"""
from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "web" / "public" / "data"
MEDIA = ROOT / "web" / "public" / "media"
OUT = Path(__file__).resolve().parent / "images"
OUT.mkdir(parents=True, exist_ok=True)

NAVY, GOLD, SKY, RED = "#0B1F3A", "#E0B800", "#2F7FC1", "#C62828"
BLUE_GOLD = LinearSegmentedColormap.from_list(
    "bg", ["#0b1f3a", "#16365f", "#1f5a96", "#3f8fc9", "#a9c96a", "#ffd200"])
DELTAS = [(-1, 0), (1, 0), (0, 1), (0, -1), (-1, 1), (-1, -1), (1, 1), (1, -1)]

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100, "savefig.dpi": 220,
    "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.25,
})

tab = json.loads((DATA / "tabular.json").read_text())
dqn = json.loads((DATA / "dqn.json").read_text())
nofuel = json.loads((DATA / "dqn_nofuel.json").read_text())
cfg = tab["config"]
MAP = cfg["map"]
OBS = np.array([[c == "#" for c in r] for r in MAP])
START, GOAL = tuple(cfg["start"]), tuple(cfg["goal"])
GAP = [tuple(g) for g in cfg["gap_cells"]]
LABEL = {"qlearning": "Q-learning", "sarsa": "SARSA"}
COLOR = {"qlearning": GOLD, "sarsa": SKY}


def follow(policy, start):
    """Ruta greedy sin resbalón desde A."""
    path = [start]
    r, c = start
    for _ in range(80):
        dr, dc = DELTAS[policy[r][c]]
        r, c = r + dr, c + dc
        if not (0 <= r < 20 and 0 <= c < 20) or OBS[r, c]:
            break
        path.append((r, c))
        if (r, c) == GOAL:
            break
    return path


def draw_board(ax, show_grid=True):
    ax.set_xlim(-0.5, 19.5)
    ax.set_ylim(19.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(0, 20, 5))
    ax.set_yticks(range(0, 20, 5))
    ax.set_xlabel("columna")
    ax.set_ylabel("fila")
    ax.grid(False)
    for r in range(20):
        for c in range(20):
            if OBS[r, c]:
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=NAVY, edgecolor="#ffd200", hatch="xx", lw=0.4))
    for r, c in GAP:
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor=RED, lw=2, ls="--"))
    for (r, c), t, fc in [(START, "A", "#fff6d5"), (GOAL, "B", "#ffd200")]:
        ax.scatter([c], [r], s=230, c=fc, edgecolors=NAVY, linewidths=1.5, zorder=5)
        ax.text(c, r, t, ha="center", va="center", fontweight="bold", color=NAVY, zorder=6)


# 1. Mapa con rutas -----------------------------------------------------------
def fig_mapa():
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    for r in range(20):
        for c in range(20):
            if not OBS[r, c]:
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor="#2a8c48" if (r + c) % 2 else "#24803f", lw=0))
    draw_board(ax)
    for algo, ls in [("qlearning", "-"), ("sarsa", "--")]:
        p = np.array(follow(tab["algos"][algo]["policy_seed0"], START))
        ax.plot(p[:, 1], p[:, 0], ls, color=RED if algo == "qlearning" else "#fff6d5", lw=2.6,
                label=f"{LABEL[algo]} (greedy, semilla 0)", zorder=4)
    ax.annotate("brecha\n(1 celda)", xy=(10, 6), xytext=(12.6, 3.2), color="white", fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="white"))
    ax.text(5, 16.5, "espacio abierto (ruta larga y segura)", color="white", fontsize=8.5, ha="center")
    leg = ax.legend(loc="lower right", facecolor=NAVY, labelcolor="white", framealpha=0.85)
    leg.get_frame().set_edgecolor(GOLD)
    ax.set_title("El Paso del Contrabandista (20×20)")
    fig.savefig(OUT / "mapa.png")
    plt.close(fig)


# 2. Convergencia ---------------------------------------------------------------
def fig_convergencia():
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.2), sharex=True, gridspec_kw={"height_ratios": [1.6, 1]})
    ep = np.arange(1, cfg["episodes"] + 1)
    for algo in ("qlearning", "sarsa"):
        a = tab["algos"][algo]
        m, s = np.array(a["ma100_mean"]), np.array(a["ma100_std"])
        axes[0].fill_between(ep, m - s, m + s, color=COLOR[algo], alpha=0.22, lw=0)
        axes[0].plot(ep, m, color=COLOR[algo], lw=1.8, label=f"{LABEL[algo]} (media ± σ, 5 semillas)")
        axes[1].plot(ep, 100 * np.array(a["crash_ma100"]), color=COLOR[algo], lw=1.4, label=LABEL[algo])
    axes[0].set_ylim(-250, 100)
    axes[0].set_ylabel("Recompensa (media móvil 100)")
    axes[0].legend(loc="lower right")
    axes[0].set_title("Recompensa promedio vs episodios")
    axes[1].set_ylim(0, 40)
    axes[1].set_ylabel("% episodios con choque")
    axes[1].set_xlabel("Episodio")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "convergencia.png")
    plt.close(fig)


# 3. Mapas de calor -------------------------------------------------------------
def fig_heatmaps(key="eval_eps", name="heatmaps.png"):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.0))
    vmax = max(np.log10(1 + np.array(tab["algos"][a]["visits"][key])).max() for a in LABEL)
    for ax, algo in zip(axes, ("qlearning", "sarsa")):
        V = np.log10(1 + np.array(tab["algos"][algo]["visits"][key], dtype=float))
        V = np.ma.masked_where(OBS, V)
        im = ax.imshow(V, cmap=BLUE_GOLD, vmin=0, vmax=vmax, extent=(-0.5, 19.5, 19.5, -0.5))
        draw_board(ax)
        ax.set_title(f"{LABEL[algo]}: {100 * tab['algos'][algo]['eval_eps']['gap_rate']:.0f} % de partidos por la brecha")
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label("log$_{10}$(1 + visitas)")
    fig.savefig(OUT / name)
    plt.close(fig)


# 4. Políticas ------------------------------------------------------------------
def fig_politicas():
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.0))
    for ax, algo in zip(axes, ("qlearning", "sarsa")):
        a = tab["algos"][algo]
        V = np.ma.masked_where(OBS, np.array(a["v_seed0"]))
        im = ax.imshow(V, cmap=BLUE_GOLD, extent=(-0.5, 19.5, 19.5, -0.5))
        P = np.array(a["policy_seed0"])
        X, Y, U, W = [], [], [], []
        for r in range(20):
            for c in range(20):
                if OBS[r, c] or (r, c) == GOAL:
                    continue
                dr, dc = DELTAS[P[r, c]]
                n = np.hypot(dr, dc)
                X.append(c), Y.append(r), U.append(dc / n), W.append(-dr / n)
        ax.quiver(X, Y, U, W, color="white", scale=26, width=0.005, headwidth=4, pivot="middle")
        draw_board(ax)
        ax.set_title(f"{LABEL[algo]}: política greedy sobre $V(s)=\\max_a Q(s,a)$")
        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    fig.savefig(OUT / "politicas.png")
    plt.close(fig)


# 5. Curva DQN con fases --------------------------------------------------------
def rolling(x, w):
    x = np.asarray(x, float)
    c = np.cumsum(np.insert(x, 0, 0))
    return np.array([(c[i + 1] - c[max(0, i + 1 - w)]) / (i + 1 - max(0, i + 1 - w)) for i in range(len(x))])


def dqn_phases(d):
    h = d["history"]
    n = len(h["reward"])
    to = rolling(np.array(h["steps"]) >= 1000, 50)
    la = rolling(np.array(h["landed"], float), 50)
    ma = np.array(h["ma100"])
    b1 = int(np.argmax(to >= 0.1)) + 1
    b2 = int(np.argmax((la >= 0.2) & (np.arange(n) > b1))) + 1
    b3 = int(np.argmax((ma >= 150) & (np.arange(n) > b2))) + 1
    return [(1, b1), (b1, b2), (b2, b3), (b3, n)]


def fig_dqn():
    h = dqn["history"]
    ep = np.arange(1, len(h["reward"]) + 1)
    ph = dqn_phases(dqn)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    names = ["1. Exploración", "2. Aprende el hover", "3. Aterrizajes\ninestables", "4. Convergencia"]
    shades = ["#f8d7da", "#d6e9f8", "#fff3c4", "#d8f3dc"]
    for (a, b), nm, sh in zip(ph, names, shades):
        ax.axvspan(a, b, color=sh, alpha=0.8, lw=0)
        ax.text((a + b) / 2, 318, nm, ha="center", va="top", fontsize=8.5, fontweight="bold")
    ax.scatter(ep, h["reward"], s=6, color=SKY, alpha=0.55, label="Recompensa por episodio")
    ax.plot(ep, h["ma100"], color=GOLD, lw=2.6, label="Media móvil (100)")
    ax.axhline(200, color="#1b7a2c", ls="--", lw=1.3, label="Umbral +200")
    ax.set_ylim(-450, 330)
    ax.set_xlabel("Episodio")
    ax.set_ylabel("Recompensa acumulada")
    ax2 = ax.twinx()
    ax2.plot(ep, h["epsilon"], color=RED, ls=":", lw=1.6, label="ε")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("ε", color=RED)
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    l1, n1 = ax.get_legend_handles_labels()
    l2, n2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, n1 + n2, loc="lower right", ncol=2, framealpha=0.9)
    ax.set_title(f"Curva de aprendizaje DQN ({dqn['env_id']}); paro en el episodio {dqn['solved_at']}")
    fig.savefig(OUT / "dqn_curva.png")
    plt.close(fig)
    return ph


# 6. Combustible ----------------------------------------------------------------
def fig_combustible():
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9), gridspec_kw={"width_ratios": [1.35, 1]})
    axes[0].plot(dqn["history"]["ma100"], color=GOLD, lw=2.2, label="Con costo de combustible")
    axes[0].plot(nofuel["history"]["ma100"], color=SKY, lw=2.2, label="Costo = 0")
    axes[0].axhline(200, color="#1b7a2c", ls="--", lw=1.2)
    axes[0].set_xlabel("Episodio")
    axes[0].set_ylabel("Media móvil (100)")
    axes[0].set_title("Recompensa que optimiza cada agente")
    axes[0].legend(loc="lower right")
    cats = ["Motor\nprincipal", "Propulsores\nlaterales", "Pasos por\nepisodio"]
    a = [dqn["eval"]["mean_main_engine"], dqn["eval"]["mean_side_engine"], dqn["eval"]["mean_steps"]]
    b = [nofuel["eval"]["mean_main_engine"], nofuel["eval"]["mean_side_engine"], nofuel["eval"]["mean_steps"]]
    x = np.arange(3)
    axes[1].bar(x - 0.2, a, 0.4, color=GOLD, label="Con costo")
    axes[1].bar(x + 0.2, b, 0.4, color=SKY, label="Costo = 0")
    for i in range(3):
        axes[1].text(x[i] - 0.2, a[i] + 8, f"{a[i]:.0f}", ha="center", fontsize=8)
        axes[1].text(x[i] + 0.2, b[i] + 8, f"{b[i]:.0f}", ha="center", fontsize=8)
    axes[1].set_xticks(x, cats)
    axes[1].set_title("Política greedy (20 episodios)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "combustible.png")
    plt.close(fig)


# 7. Escalabilidad --------------------------------------------------------------
def fig_escalabilidad():
    params = dqn["config"]["n_params"]
    # bytes = estados x 4 acciones x 4 B (float32)
    vals = [20 ** 8 * 4 * 4, 20 ** 6 * 2 ** 2 * 4 * 4, params * 4]
    labels = ["Tabla Q\n(8 variables × 20)", "Tabla Q\n(patas binarias)", f"Red DQN\n({params:,} parámetros)"]
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    bars = ax.barh(labels, vals, color=[RED, GOLD, SKY])
    ax.set_xscale("log")
    ax.set_xlim(1e4, 4e12)
    ax.set_xlabel("Memoria en float32 (bytes, escala log)")
    for bar, t in zip(bars, ["≈ 410 GB", "≈ 4.1 GB", f"≈ {params * 4 / 1000:.1f} KB"]):
        ax.text(bar.get_width() * 1.3, bar.get_y() + bar.get_height() / 2, t, va="center", fontsize=9)
    ax.invert_yaxis()
    fig.savefig(OUT / "escalabilidad.png")
    plt.close(fig)


# 8. Tira de fotogramas del aterrizaje ------------------------------------------
def fig_aterrizaje():
    r = iio.get_reader(MEDIA / "dqn_landing.mp4")
    n = r.count_frames()
    idx = np.linspace(0, n - 1, 5).astype(int)
    fig, axes = plt.subplots(1, 5, figsize=(12, 2.0))
    for ax, i in zip(axes, idx):
        ax.imshow(r.get_data(int(i))[40:360])
        ax.set_title(f"t = {i / 50:.1f} s", fontsize=9)
        ax.axis("off")
    fig.subplots_adjust(wspace=0.03)
    fig.savefig(OUT / "aterrizaje.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_mapa()
    fig_convergencia()
    fig_heatmaps()
    fig_politicas()
    ph = fig_dqn()
    fig_combustible()
    fig_escalabilidad()
    fig_aterrizaje()
    print("fases DQN:", ph)
    print("figuras en", OUT)
