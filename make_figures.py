"""Figures for the numerics (called from run_all.py).

House style for the paper:
- every figure is drawn at the text width (6.3 in) and included at 100 %, so the
  fonts below are the printed sizes;
- Computer Modern text and tick labels, matching the LaTeX body;
- fixed semantic colours: the adaptive two-radius algorithm is always blue, the
  single-radius algorithm always orange; ordered parameters (t, K) use one blue ramp
  from light to dark; categorical identities use the validated categorical order;
- all text is in neutral ink, never in a series colour; dotted lines are predicted
  slopes; panels are labelled (a), (b).
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import LogLocator, NullFormatter, NullLocator  # noqa: E402

# Validated categorical order (reference palette, light mode)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
TWO, ONE = C[0], C[1]
# Identities that are not algorithms take the slots after blue/orange, in palette order
OTHER = ["#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
# Ordinal blue ramp, steps 250 .. 700 (lightest step still clears 2:1 on white)
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#a8a69f"

TEXTWIDTH = 4.98  # inches (amsart text width, 360pt)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.7, "axes.labelcolor": INK,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7, "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5, "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.minor.size": 1.6, "ytick.minor.size": 1.6, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.which": "major", "grid.color": GRID, "grid.linewidth": 0.5,
    "lines.linewidth": 1.1, "lines.markersize": 3.2, "lines.markeredgewidth": 0.8,
    "legend.frameon": False, "legend.handlelength": 1.8, "legend.borderaxespad": 0.3,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.dpi": 300, "pdf.fonttype": 42,
})


def _save(fig, folder, name):
    os.makedirs(folder, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(folder, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _loglog_ticks(ax, nx=4, ny=4):
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=nx))
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=ny))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())


def _panel(ax, letter):
    ax.text(-0.02, 1.04, f"({letter})", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, fontweight="bold", color=INK)


def _series(W, c, rf, algo, q):
    pts = sorted((kk[4], v) for kk, v in W.items() if kk[:3] == c and kk[3] == rf and kk[5] == algo)
    return np.array([e for e, _ in pts]), np.array([v["cost"][q] for _, v in pts]), [v for _, v in pts]


def _guide(ax, x, y, slope, shift, color, label=None):
    """Dotted predicted-slope line through the log-centre of (x, y), shifted by `shift` decades.

    The label sits just beyond the small-eps end, which the inverted axis draws at the right.
    """
    lx, ly = np.log10(x), np.log10(y)
    xs = np.array([lx.min(), lx.max()])
    ys = ly.mean() + shift + slope * (xs - lx.mean())
    ax.plot(10 ** xs, 10 ** ys, ls=(0, (1, 1.6)), lw=1.0, color=color, zorder=1)
    if label:
        ax.annotate(label, (10 ** xs[0], 10 ** ys[0]), textcoords="offset points",
                    xytext=(3, 0), ha="left", va="center", fontsize=7.5, color=INK2)


def _eps_axis(ax, label=True):
    ax.invert_xaxis()
    if label:
        ax.set_xlabel(r"tolerance $\varepsilon$")


# ----------------------------------------------------------------------------- Fig. 1

def fig_cost(W, table, cfgs, folder):
    cfgs = [c for c in cfgs if c != (2, 1.5, 0.75)]  # its slack window is too short; kept in Table 1
    fig, axes = plt.subplots(2, len(cfgs), figsize=(TEXTWIDTH, 3.9))
    q = 1
    for j, c in enumerate(cfgs):
        d, s, t = c
        for i, rf in enumerate((0.0, 0.5)):
            ax = axes[i, j]
            e2, c2, v2 = _series(W, c, rf, "two", q)
            e1, c1, _ = _series(W, c, rf, "one", q)
            ax.loglog(e2, c2, "o-", color=TWO, ms=2.2, lw=0.9, label="two radii (adaptive)", zorder=3)
            ax.loglog(e1, c1, "s-", color=ONE, mfc="white", ms=2.2, mew=0.6, lw=0.9,
                      label="single radius", zorder=3)
            if rf == 0.0:
                sel = e2 <= 1e-2
                p2 = (d + 2 * s * q) / (2 * s - t)
                p1 = s * q / (s - t) + d / (2 * s - t)
            else:
                sel = np.array([v["rho"] >= 3 * (v["gap"] - v["rho"]) for v in v2])
                p2 = p1 = (d + s * q) / (s - t)
            if sel.sum() >= 2:
                span = np.log10(c1[sel].max() / c1[sel].min())
                off = 0.12 * span + 0.5
                _guide(ax, e1[sel], c1[sel], -p1, +off, ONE, rf"${p1:.2f}$")
                _guide(ax, e2[sel], c2[sel], -p2, -off, TWO, rf"${p2:.2f}$")
            _eps_axis(ax, label=(i == 1))
            _loglog_ticks(ax, nx=3, ny=4)
            # narrow panels: label only the two end decades of the eps range
            le = np.log10(np.concatenate([e1, e2]))
            ax.set_xticks([10.0 ** np.floor(le.max()), 10.0 ** np.ceil(le.min())])
            ax.xaxis.set_minor_locator(NullLocator())
            ax.margins(x=0.04, y=0.08)
            if i == 0:
                ax.set_title(rf"$(d,s,t)=({d},{s:g},{t:g})$", pad=5)
            if j == 0:
                ax.set_ylabel(("cost, $\\rho=0$" if rf == 0 else "cost, $\\rho=A/2$"))
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(w_pad=0.9, h_pad=1.0, rect=(0, 0, 1, 0.93))
    fig.legend(h, lab, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0), columnspacing=2.0)
    _save(fig, folder, "fig1_cost_vs_eps")


# ----------------------------------------------------------------------------- Fig. 2

def _panel_ratio_t(ax, W, s=1.5, ts=(0.25, 0.5, 0.75), q=2):
    ramp = [RAMP[0], RAMP[1], RAMP[3]]
    for i, t in enumerate(ts):
        g = q * t * s / ((s - t) * (2 * s - t))
        for d, mk, ls, mfc in ((1, "o", "-", ramp[i]), (2, "s", (0, (3, 1.5)), "white")):
            c = (d, s, t)
            e2, c2, _ = _series(W, c, 0.0, "two", q)
            e1, c1, _ = _series(W, c, 0.0, "one", q)
            if len(e2) == 0:
                continue
            ax.loglog(e2, c1 / c2, marker=mk, ls=ls, color=ramp[i], mfc=mfc, ms=3, zorder=3)
        e2, c2, _ = _series(W, (1, s, t), 0.0, "two", q)
        e1, c1, _ = _series(W, (1, s, t), 0.0, "one", q)
        sel = e2 <= 1e-2
        _guide(ax, e2[sel], c1[sel] / c2[sel], -g, +0.6, ramp[i], rf"$t={t:g}$: ${g:.2f}$")
    _eps_axis(ax)
    _loglog_ticks(ax, nx=5, ny=5)
    ax.set_ylabel(r"cost ratio, single / two radii")
    h = [plt.Line2D([], [], color=INK2, marker="o", ls="-", ms=3),
         plt.Line2D([], [], color=INK2, marker="s", mfc="white", ls=(0, (3, 1.5)), ms=3)]
    ax.legend(h, [r"$d=1$", r"$d=2$"], loc="upper left")


def _panel_ratio_rho(ax, Wr, cfgs, rho_ratio, q=2):
    for i, c in enumerate(cfgs):
        d, s, t = c
        gam = t * q / (2 * (s - t))
        rho, r, sel = rho_ratio(Wr, c, q)
        ax.loglog(rho, r, "o-", color=OTHER[i], ms=2.8, label=rf"$({d},{s:g},{t:g})$", zorder=3)
        if sel.sum() >= 2:
            _guide(ax, rho[sel], r[sel], -gam, +0.35, OTHER[i])
    ax.set_xlabel(r"prior slack $\rho$")
    ax.set_ylim(bottom=5e-4)
    _loglog_ticks(ax, nx=5, ny=5)
    ax.set_ylabel(r"cost ratio, single / two radii")
    ax.legend(loc="lower left", ncol=2, columnspacing=1.0)


def fig_penalty(W, Wr, cfgs, rho_ratio, folder):
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.6))
    _panel_ratio_t(axes[0], W)
    _panel_ratio_rho(axes[1], Wr, cfgs, rho_ratio)
    for ax, L in zip(axes, "ab"):
        _panel(ax, L)
    fig.tight_layout(w_pad=1.6)
    _save(fig, folder, "fig2_penalty")


def make_all(W, Wr, table, eps_map, folder, rho_ratio):
    main = [(1, 1.0, 0.5), (1, 2.0, 1.0), (1, 1.5, 0.25), (2, 1.5, 0.75), (2, 2.0, 1.0)]
    fig_cost(W, table, main, folder)
    fig_penalty(W, Wr, main, rho_ratio, folder)


# ----------------------------------------------------------------------------- Fig. 3

def fig_public_radii(tabs, folder, predicted, q=2):
    """Worst ratio over c in [1/2,1] of cost_K / cost_adaptive vs eps, K = 1..4."""
    cfgs = list(tabs)
    fig, axes = plt.subplots(1, len(cfgs), figsize=(TEXTWIDTH, 2.6))
    for ax, c, L in zip(np.atleast_1d(axes), cfgs, "ab"):
        d, s, t = c
        g = predicted(d, s, t, q)["g"]
        tab = tabs[c][q]
        Ks = sorted(tab)
        for i, K in enumerate(Ks):
            col = RAMP[::-1][i] if len(Ks) <= len(RAMP) else C[i]
            E, R, _ = tab[K]
            ax.loglog(E, R, "o-", color=col, ms=2.8, label=rf"$K={K}$", zorder=3)
            sel = E <= 1e-2
            _guide(ax, E[sel], R[sel], -g / K, +0.45, col, rf"${g / K:.2f}$")
        _eps_axis(ax)
        _loglog_ticks(ax, nx=5, ny=5)
        ax.set_title(rf"$(d,s,t)=({d},{s:g},{t:g})$", pad=5)
        ax.legend(loc="upper left")
        _panel(ax, L)
    np.atleast_1d(axes)[0].set_ylabel(r"worst cost ratio, menu / adaptive")
    fig.tight_layout(w_pad=1.6)
    _save(fig, folder, "fig4_public_radii")


# ----------------------------------------------------------------------------- Fig. 4

def fig_harmonic(Wh, names, folder, hrows, q=1):
    """Harmonic recovery: cost vs eps (rho = 0, both traces) and certificate vs error."""
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.6))
    ax = axes[0]
    for i, nm in enumerate(names):
        for algo, mk, col in (("two", "o", TWO), ("one", "s", ONE)):
            pts = sorted((kk[2], v) for kk, v in Wh.items() if kk[0] == nm and kk[1] == 0.0 and kk[3] == algo)
            e = np.array([p[0] for p in pts])
            cst = np.array([p[1]["cost"][q] for p in pts])
            ax.loglog(e, cst, marker=mk, ls="-" if i == 0 else (0, (3, 1.5)), color=col,
                      mfc=col if i == 0 else "white", ms=2.8,
                      label=("two radii" if algo == "two" else "single radius") + rf", $f_{i + 1}$",
                      zorder=3)
            if i == 0:
                sel = e <= 1e-2
                p_ = (4 * q + 2) / 3 if algo == "two" else 2 * q + 2 / 3
                _guide(ax, e[sel], cst[sel], -p_, +1.6 if algo == "one" else -1.6, col, rf"${p_:.2f}$")
    _eps_axis(ax)
    _loglog_ticks(ax, nx=5, ny=6)
    ax.set_ylabel(rf"cost ($q={q}$, $\rho=0$)")
    ax.legend(loc="upper left")
    ax = axes[1]
    ex = [r for r in hrows if r["executed"]]
    # x = error/certificate (valid certificate: x <= 1); y = certificate/eps (tolerance met: y <= 1).
    # Grey = what the two guarantees exclude; every run must land in the white quadrant.
    xlo, xhi, ylo, yhi = 1e-7, 10.0, 0.0, 1.25
    excl = dict(color="#efeeea", lw=0, zorder=0)
    ax.fill_between([xlo, xhi], 1.0, yhi, **excl)
    ax.fill_between([1.0, xhi], ylo, 1.0, **excl)
    ax.axhline(1.0, color=INK2, lw=0.8, zorder=2)
    ax.axvline(1.0, color=INK2, lw=0.8, zorder=2)
    for j, p in enumerate(("centered", "adversarial", "random")):
        rr = [r for r in ex if r["path"] == p]
        ax.semilogx([r["err"] / r["cert"] for r in rr], [r["cert"] / r["eps"] for r in rr], "o",
                    ms=2.2, mew=0, alpha=0.8, color=OTHER[[0, 2, 4][j]], label=p,
                    zorder=3)
    ax.text(3e-7, 1.06, r"certificate $>\varepsilon$ (tolerance missed)", fontsize=7.5,
            color=INK2, va="bottom")
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel(r"true error / certificate")
    ax.set_ylabel(r"certificate / $\varepsilon$")
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
    h, lab = ax.get_legend_handles_labels()
    h.append(plt.Rectangle((0, 0), 1, 1, color="#e1e0d9", lw=0))
    lab.append("excluded")
    ax.legend(h, lab, loc="lower left", markerscale=1.6, handlelength=1.2)
    for ax, L in zip(axes, "ab"):
        _panel(ax, L)
    fig.tight_layout(w_pad=1.6)
    _save(fig, folder, "fig5_harmonic")
