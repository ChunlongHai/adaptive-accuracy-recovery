#!/usr/bin/env python
"""Single entry point: reproduces every number, table and figure in numerics/.

    python run_all.py            # full run (~ a few minutes)
    python run_all.py --plots    # re-make figures/tables from results/*.json only

Outputs: results/constants.json, results/runs.csv, results/rho_sweep.csv,
results/fits.json, results/fits.md, results/reliability.md, figures/*.pdf|png.
"""
from __future__ import annotations

import csv
import json
import math
import os
import platform
import sys
import time
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torus_recovery as tr  # noqa: E402
import harmonic as hm  # noqa: E402

RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")

MAIN = [(1, 1.0, 0.5), (1, 2.0, 1.0), (1, 1.5, 0.25), (2, 1.5, 0.75), (2, 2.0, 1.0)]
EXTRA_T = [(1, 1.5, 0.5), (1, 1.5, 0.75), (2, 1.5, 0.25), (2, 1.5, 0.5)]  # for the d-independence figure
PATHS = ("centered", "adversarial", "random")
ALGOS = ("two", "one")
QS = (1, 2)
CAP = 2 ** 22                     # max grid points of any executed grid (1D and 2D)
RHO_FRAC = (0.0, 0.01, 0.5)      # rho = frac * A  (0.5 is a supplementary slack-regime run)
EPS_PER_DECADE = 6
EPS_MAX_EXP = 3                   # eps starts at 10^{-3/6}
RHO_SWEEP_EPS = {(1, 1.0, 0.5): 1e-3, (1, 2.0, 1.0): 1e-5, (1, 1.5, 0.25): 1e-5,
                 (2, 1.5, 0.75): 3e-3, (2, 2.0, 1.0): 1e-3}
FIT_EPS_MAX_RHO0 = 1e-2           # pre-asymptotic points eps > this are excluded at rho = 0

# non-polynomial torus input (item: finite smoothness)
POWER_EXTRA = 1.6                 # hat f(k) = c <k>^{-(2s+1.6)}: f in H^sigma iff sigma < 2s + 1.1
POWER_CFGS = [(1, 1.0, 0.5), (1, 2.0, 1.0)]
POWER_RHO = (0.0, 0.5)
POWER_CAP = 2 ** 20

# harmonic recovery (Section 8)
HARM_RHO = (0.0, 0.01, 0.5)
HARM_CAP = 2 ** 20
HARM_FIT_EPS_MAX = 1e-2

# K public radii
KRAD_CFGS = {(1, 1.0, 0.5): 1e-7, (1, 2.0, 1.0): 1e-10}   # cfg -> smallest eps
KRAD_KS = (1, 2, 3, 4)
KRAD_EPS_PER_DECADE = 3
KRAD_RHO_PER_DECADE = 6
KRAD_FINAL_CAP = 2 ** 14


def cfg_name(c):
    d, s, t = c
    return f"d{d}_s{s:g}_t{t:g}"


def predicted(d, s, t, q):
    return dict(
        two_rho0=(d + 2 * s * q) / (2 * s - t),
        slack=(d + s * q) / (s - t),
        one_rho0=s * q / (s - t) + d / (2 * s - t),
        one_slack=s * q / (s - t) + d / (s - t),
        g=q * t * s / ((s - t) * (2 * s - t)),
        gamma=t * q / (2 * (s - t)),
    )


# ----------------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------------

def constants(c):
    d, s, t = c
    mcap = 2 ** 22 if d == 1 else 2 ** 11
    ms, cb, cn = [], [], []
    m = 1
    while m <= mcap:
        ms.append(m)
        cb.append(tr.CB_of_m(m, d, s, t))
        cn.append(tr.CN_of_m(m, d, s, t))
        m *= 2
    CB = max(2 ** (s - t), max(cb))
    CN = max(cn)
    return dict(d=d, s=s, t=t, m=ms, CB_m=cb, CN_m=cn, CB=CB, CN=CN, CB_limit=2 ** (s - t),
                rigorous=(d == 1), m_cap=mcap,
                note=("C_B: secular-equation upper bound with conservative alias lumping; "
                      "C_N: sqrt(T_hi)/S_lo from rigorous lattice-sum enclosures; both for every "
                      "dyadic m <= m_cap" if d == 1 else
                      "C_B: conservative secular-equation bound; C_N: estimate (midpoint + "
                      "Euler-Maclaurin lattice tails, not an enclosure); dyadic m <= m_cap"))


def problem(c, rho_frac, consts, kind="poly"):
    d, s, t = c
    f = tr.default_input(d) if kind == "poly" else tr.PowerInput(2 * s + POWER_EXTRA)
    f = f.scaled(1.0 / math.sqrt(f.norm2(s)))      # normalise E = ||f||_s^2 = 1
    E = 1.0
    rho = rho_frac * E / (1 - rho_frac) if rho_frac > 0 else 0.0   # rho = frac * A
    return tr.Problem(d, s, t, f, rho, consts["CB"], consts["CN"])


def seed_of(*parts):
    return zlib.crc32(json.dumps(parts).encode())


def run_point(P, eps, cfg, tag, cap=None):
    cap = CAP if cap is None else cap
    rows = []
    for algo in ALGOS:
        for path in PATHS:
            r = tr.run(P, eps, algo, path, qs=QS, cap=cap,
                       seed=seed_of(cfg_name(cfg), tag, f"{eps:.6e}", f"{P.rho:.6e}", algo, path))
            rows.append(r)
    return rows


# ----------------------------------------------------------------------------
# experiments
# ----------------------------------------------------------------------------

def eps_scan(c, consts, rho_frac, kind="poly", cap=None):
    cap = CAP if cap is None else cap
    P = problem(c, rho_frac, consts, kind)
    out = []
    j = EPS_MAX_EXP
    while True:
        eps = 10 ** (-j / EPS_PER_DECADE)
        # feasibility first (cheap: no final stage)
        ok = all(tr.run(P, eps, a, p, final=False)["m"] ** c[0] <= cap
                 for a in ALGOS for p in PATHS)
        if not ok or j > 12 * EPS_PER_DECADE:
            break
        for r in run_point(P, eps, c, "eps" if kind == "poly" else kind, cap):
            r["rho_frac"] = rho_frac
            r["input"] = kind
            out.append(r)
        j += 1
    return out


def rho_sweep(c, consts):
    """Penalty vs rho at (nearly) fixed eps.  To smooth the dyadic step structure,
    each rho is run at 4 tolerances eps0 * 10^{j/8}, j=-2..1 (half a decade around
    eps0); the penalty is reported as the geometric mean over these 4 values."""
    eps0 = RHO_SWEEP_EPS[c]
    epss = [eps0 * 10 ** (j / 8) for j in range(-2, 2)]
    out = []
    for jr in range(-28, 1):                 # rho/A from 1e-7 to 0.9, 4 per decade
        frac = min(10 ** (jr / 4), 0.9)
        P = problem(c, frac, consts)
        ok = all(tr.run(P, e, a, p, final=False)["m"] ** c[0] <= CAP
                 for e in epss for a in ALGOS for p in PATHS)
        if not ok:
            continue
        for e in epss:
            for r in run_point(P, e, c, "rho"):
                r["rho_frac"] = frac
                out.append(r)
    return out


def rho_ratio(Wr, c, q):
    """(rho, geometric-mean penalty over the eps window, slack mask) for the rho sweep."""
    fr = sorted({kk[3] for kk in Wr if kk[:3] == c})
    rho, rat, slack = [], [], []
    for f in fr:
        two = {kk[4]: v for kk, v in Wr.items() if kk[:3] == c and kk[3] == f and kk[5] == "two"}
        one = {kk[4]: v for kk, v in Wr.items() if kk[:3] == c and kk[3] == f and kk[5] == "one"}
        lr = [math.log(one[e]["cost"][q] / two[e]["cost"][q]) for e in two]
        v0 = next(iter(two.values()))
        rho.append(v0["rho"])
        rat.append(math.exp(sum(lr) / len(lr)))
        slack.append(all(v["rho"] >= 3 * (v["gap"] - v["rho"]) for v in two.values()))
    return np.array(rho), np.array(rat), np.array(slack, dtype=bool)




# ----------------------------------------------------------------------------
# new experiments: harmonic recovery, K public radii
# ----------------------------------------------------------------------------

def harm_problem(f, rho_frac):
    f = f.scaled(1.0 / math.sqrt(f.energy()))          # E = int|f'|^2 = 1
    rho = rho_frac / (1 - rho_frac) if rho_frac > 0 else 0.0   # rho = frac * A
    return hm.HProblem(f, rho)


def harm_scan(f, rho_frac):
    P = harm_problem(f, rho_frac)
    out = []
    j = EPS_MAX_EXP
    while j <= 12 * EPS_PER_DECADE:
        eps = 10 ** (-j / EPS_PER_DECADE)
        if not all(hm.run(P, eps, a, p, final=False)["m"] <= HARM_CAP for a in ALGOS for p in PATHS):
            break
        for a in ALGOS:
            for p in PATHS:
                r = hm.run(P, eps, a, p, qs=QS, cap=HARM_CAP,
                           seed=seed_of("harm", f.name, f"{eps:.6e}", f"{rho_frac:g}", a, p))
                r["rho_frac"] = rho_frac
                out.append(r)
        j += 1
    return out


def kradii_experiment(c, consts):
    """max over f_c = c f0 (c in [1/2, 1], A = ||f0||_s^2 = 1, rho = 1 - c^2) of
    cost_K / cost_two, for K = 1..4, on a grid of eps."""
    d, s, t = c
    f0 = tr.default_input(d)
    f0 = f0.scaled(1.0 / math.sqrt(f0.norm2(s)))
    eps_min = KRAD_CFGS[c]
    rows = []
    j = 1
    while True:
        eps = 10 ** (-j / KRAD_EPS_PER_DECADE)
        if eps < eps_min * 0.999:
            break
        lo_exp = math.log10(eps ** (2 * s / (2 * s - t))) - 1.0
        nr = int(math.ceil((math.log10(0.75) - lo_exp) * KRAD_RHO_PER_DECADE))
        rhos = [0.0] + [10 ** (lo_exp + i / KRAD_RHO_PER_DECADE) for i in range(nr)] + [0.75]
        for rho in rhos:
            cc = math.sqrt(1.0 - rho)
            P = tr.Problem(d, s, t, f0.scaled(cc), rho, consts["CB"], consts["CN"])
            for algo, K in [("two", 0)] + [("K", K) for K in KRAD_KS]:
                for path in PATHS:
                    r = tr.run(P, eps, algo, path, qs=QS, final=True, cap=KRAD_FINAL_CAP, K=max(K, 1),
                               seed=seed_of("krad", cfg_name(c), f"{eps:.6e}", f"{rho:.6e}", algo, K, path))
                    keep = {kk: r[kk] for kk in ("d", "s", "t", "rho", "eps", "algo", "path", "k", "m",
                                                 "delta", "b", "gap", "pilot_lemma_ok", "executed",
                                                 "cost_q1", "cost_q2")}
                    keep["K"] = K
                    keep["c"] = cc
                    if r["executed"]:
                        keep.update(err=r["err"], cert=r["cert"])
                    rows.append(keep)
        j += 1
    return rows


def kradii_table(rows, c, q):
    """{K: (eps array, max_c ratio array, argmax rho)}"""
    d, s, t = c
    agg = {}
    for r in rows:
        if (r["d"], r["s"], r["t"]) != c:
            continue
        kk = (r["eps"], r["rho"], r["K"])
        agg[kk] = max(agg.get(kk, 0.0), r[f"cost_q{q}"])
    epss = sorted({kk[0] for kk in agg}, reverse=True)
    out = {}
    for K in KRAD_KS:
        E, R, Rho = [], [], []
        for e in epss:
            best, arg = 0.0, None
            for (e2, rho, K2), v in agg.items():
                if e2 == e and K2 == K:
                    rat = v / agg[(e, rho, 0)]
                    if rat > best:
                        best, arg = rat, rho
            E.append(e)
            R.append(best)
            Rho.append(arg)
        out[K] = (np.array(E), np.array(R), Rho)
    return out


# ----------------------------------------------------------------------------
# aggregation and fits
# ----------------------------------------------------------------------------

def worst_cost(rows, key=("d", "s", "t")):
    """cost per (cfg, rho_frac, eps, algo) = max over response paths."""
    agg = {}
    for r in rows:
        kk = tuple(r[x] for x in key) + (r["rho_frac"], r["eps"], r["algo"])
        agg.setdefault(kk, []).append(r)
    out = {}
    for kk, rs in agg.items():
        out[kk] = dict(cost={q: max(x[f"cost_q{q}"] for x in rs) for q in QS},
                       gap=max(x["gap"] for x in rs), rho=rs[0]["rho"],
                       m=max(x["m"] for x in rs), k=rs[0]["k"])
    return out


def fitw(eps, y):
    """Least-squares slope of -log y vs log eps over the given window, reported with
    its window instead of a standard error (dyadic step data have correlated
    residuals).  Also the slopes over the upper and lower halves of the window
    (in log eps) as a stability indicator."""
    e, y = np.asarray(eps, float), np.asarray(y, float)
    if len(e) < 3:
        return None
    x, ly = np.log(e), np.log(y)
    sl = -float(np.polyfit(x, ly, 1)[0])
    mid = 0.5 * (x.min() + x.max())
    halves = []
    for sel in (x >= mid, x <= mid):
        halves.append(-float(np.polyfit(x[sel], ly[sel], 1)[0]) if sel.sum() >= 3 else float("nan"))
    return dict(slope=sl, eps_hi=float(e.max()), eps_lo=float(e.min()), n=int(len(e)),
                decades=float(np.log10(e.max() / e.min())), half_hi=halves[0], half_lo=halves[1])


def series(W, c, rho_frac, algo, q):
    pts = sorted((kk[-2], v) for kk, v in W.items()
                 if kk[:-3] == tuple(c) and kk[-3] == rho_frac and kk[-1] == algo)
    return [e for e, _ in pts], [v["cost"][q] for _, v in pts], [v for _, v in pts]


def slack_mask(vals_two):
    """rho dominates the measured pilot excess: rho >= 3 (gap - rho)."""
    return [v["rho"] > 0 and v["rho"] >= 3 * (v["gap"] - v["rho"]) for v in vals_two]


def fit_block(W, c, rho_frac, q, eps_max=None, slack=False):
    e2, c2, v2 = series(W, c, rho_frac, "two", q)
    e1, c1, _ = series(W, c, rho_frac, "one", q)
    if not e2:
        return {}
    if slack:
        sel = [i for i, b in enumerate(slack_mask(v2)) if b]
    else:
        sel = [i for i, e in enumerate(e2) if e <= eps_max]
    xe = [e2[i] for i in sel]
    return dict(two=fitw(xe, [c2[i] for i in sel]), one=fitw(xe, [c1[i] for i in sel]),
                ratio=fitw(xe, [c1[i] / c2[i] for i in sel]))


def fits(W, Wr):
    table = []
    for c in MAIN + EXTRA_T:
        d, s, t = c
        for q in QS:
            row = dict(cfg=cfg_name(c), d=d, s=s, t=t, q=q, pred=predicted(d, s, t, q))
            row["rho0"] = fit_block(W, c, 0.0, q, eps_max=FIT_EPS_MAX_RHO0)
            for rf in RHO_FRAC[1:]:
                row[f"rho{rf:g}"] = fit_block(W, c, rf, q, slack=True)
            if c in RHO_SWEEP_EPS:
                xs, ys, sl = rho_ratio(Wr, c, q)
                if sl.sum() >= 3:
                    f = fitw(xs[sl], ys[sl])          # slope of -log ratio vs log rho = gamma
                    f["eps0"] = RHO_SWEEP_EPS[c]
                    row["ratio_vs_rho"] = f
            table.append(row)
    return table


def fits_power(Wp):
    out = []
    for c in POWER_CFGS:
        d, s, t = c
        for q in QS:
            row = dict(cfg=cfg_name(c), d=d, s=s, t=t, q=q, pred=predicted(d, s, t, q))
            row["rho0"] = fit_block(Wp, c, 0.0, q, eps_max=FIT_EPS_MAX_RHO0)
            row["rho0.5"] = fit_block(Wp, c, 0.5, q, slack=True)
            out.append(row)
    return out


def harm_pred(q):
    return dict(two_rho0=(4 * q + 2) / 3, one_rho0=2 * q + 2 / 3, g=2 * q / 3, slack=2 * q + 2,
                one_slack=2 * q + 2)


def fits_harm(Wh, names):
    out = []
    for nm in names:
        for q in QS:
            row = dict(f=nm, q=q, pred=harm_pred(q))
            row["rho0"] = fit_block(Wh, (nm,), 0.0, q, eps_max=HARM_FIT_EPS_MAX)
            for rf in HARM_RHO[1:]:
                row[f"rho{rf:g}"] = fit_block(Wh, (nm,), rf, q, slack=True)
            out.append(row)
    return out


def fits_kradii(KR):
    out = []
    for c, rows in KR.items():
        d, s, t = c
        for q in QS:
            g = predicted(d, s, t, q)["g"]
            tab = kradii_table(rows, c, q)
            for K in KRAD_KS:
                E, R, _ = tab[K]
                sel = E <= FIT_EPS_MAX_RHO0
                f = fitw(E[sel], R[sel])             # ratio ~ eps^{-slope}
                out.append(dict(cfg=cfg_name(c), d=d, s=s, t=t, q=q, K=K, pred=g / K, fit=f,
                                ratio_at_min_eps=float(R[sel][-1]), eps_min=float(E[sel][-1])))
    return out


# ----------------------------------------------------------------------------
# tables
# ----------------------------------------------------------------------------

def fw(f, pred=None):
    if not f:
        return "–"
    s = f"{f['slope']:.2f}"
    if pred is not None:
        s += f" / **{pred:.2f}**"
    return s


def win(f):
    if not f:
        return "–"
    return (f"{f['eps_lo']:.0e}–{f['eps_hi']:.0e} ({f['decades']:.1f} dec, n={f['n']}; "
            f"halves {f['half_hi']:.2f}/{f['half_lo']:.2f})")


def write_fit_table(table, ptable, htable, ktable, path):
    L = ["# Fitted exponents (fit / **predicted**)", "",
         "Slopes are least-squares slopes of log cost vs log ε over the stated window. "
         "No standard errors are given: the costs are dyadic step functions with "
         "correlated residuals. Instead each fit lists its window (ε-range, decades, "
         "number of tolerances, 6 per decade) and the slopes over the upper/lower half "
         "of the window as a stability indicator. Cost = max over the three reply paths.", "",
         "## 1. Torus, saturated prior ρ = 0 (ε ≤ 1e-2)", "",
         "| (d,s,t) | q | two-radius | one-radius | penalty g | window (two-radius; halves = penalty) |",
         "|---|---|---|---|---|---|"]
    for r in table:
        p, b = r["pred"], r["rho0"]
        if not b:
            continue
        w = win(b["two"]).split(";")[0] + f"; penalty halves {b['ratio']['half_hi']:.2f}/{b['ratio']['half_lo']:.2f})"
        L.append(f"| ({r['d']},{r['s']:g},{r['t']:g}) | {r['q']} | {fw(b['two'], p['two_rho0'])} | "
                 f"{fw(b['one'], p['one_rho0'])} | {fw(b['ratio'], p['g'])} | {w} |")
    L += ["", "## 2. d-independence of g (s = 1.5, ρ = 0); Fig. 2 shows q = 2", "",
          "| t | pred g (q=1) | d=1 | d=2 | pred g (q=2) | d=1 | d=2 |", "|---|---|---|---|---|---|---|"]
    by = {(r["d"], r["t"], r["q"]): r for r in table if r["s"] == 1.5}
    for t in (0.25, 0.5, 0.75):
        cells = []
        for q in QS:
            g = predicted(1, 1.5, t, q)["g"]
            cells += [f"**{g:.2f}**"] + [f"{by[(d, t, q)]['rho0']['ratio']['slope']:.2f}" for d in (1, 2)]
        L.append(f"| {t:g} | " + " | ".join(cells) + " |")
    for rf, lab in ((0.5, "ρ = A/2 (supplementary)"), (0.01, "ρ = 0.01·A (requested)")):
        L += ["", f"## 3. Torus, slack regime {lab}", "",
              "Only tolerances where the data show the slack regime (ρ ≥ 3·(A−Λ−ρ) on every path) are used. "
              "Predicted: both algorithms (d+sq)/(s−t), penalty exponent 0.", "",
              "| (d,s,t) | q | two-radius | one-radius | penalty exp. (pred 0) | window | note |",
              "|---|---|---|---|---|---|---|"]
        for r in table:
            b = r.get(f"rho{rf:g}") or {}
            if r["cfg"] not in [cfg_name(c) for c in MAIN]:
                continue
            p = r["pred"]
            if not b or not b.get("two"):
                L.append(f"| ({r['d']},{r['s']:g},{r['t']:g}) | {r['q']} | – | – | – | – | "
                         "no slack window inside the grid cap |")
                continue
            note = ""
            if b["two"]["n"] < 6 or b["two"]["decades"] < 1.0:
                note = "insufficient range: not evidence"
            elif abs(b["ratio"]["slope"]) > 0.1:
                note = "pre-asymptotic (transition (ε^{2s/(2s−t)}+ρ)^{−γ} still curved)"
            L.append(f"| ({r['d']},{r['s']:g},{r['t']:g}) | {r['q']} | {fw(b['two'], p['slack'])} | "
                     f"{fw(b['one'], p['one_slack'])} | {b['ratio']['slope']:.2f} | {win(b['two'])} | {note} |")
    L += ["", "## 4. Penalty vs ρ at fixed ε (slope of −log penalty vs log ρ, predicted γ = tq/(2(s−t)))", "",
          "| (d,s,t) | q | fit / **γ** | fraction of γ | ρ-window |", "|---|---|---|---|---|"]
    for r in table:
        f = r.get("ratio_vs_rho")
        if not f:
            continue
        gam = r["pred"]["gamma"]
        L.append(f"| ({r['d']},{r['s']:g},{r['t']:g}) | {r['q']} | {f['slope']:.2f} / **{gam:.2f}** | "
                 f"{f['slope'] / gam:.0%} | {f['eps_lo']:.1e}–{f['eps_hi']:.1e} (n={f['n']}) |")
    L += ["", f"## 5. Non-polynomial torus input  f̂(k) = c⟨k⟩^(−(2s+{POWER_EXTRA:g}))  (f ∈ H^σ iff σ < 2s+{POWER_EXTRA - 0.5:g})", "",
          "| (d,s,t) | q | ρ | two-radius | one-radius | penalty | window |", "|---|---|---|---|---|---|---|"]
    for r in ptable:
        p = r["pred"]
        for key, lab, pt, po, pg in (("rho0", "0", p["two_rho0"], p["one_rho0"], p["g"]),
                                     ("rho0.5", "A/2", p["slack"], p["one_slack"], 0.0)):
            b = r.get(key) or {}
            if not b or not b.get("two"):
                L.append(f"| ({r['d']},{r['s']:g},{r['t']:g}) | {r['q']} | {lab} | – | – | – | no window |")
                continue
            L.append(f"| ({r['d']},{r['s']:g},{r['t']:g}) | {r['q']} | {lab} | {fw(b['two'], pt)} | "
                     f"{fw(b['one'], po)} | {fw(b['ratio'], pg)} | {win(b['two'])} |")
    L += ["", "## 6. Harmonic recovery (Section 8): piecewise-linear algorithm, Dirichlet energy target", "",
          "Predicted at ρ = 0: two-radius (4q+2)/3, one-radius 2q+2/3, penalty 2q/3; slack: both 2q+2, penalty 0.", "",
          "| trace | q | ρ | two-radius | one-radius | penalty | window |", "|---|---|---|---|---|---|---|"]
    for r in htable:
        p = r["pred"]
        for key, lab in (("rho0", "0"), ("rho0.01", "0.01A"), ("rho0.5", "A/2")):
            b = r.get(key) or {}
            pt, po, pg = (p["two_rho0"], p["one_rho0"], p["g"]) if key == "rho0" else (p["slack"], p["one_slack"], 0.0)
            if not b or not b.get("two"):
                L.append(f"| {r['f']} | {r['q']} | {lab} | – | – | – | no slack window inside the grid cap |")
                continue
            L.append(f"| {r['f']} | {r['q']} | {lab} | {fw(b['two'], pt)} | {fw(b['one'], po)} | "
                     f"{fw(b['ratio'], pg)} | {win(b['two'])} |")
    L += ["", "## 7. K public radii: max over c ∈ [1/2,1] of cost_K / cost_adaptive (f_c = c f0, ρ = A(1−c²))", "",
          "Predicted slope g/K (K = 1 is the one-radius algorithm).", "",
          "| (d,s,t) | q | K | fit / **g/K** | max ratio at smallest ε | window |", "|---|---|---|---|---|---|"]
    for r in ktable:
        L.append(f"| ({r['d']},{r['s']:g},{r['t']:g}) | {r['q']} | {r['K']} | {r['fit']['slope']:.2f} / **{r['pred']:.2f}** | "
                 f"{r['ratio_at_min_eps']:.3g} (ε={r['eps_min']:.0e}) | {win(r['fit'])} |")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


def reliability_summary(rows, paths=PATHS):
    ex = [r for r in rows if r.get("executed")]
    bad = [r for r in ex if not (r["err"] <= r["cert"] * (1 + 1e-9))]
    over = [r for r in ex if r["cert"] > r["eps"] * (1 + 1e-9) or r["err"] > r["eps"]]
    bad_parts = [r for r in ex if "err_interp" in r and (r["err_interp"] > r["cert_interp"] * (1 + 1e-9)
                 or r["err_noise"] > r["cert_noise"] * (1 + 1e-9))]
    lemma = [r for r in rows if not r["pilot_lemma_ok"]]
    ratio = np.array([r["cert"] / r["err"] for r in ex if r["err"] > 0])
    per_path = {}
    for p in paths:
        rr = np.array([r["cert"] / r["err"] for r in ex if r["path"] == p and r["err"] > 0])
        ee = np.array([r["err"] / r["eps"] for r in ex if r["path"] == p])
        if len(rr):
            per_path[p] = dict(n=int(len(rr)), min_cert_over_err=float(rr.min()),
                               median_cert_over_err=float(np.median(rr)),
                               max_err_over_eps=float(ee.max()))
    out = dict(n_runs=len(rows), n_executed=len(ex), n_cert_violations=len(bad),
               n_part_violations=len(bad_parts), n_over_eps=len(over),
               n_pilot_lemma_violations=len(lemma),
               min_cert_over_err=float(ratio.min()) if len(ratio) else None,
               median_cert_over_err=float(np.median(ratio)) if len(ratio) else None,
               per_path=per_path)
    taus = [r["tau"] for r in rows if r.get("tau") is not None and r["tau"] == r["tau"]]
    if taus:
        rel = [r["tau"] / (r["k"] ** (-2 * r["s"]) / max(1.0, r["A"])) for r in rows
               if r.get("tau") is not None and r["tau"] == r["tau"]]
        out["enclosure"] = dict(n_pilots_enclosed=len(taus), max_rel_width=float(max(taus)),
                                max_width_over_k_minus_2s=float(max(rel)),
                                max_Lambda_shift=float(max(r["enc"] for r in rows if r.get("tau") == r.get("tau") and "enc" in r)))
    if any("const_ok" in r for r in ex):
        out["n_constant_check_failures"] = sum(1 for r in ex if not r["const_ok"])
    return out


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def _write_csv(path, data):
    keys = sorted({kk for r in data for kk in r})
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in data:
            w.writerow(r)


def main_compute():
    os.makedirs(RES, exist_ok=True)
    t0 = time.time()
    allc, rows, rrows, prows = {}, [], [], []
    for c in MAIN + EXTRA_T:
        tc = time.time()
        tr.clear_caches()
        k = constants(c)
        allc[cfg_name(c)] = k
        rfs = RHO_FRAC if c in MAIN else (0.0,)
        for rf in rfs:
            rows += eps_scan(c, k, rf)
        if c in RHO_SWEEP_EPS:
            rrows += rho_sweep(c, k)
        if c in POWER_CFGS:
            for rf in POWER_RHO:
                prows += eps_scan(c, k, rf, kind="power", cap=POWER_CAP)
        print(f"{cfg_name(c)}: C_B={k['CB']:.6f} C_N={k['CN']:.6f}  [{time.time() - tc:.1f}s]", flush=True)
    tc = time.time()
    krad = {}
    for c in KRAD_CFGS:
        tr.clear_caches()
        krad[cfg_name(c)] = kradii_experiment(c, allc[cfg_name(c)])
    print(f"K public radii: [{time.time() - tc:.1f}s]", flush=True)
    tr.clear_caches()
    tc = time.time()
    hrows = []
    for f in hm.default_traces():
        for rf in HARM_RHO:
            hrows += harm_scan(f, rf)
    print(f"harmonic: [{time.time() - tc:.1f}s]", flush=True)
    allc["harmonic"] = dict(CB=hm.CB_H, CN=hm.CN_H, coth_pi=hm.COTH_PI,
                            note="certificate C_B m^{-1/2}(A-Lambda)^{1/2} + C_N (sum_i eta_i^2)^{1/2}; "
                                 "C_B = sqrt(coth(pi)/pi), C_N = sqrt(2 coth(pi)) (analytic)")
    with open(os.path.join(RES, "constants.json"), "w") as fh:
        json.dump(allc, fh, indent=1)
    _write_csv(os.path.join(RES, "runs.csv"), rows)
    _write_csv(os.path.join(RES, "rho_sweep.csv"), rrows)
    _write_csv(os.path.join(RES, "power_runs.csv"), prows)
    _write_csv(os.path.join(RES, "harmonic_runs.csv"), hrows)
    _write_csv(os.path.join(RES, "public_radii_runs.csv"), [r for v in krad.values() for r in v])
    with open(os.path.join(RES, "runs.json"), "w") as fh:
        json.dump(dict(runs=rows, rho_sweep=rrows, power=prows, harmonic=hrows, public_radii=krad), fh)
    env = dict(python=sys.version.split()[0], numpy=np.__version__,
               platform=platform.platform(), runtime_s=round(time.time() - t0, 1))
    import matplotlib
    env["matplotlib"] = matplotlib.__version__
    with open(os.path.join(RES, "environment.json"), "w") as fh:
        json.dump(env, fh, indent=1)
    print(f"compute done in {time.time() - t0:.1f}s")


def main_analyse():
    with open(os.path.join(RES, "runs.json")) as fh:
        J = json.load(fh)
    rows, rrows, prows, hrows = J["runs"], J["rho_sweep"], J["power"], J["harmonic"]
    krad = {}
    for name, v in J["public_radii"].items():
        c = next(cc for cc in KRAD_CFGS if cfg_name(cc) == name)
        krad[c] = v
    W, Wr, Wp = worst_cost(rows), worst_cost(rrows), worst_cost(prows)
    Wh = worst_cost(hrows, key=("f",))
    names = [f.name for f in hm.default_traces()]
    table, ptable = fits(W, Wr), fits_power(Wp)
    htable, ktable = fits_harm(Wh, names), fits_kradii(krad)
    with open(os.path.join(RES, "fits.json"), "w") as fh:
        json.dump(dict(torus=table, power=ptable, harmonic=htable, public_radii=ktable), fh, indent=1)
    write_fit_table(table, ptable, htable, ktable, os.path.join(RES, "fits.md"))
    rel = dict(eps_scan=reliability_summary(rows), rho_sweep=reliability_summary(rrows),
               eps_scan_1d=reliability_summary([r for r in rows if r["d"] == 1]),
               eps_scan_2d=reliability_summary([r for r in rows if r["d"] == 2]),
               power_input=reliability_summary(prows), harmonic=reliability_summary(hrows),
               public_radii=reliability_summary([r for v in krad.values() for r in v]))
    with open(os.path.join(RES, "reliability.json"), "w") as fh:
        json.dump(rel, fh, indent=1)
    import make_figures
    make_figures.make_all(W, Wr, table, RHO_SWEEP_EPS, FIG, rho_ratio)
    make_figures.fig_public_radii({c: {q: kradii_table(krad[c], c, q) for q in QS} for c in krad},
                                  FIG, predicted)
    make_figures.fig_harmonic(Wh, names, FIG, hrows)
    print(json.dumps(rel, indent=1))
    print(open(os.path.join(RES, "fits.md")).read())


if __name__ == "__main__":
    if "--plots" not in sys.argv:
        main_compute()
    main_analyse()
