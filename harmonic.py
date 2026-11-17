"""Harmonic recovery from noisy Dirichlet data (Section 8 of the paper, Corollary 8.2).

Setting.  Omega = (0,1)^2, data on the bottom edge, zero Dirichlet data on the other
three edges.  For a trace v = sum_j b_j sqrt2 sin(pi j x) on [0,1],
    target   ||H v||_V^2 = int_Omega |grad H v|^2 = sum_j pi j coth(pi j) b_j^2,
    prior    int_0^1 |v'|^2 = sum_j pi^2 j^2 b_j^2 <= A,
    ||v||_*^2 = sum_j pi j b_j^2  (so ||v||_*^2 <= ||Hv||_V^2 <= coth(pi) ||v||_*^2).

Algorithm (piecewise-linear interpolation I_m on x_i = i/m, y_0 = y_m = 0 known).
    pilot      k = least power of 2 with k^{-3/2} <= eps, delta = k^{-2}
               Lambda = max{0, k sum_{i<k} (y_{i+1}-y_i)^2 - 2 delta k sum_{0<i<k} |2y_i - y_{i-1} - y_{i+1}|}
    refine     m = least power of 2 >= k with  C_B m^{-1/2} (A - Lambda)^{1/2} <= eps/2
               b = eps / (2 C_N sqrt(m-1));  query x_1..x_{m-1} at radius b;  output H I_m y.
    certificate ||H v - H I_m y||_V <= C_B m^{-1/2} (A - Lambda)^{1/2} + C_N (sum_i eta_i^2)^{1/2}
with the explicit constants (derivation in README):
    C_B = sqrt(coth(pi)/pi)  [ ||g||_*^2 <= ||g|| ||g'|| <= (h/pi) ||g'||^2 for g = v - I_m v,
                               cellwise Poincare;  ||g'||^2 = int|v'|^2 - E_m(v) <= A - Lambda ]
    C_N = sqrt(2 coth(pi))   [ z = I_m e:  ||z||^2 <= h sum e_i^2, ||z'||^2 <= (4/h) sum e_i^2 ]
One-radius variant: m_max from A, mbar = max{k, m_max},
eta_1 = min{k^{-2}, eps / (2 C_N sqrt(mbar - 1))}  (C_N sqrt(m-1) increases in m).

Everything the algorithm uses is a finite computation (no lattice sums): the
certificate is rigorous up to floating-point rounding.

The true error is evaluated in closed form from sine coefficients (no quadrature):
for f = c1 x(1-x) + sum_p a_p sin(p pi x) and nodal errors e,
    beta_j = b_j(f - I_m f) - b_j(I_m e)
is known exactly for every j; for j >= J (J a multiple of 2m) it has the form
alpha(rho)/j^2 + beta(rho)/j^3 in each residue class rho = j mod 2m, and the tail
is summed with Hurwitz zeta functions.  Low-j coefficients use cancellation-free
formulas (series for sin t - t cos t and 1 - sinc^2).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from torus_recovery import hurwitz, price

COTH_PI = 1.0 / math.tanh(math.pi)
CB_H = math.sqrt(COTH_PI / math.pi)
CN_H = math.sqrt(2.0 * COTH_PI)
SQ2 = math.sqrt(2.0)


# ----------------------------------------------------------------------------
# stable elementary functions
# ----------------------------------------------------------------------------

def _s_minus_tc(th):
    """sin(th) - th cos(th), cancellation-free for small th."""
    th = np.asarray(th, dtype=float)
    out = np.sin(th) - th * np.cos(th)
    sm = np.abs(th) < 0.1
    t = th[sm]
    t2 = t * t
    out[sm] = t ** 3 * (1 / 3 - t2 * (1 / 30 - t2 * (1 / 840 - t2 * (1 / 45360 - t2 / 3991680))))
    return out


def _one_minus_sinc2(th):
    """1 - (sin th / th)^2, cancellation-free for small th."""
    th = float(th)
    if abs(th) < 0.1:
        t2 = th * th
        tms = th ** 3 * (1 / 6 - t2 * (1 / 120 - t2 * (1 / 5040 - t2 / 362880)))   # th - sin th
    else:
        tms = th - math.sin(th)
    return tms * (th + math.sin(th)) / (th * th)


# ----------------------------------------------------------------------------
# input
# ----------------------------------------------------------------------------

@dataclass
class Trace:
    """f(x) = c1 x(1-x) + sum_p a_p sin(p pi x) on [0,1]; f(0) = f(1) = 0,
    f''(0) = -2 c1 - 0 (sine part has zero 2nd derivative at 0), so f''(0) != 0 iff c1 != 0.
    The odd extension has a jump of L f = -f'' at the reflection points."""
    c1: float
    sines: dict = field(default_factory=dict)
    name: str = ""

    def nodal(self, m):
        x = np.arange(1, m) / m
        v = self.c1 * x * (1 - x)
        for p, a in self.sines.items():
            v = v + a * np.sin(p * math.pi * x)
        return v

    def energy(self):
        """int_0^1 |f'|^2 (closed form)."""
        c1 = self.c1
        E = c1 * c1 / 3
        for p, a in self.sines.items():
            E += a * a * p * p * math.pi ** 2 / 2
            E += 4 * c1 * a * (1 - (-1) ** p) / (p * math.pi)
        return E

    def scaled(self, c):
        return Trace(self.c1 * c, {p: a * c for p, a in self.sines.items()}, self.name)


def default_traces():
    return [Trace(1.0, {1: 1.0}, "sin(pi x) + x(1-x)"),
            Trace(1.0, {2: 0.4}, "x(1-x) + 0.4 sin(2 pi x)")]


# ----------------------------------------------------------------------------
# exact sine-coefficient norms of  f - I_m f - I_m e
# ----------------------------------------------------------------------------

def _dst_classes(e, m):
    """E_rho = sum_{i=1}^{m-1} e_i sin(rho pi i / m), rho = 0..2m-1."""
    pad = np.zeros(2 * m)
    pad[1:m] = e
    return -np.imag(np.fft.fft(pad))


def error_norms(f: Trace, m: int, e, with_f=True):
    """Sine-scale norms of g = (f - I_m f) - I_m e  (with_f=False: g = -I_m e).
    Returns dict(harm=sum pi j coth(pi j) b_j^2, star=sum pi j b_j^2,
                 energy=sum pi^2 j^2 b_j^2, l2=sum b_j^2)."""
    h = 1.0 / m
    L0 = max(1, -(-64 // (2 * m)))
    J = 2 * m * L0
    j = np.arange(1, J, dtype=float)
    th = j * math.pi / (2 * m)
    sinc2 = (np.sin(th) / th) ** 2
    beta = np.zeros_like(j)
    rho_all = np.arange(2 * m)
    phi = rho_all * math.pi / (2 * m)
    alpha_r = np.zeros(2 * m)
    beta_r = np.zeros(2 * m)
    jm = (j.astype(np.int64)) % (2 * m)
    if with_f:
        # bump train  c1 [x(1-x) - I_m x(1-x)]  (odd j only)
        odd = (j.astype(np.int64) % 2) == 1
        beta[odd] += f.c1 * 4 * SQ2 * _s_minus_tc(th[odd]) / ((j[odd] * math.pi) ** 3 * np.sin(th[odd]))
        oddr = (rho_all % 2) == 1
        beta_r[oddr] += f.c1 * 4 * SQ2 / math.pi ** 3
        with np.errstate(divide="ignore"):
            alpha_r[oddr] += -f.c1 * 2 * SQ2 / (m * math.pi ** 2) / np.tan(phi[oddr])
        # sine modes  a_p [sin - I_m sin]
        for p, a in f.sines.items():
            rp, rn = p % (2 * m), (-p) % (2 * m)
            sgn = np.zeros(2 * m)
            sgn[rp] += 1.0
            sgn[rn] -= 1.0
            Ipart = a / SQ2 * sinc2 * sgn[jm]
            beta -= Ipart
            if p < J:
                if sgn[rp] == 1.0:      # j = p: a/sqrt2 (1 - sinc^2), cancellation-free
                    beta[p - 1] = beta[p - 1] + Ipart[p - 1] + a / SQ2 * _one_minus_sinc2(th[p - 1])
                else:
                    beta[p - 1] += a / SQ2
            alpha_r += -a / SQ2 * (2 * m / math.pi) ** 2 * np.sin(phi) ** 2 * sgn
    # noise part  -I_m e
    if e is not None and np.any(e):
        Er = _dst_classes(e, m)
        beta -= SQ2 * h * sinc2 * Er[jm]
        alpha_r += -SQ2 * h * (2 * m / math.pi) ** 2 * np.sin(phi) ** 2 * Er
    b2 = beta * beta
    coth = 1.0 / np.tanh(math.pi * j)
    out = dict(harm=float(np.sum(math.pi * j * coth * b2)), star=float(np.sum(math.pi * j * b2)),
               energy=float(np.sum(math.pi ** 2 * j * j * b2)), l2=float(np.sum(b2)))
    # tail j >= J, j = 2m l + rho, l >= L0: b_j = alpha/j^2 + beta/j^3 (coth(pi j) = 1 to 1e-170)
    q = L0 + rho_all / (2 * m)
    M = 2.0 * m
    Z = {r: hurwitz(float(r), q)[0] * M ** (-r) for r in (2, 3, 4, 5, 6)}
    A2, AB, B2 = alpha_r ** 2, alpha_r * beta_r, beta_r ** 2
    for key, w0 in (("harm", 1), ("star", 1), ("energy", 2), ("l2", 0)):
        # weight pi^{w0} j^{w0}:  sum j^{w0} (alpha/j^2 + beta/j^3)^2
        tail = float(np.sum(A2 * Z[4 - w0] + 2 * AB * Z[5 - w0] + B2 * Z[6 - w0]))
        out[key] += (math.pi ** w0) * tail
    return out


# ----------------------------------------------------------------------------
# algorithm
# ----------------------------------------------------------------------------

def nodal_energy(y, k):
    """E_k(y) = k sum_{i=0}^{k-1} (y_{i+1} - y_i)^2 with y_0 = y_k = 0."""
    yy = np.concatenate(([0.0], y, [0.0]))
    return k * math.fsum(np.diff(yy) ** 2)


def second_diff(y):
    yy = np.concatenate(([0.0], y, [0.0]))
    return 2 * yy[1:-1] - yy[:-2] - yy[2:]


def make_noise(path, f, m, radius, stage, rng):
    if m <= 1:
        return np.zeros(0)
    if path == "centered":
        return np.zeros(m - 1)
    if path == "random":
        return rng.uniform(-radius, radius, size=m - 1)
    if path == "adversarial":
        if stage == "pilot":
            # drive the nodal energy down: e_i = -delta sign(2f_i - f_{i-1} - f_{i+1})
            return -radius * np.where(second_diff(f.nodal(m)) >= 0, 1.0, -1.0)
        # final stage: alternating (Nyquist) errors maximise the noise energy
        return radius * (-1.0) ** np.arange(1, m)
    raise ValueError(path)


@dataclass
class HProblem:
    f: Trace
    rho: float

    @property
    def E(self):
        return self.f.energy()

    @property
    def A(self):
        return self.E + self.rho


def pilot(P: HProblem, k, delta, e):
    """Lambda (corollary formula) and the gap A - Lambda, evaluated by the exact
    rearrangement  A - Lambda = rho + int|(f - I_k f)'|^2 - [E_k(y) - E_k(f)] + 2 delta k sum|D2 y|."""
    if k <= 1:
        gap = P.A
        return dict(gap=gap, Lambda=0.0, lemma1_ok=True, resid=P.E)
    fn = P.f.nodal(k)
    y = fn + e
    yy = np.concatenate(([0.0], y, [0.0]))
    ff = np.concatenate(([0.0], fn, [0.0]))
    de = np.diff(yy) - np.diff(ff)
    dE = k * math.fsum(2 * np.diff(ff) * de + de * de)            # E_k(y) - E_k(f)
    pen = 2 * delta * k * math.fsum(np.abs(second_diff(y)))
    resid = error_norms(P.f, k, None)["energy"]                   # int|f'|^2 - E_k(f) >= 0
    raw = P.rho + resid - dE + pen
    gap = min(P.A, raw)
    # data-only Lambda (what the algorithm computes in floating point), for the record
    Lam_data = max(0.0, nodal_energy(y, k) - pen)
    return dict(gap=gap, Lambda=P.A - gap, Lambda_data=Lam_data, resid=resid,
                lemma1_ok=bool(gap >= P.rho + resid * (1 - 1e-9) - 1e-300))


def _pilot_k(eps):
    k = 1
    while k ** (-1.5) > eps:
        k *= 2
    return k


def bm(eps, m):
    return eps / (2 * CN_H * math.sqrt(max(m - 1, 1)))


_INTERP_CACHE: dict = {}


def run(P: HProblem, eps, algo, path, qs=(1, 2), cap=2 ** 22, seed=0, final=True):
    rng = np.random.default_rng(seed)
    k = _pilot_k(eps)
    if algo == "two":
        delta = k ** -2.0
    else:
        mmax = 1
        while CB_H * mmax ** -0.5 * math.sqrt(P.A) > eps / 2:
            mmax *= 2
        mbar = max(k, mmax)
        delta = min(k ** -2.0, bm(eps, mbar))
    e_p = make_noise(path, P.f, k, delta, "pilot", rng)
    pil = pilot(P, k, delta, e_p)
    m = k
    while CB_H * m ** -0.5 * math.sqrt(pil["gap"]) > eps / 2:
        m *= 2
    b = bm(eps, m) if algo == "two" else delta
    rec = dict(f=P.f.name, rho=P.rho, A=P.A, eps=eps, algo=algo, path=path, k=k, m=m,
               delta=delta, b=b, gap=pil["gap"], Lambda=pil["Lambda"],
               Lambda_data=pil.get("Lambda_data", 0.0), pilot_lemma_ok=pil["lemma1_ok"])
    for q in qs:
        rec[f"cost_q{q}"] = (k - 1) * price(delta, q) + (m - 1) * price(b, q)
    rec["executed"] = False
    if final and m <= cap:
        e_f = make_noise(path, P.f, m, b, "final", rng)
        full = error_norms(P.f, m, e_f)
        key = (P.f.c1, tuple(sorted(P.f.sines.items())), m)
        if key not in _INTERP_CACHE:                      # path-independent: f - I_m f
            _INTERP_CACHE[key] = error_norms(P.f, m, None)
        interp = _INTERP_CACHE[key]
        noise = error_norms(P.f, m, e_f, with_f=False) if m > 1 else dict(harm=0.0, star=0.0, energy=0.0, l2=0.0)
        cert1 = CB_H * m ** -0.5 * math.sqrt(pil["gap"])
        cert2 = CN_H * math.sqrt((m - 1) * b * b)           # (sum_i eta_i^2)^{1/2}, eta_i = b
        # constant checks (each inequality used in the certificate, on this run's g and z)
        g_ok = (interp["star"] <= math.sqrt(interp["l2"] * interp["energy"]) * (1 + 1e-9)
                and interp["l2"] <= interp["energy"] / (math.pi * m) ** 2 * (1 + 1e-9))
        z_ok = noise["harm"] <= CN_H ** 2 * float(np.sum(e_f ** 2)) * (1 + 1e-9)
        rec.update(executed=True, err=math.sqrt(full["harm"]), err_interp=math.sqrt(interp["harm"]),
                   err_noise=math.sqrt(noise["harm"]), cert=cert1 + cert2, cert_interp=cert1,
                   cert_noise=cert2, const_ok=bool(g_ok and z_ok))
    return rec
