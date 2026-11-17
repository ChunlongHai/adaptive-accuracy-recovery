"""Core numerics for the precision-adaptivity theorem on T^d (theory_core.tex, Secs. 2-3).

Conventions (as in the note): period-1 torus, hat v(k) = int v e^{-2 pi i k.x},
<k> = (1+|k|^2)^{1/2} (no 2 pi), ||v||_r^2 = sum <k>^{2r} |hat v(k)|^2.
Grid G_m = {j/m}, aliasing classes kappa in R_m = (-m/2, m/2]^d.

Everything is done on the Fourier side, aliasing class by aliasing class:
  D_v(kappa) = sum_l hat v(kappa+ml),   S(kappa) = sum_l <kappa+ml>^{-2s},
  hat(I_m v)(k) = <k>^{-2s} D_v(kappa)/S(kappa)              (Lemma interp)
The input f is a trigonometric polynomial (finitely many modes) and grid data are
y = f(G) + e.  We never form f(G): D_y = D_f + D_e with D_f assembled exactly from
the modes and D_e = FFT(e)/n.  Quantities that are mathematically small differences
of O(1) numbers (E - ||I_k y||_s^2, ||f - I_m y||_t) are evaluated through exact
algebraic rearrangements that avoid cancellation; this emulates the exact-arithmetic
model of the note.  The algorithm itself only uses data-computable quantities.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

# ----------------------------------------------------------------------------
# Lattice (aliasing) sums
# ----------------------------------------------------------------------------

_J_CACHE: dict = {}


def _J(q: float, R: float, d: int) -> float:
    """int_{|x|_inf > R} |x|^{-q} dx  (q > d)."""
    key = (q, d)
    if key not in _J_CACHE:
        if d == 1:
            _J_CACHE[key] = 2.0 / (q - 1.0)
        elif d == 2:
            # = 1/(q-2) * int_0^{2pi} c(theta)^{q-2}, c = max(|cos|,|sin|); 8-fold symmetry
            th = np.linspace(0.0, np.pi / 4, 4001)
            g = np.cos(th) ** (q - 2.0)
            h = th[1] - th[0]
            simpson = h / 3 * (g[0] + g[-1] + 4 * g[1:-1:2].sum() + 2 * g[2:-1:2].sum())
            _J_CACHE[key] = 8.0 * simpson / (q - 2.0)
        else:
            raise NotImplementedError
    return _J_CACHE[key] * R ** (d - q)


def lattice_prime(m: int, p: float, d: int, L: int) -> np.ndarray:
    """Sigma'(a) = sum_{l != 0} <a + m l>^{-p} for a in the quarter domain
    {0..m//2}^d (the sum depends on |kappa_i| only).  Explicit terms |l|_inf <= L,
    tail by the cell-midpoint integral with 2nd-order corrections (shift a/m,
    the '1+' in <.>, and the Euler-Maclaurin Laplacian term); relative tail
    error is O(L^{-4}) of the tail itself."""
    a1 = np.arange(m // 2 + 1, dtype=float)
    if d == 1:
        grids = [a1]
    else:
        grids = [a1[:, None], a1[None, :]]
    acc = np.zeros((m // 2 + 1,) * d)
    for l in itertools.product(range(-L, L + 1), repeat=d):
        if not any(l):
            continue
        x2 = 1.0
        for gi, li in zip(grids, l):
            x2 = x2 + (gi + m * li) ** 2
        acc += x2 ** (-p / 2)
    R = L + 0.5
    a2 = sum(gi ** 2 for gi in grids) / m ** 2  # |kappa/m|^2
    J0, J2 = _J(p, R, d), _J(p + 2, R, d)
    tail = J0 + J2 * (a2 * p * (p + 2 - d) / (2 * d)
                      - 0.5 * p / m ** 2
                      - p * (p + 2 - d) / 24.0)
    return acc + m ** (-p) * tail


# ----------------------------------------------------------------------------
# Rigorous enclosures (d = 1)
# ----------------------------------------------------------------------------

# Bernoulli numbers B_2, B_4, ..., B_16
_BERN = [1 / 6, -1 / 30, 1 / 42, -1 / 30, 5 / 66, -691 / 2730, 7 / 6, -3617 / 510]


def hurwitz(r: float, q, N: int = 12, J: int = 6):
    """Hurwitz zeta zeta(r, q) = sum_{n>=0} (q+n)^{-r}, r > 1, q > 0 (array).

    N explicit terms, then Euler-Maclaurin with J Bernoulli corrections at
    x = q + N.  For the completely monotone x^{-r} the EM remainder is bounded by
    the first omitted correction; we return 2x that as the error bound.
    Returns (value, bound)."""
    q = np.asarray(q, dtype=float)
    val = np.zeros_like(q)
    for n in range(N):
        val += (q + n) ** (-r)
    x = q + N
    val += x ** (1 - r) / (r - 1) + 0.5 * x ** (-r)
    poch, fact = r, 2.0          # (r)_{2j-1}, (2j)!
    nxt = None
    for j in range(1, J + 2):
        term = _BERN[j - 1] / fact * poch * x ** (-r - 2 * j + 1)
        if j <= J:
            val += term
        else:
            nxt = np.abs(term)
        poch *= (r + 2 * j - 1) * (r + 2 * j)
        fact *= (2 * j + 1) * (2 * j + 2)
    return val, 2.0 * nxt


def lattice_prime_enclosure(m: int, p: float, L: int = 8):
    """Rigorous enclosure [lo, hi] of Sigma'(a) = sum_{l != 0} <a + m l>^{-p}, d = 1,
    for a = 0..m//2 (quarter domain).  Terms 1 <= |l| <= L are explicit.  For
    |l| > L write x = m l + b (b = +-a, |b| <= m/2, so x >= m(L+1/2) > 0) and
    <x>^{-p} = x^{-p} (1 + x^{-2})^{-p/2} with
        1 - (p/2) u  <=  (1+u)^{-p/2}  <=  1 - (p/2) u + (p/2)(p/2+1) u^2 / 2,   u = x^{-2};
    the sums of x^{-r} are m^{-r} zeta(r, L+1+b/m), enclosed by `hurwitz`.
    Exact up to floating-point rounding."""
    a = np.arange(m // 2 + 1, dtype=float)
    acc = np.zeros_like(a)
    for l in range(1, L + 1):
        acc += (1.0 + (a + m * l) ** 2) ** (-p / 2) + (1.0 + (m * l - a) ** 2) ** (-p / 2)
    al = p / 2
    lo, hi = acc.copy(), acc.copy()
    width = np.zeros_like(a)      # hi - lo, accumulated from the bound terms (no cancellation)
    for b in (a, -a):
        q = L + 1 + b / m
        Z0, e0 = hurwitz(p, q)
        Z2, e2 = hurwitz(p + 2, q)
        Z4, e4 = hurwitz(p + 4, q)
        lo += m ** (-p) * (Z0 - e0) - al * m ** (-p - 2) * (Z2 + e2)
        hi += (m ** (-p) * (Z0 + e0) - al * m ** (-p - 2) * (Z2 - e2)
               + al * (al + 1) / 2 * m ** (-p - 4) * (Z4 + e4))
        width += (2 * m ** (-p) * e0 + 2 * al * m ** (-p - 2) * e2
                  + al * (al + 1) / 2 * m ** (-p - 4) * (Z4 + e4))
    return lo, hi, width


_P1_CACHE: dict = {}


def prime1d(m: int, p: float, L: int = 8):
    """Cached quarter-domain enclosure (mid, lo, hi) of Sigma'_p on G_m, d = 1."""
    key = (m, p, L)
    if key not in _P1_CACHE:
        lo, hi, w = lattice_prime_enclosure(m, p, L)
        _P1_CACHE[key] = (0.5 * (lo + hi), lo, hi, w)
    return _P1_CACHE[key][:3]


def prime1d_width(m: int, p: float, L: int = 8):
    prime1d(m, p, L)
    return _P1_CACHE[(m, p, L)][3]


def clear_caches():
    _GRID_CACHE.clear()
    _P1_CACHE.clear()


# ----------------------------------------------------------------------------
# Grid data structure
# ----------------------------------------------------------------------------

@dataclass
class Grid:
    """Per-(m, s, t, d) aliasing data, quarter-domain arrays expanded on demand."""
    m: int
    d: int
    s: float
    t: float
    L: int
    absk: np.ndarray = field(init=False)
    br2: np.ndarray = field(init=False)   # <kappa>^2, full FFT layout
    Sp: np.ndarray = field(init=False)    # S'(kappa)  (l != 0 part), p = 2s
    Tp: np.ndarray = field(init=False)    # T'(kappa), p = 4s - 2t
    S: np.ndarray = field(init=False)
    T: np.ndarray = field(init=False)

    def __post_init__(self):
        m, d = self.m, self.d
        idx = np.arange(m)
        self.absk = np.minimum(idx, m - idx)
        self.rigorous = d == 1
        if d == 1:
            # rigorous enclosures; mid-point used as the value, [lo, hi] kept
            sm, slo, shi = prime1d(m, 2 * self.s)
            tm, tlo, thi = prime1d(m, 4 * self.s - 2 * self.t)
            ak = self.absk
            self.Sp, self.Tp = sm[ak], tm[ak]
            self.Sp_lo, self.Sp_hi = slo[ak], shi[ak]
            self.Tp_lo, self.Tp_hi = tlo[ak], thi[ak]
            self.Sp_w = prime1d_width(m, 2 * self.s)[ak]     # exact-arithmetic width hi - lo
            self.br2 = 1.0 + self.absk.astype(float) ** 2
        else:
            qS = lattice_prime(m, 2 * self.s, d, self.L)
            qT = lattice_prime(m, 4 * self.s - 2 * self.t, d, self.L)
            ix = np.ix_(self.absk, self.absk)
            self.Sp, self.Tp = qS[ix], qT[ix]
            a = self.absk.astype(float)
            self.br2 = 1.0 + a[:, None] ** 2 + a[None, :] ** 2
        self.S = self.br2 ** (-self.s) + self.Sp
        self.T = self.br2 ** (self.t - 2 * self.s) + self.Tp
        if d == 1:
            self.S_lo = self.br2 ** (-self.s) + self.Sp_lo
            self.S_hi = self.br2 ** (-self.s) + self.Sp_hi

    def prime(self, p):
        """Sigma'_p(kappa) = sum_{l != 0} <kappa + m l>^{-p} in full FFT layout (d = 1)."""
        assert self.d == 1
        return prime1d(self.m, p)[0][self.absk]

    @property
    def n(self):
        return self.m ** self.d

    @property
    def shape(self):
        return (self.m,) * self.d


_GRID_CACHE: dict = {}


def get_grid(m, d, s, t, L=None):
    if L is None:
        L = 16 if d == 1 else 8
    key = (m, d, s, t, L)
    if key not in _GRID_CACHE:
        _GRID_CACHE[key] = Grid(m, d, s, t, L)
    return _GRID_CACHE[key]


# ----------------------------------------------------------------------------
# Sharp constants C_B (Lemma sampling, r = t) and C_N (Lemma noise, r = t)
# ----------------------------------------------------------------------------

def _tail_mass_upper(m, s, d, L):
    """Upper bound for sum_{|l|_inf > L} <kappa+ml>^{-2s}, any kappa in R_m,
    using |kappa+ml| >= m(|l|_inf - 1/2)."""
    j = np.arange(L + 1, 200001, dtype=float)
    Nj = (2 * j + 1) ** d - (2 * j - 1) ** d
    tot = np.sum(Nj * (j - 0.5) ** (-2 * s))
    J = j[-1]
    tot += 2 * d * 3 ** (d - 1) * 2 ** (2 * s) * J ** (d - 2 * s) / (2 * s - d)  # crude tail
    return m ** (-2 * s) * tot


def CB_of_m(m, d, s, t, L=None, iters=64):
    """Sharp C_B(m) = m^{s-t} * sup_{g|G_m=0} ||g||_t/||g||_s.
    Per class kappa: maximise sum D_l|b_l|^2 over |b|=1, b ⟂ u, with
    D_l=<kappa+ml>^{-2(s-t)}, u_l^2=<kappa+ml>^{-2s}; the value is the largest
    root of the secular equation sum u_l^2/(D_l-lam)=0 in (D_(2), D_(1)).
    Aliases |l|_inf>L are lumped into one pseudo-component with u^2 and D replaced
    by upper bounds, which can only increase the root (conservative)."""
    if L is None:
        L = 4
    a1 = np.arange(m // 2 + 1, dtype=float)
    if d == 1:
        pts = [a1]
    else:
        A1, A2 = np.meshgrid(a1, a1, indexing="ij")
        keep = A1 >= A2  # symmetry kappa1 <-> kappa2
        pts = [A1[keep], A2[keep]]
    Dl, Wl = [], []
    for l in itertools.product(range(-L, L + 1), repeat=d):
        x2 = 1.0
        for gi, li in zip(pts, l):
            x2 = x2 + (gi + m * li) ** 2
        Dl.append(x2 ** (-(s - t)))
        Wl.append(x2 ** (-s))
    Dl, Wl = np.array(Dl), np.array(Wl)
    Dt = (1.0 + (m * (L + 0.5)) ** 2) ** (-(s - t))
    Wt = _tail_mass_upper(m, s, d, L)
    Ds = np.sort(Dl, axis=0)
    D1, D2 = Ds[-1], Ds[-2]
    lo, hi = D2.copy(), D1.copy()
    for _ in range(iters):
        mid = np.sqrt(lo * hi)  # geometric bisection: D_(2) can be ~1e-17 while D_(1) ~ 1
        with np.errstate(divide="ignore", invalid="ignore"):
            phi = np.sum(Wl / (Dl - mid), axis=0) + Wt / (Dt - mid)
        pos = ~(phi <= 0)
        hi = np.where(pos, mid, hi)
        lo = np.where(pos, lo, mid)
    lam = np.where(D1 - D2 <= 1e-13 * D1, D1, hi)  # hi: upper end of bracket
    return math.sqrt(lam.max()) * m ** (s - t)


def CN_of_m(m, d, s, t):
    """Sharp C_N(m) = m^{-t} * max_kappa sqrt(T_t(kappa))/S(kappa) (RMS form).
    d = 1: rigorous upper bound sqrt(T_hi)/S_lo from the lattice-sum enclosures
    (computed on the quarter domain, nothing cached).  d = 2: estimate."""
    if d == 1:
        a2 = 1.0 + np.arange(m // 2 + 1, dtype=float) ** 2
        slo = lattice_prime_enclosure(m, 2 * s)[0]
        thi = lattice_prime_enclosure(m, 4 * s - 2 * t)[1]
        r = (a2 ** (t - 2 * s) + thi) / (a2 ** (-s) + slo) ** 2
        return math.sqrt(np.max(r)) * m ** (-t)
    g = get_grid(m, d, s, t)
    return math.sqrt(np.max(g.T / g.S ** 2)) * m ** (-t)


# ----------------------------------------------------------------------------
# Input functions
# ----------------------------------------------------------------------------

@dataclass
class TrigPoly:
    """Real trigonometric polynomial: dict {k (tuple): hat f(k)} closed under k -> -k."""
    d: int
    coef: dict

    def norm2(self, r):
        return sum((1 + sum(ki * ki for ki in k)) ** r * abs(c) ** 2 for k, c in self.coef.items())

    def scaled(self, a):
        return TrigPoly(self.d, {k: a * c for k, c in self.coef.items()})

    def classes(self, m):
        """{class index tuple (k mod m): [(k, hat f(k)), ...]}"""
        out: dict = {}
        for k, c in self.coef.items():
            out.setdefault(tuple(ki % m for ki in k), []).append((k, c))
        return out

    def Df(self, m):
        D = np.zeros((m,) * self.d, dtype=complex)
        for idx, lst in self.classes(m).items():
            D[idx] = sum(c for _, c in lst)
        return D

    def W(self, s):
        return sum((1 + sum(ki * ki for ki in k)) ** s * abs(c) for k, c in self.coef.items())


@dataclass
class PowerInput:
    """Non-polynomial input on T^1:  hat f(k) = c <k>^{-a} for every k in Z.
    f is in H^sigma exactly for sigma < a - 1/2; with a = 2s + 1.6 this is
    sigma < 2s + 1.1, just above the threshold 2s + d/2 of Assumption 2.3.
    Every Fourier quantity is an aliasing (lattice) sum, enclosed rigorously."""
    a: float
    c: float = 1.0
    d: int = 1

    def _full(self, r_exp):
        """sum_k <k>^{-r_exp} over Z."""
        mid, _, _ = prime1d(1, r_exp)
        return 1.0 + float(mid[0])

    def norm2(self, r):
        return self.c ** 2 * self._full(2 * self.a - 2 * r)

    def scaled(self, b):
        return PowerInput(self.a, self.c * b)

    def W(self, s):
        return abs(self.c) * self._full(self.a - 2 * s)

    def Df(self, m):
        idx = np.arange(m)
        ak = np.minimum(idx, m - idx)
        mid, _, _ = prime1d(m, self.a)
        return (self.c * ((1.0 + ak.astype(float) ** 2) ** (-self.a / 2) + mid[ak])).astype(complex)


def default_input(d):
    if d == 1:
        # 0.2 + cos(2pi x) + 0.5 sin(4pi x) + 0.25 cos(6pi x)
        c = {(0,): 0.2, (1,): 0.5, (-1,): 0.5, (2,): -0.25j, (-2,): 0.25j,
             (3,): 0.125, (-3,): 0.125}
    else:
        # 0.2 + cos(2pi x1) + 0.5 sin(2pi(x1+x2)) + 0.3 cos(2pi(x2-x1))
        c = {(0, 0): 0.2, (1, 0): 0.5, (-1, 0): 0.5, (1, 1): -0.25j, (-1, -1): 0.25j,
             (-1, 1): 0.15, (1, -1): 0.15}
    return TrigPoly(d, c)


# ----------------------------------------------------------------------------
# Problem / interpolant / pilot / certificate
# ----------------------------------------------------------------------------

def _br2(k):
    return 1.0 + float(sum(ki * ki for ki in k))


@dataclass
class Problem:
    d: int
    s: float
    t: float
    f: TrigPoly
    rho: float
    CB: float
    CN: float

    @property
    def E(self):
        return self.f.norm2(self.s)

    @property
    def A(self):
        return self.E + self.rho


def _excl(Sprime_k, br2_kappa, br2_list, expo):
    """S - sum_{k in K} <k>^{expo}  where S = <kappa>^{expo} + S'.
    One k with |k| = |kappa| cancels the <kappa> term exactly."""
    rest = list(br2_list)
    matched = False
    for i, b in enumerate(rest):
        if b == br2_kappa:
            rest.pop(i)
            matched = True
            break
    val = Sprime_k - sum(b ** expo for b in rest)
    if not matched:
        val += br2_kappa ** expo
    return val


def interp_coeffs(P: Problem, g: Grid, De):
    """Return C(kappa) = D_y/S and the interpolation coefficients c_j (Lemma interp)."""
    Dy = P.f.Df(g.m) + De
    C = Dy / g.S
    c = np.real(np.fft.ifftn(C))
    return C, c


def interp_norm2(P: Problem, g: Grid, C, r):
    """||sum_k <k>^{-2s} C(kappa)||_r^2 for r in {s, t} (generic r computed directly)."""
    if r == P.s:
        return float(np.sum(np.abs(C) ** 2 * g.S))
    if r == P.t:
        return float(np.sum(np.abs(C) ** 2 * g.T))
    raise ValueError


def _resid_s2_power(P: Problem, g: Grid, De):
    """Vectorised resid_s2 for PowerInput: per class
    R(kappa) = c^2 [S_{2a-2s} S - S_a^2]/S, expanded so that the l = 0 products cancel
    exactly (X_2 X_s = X_a^2)."""
    s, a, c = P.s, P.f.a, P.f.c
    w = g.br2
    Xs, Xa, X2 = w ** (-s), w ** (-a / 2), w ** (s - a)
    Sp, Sap, S2p = g.Sp, g.prime(a), g.prime(2 * a - 2 * s)
    R = c ** 2 * float(np.sum((X2 * Sp + S2p * Xs + S2p * Sp - 2 * Xa * Sap - Sap ** 2) / g.S))
    Df = P.f.Df(g.m)
    cross = float(np.sum(2 * np.real(np.conj(Df) * De) / g.S))
    noise = float(np.sum(np.abs(De) ** 2 / g.S))
    return R - cross - noise, R


def _error_t_power(P: Problem, g: Grid, De):
    s, t, a, c = P.s, P.t, P.f.a, P.f.c
    w = g.br2
    Xs, Xa = w ** (-s), w ** (-a / 2)
    Sp, Sap = g.Sp, g.prime(a)
    P1, P2 = g.prime(2 * a - 2 * t), g.prime(a + 2 * s - 2 * t)
    Tp = g.Tp

    def tot(de):
        C = (c * (Xa + Sap) + de) / g.S
        num0 = (c * Xa * Sp - c * Xs * Sap - de * Xs) / g.S     # hat f(kappa) - <kappa>^{-2s} C
        l0 = w ** t * np.abs(num0) ** 2
        lr = c ** 2 * P1 - 2 * c * np.real(C) * P2 + np.abs(C) ** 2 * Tp
        return float(np.sum(l0 + lr))
    full = tot(De)
    interp = tot(0.0 * De)
    noise_only = float(np.sum(np.abs(De / g.S) ** 2 * g.T))
    return math.sqrt(max(full, 0.0)), math.sqrt(max(interp, 0.0)), math.sqrt(noise_only)


def resid_s2(P: Problem, g: Grid, De):
    """Q = E - ||I_m y||_s^2 evaluated without cancellation, and R = ||f - I_m f||_s^2."""
    if isinstance(P.f, PowerInput):
        return _resid_s2_power(P, g, De)
    s = P.s
    R = 0.0
    cross = 0.0
    for idx, lst in P.f.classes(g.m).items():
        S, Sp, b2k = g.S[idx], g.Sp[idx], g.br2[idx]
        Df = sum(c for _, c in lst)
        Cf = Df / S
        brs = [_br2(k) for k, _ in lst]
        for (k, c), b in zip(lst, brs):
            Sex = _excl(Sp, b2k, [b], -s)
            num = c * Sex - b ** (-s) * (Df - c)
            R += b ** s * abs(num / S) ** 2
        R += abs(Cf) ** 2 * _excl(Sp, b2k, brs, -s)
        cross += 2 * np.real(np.conj(Df) * De[idx]) / S
    noise = float(np.sum(np.abs(De) ** 2 / g.S))
    return R - cross - noise, R


def error_t(P: Problem, g: Grid, De):
    """||f - I_m y||_t (exact up to lattice-sum accuracy), plus its two parts
    ||f - I_m f||_t and ||I_m e||_t."""
    if isinstance(P.f, PowerInput):
        return _error_t_power(P, g, De)
    s, t = P.s, P.t
    C = (P.f.Df(g.m) + De) / g.S
    Wn = np.abs(C) ** 2 * g.T
    fcl = P.f.classes(g.m)
    for idx in fcl:
        Wn[idx] = 0.0
    tot = float(Wn.sum())
    interp_only = 0.0
    for idx, lst in fcl.items():
        S, Sp, Tp, b2k = g.S[idx], g.Sp[idx], g.Tp[idx], g.br2[idx]
        Df = sum(c for _, c in lst)
        brs = [_br2(k) for k, _ in lst]
        for which, de in (("full", De[idx]), ("f", 0.0)):
            Cc = (Df + de) / S
            acc = 0.0
            for (k, c), b in zip(lst, brs):
                Sex = _excl(Sp, b2k, [b], -s)
                num = c * Sex - b ** (-s) * (Df - c) - b ** (-s) * de
                acc += b ** t * abs(num / S) ** 2
            acc += abs(Cc) ** 2 * _excl(Tp, b2k, brs, t - 2 * s)
            if which == "full":
                tot += acc
            else:
                interp_only += acc
    noise_only = float(np.sum(np.abs(De / g.S) ** 2 * g.T))
    return math.sqrt(max(tot, 0.0)), math.sqrt(max(interp_only, 0.0)), math.sqrt(noise_only)


def pilot(P: Problem, k: int, delta: float, e: np.ndarray):
    """Lemma pilot on G_k with radius delta.  Returns gap = A - Lambda (stable) etc."""
    g = get_grid(k, P.d, P.s, P.t)
    De = np.fft.fftn(e) / g.n
    C, c = interp_coeffs(P, g, De)
    sumc = float(np.abs(c).sum())
    Q, R = resid_s2(P, g, De)
    raw_gap = P.rho + Q + 2 * delta * sumc          # = A - (||I_k y||^2 - 2 delta sum|c|)
    enc, tau = 0.0, float("nan")
    if g.rigorous:
        # Conservative use of the enclosure S_lo <= S <= S_hi (Remark comp):
        #   ||I_k y||^2 >= sum |D_y|^2 / S_hi              (upper enclosure of S)
        #   sum_j |c_j(y)| <= sum_j |c_j(S_mid)| + sum_kappa |D_y| (S_hi - S_lo) / S_lo^2
        # so Lambda_enc = A - gap_enc  <=  the exact-S Lambda  <=  ||I_k v||^2.
        Dy = P.f.Df(g.m) + De
        w = g.Sp_w                                   # S_hi - S_lo; S_hi - S_mid = w/2
        tau = float(np.max(w / g.S_lo))
        corr1 = float(np.sum(np.abs(Dy) ** 2 * (0.5 * w) / (g.S * g.S_hi)))
        extra = float(np.sum(np.abs(Dy) * w / g.S_lo ** 2))
        enc = corr1 + 2 * delta * extra
        raw_gap += enc
    gap = min(P.A, raw_gap)                          # Lambda = max(0, .)
    Lam = P.A - gap
    return dict(gap=gap, Lambda=Lam, sumc=sumc, Q=Q, R=R, enc=enc, tau=tau,
                lemma1_ok=bool(gap >= P.rho + R * (1 - 1e-9) - 1e-300))


# ----------------------------------------------------------------------------
# Response paths
# ----------------------------------------------------------------------------

def make_noise(path: str, P: Problem, m: int, radius: float, stage: str, rng):
    g = get_grid(m, P.d, P.s, P.t)
    if path == "centered":
        return np.zeros(g.shape)
    if path == "random":
        return rng.uniform(-radius, radius, size=g.shape)
    if path == "adversarial":
        if stage == "pilot":
            # push ||I_k y||_s^2 down: e_j = -radius * sign(c_j(f))
            cf = np.real(np.fft.ifftn(P.f.Df(m) / g.S))
            return -radius * np.where(cf >= 0, 1.0, -1.0)
        # final stage: square wave at the most amplified frequency kappa*
        amp = g.T / g.S ** 2
        kst = np.unravel_index(np.argmax(amp), amp.shape)
        kst = [ki if ki <= m // 2 else ki - m for ki in kst]
        J = np.indices(g.shape)
        ph = sum(kk * Jj for kk, Jj in zip(kst, J)) / m
        return radius * np.where(np.cos(2 * np.pi * ph) >= 0, 1.0, -1.0)
    raise ValueError(path)


# ----------------------------------------------------------------------------
# Algorithms (Theorems thm:two and thm:onealg)
# ----------------------------------------------------------------------------

def price(eta, q):
    return max(1.0, eta ** (-q))


def _pilot_k(P, eps):
    k = 1
    while k ** (-(2 * P.s - P.t)) > eps:
        k *= 2
    return k


def _choose_m(P, k, gap, eps):
    m = k
    while P.CB * m ** (-(P.s - P.t)) * math.sqrt(gap) > eps / 2:
        m *= 2
    return m


TAU_C = 0.01   # enclosure accuracy required: relative width <= TAU_C * k^{-2s} / max(1, A)


def public_radii(P: Problem, eps: float, k: int, K: int):
    """Public radius set of the K-radius algorithm:
    hi = k^{-2s}, lo = min(hi, eps mbar^{-t} / (2 C_N)), mbar = max{k, m_max},
    R = {lo r^{i-1}: i = 1..K}, r = (hi/lo)^{1/K}.  K = 1 gives the one-radius
    algorithm of Thm onealg (R = {eta_1})."""
    s, t = P.s, P.t
    mmax = 1
    while P.CB * mmax ** (-(s - t)) * math.sqrt(P.A) > eps / 2:
        mmax *= 2
    mbar = max(k, mmax)
    hi = k ** (-2 * s)
    lo = min(hi, eps * mbar ** (-t) / (2 * P.CN))
    r = (hi / lo) ** (1.0 / K)
    return [lo * r ** i for i in range(K)], mbar


def run(P: Problem, eps: float, algo: str, path: str, qs=(1, 2), cap=2 ** 22,
        seed=0, final=True, K=1):
    """Execute the two-radius ('two'), one-radius ('one', Thm onealg with
    mbar = max{k, m_max}) or K-public-radii ('K') algorithm at f on a response
    path.  Returns a record with costs for each q, the certificate and the
    realised error.  If m^d > cap the final stage is not executed."""
    rng = np.random.default_rng(seed)
    d, s, t = P.d, P.s, P.t
    k = _pilot_k(P, eps)
    mbar, R = None, None
    if algo == "two":
        delta = k ** (-2 * s)
    else:
        R, mbar = public_radii(P, eps, k, 1 if algo == "one" else K)
        delta = R[-1]                       # pilot at the top element
    e_p = make_noise(path, P, k, delta, "pilot", rng)
    pil = pilot(P, k, delta, e_p)
    if pil["tau"] == pil["tau"] and pil["tau"] > TAU_C * k ** (-2 * s) / max(1.0, P.A):
        raise RuntimeError(f"enclosure too wide: tau={pil['tau']:.3e} at k={k}")
    m = _choose_m(P, k, pil["gap"], eps)
    bm = eps * m ** (-t) / (2 * P.CN)
    if algo == "two":
        b = bm
    else:
        assert m <= mbar
        b = max(x for x in R if x <= bm * (1 + 1e-12))   # largest public radius <= b(m)
    rec = dict(d=d, s=s, t=t, rho=P.rho, A=P.A, eps=eps, algo=algo, path=path,
               K=(K if algo == "K" else (1 if algo == "one" else 0)),
               k=k, m=m, mbar=mbar, delta=delta, b=b, Lambda=pil["Lambda"], gap=pil["gap"],
               enc=pil["enc"], tau=pil["tau"], pilot_lemma_ok=pil["lemma1_ok"])
    for q in qs:
        rec[f"cost_q{q}"] = k ** d * price(delta, q) + m ** d * price(b, q)
    rec["executed"] = False
    if final and m ** d <= cap:
        g = get_grid(m, d, s, t)
        e_f = make_noise(path, P, m, b, "final", rng)
        De = np.fft.fftn(e_f) / g.n
        err, err_i, err_n = error_t(P, g, De)
        cert1 = P.CB * m ** (-(s - t)) * math.sqrt(pil["gap"])
        cert2 = P.CN * m ** t * b
        rec.update(executed=True, err=err, err_interp=err_i, err_noise=err_n,
                   cert=cert1 + cert2, cert_interp=cert1, cert_noise=cert2,
                   rms_noise=float(np.sqrt(np.mean(e_f ** 2))))
    return rec
