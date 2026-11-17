# adaptive-accuracy-recovery

Code and numerical records for the paper

> C. Hai, H. Tang, Y. Li and J. Wang, *Adaptive observation accuracy in reliable recovery from priced noisy point values*, submitted to Mathematics of Computation, 2026.

They reproduce every figure, table and number in Section 9 of the paper. The code implements
- Algorithm 1 (pilot and refine, Theorem 4.1);
- the single-accuracy algorithm of Theorem 4.2;
- the public-menu algorithm with K geometric radii (proof of Theorem 6.1);
- the harmonic-recovery algorithm of Corollary 8.2.

Each algorithm is run on three reply paths, and every run records its certificate, its true error and its cost.

## Reproduce

```
python -m venv venv && venv/bin/pip install -r requirements.txt
venv/bin/python run_all.py            # all experiments, tables and figures (about 8 min on a laptop)
venv/bin/python run_all.py --plots    # tables and figures only, from results/runs.json (about 20 s)
```

Tested with Python 3.9.6, NumPy 2.0.2, Matplotlib 3.9.4 on macOS (arm64); see `results/environment.json`. SciPy is not needed. The random reply path uses fixed seeds, so reruns give identical output.

## Files

| file | content |
|---|---|
| `torus_recovery.py` | torus experiments: lattice-sum enclosures, constants C_B and C_N, interpolant, energy pilot, certificate, and the two-radius, one-radius and K-radius algorithms |
| `harmonic.py` | harmonic recovery (Section 8): piecewise-linear algorithm, explicit constants, exact error evaluation |
| `run_all.py` | single entry point: experiments, fits, tables |
| `make_figures.py` | Figures 1–4 of the paper |
| `results/runs.csv`, `results/rho_sweep.csv` | torus runs: scans in ε and in ρ (trigonometric-polynomial inputs) |
| `results/power_runs.csv` | torus runs, finite-smoothness input |
| `results/harmonic_runs.csv` | harmonic runs |
| `results/public_radii_runs.csv` | public-menu runs (one row per c, ε, K, path) |
| `results/runs.json` | all runs, as read by `--plots` |
| `results/constants.json` | C_B(m) and C_N(m) for every dyadic m used, and the harmonic constants |
| `results/fits.md`, `results/fits.json` | every fitted exponent with its fitting window |
| `results/reliability.json` | certificate checks per experiment and per reply path, and enclosure widths |
| `figures/` | Figures 1–4 (`fig1_cost_vs_eps`, `fig2_penalty`, `fig4_public_radii`, `fig5_harmonic`), PDF and PNG |

## Torus experiments

**Setting.** As in the paper: frequencies k ∈ Z^d, ⟨k⟩ = (1+|k|²)^{1/2}, λ(k) = ⟨k⟩^{2s}, grids G_m = {j/m} with m a power of two. Triples (d,s,t) are (1,1,½), (1,2,1), (1,3/2,¼), (1,3/2,½), (1,3/2,¾), (2,3/2,¼), (2,3/2,½), (2,3/2,¾) and (2,2,1); price exponents q ∈ {1,2}.

**Inputs.**
- Trigonometric polynomials with ‖f‖_λ = 1:
  - d = 1: 0.2 + cos 2πx + 0.5 sin 4πx + 0.25 cos 6πx;
  - d = 2: 0.2 + cos 2πx₁ + 0.5 sin 2π(x₁+x₂) + 0.3 cos 2π(x₂−x₁).
- A finite-smoothness input (d = 1): f̂(k) ∝ ⟨k⟩^{−(2s+1.6)} for all k, normalised to ‖f‖_λ = 1. It lies in H^σ exactly for σ < 2s + 1.1, so it just satisfies the sufficient condition σ > 2s + d/2 of Section 2.

**Prior.** A = ‖f‖_λ² + ρ with ρ ∈ {0, 0.01A, A/2}; the ρ-sweep runs ρ/A from 1e-7 to 0.9.

**Algorithms.** As in Section 4:
- m₀ is the least power of two with m₀^{−(2s−t)} ≤ ε, and δ = m₀^{−2s};
- m ≥ m₀ is the least power of two with C_B m^{−(s−t)}(A−Λ)^{1/2} ≤ ε/2;
- b(m) = εm^{−t}/(2C_N);
- single accuracy: η₁ = min{m₀^{−2s}, εm̄^{−t}/(2C_N)} with m̄ = max{m₀, m_max};
- price max(1, η^{−q}); certificate C_B m^{−(s−t)}(A−Λ)^{1/2} + C_N m^t b.

**Reply paths.**
- centered: exact values;
- adversarial: in the pilot, errors −δ·sign α_j(f), which lower Λ; in the final sweep, a square wave at the most amplified frequency;
- random: uniform in [−η, η].

The reported cost is the maximum over the three paths.

**Lattice sums in one dimension (Remark 3.8).** The sums Σ_{l≠0}⟨κ+ml⟩^{−p} are enclosed rigorously:
- terms with |l| ≤ 8 are summed explicitly;
- for |l| > 8, ⟨x⟩^{−p} = x^{−p}(1+x^{−2})^{−p/2} is bracketed by two-term expansions in x^{−2}, which reduces the tail to Hurwitz zeta values;
- the Hurwitz zeta values are evaluated by Euler–Maclaurin with an explicit remainder bound (twice the first omitted Bernoulli term, valid by complete monotonicity).

The pilot uses the upper enclosure in ‖I_{m₀}y‖_λ² and an upper bound for Σ_j|α_j(y)|, so Λ remains a lower bound. The enclosure width is required to be at most 0.01·m₀^{−2s}; measured over all one-dimensional pilots, it is at most 3.0e-5·m₀^{−2s}, and the induced shift in Λ is at most 9.6e-9.

**Constants.**
- C_B(m) = m^{s−t} sup{‖g‖_t/‖g‖_λ : g = 0 on G_m}. In each aliasing class this is the largest root of a secular equation, bracketed from above by bisection; the code uses C_B = max{2^{s−t}, max_m C_B(m)}.
- C_N(m) = m^{−t} max_κ T(κ)^{1/2}/Φ(κ), computed from the enclosures.
- Both are evaluated for every dyadic m ≤ 2^22 (d = 1) and m ≤ 2^11 (d = 2), which covers every grid used.

| (d,s,t) | C_B | C_N |
|---|---|---|
| (1,1,½) | 1.41421 | 0.45717 |
| (1,2,1) | 2.00000 | 0.69798 |
| (1,3/2,¼) | 2.37841 | 0.69427 |
| (2,3/2,¾) | 1.68179 | 0.33477 (estimate) |
| (2,2,1) | 2.00000 | 0.40907 (estimate) |

## Harmonic experiments (Section 8)

- Traces f₁ = sin πx + x(1−x) and f₂ = x(1−x) + 0.4 sin 2πx, normalised to ∫|f′|² = 1. Both have f″(0) ≠ 0, so Lf jumps at the reflection points.
- The algorithm is that of Corollary 8.2: m₀ is the least power of two with m₀^{−3/2} ≤ ε, δ = m₀^{−2}, the pilot Λ as displayed there, and final radius b = ε/(2C_N^H (m−1)^{1/2}).
- Constants C_B^H = (coth π/π)^{1/2} = 0.5652 and C_N^H = (2 coth π)^{1/2} = 1.4169. Each run also checks the three inequalities behind them on its own residual and noise interpolant; there were no failures.
- Cost counts interior nodes only: (m₀−1)·price(δ) + (m−1)·price(b).
- The true error is computed in closed form from the sine coefficients of f − I_m f − I_m e. Coefficients with j < 2m are summed explicitly; for j ≥ 2m each residue class mod 2m has the form α/j² + β/j³ and its tail is a Hurwitz zeta value. No quadrature is used.

## What is rigorous

- **One-dimensional torus runs and the harmonic runs.** The lattice sums are enclosed and rounded in the safe direction, and C_B, C_N are proven upper bounds on every grid used. Apart from floating-point rounding, the certificates are therefore rigorous. At ρ = 0 the gap A − Λ is far below double precision relative to A; it is evaluated by exact algebraic rearrangement, not by interval arithmetic.
- **Two-dimensional torus runs.** The lattice tails and C_N are accurate estimates (relative accuracy about 1e-5), not enclosures. The reliability figures for d = 2 are consistency checks only.

## Results

All numbers quoted in Section 9 come from `results/reliability.json` and `results/fits.md`.

**Reliability** (no run had a certificate below its true error or a true error above ε):

| experiment | runs | min certificate / error | max error / ε |
|---|---|---|---|
| torus ε-scans, d = 1 | 3102 | 1.49 | 0.48 |
| torus ε-scans, d = 2 | 1056 | 1.51 | 0.48 |
| torus ρ-sweep | 3096 | 1.49 | 0.48 |
| torus, finite-smoothness input | 900 | 1.94 | 0.45 |
| harmonic | 930 | 1.66 | 0.37 |
| public menus (final sweep executed when m ≤ 2^14) | 11729 | 1.93 | 0.45 |

The adversarial path is always the tightest.

**Exponents.** Fits are least-squares slopes of log cost against log ε over a stated window. Because grid sizes are powers of two, costs are step functions with correlated residuals, so no standard errors are given; instead `fits.md` lists each window and the slopes over its two halves.

| (d,s,t), ρ = 0, q = 1 | two radii | one radius | penalty g | window |
|---|---|---|---|---|
| (1,1,½) | 1.99 / 2.00 | 2.66 / 2.67 | 0.66 / 0.67 | 1e-7 to 1e-2 |
| (1,2,1) | 1.65 / 1.67 | 2.32 / 2.33 | 0.67 / 0.67 | 1e-12 to 1e-2 |
| (1,3/2,¼) | 1.45 / 1.45 | 1.55 / 1.56 | 0.10 / 0.11 | 1e-12 to 1e-2 |
| (2,3/2,¾) | 2.16 / 2.22 | 2.73 / 2.89 | 0.57 / 0.67 | 1e-4 to 1e-2 |
| (2,2,1) | 2.02 / 2.00 | 2.64 / 2.67 | 0.62 / 0.67 | 1e-6 to 1e-2 |

(fit / prediction). For (2,3/2,¾) the two-dimensional grid limit leaves less than two decades, and there is no usable slack window.

## Figures

| file | paper | content |
|---|---|---|
| `fig1_cost_vs_eps` | Figure 1 | cost against ε (decreasing to the right), q = 1, ρ = 0 (top) and ρ = A/2 (bottom) |
| `fig2_penalty` | Figure 2 | (a) cost ratio at ρ = 0, q = 2, s = 3/2, t ∈ {¼,½,¾}, d ∈ {1,2}; (b) cost ratio against ρ at fixed ε |
| `fig4_public_radii` | Figure 3 | worst ratio over the family of public-menu cost to adaptive cost, K = 1,…,4, q = 2 |
| `fig5_harmonic` | Figure 4 | (a) harmonic cost against ε at ρ = 0, q = 1; (b) certificate and true error of all 930 harmonic runs |

In Figure 2(b) the ratio has small bumps: m is a power of two chosen by a threshold on A − Λ, so a small change in ρ can move m across a dyadic jump for one algorithm and not the other.

## License

MIT; see `LICENSE`.
