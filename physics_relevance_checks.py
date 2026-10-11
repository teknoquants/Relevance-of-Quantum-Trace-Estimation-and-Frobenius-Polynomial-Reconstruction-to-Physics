# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Roger Cooperman; A.E. Petrunina V.V.
"""Validation of the six statements of the white paper
"Relevance of Quantum Trace Estimation and Frobenius Polynomial Reconstruction to Physics: A Formal Statement".

  A  Statement 1 (DQC1 / Hadamard test): Pr(0) = 1/2 + Re Tr(X)/(2n) for Haar-random X in U(n), n = 2..16.
  B  Statement 2 (Heisenberg vs standard quantum limit): queries needed to guarantee additive precision eps on
     the acceptance probability with success >= 8/pi^2, for amplitude estimation (exact worst-case law) and for
     independent repetitions (exact binomial at a = 1/2); fitted log-log slopes ~1 and ~2.
  C  Statement 3 (two-point lower bound, [CP] Thm 3.4): U_0 = I, U_theta = e^{i theta} I, theta = 1/(d^2 P(d)),
     P(d) = d: trace distance <= 1/P(d) and coefficient gap at k = floor(d/2) >= (2/pi) k theta C(d,k), > 1/3
     from moderate d on; exact closed forms evaluated in high precision.
  D  Statement 4 (Hilbert-Polya form): for test curves, every zero of zeta_C(s) = Z_C(Q^-s) has Re s = 1/2 and
     s = 1/2 + i(theta + 2 pi k)/log Q with e^{i theta} an eigenvalue of U_C; checked by evaluating L_C.
  E  Statement 5 (symplectic random-matrix moments): Weyl-integration quadrature on USp(2g), g = 1..4:
     E Tr U^j = -eta_j (all j <= 2g + 2) and E (Tr U^j)^2 = j + eta_j (1 <= j <= g); plus the genus-1 (USp(2) = SU(2))
     average of tau_1, tau_2, tau_1^2 over ALL elliptic curves y^2 = x^3 + ax + b over F_p, p = 101, 211, 401.
  F  Statement 6 (free-field identities): exact rational checks that the coefficients of exp(sum s_m z^m/m) are
     the complete symmetric h_k and equal the series of det(I - zA)^-1, and that det(I - zA) = sum (-1)^k e_k z^k,
     for random integer matrices; and c_k = (-1)^k e_k(beta) for the normalised Frobenius spectrum of test curves.
Run: python3 -I physics_relevance_checks.py      Output: results/physics_checks.json
"""
import sys, math, time, itertools
from fractions import Fraction
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frobenius_core import (test_curves, L_from_counts, exact_spectrum, haar_unitary, hadamard_state,
                            good_projector_diag, ae_distribution_analytic, ae_estimate_values, field, save_json)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
P0 = 8 / math.pi ** 2


def check_A(rng):
    worst = 0.0
    for n in (2, 3, 4, 8, 16):
        for _ in range(20):
            X = haar_unitary(n, rng)
            a = float(np.sum(np.abs(hadamard_state(X)) ** 2 * good_projector_diag(n)))
            worst = max(worst, abs(a - (0.5 + np.real(np.trace(X)) / (2 * n))))
    return {"cases": 100, "max_err": worst, "ok": worst < 1e-12}


def ae_queries(eps, grid=np.linspace(0.05, 0.95, 181)):
    M = 2
    while True:
        vals = ae_estimate_values(M)
        if all(ae_distribution_analytic(a, M)[np.abs(vals - a) <= eps].sum() >= P0 for a in grid):
            return 2 * M - 1, M
        M *= 2


def sampling_queries(eps):
    from scipy.stats import binom
    lo, hi = 1, 1
    def ok(N):
        k = np.arange(N + 1)
        return binom.pmf(k, N, 0.5)[np.abs(k / N - 0.5) <= eps].sum() >= P0
    while not ok(hi):
        hi *= 2
    lo = hi // 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ok(mid): hi = mid
        else: lo = mid
    return hi


def check_B():
    rows = []
    for e in [2.0 ** (-k) for k in range(3, 11)]:
        kae, M = ae_queries(e)
        rows.append({"eps": e, "ae_queries": kae, "M": M, "sampling_queries": sampling_queries(e)})
    x = np.log([1 / r["eps"] for r in rows])
    s_ae = float(np.polyfit(x, np.log([r["ae_queries"] for r in rows]), 1)[0])
    s_cl = float(np.polyfit(x, np.log([r["sampling_queries"] for r in rows]), 1)[0])
    return {"rows": rows, "slope_amplitude_estimation": s_ae, "slope_independent_sampling": s_cl,
            "ok": 0.85 <= s_ae <= 1.15 and 1.85 <= s_cl <= 2.15}


def check_C():
    import mpmath
    mpmath.mp.dps = 60
    rows = []
    for d in (8, 16, 32, 64, 128):
        P = d
        th = mpmath.mpf(1) / (d * d * P)
        tr = max(abs(d * (mpmath.e ** (1j * j * th) - 1)) for j in range(1, d + 1))
        k = d // 2
        gap = mpmath.binomial(d, k) * abs(mpmath.e ** (1j * k * th) - 1)
        lb = 2 / mpmath.pi * k * th * mpmath.binomial(d, k)
        rows.append({"d": d, "sup_trace_diff": float(tr), "one_over_P": 1 / P, "coef_gap": float(gap), "lower_bound": float(lb),
                     "trace_ok": bool(tr <= mpmath.mpf(1) / P), "gap_ge_bound": bool(gap >= lb), "gap_gt_one_third": bool(gap > mpmath.mpf(1) / 3)})
    return {"rows": rows, "first_d_with_gap_over_one_third": next((r["d"] for r in rows if r["gap_gt_one_third"]), None),
            "ok": all(r["trace_ok"] and r["gap_ge_bound"] for r in rows) and rows[-1]["gap_gt_one_third"]}


def check_D():
    rows = []
    for C in test_curves():
        N = C.counts(C.g)
        L, _ = L_from_counts(N, C.g, C.Q)
        beta = exact_spectrum(L, C.Q)
        lq = math.log(C.Q)
        worst_val, worst_re = 0.0, 0.0
        for b in beta:
            th = math.atan2(b.imag, b.real)
            for k in (-1, 0, 1):
                s = 0.5 + 1j * (th + 2 * math.pi * k) / lq
                u = C.Q ** (-s)
                val = abs(sum(c * u ** i for i, c in enumerate(L))) / sum(abs(c) * abs(u) ** i for i, c in enumerate(L))
                worst_val = max(worst_val, val)
        # every zero u of L(u) gives Re s = 1/2 where u = Q^{-s}
        for z in np.roots([float(c) for c in L][::-1]):
            s_re = -math.log(abs(z)) / lq
            worst_re = max(worst_re, abs(s_re - 0.5))
        rows.append({"curve": C.name, "max_rel_L_at_predicted_zero": worst_val, "max_dev_Re_s": worst_re})
    return {"rows": rows, "ok": all(r["max_rel_L_at_predicted_zero"] < 1e-9 and r["max_dev_Re_s"] < 1e-6 for r in rows)}


def weyl_moments(g, jmax, npts=48):
    x, w = np.polynomial.legendre.leggauss(npts)
    th = (x + 1) * math.pi / 2; wt = w * math.pi / 2
    G = np.meshgrid(*([th] * g), indexing="ij")
    W = np.ones_like(G[0])
    for gw in np.meshgrid(*([wt] * g), indexing="ij"):
        W = W * gw
    dens = np.ones_like(W)
    for i in range(g):
        dens *= np.sin(G[i]) ** 2
        for k in range(i + 1, g):
            dens *= (np.cos(G[i]) - np.cos(G[k])) ** 2
    Z = float((W * dens).sum())
    out = []
    for j in range(1, jmax + 1):
        tr = sum(2 * np.cos(j * G[i]) for i in range(g))
        out.append((float((W * dens * tr).sum() / Z), float((W * dens * tr ** 2).sum() / Z)))
    return out


def check_E():
    rows, ok = [], True
    for g in (1, 2, 3, 4):
        m = weyl_moments(g, 2 * g + 2, npts=48 if g <= 3 else 28)
        for j, (mean, second) in enumerate(m, start=1):
            eta = 1 if (j % 2 == 0 and j <= 2 * g) else 0
            mean_ok = abs(mean + eta) < 1e-8
            var_ok = (abs(second - (j + eta)) < 1e-8) if j <= g else None
            ok &= mean_ok and (var_ok in (True, None))
            rows.append({"g": g, "j": j, "E_Tr": mean, "minus_eta": -eta, "E_Tr2": second,
                         "j_plus_eta_if_j_le_g": (j + eta) if j <= g else None, "mean_ok": mean_ok, "second_ok": var_ok})
    # genus 1, all curves
    st = []
    for p in (101, 211, 401):
        F = field(p, 1)
        xs = np.arange(p)
        chi = F.chi(xs)
        x3 = (xs ** 3) % p
        t1, t2, t11, n = 0.0, 0.0, 0.0, 0
        for a in range(p):
            base = (x3 + a * xs) % p
            for b in range(p):
                if (4 * a ** 3 + 27 * b * b) % p == 0:
                    continue
                N1 = p + 1 + int(chi[(base + b) % p].sum())
                ap = p + 1 - N1
                tau1 = ap / math.sqrt(p); tau2 = (ap * ap - 2 * p) / p
                t1 += tau1; t2 += tau2; t11 += tau1 * tau1; n += 1
        st.append({"p": p, "curves": n, "mean_tau1": t1 / n, "mean_tau2": t2 / n, "mean_tau1_sq": t11 / n,
                   "haar_values": [0, -1, 1]})
    st_ok = all(abs(r["mean_tau1"]) < 0.05 and abs(r["mean_tau2"] + 1) < 0.05 and abs(r["mean_tau1_sq"] - 1) < 0.05 for r in st)
    return {"rows": rows, "genus1_census": st, "ok": bool(ok and st_ok)}


def check_F(rng):
    import sympy as sp
    z = sp.Symbol("z")
    ok, cases = True, 0
    for _ in range(12):
        d = int(rng.integers(2, 6))
        A = sp.Matrix(rng.integers(-3, 4, size=(d, d)).tolist())
        K = 8
        s = [int((A ** m).trace()) for m in range(1, K + 1)]
        h = [Fraction(1)]
        for k in range(1, K + 1):
            h.append(sum(Fraction(s[m - 1]) * h[k - m] for m in range(1, k + 1)) / k)
        e = [Fraction(1)]
        for k in range(1, d + 1):
            e.append(sum((-1) ** (m - 1) * e[k - m] * Fraction(s[m - 1]) for m in range(1, k + 1)) / k)
        det = sp.expand((sp.eye(d) - z * A).det())
        detc = [sp.Rational(det.coeff(z, k)) for k in range(d + 1)]
        ok &= all(detc[k] == (-1) ** k * sp.Rational(e[k].numerator, e[k].denominator) for k in range(d + 1))
        ser = sp.series(1 / det, z, 0, K + 1).removeO()
        ok &= all(sp.Rational(ser.coeff(z, k)) == sp.Rational(h[k].numerator, h[k].denominator) for k in range(K + 1))
        cases += 1
    # curves: c_k = (-1)^k e_k(beta), c_k = b_k / Q^{k/2}
    worst = 0.0
    for C in test_curves():
        N = C.counts(C.g)
        L, _ = L_from_counts(N, C.g, C.Q)
        beta = exact_spectrum(L, C.Q)
        ek = np.poly(beta)          # coefficients of prod (x - beta): x^n - e1 x^{n-1} + ...
        for k in range(len(L)):
            ck = L[k] / C.Q ** (k / 2)
            worst = max(worst, abs(ck - ek[k].real) / max(1, abs(ck)))     # np.poly already carries (-1)^k
    return {"matrix_cases": cases, "exact_identities_ok": bool(ok), "curve_max_rel_err": worst,
            "ok": bool(ok and worst < 1e-9)}


if __name__ == "__main__":
    t0 = time.time()
    rng = np.random.default_rng(20261011)
    res = {}
    res["A_dqc1"] = check_A(rng); print("A:", res["A_dqc1"])
    res["B_heisenberg"] = check_B(); print("B: slopes AE %.3f, sampling %.3f" % (res["B_heisenberg"]["slope_amplitude_estimation"], res["B_heisenberg"]["slope_independent_sampling"]))
    res["C_two_point"] = check_C(); print("C:", [(r["d"], "%.3g" % r["coef_gap"]) for r in res["C_two_point"]["rows"]], res["C_two_point"]["ok"])
    res["D_hilbert_polya"] = check_D(); print("D:", res["D_hilbert_polya"]["ok"], max(r["max_rel_L_at_predicted_zero"] for r in res["D_hilbert_polya"]["rows"]))
    res["E_random_matrix"] = check_E(); print("E:", res["E_random_matrix"]["ok"], res["E_random_matrix"]["genus1_census"])
    res["F_free_field"] = check_F(rng); print("F:", res["F_free_field"])
    res["all_ok"] = all(v["ok"] for v in res.values() if isinstance(v, dict))
    res["seconds"] = round(time.time() - t0, 1)
    print("ALL OK:", res["all_ok"], res["seconds"], "s")
    save_json(OUT / "physics_checks.json", res)
    sys.exit(0 if res["all_ok"] else 1)
