# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Roger Cooperman; A.E. Petrunina V.V.
"""Shared core for the validation scripts of
"The Quantum Trace of Frobenius as a Formal Object".

Contents
  FiniteField            F_{p^e} with exponential/log tables (vectorised with numpy)
  Hyperelliptic          y^2 = f(x) over F_Q, Q = p^r, p odd; point counts over F_{Q^m}
  newton / L-polynomials exact Newton identities, functional equation, decoder
  realisations           normalised Frobenius spectrum, USp(2g) form
  hadamard / AE          real Hadamard-test state, amplitude estimation (statevector and analytic)

Everything exact uses fractions.Fraction or Python integers; numerical parts use numpy.
"""
from __future__ import annotations

import math
from fractions import Fraction
from functools import lru_cache
import numpy as np


# ============================================================ finite fields
class FiniteField:
    """F_q, q = p^e, elements are integers 0..q-1 whose base-p digits are the
    coefficients (low degree first) in F_p[t]/(h), h primitive of degree e."""

    def __init__(self, p: int, e: int):
        self.p, self.e, self.q = p, e, p ** e
        self.powp = np.array([p ** i for i in range(e)], dtype=np.int64)
        self.h = self._primitive_poly()
        self._tables()

    # --- construction --------------------------------------------------
    def _powers_of_t(self, h):
        """Yield digit vectors of t^k, k = 0.. until it returns to 1 (or exceeds q-1)."""
        p, e = self.p, self.e
        v = [1] + [0] * (e - 1)
        out = [tuple(v)]
        for _ in range(self.q - 1):
            top = v[-1]
            v = [0] + v[:-1]
            if top:
                v = [(v[i] - top * h[i]) % p for i in range(e)]
            if v == [1] + [0] * (e - 1):
                return out, True
            out.append(tuple(v))
        return out, False

    def _primitive_poly(self):
        p, e = self.p, self.e
        if e == 1:
            # find a primitive root g mod p; h(t) = t - g
            for g in range(2, p) if p > 2 else [1]:
                if all(pow(g, (p - 1) // r, p) != 1 for r in _prime_factors(p - 1)):
                    return [(-g) % p, 1]
            return [p - 1, 1]
        import itertools
        q1 = self.q - 1
        factors = _prime_factors(q1)

        def mulmod(a, b, h):
            r = [0] * (2 * e - 1)
            for i, x in enumerate(a):
                if x:
                    for j, y in enumerate(b):
                        r[i + j] = (r[i + j] + x * y) % p
            for k in range(2 * e - 2, e - 1, -1):
                c = r[k]
                if c:
                    for i in range(e + 1):
                        r[k - e + i] = (r[k - e + i] - c * h[i]) % p
            return r[:e]

        def powmod_t(k, h):
            res = [1] + [0] * (e - 1)
            base = [0, 1] + [0] * (e - 2)
            while k:
                if k & 1:
                    res = mulmod(res, base, h)
                base = mulmod(base, base, h)
                k >>= 1
            return res

        one = [1] + [0] * (e - 1)
        # order of t is exactly q-1  <=>  h is primitive (hence irreducible)
        for tail in itertools.product(range(p), repeat=e):
            h = list(tail) + [1]
            if h[0] == 0:
                continue
            if powmod_t(q1, h) != one:
                continue
            if all(powmod_t(q1 // r, h) != one for r in factors):
                return h
        raise RuntimeError("no primitive polynomial")

    def _tables(self):
        p, e, q = self.p, self.e, self.q
        if e == 1:
            g = (-self.h[0]) % p
            exp = [1]
            for _ in range(q - 2):
                exp.append(exp[-1] * g % p)
            self.exp = np.array(exp, dtype=np.int64)
        else:
            pw, ok = self._powers_of_t(self.h)
            assert ok and len(pw) == q - 1
            self.exp = np.array([sum(d * p ** i for i, d in enumerate(v)) for v in pw], dtype=np.int64)
        self.log = np.full(q, -1, dtype=np.int64)
        self.log[self.exp] = np.arange(q - 1)
        assert len(set(self.exp.tolist())) == q - 1

    # --- arithmetic (vectorised) ---------------------------------------
    def digits(self, a):
        a = np.asarray(a, dtype=np.int64)
        return (a[..., None] // self.powp) % self.p

    def from_digits(self, d):
        return (np.asarray(d, dtype=np.int64) % self.p) @ self.powp

    def add(self, a, b):
        return self.from_digits(self.digits(a) + self.digits(b))

    def neg(self, a):
        return self.from_digits(-self.digits(a))

    def add_const(self, a, c):
        """a + c with c in F_p (an integer)."""
        a = np.asarray(a, dtype=np.int64)
        d0 = a % self.p
        return a - d0 + (d0 + c) % self.p

    def mul(self, a, b):
        a = np.asarray(a, dtype=np.int64); b = np.asarray(b, dtype=np.int64)
        r = self.exp[(self.log[a] + self.log[b]) % (self.q - 1)]
        return np.where((a == 0) | (b == 0), 0, r)

    def power(self, a, k):
        a = np.asarray(a, dtype=np.int64)
        r = self.exp[(self.log[a] * k) % (self.q - 1)]
        if k == 0:
            return np.ones_like(a)
        return np.where(a == 0, 0, r)

    def chi(self, a):
        """Quadratic character (q odd)."""
        a = np.asarray(a, dtype=np.int64)
        return np.where(a == 0, 0, np.where(self.log[a] % 2 == 0, 1, -1))

    def sqrt(self, a):
        """One square root of a square a (a != 0)."""
        a = np.asarray(a, dtype=np.int64)
        return np.where(a == 0, 0, self.exp[(self.log[a] // 2) % (self.q - 1)])

    def elements(self):
        return np.arange(self.q, dtype=np.int64)


def _prime_factors(n):
    out, d = set(), 2
    while d * d <= n:
        while n % d == 0:
            out.add(d); n //= d
        d += 1
    if n > 1:
        out.add(n)
    return out


@lru_cache(maxsize=None)
def field(p, e):
    return FiniteField(p, e)


# ============================================================ polynomials over F_p
def poly_trim(a, p):
    a = [x % p for x in a]
    while len(a) > 1 and a[-1] == 0:
        a.pop()
    return a


def poly_mod(a, b, p):
    a, b = poly_trim(a, p), poly_trim(b, p)
    inv = pow(b[-1], -1, p)
    while len(a) >= len(b) and any(a):
        c = a[-1] * inv % p
        s = len(a) - len(b)
        for i in range(len(b)):
            a[s + i] = (a[s + i] - c * b[i]) % p
        a = poly_trim(a, p)
        if len(a) < len(b):
            break
    return a


def squarefree(f, p):
    f = poly_trim(f, p)
    df = poly_trim([(i * c) % p for i, c in enumerate(f)][1:], p)
    if df == [0]:
        return False
    a, b = f, df
    while any(b):
        a, b = b, poly_mod(a, b, p)
    return len(poly_trim(a, p)) == 1


# ============================================================ curves
class Hyperelliptic:
    """Smooth projective model of y^2 = f(x), f in F_p[x] squarefree, deg f in {2g+1, 2g+2},
    regarded over F_Q with Q = p^r (coefficients lie in F_p subset F_Q)."""

    def __init__(self, f, p, r=1, name=None):
        assert p % 2 == 1
        self.f = poly_trim(list(f), p)
        self.p, self.r, self.Q = p, r, p ** r
        self.d = len(self.f) - 1
        assert self.d >= 3 and squarefree(self.f, p), "need squarefree f of degree >= 3"
        self.g = (self.d - 1) // 2
        self.name = name or f"y^2 = {self.poly_str()} over F_{self.Q}"

    def poly_str(self):
        terms = []
        for i in range(self.d, -1, -1):
            c = self.f[i]
            if c == 0:
                continue
            cs = "" if (c == 1 and i > 0) else str(c)
            terms.append(f"{cs}x^{i}" if i > 1 else (f"{cs}x" if i == 1 else str(c)))
        return " + ".join(terms)

    def eval_f(self, F, xs):
        v = np.zeros_like(xs)
        for c in reversed(self.f):
            v = F.add_const(F.mul(v, xs), c)
        return v

    def count(self, m):
        """N_m = #C(F_{Q^m}) by exhaustive enumeration."""
        F = field(self.p, self.r * m)
        xs = F.elements()
        fx = self.eval_f(F, xs)
        affine = int(np.sum(1 + F.chi(fx)))
        if self.d % 2 == 1:
            inf = 1
        else:
            inf = 1 + int(F.chi(np.array([self.f[-1]]))[0])
        return affine + inf

    def counts(self, jmax):
        return [self.count(j) for j in range(1, jmax + 1)]

    def twist(self, nonsquare):
        """Quadratic twist y^2 = c f(x), c a non-square in F_p (valid model when r is odd)."""
        return Hyperelliptic([nonsquare * a for a in self.f], self.p, self.r, name=self.name + " (twist)")


# ============================================================ L-polynomials
def newton_b_from_s(s, n):
    """b_0..b_n from s_1..s_n (exact). Returns list of Fractions."""
    b = [Fraction(1)]
    for k in range(1, n + 1):
        acc = Fraction(s[k - 1])
        for j in range(1, k):
            acc += Fraction(s[j - 1]) * b[k - j]
        b.append(-acc / k)
    return b


def power_sums_from_L(L, m):
    """s_1..s_m of reciprocal roots of L (L[0] = 1), exact integers when L is integral."""
    L = [Fraction(x) for x in L]
    d = len(L) - 1
    s = []
    for j in range(1, m + 1):
        acc = j * (L[j] if j <= d else 0)
        for i in range(1, j):
            acc += (L[i] if i <= d else 0) * s[j - i - 1]
        s.append(-acc)
    return s


def L_from_counts(N, g, Q):
    """L(T) from N_1..N_g via s_k = Q^k + 1 - N_k, Newton and the functional equation.
    Returns (L as list of ints, integrality flag)."""
    s = [Q ** k + 1 - N[k - 1] for k in range(1, g + 1)]
    b = newton_b_from_s(s, g)
    integral = all(x.denominator == 1 for x in b)
    bi = [int(x) for x in b]
    L = bi + [Q ** (g - k) * bi[k] for k in range(g - 1, -1, -1)]
    return L, integral


def reciprocal_roots(L):
    """alpha_i with L(T) = prod (1 - alpha_i T)."""
    return np.roots([float(x) for x in L])


def normalised_spectrum(L, Q):
    return reciprocal_roots(L) / math.sqrt(Q)


def tau_from_counts(N, Q):
    return [(Q ** j + 1 - N[j - 1]) / Q ** (j / 2) for j in range(1, len(N) + 1)]


# ============================================================ decoder
def congruence_decoder(s_hat, g, Q):
    """Recover L from real estimates s_hat[k-1] of the integer power sums s_k (k = 1..g):
    round s_k inside the residue class fixed by Newton integrality, then exact Newton + FE.
    Exact whenever |s_hat_k - s_k| < k/2 for every k <= g and L is an integral Weil polynomial."""
    s, b = [], [Fraction(1)]
    for k in range(1, g + 1):
        R = int(sum(Fraction(s[j - 1]) * b[k - j] for j in range(1, k)))
        bk = round((-R - s_hat[k - 1]) / k)
        s.append(-R - k * bk)
        b.append(Fraction(bk))
    bi = [int(x) for x in b]
    return bi + [Q ** (g - k) * bi[k] for k in range(g - 1, -1, -1)]


def delta_cong(g, Q):
    return min(k / (2 * Q ** (k / 2)) for k in range(1, g + 1))


# ============================================================ realisations
J_cache = {}


def J(g):
    if g not in J_cache:
        Z, I = np.zeros((g, g)), np.eye(g)
        J_cache[g] = np.block([[Z, I], [-I, Z]])
    return J_cache[g]


def pair_spectrum(beta, tol=1e-7):
    """Split a multiset of unimodular numbers, closed under conjugation with +-1 of even
    multiplicity, into B = (beta_1..beta_g) such that spectrum = B u conj(B)."""
    beta = list(beta)
    B = []
    used = [False] * len(beta)
    order = np.argsort([-np.angle(z) for z in beta])
    for i in order:
        if used[i]:
            continue
        z = beta[i]
        used[i] = True
        # find conjugate partner
        cand = [k for k in range(len(beta)) if not used[k] and abs(beta[k] - np.conj(z)) < tol]
        if not cand:
            raise ValueError("spectrum not closed under conjugation")
        k = cand[0]
        used[k] = True
        B.append(z if np.angle(z) >= 0 else np.conj(z))
    return np.array(B)


def usp_realisation(L, Q):
    """Diagonal element diag(B, conj B) of USp(2g) with det(I - tU) = L(t/sqrt Q)."""
    beta = normalised_spectrum(L, Q)
    B = pair_spectrum(beta)
    return np.diag(np.concatenate([B, np.conj(B)]))


def haar_unitary(n, rng):
    z = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / math.sqrt(2)
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / abs(np.diag(r)))


def padded(U, n):
    d = U.shape[0]
    V = np.eye(n, dtype=complex)
    V[:d, :d] = U
    return V


# ============================================================ Hadamard test & amplitude estimation
def hadamard_state(X):
    """Real Hadamard-test state of Lemma 3.1 of [CP] for an n x n unitary X.
    Basis ordering: control (2) x register A (n) x register B (n)."""
    n = X.shape[0]
    Phi = np.eye(n, dtype=complex).reshape(n * n) / math.sqrt(n)        # sum_x |x>|x> / sqrt n
    phiX = (np.eye(n) @ np.eye(n)).reshape(n * n) * 0                       # placeholder
    phiX = np.kron(np.eye(n), X) @ Phi                                     # (I (x) X)|Phi>
    psi = np.concatenate([(Phi + phiX) / 2, (Phi - phiX) / 2])
    return psi


def good_projector_diag(n):
    """Good outcome = control qubit 0."""
    return np.concatenate([np.ones(n * n), np.zeros(n * n)])


def ae_distribution_statevector(psi, good, M):
    """Exact output distribution of canonical amplitude estimation (phase estimation of the
    Grover iterate G = (2|psi><psi| - I)(I - 2 Pi_good) with an M-point register),
    obtained by evolving the actual state: amplitude(y) = (1/M) sum_t e^{-2 pi i y t / M} G^t psi."""
    D = psi.shape[0]
    V = np.empty((M, D), dtype=complex)
    v = psi.copy()
    for t in range(M):
        V[t] = v
        w = v * (1 - 2 * good)                       # (I - 2 Pi_good)
        v = 2 * psi * np.vdot(psi, w) - w            # (2|psi><psi| - I)
    out = np.fft.fft(V, axis=0) / M
    return np.sum(np.abs(out) ** 2, axis=1)


def ae_distribution_analytic(a, M):
    """Analytic law of canonical amplitude estimation for amplitude a = sin^2(theta)."""
    theta = math.asin(math.sqrt(min(max(a, 0.0), 1.0)))
    y = np.arange(M)
    P = np.zeros(M)
    for phi in (theta / math.pi, 1 - theta / math.pi):
        x = phi - y / M
        num = np.sin(M * math.pi * x) ** 2
        den = (M ** 2) * np.sin(math.pi * x) ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(np.abs(np.sin(math.pi * x)) < 1e-15, 1.0, num / den)
        P += 0.5 * term
    return P / P.sum()


def ae_estimate_values(M):
    return np.sin(math.pi * np.arange(M) / M) ** 2


def ae_error_bound(a, M):
    """Brassard-Hoyer-Mosca-Tapp (2002), Thm 12 with k = 1."""
    return 2 * math.pi * math.sqrt(a * (1 - a)) / M + math.pi ** 2 / M ** 2


def median_failure_probability(p_single_fail, r):
    """Exact probability that the median of r independent runs fails, if each run fails
    independently with probability at most p_single_fail (r odd)."""
    from math import comb
    k0 = r // 2 + 1
    return sum(comb(r, k) * p_single_fail ** k * (1 - p_single_fail) ** (r - k) for k in range(k0, r + 1))


def runs_for_confidence(eta):
    """Odd r with exp(-2 r (1/2 - (1 - 8/pi^2))^2) <= eta (Hoeffding)."""
    gap = 0.5 - (1 - 8 / math.pi ** 2)
    r = math.ceil(math.log(1 / eta) / (2 * gap * gap))
    return r if r % 2 == 1 else r + 1


def M_for_precision(n, delta):
    """Smallest power of two M >= 4 pi n / delta (Theorem 1)."""
    target = 4 * math.pi * n / delta
    return 1 << math.ceil(math.log2(target))


# ============================================================ standard test curves
def test_curves():
    """Deterministic set of curves used across the scripts (genus 1, 2, 3; Q = p and Q = 9)."""
    C = []
    # genus 1
    C += [Hyperelliptic([1, 1, 0, 1], 5), Hyperelliptic([1, 1, 0, 1], 7), Hyperelliptic([3, 1, 0, 1], 11),
          Hyperelliptic([2, 5, 0, 1], 13), Hyperelliptic([0, -1, 0, 1], 3, name="y^2 = x^3 - x over F_3 (supersingular)"),
          Hyperelliptic([0, -1, 0, 1], 3, r=2, name="y^2 = x^3 - x over F_9 (eigenvalue -1 twice)")]
    # genus 2
    C += [Hyperelliptic([0, -1, 0, 0, 0, 1], 3, name="y^2 = x^5 - x over F_3 ([CP] Prop. 6.2)"),
          Hyperelliptic([1, 1, 0, 0, 0, 1], 5), Hyperelliptic([2, 0, 1, 0, 0, 1], 7),
          Hyperelliptic([1, 0, 1, 1, 0, 0, 2], 5, name="y^2 = 2x^6 + x^3 + x^2 + 1 over F_5 (even degree, non-square lead)"),
          Hyperelliptic([1, 0, 1, 0, 0, 1], 3, r=2)]
    # genus 3
    C += [Hyperelliptic([1, 1, 0, 0, 0, 0, 0, 1], 3), Hyperelliptic([2, 1, 0, 0, 1, 0, 0, 1], 5)]
    return C


def jmax_for(curve, cap=200_000):
    j = 1
    while curve.Q ** (j + 1) <= cap:
        j += 1
    return j


# ============================================================ exact Weil test and Weil classes
def is_weil_exact(L, Q):
    """Exact test that every reciprocal root of the integer polynomial L (with functional
    equation) has modulus sqrt(Q): writing L(T)/T^g = b_g + sum_{k<g} b_k V_{g-k}(y),
    y = 1/T + Q T, V_n = y V_{n-1} - Q V_{n-2}, the condition is that h(y) has g real roots
    (with multiplicity) in [-2 sqrt Q, 2 sqrt Q]."""
    import sympy as sp
    y = sp.Symbol("y")
    g = (len(L) - 1) // 2
    V = [sp.Integer(2), y]
    for n in range(2, g + 1):
        V.append(sp.expand(y * V[-1] - Q * V[-2]))
    h = sp.Poly(L[g] + sum(L[k] * V[g - k] for k in range(g)), y)
    _, facs = h.factor_list()
    nreal = 0
    for f, mult in facs:
        rr = f.real_roots()
        if len(rr) != f.degree():
            return False
        if f == sp.Poly(y ** 2 - 4 * Q, y):
            nreal += 2 * mult
            continue
        for r_ in rr:
            if r_.is_rational:
                if r_ ** 2 > 4 * Q:
                    return False
            elif sp.N(r_ ** 2 - 4 * Q, 60) > 0:
                return False
        nreal += f.degree() * mult
    return nreal == g


def weil_class(g, Q):
    """All integer Weil polynomials of genus g (b_0 = 1, functional equation, roots of modulus sqrt Q)."""
    import itertools
    out = []
    ranges = [range(-math.floor(math.comb(2 * g, k) * Q ** (k / 2)), math.floor(math.comb(2 * g, k) * Q ** (k / 2)) + 1)
              for k in range(1, g + 1)]
    for bs in itertools.product(*ranges):
        b = [1] + list(bs)
        L = b + [Q ** (g - k) * b[k] for k in range(g - 1, -1, -1)]
        r = np.roots([L[k] / Q ** (k / 2) for k in range(2 * g + 1)][::-1])
        if np.max(np.abs(np.abs(r) - 1)) < 1e-2 and is_weil_exact(L, Q):
            out.append(tuple(L))
    return out


def random_curves(rng, g, p, count, even_degree=False):
    out, seen = [], set()
    d = 2 * g + 2 if even_degree else 2 * g + 1
    tries = 0
    while len(out) < count and tries < 10000:
        tries += 1
        f = [int(x) for x in rng.integers(0, p, size=d)] + [int(rng.integers(1, p))]
        if tuple(f) in seen or not squarefree(f, p):
            continue
        seen.add(tuple(f))
        out.append(Hyperelliptic(f, p))
    return out


def save_json(path, obj):
    import json
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, Fraction):
            return str(o)
        if isinstance(o, complex):
            return [o.real, o.imag]
        raise TypeError(type(o))
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=conv)


def exact_spectrum(L, Q, digits=30):
    """Normalised reciprocal roots alpha/sqrt(Q) of L with multiplicity, computed factor by
    factor over Q (each irreducible factor has simple roots, so high accuracy)."""
    import sympy as sp
    T = sp.Symbol("T")
    P = sp.Poly(sum(int(c) * T ** k for k, c in enumerate(L)), T)
    _, facs = P.factor_list()
    out = []
    for f, mult in facs:
        if f.degree() == 0:
            continue
        # reciprocal roots of f are roots of its reversal
        rev = sp.Poly(list(reversed(f.all_coeffs())), T)
        for z in rev.nroots(n=digits):
            out += [complex(z) / math.sqrt(Q)] * mult
    return np.array(out)


def usp_realisation_exact(L, Q):
    beta = exact_spectrum(L, Q)
    B = pair_spectrum(beta, tol=1e-9)
    return np.diag(np.concatenate([B, np.conj(B)]))


# ============================================================ points and the Frobenius permutation
def curve_points(C, m):
    """All points of the smooth projective model over F_{Q^m}: affine (0, x, y) and points at
    infinity (1, s, 0) with s^2 = lead (even degree) or (1, 0, 0) (odd degree)."""
    F = field(C.p, C.r * m)
    xs = F.elements()
    fx = C.eval_f(F, xs)
    pts = []
    for x, v in zip(xs.tolist(), fx.tolist()):
        if v == 0:
            pts.append((0, x, 0))
        elif F.chi(np.array([v]))[0] == 1:
            y = int(F.sqrt(np.array([v]))[0])
            pts.append((0, x, y)); pts.append((0, x, int(F.neg(np.array([y]))[0])))
    if C.d % 2 == 1:
        pts.append((1, 0, 0))
    else:
        c = np.array([C.f[-1]])
        if F.chi(c)[0] == 1:
            s = int(F.sqrt(c)[0])
            pts.append((1, s, 0)); pts.append((1, int(F.neg(np.array([s]))[0]), 0))
    return F, pts


def frobenius_permutation(C, m):
    """perm[i] = index of phi(P_i), phi = coordinatewise x -> x^Q, on X_m = C(F_{Q^m})."""
    F, pts = curve_points(C, m)
    idx = {P: i for i, P in enumerate(pts)}
    perm = []
    for (tag, a, b) in pts:
        a2 = int(F.power(np.array([a]), C.Q)[0]); b2 = int(F.power(np.array([b]), C.Q)[0])
        perm.append(idx[(tag, a2, b2)])
    return F, pts, np.array(perm)


def mobius(n):
    res, d, k = 1, 2, n
    while d * d <= k:
        if k % d == 0:
            k //= d
            if k % d == 0:
                return 0
            res = -res
        d += 1
    if k > 1:
        res = -res
    return res
