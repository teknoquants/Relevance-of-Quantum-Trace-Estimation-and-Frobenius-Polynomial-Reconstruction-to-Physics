# Relevance to Physics — validation script

Checks every computable claim of the white paper *Relevance of Quantum Trace Estimation and Frobenius Polynomial
Reconstruction to Physics: A Formal Statement* (R. Cooperman and A.E. Petrunina V.V., 11 October 2026).

| Check | Statement | What is tested |
|---|---|---|
| A | 1 (DQC1) | Hadamard-test acceptance probability = 1/2 + Re Tr(X)/(2n), Haar-random X, n = 2..16 |
| B | 2 (Heisenberg scaling) | queries for precision eps: amplitude estimation (exact law) vs independent sampling (exact binomial); log-log slopes |
| C | 3 (ill-conditioning) | two-point construction U_0 = I, U_theta = e^{i theta} I in 60-digit arithmetic |
| D | 4 (Hilbert-Polya form) | zeros of zeta_C(s) at s = 1/2 + i(theta + 2 pi k)/log Q for 13 curves |
| E | 5 (symplectic moments) | Weyl-integration moments on USp(2g), g = 1..4; averages over all elliptic curves over F_p, p = 101, 211, 401 |
| F | 6 (free-field identities) | exact rational checks of det(I - zA)^{-1} = exp(sum s_m z^m/m) = sum h_k z^k and det(I - zA) = sum (-1)^k e_k z^k |

Scope: the checks test the statements on finite instances. They do not prove the asymptotic rates of Statement 2
or the equidistribution theorem of Statement 5 (cited from Katz-Sarnak).

## Run
    python3 -m pip install -r requirements.txt
    python3 -I physics_relevance_checks.py      # about 10 s; writes results/physics_checks.json

`results/physics_checks.json` and `results/physics_checks.log` are the outputs of the reference run;
`SHA256SUMS` covers every file (verify with `sha256sum -c SHA256SUMS`).

## License
MIT License, Copyright (c) 2026 Roger Cooperman; A.E. Petrunina V.V. See `LICENSE`.
