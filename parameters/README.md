# RiVeR Parameter-Setting Artifact

This artifact contains the final RiVeR OOM parameter-setting code and data for reproduction and inspection.
It is intentionally curated: macOS metadata, Python caches, and the very large challenge
precomputation tables are not included.

The retained OOM ring-count values are `N in {8,16,64,128,256}`.

## Quick Start

Run the deterministic table and repetition checks:

```bash
make table-check
```

Run all six checks, including the bundled estimators and expanded finite-grid
diagnostic:

```bash
make check
```

The estimator step may print progress lines such as
`running selector A-MSIS estimator for N=...`. The deterministic table,
product-threshold, estimator-rerun, expanded minimality, and all-parameter
generation steps should report `PASS`; see "Search Scope And Minimality
Diagnostic" below for the finite search scope.

```json
{"repeat_target": 10.0, "rows": 5, "status": "PASS"}
{"rows": 5, "status": "PASS"}
{"rows": 5, "status": "PASS"}
{"rows": 5, "status": "PASS"}
{"smaller_full_pass_rows": 0, "status": "PASS"}
{"markdown": "report/final_oom_all_parameters.md", "rows": 5, "status": "PASS", "tsv": "data/final_oom_all_parameters.tsv"}
```

The fourth command reruns the real LWE estimator and the bundled selector A-MSIS estimator. The
fifth command evaluates the finite-grid minimality claim and may print progress lines such as
`minimality search N=...`. These two Sage commands are the slowest steps.
`--skip-msis` is available only for local diagnostics and deliberately does
not produce top-level `PASS`.

## Requirements

- SageMath, with `sage` available on `PATH`.
- Python 3 for the non-Sage helper scripts.
- No network access is required.  The LWE estimator and A-MSIS estimator source used by the artifact
  are bundled under `external/`.

## Artifact Layout

| Path | Purpose |
|---|---|
| `README.md` | Reproduction guide. |
| `Makefile` | Deterministic/full checks and artifact-safe cleanup. |
| `data/final_oom_parameters.tsv` | Compact final table, including explicit `p`, `q`, and `hat_q`. |
| `data/final_oom_parameters.json` | Same compact final rows plus formula metadata. |
| `data/final_oom_security.json` | Deterministic recomputation of moduli, bounds, repeat accounting, checked instances, and pass/fail flags. |
| `data/final_oom_estimator_rerun.json` | Actual estimator rerun output. |
| `data/final_oom_all_parameters.tsv` | Complete flat all-parameter table. |
| `data/oom_search_minimality_diagnostic.json` | Finite-grid diagnostic for the selected OOM rows; its current status is `PASS`. |
| `data/product_tau_validation.csv` | One-million-trial product-threshold validation inputs for `epsilon_g^U`. |
| `data/product_tau_validation_metadata.json` | Seed rule and source metadata for the product-threshold validation runs. |
| `report/final_oom_parameters.md` | Generated compact verification report. |
| `report/final_oom_all_parameters.md` | Generated complete human-readable report. |
| `scripts/reproduce_final_table.py` | Regenerates compact final rows and compact report. |
| `scripts/river_oom_math_checks.sage.py` | Recomputes deterministic formulas and writes `data/final_oom_security.json`. |
| `scripts/validate_product_tau_inputs.py` | Checks `tau_g0/tau_g1`, failures/trials, and Clopper-Pearson upper bounds from `data/product_tau_validation.csv`. |
| `scripts/estimate_public_restart_bounds.sage.py` | Deterministically regenerates the product-threshold aggregates from the recorded seed rule; its default five-million-trial run is intentionally separate from the fast checks. |
| `scripts/run_final_oom_estimators.sage.py` | Reruns MLWR/MLWE and selector A-MSIS estimator checks. |
| `scripts/verify_oom_search_minimality.sage.py` | Enumerates the finite OOM search grid and checks that no strictly smaller row passes all checks. |
| `scripts/make_all_parameters_table.py` | Builds `data/final_oom_all_parameters.tsv` and `report/final_oom_all_parameters.md`. |
| `scripts/run_all_checks.sh` | Convenience wrapper for the six final OOM reproduction commands. |
| `external/lattice-estimator/` | Bundled LWE estimator package used by the OOM scripts. |
| `external/lattice-estimator/LANES.sage` | Standalone LANES size and KLSS/MSIS/MLWE cross-check script. |
| `external/lattice-estimator/UPSTREAM.txt` | Upstream commit provenance for the bundled lattice-estimator snapshot. |
| `external/lattice-estimator/COPYING.LESSER-3.0.txt` | LGPLv3 license text referenced by the upstream lattice-estimator README. |
| `external/msis-security/` | Bundled selector A-MSIS estimator backend. |
| `external/msis-security/UPSTREAM.txt` | Upstream commit, file hashes, and artifact-local modification notes for the A-MSIS helpers. |
| `external/msis-security/LICENSE-BSD-0-Clause.txt` | Upstream BSD-0-Clause license for the A-MSIS helper source. |
| `optional_challenge_invertibility/` | Optional MatRiCT+ challenge-difference invertibility scripts; not needed for OOM table reproduction. |

The parameter package deliberately contains no collaborator change log,
generated file manifest, or checksum list. The package itself is the artifact
distributed by the surrounding repository.

`THIRD_PARTY.md` records the upstream provenance, artifact-local modifications,
and redistribution licenses for both bundled estimator components.

## Final OOM Parameters

The retained OOM ring-count values are `N in {8,16,64,128,256}`. `N=32` is intentionally excluded.
The table below contains the parameter values that are used by the final OOM rows; longer diagnostic fields are in `data/final_oom_all_parameters.tsv`.

| N | d | w | gamma | q0 | B_e | beta | r' | n | ell | hat n | hat k | K_b | K_a | s_c | phi_a | phi_s | phi_m | phi_b | tau_g0 | tau_g1 | OOM KiB | mu_RiVeR |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 32 | 32 | 16 | 61 | 30 | 1 | 1 | 44 | 54 | 42 | 46 | 5 | 28 | 3 | 32 | 26 | 32 | 2 | 3.14 | 2.68 | 20.133209 | 8.344282 |
| 16 | 32 | 32 | 16 | 61 | 30 | 1 | 1 | 41 | 59 | 43 | 49 | 5 | 28 | 3 | 40 | 22 | 32 | 2 | 3.09 | 3.08 | 21.409120 | 8.435179 |
| 64 | 32 | 32 | 16 | 61 | 30 | 1 | 1 | 44 | 54 | 50 | 51 | 5 | 28 | 3 | 34 | 24 | 32 | 2 | 3.05 | 3.33 | 25.535994 | 8.625686 |
| 128 | 32 | 32 | 16 | 61 | 30 | 1 | 1 | 45 | 54 | 49 | 51 | 5 | 28 | 3 | 24 | 34 | 32 | 2 | 3.09 | 3.58 | 28.952209 | 8.598617 |
| 256 | 32 | 32 | 16 | 61 | 30 | 1 | 1 | 42 | 59 | 48 | 52 | 5 | 28 | 3 | 22 | 40 | 32 | 2 | 3.06 | 3.84 | 36.040999 | 8.526236 |

The concrete Rej1 definition fixes its internal constant to `12` globally.
In this artifact, the exact-layer repetition factor is `1`, so
`mu_RiVeR = mu_OOM`.

## Selected Moduli

The selected outer prime `p` has split factor 2 and satisfies `p == 5 mod 8`.
The exact modulus is `q = p q_0`, with fixed `q_0 = 61 == 5 mod 8`.  The selector
modulus `hat_q` is prime and satisfies `hat_q == 5 mod 8`.

| N | p | q | hat q | p mod 8 | q0 mod 8 | hat q mod 8 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 17592186043877 | 1073123348676497 | 8796093022237 | 5 | 5 | 5 |
| 16 | 281474976710597 | 17169973579346417 | 35184372088997 | 5 | 5 | 5 |
| 64 | 17592186043877 | 1073123348676497 | 140737488355333 | 5 | 5 | 5 |
| 128 | 17592186043877 | 1073123348676497 | 140737488355333 | 5 | 5 | 5 |
| 256 | 281474976710597 | 17169973579346417 | 281474976710677 | 5 | 5 | 5 |

## Formulas Recomputed

The deterministic checker recomputes these quantities for every row:

```text
B_s     = w gamma B_e sqrt(d(n+ell))
eta_m   = w gamma B_e sqrt(d)
phi_m   = max { phi in Z_>0 : 67107713 > 24 phi eta_m }
sigma_s = phi_s B_s
sigma_m = phi_m eta_m
B_response = w gamma sqrt(d(ell beta^2 + (n+r') B_e^2))

beta_sis_1 = 2.4 sqrt(sigma_s^2 (ell+n) d)
beta_sis_2 = 2.4 sqrt(d(ell+n) sigma_s^2 + d sigma_m^2)
beta_sis   = max(4 w gamma beta_sis_1, beta_sis_1 + 2 B_response)
```

It also checks the auxiliary exact-response condition:

```text
q > max(beta_sis_2, 12 sigma_s, 12 sigma_m).
```

## Search Scope And Minimality Diagnostic

This artifact includes a finite-grid check of the selected OOM rows. The script
`scripts/verify_oom_search_minimality.sage.py` encodes the search grid explicitly, enumerates every
candidate in that grid, and checks every candidate whose OOM communication is strictly smaller than
the selected row.  A smaller candidate is rejected only if it fails a deterministic parameter-setting
condition or one of the real estimator delta checks. The generated diagnostic is
`data/oom_search_minimality_diagnostic.json`.

The grid includes values below each selected `hat_n` and `hat_k`, including
the two largest profiles. No strictly smaller row in this explicit grid passes
both the deterministic conditions and the estimator-delta checks. This is a
finite-grid certificate for the published search space, not a proof of global
optimality outside that space.

| N | grid rows | strictly smaller rows | deterministic-pass smaller rows | estimator-fail smaller rows | full-pass smaller rows |
|---:|---:|---:|---:|---:|---:|
| 8 | 12096 | 2230 | 664 | 664 | 0 |
| 16 | 19200 | 5484 | 358 | 358 | 0 |
| 64 | 2880 | 1994 | 1154 | 1154 | 0 |
| 128 | 7938 | 414 | 341 | 341 | 0 |
| 256 | 11520 | 2772 | 1214 | 1214 | 0 |

The selector product bounds are:

```text
B_a       = gamma sqrt(2w)
sigma_a   = phi_a B_a
mathcal_B = gamma w sqrt(d hat_k)
B_g,0     = tau_g0 d(N-1)(phi_a B_a)^2 / 3
B_g,1     = tau_g1 d(phi_a B_a)^2 / 2
```

Both `B_g,0` and `B_g,1` are recomputed and recorded for each row.

The communication estimate uses the current response split:

```text
z_s,z_key : d(ell+n) coefficients encoded at entropy h(sigma_s)
z_eval    : d coefficients encoded at entropy h(sigma_m)
```

## Repeat Accounting

The OOM expected-attempt bound is:

```text
mu_OOM = mu_a mu_b mu_s mu_m
  / ((1-epsilon_a)(1-epsilon_b)
     ((1-epsilon_s)(1-epsilon_m)-epsilon_2)
     (1-epsilon_g^U)(1-epsilon_cmp^U)).
```

The infinity-tail terms are:

```text
epsilon_a_tail = 2 d (N-1) exp(-18)
epsilon_b_tail = 2 d hat_k exp(-18)
epsilon_s_tail = 2 d (n+ell) exp(-18)
epsilon_m_tail = 2 d exp(-18).
```

The joint Euclidean response check contributes:

```text
t2 = 1.2 sqrt((ell+n+(sigma_m/sigma_s)^2)/(ell+n+1)),
epsilon_2 <= t2^(d(ell+n+1)) exp(d(ell+n+1)(1-t2^2)/2).
```

The response checks are applied sequentially, with each failure probability
conditioned on preceding checks having succeeded. Their success probabilities
therefore multiply without an independence assumption. The checker also verifies both
`sigma_m <= sigma_s` and:

```text
1.2 sqrt(sigma_s^2 d(ell+n) + sigma_m^2 d)
  = t2 sigma_s sqrt(d(ell+n+1)).
```

The product-threshold term `epsilon_g^U` is not hand-filled: it is checked from
`data/product_tau_validation.csv`, which records `1,000,000` fresh validation trials per row.
The script recomputes the Clopper-Pearson upper bound from the recorded failures and confirms it
matches the final `epsilon_g^U`.  The final artifact uses the scale-free `tau_g0/tau_g1` values and recomputes
`B_g,0/B_g,1` from the final `phi_a`.
The companion file `data/product_tau_validation_metadata.json` records the source-script name,
default seed, per-row seed rule, validation grouping, chunk size, product block size, and
confidence-bound convention. The supplied N=8 and N=16 runs restart their row index and therefore
share an initial seed and challenge prefix; their streams diverge after Gaussian sampling, and no
cross-profile independence is used in the per-row confidence bounds. Each run is named explicitly
in the CSV rather than hidden behind one generic source label.

The supplied experiment samples from a centered Gaussian cut at `16 sigma` without conditioning on
the protocol's preceding `6 sigma` infinity check. It is therefore reproduced as an unconditioned,
wider-support proxy, not asserted to equal the conditional probability in the protocol analysis.
Raw sample vectors are not stored. The fast checks validate the recorded aggregates and independently
recompute their Clopper--Pearson bounds; the full experiment can be replayed explicitly with
`make product-check` (five million trials, so it is not part of `make check`). Integer failure counts
must match exactly. The floating quantile is compared within `5e-15`, since different
numerical-library builds can differ in its last few digits.

The compression-stability term `epsilon_cmp^U` is deterministic/modelled, not a Monte Carlo input.
It is recomputed by exactly counting residues that simultaneously satisfy the
low-bit and centered-modulus predicates; multiplying their marginals would not
be justified because both inspect the same residue. The resulting
per-coefficient probability is raised to `hat_n*d` under the artifact's
independent-uniform coefficient model; the exact residue count does not by
itself prove that distributional assumption for the protocol transcript.

## Security Instances Checked

The artifact's row-selection acceptance criterion for every reported root-Hermite factor is `delta <= 1.004690`.
This is a stated root-Hermite-factor convention, not an independent reproduction
of a 128-bit work factor. Diagnostic cost-bit fields are not acceptance tests.
The rows below follow the assumption order used by the parameter setting and use the paper-level
instance names.

| Paper instance | Estimator/check used |
|---|---|
| `MLWR_{p,q,ell,n,U_beta,U_{B_e}}` | Expanded to LWE with dimension `d ell`, modulus `q`, secret `U_beta`, error `U_{B_e}`; checked with `estimator.LWE.primal_usvp`. Arora-GB bits are diagnostic only. |
| `MLWE_{q,ell+r',n,U_{B_e},U_beta}` | Expanded to LWE with coefficient dimension `d(ell+r')` and `dn` samples; checked with `estimator.LWE.dual`. |
| `MSIS_{q,n,n+ell+r',beta_SIS}` | Systematic matrix `[A | -I_n | B]`; checked by `q > beta_SIS` and the MR09 l2 root-Hermite-factor test. |
| `MSIS_{q,n+r',ell+n+r',beta_{SIS,2}}` | Auxiliary exact-response MSIS check; verifies `q > max(beta_{SIS,2}, 12 sigma_s, 12 sigma_m)` and the MR09 required-delta test. |
| `MLWE_{hat q,hat k,hat n,U_beta,D_{2^{K_b}/sqrt(12)}}` | Expanded to LWE with dimension `d hat k`, `d hat n` samples, modulus `hat q`, secret `U_beta`, and error width `2^{K_b}/sqrt(12)`; checked with `estimator.LWE.dual`. |
| `A-MSIS^infty_{hat n,m_sel,hat q,beta_sel}` | Merged five-block selector-binding instance over `hat q`; checked with the bundled `MSIS_security.MSIS_summarize_attacks`. Cost bits are diagnostic; acceptance uses delta. |

## Security Estimates

The table follows the artifact's stated row-selection rule
`delta <= 1.004690`. Values are root-Hermite factors unless the metric column
says otherwise. This convention is not a standalone reproduction of a
128-bit work factor; the estimator inputs, cost model, and acceptance rule
must be read together.

| Paper instance | Metric | N=8 | N=16 | N=64 | N=128 | N=256 | Target/check |
|---|---|---:|---:|---:|---:|---:|---|
| `MLWR_{p,q,ell,n,U_beta,U_{B_e}}` | delta | 1.00464751988 | 1.00461836234 | 1.00464751988 | 1.00464751988 | 1.00461836234 | `<= 1.004690` |
| `MLWE_{q,ell+r',n,U_{B_e},U_beta}` | delta | 1.00446957841 | 1.00436489062 | 1.00446957841 | 1.00448758225 | 1.00438195254 | `<= 1.004690` |
| `MSIS_{q,n,n+ell+r',beta_SIS}` | `q / beta_SIS` | 174.328275 | 3230.461414 | 188.855631 | 131.963293 | 1759.162156 | `> 1` |
| `MSIS_{q,n+r',ell+n+r',beta_{SIS,2}}` | required delta | 1.00254658498 | 1.00249321187 | 1.0025284924 | 1.00255321275 | 1.00256943698 | `<= 1.004690` |
| `MSIS_{q,n+r',ell+n+r',beta_{SIS,2}}` | `q / max(beta_{SIS,2},12sigma_s,12sigma_m)` | 356996.154447 | 6615285.214404 | 386740.539827 | 270248.611939 | 3602651.084433 | `> 1` |
| `MLWE_{hat q,hat k,hat n,U_beta,D_{2^{K_b}/sqrt(12)}}` | delta | 1.00468705157 | 1.00460873487 | 1.00465733193 | 1.00465733193 | 1.00466719099 | `<= 1.004690` |
| `A-MSIS^infty_{hat n,m_sel,hat q,beta_sel}` | delta | 1.00466719099 | 1.00467709742 | 1.00462803543 | 1.00468705157 | 1.00465733193 | `<= 1.004690` |

The JSON payloads keep the exact estimator inputs under `checked_instances`. The full TSV also records block sizes, MR09 sides, selector bounds, and diagnostic cost bits.

## Repeat And Size Summary

| N | mu_a | mu_b | mu_s | mu_m | epsilon_g^U | epsilon_cmp^U | log2 epsilon_2 | response success | success denominator | mu_RiVeR | OOM KiB | pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 8 | 1.455702 | 2.266297 | 1.587687 | 1.455702 | 0.007953415 | 0.078766056 | -162.321916 | 0.999903503 | 0.913772 | 8.344282 | 20.133209 | yes |
| 16 | 1.350281 | 2.266297 | 1.727176 | 1.455702 | 0.007791315 | 0.080562037 | -165.855193 | 0.999901554 | 0.912128 | 8.435179 | 21.409120 | yes |
| 64 | 1.423863 | 2.266297 | 1.650153 | 1.455702 | 0.008953919 | 0.093047660 | -162.348550 | 0.999903503 | 0.898645 | 8.625686 | 25.535994 | yes |
| 128 | 1.650153 | 2.266297 | 1.423863 | 1.455702 | 0.007711270 | 0.091274372 | -163.995628 | 0.999902528 | 0.901474 | 8.598617 | 28.952209 | yes |
| 256 | 1.727176 | 2.266297 | 1.350281 | 1.455702 | 0.008518571 | 0.089497535 | -167.446394 | 0.999900579 | 0.902386 | 8.526236 | 36.040999 | yes |

## LANES Size Cross-Check

`external/lattice-estimator/LANES.sage` is included as a standalone LANES size and
KLSS/MSIS/MLWE cross-check.  It is separate from the final OOM table reproduction above.
Run it from the bundled lattice-estimator directory so that Sage can import the local `estimator`
package:

```bash
cd external/lattice-estimator
sage LANES.sage
```

The default run checks only the selected admissible LANES profile and prints `status = PASS`.
The selected profile is
`(d_hat,n_hat,ell_hat,N,alpha,D) = (256,4,4,6,3,17)`, with
`q_hat = 67107713`, `q_hat prime = yes`, `q_hat mod 256 = 129`, `w_hat = 44`,
`delta_MSIS = 1.00373193`, `delta_MLWE = 1.00399599`, and
LANES Eq.(12) size `13.5050 KiB`.  The script includes the selected modulus in
its final PASS condition by checking `q_hat.is_prime()` and
`q_hat % (4*L_split) == 2*L_split + 1`.
The MLWE diagnostic uses coefficient dimensions `(n,m)=(1024,3328)` and only accepts finite-cost
estimator attacks when reporting a delta.
This LANES script is included to document the LANES-size calculation; it is not used to generate
`data/final_oom_all_parameters.tsv`.

## Optional Challenge Invertibility Scripts

The `optional_challenge_invertibility/` directory contains the MatRiCT+ Sage scripts used for
challenge-difference invertibility experiments.  These scripts are not required to reproduce the OOM
parameter table above.  The large precomputed `STable_*.csv` files are intentionally omitted from this
artifact because they are hundreds of megabytes and can be regenerated by the precompute scripts.
The bundled d=256 scripts are configured for the current LANES modulus `q_hat=67107713`.
For `d=256`, `L=64`, and `w=11`, `d256_current_q_result.txt` records
`logp=-90.5`. The fast one-command check pins the recorded modulus and
value; independently re-deriving the value requires regenerating the
omitted large S-table.
