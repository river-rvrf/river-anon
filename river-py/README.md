# river-py

A dependency-free **executable prototype** of RiVeR's outer protocol,
serialization and test-vector machinery, with a deterministic vector mode.

**Targets the published paper.**  The OOM response is split as
`r_0 = (s, e_key)` at `sigma_s` against `r_1 = e_eval` at `sigma_m`, with
four rejection samplers in the figure's order.

The outer protocol, codec, opening backend, vectors **and the LANES
parameters** are complete against it — the revision publishes the whole
Hint-MLWE chain in closed form and `lanes_params.py` re-derives every
printed figure.  The LANES *backend* is gated on security evidence; see
"The exact layer".

It is not a production or security-preserving RiVeR implementation.  The
`opening` backend carries the revised protocol end to end but deliberately
reveals its opening; the zero-knowledge `lanes` layer runs under
`exact_backend="lanes-experimental"`, with the production `"lanes"` name
**gated** on security evidence — see "The exact layer".  What it is good
for is the OOM and parameter layers, composition, wire format, and being
the byte-for-byte reference for `river-rs`.

**It makes no timing claim at all, and cannot.**  Python's integers are
arbitrary-precision with data-dependent limb counts, its comparisons branch
freely, and several hot loops skip zero coefficients.  No arrangement of
this code could carry a constant-time property, so none is attempted or
asserted — the deliberate posture is that this tree is the *specification
and the KAT oracle*, and `river-rs` enforces the constant-time behaviour of
secret-bearing arithmetic and sampling independently.  The two agree on
**bytes**; they are not expected to agree on timing, and a timing
difference between them is not a port defect.  `../river-rs/README.md`
§Timing states which properties that implementation holds itself to, and
which — the rejection loops and the whole-proof retry — are structural in
both.

Every source of randomness is a SHAKE-256 stream, so a complete execution —
setup, keys, evaluation, proof bytes — is reproducible *when asked for*.  `Eval`
defaults to fresh `os.urandom`; `eval_deterministic` is the explicit pinned
path, and is what `vectors.py` exports.  The default is fresh coins because a
seed argument that silently becomes the whole nonce is a footgun — see

## Quick start

```bash
cd river-py
make test            # unit / end-to-end tests
make test-all        # the same, plus every published profile end to end
make selftest        # per-module __main__ self-checks
make check-vectors   # re-derive and diff against the shipped vectors.json
```

A single evaluation:

```python
from params import get
from river import RiVeR

par = get("RiVeR-N8")
scheme = RiVeR(par)
pp = scheme.setup(b"\x00" * 32)

# A ring is an ordered tuple of exactly `N` distinct keys.  There is no
# padding and no canonical reordering: the order is part of the statement.
keys = [scheme.keygen(pp, bytes([i]) + b"\x00" * 31) for i in range(par.N)]
ring = [pk for _, pk in keys]
sk, pk = keys[1]

v, pi = scheme.eval(pp, pk, sk, ring, b"hello", b"\xAA" * 32)
assert scheme.verify(pp, ring, b"hello", v, pi)
```

`python params.py` prints every profile with its derived bounds;
`python river.py` runs one toy evaluation end to end.

## Modules

| file | contents |
|---|---|
| `manifest.py` | the wire-visible numeric choices, frozen per profile and field |
| `params.py` | the five published profiles plus a fast toy profile, `BoundGen`, concrete moduli, all derived bounds and the repetition estimate |
| `ring.py` | `R_q = Z_q[X]/(X^d+1)`: schoolbook and Kronecker multiplication, norms, rounding `floor(.)_p`, Power2Round |
| `sample.py` | SHAKE-256 counter-mode XOF, uniform / ternary / discrete-Gaussian samplers, hash-to-`C^d_{w,gamma}`, `Rej_1` and `Rej_2` |
| `dgs.py` | exact-`Decimal` Gaussian tail arithmetic: where the sampler truncates, and why that is not the verifier's `6 sigma` |
| `test_kat.py` | known-answer tests for the XOF, samplers, thresholds and codec, in dependency order |
| `codec.py` | the bit-level codec: `Uniform` / `Signed` / `Rice` coders, layout-driven encode and decode, and the Fiat–Shamir transcript |
| `exact.py` | the exact layer `Pi_ex`: radix-`(1,3,9,17)` range encoding, witness packing, BDLOP commitment over `R_q~`, backend interface |
| `oom.py` | the relaxed one-out-of-many proof: `Com`, `Prove`, `Ver` |
| `river.py` | ring admissibility, `Setup`, `KeyGen`, `Eval`, `Verify` |
| `vectors.py` | deterministic test-vector generator and re-derivation checker |
| `test_review.py` | exact properties of the parameter set and the challenge algebra the security argument rests on |
| `lanes_*.py` | **optional**: a port of the LANES exact proof, usable as a second `Pi_ex` backend (see below) |

Each module has a `__main__` block with self-checks that double as usage
examples.

## What is and is not implemented

**Implemented, matching the paper:** the parameter layer (all five published
profiles reproduce every derived column of the final table — `B`, `B_s`,
`B_g0`, `B_g1`, `beta_SIS,1`, `beta_SIS,2`, `beta_SIS`, `beta_sel,inf`, the
repeat bound, `|pi_OOM|` and the total — to the two significant figures they
are printed with), the outer MLWR key relation, exact-`N` ordered rings, the
complete OOM layer with its split `(z_s, z_m)` response and all four
rejection samplers, the centred `[[.]]_K` of the Preliminaries, the exact
relation `R^_ex` with its radix-3 range encoding and six padded `N_ex = 6`
message blocks, the two-stage commit ordering that binds `W` into `rho'`
before the challenge, serialization, and verification including the two
checks the figure omits.

## The exact layer

`Pi_ex` sits behind a backend interface, with three selectable names.

| backend | link | witness | state |
|---|---|---|---|
| `"opening"` (default) | over `Z` | **transmitted** | runs; a mock, about 9.3 KB |
| `"lanes-experimental"` | mod `q~` | not transmitted | runs, at the paper's parameters; about 13.9 KB |
| `"lanes"` | mod `q~` | not transmitted | **gated** on security evidence — see below |

**`opening`** is a mock.  It enforces every clause of the relation, including
the integer link, but `sigma_ex` *is* the opening: `e_eval` leaks, and with it
`<G(m), s>`, so the secret key falls out after about `ell` distinct messages.
It carries the revised protocol end to end, which is what it is for.

**`lanes`** is a port of [ENS20].  Every layer of it runs at the paper's
own parameters, and `exact_backend="lanes-experimental"` exercises it end
to end — commit, product proof, linear proof, hint recovery, serialization
and verification — at `RiVeR-TOY` and `RiVeR-N8`, **byte-exactly against
`river-rs`**.  `vectors.json` ships both cases and
`../river-rs/tests/sampler_kat.json` bisects them through its
`lanes_ring`, `lanes_params` and `lanes_proof` blocks.

The paper publishes the Hint-MLWE parameterization in closed form, with no
free constant, so nothing here is searched:

    eps = 2^-100,  s_0 = sqrt(ln(2 d~ (1 + 1/eps))) / pi
    s_1 = 2 s_0,   s_2 = 2 w_hat s_0,   s = 2 sqrt(2) w_hat s_0

`lanes_params.py` re-derives every printed figure to the last digit —
`beta' = 45430.6`, `B_MSIS = 15991562`, `q~/B_MSIS = 4.2`,
`delta_MSIS = 1.0037`, `D = 17` — and `test_lanes.py` pins each against the
printed digits.  Two identities the paper does not state fall out and are
pinned too: `s^2 = s_2^2 + w_hat^2 s_1^2`, so the published `s` is already
the worst-case-`l1` response width; and `sigma_MLWE = s_0` exactly, which
is *why* the widths are these.

**What gates the production name is security evidence, not parameters.**
An independent lattice-estimator run (commit `53da598`, recorded in
`lanes_security.json`) confirms M-SIS at 128.2 bits and `delta = 1.003732`
against the printed 1.0037.  The MLWE side does not reproduce:
`delta_MLWE = 1.0020` is not obtainable under either reading of the paper's
own Gaussian convention (116.2 bits reading `s_0` as a standard deviation,
which is what the paper says it is; 134.3 reading `sigma_0 = s_0 sqrt(2 pi)`
as one), and [KLSS23] Theorem 1 loses `(d+m) 2 eps ~ 2^-94.9` on its own.
The recovery-hint rules are still this implementation's.  See
`lanes_security.json` and

Two steps in the chain are deliberately worst-case, so 116.2 is a lower
bound on this instantiation rather than a claim the paper is short —
and neither is tightened here, because tightening a security bound to make
a parameter set pass is the failure the gate exists to prevent.

**What the cross-check found.**  The two implementations disagreed on
every LANES byte, and the cause was one unit: `bound_z`, the codec's cap on
`z_eval`, was `int(math.ceil(par.zm_inf_bound))` here — a *float* ceiling —
giving 16682743, against `river-rs`'s exact `floor_sqrt` giving 16682742.
`(6 sigma_m)^2 = 278313880780800` exactly, so 16682742 is the largest
coefficient the verifier's own test admits: the cap was one unit looser
than the accept/reject bound it was meant to mirror, and it fed the
statement hash.  Fixed here, to match; it is the repository's own rule
("no float reaches an accept/reject decision") applied at the one place
that had slipped it, and nothing short of a byte comparison would have
found it.

It also reports, in a footnote to §5, a LANES challenge-difference
noninvertibility probability of about `2^-93.5` for its re-optimized
parameters (`2^-70` for the original) and an outer figure of about
`2^-91.5`.  Those two are the only published quantities that separate the
LANES parameters from any other set, so a manifest has to
reproduce them; neither reaches 128 bits.

**Separate gates, not one.**  `exact.LANES_PARAMETER_MANIFEST` is the
frozen table, carrying data (not headings) in every section and *selecting
a value* for every constant in `exact.GATED_LANES_CONSTANTS` under a
Paper/Derived/Repair label — a label alone would let a wrong width be
relabelled **Paper** and travel on, so the value is compared against what
the code consumes.  `exact.LANES_SECURITY_EVIDENCE` carries a *verdict*:
having a recorded estimator run is not the same as passing, and this one
does not.  `exact.LANES_BACKEND_READY` records whether the implementation
has passed its KAT, serialization, negative-test and vector gates.  All are
required.  With one flag, obtaining a parameter table would have run the
backend on unvalidated security with a table of the right numbers sitting
beside it.

the manifest is `status: "final"` — every wire- and
security-visible value is Paper or a stated derivation from it — and the
live cause is `security-evidence-pending`, in both implementations.

The manifest check is deliberately *not* a comparison against the old
dimensions — those have already moved, so such a check could only ever
answer "gated", and the one way to make it answer "available" was to move
the exact parameters backwards.  `test_exact.py` drives every such move and
asserts the verdict does not change, and walks the constant list to require
the gate closed while any of them has drifted.

`exact.lanes_gate_cause()` reports which blocker applies as one of eight
short tokens — `audit-drift`, `constant-changed`, `no-parameter-manifest`,
`manifest-invalid`, `manifest-experimental`, `no-security-evidence`,
`security-evidence-pending`, `backend-not-ready` — so a generated artifact
records *why* a layer is missing in a form the Rust tree can compare
against its own gate.  It does compare, directly and by equality:
`river-rs/src/lanes_manifest.rs` is generated from `lanes_manifest.json`,
so both implementations are gated on the same table and a difference in
cause is a real divergence rather than one side lacking an input.  The
prose reason names each language's own API and is not compared.

Until then `LanesBackend(par)` **refuses to construct** under the
production name; `LanesBackend.experimental(par)` does not, and is what
`test_lanes.py`, the five `lanes_*` self-checks, the two
`lanes-experimental` vector cases and the three `lanes_*` KAT blocks all
run against.  None of them skips: a self-check that skips while the code
demonstrably works is one gate away from silence, which is how
`lanes_ring`'s twiddle tree stayed wrong for as long as it did.

What the gate still prevents is the thing it was built for.  Substituting
new dimensions into old widths and hint constants — or tightening a
security bound to make a parameter set pass — would produce something that
ran, verified against itself, and could be described as "the paper's LANES
instantiation" while being no such thing.

### The modulus condition, and why the centred range is load-bearing

LANES has a single modulus, so it checks the link only modulo `q~`.  That
pins an integer only when no accepted response can wrap.  Two accepted error
responses each satisfy `||z_m||_inf <= 6 sigma_m`, so their difference is at
most `12 sigma_m`, and a unique centred lift of `z_eval - x e_eval` needs

    q~ > 24 phi_m eta_m.

`z_eval` sits in the error block at `sigma_m = phi_m eta_m`, and
`eta_m = w gamma B_e sqrt(d)` does not depend on `ell` — it is the same
86889.3 for every profile — so this is one number for all five:
66730968.02, against the selected `q~ = 67107713`.

The margin is 376744.98, about **0.56%**, and it exists only because
`B_e = 30`.  The rounding relation is written on `[0, q_0-1]`; the parameter
table's norm bounds use the centred range; the algorithms never define the
translation.  Substituting the literal 60 doubles the requirement to
133461936.03 and the selected modulus fails outright.  So the shift is
carried explicitly (`ring.to_centered_error`), and because 0.56% is inside
what a float `sqrt` chain can move, `ExactParams.q_tilde_clears` decides the
condition over the integers as `q~^2 > (24 phi_m w gamma B_e)^2 d`.

## Sizes

`river-py` ships **no benchmark**.  It is the golden reference and the
vector generator; a timing taken from it would measure CPython rather than
the protocol, so benchmarking belongs to `river-rs` alone
(`make -C ../river-rs bench`, and `bench-sizes` for the per-profile table).

The sizes below are properties of the shared wire format: both
implementations produce byte-identical proofs, which `make check-vectors`
enforces. They are one deterministic measurement, not a restatement of the
paper's formula.

The paper's size model charges an *ideal* entropy cost of
`h(sigma) = log2(4.13 sigma)` bits per Gaussian coefficient without naming a
coder.  Golomb-Rice is this repository's concrete approximation to that, and
lands about half a bit per coefficient above it. Measured candidate-LANES
proofs against the current ideal model:

| profile | measured OOM | ideal OOM | measured exact | paper exact | framed proof |
|---|---:|---:|---:|---:|---:|
| `RiVeR-N8` | 20.245 KiB | 20.133 KiB | 13.890 KiB | 13.5 KiB | 34.143 KiB |
| `RiVeR-N16` | 21.548 KiB | 21.409 KiB | 13.883 KiB | 13.5 KiB | 35.438 KiB |
| `RiVeR-N64` | 25.729 KiB | 25.536 KiB | 13.887 KiB | 13.5 KiB | 39.623 KiB |
| `RiVeR-N128` | 29.366 KiB | 29.120 KiB | 13.893 KiB | 13.5 KiB | 43.267 KiB |
| `RiVeR-N256` | 36.487 KiB | 36.213 KiB | 13.888 KiB | 13.5 KiB | 50.383 KiB |

Those are single measurements, and Rice makes length data-dependent, so the
last digit moves between proofs — the comparison is good to about 0.1%, not to
the three decimals shown.  The residue is the gap between Rice and true
entropy, about half a bit per coefficient, so this is an independent check on
the paper's `|pi_OOM|` accounting rather than a restatement of its formula.

The exact-layer comparison uses the witness-hiding candidate LANES backend,
not the opening test backend that transmits its witness:

    make -C ../river-rs bench      # backend `lanes-experimental`

measures `|pi_ex|` at **13.88 KiB** against the stated 13.5, a 2.9% excess
that is the recovery hint (2048 bits) and the transmitted challenge — for
neither of which the paper's figure has any accounting.

**Proof length varies between proofs.**  Entropy coding makes it depend on
the sample; `Layout.max_bytes` is the worst case.

Length is *as* witness-independent as the accepted responses themselves, and
no more.  Rejection sampling targets witness-independent distributions for the
released `f_1`, `z_b`, `z_s` and `z_m`, which are the only variable-length
fields.  The current proof mirrors the later public `6 sigma` checks in its
simulator and treats them as deterministic post-processing, so their Gaussian
tail is not a separate distinguishing loss.  What remains open is the
proof's fixed `2^-100` rejection-sampling loss and how it composes with the
asymptotic and concrete query budgets.  The response split gives the two blocks their
own Rice parameters — `sigma_m / sigma_s` runs from 3.9 to 5.7 across the
published profiles, so a single parameter would have cost about a bit per
coefficient on whichever block it did not fit.  The honest statement is that
length leaks no more than the accepted proof bytes already do, not that it
leaks nothing.  Measured spread is 7 bytes in 8580, identical across signers;
padding to `max_bytes` would close the channel at the cost of the entire coding
gain.

Wall clock on one core of a modern x86-64 box, CPython 3.14:

| profile | eval / attempt | verify |
|---|---|---|
| `RiVeR-TOY` | 0.08 s | 0.03 s |
| `RiVeR-N8` | 0.67 s | 0.12 s |
| `RiVeR-N16` | 0.75 s | 0.15 s |
| `RiVeR-N64` | 1.13 s | 0.29 s |
| `RiVeR-N128` | 1.65 s | 0.50 s |
| `RiVeR-N256` | 2.40 s | 0.83 s |

Total `Eval` time is that times the attempt count, which averages about
`mu-tilde_RiVeR` — 8.3 to 8.6 for the published profiles, against 5.4 to 5.6
before the paper.  The increase is the fourth rejection sampler: the
response split turns one `Rej_1` call into two, and `mu_m` is a genuinely new
factor.  The paper's design target moved from `< 3` to `<= 10` to accommodate
it.  A full `RiVeR-N256` evaluation is therefore about 20 s here, which is
what `river-rs` exists to fix.

The one performance trick in the code is Kronecker substitution for
polynomial multiplication (`ring.py::mul`): pack both operands into single
big integers with a limb wide enough that no product coefficient can carry,
multiply once with Python's bignum, unpack and fold `X^d = -1`.  It is about
3x faster than schoolbook here and `test_ring.py` checks the two agree.

## The numeric manifest

Every value two implementations must agree on *exactly*, and that the paper
does not state, is collected in `manifest.py` and pinned by
`test_manifest.py`: the rational each Gaussian width is pinned to, the Rice
parameter per field, the largest coefficient that can pass each bound, and
the fixed field widths.  `make manifest` prints it; `python3 manifest.py
--json` emits it as an artifact a port can read.

The point is *which failure you get first*.  Without it, a changed sampler
width or a one-off Rice parameter surfaces as "proof bytes differ at byte 4"
in a cross-language vector, which names neither the field nor the cause.

It is meant to be readable standalone, as the handoff artifact `R1` starts
from, so it carries its own provenance (paper revision and SHA-256), both
wire layouts walked field by field in wire order, the exact layer's
dimensions including which rank plays which structural role, and the
framing overhead.  A port that reproduces the manifest has nothing left to
infer from prose.

Four things it made explicit that were previously implicit:

* `rational_sigma` does not represent sigma "exactly" — no rational does,
  since the widths are irrational.  It pins `round(sigma * 2^20) / 2^20`,
  and the `2^20` is part of the wire format.  The `round` removes only the
  *final* float error, so a port must compute the input in the same
  operation order.
* The Rice constant `sqrt(2 ln 2)` was a 4-digit rational, `11774/10000`.
  It is now 30 digits, and a test measures how far each field sits from the
  power-of-two boundary where `k` would move — at least 1% at every field of
  every profile, so the old constant was in fact safe here, and now that is
  checked rather than assumed.
* Every verifier bound has the shape `K sqrt(M)`, so the acceptance tests
  are decided by squaring — exact rationals, no `sqrt`, no float on the
  accept/reject path.  The codec's field caps are `floor(sqrt(bound_sq))`,
  which is exactly the largest coefficient that can pass, so the encoder and
  the verifier cannot disagree about the boundary.
* Which of `n~` and `l~` is the identity rank and which is the shared tail.
  They are both 4, so no
  dimension check can tell them apart — and the `lanes_*` modules were
  internally inconsistent about it. The manifest states the roles, and
  `test_exact.py` checks them against `ExactParams` even while the LANES
  backend is gated.

## Test vectors

`vectors.json` holds two pinned executions — the toy profile and `RiVeR-N8`,
each under the `opening` backend — with hex for every object plus intermediate
values (`j*`, the ring, the challenge, response norms, `W`).  Fixed seeds:
setup `00 01 .. 1f`, key `i` uses `bytes([i ^ 0x40]) + 00*31`, evaluation
`aa..aa`.  Schema 2, stamped with the paper revision and its SHA-256, so an
artifact produced before the migration cannot be mistaken for one produced
after it.

The two `lanes` cases are **withheld**, not dropped: `vectors.WITHHELD_CASES`
lists them, so the gap is visible in the artifact rather than only in a commit
message.  Pinning a retuned guess as a normative vector would give a second
implementation a target that is not the paper's.

```bash
python vectors.py --out vectors.json      # regenerate
python vectors.py --verify vectors.json   # re-derive and compare
```

`--verify` re-runs setup, key generation and evaluation from the seeds and
diffs every field, reporting the first differing byte of the proof.  A second
implementation reproduces this file byte-for-byte or it is not compatible.

## Where the paper is open

Every implementation-facing ambiguity is resolved behind a named helper, so
a clarification changes one boundary rather than the scheme, and each
resolution is labelled **Repair** where it is not derivable from the paper.

**Open questions, asked back to the paper.**

* The centred range shift is absent from the algorithms even though the
  selected `q~` depends on it.  The relation is stated over canonical
  errors in `[0, 60]` while `BoundGen` bounds them with `B_e = 30`; taking
  the paper literally almost exactly doubles every bound.  This
  implementation carries the translation explicitly in
  `ring.to_centered_error`.
* The challenge-invertibility assumption is stated unqualified while the
  concrete parameters give `1 - 2^-91.5`, below the 128-bit target.
  `test_review.py` reproduces the exact figures, `2155/131072` per paired
  coordinate and `p_nonunit ~ 2^-93.82`.
* The embedded-key assumption preamble uses a `beta = 1` bound while the
  concrete section samples `U_{B_e}`.
* `|pi_ex| = 13.5 KB` is stated with no field-by-field derivation, so it
  cannot be reproduced; a measured proof is 13.88 KB.
* Statistical accounting is asymptotic: `eps_1 <= 2^-100` per `Rej_1` call
  at `tau_rej = 12`, about 25 calls per returned proof, and neither a
  concrete statistical target nor a proof budget is stated.

**Two places the paper states something twice**, both pinned by tests
rather than normalised away:

* `(tau_g0, tau_g1)` is printed to two decimals, with a table note saying
  why: rounded to one decimal the same values reproduce only 8 of the 10
  `(B_g0, B_g1)` entries in the same table — `N = 256` fails both.  The
  printed pairs are used, labelled **Paper**, and a test pins that the
  coarser figures would not reproduce the table.
* `BoundGen` returns its bounds in one order and the three OOM algorithms
  parse another, with `phi_s` and `phi_b` swapped and nothing else.
  `phi_s` is 22 to 32 and `phi_b` is 2, so a positional implementation
  would sample and test both responses at each other's widths.  Neither
  implementation is positional — every value is a named field — which is
  why this is pinned by a test rather than left to a comment.

**Gated.**  The production `lanes` backend name; see "The exact layer".

Every constant in `params.py` carries one of three provenance labels —
**Paper**, **Derived**, **Repair** — and a Derived or Repair value is never
described as though the paper printed it.  `test_params.py` pins the
distinction.

## References

The exact layer follows the published protocols rather than any particular
implementation of them.

* **[ENS20]** Muhammed F. Esgin, Ngoc Khanh Nguyen, Gregor Seiler.
  *Practical Exact Proofs from Lattices: New Techniques to Exploit
  Fully-Splitting Rings.*  ASIACRYPT 2020, LNCS 12492, 259–288.
  The LANES protocol itself: the incomplete-NTT ring, the challenge space,
  and Figures 3 and 4, which `lanes_proof.py` implements.
* **[ESLR23]** Muhammed F. Esgin, Ron Steinfeld, Dongxi Liu, Sushmita Ruj.
  *Efficient Hybrid Exact/Relaxed Lattice Proofs and Applications to Rounding
  and VRFs.*  CRYPTO 2023, LNCS 14085, 484–517.
  LANES+, the hybrid exact/relaxed framework RiVeR builds on, the rounding
  relation behind the radix-`(1,3,9,17)` encoding, the LANES parameter set of
  Section 6.5, and the `D`-bit commitment compression the paper cites for
  `|pi_ex| = 7.898` KB.
* **[KLSS23]** Duhyeong Kim, Dongwon Lee, Jinyeong Seo, Yongsoo Song.
  *Toward Practical Lattice-Based Proof of Knowledge from Hint-MLWE.*
  CRYPTO 2023, LNCS 14085, 549–580.
  The Hint-MLWE instantiation, which is why the exact layer needs no internal
  rejection sampling.
* **[ALS20]** Thomas Attema, Vadim Lyubashevsky, Gregor Seiler.
  *Practical Product Proofs for Lattice Commitments.*  CRYPTO 2020,
  LNCS 12171, 470–499.  The cubic product proof in `lanes_mp.py`.
* **[BDLOP18]** Carsten Baum, Ivan Damgård, Vadim Lyubashevsky, Sabine
  Oechsner, Chris Peikert.  *More Efficient Commitments from Structured
  Lattice Assumptions.*  SCN 2018, LNCS 11035, 368–385.
  The commitment in `lanes_commit.py` and `exact.py`.
* **[HPRR19]** James Howe, Thomas Prest, Thomas Ricosset, Mélissa Rossi.
  *Isochronous Gaussian Sampling: From Inception to Implementation.*
  ePrint 2019/1411.  The Rényi-divergence sampler analysis `dgs.py` computes
  for comparison; see
