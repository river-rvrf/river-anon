# river-py

A dependency-free **executable prototype** of RiVeR's outer protocol,
serialization and test-vector machinery, with a deterministic vector mode.

**Targets the published paper.**  The OOM response is split as
`r_0 = (s, e_key)` at `sigma_s` against `r_1 = e_eval` at `sigma_m`, with
four rejection samplers in the figure's order.

The outer protocol, codec, opening backend, vectors **and the LANES
parameters** are complete against it. `lanes_params.py` re-derives the
published Hint-MLWE quantities and `delta_MLWE = 1.0040`. The candidate
LANES backend is available as `lanes-experimental`; see "The exact layer".

It is not a production or security-preserving RiVeR implementation.  The
`opening` backend carries the protocol end to end but deliberately
reveals its opening; the zero-knowledge `lanes` layer runs under
`exact_backend="lanes-experimental"`, with the production `"lanes"` name
reserved pending a reduction for the concrete compression/recovery
composition — see "The exact layer". What it is good
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
path, and is what `vectors.py` exports. The default is fresh coins so a seed
argument cannot silently become the nonce for every evaluation.

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

# A ring is an ordered tuple of exactly `N` valid keys. Duplicates are
# permitted; the evaluator uses the first matching position. There is no
# padding or canonical reordering: the order is part of the statement.
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
before the challenge, serialization, and the complete verification boundary.

## The exact layer

`Pi_ex` sits behind a backend interface, with three selectable names.

| backend | link | witness | state |
|---|---|---|---|
| `"opening"` (default) | over `Z` | **transmitted** | runs; a mock, about 9.3 KB |
| `"lanes-experimental"` | mod `q~` | not transmitted | runs, at the paper's parameters; about 13.9 KB |
| `"lanes"` | mod `q~` | not transmitted | reserved production alias; see below |

**`opening`** is a mock.  It enforces every clause of the relation, including
the integer link, but `sigma_ex` *is* the opening: `e_eval` leaks, and with it
`<G(m), s>`, so the secret key falls out after about `ell` distinct messages.
It carries the protocol end to end, which is what it is for.

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
printed digits. Two derived identities are pinned too:
`s^2 = s_2^2 + w_hat^2 s_1^2`, so the published `s` is already
the worst-case-`l1` response width; and `sigma_MLWE = s_0` exactly, which
is *why* the widths are these.

The published parameters are recorded in `lanes_manifest.json` and checked
against the values consumed by the code. The same manifest is projected into
Rust, and the KAT and vector suites cover commitment generation, proof fields,
serialization, verification, and negative cases in both languages.

The paper treats LANES as an exact-proof backend and gives the dimensions,
Gaussian widths, response model, compression parameter, and entropy estimate.
This executable candidate additionally fixes a concrete wire codec and a
compression/recovery-hint construction. Those choices are labelled
**Derived** or **Repair** in the manifest. The artifact does not supply a
security reduction for that exact composition, so the working backend is
named `lanes-experimental`; the production alias `lanes` remains reserved.
`lanes_security.json` is a reproducible estimator diagnostic, not a normative
security verdict for the paper.

### The exact-modulus margin

LANES has a single modulus, so it checks the link only modulo `q~`.  That
pins an integer only when no accepted response can wrap.  Two accepted error
responses each satisfy `||z_m||_inf <= 6 sigma_m`, so their difference is at
most `12 sigma_m`, and a unique centred lift of `z_eval - x e_eval` needs

    q~ > 24 phi_m eta_m.

`z_eval` sits in the error block at `sigma_m = phi_m eta_m`, and
`eta_m = w gamma B_e sqrt(d)` does not depend on `ell` — it is the same
86889.3 for every profile — so this is one number for all five:
66730968.02, against the selected `q~ = 67107713`.

The margin is 376744.98, about **0.56%**. The construction translates
canonical rounding errors to centred representatives before applying the
bound; `ring.to_centered_error` implements that translation. Because the
margin is small, `ExactParams.q_tilde_clears` decides the condition over the
integers as `q~^2 > (24 phi_m w gamma B_e)^2 d` rather than through a floating
point square-root chain.

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

| profile | measured OOM | ideal OOM | measured exact | exact entropy estimate | framed proof |
|---|---:|---:|---:|---:|---:|
| `RiVeR-N8` | 20.245 KiB | 20.133 KiB | 13.890 KiB | 13.5 KiB | 34.143 KiB |
| `RiVeR-N16` | 21.548 KiB | 21.409 KiB | 13.883 KiB | 13.5 KiB | 35.438 KiB |
| `RiVeR-N64` | 25.729 KiB | 25.536 KiB | 13.887 KiB | 13.5 KiB | 39.623 KiB |
| `RiVeR-N128` | 29.224 KiB | 28.952 KiB | 13.886 KiB | 13.5 KiB | 43.117 KiB |
| `RiVeR-N256` | 36.315 KiB | 36.041 KiB | 13.888 KiB | 13.5 KiB | 50.211 KiB |

Those are single measurements, and Rice makes length data-dependent, so the
last digit moves between proofs — the comparison is good to about 0.1%, not to
the three decimals shown.  The residue is the gap between Rice and true
entropy, about half a bit per coefficient, so this is an independent check on
the paper's `|pi_OOM|` accounting rather than a restatement of its formula.

The exact-layer comparison uses the witness-hiding candidate LANES backend,
not the opening test backend that transmits its witness:

    make -C ../river-rs bench      # backend `lanes-experimental`

measures `|pi_ex|` at **13.88 KiB** against the paper's 13.5 KiB
entropy-based estimate. The concrete encoder also carries the recovery hint
and transmitted challenge; the side-by-side values distinguish encoded bytes
from the entropy estimate.

**Proof length varies between proofs.**  Entropy coding makes it depend on
the sample; `Layout.max_bytes` is the worst case.

The response blocks have separate Rice parameters because
`sigma_m / sigma_s` runs from 3.9 to 5.7 across the published profiles.
Encoded length is data-dependent and this artifact does not claim a
length-hiding wire format; applications that require fixed-length messages can
pad to `Layout.max_bytes`.

The one performance trick in the code is Kronecker substitution for
polynomial multiplication (`ring.py::mul`): pack both operands into single
big integers with a limb wide enough that no product coefficient can carry,
multiply once with Python's bignum, unpack and fold `X^d = -1`.  It is about
3x faster than schoolbook here and `test_ring.py` checks the two agree.

## The numeric manifest

Every implementation-level value the two implementations must agree on
*exactly* is collected in `manifest.py` and pinned by
`test_manifest.py`: the rational each Gaussian width is pinned to, the Rice
parameter per field, the largest coefficient that can pass each bound, and
the fixed field widths.  `make manifest` prints it; `python3 manifest.py
--json` emits it as an artifact a port can read.

The point is *which failure you get first*.  Without it, a changed sampler
width or a one-off Rice parameter surfaces as "proof bytes differ at byte 4"
in a cross-language vector, which names neither the field nor the cause.

It is readable standalone: both wire layouts are listed field by field in
wire order, along with the exact-layer dimensions, structural rank roles, and
framing overhead. A port that reproduces the manifest has nothing left to
infer about the wire format from prose.

The manifest makes the following choices explicit:

* `rational_sigma` does not represent sigma "exactly" — no rational does,
  since the widths are irrational.  It pins `round(sigma * 2^20) / 2^20`,
  and the `2^20` is part of the wire format.  The `round` removes only the
  *final* float error, so a port must compute the input in the same
  operation order.
* The Rice calculation uses a 30-digit value of `sqrt(2 ln 2)`. A test
  measures how far each field sits from the power-of-two boundary where `k`
  would move — at least 1% at every field of every profile.
* Every verifier bound has the shape `K sqrt(M)`, so the acceptance tests
  are decided by squaring — exact rationals, no `sqrt`, no float on the
  accept/reject path.  The codec's field caps are `floor(sqrt(bound_sq))`,
  which is exactly the largest coefficient that can pass, so the encoder and
  the verifier cannot disagree about the boundary.
* Which of `n~` and `l~` is the identity rank and which is the shared tail.
  They are both 4, so a numeric dimension check alone cannot distinguish the
  roles; `test_exact.py` checks them against `ExactParams` explicitly.

## Test vectors

`vectors.json` holds four pinned executions — the toy profile and `RiVeR-N8`,
each under both `opening` and `lanes-experimental` — with hex for every object
plus intermediate values (`j*`, the ring, the challenge, response norms,
`W`). Fixed seeds:
setup `00 01 .. 1f`, key `i` uses `bytes([i ^ 0x40]) + 00*31`, evaluation
`aa..aa`. The top-level metadata identifies `river-py` as the generator; the
profile and backend are recorded in each case.

The two production-alias `lanes` cases are **withheld**, not dropped:
`vectors.WITHHELD_CASES` lists them. They use the same code as
`lanes-experimental` but remain withheld while the production alias is
reserved.

```bash
python vectors.py --out vectors.json      # regenerate
python vectors.py --verify vectors.json   # re-derive and compare
```

`--verify` re-runs setup, key generation and evaluation from the seeds and
diffs every field, reporting the first differing byte of the proof.  A second
implementation reproduces this file byte-for-byte or it is not compatible.

## Specification and artifact scope

The paper fixes the algorithms, profiles, exact-layer dimensions, LANES
widths, response bounds, and entropy-based size model. The executable wire
format additionally needs exact rational approximations to irrational widths,
sampler tail cuts, coefficient coders, framing, transcript byte order, and a
concrete LANES compression/recovery procedure. These choices are collected in
the manifests and labelled **Paper**, **Derived**, or **Repair** according to
their source.

The tests establish internal arithmetic consistency, cross-language byte
interoperability, rejection boundaries, and honest-path/negative-case
behaviour. They do not constitute a proof of the paper's security theorems or
a reduction for the artifact's concrete LANES recovery composition. The
production alias remains reserved for that reason; the fully tested candidate
is explicitly named `lanes-experimental`.

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
