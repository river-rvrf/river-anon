# river-rs

Rust implementation of **RiVeR** — compact ring verifiable random functions
from lattices.  Test-vector compatible with the Python reference in
[`../river-py`](../river-py/), and the implementation the paper's performance
numbers are meant to come from.

**Status: byte-exact against the reference implementation.**
`Setup` / `KeyGen` / `Eval` / `Verify` re-derive every one of `river-py`'s
shipped test vectors byte for byte — the same proof, the same attempt
count, the same value — under both the witness-revealing `opening` backend
and `lanes-experimental`.

The OOM response is split as `r_0 = (s, e_key)` at `sigma_s` against
`r_1 = e_eval` at `sigma_m`, and the concrete moduli are the paper's
printed ones — including `q_hat`, which follows a *different* rule from
`p`: largest prime below `2^bits` for `p`, smallest above `2^{bits-1}` for
`q_hat`, both `5 mod 8`.

The candidate `lanes` backend has the following status:

* **complete** — the ring `R_q~` at `(d~, l, q~) = (256, 64, 67107713)`.
  The exact layer commits over it, and its block of `sampler_kat.json` is
  generated and driven, so it is cross-checked against `river-py` rather
  than against itself;
* **complete** — `lanes::params`. The paper publishes the Hint-MLWE chain in
  closed form, so no parameter search is needed: the widths, `beta'`,
  `B_MSIS`, `delta_MSIS`, `delta_MLWE`, and `D` all
  re-derive from `s_0 = sqrt(ln(2 d~ (1 + 1/eps)))/pi` at `eps = 2^-100`.
  Tests pin each against the paper's printed digits;
* **complete** — `lanes::{commit, mp, proof, backend}`.  The proof layer
  runs end to end at the current parameters and is **byte-exact against
  `river-py`**: `vectors.json` ships two `lanes-experimental` cases and
  this crate re-derives both from their seeds, with `sampler_kat.json`'s
  `lanes_ring`, `lanes_params` and `lanes_proof` blocks bisecting them
  primitive by primitive;
* **reserved** — the production `lanes` *name*. `BackendKind::LanesExperimental`
  is the tested name for the same code while the concrete recovery/compression
  composition remains an artifact-level completion without a supplied
  reduction.

Gates guard the production alias. `exact::LANES_PARAMETER_MANIFEST` is the
frozen table, carrying data for every required field and selecting
a *value* for every constant on the audit list, compared against what
`lanes::params` actually consumes.  It is **generated**:
`src/lanes_manifest.rs` comes from `../river-py/lanes_manifest.json` via
`scripts/gen_lanes_manifest.py`, and `make manifest-check` requires an
empty diff — so the two implementations are gated on the same table rather
than each on its own opinion of one, and `sampler_kat.rs` asserts they
report the same cause.

A typed projection of a table that also carries prose is lossy by
construction, so the generated file additionally carries `source_sha256`,
the digest of the canonical source JSON.  Without it, editing a `how`
string with no typed consequence would leave the check green while the two
trees documented the same field differently; with it, any edit to the
source table has to be regenerated.  The projection itself is complete —
coder parameters and not just coder names, `Option<u64>` bits so a
variable-length field is distinguishable from a zero-width one, real row
counts, the dimensions, the three transcript rounds with their challenge
separators, and the four recovery counts.

`exact::LANES_SECURITY_MEETS_TARGET` is the production-alias policy switch;
it remains `false` because this artifact does not supply a reduction for its
exact compression/recovery composition. It is not a claim that the paper's
published parameter derivation fails. `exact::LANES_BACKEND_READY` records
that the implementation has passed its KAT, serialization, negative-test and
vector gates.

A table is not evidence and evidence is not an implementation, so none of
the three lifts another.  `BackendKind::Lanes` refuses to construct until
all are set, and the reference withholds the two matching production vector
cases; `BackendKind::LanesExperimental` is the ungated name and ships two
of its own.

`exact::lanes_gate_cause()` names the blocker as a short token shared with
`river-py`, which is what `tests/sampler_kat.json` records for the blocks it
withholds; the prose reason names this crate's API and is not comparable
across the two languages.

Not constant time end to end, but the secret-bearing arithmetic and the
signer index are; see *Timing* for exactly which properties are enforced,
which are structural, and which are unverified.  Not intended for production
use.

```bash
make test        # unit + integration + cross-language KAT
make kat         # the cross-language KAT alone, ~0.1 s
make selftest    # unit tests only
make bench       # micro-benchmarks, self-contained
make bench-lanes # focused LANES ring/backend/codec benchmarks
make bench-sizes # one measured LANES proof at every published profile
```

`make check-vectors` checks **all four** shipped cases,
`{RiVeR-TOY, RiVeR-N8}` against each of `opening` and `lanes-experimental`.
The two production-alias `lanes` cases are *withheld* by the reference, not
dropped, and this crate's backend is reserved to match. The accounting is
enforced by `every_case_is_checked`, which fails both if a case names a backend
this crate does not have *and* if the LANES gate ever lifts without the
withheld cases coming back, so coverage cannot shrink in silence in either
direction.

The `lanes` layer runs at the paper's parameters — `D = 17`,
`q~ = 67107713` over a degree-256 ring, the published widths — and produces
byte-identical proofs to `river-py`. A measured exact proof at the paper's
parameters is **13.88 KB** beside its 13.5 KB entropy estimate; `make bench`
reports the whole `lanes-experimental` proof in its `Scheme` section
and the OOM half in its size table, and the exact half is the difference.
`river-py` ships no benchmark: it is the reference and the vector
generator. The recovery-hint construction and concrete codec are
implementation-level completions of the black-box exact layer.

The modulus condition is now decided over the integers
(`ExactParams::q_tilde_clears`) rather than in floating point, because the
margin is 0.56% — inside what a float `sqrt` and a multiplication chain can
move.

## Goals

- **Byte-for-byte interop with the Python reference.**  Every seed-derived
  stream, every serialized object, every Fiat–Shamir input matches
  `river-py` exactly.  That is the primary constraint, and it decides
  things that would otherwise be free choices — see *What byte
  compatibility costs* below.
- **Performance-oriented.**  The CRT-NTT design of
  four 32-bit auxiliary primes, so products stay in `u64`; pre-transformed matrices for
  `G'` and `A`, which are fixed by `rho` forever and are 76–88% of
  per-attempt work.
- **Fail-closed on unsupported parameter sets.**  `RiVeRParams::check` is
  `BoundGen`'s abort, including the compression margin and modulus
  primality — a composite `q_hat` that happens to be `5 mod 8` is refused,
  not accepted.
- **`Rej_1`'s repetition constant is an argument, not a literal.**  The
  paper parameterises it as `tau_rej` and fixes `tau_rej = 12`.  Naming it
  only in the parameter report would have been the wrong half.  `params::REJ_TAU` feeds `mu_a`/`mu_s`/`mu_m` *and* is
  `sample::rej1`'s fifth argument, required rather than defaulted, so a
  test over the reporting formula can no longer pass while the sampler
  uses a different number.  `sampler_kat.json` records `tau_rej` per
  `rej1` case and reads it from the artifact rather than from `REJ_TAU`.
- **No division on secret data.**  Every modular reduction on a value
  derived from a key or a mask goes through Barrett or the auxiliary
  primes' pseudo-Mersenne identity, not `%` — in `R_q`, in the CRT
  backend, and in the LANES ring `R_q~`, which multiplies commitment
  randomness and proof masks.  (`%` by a constant is usually
  strength-reduced by the compiler, but "usually" is not a guarantee, and
  the guarantee is the point.)  The rejection samplers remain
  variable-time — that is structural, and the reference is the same.  See
  *Timing*.
- **Robust verification.**  Every malformed public input is a `false`, not a
  panic, and that now means the whole path rather than the codec alone.  No
  decoder panics, and every malformation — truncation, a non-canonical
  residue, a runaway unary run, nonzero padding, trailing bytes, a hostile
  length prefix — is a `CodecError`; above it, `RiVeR::verify` is total on
  the ring, the message, the value and the proof, including values handed in
  as structs rather than decoded from bytes.

## Layout

| file | contents |
|---|---|
| `src/fixed.rs` | arbitrary-precision naturals and the *exact* fixed-point exponential the acceptance thresholds need |
| `src/params.rs` | the five published profiles plus a toy one, `BoundGen`, concrete moduli, every derived bound |
| `src/ring.rs` | `R_q = Z_q[X]/(X^d+1)`: schoolbook multiplication, norms, rounding `floor(·)_p`, Power2Round |
| `src/aux_ntt.rs` | the CRT-NTT matrix backend over four auxiliary primes |
| `src/fastexp.rs` | the acceptance test in fixed width — same predicate as `fixed`, no allocation, with `fixed` as the fallback |
| `src/sample.rs` | SHAKE-256 counter-mode XOF, uniform / ternary / Gaussian samplers, hash-to-`C^d_{w,gamma}`, `Rej_1` and `Rej_2` |
| `src/bin/bench.rs` | the micro-benchmarks behind the numbers below |
| `src/codec.rs` | the bit-level wire format: `Uniform` / `Signed` / Golomb–Rice coders, layout-driven encode and decode, the Fiat–Shamir transcript digests |
| `src/oom.rs` | the relaxed one-out-of-many proof: `Com`, `Prove`, `Ver`, and the structural statement they run against |
| `src/exact.rs` | the exact layer `Pi_ex`: radix-`(1,3,9,17)` range encoding, witness packing, the BDLOP commitment over `R_q~`, and the opening backend |
| `src/river.rs` | the scheme: `Setup`, `KeyGen`, ring admissibility, `Eval`, `Verify`, and the framing |
| `src/lanes/` | the LANES exact backend, complete: `ring` (the incomplete NTT, with shape and domain in the type), `params`, the `[BDLOP18]` `commit`, the `[ALS20]` product proof `mp`, the `gamma`-compressed linear `proof`, and the `Pi_ex` `backend` |
| `tests/vectors.rs` | byte-for-byte interop against `../river-py/vectors.json` |
| `tests/sampler_kat.rs` | the cross-language KAT loader |
| `tests/sampler_kat.json` | the KAT itself, generated from `river-py` |
| `scripts/gen_kat.py` | that generator |

## What byte compatibility costs

`vectors.json` pins whole executions, so several natural implementation
choices are unavailable.  Each of these is a deliberate deviation from what
a from-scratch Rust implementation would do:

- **The Gaussian sampler is a uniform-proposal rejection sampler.**  Both
  sibling implementations use a CDT (`../../lotrs-dev`) or FACCT.  Neither
  reproduces the reference's accept/reject decisions, and those decisions
  are the transcript: FACCT draws extra XOF bits for its `Ber(2^-k)` test
  and finishes with an explicitly approximate polynomial, and a CDT
  consumes one uniform per *sample* rather than one per *proposal* and
  would need a `2.5·10^8`-entry table at `σ ≈ 1.8·10^7`.  Changing the
  sampler is a specification change, not an optimisation — see

  What *is* an optimisation is how the exact acceptance test is
  evaluated, and that was the whole cost: 40 µs per proposal through
  arbitrary-precision integers, because `mag << 128 / den` is a
  bit-at-a-time long division with two heap allocations per bit.
  `src/fastexp.rs` decides the same predicate in `u64` mantissas with
  `u128` intermediates — **10 ns**, no allocation — by bracketing the
  threshold instead of computing it, and deferring to `src/fixed.rs` on
  the roughly one proposal in `2^55` whose bracket is ambiguous.  Same
  distribution, same XOF consumption, same bytes; the cross-language KAT
  is unchanged and still passes.
- **The acceptance threshold is computed exactly.**  `river-py` reaches
  `floor(scale · exp(num/den))` through `decimal` at a pinned precision;
  `std` has no equivalent and an `f64::exp` comparison would fork a vector
  on the last ulp.  `src/fixed.rs` computes the mathematically exact floor
  in fixed point instead, escalating precision until the bracket pins one
  integer.  Agreement with the reference is *measured*, not assumed: the
  KAT carries 519 thresholds spanning the reachable range, including the
  exact exponents each published profile produces — all six of them.  It
  covered three until a review pointed out that N16, N64 and N128 had
  their widths pinned by the parameter table and their accept/reject
  decisions pinned by nothing.
- **`Ring::mul` is schoolbook.**  At `d = 32` an isolated CRT-NTT multiply
  costs six transforms against 1024 multiply-accumulates and loses; the
  transform earns its keep only across a matrix product with a fixed
  matrix.  This follows the design note rather than reflexively
  transforming everything.
- **`[[·]]_K` follows the canonical-representative convention** the OOM
  layer uses.  Aligning
  moves protocol bytes, so it waits for the reference to move first —
- **The Rice parameter is computed in integers, and a proof is padded to a
  byte boundary exactly once.**  `k = floor(log2(sqrt(2 ln 2)·sigma))` is
  wire-visible: evaluated in `f64` it can differ by one between two
  implementations at a half-ulp, and one is a different encoding, not a
  rounding difference.  It is evaluated over the exact rational `sigma`
  instead — the same rational the sampler uses.  Padding is likewise a
  format decision and not an obvious one: the sibling `lotrs-rs` aligns
  after every polynomial, which is friendlier to a streaming decoder and
  costs a few bytes per field.  `river-py` aligns once for the whole
  layout, so this does too.

## The arithmetic backend

Four auxiliary primes `2^32 - c`, each `≡ 1 mod 64`, with
`P = p0·p1·p2·p3 > 2^127`.  A product in `Z[X]/(X^d+1)` is an integer
computation, so it is carried over the auxiliary primes, reconstructed
exactly by Garner, and reduced mod `q` at the end — RiVeR's own moduli are
all `5 mod 8`, which splits `X^32 + 1` into just two factors and admits no
useful transform of its own.

The reconstruction bound `P > 2·m·d·A²` is checked at construction against
the **unsigned** `A = q-1`, not the centred `A = (q-1)/2`, even though the
backend centres its inputs.  The design note records a prototype that sized
against the centred bound while feeding unsigned inputs and passed every
random test, because the worst case needs every coefficient at the extreme
simultaneously.  `tests` cover that case explicitly:
`mul_agrees_on_saturated_inputs` and `accumulation_agrees_at_the_extreme`
are the ones an undersized `P` fails.

Two bugs this arrangement caught while it was being written, both of the
kind random tests miss:

- `P` is just under `2^128`, so it does not fit `i128`.  Casting the Garner
  reconstruction to a signed type wraps for every value above `2^127` —
  half of them.  The centring stays in `u128` and carries the sign
  separately.
- The acceptance bracket at `f = 128` fractional bits is `2^64` wide
  against a `2^192` scale, so it can never pin the threshold *integer*.
  It is still the right first rung for the *comparison*, because a uniform
  `u` falls inside it with probability about `2^-109`.  The two ladders are
  separate for that reason.

## Status and what is next

| layer | state |
|---|---|
| exact thresholds (`fixed`) | complete, cross-checked against 519 reference values |
| parameters, `BoundGen` (`params`) | complete, reproduces the paper's table and the reference's exact widths |
| ring arithmetic, rounding, bit dropping (`ring`) | complete |
| CRT-NTT matrix backend (`aux_ntt`) | complete, saturated-input tested |
| XOF and samplers (`sample`, `fastexp`) | complete, byte-exact; acceptance test 4000x faster than the bignum path it defers to |
| bit codec, transcript (`codec`) | complete, byte-exact — including a whole `pi_OOM` encoding and both Fiat–Shamir digests |
| OOM layer (`oom`) | complete, byte-exact — one whole `Com`/`Prove`/`Ver` trajectory matched against the reference |
| exact layer `Pi_ex` (`exact`) | both backends complete, byte-exact |
| LANES exact backend (`lanes`) | complete, byte-exact — `ring`, `params`, `commit`, the product proof, the linear proof and the `Pi_ex` backend |
| `Setup` / `KeyGen` / `Eval` / `Verify` (`river`) | complete, byte-exact |
| interop against `vectors.json` | all four shipped cases, `opening` and `lanes-experimental`; 2 production `lanes` cases withheld, the accounting enforced by a test |

In that order.  The OOM layer is the first one that *produces* the values
the codec serializes, so it is the first that could be wrong in a way the
codec tests could not see.  It is pinned as a **trajectory** rather than as
one successful proof: each attempt draws from three rejection samplers and
can abort at one of six places, so an extra XOF draw, a reordered bound
check or an early return from the wrong test shows up as a different
sequence of aborts long before it shows up as different proof bytes.  The
KAT fixes eight consecutive attempts at the toy profile — six aborts and
two accepted proofs, interleaved — and both implementations produce the
same eight.

The exact layer sits on top of it and is complete at the paper
parameters for both backends -- `opening` and, under the experimental
name, LANES.  Only the production `lanes` name is gated.

Two parameter requirements are enforced here: a declared sampler tail cut beyond what
`PROB_BITS` supports is an error rather than a silently unchanged
distribution (`sample::check_probability_width`, called from
`GaussCtx::new`), and both response bounds are *derived* from the response
variance rather than kept as literals — `lanes::params::Z_INF_BOUND` is
`ceil(t sqrt(Var))` in `const fn` integer arithmetic, and the Euclidean one
is `ceil(4 (sigma_y^2 + w_hat^2 sigma_r^2) N_Z)` — the paper's
`beta' = 2 s sqrt(N_z)` rule, evaluated on the rounded widths so both
implementations reach the same integer rather than agreeing to within a
rounding.  Deriving the first of those is what found it was 13.99 sd under
a comment claiming 14.

The codec is where the KAT stops being a transcription of integers and
starts pinning bytes, so it is checked at three levels: the coders alone
(Rice parameters, field widths, the bit writer's output, a pinned Rice
blob), each profile's derived layout — every width, bound and Rice
parameter, plus a four-value probe per field so a drift names the field
that drifted — and one whole `pi_OOM` encoding at the toy profile.  All of
it agreed with `river-py` on the first run; adding `1` to the Rice
parameter fails four of the six codec cases and none of the others, which
is the property the layering is for.

The exact-layer block exists now, so the full-proof framing is tested
against the real thing rather than a stand-in layout.  `RiVeRCodec::proof_encode` takes the
exact block already encoded, because its layout belongs to whichever exact
backend is in use and this module does not need to know which.

## Measured

`make bench`. Single-threaded, `--release`, one machine — indicative, and
the run prints its own CPU, target triple and compiler so the numbers stay
attributable. `make bench-lanes` is the short reproducible run for the
NTT over `R_q~`, the two reduction strategies, and the exact
backend itself, which runs under the `lanes-experimental` name.
`make bench-sizes` generates one deterministic proof per published profile
under both backends and prints actual framed bytes beside the paper's model.
Read the `opening` exact half as the cost of revealing the witness; the
`lanes-experimental` half is the concrete encoding shown beside the paper's
13.5 KiB entropy estimate.

The headline end-to-end measurements below use `lanes-experimental` on an
AMD Ryzen AI 9 HX 370. `Eval` is aggregated over five independent seeds as
total time divided by total attempts; `Verify` and decode are medians of
repeated batches.

| profile | Eval/attempt | attempts sampled | Verify | decode |
|---|---:|---:|---:|---:|
| `RiVeR-N8` | 25.6 ms | 26 | 4.77 ms | 138 µs |
| `RiVeR-N16` | 30.6 ms | 11 | 5.84 ms | 141 µs |
| `RiVeR-N64` | 40.4 ms | 35 | 16.0 ms | 161 µs |
| `RiVeR-N128` | 65.7 ms | 59 | 21.4 ms | 182 µs |
| `RiVeR-N256` | 118.6 ms | 32 | 37.3 ms | 241 µs |

These are performance observations, not interoperability evidence.
Byte-for-byte agreement is established separately by `make check-vectors`,
which pins the attempt count and complete proof encoding.

The profile trend is dominated by drawing Gaussian polynomials, so the
`Eval` column is substantially the sampler's number rather than the proof
system's, and the sampler is XOF-bound.

Two things shape that picture.

**The sampler is now XOF-bound, and by specification rather than by
implementation.**  Each proposal consumes a `PROB_BITS = 192` uniform
plus the proposal itself — about 30 bytes — and there are ~11 proposals
per accepted coefficient.  That is ~330 bytes of SHAKE-256 per
coefficient, and SHAKE-256 costs what it costs (~10 cycles/byte without
SIMD or SHA-3 instructions).  The exponential is now 0.5% of the sampler.
Going faster means consuming less randomness, which means changing
`PROB_BITS` or the sampler — a specification decision, not a code one.

**The CRT-NTT backend does win, but only pre-transformed.**  486 ns
against 1394 ns per ring product, 2.9×, exactly the case the design note
describes: `G'` and `A` are fixed by `rho` forever.  Transforming
per call loses badly — the one-off `mat_to_ntt` for an 8×64 matrix is
~395 µs — so `Ring::mat_vec`, which transforms on every call, is *slower*
than schoolbook and exists only as a correctness reference.  Callers that
care must hold the transform.

**What this predicts, and what it does not.**  At `RiVeR-N256` a proof
draws 13 184 Gaussian coefficients per attempt, so the sampler alone puts a
floor of tens of milliseconds on every attempt.  The measured per-attempt
`Eval` sits above that floor for the reason it predicts: most of an attempt
is `OM.Com`, and most of `OM.Com` is the sampler.

Every published profile *is* benchmarked end to end, including
`RiVeR-N256`.  What is **not** claimed is a converged mean proof time: that
is the per-attempt cost times a geometric attempt count, and the benchmark
samples too few seeds to pin the count.

## Timing

Not constant time, and the parts that are not are worth naming rather
than disclaiming in general.

**Whose property this is.**  `river-py` is the specification and the KAT
oracle, and nothing more: Python's integers are arbitrary-precision with
data-dependent limb counts, its comparisons and its remaining
zero-coefficient skips branch freely, and no arrangement of that code
could carry a constant-time claim.  So the two implementations agree on
*bytes*, not on timing, and this one enforces the property below
independently.  A divergence between them here is expected and is not a
port bug; a divergence in bytes is.

*Fixed, with the exceptions named:* **no value derived from a secret
reaches a division, and no secret chooses a branch or an address.**  The
qualification matters — the earlier form of this sentence had no
exceptions and was therefore too strong.  What is actually enforced:

| removed | what it carried |
|---|---|
| `%` / `rem_euclid` finishing every ring multiply | every product in `R_q`, `R_qhat`, `R_q~` — now `ring::Barrett`, the auxiliary primes' pseudo-Mersenne identity, or `lanes::ring`'s |
| `c / q_0` per coefficient in `round_p` | `A s` and `<h_m, s>` — now one reciprocal formed from the *public* `q_0` and a widening multiply each |
| `rem_euclid(q)` in `rounding_error` | the secret key's rounding error — now a masked add, since both terms are below `q` |
| `rem_euclid(2^K)` and the tie `if` in `mod_pm` | coefficients of `u_A`, `u_B`, which are products of the masks |
| `if c > half_q` in `Ring::centered` | the secret key, every mask, every response — the busiest secret branch in the crate |
| `if c == 0` in `Ring::neg` | whether a secret coefficient is zero, on data a third zeros when ternary |
| `if i == j_star`, `b[j_star] = 1`, `shift[j_star - 1] = x` | **`j*` itself** — the signer's index, which is the secret the whole OOM proof exists to hide |
| `ring[j_star]` in `Eval` | the same, through the cache line a secret-indexed read touches |
| `rem_euclid` in `uniform_beta_poly` / `gaussian_poly` | the secret key and every Gaussian mask |
| `% P` in `aux_ntt::crt_combine` | the auxiliary residues of a secret polynomial |

Each is an arithmetic-shift mask and a wrapping add — the same
instruction sequence either way.  LLVM usually picks a conditional move
for the `if` forms; "usually" is not the claim this section makes, and
the masked form does not depend on it.

*Divisions that remain, and why they are not exceptions:*

* `Barrett::new`, `Reciprocal::new`, `half_q` — setup, on a public
  modulus, once per object.
* `Ring::scale` and `Ring::const_poly` reduce their *scalar* once per
  call.  Every call site in the scheme passes the public `q_0`.
* `Ring::from_centered` and `rounding_error` keep a `rem_euclid`
  fallback, and `round_p` keeps a `/` one.  Their branch outcomes are
  invariant over every input the domain admits — the fast path is taken
  for all of them — so the tests are not secret-dependent; they exist to
  keep the functions total on a hand-built argument.

*Fixed:* `q~ = 2^26 - 1151` is pseudo-Mersenne, so a canonical product
reduces with two base-`2^26` folds and one masked subtraction — fixed
work, and **1.14 ns against Barrett's 1.83 ns**, a factor of 1.61
(`river-bench --lanes`).  Its wide entry point folds a fixed five times
rather than looping until the value fits, so an accumulator whose
magnitude depends on a secret does not change the instruction count.

Those figures are throughput, not latency: the harness folds 1024
independent reductions and a superscalar core overlaps them, which is
what a *stream* of reductions costs — and a stream is what the transforms
issue.  Inputs and outputs both cross `std::hint::black_box`; without the
input barrier the fold is a pure function of a fixed array and may be
hoisted out of the timing loop, which would report one evaluation divided
by the iteration count.

*Not fixed, and structural:* the rejection samplers.  `gaussian_int`
loops until it accepts, `Rej_1` / `Rej_2` short-circuit, and the number
of iterations is the secret.  The reference is the same, and making it
otherwise is a specification change.  `fastexp` inherits this: its binary
decompositions branch on bits of the exponent.

The same goes for the whole-proof retry loop: `Eval` restarts until an
attempt is accepted, so its wall clock is a function of the mask stream.
Evaluating all four rejection predicates and all four norm checks on every
attempt — accumulating a flag rather than short-circuiting — would make
each *attempt* more uniform without making either count constant, since
every attempt draws from an independently derived XOF and a rejected
attempt's state is discarded whole.  Truly fixed wall clock would mean
running all `max_attempts = 1000` attempts and selecting the first
success, which is not a trade this implementation makes.

*Fixed, and it was a real leak:* two zero-coefficient skips are gone.
`mul_schoolbook` used to skip zero coefficients of its first operand,
documented as seeing only public challenge polynomials, and
`OomStatement::combine_c` skipped all-zero `coeff_i`.  Both claims were
wrong.  `combine_c` passes the mask `a_i` as that first operand and
`Oom::com` squares every `a_i`, so the skip saw secrets; the leak is
per-coefficient rather than the measure-zero event that a whole polynomial
is zero; and because `f_i = a_i + x b_i` is published, an observer who
learns where `a_i` has zeros can test each `i` against `f_i` and `f_i - x`,
so "`a_i` is independent of `j*`" does not settle it.

It also bought nothing where it was supposed to: `w == d == 32` at every
shipped profile, so a challenge polynomial has **no** zero coefficients and
the branch never fired on the public data it was justified by.  It fired on
ternary secrets, where a third of the coefficients are zero — which is why
removing it costs 5–8% on the commitment paths and nothing elsewhere.

*Not verified:* that the emitted machine code keeps these properties.  The
masks are written so a compiler has nothing to branch on, and that is an
argument about the source, not a measurement of the binary.  Checking it
on x86-64 and AArch64 — by disassembly or a dudect-style harness — is
open, and neither this crate nor the reference does it today.

**`Cargo.lock` is committed** — this is destined to be a reproducibility
artifact, and the "libraries do not commit their lockfile" convention
applies to published crates, not to a reproducibility artifact rebuilt
byte-for-byte.

## Dependencies

One: `sha3`, for SHAKE-256.  Everything else — the big integers, the
fixed-point exponential, the NTT, the samplers, the codec — is in-crate,
matching the dependency-free posture of `river-py`.  `rayon` and
`zeroize` were declared ahead of the layers that will use them and have
been removed until then; an unused dependency in a reproducibility
artifact is audit surface with nothing behind it.

`Cargo.lock` pins the dependency graph but not the compiler, so
`rust-version = "1.87"` records what the source needs.  Declaring it is
also what gets it checked: the value started at 1.77, picked for
`round_ties_even` — the binding one on paper, since it pins the Gaussian
width and therefore the transcript — and clippy immediately pointed at
`u64::is_multiple_of`, stable only since 1.87.  A stated MSRV nobody
verifies is worth about as much as no MSRV.  The committed results were
produced with `rustc 1.97.1 (8bab26f4f 2026-07-14)`.
