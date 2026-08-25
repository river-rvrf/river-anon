# RiVeR — reference implementation and port

Two implementations of RiVeR, a compact ring verifiable random function from
lattices, provided so that the paper's concrete claims can be checked rather
than taken on trust.

* **`river-py`** — a dependency-free Python reference.  It exists to be read
  beside the paper: every algorithm follows the figures step by step, every
  constant carries its provenance, and it owns the shipped test vectors and
  the frozen parameter manifests.  It is not optimised and makes no timing
  claim.
* **`river-rs`** — a Rust **port** of the same protocol, byte-for-byte
  identical on the wire.  It is what the performance figures are measured
  from.
* **`parameters`** — the self-contained parameter-setting artifact. It
  reproduces the selected rows, derived bounds, repetition accounting, and
  the bundled estimator checks without referring to private documentation.

These are not independent reimplementations: the Rust port was written
against the Python reference and consumes manifests and known-answer tests
generated from it.  What the cross-check establishes is that two separate
codebases in two languages agree on **every byte of all four shipped vector
executions**, and on every primitive the known-answer tests cover.  That
catches porting errors and under-specified conventions; it is not two
independent readings of the paper.

Nothing here is claimed to be constant-time-hardened production code, albeit the Rust codebase is closer to that aim.

## Requirements

| | |
|---|---|
| `river-py` | Python 3.9+, no packages, no network |
| `river-rs` | Rust 1.87+ (`cargo`); see below |
| `parameters` | SageMath and Python 3; bundled estimators, no network |

`river-py` needs nothing but a Python interpreter: no packages, no network,
no external test data. Everything it checks is in this tree.

**`river-rs` needs crates.io for its first build**, unless the evaluating
machine already has a populated Cargo cache. The dependency set is small and
deliberately so:

* **one** runtime dependency, `sha3`, for SHAKE-256 — the only randomness
  source in the protocol. That is 7 crates linked into the library, plus
  `version_check`, which `generic-array` uses at build time only and which
  does not end up in the binary.
* **three** dev-dependencies used only by the test harness to read the
  shipped JSON: `serde`, `serde_json`, `hex`.

`Cargo.lock` is committed, so the resolved versions are fixed; 22 external
crates are pinned in it. To build with no network at all, vendor once on a
connected machine and point Cargo at the result:

```bash
cd river-rs
cargo vendor ../vendor > vendor-config.toml   # needs network, once
mkdir -p .cargo && cp vendor-config.toml .cargo/config.toml
cargo build --release --offline               # no network from here on
```

## Quick start

```bash
make test            # both implementations (Python ~8 min, Rust ~1 min)
make kat             # primitive known-answer tests only, ~1 s
make check-vectors   # re-derive the shipped vectors in both languages and diff
make bench           # all measurements below
make -C parameters table-check  # deterministic parameter/table checks
```

For the complete parameter-artifact run, including the bundled estimators
and expanded finite-grid diagnostic, use `make -C parameters check`. The
published rows pass the explicit finite grid; [parameters/README.md](parameters/README.md)
documents its scope and the acceptance tests. Generated tables under `parameters/data/`
and `parameters/report/` are shipped outputs and are preserved by `make clean`.
The five-million-trial product-threshold replay is intentionally separate:
`make -C parameters product-check` regenerates it without writing and compares
the resulting aggregate counts with the shipped CSV.

## Reproducing the paper's claims

The paper's parameter and size claims are checked by something you can run.
The table says which command; this is not an exhaustive audit of every
statement in the paper.

| claim in the paper | how to check it | what happens |
|---|---|---|
| the parameter table | `make -C parameters table-check` | re-derives the selected rows, bounds, repetition accounting, and generated tables |
| public-key and proof sizes | `make bench-sizes` | measured against the paper's size model, both shown side by side |
| exact-proof entropy estimate `\|pi_ex\| = 13.5 KB` | `make bench-sizes` | the concrete candidate LANES encoding measures **13.88–13.89 KB** at every profile |
| the two implementations agree | `make check-vectors` | all four shipped cases re-derived from seeds in both languages and diffed |
| the primitives agree | `make kat` | XOF, samplers, thresholds and codec, field by field |

### Ideal versus measured sizes

The paper's size model charges an **ideal** entropy cost of
`h(sigma) = log2(4.13 sigma)` bits per Gaussian coefficient and does not name
a coder. This implementation uses Golomb–Rice, which lands about half a bit
per coefficient above that ideal. Both are reported, always side by side, and
never conflated:

* **ideal** columns are the paper's own formula, evaluated at the published
  parameters.
* **measured** columns are the bytes this encoder actually emits.

`make bench-sizes` prints both. The final manuscript table uses the same
response split as the implementation, so its displayed OOM column is the
ideal-model column directly.

Key sizes are measured the same way. A ring is `N` public keys verbatim: no
padding, no compression, so ring size is exactly `N × pk`.

### What is measured (single machine, single thread, release build)

Public keys and one deterministic proof per profile:

| profile | `N` | pk B | sk B | OOM KiB | exact KiB | wire KiB |
|---|---:|---:|---:|---:|---:|---:|
| RiVeR-N8 | 8 | 8448 | 1728 | 20.245 | 13.890 | 34.143 |
| RiVeR-N16 | 16 | 7872 | 1888 | 21.548 | 13.883 | 35.438 |
| RiVeR-N64 | 64 | 8448 | 1728 | 25.729 | 13.887 | 39.623 |
| RiVeR-N128 | 128 | 8640 | 1728 | 29.224 | 13.886 | 43.117 |
| RiVeR-N256 | 256 | 8064 | 1888 | 36.315 | 13.888 | 50.211 |

Exact layer under `lanes-experimental`, which does not transmit the witness;
`wire`
includes two 4-byte length prefixes. Proof length is data-dependent — entropy coding moves the
last few bytes between proofs.

`Eval` retries until every rejection sampler accepts, so its total cost is a
geometric variable; the per-attempt figure is the stable one. The paper's
model puts the mean attempt count at 8.3–8.6.

Backend `lanes-experimental` (the candidate that does not transmit the
witness) throughout. `Eval` is
aggregated over 5 independent seeds — total time over total attempts, since
attempts abort at different points and do not cost the same:

| profile | Eval per attempt | attempts sampled | Verify | decode |
|---|---:|---:|---:|---:|
| RiVeR-N8 | 25.6 ms | 26 | 4.77 ms | 138 µs |
| RiVeR-N16 | 30.6 ms | 11 | 5.84 ms | 141 µs |
| RiVeR-N64 | 40.4 ms | 35 | 16.0 ms | 161 µs |
| RiVeR-N128 | 65.7 ms | 59 | 21.4 ms | 182 µs |
| RiVeR-N256 | 118.6 ms | 32 | 37.3 ms | 241 µs |

**Total time per proof is deliberately not tabulated.** It is the
per-attempt cost times a geometric attempt count, and 5 seeds is far too few
to pin that count — the samples above span 11 to 59 attempts. The benchmark
prints a mean proof time, but treat it as one observation, not an estimate.
A converged figure needs a proper sweep.

Measured on an AMD Ryzen AI 9 HX 370, `x86_64-unknown-linux-gnu`,
`rustc 1.97.1`. The benchmark prints CPU, target triple and compiler on
every run, so a different machine's figures stay attributable. Timings will
not reproduce to the digit elsewhere; the size columns above *do* reproduce
exactly.

## Re-running the benchmarks

All measurements come from one binary, and it needs no arguments:

```bash
make bench          # everything: primitives, scheme, keys, sizes
make bench-sizes    # keys and communication only, all five profiles
make bench-lanes    # the exact layer's ring, backend and codec
```

Or directly, which is the same thing:

```bash
cd river-rs
cargo run --release --bin river-bench             # full run, ~42 s
cargo run --release --bin river-bench -- --sizes  # sizes only, ~15 s
cargo run --release --bin river-bench -- --lanes  # exact layer only
```

Notes on reading the output:

* **Build in release.** A debug build is 30–100× slower and its numbers mean
  nothing.
* **Timings are single-threaded and unpinned.** `Verify` and `proof_decode`
  are medians of repeated batches. **`Eval` is aggregated over 5 seeds** as
  total time over total attempts — never one run divided by its own attempt
  count, which weights a lucky single-attempt run as heavily as a
  twenty-attempt one. Even aggregated it is a small sample. For absolute
  figures, pin a core and fix the CPU governor.
* **The run prints its own environment** — CPU, target triple and `rustc`
  version — as its first four lines.
* **Sizes are exact.** They are byte counts of real encoded proofs, not
  estimates, and they are reproducible to the byte for a fixed seed. The
  benchmark uses fixed seeds throughout.
* **Attempt counts vary widely.** They are geometric; a single run tells you
  little. Compare the per-attempt column instead.

## The two exact-proof backends

The one-out-of-many layer is always the same. The exact layer `Pi_ex` sits
behind an interface with two selectable implementations:

| backend | witness | size | what it is |
|---|---|---|---|
| `opening` | **transmitted** | ~9.3 KB | a deliberate mock: it reveals the witness, so it is *not* zero-knowledge and its size is not comparable to the paper's 13.5 KB. Useful for exercising everything above it. |
| `lanes-experimental` | **not transmitted** | ~13.9 KB | the candidate LANES exact layer at the paper's parameters: it does not transmit the witness, and its encoded size is shown beside the 13.5 KB entropy estimate. |

`opening` is labelled a mock everywhere it appears, including in the
benchmark output, because its 9.3 KB would otherwise read as a smaller proof
rather than as the cost of a leak.

`lanes-experimental` is a **candidate** concrete instantiation. The paper
treats LANES as a black-box exact layer; its response compression, recovery
rules, codec and transcript encoding are fixed by this artifact and labelled
**Derived** or **Repair** in the manifest. Byte interoperability and algebraic
correctness are tested; the artifact does not supply a reduction for that
exact composition.

A third name, `lanes`, is the same code under a reserved production alias and
refuses to construct. The candidate remains available as
`lanes-experimental`; `river-py/lanes_security.json` records a reproducible
diagnostic, not an authoritative security verdict for the paper.

## Layout

```
river-py/    Python reference; owns vectors.json and the frozen manifests
river-rs/    Rust implementation and the benchmark binary
parameters/  Self-contained parameter-setting and estimator artifact
Makefile     dispatches to implementations and recursively cleans components
```

Each component has its own README with the detail: `river-py/README.md` for
the algorithms and implementation choices, `river-rs/README.md` for the
performance work and what byte compatibility costs, and
`parameters/README.md` for parameter reproduction and its documented limits.

## Tests

```bash
make test            # both implementations
make test-all        # plus the slow all-profile sweep
make kat             # primitives only, ~1 s
make check-vectors   # cross-language byte equality
```

`river-py` carries 370 tests, `river-rs` 290 (260 library, 26 primitive KAT,
and 4 vector tests). They include the negative
cases: every malformed public input must be rejected rather than crash —
truncation, non-canonical residues, runaway unary runs, nonzero padding,
trailing bytes, hostile length prefixes.
