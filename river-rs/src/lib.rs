//! # river — Compact Ring Verifiable Random Functions from Lattices
//!
//! Rust implementation of **RiVeR**, test-vector compatible with the
//! Python reference in `../river-py`.
//!
//! A ring VRF lets one member of a public-key ring publish a
//! pseudorandom value together with a proof that *some* ring member
//! computed it correctly, without revealing which.  RiVeR builds this
//! from two linked lattice proofs: a relaxed one-out-of-many proof that
//! hides the evaluator's index, and a small exact proof certifying the
//! canonical range of the evaluation-side rounding error.
//!
//! ## Byte compatibility is the primary constraint
//!
//! Every seed-derived stream, every serialized object and every
//! Fiat-Shamir input has to match `river-py` exactly, because
//! `../river-py/vectors.json` is the interface between the two.  That
//! decides several things that would otherwise be free choices:
//!
//! * the Gaussian sampler is a uniform-proposal rejection sampler with
//!   an *exactly* computed acceptance threshold, not a CDT or FACCT — a
//!   different sampler is a different transcript.  [`fixed`] is that
//!   threshold's specification and [`fastexp`] is how it is actually
//!   evaluated: the same predicate in fixed width, with [`fixed`] as the
//!   fallback for the roughly one proposal in `2^55` it cannot settle.
//!   Its three tail parameters are independent by specification, and a
//!   declared cut `PROB_BITS` cannot reach is an error — see
//!   [`sample::check_probability_width`];
//! * `[[·]]_K` is taken on the **centred** representative, as the
//!   preliminaries define it, so about half the high parts are
//!   negative and the transmitted `B` field is signed;
//! * float-derived widths are pinned to exact rationals.  Production
//!   reads them, and the Rice parameters and integer bounds beside them,
//!   from [`manifest`] — a table generated from the reference's own
//!   frozen `manifest.json` — rather than re-deriving them from an `f64`
//!   chain whose evaluation order is the thing that has to match;
//! * every accept/reject bound is an exact rational ([`params::Rat`]),
//!   never a float: each has the shape `K sqrt(M)`, so squaring removes
//!   the `sqrt` and with it the last place two implementations could
//!   disagree about a coefficient on the boundary.
//!
//! ## Status
//!
//! Targeting the published paper.
//! the protocol and wire format are unchanged.
//!
//! | layer | module | state |
//! |---|---|---|
//! | exact thresholds | [`fixed`] | complete — the specification of the acceptance test |
//! | fixed-width acceptance | [`fastexp`] | complete — same predicate, 10 ns instead of 40 µs |
//! | frozen wire manifest | [`manifest`] | generated from `river-py/manifest.json`; `make manifest-check` |
//! | parameters, `BoundGen` | [`params`] | complete, reproduces the revision's table |
//! | ring arithmetic, rounding, centred bit dropping | [`ring`] | complete |
//! | CRT-NTT matrix backend | [`aux_ntt`] | complete |
//! | XOF and samplers | [`sample`] | complete |
//! | bit codec, transcript | [`codec`] | complete |
//! | relaxed one-out-of-many proof | [`oom`] | complete — split `z_s`/`z_m`, four rejection samplers, matched attempt-by-attempt against the reference |
//! | exact layer `Pi_ex` | [`exact`] | complete against `opening` and `lanes-experimental` |
//! | `Setup` / `KeyGen` / `Eval` / `Verify` | [`river`] | complete — byte-exact against `vectors.json` |
//! | interop against `vectors.json` | `tests/vectors.rs` | all four shipped cases; two production `lanes` cases withheld |
//! | LANES ring `R_q~` (R3) | [`lanes::ring`] | **complete** at `(d~, l, q~) = (256, 64, 67107713)`, cross-checked by an active KAT |
//! | LANES parameters (R3) | [`lanes::params`] | **complete** — the paper's closed form; every printed figure re-derived |
//! | LANES commitment (R3) | [`lanes::commit`] | **complete** — exercised at the current parameters by the shipped `lanes-experimental` cases |
//! | LANES proof and backend (R4) | [`lanes`] | **complete** — runs end to end, byte-exact against `river-py`; the production `lanes` *name* stays gated, see [`exact::lanes_unavailable_reason`] |
//!
//! *complete* means active and covered by a discriminating test.
//!
//! The LANES *ring* is complete because [`exact`] commits over it and its
//! block of `tests/sampler_kat.json` is generated and driven.  The
//! *parameters* became complete, which publishes the whole
//! Hint-MLWE chain in closed form: the searched integers `K_S1`/`K_S2`
//! are retired, and the widths, `beta'`, `B_MSIS`, `delta_MSIS` and `D`
//! all re-derive from `s_0 = sqrt(ln(2 d~ (1 + 1/eps)))/pi`.
//!
//! The implementation is now current too: [`lanes::proof`] and
//! [`lanes::backend`] run the proof end to end and are byte-exact against
//! `river-py` for both shipped `lanes-experimental` cases.  What is not
//! *established* is the security evidence, and that alone is what still
//! withholds the production name.  Four gates guard it, in this order: the
//! live constants still match the paper's closed form; a frozen
//! [`exact::LANES_PARAMETER_MANIFEST`] exists, validates and is final;
//! [`exact::LANES_SECURITY_MEETS_TARGET`], the recorded verdict — which is
//! `false`, and is the live blocker; and [`exact::LANES_BACKEND_READY`],
//! whether the implementation has passed its gates — which is now `true`.
//! A table is not evidence and evidence is not an implementation, so all
//! four are required, and dimensions do not enter any of the decisions.
//! [`river::BackendKind::Lanes`] refuses to construct and says why;
//! [`exact::lanes_gate_cause`] says which blocker, as a token shared with
//! `river-py`.  `lanes-experimental` is the same code under a different
//! instance name and is not gated.
//!
//! See `README.md` for what remains and in what order.

#![deny(unsafe_code)]
#![warn(rust_2018_idioms)]

pub mod aux_ntt;
pub mod codec;
pub mod exact;
pub mod fastexp;
pub mod fixed;
pub mod lanes;
pub mod lanes_manifest;
pub mod manifest;
pub mod oom;
pub mod params;
pub mod ring;
pub mod river;
pub mod sample;

pub use aux_ntt::CrtBackend;
pub use codec::{BitReader, BitWriter, CodecError, Coder, Field, FieldValue, Layout, RiVeRCodec};
pub use exact::{ExactParams, ExactStatement, ExactWitness, OpeningBackend};
pub use fastexp::ExpCtx;
pub use fixed::{Int, Nat};
pub use manifest::{ProfileManifest, MANIFEST};
pub use oom::{Oom, OomCommitment, OomProof, OomState, OomStatement};
pub use params::{
    Rat, RiVeRParams, DEFAULT_PARAMS, PROFILES, PUBLISHED, RIVER_N128, RIVER_N16, RIVER_N256,
    RIVER_N64, RIVER_N8, RIVER_TOY,
};
pub use ring::{Barrett, Poly, PolyMat, PolyVec, Ring};
pub use river::{EvalStats, Proof, PublicParams, RiVeR};
pub use sample::{GaussCtx, Part, Xof};
