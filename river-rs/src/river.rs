//! The scheme — `Setup`, `KeyGen`, `Eval`, `Verify`.  Port of
//! `river-py/river.py`.
//!
//! A ring VRF lets one member of a public-key ring publish a pseudorandom
//! value with a proof that *some* ring member computed it correctly.  This
//! module is the composition: it derives the statement, drives the attempt
//! loop, and binds the two proof layers to one execution.
//!
//! ## The two-stage ordering is the point
//!
//! Each attempt runs [`Oom::com`] and then the exact layer's `Com`, and
//! only then derives `rho'` — the Fiat–Shamir nonce — from `(R~, v, W)`.
//! So the exact commitment `W` is fixed *before* the OOM challenge exists,
//! which is what ties the two components to one execution rather than
//! letting a prover shop for a `W` after seeing `x`.  Reordering those
//! three lines is not a refactor.
//!
//! ## The seed is always the caller's
//!
//! There is no fresh-coin default here: the crate has no RNG dependency,
//! so [`RiVeR::eval_deterministic`] takes a seed and its name says the
//! caller meant it.  A production caller supplies fresh randomness per
//! evaluation; the reference's `eval` defaults to `os.urandom` and keeps
//! `eval_deterministic` as the pinned path, which is the same distinction
//! drawn one layer up.
//!
//! The nonce actually used is `H(seed || sk || ring || v || m)`.  That
//! derivation is load-bearing, not defensive: two evaluations that share a
//! mask `y` but get different challenges publish `z_1 = y + x_1 r` and
//! `z_2 = y + x_2 r`, so `z_1 - z_2 = (x_1 - x_2) r` and the whole witness
//! falls out by one linear solve.  Binding the message and the key into
//! the nonce means a caller who reuses a seed across messages still gets
//! independent masks — the only form of the guarantee an API can enforce.
//!
//! ## `Verify` returns a bit
//!
//! Total on `ring_pks`, `m`, `v` and `pi` for *any* value: those four come
//! from a peer.  Every stage returns `false` rather than propagating, and
//! the typed proof is re-encoded before use so that a value arriving as a
//! struct is held to exactly the ranges the byte decoder would enforce.

use crate::codec::{
    proof_frame, proof_unframe, ring_digest, statement_digest, Layout, Result as CodecResult,
    RiVeRCodec,
};
use crate::exact::{ExactCommitment, ExactStatement, ExactWitness, OpeningBackend, OpeningProof};
use crate::lanes::backend::{LanesBackend, LanesState, LanesStatement};
use crate::lanes::commit::Commitment as LanesCommitment;
use crate::lanes::proof::LanesProof;
use crate::oom::{Oom, OomProof, OomStatement};
use crate::params::RiVeRParams;
use crate::ring::{round_p, rounding_error, to_centered_error, Poly, PolyMat, PolyVec, Ring};
use crate::sample::{
    hash_bytes, sam_mat, uniform_beta_vec, uniform_poly, Part, Xof, DS_COMMIT, DS_EXACT, DS_G,
    DS_KEYGEN,
};

// ---- the exact backend, as a choice --------------------------------------
//
// `Pi_ex` is a black box to everything above it, and the reference makes it
// a constructor argument: `RiVeR(par, exact_backend="lanes")`.  Two
// instantiations ship, they produce different proofs, and `vectors.json`
// carries cases for both — so the choice has to survive into the type
// rather than being fixed at compile time.
//
// An enum rather than a trait object: there are exactly two, both are in
// this crate, and every method below dispatches on a value the caller
// already has.  What that costs is the pair of "wrong variant" arms in
// `prove` and `verify`, which are reachable — a caller can build a `Proof`
// by hand — and are `None`/`false` rather than a panic.

/// Which `Pi_ex` instantiation a scheme runs with.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BackendKind {
    /// [`OpeningBackend`] — complete and binding, **not** zero knowledge.
    Opening,
    /// [`LanesBackend`] — the ported `[ENS20]` prover under the reserved
    /// production alias; see [`crate::exact::lanes_unavailable_reason`].
    Lanes,
    /// The same prover under a different name, ungated.
    ///
    /// It runs at the paper's parameters with the artifact's concrete
    /// compression/recovery composition. The separate name is load-bearing:
    /// a vector case or benchmark row recording
    /// `"lanes-experimental"` reconstructs *this*, and recording `"lanes"`
    /// for both would make an experimental artifact reconstruct a backend
    /// that refuses to exist.
    LanesExperimental,
}

impl BackendKind {
    /// The name `vectors.json` records in `exact_backend`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Opening => OpeningBackend::NAME,
            Self::Lanes => LanesBackend::NAME,
            Self::LanesExperimental => LanesBackend::EXPERIMENTAL_NAME,
        }
    }

    /// Inverse of [`BackendKind::name`]; `None` on anything else.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            OpeningBackend::NAME => Some(Self::Opening),
            LanesBackend::NAME => Some(Self::Lanes),
            LanesBackend::EXPERIMENTAL_NAME => Some(Self::LanesExperimental),
            _ => None,
        }
    }
}

/// The instantiated backend, held by [`PublicParams`].
pub enum ExactBackend {
    Opening(Box<OpeningBackend>),
    Lanes(Box<LanesBackend>),
}

/// `W` — whichever commitment the backend in use produces.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExactCom {
    Opening(ExactCommitment),
    Lanes(LanesCommitment),
}

/// `sigma_ex` — whichever proof the backend in use produces.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExactSigma {
    Opening(OpeningProof),
    Lanes(Box<LanesProof>),
}

/// Prover state carried from `Com` to `Prove`.
enum ExactState {
    Opening(crate::exact::OpeningState),
    Lanes(Box<LanesState>),
}

impl ExactCom {
    /// The opening backend's commitment, or `None` under LANES.
    pub fn opening(&self) -> Option<&ExactCommitment> {
        match self {
            Self::Opening(w) => Some(w),
            Self::Lanes(_) => None,
        }
    }

    /// Mutable form, for the tests that move one coefficient.
    pub fn opening_mut(&mut self) -> Option<&mut ExactCommitment> {
        match self {
            Self::Opening(w) => Some(w),
            Self::Lanes(_) => None,
        }
    }

    pub fn lanes(&self) -> Option<&LanesCommitment> {
        match self {
            Self::Lanes(w) => Some(w),
            Self::Opening(_) => None,
        }
    }

    pub fn lanes_mut(&mut self) -> Option<&mut LanesCommitment> {
        match self {
            Self::Lanes(w) => Some(w),
            Self::Opening(_) => None,
        }
    }
}

impl ExactSigma {
    pub fn opening(&self) -> Option<&OpeningProof> {
        match self {
            Self::Opening(s) => Some(s),
            Self::Lanes(_) => None,
        }
    }

    pub fn opening_mut(&mut self) -> Option<&mut OpeningProof> {
        match self {
            Self::Opening(s) => Some(s),
            Self::Lanes(_) => None,
        }
    }

    pub fn lanes(&self) -> Option<&LanesProof> {
        match self {
            Self::Lanes(s) => Some(s),
            Self::Opening(_) => None,
        }
    }

    pub fn lanes_mut(&mut self) -> Option<&mut LanesProof> {
        match self {
            Self::Lanes(s) => Some(s),
            Self::Opening(_) => None,
        }
    }
}

impl ExactBackend {
    /// `Err` names why the backend cannot be built at this profile.
    ///
    /// [`BackendKind::Lanes`] is a reserved production alias
    /// — see [`crate::exact::lanes_unavailable_reason`] — so this is where
    /// selecting it fails, rather than three layers down in a proof that
    /// verifies against itself.
    fn new(kind: BackendKind, par: RiVeRParams, ck_seed: &[u8]) -> Result<Self, String> {
        Ok(match kind {
            BackendKind::Opening => Self::Opening(Box::new(OpeningBackend::new(par, ck_seed)?)),
            BackendKind::Lanes => Self::Lanes(Box::new(LanesBackend::new(par, ck_seed)?)),
            BackendKind::LanesExperimental => {
                Self::Lanes(Box::new(LanesBackend::experimental(par, ck_seed)?))
            }
        })
    }

    pub fn kind(&self) -> BackendKind {
        match self {
            Self::Opening(_) => BackendKind::Opening,
            Self::Lanes(b) => {
                if b.name() == LanesBackend::EXPERIMENTAL_NAME {
                    BackendKind::LanesExperimental
                } else {
                    BackendKind::Lanes
                }
            }
        }
    }

    /// The opening backend, or `None` under LANES — for the tests and
    /// benchmarks that want at its `bound_y`.
    pub fn opening(&self) -> Option<&OpeningBackend> {
        match self {
            Self::Opening(b) => Some(b),
            Self::Lanes(_) => None,
        }
    }

    pub fn lanes(&self) -> Option<&LanesBackend> {
        match self {
            Self::Lanes(b) => Some(b),
            Self::Opening(_) => None,
        }
    }

    fn com(&self, witness: &ExactWitness, xof: &mut Xof) -> Option<(ExactCom, ExactState)> {
        match self {
            Self::Opening(b) => b
                .com(witness, xof)
                .map(|(w, st)| (ExactCom::Opening(w), ExactState::Opening(st))),
            Self::Lanes(b) => b
                .com(witness, xof)
                .map(|(w, st)| (ExactCom::Lanes(w), ExactState::Lanes(Box::new(st)))),
        }
    }

    /// `Pi_ex.Prove`.  `xof` is the one [`ExactBackend::com`] drew from —
    /// LANES continues that stream for `g` and `y`, so the two calls share
    /// it, exactly as the reference's state does.
    fn prove(
        &self,
        witness: &ExactWitness,
        w: &ExactCom,
        z_eval: &[i64],
        x: &[i64],
        state: &ExactState,
        xof: &mut Xof,
    ) -> Option<ExactSigma> {
        match (self, w, state) {
            (Self::Opening(b), ExactCom::Opening(_), ExactState::Opening(st)) => {
                Some(ExactSigma::Opening(b.prove(witness, st)))
            }
            (Self::Lanes(b), ExactCom::Lanes(w), ExactState::Lanes(st)) => {
                let statement = LanesStatement { w, z_eval, x };
                b.prove(&statement, st, xof)
                    .map(|s| ExactSigma::Lanes(Box::new(s)))
            }
            _ => None,
        }
    }

    /// `Pi_ex.Ver`.  A `W` or `sigma_ex` from the other backend is `false`,
    /// not a panic: a `Proof` can be built by hand.
    fn verify(&self, w: &ExactCom, z_eval: &[i64], x: &[i64], sigma: &ExactSigma) -> bool {
        match (self, w, sigma) {
            (Self::Opening(b), ExactCom::Opening(w), ExactSigma::Opening(s)) => {
                b.verify(&ExactStatement { w, z_eval, x }, s)
            }
            (Self::Lanes(b), ExactCom::Lanes(w), ExactSigma::Lanes(s)) => {
                b.verify(&LanesStatement { w, z_eval, x }, s)
            }
            _ => false,
        }
    }

    pub fn w_encode(&self, w: &ExactCom) -> CodecResult<Vec<u8>> {
        use crate::codec::CodecError;
        match (self, w) {
            (Self::Opening(b), ExactCom::Opening(w)) => b.w_encode(w),
            (Self::Lanes(b), ExactCom::Lanes(w)) => b.w_encode(w),
            _ => Err(CodecError::LengthMismatch),
        }
    }

    pub fn w_bytes(&self) -> usize {
        match self {
            Self::Opening(b) => b.w_bytes(),
            Self::Lanes(b) => b.w_bytes(),
        }
    }

    pub fn proof_encode(&self, w: &ExactCom, sigma: &ExactSigma) -> CodecResult<Vec<u8>> {
        use crate::codec::CodecError;
        match (self, w, sigma) {
            (Self::Opening(b), ExactCom::Opening(w), ExactSigma::Opening(s)) => {
                b.proof_encode(w, s)
            }
            (Self::Lanes(b), ExactCom::Lanes(w), ExactSigma::Lanes(s)) => b.proof_encode(w, s),
            _ => Err(CodecError::LengthMismatch),
        }
    }

    pub fn proof_decode(&self, data: &[u8]) -> CodecResult<(ExactCom, ExactSigma)> {
        match self {
            Self::Opening(b) => b
                .proof_decode(data)
                .map(|(w, s)| (ExactCom::Opening(w), ExactSigma::Opening(s))),
            Self::Lanes(b) => b
                .proof_decode(data)
                .map(|(w, s)| (ExactCom::Lanes(w), ExactSigma::Lanes(Box::new(s)))),
        }
    }

    pub fn proof_layout(&self) -> &Layout {
        match self {
            Self::Opening(b) => b.proof_layout(),
            Self::Lanes(b) => b.proof_layout(),
        }
    }

    pub fn proof_bytes(&self) -> usize {
        match self {
            Self::Opening(b) => b.proof_bytes(),
            Self::Lanes(b) => b.proof_bytes(),
        }
    }
}

/// `pp` — `Setup`'s output, held by every party.
///
/// **Opaque.**  The CRS is assumed honestly generated and this does not
/// pretend to validate an adversarial one.  What opacity buys is
/// narrower and worth having: a `pp` that came out of [`RiVeR::setup`]
/// cannot subsequently be edited into one that is internally
/// *inconsistent*.  Every field here is derived from `rho` and the
/// profile, and several are cached — `G'`, three Barrett reductions, the
/// commitment key — so a public `oom.par.K_b` was a way to make
/// verification shift out of range without touching `rho` at all.
///
/// Read access is unrestricted; there is nothing secret in a CRS.
pub struct PublicParams {
    seed: Vec<u8>,
    rho: Vec<u8>,
    a_mat: PolyMat,
    oom: Oom,
    exact: ExactBackend,
}

impl PublicParams {
    /// The setup seed this was derived from.
    pub fn seed(&self) -> &[u8] {
        &self.seed
    }

    /// `rho`, the public randomness every matrix is expanded from.
    pub fn rho(&self) -> &[u8] {
        &self.rho
    }

    /// `A ← SamMat(rho, q, n, ell, "RiVeR.A")`.
    pub fn a_mat(&self) -> &PolyMat {
        &self.a_mat
    }

    pub fn oom(&self) -> &Oom {
        &self.oom
    }

    pub fn exact(&self) -> &ExactBackend {
        &self.exact
    }
}

/// `pi = (pi_OOM, pi_ex)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proof {
    pub oom: OomProof,
    pub w: ExactCom,
    pub sigma_ex: ExactSigma,
}

/// What an evaluation did, for the tests that compare the measured restart
/// rate against `mu-tilde_RiVeR`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EvalStats {
    pub attempts: usize,
    /// Why each failed attempt aborted, in order.
    ///
    /// `mu-tilde_RiVeR = mu_OOM mu_ex` charges both layers, so an
    /// evaluation that restarts has to say which one asked for it —
    /// otherwise the measured attempt rate cannot be compared against the
    /// model.  See [`AbortReason`].
    pub aborts: Vec<AbortReason>,
}

/// Which layer returned `bot` and discarded an attempt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AbortReason {
    /// `OM.Prove` rejected: a rejection sampler, a norm bound, the
    /// compression margin, or the `A' = A` wrap check.
    Oom,
    /// `Pi_ex.Prove` returned `bot`.  `W` is already bound into the
    /// challenge, so the OOM proof cannot be reused with a fresh exact
    /// commitment — the whole attempt is discarded.  The shipped
    /// `opening` backend never takes this path.
    Exact,
}

/// Why an evaluation could not produce a proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EvalError {
    /// The ring is malformed or is not exactly `N` keys.
    Ring(&'static str),
    /// `pk` is not in the ring.
    NotAMember,
    /// The caller's own key material is the wrong shape or out of range.
    Key(&'static str),
    /// The rounding error left `[0, q_0-1]`, which cannot happen for a
    /// witness this module derived — it is a guard on the arithmetic, not
    /// on the caller.
    Witness,
    /// An invariant the scheme rests on did not hold.
    ///
    /// Not a caller error and not a restartable event: reaching one means
    /// the parameters or the statement construction are wrong.  It is an
    /// error rather than an assertion because `--release` strips
    /// `debug_assert!`, and these are the two equations that make a proof
    /// mean anything.
    Invariant(&'static str),
    /// No accepting attempt in `max_attempts`.
    Exhausted(usize),
    /// `pp` was produced by a different parameter profile or a different
    /// exact backend than this instance was configured for.
    BackendMismatch,
}

/// The scheme, bound to one parameter profile.
pub struct RiVeR {
    /// Private for the same reason [`Oom`]'s is: `codec` and `rq` are
    /// derived from it at construction.
    par: RiVeRParams,
    codec: RiVeRCodec,
    rq: Ring,
    backend: BackendKind,
}

impl RiVeR {
    /// `None` on a profile either `BoundGen` or the exact backend would
    /// abort.
    ///
    /// The check has to come **first**, and has to cover **both** layers.
    /// It used to run inside `setup`, which is one constructor too late:
    /// `RiVeRCodec::new` shifts by `K_b` and `Ring::new` builds a Barrett
    /// reduction for `q_0 · p`, so a zero modulus or an out-of-range `K_b`
    /// reached derived state before anything looked at the profile.  And
    /// it used to check only the outer parameters, so a profile whose
    /// `q_0` broke the exact layer's radix cover, or whose `phi_rs B_rs`
    /// outgrew `q~`, panicked inside `setup` instead.
    pub fn try_new(par: RiVeRParams) -> Option<Self> {
        Self::try_new_with(par, BackendKind::Opening)
    }

    /// [`RiVeR::try_new`] against a chosen exact backend.
    ///
    /// The reference's `RiVeR(par, exact_backend=...)`.  The default stays
    /// [`BackendKind::Opening`] on both sides — it is the one that checks
    /// every clause of `R^_ex` including the integer link, and the one the
    /// crate's own tests are written against.
    pub fn try_new_with(par: RiVeRParams, backend: BackendKind) -> Option<Self> {
        Self::build(par, backend).ok()
    }

    /// [`RiVeR::try_new_with`] with the reason kept.
    ///
    /// `Err` is either `BoundGen`'s abort, the exact layer's, or — for the
    /// reserved [`BackendKind::Lanes`] alias — the gate in
    /// [`crate::exact::lanes_unavailable_reason`].
    pub fn build(par: RiVeRParams, backend: BackendKind) -> Result<Self, String> {
        let domains = par.check_domains();
        if !domains.is_empty() {
            return Err(domains.join("; "));
        }
        let conditions = par.check();
        if !conditions.is_empty() {
            return Err(conditions.join("; "));
        }
        // The **exact backend** too, and here rather than in `setup`.
        // An outer-valid profile that violates the six-block layout or
        // `q~ > 24 phi_m eta_m` used to surface one call later, after the
        // codec and both rings had been built from it.
        crate::exact::ExactParams::new(&par)?;
        // And the backend's own **readiness**, which is not a property of
        // the parameters at all: `Lanes` is unavailable while the security
        // evidence does not reach the paper's own target.  Failing at
        // construction is what keeps a caller from getting a proof system
        // that runs and verifies against itself under a name that claims
        // more than it has.  `LanesExperimental` is the ungated name and
        // is deliberately not covered by this check.
        if backend == BackendKind::Lanes {
            if let Some(reason) = crate::exact::lanes_unavailable_reason() {
                return Err(reason);
            }
        }
        Ok(Self {
            par,
            codec: RiVeRCodec::new(par),
            rq: Ring::new(par.q(), par.d),
            backend,
        })
    }

    /// Which `Pi_ex` this instance runs with.
    pub fn backend(&self) -> BackendKind {
        self.backend
    }

    /// `pp` really is *this* instance's CRS: same profile, same backend.
    ///
    /// Every method below reads dimensions from `self.par` and matrices
    /// from `pp`, so the two have to be the same parameter set. Neither
    /// half of that was checked:
    ///
    /// * **the backend.** Everything dispatches through `pp.exact`, so
    ///   `backend()` was a claim rather than a constraint: a scheme
    ///   configured for [`BackendKind::Lanes`], handed `PublicParams` from
    ///   an `opening` setup, evaluated happily and returned an opening
    ///   proof — which *carries the witness in the clear*. It verified, so
    ///   nothing downstream noticed, and the caller had asked for zero
    ///   knowledge.
    /// * **the profile.** Comparing only the backend still admitted a
    ///   `RiVeR-N8` CRS to a `RiVeR-TOY` scheme. `A` is then `43 x 56`
    ///   against a 6-element key, and [`Ring::inner`] indexes past the end
    ///   of it: a panic in `ring.rs`, not a returned error.
    ///
    /// The profile is compared whole rather than field by field. Every
    /// derived quantity is a method on [`RiVeRParams`], so two profiles
    /// that compare equal *are* the same parameter set — which is why
    /// [`PROFILES`] entries are literals.
    ///
    /// The instance is authoritative and `pp` is checked against it, not
    /// the reverse: `par()` and `backend()` are what the caller chose, and
    /// a mismatch is their error to see rather than a silent downgrade.
    ///
    /// [`PROFILES`]: crate::params::PROFILES
    /// [`Ring::inner`]: crate::ring::Ring::inner
    fn accepts(&self, pp: &PublicParams) -> bool {
        pp.exact.kind() == self.backend && *pp.oom.par() == self.par
    }

    pub fn par(&self) -> &RiVeRParams {
        &self.par
    }

    pub fn codec(&self) -> &RiVeRCodec {
        &self.codec
    }

    /// # Panics
    ///
    /// On a profile [`RiVeR::try_new`] would refuse.  Every profile this
    /// crate ships passes; this is the convenience form for them.
    pub fn new(par: RiVeRParams) -> Self {
        Self::new_with(par, BackendKind::Opening)
    }

    /// [`RiVeR::new`] against a chosen exact backend.
    ///
    /// # Panics
    ///
    /// On a profile [`RiVeR::try_new_with`] would refuse.
    pub fn new_with(par: RiVeRParams, backend: BackendKind) -> Self {
        Self::build(par, backend).unwrap_or_else(|why| {
            panic!(
                "profile {} is not supported with {backend:?}: {why}",
                par.name
            )
        })
    }

    // ---- Setup (Figure 3) ------------------------------------------------

    /// `RiVeR.Setup(1^lambda)`, made deterministic in `seed`.
    ///
    /// The paper samples `rho` inside `OM.Setup`; this derives it from the
    /// caller's seed so a whole run is reproducible.
    ///
    /// # Panics
    ///
    /// On a profile `BoundGen` would abort — a compression margin too thin
    /// the compression margin, a modulus condition violated, a composite modulus.  That
    /// abort is the specification's; raising here is what makes it one
    /// rather than a diagnostic nobody calls.
    pub fn setup(&self, seed: &[u8]) -> PublicParams {
        // No profile check here: [`RiVeR::try_new`] is the gate, and it runs
        // before any derived state exists.  Re-checking would be free but
        // would also suggest this is where the abort lives, which is the
        // mistake that let a bad profile reach `Ring::new` in the first
        // place.
        let par = self.par;
        let rho = hash_bytes(32, &[DS_KEYGEN, b".rho"].concat(), &[Part::Bytes(seed)]);
        let a_mat = sam_mat(&rho, par.q(), par.n, par.ell, par.d, "RiVeR.A");
        let ck_seed = hash_bytes(32, &[DS_EXACT, b".ck"].concat(), &[Part::Bytes(&rho)]);
        PublicParams {
            seed: seed.to_vec(),
            oom: Oom::new(par, &rho),
            // `RiVeR::build` is the gate and has already run: it validates
            // the profile, the exact parameters and the backend's own
            // availability before any derived state exists.  Reaching an
            // `Err` here would be a defect in that gate rather than in the
            // caller's input.
            exact: ExactBackend::new(self.backend, par, &ck_seed)
                .expect("RiVeR::build validated this backend"),
            a_mat,
            rho,
        }
    }

    // ---- KeyGen (Figure 3) -----------------------------------------------

    /// `s ← S_beta^ell`, `t ← floor(A s)_p`.  Returns `(sk, pk)`.
    ///
    /// `Err` on a `pp` from a different profile or backend. This gate is
    /// not decorative: `A` is `pp`'s and its column count is the *CRS's*
    /// `ell`, while `s` is drawn at *this* profile's, so a mismatch reads
    /// past the end of `s` inside [`Ring::inner`] rather than producing a
    /// wrong key. It was the one entry point taking `pp` with no check at
    /// all.
    ///
    /// [`Ring::inner`]: crate::ring::Ring::inner
    pub fn keygen(&self, pp: &PublicParams, seed: &[u8]) -> Result<(PolyVec, PolyVec), EvalError> {
        if !self.accepts(pp) {
            return Err(EvalError::BackendMismatch);
        }
        let par = self.par;
        let mut xof = Xof::new(DS_KEYGEN, &[Part::Bytes(seed)]);
        let s = uniform_beta_vec(&mut xof, par.beta, par.d, par.ell, par.q());
        let t = self
            .rq
            .mat_vec(&pp.a_mat, &s)
            .iter()
            .map(|row| round_p(row, par.q0))
            .collect();
        Ok((s, t))
    }

    /// `h_m ← G(m)`.  One XOF, `ell` draws — a fresh XOF per polynomial
    /// would be a different `G`.
    pub fn hash_message(&self, m: &[u8]) -> PolyVec {
        let par = self.par;
        let mut xof = Xof::new(DS_G, &[Part::Bytes(m)]);
        (0..par.ell)
            .map(|_| uniform_poly(&mut xof, par.q(), par.d))
            .collect()
    }

    // ---- ring admissibility ----------------------------------------------

    /// Check the ordered ring `R` and return it unchanged.
    ///
    /// Duplicates are admissible, and the tie-break is first occurrence:
    /// `j* = min{j in [N] : t_j = pk}`.
    ///
    /// A ring with `k` copies of the evaluator's key contains only
    /// `N - k + 1` distinct identities; repeated positions do not create
    /// additional identities.
    ///
    /// What is enforced here, on both the prover and the verifier side, so
    /// that the two hash the same admissible domain:
    ///
    /// * exactly `N` keys — not "at most", and no padding;
    /// * in caller-supplied order — the order is part of the statement, and
    ///   two rings with the same members in a different order are
    ///   different statements;
    /// * every key structurally valid and canonical.
    ///
    /// `Eval` additionally requires the evaluator's key to occur, which
    /// [`RiVeR::ring_index`] establishes.  `Verify` does not need to locate
    /// an evaluator, but applies everything above.
    ///
    pub fn validate_ring(&self, ring_pks: &[PolyVec]) -> Result<Vec<PolyVec>, EvalError> {
        let par = self.par;
        if ring_pks.len() != par.N {
            return Err(EvalError::Ring("ring is not exactly N keys"));
        }
        for pk in ring_pks {
            if !self.is_valid_pk(pk) {
                return Err(EvalError::Ring("malformed public key"));
            }
            // Encodability is part of admissibility: a key that cannot be
            // encoded cannot be hashed into the statement.
            self.codec
                .pk_encode(pk)
                .map_err(|_| EvalError::Ring("public key does not encode"))?;
        }
        Ok(ring_pks.to_vec())
    }

    /// `s in S_beta^ell`: the shape *and* the range `KeyGen` produces.
    ///
    /// The range matters as much as the shape.  A `sk` inside `R_q` but
    /// outside `[-beta, beta]` would still evaluate, and would still
    /// produce a proof — one whose `z = y + x r` is wider than the
    /// verifier's bound, so it would fail after the whole attempt loop
    /// rather than at the call.
    fn is_valid_sk(&self, sk: &[Poly]) -> bool {
        let par = self.par;
        let beta = par.beta as i64;
        sk.len() == par.ell
            && sk.iter().all(|poly| {
                poly.len() == par.d
                    && poly.iter().all(|&c| c < par.q())
                    && self
                        .rq
                        .centered(poly)
                        .iter()
                        .all(|&c| (-beta..=beta).contains(&c))
            })
    }

    /// "Admissible: non-dummy public keys valid" is an input *type* in
    /// Figure 5, not a check, so it has to happen somewhere.
    fn is_valid_pk(&self, pk: &[Poly]) -> bool {
        let par = self.par;
        pk.len() == par.n
            && pk
                .iter()
                .all(|poly| poly.len() == par.d && poly.iter().all(|&c| c < par.p))
    }

    /// `j* = min{j in [N] : t_j = pk}`, the paper's hidden index.
    ///
    /// Duplicate ring entries are admissible; the tie-break is the first
    /// occurrence.
    ///
    /// **The index is the secret**, so the scan is fixed work and the
    /// index is accumulated by masking rather than by assignment — the
    /// `min` is taken with a mask too, not with an early `break`, which
    /// would publish `j*` through the loop count.  What this replaces
    /// compared *encoded* keys with slice equality, which short-circuits
    /// at the first differing byte and so leaks how far each key agrees
    /// with the evaluator's, and then took a branch on the match.
    ///
    /// Whether the key is in the ring at all is public: the caller
    /// supplied both, and a wrong answer is their input error.  How many
    /// times it occurs is public in the same sense — it is a property of
    /// the ring, which is part of the statement.
    pub fn ring_index(&self, ring: &[PolyVec], pk: &[Poly]) -> Option<usize> {
        if !self.is_valid_pk(pk) {
            return None;
        }
        let mut found = 0usize; // 0 or 1: any match seen so far
        let mut index = 0usize;
        for (i, t) in ring.iter().enumerate() {
            // 1 when the whole key matches, 0 otherwise — no early exit
            let eq = keys_equal(t, pk, self.par.n, self.par.d);
            // ...and take the index only on the *first* match, so a later
            // duplicate cannot overwrite it.  `found ^ 1` is `!found`.
            let take = eq & (found ^ 1);
            index |= i & 0usize.wrapping_sub(take);
            found |= eq;
        }
        (found == 1).then_some(index)
    }

    // ---- Eval (Figure 4) -------------------------------------------------

    /// `(v, pi) ← RiVeR.Eval(pp, pk, sk, R, m)` with the auxiliary
    /// randomness pinned.
    ///
    /// The name says the caller meant it: `seed` is *auxiliary* randomness
    /// and a production caller wants it fresh per evaluation.  See the
    /// module docs for why reuse is survivable but not something to rely
    /// on.
    pub fn eval_deterministic(
        &self,
        pp: &PublicParams,
        pk: &[Poly],
        sk: &[Poly],
        ring_pks: &[PolyVec],
        m: &[u8],
        seed: &[u8],
    ) -> Result<(Poly, Proof, EvalStats), EvalError> {
        let par = self.par;
        let r_ring = &self.rq;

        if !self.accepts(pp) {
            return Err(EvalError::BackendMismatch);
        }

        // The key material, before it reaches `Ring::inner` — which
        // iterates `u.len()` and indexes `v`, so a short `sk` is an
        // out-of-bounds read rather than an error.  `sk` is the caller's
        // own, not a peer's, but "the caller passed their own key wrong" is
        // still an error to return rather than a panic to propagate.
        if !self.is_valid_sk(sk) {
            return Err(EvalError::Key("secret key is malformed"));
        }
        if !self.is_valid_pk(pk) {
            return Err(EvalError::Key("public key is malformed"));
        }

        let ring = self.validate_ring(ring_pks)?;
        let j_star = self.ring_index(&ring, pk).ok_or(EvalError::NotAMember)?;
        let h_m = self.hash_message(m);

        // Rounding errors are canonical, in `[0, q_0-1]`; the OOM witness
        // is the centred `e^c = e - B_e`, and every public target carries
        // the matching `+B_e` (see `ring::to_centered_error`, and
        // `OomStatement::c_i`).  Both sides use the one conversion.
        let inner = r_ring.inner(&h_m, sk);
        let v = round_p(&inner, par.q0);
        let epsilon_eval = rounding_error(r_ring, &inner, &v, par.q0);
        let e_eval = to_centered_error(&epsilon_eval, par.B_e()).map_err(|_| EvalError::Witness)?;
        let e_eval_res = r_ring.from_centered(&e_eval);

        // Key-side errors use the same public centring offset.
        let as_ = r_ring.mat_vec(&pp.a_mat, sk);

        // The evaluator's own row, selected **obliviously**.
        //
        // `ring[j_star]` is a secret-indexed read: `j*` is the one thing
        // the whole OOM proof exists to hide, and the cache line this
        // touches publishes it.  One masked pass over all `N` keys costs
        // `N n d` word operations — nothing beside the eight-odd attempts
        // that follow, and it happens once per evaluation rather than per
        // attempt.
        let mut own: PolyVec = vec![vec![0u64; par.d]; par.n];
        for (j, key) in ring.iter().enumerate() {
            let m = select_mask(j, j_star);
            for i in 0..par.n {
                for k in 0..par.d {
                    own[i][k] |= key[i][k] & m;
                }
            }
        }

        // ...and, for free, the keypair relation `pk = round_p(A s)`.
        // Shape and range were checked independently, which admits a
        // well-formed *mismatched* pair: it would reach the whole attempt
        // loop and fail there, or interact with the canonical-versus-
        // centred error convention in a way that is
        // harder to read than an error at the call.  `A s` is computed
        // here anyway, so this costs a comparison.
        for i in 0..par.n {
            if round_p(&as_[i], par.q0) != own[i] {
                return Err(EvalError::Key("public key is not round_p(A sk)"));
            }
        }

        let mut r: PolyVec = sk.to_vec();
        for i in 0..par.n {
            let canonical = rounding_error(r_ring, &as_[i], &own[i], par.q0);
            let centered =
                to_centered_error(&canonical, par.B_e()).map_err(|_| EvalError::Witness)?;
            r.push(r_ring.from_centered(&centered));
        }
        r.push(e_eval_res.clone());
        if r.len() != par.r_dim() {
            return Err(EvalError::Invariant("opening has the wrong rank"));
        }

        let statement = OomStatement::new(&pp.oom, &pp.a_mat, &h_m, &ring, &v)
            .ok_or(EvalError::Ring("statement does not build"))?;
        let ck_digest = statement_digest(&self.codec, &pp.rho, &h_m)
            .map_err(|_| EvalError::Ring("statement digest"))?;

        // The honest opening really does open `c_{j*}`.  This is the
        // invariant the whole OOM layer rests on, so it is checked once per
        // evaluation — and as a real check, not a `debug_assert!`, because
        // `--release` strips those and this is the one equation that makes
        // the proof mean anything.  The keypair check above catches the
        // reachable cause; if this still fails, the parameters or the
        // statement construction are wrong, not the caller's input.
        // ...against `own`, not against `c_i(j_star)`.  `c_i` indexes the
        // ring, and doing that with `j*` would put back the
        // secret-indexed read the oblivious selection above removed.
        if statement.apply_ck(&r) != statement.c_for_key(&own) {
            return Err(EvalError::Invariant(
                "c_{j*} != Com(0; r): the honest opening does not open the claimed ring position",
            ));
        }

        let nonce = self.nonce(pp, sk, &ring, &v, m, seed)?;

        let mut attempts = 0usize;
        let mut aborts: Vec<AbortReason> = Vec::new();
        for attempt in 0..par.max_attempts {
            attempts += 1;
            let label = attempt.to_le_bytes();
            let mut com_xof = Xof::new(DS_COMMIT, &[Part::Bytes(&nonce), Part::Bytes(&label)]);
            let mut ex_xof = Xof::new(DS_EXACT, &[Part::Bytes(&nonce), Part::Bytes(&label)]);

            let (commitment, state) = pp.oom.com(&statement, j_star, &mut com_xof);

            // w_ex = (e_eval, y_eval), committed *before* the challenge
            let y_eval = r_ring.centered(&state.y_om[par.ell + par.n]);
            let witness = ExactWitness {
                e_eval: e_eval.clone(),
                y_eval: y_eval.clone(),
            };
            let (w, ex_state) = pp
                .exact
                .com(&witness, &mut ex_xof)
                .ok_or(EvalError::Witness)?;

            let rho_digest = self.rho_digest(pp, &ring, &v, &w)?;
            let Some(pi_oom) = pp.oom.prove(
                &r,
                &commitment,
                &state,
                &ck_digest,
                &rho_digest,
                &mut com_xof,
            ) else {
                // The OOM layer aborted.  Nothing from this attempt
                // survives: both XOFs are rebuilt from `(nonce, attempt)`
                // at the top of the next pass, so the masks, the selector
                // state and the exact commitment randomness are all
                // discarded together.
                aborts.push(AbortReason::Oom);
                continue;
            };

            // Only now is `z_eval` defined.  The bottom test above happens
            // first, unconditionally, matching the `Eval` figure.
            let z_eval = r_ring.centered(&pi_oom.z[par.ell + par.n]);

            // Correctness invariant: the response equation is exact over
            // Z, not merely modulo q.  If this ever fails, q is too small
            // for sigma_m — a parameter defect, not a restartable event,
            // so it is an error rather than an abort.
            let xe = Ring::mul_int(
                &pi_oom.x.iter().map(|&c| c as i128).collect::<Vec<_>>(),
                &e_eval.iter().map(|&c| c as i128).collect::<Vec<_>>(),
            );
            if !(0..par.d).all(|k| xe[k] + y_eval[k] as i128 == z_eval[k] as i128) {
                return Err(EvalError::Invariant(
                    "z_eval != x e_eval + y_eval over Z: q is too small for sigma_m",
                ));
            }

            let Some(sigma_ex) =
                pp.exact
                    .prove(&witness, &w, &z_eval, &pi_oom.x, &ex_state, &mut ex_xof)
            else {
                // The exact layer aborted.  Its own test, separate from
                // the OOM one.
                // `W` is already bound into the challenge, so the OOM
                // proof cannot be reused with a fresh exact commitment —
                // the whole attempt is discarded, which is what
                // `mu_RiVeR = mu_OOM mu_ex` accounts for.
                aborts.push(AbortReason::Exact);
                continue;
            };
            return Ok((
                v,
                Proof {
                    oom: pi_oom,
                    w,
                    sigma_ex,
                },
                EvalStats { attempts, aborts },
            ));
        }
        Err(EvalError::Exhausted(attempts))
    }

    /// `H(seed || sk || ring || m)` — see the module docs.
    fn nonce(
        &self,
        _pp: &PublicParams,
        sk: &[Poly],
        padded: &[PolyVec],
        v: &[u64],
        m: &[u8],
        seed: &[u8],
    ) -> Result<Vec<u8>, EvalError> {
        let sk_bytes = self
            .codec
            .sk_encode(sk)
            .map_err(|_| EvalError::Ring("secret key does not encode"))?;
        let ring =
            ring_digest(&self.codec, padded, v).map_err(|_| EvalError::Ring("ring digest"))?;
        Ok(hash_bytes(
            32,
            &[DS_COMMIT, b".nonce"].concat(),
            &[
                Part::Bytes(seed),
                Part::Bytes(&sk_bytes),
                Part::Bytes(&ring),
                Part::Bytes(m),
            ],
        ))
    }

    /// Digest of `rho' = (R~, v, W)`.
    fn rho_digest(
        &self,
        pp: &PublicParams,
        padded: &[PolyVec],
        v: &[u64],
        w: &ExactCom,
    ) -> Result<Vec<u8>, EvalError> {
        let base =
            ring_digest(&self.codec, padded, v).map_err(|_| EvalError::Ring("ring digest"))?;
        let w_bytes = pp
            .exact
            .w_encode(w)
            .map_err(|_| EvalError::Ring("W does not encode"))?;
        Ok(hash_bytes(
            32,
            &[DS_COMMIT, b".rho'"].concat(),
            &[Part::Bytes(&base), Part::Bytes(&w_bytes)],
        ))
    }

    // ---- Verify (Figure 5) -----------------------------------------------

    /// `RiVeR.Verify(pp, R, m, v, pi)` in `{0, 1}`.
    ///
    /// Total on all four peer-supplied arguments.  `pp` is not in that
    /// class: it is the CRS, which every party is assumed to hold the same
    /// honestly generated copy of.
    pub fn verify(
        &self,
        pp: &PublicParams,
        ring_pks: &[PolyVec],
        m: &[u8],
        v: &[u64],
        pi: &Proof,
    ) -> bool {
        let par = self.par;
        if !self.accepts(pp) {
            return false;
        }
        let Ok(padded) = self.validate_ring(ring_pks) else {
            return false;
        };
        // **Choice.**  Figure 5 never checks that `v` is canonically
        // reduced; a non-canonical `v` changes `q_0 v mod q` and hence the
        // whole statement.  The ring check above moved the other way — the
        // The paper drops the admissible-ring set from every
        // experiment, so no admissibility condition is stated anywhere now.
        // Both are kept.
        if v.len() != par.d || v.iter().any(|&c| c >= par.p) {
            return false;
        }

        let h_m = self.hash_message(m);
        let Some(statement) = OomStatement::new(&pp.oom, &pp.a_mat, &h_m, &padded, v) else {
            return false;
        };
        let Ok(ck_digest) = statement_digest(&self.codec, &pp.rho, &h_m) else {
            return false;
        };

        // `pi` reaches here either from `proof_decode`, which has validated
        // every field, or straight from a caller's struct, which has not.
        // Re-encoding applies exactly the same checks to both: a wrong
        // shape, a coefficient outside its declared range, a non-canonical
        // residue.  A `t_g + q~` or a shifted `z` must not verify just
        // because it arrived as a value rather than as bytes.
        if self.encode_proof(pp, pi).is_err() {
            return false;
        }

        let Ok(rho_digest) = self.rho_digest(pp, &padded, v, &pi.w) else {
            return false;
        };
        if !pp.oom.verify(&statement, &pi.oom, &ck_digest, &rho_digest) {
            return false;
        }

        let z_eval = self.rq.centered(&pi.oom.z[par.ell + par.n]);
        pp.exact.verify(&pi.w, &z_eval, &pi.oom.x, &pi.sigma_ex)
    }

    // ---- serialization ---------------------------------------------------

    fn encode_proof(&self, pp: &PublicParams, pi: &Proof) -> Result<(Vec<u8>, Vec<u8>), ()> {
        if !self.accepts(pp) {
            return Err(());
        }
        let values = self
            .codec
            .oom_field_values(&pi.oom.b_hi, &pi.oom.x, &pi.oom.f1, &pi.oom.zb, &pi.oom.z)
            .map_err(|_| ())?;
        let oom = self.codec.oom_encode(&values).map_err(|_| ())?;
        let ex = pp.exact.proof_encode(&pi.w, &pi.sigma_ex).map_err(|_| ())?;
        Ok((oom, ex))
    }

    /// `pi` as two length-prefixed blocks.
    pub fn proof_encode(&self, pp: &PublicParams, pi: &Proof) -> Option<Vec<u8>> {
        let (oom, ex) = self.encode_proof(pp, pi).ok()?;
        Some(proof_frame(&oom, &ex))
    }

    /// Inverse of [`RiVeR::proof_encode`].  `None` on any malformation.
    pub fn proof_decode(&self, pp: &PublicParams, data: &[u8]) -> Option<Proof> {
        if !self.accepts(pp) {
            return None;
        }
        let (oom_bytes, ex_bytes) =
            proof_unframe(data, &self.codec.oom_layout, pp.exact.proof_layout()).ok()?;
        let mut f = self.codec.oom_decode(oom_bytes).ok()?.into_iter();
        let b_hi = ints(f.next()?);
        let x = ints(f.next()?).into_iter().next().unwrap_or_default();
        let f1 = ints(f.next()?);
        let zb = ints(f.next()?);
        // `z` arrives split — `z_s` at `sigma_s`, `z_m` at `sigma_m` — and
        // is reassembled immediately, because the protocol and the
        // verifier's Euclidean check operate on the whole vector.
        let zs = residues(f.next()?);
        let zm = residues(f.next()?);
        let z = self.codec.oom_z_from_values(zs, zm);
        let (w, sigma_ex) = pp.exact.proof_decode(ex_bytes).ok()?;
        Some(Proof {
            oom: OomProof { b_hi, x, f1, zb, z },
            w,
            sigma_ex,
        })
    }
}

/// `1` when two public keys are equal, `0` otherwise — fixed work.
///
/// No short-circuit: slice equality stops at the first differing byte,
/// which tells an observer how far each ring member agrees with the
/// evaluator's key and therefore which one it is.  The shapes are checked
/// first and are public (`validate_ring` and `is_valid_pk` have run), so
/// only the *coefficients* are folded into the difference mask.
#[inline]
fn keys_equal(a: &[Poly], b: &[Poly], n: usize, d: usize) -> usize {
    if a.len() != n || b.len() != n {
        return 0;
    }
    let mut diff = 0u64;
    for (x, y) in a.iter().zip(b.iter()) {
        if x.len() != d || y.len() != d {
            return 0;
        }
        for (&p, &q) in x.iter().zip(y.iter()) {
            diff |= p ^ q;
        }
    }
    // 1 exactly when `diff == 0`
    (((diff | diff.wrapping_neg()) >> 63) ^ 1) as usize
}

/// All-ones when `j == j_star`, zero otherwise.
///
/// The same device `oom::eq_mask` uses, and for the same reason: `j*`
/// must not choose an address.  Kept here rather than shared so neither
/// module's timing property depends on the other's visibility.
#[inline(always)]
fn select_mask(j: usize, j_star: usize) -> u64 {
    let diff = (j ^ j_star) as u64;
    let nonzero = ((diff | diff.wrapping_neg()) >> 63) & 1;
    nonzero.wrapping_sub(1)
}

fn ints(v: crate::codec::FieldValue) -> Vec<Vec<i64>> {
    match v {
        crate::codec::FieldValue::Ints(r) => r,
        crate::codec::FieldValue::Residues(r) => r
            .into_iter()
            .map(|row| row.into_iter().map(|c| c as i64).collect())
            .collect(),
    }
}

fn residues(v: crate::codec::FieldValue) -> PolyVec {
    match v {
        crate::codec::FieldValue::Residues(r) => r,
        crate::codec::FieldValue::Ints(r) => r
            .into_iter()
            .map(|row| row.into_iter().map(|c| c as u64).collect())
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::{RIVER_N8, RIVER_TOY};

    // Tests that need the LANES layer used to skip, because
    // `BackendKind::Lanes` is gated; see the note in
    // `crate::lanes::backend`.  They run under
    // `BackendKind::LanesExperimental` now — the same code under the name
    // an artifact can honestly record.

    fn ring_of(scheme: &RiVeR, pp: &PublicParams, n: usize) -> Vec<(PolyVec, PolyVec)> {
        (0..n)
            .map(|i| {
                let mut seed = [0u8; 32];
                seed[0] = i as u8;
                scheme.keygen(pp, &seed).expect("pp is this scheme's")
            })
            .collect()
    }

    #[test]
    fn eval_then_verify() {
        for par in [RIVER_TOY, RIVER_N8] {
            let scheme = RiVeR::new(par);
            let pp = scheme.setup(&[0u8; 32]);
            let keys = ring_of(&scheme, &pp, par.N);
            let ring: Vec<PolyVec> = keys.iter().map(|(_, pk)| pk.clone()).collect();
            let (sk, pk) = &keys[1];

            let (v, pi, stats) = scheme
                .eval_deterministic(&pp, pk, sk, &ring, b"hello", &[0xAA; 32])
                .expect("eval");
            assert!(stats.attempts >= 1);
            assert!(scheme.verify(&pp, &ring, b"hello", &v, &pi), "{}", par.name);

            // the value is a canonical R_p element
            assert_eq!(v.len(), par.d);
            assert!(v.iter().all(|&c| c < par.p));
        }
    }

    #[test]
    fn a_proof_round_trips_through_bytes() {
        let par = RIVER_TOY;
        let scheme = RiVeR::new(par);
        let pp = scheme.setup(&[1u8; 32]);
        let keys = ring_of(&scheme, &pp, par.N);
        let ring: Vec<PolyVec> = keys.iter().map(|(_, pk)| pk.clone()).collect();
        let (sk, pk) = &keys[0];

        let (v, pi, _) = scheme
            .eval_deterministic(&pp, pk, sk, &ring, b"bytes", &[7u8; 32])
            .unwrap();
        let blob = scheme.proof_encode(&pp, &pi).unwrap();
        let back = scheme.proof_decode(&pp, &blob).unwrap();
        assert_eq!(back, pi);
        assert!(scheme.verify(&pp, &ring, b"bytes", &v, &back));

        // every truncation and every trailing byte is refused
        for cut in [0, 1, 4, blob.len() / 2, blob.len() - 1] {
            assert!(
                scheme.proof_decode(&pp, &blob[..cut]).is_none(),
                "cut {cut}"
            );
        }
        let mut longer = blob.clone();
        longer.push(0);
        assert!(scheme.proof_decode(&pp, &longer).is_none());
    }

    /// Everything the proof is bound to, changed one at a time.
    #[test]
    fn the_proof_is_bound_to_its_statement() {
        let par = RIVER_TOY;
        let scheme = RiVeR::new(par);
        let pp = scheme.setup(&[2u8; 32]);
        let keys = ring_of(&scheme, &pp, par.N);
        let ring: Vec<PolyVec> = keys.iter().map(|(_, pk)| pk.clone()).collect();
        let (sk, pk) = &keys[2];

        let (v, pi, _) = scheme
            .eval_deterministic(&pp, pk, sk, &ring, b"bind", &[3u8; 32])
            .unwrap();
        assert!(scheme.verify(&pp, &ring, b"bind", &v, &pi));

        // a different message
        assert!(!scheme.verify(&pp, &ring, b"other", &v, &pi));

        // a different value
        let mut moved = v.clone();
        moved[0] = (moved[0] + 1) % par.p;
        assert!(!scheme.verify(&pp, &ring, b"bind", &moved, &pi));

        // a different ring, same size
        let extra = ring_of(&scheme, &pp, par.N + 1);
        let other_ring: Vec<PolyVec> = extra[1..=par.N].iter().map(|(_, p)| p.clone()).collect();
        assert!(!scheme.verify(&pp, &other_ring, b"bind", &v, &pi));

        // a different setup
        let pp2 = scheme.setup(&[9u8; 32]);
        assert!(!scheme.verify(&pp2, &ring, b"bind", &v, &pi));

        // and each proof component
        let mut bad = pi.clone();
        bad.oom.x[0] = -bad.oom.x[0];
        assert!(!scheme.verify(&pp, &ring, b"bind", &v, &bad));
        let mut bad = pi.clone();
        bad.sigma_ex.opening_mut().unwrap().y_eval[0] += 1;
        assert!(!scheme.verify(&pp, &ring, b"bind", &v, &bad));
        let mut bad = pi.clone();
        let w = bad.w.opening_mut().unwrap();
        w.t0[0][0] = (w.t0[0][0] + 1) % crate::exact::ExactParams::Q_TILDE;
        assert!(!scheme.verify(&pp, &ring, b"bind", &v, &bad));
    }

    /// `Verify` is total: no shape of peer input is a panic.
    #[test]
    fn verify_is_total_on_peer_input() {
        let par = RIVER_TOY;
        let scheme = RiVeR::new(par);
        let pp = scheme.setup(&[4u8; 32]);
        let keys = ring_of(&scheme, &pp, par.N);
        let ring: Vec<PolyVec> = keys.iter().map(|(_, pk)| pk.clone()).collect();
        let (sk, pk) = &keys[0];
        let (v, pi, _) = scheme
            .eval_deterministic(&pp, pk, sk, &ring, b"total", &[5u8; 32])
            .unwrap();

        // rings: empty, over-long, ragged, duplicated, non-canonical
        assert!(!scheme.verify(&pp, &[], b"total", &v, &pi));
        let mut too_many = ring.clone();
        while too_many.len() <= par.N {
            too_many.push(ring[0].clone());
        }
        assert!(!scheme.verify(&pp, &too_many, b"total", &v, &pi));
        let dup = vec![ring[0].clone(), ring[0].clone()];
        assert!(!scheme.verify(&pp, &dup, b"total", &v, &pi));
        let mut ragged = ring.clone();
        ragged[0][0].pop();
        assert!(!scheme.verify(&pp, &ragged, b"total", &v, &pi));
        let mut noncanon = ring.clone();
        noncanon[0][0][0] = par.p;
        assert!(!scheme.verify(&pp, &noncanon, b"total", &v, &pi));
        let short_pk = vec![ring[0][..par.n - 1].to_vec()];
        assert!(!scheme.verify(&pp, &short_pk, b"total", &v, &pi));

        // values: wrong length, non-canonical
        assert!(!scheme.verify(&pp, &ring, b"total", &[], &pi));
        assert!(!scheme.verify(&pp, &ring, b"total", &vec![0u64; par.d + 1], &pi));
        let mut bad_v = v.clone();
        bad_v[0] = par.p;
        assert!(!scheme.verify(&pp, &ring, b"total", &bad_v, &pi));

        // proofs: every field, wrong shape and extreme value
        let mut empty = pi.clone();
        empty.oom.x = vec![];
        assert!(!scheme.verify(&pp, &ring, b"total", &v, &empty));
        let mut extreme = pi.clone();
        extreme.oom.x = vec![i64::MIN; par.d];
        assert!(!scheme.verify(&pp, &ring, b"total", &v, &extreme));
        let mut extreme = pi.clone();
        extreme.oom.b_hi[0][0] = i64::MIN;
        assert!(!scheme.verify(&pp, &ring, b"total", &v, &extreme));
        let mut extreme = pi.clone();
        extreme.oom.zb[0][0] = i64::MAX;
        assert!(!scheme.verify(&pp, &ring, b"total", &v, &extreme));
        let mut extreme = pi.clone();
        extreme.sigma_ex.opening_mut().unwrap().e_eval = vec![i64::MAX; par.d];
        assert!(!scheme.verify(&pp, &ring, b"total", &v, &extreme));
        let mut extreme = pi.clone();
        extreme.sigma_ex.opening_mut().unwrap().randomness = vec![];
        assert!(!scheme.verify(&pp, &ring, b"total", &v, &extreme));
        let mut extreme = pi;
        extreme.oom.z = vec![vec![u64::MAX; par.d]; par.r_dim()];
        assert!(!scheme.verify(&pp, &ring, b"total", &v, &extreme));

        // and arbitrary bytes never decode to something that verifies
        for len in [0usize, 1, 8, 64, 4096] {
            let junk = vec![0xABu8; len];
            assert!(scheme.proof_decode(&pp, &junk).is_none(), "len {len}");
        }
    }

    /// The evaluation is a function of its inputs, and the nonce binds the
    /// message and the key — a reused seed does not reuse masks.
    #[test]
    fn evaluation_is_deterministic_and_the_nonce_binds() {
        let par = RIVER_TOY;
        let scheme = RiVeR::new(par);
        let pp = scheme.setup(&[6u8; 32]);
        let keys = ring_of(&scheme, &pp, par.N);
        let ring: Vec<PolyVec> = keys.iter().map(|(_, pk)| pk.clone()).collect();
        let (sk, pk) = &keys[1];

        let one = scheme
            .eval_deterministic(&pp, pk, sk, &ring, b"m", &[8u8; 32])
            .unwrap();
        let two = scheme
            .eval_deterministic(&pp, pk, sk, &ring, b"m", &[8u8; 32])
            .unwrap();
        assert_eq!(one.0, two.0);
        assert_eq!(one.1, two.1);

        // the value is a function of (sk, m) only
        let other_ring: Vec<PolyVec> = keys.iter().rev().map(|(_, p)| p.clone()).collect();
        let three = scheme
            .eval_deterministic(&pp, pk, sk, &other_ring, b"m", &[8u8; 32])
            .unwrap();
        assert_eq!(three.0, one.0, "v must not depend on the ring order");

        // the same seed on a different message must not reuse the mask
        let four = scheme
            .eval_deterministic(&pp, pk, sk, &ring, b"m2", &[8u8; 32])
            .unwrap();
        assert_ne!(four.1.oom.z, one.1.oom.z);
        assert_ne!(four.1.w, one.1.w);
    }

    /// `Eval` returns an error on malformed key material, never a panic.
    ///
    /// `sk` is the caller's own rather than a peer's, but `Ring::inner`
    /// iterates `u.len()` and indexes `v`, so a short key was an
    /// out-of-bounds read before anything looked at it.
    #[test]
    fn malformed_key_material_is_an_error_not_a_panic() {
        let par = RIVER_TOY;
        let scheme = RiVeR::new(par);
        let pp = scheme.setup(&[11u8; 32]);
        let keys = ring_of(&scheme, &pp, par.N);
        let ring: Vec<PolyVec> = keys.iter().map(|(_, pk)| pk.clone()).collect();
        let (sk, pk) = &keys[1];
        assert!(scheme
            .eval_deterministic(&pp, pk, sk, &ring, b"k", &[1u8; 32])
            .is_ok());

        for bad in [
            vec![],
            sk[..par.ell - 1].to_vec(),
            {
                let mut s = sk.clone();
                s.push(sk[0].clone());
                s
            },
            {
                let mut s = sk.clone();
                s[0].pop();
                s
            },
            {
                // canonical, but outside `S_beta`: this one would have
                // evaluated and produced a proof its own verifier rejects
                let mut s = sk.clone();
                s[0][0] = par.beta + 1;
                s
            },
            {
                let mut s = sk.clone();
                s[0][0] = par.q();
                s
            },
        ] {
            assert_eq!(
                scheme.eval_deterministic(&pp, pk, &bad, &ring, b"k", &[1u8; 32]),
                Err(EvalError::Key("secret key is malformed")),
            );
        }

        for bad_pk in [vec![], pk[..par.n - 1].to_vec()] {
            assert!(matches!(
                scheme.eval_deterministic(&pp, &bad_pk, sk, &ring, b"k", &[1u8; 32]),
                Err(EvalError::Key(_))
            ));
        }
    }

    /// A well-formed but *mismatched* keypair is refused at the call.
    #[test]
    fn the_keypair_relation_is_checked() {
        let par = RIVER_TOY;
        let scheme = RiVeR::new(par);
        let pp = scheme.setup(&[12u8; 32]);
        let keys = ring_of(&scheme, &pp, par.N);
        let ring: Vec<PolyVec> = keys.iter().map(|(_, pk)| pk.clone()).collect();

        // key 0's secret with key 1's public: both well-formed, unrelated
        assert_eq!(
            scheme.eval_deterministic(&pp, &keys[1].1, &keys[0].0, &ring, b"m", &[1u8; 32]),
            Err(EvalError::Key("public key is not round_p(A sk)")),
        );
        // and the matching pair still works
        assert!(scheme
            .eval_deterministic(&pp, &keys[1].1, &keys[1].0, &ring, b"m", &[1u8; 32])
            .is_ok());
    }

    /// The whole scheme runs on the candidate LANES backend too.
    ///
    /// `tests/vectors.rs` pins the *bytes* against the reference; this
    /// checks the properties the bytes cannot show — that the proof still
    /// verifies after a round trip, that it is bound to its statement, and
    /// that it does not carry the witness the opening backend transmits.
    #[test]
    fn the_scheme_runs_on_the_lanes_backend() {
        for par in [RIVER_TOY, RIVER_N8] {
            let scheme = RiVeR::new_with(par, BackendKind::LanesExperimental);
            assert_eq!(scheme.backend(), BackendKind::LanesExperimental);
            let pp = scheme.setup(&[0x31; 32]);
            assert_eq!(pp.exact().kind(), BackendKind::LanesExperimental);
            let keys = ring_of(&scheme, &pp, par.N);
            let ring: Vec<PolyVec> = keys.iter().map(|(_, pk)| pk.clone()).collect();
            let (sk, pk) = &keys[1];

            let (v, pi, _) = scheme
                .eval_deterministic(&pp, pk, sk, &ring, b"zk", &[0x32; 32])
                .expect("eval");
            assert!(scheme.verify(&pp, &ring, b"zk", &v, &pi), "{}", par.name);
            assert!(pi.sigma_ex.lanes().is_some(), "wrong sigma variant");
            assert!(pi.w.lanes().is_some(), "wrong W variant");

            // bytes round trip, and the decoded proof still verifies
            let blob = scheme.proof_encode(&pp, &pi).unwrap();
            let back = scheme.proof_decode(&pp, &blob).unwrap();
            assert_eq!(back, pi, "{}", par.name);
            assert!(scheme.verify(&pp, &ring, b"zk", &v, &back));

            // bound to the message, the value and the ring
            assert!(!scheme.verify(&pp, &ring, b"other", &v, &pi));
            let mut moved = v.clone();
            moved[0] = (moved[0] + 1) % par.p;
            assert!(!scheme.verify(&pp, &ring, b"zk", &moved, &pi));

            // and every field of `sigma_ex` is bound
            let mut bad = pi.clone();
            let z = &mut bad.sigma_ex.lanes_mut().unwrap().z[0];
            let mut coeffs = z.to_vec();
            coeffs[0] = (coeffs[0] + 1) % crate::exact::ExactParams::Q_TILDE;
            *z = crate::lanes::ring::CoeffPoly::new(&coeffs).unwrap();
            assert!(!scheme.verify(&pp, &ring, b"zk", &v, &bad));
        }
    }

    /// The two backends are not interchangeable, and saying so is `false`.
    ///
    /// The mismatched-variant arms of [`ExactBackend::prove`],
    /// `verify` and `proof_encode` are reachable — a caller can build a
    /// [`Proof`] by hand out of two evaluations — so they are checked
    /// rather than assumed unreachable.
    #[test]
    fn a_proof_from_the_other_backend_is_refused() {
        let par = RIVER_TOY;
        let opening = RiVeR::new_with(par, BackendKind::Opening);
        let lanes = RiVeR::new_with(par, BackendKind::LanesExperimental);
        let pp_o = opening.setup(&[0x41; 32]);
        let pp_l = lanes.setup(&[0x41; 32]);

        let keys = ring_of(&opening, &pp_o, par.N);
        let ring: Vec<PolyVec> = keys.iter().map(|(_, pk)| pk.clone()).collect();
        let (sk, pk) = &keys[0];

        let (v_o, pi_o, _) = opening
            .eval_deterministic(&pp_o, pk, sk, &ring, b"x", &[0x42; 32])
            .unwrap();
        let (v_l, pi_l, _) = lanes
            .eval_deterministic(&pp_l, pk, sk, &ring, b"x", &[0x42; 32])
            .unwrap();
        assert!(opening.verify(&pp_o, &ring, b"x", &v_o, &pi_o));
        assert!(lanes.verify(&pp_l, &ring, b"x", &v_l, &pi_l));

        // the value is a function of `(sk, m)` and does not see the backend
        assert_eq!(v_o, v_l, "the VRF value must not depend on Pi_ex");

        // each proof against the other's public parameters
        assert!(
            !lanes.verify(&pp_l, &ring, b"x", &v_o, &pi_o),
            "opening/lanes"
        );
        assert!(
            !opening.verify(&pp_o, &ring, b"x", &v_l, &pi_l),
            "lanes/opening"
        );

        // and the encoders refuse the mismatch rather than producing bytes
        assert!(pp_l.exact().proof_encode(&pi_o.w, &pi_o.sigma_ex).is_err());
        assert!(pp_o.exact().w_encode(&pi_l.w).is_err());
        assert!(pp_o.exact().opening().is_some() && pp_o.exact().lanes().is_none());
        assert!(pp_l.exact().lanes().is_some() && pp_l.exact().opening().is_none());
    }

    /// Smoke test: proof length shows no signer dependence at this sample
    /// size.  It does **not** establish signer independence.
    ///
    /// What it checks is that twelve proofs per backend sit in a band a
    /// tenth of a percent wide and that one signer alone spans most of it.
    /// Four samples per signer cannot separate the three signers'
    /// *distributions*: a small signer-dependent shift passes both
    /// inequalities comfortably.  Establishing the property would want
    /// pairwise distribution tests at a far larger sample count, or
    /// fixed-length padding if the hiding has to be unconditional.  This
    /// is here to catch a gross regression — a field whose width tracks
    /// the witness — not to be the evidence.
    ///
    /// The *argument* is structural, and differs by backend:
    ///
    /// * `opening` Rice-codes `y_eval`, a coordinate of the OOM mask.
    ///   After rejection sampling it is distributed as `D_sigma`
    ///   independently of the witness, which is what makes its length a
    ///   witness-independent random variable.
    /// * `lanes` Rice-codes `z = y + c r`, and has **no** rejection
    ///   sampling at all — the `[KLSS23]` Hint-MLWE treatment is what
    ///   removes it, so citing rejection here (as an earlier version of
    ///   this comment did) is citing a mechanism the backend deletes.  The
    ///   argument instead is that `y` and `r` are drawn independently of
    ///   the witness and the witness reaches `z` only through `c`, which
    ///   ranges over a challenge space of fixed weight `w_hat` whatever was
    ///   committed.  Conditioned on `c`, the law of `z` does not move.
    ///
    /// The shipped vectors have one signer per case and so cannot cover
    /// this at all, which is why it is here rather than there.
    #[test]
    fn proof_length_smoke_test_shows_no_signer_dependence() {
        let par = RIVER_TOY;
        for kind in [BackendKind::Opening, BackendKind::LanesExperimental] {
            let scheme = RiVeR::new_with(par, kind);
            let pp = scheme.setup(&[0x64; 32]);
            let keys = ring_of(&scheme, &pp, par.N);
            let ring: Vec<PolyVec> = keys.iter().map(|(_, pk)| pk.clone()).collect();

            let per_signer: Vec<Vec<usize>> = keys
                .iter()
                .map(|(sk, pk)| {
                    (0..4u8)
                        .map(|seed| {
                            let (v, pi, _) = scheme
                                .eval_deterministic(&pp, pk, sk, &ring, b"anon", &[seed; 32])
                                .expect("eval");
                            assert!(scheme.verify(&pp, &ring, b"anon", &v, &pi));
                            scheme.proof_encode(&pp, &pi).unwrap().len()
                        })
                        .collect()
                })
                .collect();

            let every: Vec<usize> = per_signer.iter().flatten().copied().collect();
            let spread = every.iter().max().unwrap() - every.iter().min().unwrap();
            let mean = every.iter().sum::<usize>() as f64 / every.len() as f64;

            // the whole population sits in a band far too narrow to be a
            // per-signer offset — necessary, nowhere near sufficient
            assert!(
                (spread as f64) < 0.01 * mean,
                "{}: spread {spread} of mean {mean}",
                kind.name()
            );
            // and one signer alone already covers most of it, so the
            // variation is randomness rather than identity
            let widest = per_signer
                .iter()
                .map(|row| row.iter().max().unwrap() - row.iter().min().unwrap())
                .max()
                .unwrap();
            assert!(
                widest * 2 >= spread,
                "{}: widest single signer {widest} against spread {spread}",
                kind.name()
            );
        }
    }

    /// The instance's backend is enforced against `pp`, not merely claimed.
    ///
    /// Every method dispatches through `pp.exact`, so before this a scheme
    /// configured for LANES, handed `opening` public parameters, evaluated
    /// happily and returned an *opening* proof — which carries `e_eval` in
    /// the clear.  It verified, so nothing downstream noticed, and the
    /// caller had asked for zero knowledge.
    #[test]
    fn public_parameters_from_the_other_backend_are_refused() {
        let par = RIVER_TOY;
        let opening = RiVeR::new_with(par, BackendKind::Opening);
        let lanes = RiVeR::new_with(par, BackendKind::LanesExperimental);
        let pp_o = opening.setup(&[0x51; 32]);
        let pp_l = lanes.setup(&[0x51; 32]);

        let keys = ring_of(&opening, &pp_o, par.N);
        let ring: Vec<PolyVec> = keys.iter().map(|(_, pk)| pk.clone()).collect();
        let (sk, pk) = &keys[1];

        // the mismatch that used to downgrade zero knowledge in silence
        assert_eq!(
            lanes.eval_deterministic(&pp_o, pk, sk, &ring, b"m", &[0x52; 32]),
            Err(EvalError::BackendMismatch),
        );
        assert_eq!(
            opening.eval_deterministic(&pp_l, pk, sk, &ring, b"m", &[0x52; 32]),
            Err(EvalError::BackendMismatch),
        );

        // and each matching pair still works
        let (v, pi, _) = lanes
            .eval_deterministic(&pp_l, pk, sk, &ring, b"m", &[0x52; 32])
            .unwrap();
        assert!(pi.sigma_ex.lanes().is_some(), "zero knowledge, as asked");
        assert!(lanes.verify(&pp_l, &ring, b"m", &v, &pi));

        // `Verify` and both codec entry points refuse it too, rather than
        // reading the wrong layout out of the wrong `pp`
        assert!(!opening.verify(&pp_l, &ring, b"m", &v, &pi));
        assert!(opening.proof_encode(&pp_l, &pi).is_none());
        let blob = lanes.proof_encode(&pp_l, &pi).unwrap();
        assert!(opening.proof_decode(&pp_l, &blob).is_none());
        assert!(lanes.proof_decode(&pp_o, &blob).is_none());
    }

    /// A CRS from a different *profile* is refused, at every entry point.
    ///
    /// `accepts` used to compare only the backend, so a `RiVeR-N8` CRS
    /// reached a `RiVeR-TOY` scheme: `A` is then `43 x 56` against a
    /// 6-element key, and `keygen` — which had no gate at all — indexed
    /// past the end of `s` inside `Ring::inner` and panicked in `ring.rs`.
    /// Every method here takes dimensions from `self.par` and matrices
    /// from `pp`, so nothing downstream can notice the substitution.
    #[test]
    fn public_parameters_from_a_different_profile_are_refused() {
        let toy = RiVeR::new(RIVER_TOY);
        let n8 = RiVeR::new(RIVER_N8);
        let pp_toy = toy.setup(&[0x61; 32]);
        let pp_n8 = n8.setup(&[0x61; 32]);
        // the two profiles really do disagree on the shape that matters
        assert_ne!((RIVER_TOY.ell, RIVER_TOY.n), (RIVER_N8.ell, RIVER_N8.n));

        // `keygen` first: it is the one that used to panic rather than
        // return, and it is what every later call is built on
        assert_eq!(
            toy.keygen(&pp_n8, &[0u8; 32]),
            Err(EvalError::BackendMismatch)
        );
        assert_eq!(
            n8.keygen(&pp_toy, &[0u8; 32]),
            Err(EvalError::BackendMismatch)
        );

        let keys = ring_of(&toy, &pp_toy, RIVER_TOY.N);
        let ring: Vec<PolyVec> = keys.iter().map(|(_, pk)| pk.clone()).collect();
        let (sk, pk) = &keys[1];
        assert_eq!(
            toy.eval_deterministic(&pp_n8, pk, sk, &ring, b"m", &[0x62; 32]),
            Err(EvalError::BackendMismatch),
        );

        let (v, pi, _) = toy
            .eval_deterministic(&pp_toy, pk, sk, &ring, b"m", &[0x62; 32])
            .unwrap();
        assert!(toy.verify(&pp_toy, &ring, b"m", &v, &pi));
        assert!(!toy.verify(&pp_n8, &ring, b"m", &v, &pi));
        assert!(toy.proof_encode(&pp_n8, &pi).is_none());
        let blob = toy.proof_encode(&pp_toy, &pi).unwrap();
        assert!(toy.proof_decode(&pp_n8, &blob).is_none());

        // and the profile is compared whole, not by the backend alone
        assert_eq!(pp_n8.exact().kind(), pp_toy.exact().kind());
    }

    /// The backend a case names round trips through its own spelling.
    #[test]
    fn the_backend_names_are_the_references() {
        for kind in [BackendKind::Opening, BackendKind::LanesExperimental] {
            assert_eq!(BackendKind::from_name(kind.name()), Some(kind));
        }
        assert_eq!(BackendKind::Opening.name(), "opening");
        assert_eq!(BackendKind::Lanes.name(), "lanes");
        assert_eq!(BackendKind::from_name("nope"), None);
        // the default is the opening backend, on both sides
        assert_eq!(RiVeR::new(RIVER_TOY).backend(), BackendKind::Opening);
    }

    /// The profile is checked before anything is derived from it.
    #[test]
    fn an_unsupported_profile_is_refused_before_any_derived_state() {
        assert!(RiVeR::try_new(RIVER_TOY).is_some());
        for par in crate::params::PROFILES {
            assert!(RiVeR::try_new(par).is_some(), "{}", par.name);
        }

        // `RiVeRCodec::new` shifts by `K_b` and `Ring::new` builds a
        // Barrett reduction for `q_0 · p`; both used to run first.
        let mut zero_p = RIVER_TOY;
        zero_p.p = 0;
        assert!(RiVeR::try_new(zero_p).is_none(), "p = 0");

        let mut wide_kb = RIVER_TOY;
        wide_kb.K_b = 200;
        assert!(RiVeR::try_new(wide_kb).is_none(), "K_b = 200");

        let mut overflow = RIVER_TOY;
        overflow.p = u64::MAX;
        assert!(RiVeR::try_new(overflow).is_none(), "q_0 · p overflows");

        let mut zero_d = RIVER_TOY;
        zero_d.d = 0;
        assert!(RiVeR::try_new(zero_d).is_none(), "d = 0");
    }

    /// A ring is an ordered tuple of exactly `N` valid keys; duplicates are
    /// admissible and order is part of the statement.
    ///
    /// The order is part of the statement, so validation returns the ring
    /// unchanged and two orderings of the same members are two different
    /// statements.  There is no sorting or padding.
    #[test]
    fn a_ring_is_ordered_and_exactly_n_keys() {
        let par = RIVER_TOY;
        let scheme = RiVeR::new(par);
        let pp = scheme.setup(&[10u8; 32]);
        let keys = ring_of(&scheme, &pp, par.N);
        let ring: Vec<PolyVec> = keys.iter().map(|(_, pk)| pk.clone()).collect();

        // exactly `N`, unchanged, in the caller's order
        let validated = scheme.validate_ring(&ring).unwrap();
        assert_eq!(validated, ring);

        // order is part of the statement: reversing gives a different one
        let reversed: Vec<PolyVec> = ring.iter().rev().cloned().collect();
        let other = scheme.validate_ring(&reversed).unwrap();
        assert_eq!(other, reversed);
        assert_ne!(other, validated, "the ring is not reordered");
        assert_ne!(
            scheme.ring_index(&validated, &ring[0]),
            scheme.ring_index(&other, &ring[0]),
            "the evaluator's index moves with the order"
        );

        // and every key in this distinct fixture is findable
        for pk in &ring {
            assert!(scheme.ring_index(&validated, pk).is_some());
        }

        // short, long and malformed all fail cleanly
        assert!(scheme.validate_ring(&ring[..par.N - 1]).is_err(), "short");
        let mut long = ring.clone();
        long.push(ring[0].clone());
        assert!(scheme.validate_ring(&long).is_err(), "long");
        let mut malformed = ring.clone();
        malformed[0][0][0] = par.p; // non-canonical residue
        assert!(scheme.validate_ring(&malformed).is_err(), "non-canonical");
        let mut short_key = ring.clone();
        short_key[0].pop();
        assert!(scheme.validate_ring(&short_key).is_err(), "short key");
    }

    /// Duplicate entries are admissible and `j*` is the first matching
    /// position, so a duplicated ring must prove and verify end to end.
    #[test]
    fn a_duplicated_ring_is_admissible_and_uses_the_first_occurrence() {
        let par = RIVER_TOY;
        let scheme = RiVeR::new(par);
        let pp = scheme.setup(&[11u8; 32]);
        let keys = ring_of(&scheme, &pp, par.N);
        let ring: Vec<PolyVec> = keys.iter().map(|(_, pk)| pk.clone()).collect();

        let mut dup = ring.clone();
        dup[2] = ring[0].clone();
        assert_eq!(scheme.validate_ring(&dup).unwrap(), dup);

        // `min`, exhaustively over every arrangement of one key at two
        // positions: "first occurrence" and "any occurrence" agree on most
        // inputs and differ only where a later duplicate could be picked.
        for i in 0..par.N {
            for j in 0..par.N {
                if i == j {
                    continue;
                }
                let mut r = ring.clone();
                r[i] = ring[i].clone();
                r[j] = ring[i].clone();
                let validated = scheme.validate_ring(&r).unwrap();
                assert_eq!(
                    scheme.ring_index(&validated, &ring[i]),
                    Some(i.min(j)),
                    "i = {i}, j = {j}"
                );
            }
        }

        // ...and it proves and verifies.
        let (sk, pk) = &keys[0];
        let (v, pi, _) = scheme
            .eval_deterministic(&pp, pk, sk, &dup, b"m", &[7u8; 32])
            .unwrap();
        assert!(scheme.verify(&pp, &dup, b"m", &v, &pi));
    }

    /// The cost of admitting duplicates, stated as a test rather than
    /// prose: `k` copies of one key leave `N - k + 1` distinct identities.
    #[test]
    fn a_duplicated_key_shrinks_the_anonymity_set() {
        let par = RIVER_TOY;
        let scheme = RiVeR::new(par);
        let pp = scheme.setup(&[12u8; 32]);
        let keys = ring_of(&scheme, &pp, par.N);
        let ring: Vec<PolyVec> = keys.iter().map(|(_, pk)| pk.clone()).collect();

        let distinct = |r: &[PolyVec]| -> usize {
            let mut e: Vec<Vec<u8>> = r
                .iter()
                .map(|pk| scheme.codec.pk_encode(pk).unwrap())
                .collect();
            e.sort_unstable();
            e.dedup();
            e.len()
        };
        assert_eq!(distinct(&ring), par.N);

        let mut dup = ring.clone();
        dup[2] = ring[0].clone();
        dup[3] = ring[0].clone();
        scheme.validate_ring(&dup).unwrap();
        assert_eq!(distinct(&dup), par.N - 2);
    }
    /// The LANES backend runs end to end at the paper's own parameters,
    /// under the experimental name.
    ///
    /// This is the coverage the gate would otherwise cost: `BackendKind::Lanes`
    /// is reserved while `LanesExperimental` runs the same implementation, and a
    /// proof layer with no test at all is how an unconstrained message-block
    /// padding survived to be found by inspection.
    ///
    /// The proof is byte-identical to `river-py`'s at the same seeds; the
    /// vector case `("RiVeR-TOY", "lanes-experimental")` is what pins that
    /// across the two, and this is the same run without the file.
    #[test]
    fn the_experimental_lanes_backend_runs_end_to_end() {
        use crate::params::RIVER_TOY;
        let scheme = RiVeR::new_with(RIVER_TOY, BackendKind::LanesExperimental);
        assert_eq!(scheme.backend(), BackendKind::LanesExperimental);
        assert_eq!(scheme.backend().name(), "lanes-experimental");

        let seed: Vec<u8> = (0u8..32).collect();
        let pp = scheme.setup(&seed);
        let keys: Vec<_> = (0..RIVER_TOY.N)
            .map(|i| {
                let mut s = vec![0u8; 32];
                s[0] = i as u8;
                scheme.keygen(&pp, &s).expect("keygen")
            })
            .collect();
        let ring: Vec<_> = keys.iter().map(|(_, pk)| pk.clone()).collect();
        let (sk, pk) = &keys[1];
        let (v, pi, _) = scheme
            .eval_deterministic(&pp, pk, sk, &ring, b"RiVeR test vector", &[0xAAu8; 32])
            .expect("eval");
        assert!(scheme.verify(&pp, &ring, b"RiVeR test vector", &v, &pi));

        // ...and it round-trips through the wire, with the exact half
        // genuinely a LANES proof rather than an opening.
        let blob = scheme.proof_encode(&pp, &pi).expect("encode");
        let back = scheme.proof_decode(&pp, &blob).expect("decode");
        assert!(scheme.verify(&pp, &ring, b"RiVeR test vector", &v, &back));
        assert!(matches!(pi.sigma_ex, ExactSigma::Lanes(_)));

        // The production name stays shut, and says why.
        assert!(RiVeR::try_new_with(RIVER_TOY, BackendKind::Lanes).is_none());
    }

    /// A tampered proof is rejected under the LANES backend too.
    #[test]
    fn the_experimental_lanes_backend_rejects_tampering() {
        use crate::params::RIVER_TOY;
        let scheme = RiVeR::new_with(RIVER_TOY, BackendKind::LanesExperimental);
        let pp = scheme.setup(&[3u8; 32]);
        let keys: Vec<_> = (0..RIVER_TOY.N)
            .map(|i| {
                let mut s = vec![0u8; 32];
                s[0] = 0x40 | i as u8;
                scheme.keygen(&pp, &s).expect("keygen")
            })
            .collect();
        let ring: Vec<_> = keys.iter().map(|(_, pk)| pk.clone()).collect();
        let (sk, pk) = &keys[0];
        let (v, pi, _) = scheme
            .eval_deterministic(&pp, pk, sk, &ring, b"m", &[5u8; 32])
            .expect("eval");
        assert!(scheme.verify(&pp, &ring, b"m", &v, &pi));

        // wrong message, wrong value, wrong ring
        assert!(!scheme.verify(&pp, &ring, b"m'", &v, &pi));
        let mut other = v.clone();
        other[0] = (other[0] + 1) % RIVER_TOY.p;
        assert!(!scheme.verify(&pp, &ring, b"m", &other, &pi));
        let mut swapped = ring.clone();
        swapped.swap(0, 1);
        assert!(!scheme.verify(&pp, &swapped, b"m", &v, &pi));

        // a flipped byte anywhere in the serialized proof
        let blob = scheme.proof_encode(&pp, &pi).expect("encode");
        for k in [0usize, blob.len() / 2, blob.len() - 1] {
            let mut bad = blob.clone();
            bad[k] ^= 1;
            let ok = scheme
                .proof_decode(&pp, &bad)
                .is_some_and(|p| scheme.verify(&pp, &ring, b"m", &v, &p));
            assert!(!ok, "byte {k}");
        }
    }
}
