//! The relaxed one-out-of-many proof (Figure 7) — port of
//! `river-py/oom.py`.
//!
//! `OM` proves knowledge of a short opening of *one* vector `c_{j*}` in a
//! public list `(c_i)_{i<N}`, without revealing `j*`:
//!
//! ```text
//! c_{j*} = Com^0_{ck_r}(r) = ck_r · r   (mod q)
//! ```
//!
//! The selector machinery lives in `R_qhat`; the opening lives in `R_q`.
//! The challenge `x` is a single integer polynomial canonically embedded
//! into both.
//!
//! ## Representation
//!
//! Selector-side quantities (`a`, `b`, `c_sel`, `d`, `f`, `g`, `z^bin`,
//! `x`) are *integer* polynomials whose coefficients stay far below
//! `qhat/2`, so the code works in `R_qhat` and recovers exact integers by
//! centring.  That is what makes `g_i = f_i(x - f_i)` a genuine integer
//! product and the bound checks `||g||_inf <= B_g` meaningful.  Every such
//! quantity is `Vec<i64>` here and `Vec<u64>` only while it is a residue.
//!
//! The commitments `A` and `B` are the **high bits** of a `G'` product,
//! taken on the canonical representative in `[0, qhat)`.  The paper never
//! defined `[[·]]_K` and the paper defines it on the
//! *centred* representative; this follows the reference, which is the
//! Dilithium Power2Round convention on the canonical one.
//!
//! ## What is wire-visible
//!
//! All of it.  Every rejection-sampler call consumes XOF bytes, so an
//! extra or missing draw shifts the whole stream; the order of the
//! `Rej_1`/`Rej_2` calls, the order of the bound checks that can return
//! `⊥` before them, and the exact challenge preimage are as much a part
//! of the format as the codec is.

use crate::aux_ntt::CrtNttMat;
use crate::codec::{pack_signed, pack_unsigned, width_for_bound};
use crate::params::Rat;
use crate::params::RiVeRParams;
use crate::ring::{mod_pm, power2round, Poly, PolyMat, PolyVec, Ring};
use crate::sample::{
    challenge_from_hash, gaussian_vec, rational_sigma, rej1, rej2, sam_mat, uniform_beta_vec, Part,
    Xof,
};

/// The public statement `(ck_{r,m}, (c_i)_{i<N})`, kept structural.
///
/// `ck_{r,m} = [A | -I_n | 0 ; h_m^T | 0 | -1]` is never materialised as a
/// dense matrix: [`OomStatement::apply_ck`] uses the block structure
/// directly, which turns an `(n+1) × (ell+n+1)` product into one `n × ell`
/// product plus an inner product.  `c_i = (q_0 t_i, q_0 v)` is likewise
/// stored as `(t_i, v)`.
///
/// **Opaque, and constructed only through [`OomStatement::new`].**  The
/// fields were public and unvalidated, while `apply_ck` and `combine_c`
/// slice and index them directly — so a short `a_mat`, a ragged
/// `ring_pks`, or a statement built against a *different* profile from the
/// [`Oom`] verifying it, panicked inside verification.  Two of these come
/// from a peer: `Verify(pp, ring_pks, m, v, pi)` takes the ring and the
/// value from whoever sent the proof.  The reference survives this because
/// Python indexes are checked and `RiVeR.verify` has an outermost
/// `MALFORMED` boundary.  Here the constructor is that boundary.
/// [`crate::river::RiVeR::verify`] validates the ring and the value before
/// building a statement at all, so in the scheme this is the second line
/// rather than the only one — but this type is public, and a caller who
/// uses the OOM layer directly gets the same guarantee.
pub struct OomStatement<'a> {
    /// The `Oom` this was built against.
    ///
    /// Held for identity, not for access: a lifetime says the borrow is
    /// live, not that it is the *same object*, so `OomStatement::new(&n8,
    /// ..)` followed by `toy.verify(&statement, ..)` type-checked and then
    /// sliced an 11-element `z` with `ell = 56`.  [`Oom::verify`] compares
    /// this by pointer before touching anything.
    owner: &'a Oom,
    par: &'a RiVeRParams,
    rq: &'a Ring,
    /// `A`, `n × ell` over `R_q`.
    a_mat: &'a [PolyVec],
    /// `h_m`, `ell` elements of `R_q`.
    h_m: &'a [Poly],
    /// `N` public keys, each `n` elements of `R_p`.
    ring_pks: &'a [PolyVec],
    /// `v` in `R_p`.
    value: &'a [u64],
}

impl<'a> OomStatement<'a> {
    /// Every dimension, modulus and residue checked, against the profile
    /// the `oom` itself carries.
    ///
    /// `None` rather than a panic, and `None` rather than a silent
    /// truncation: an over-long `ring_pks` is as wrong as a short one, and
    /// a non-canonical residue centres to a different integer than it
    /// encodes.
    pub fn new(
        oom: &'a Oom,
        a_mat: &'a [PolyVec],
        h_m: &'a [Poly],
        ring_pks: &'a [PolyVec],
        value: &'a [u64],
    ) -> Option<Self> {
        let par = &oom.par;
        let (d, q, p) = (par.d, par.q(), par.p);
        let shaped = |v: &[Poly], rows: usize, modulus: u64| {
            v.len() == rows
                && v.iter()
                    .all(|r| r.len() == d && r.iter().all(|&c| c < modulus))
        };
        if a_mat.len() != par.n || !a_mat.iter().all(|row| shaped(row, par.ell, q)) {
            return None;
        }
        if !shaped(h_m, par.ell, q) {
            return None;
        }
        if ring_pks.len() != par.N || !ring_pks.iter().all(|pk| shaped(pk, par.n, p)) {
            return None;
        }
        if value.len() != d || value.iter().any(|&c| c >= p) {
            return None;
        }
        Some(Self {
            owner: oom,
            par,
            rq: &oom.rq,
            a_mat,
            h_m,
            ring_pks,
            value,
        })
    }

    pub fn par(&self) -> &RiVeRParams {
        self.par
    }

    /// Whether this statement was built against `oom`.
    ///
    /// Pointer identity, not profile equality: two `Oom`s at the same
    /// profile still have different `G'` matrices unless they share a seed,
    /// and a statement carries the ring the *other* one's `apply_ck` would
    /// use.  Comparing profiles would let that through.
    pub fn belongs_to(&self, oom: &Oom) -> bool {
        std::ptr::eq(self.owner, oom)
    }

    /// `ck_{r,m} · y` for `y = (y_s, y_key, y_eval)`.
    pub fn apply_ck(&self, y: &[Poly]) -> PolyVec {
        let (par, r) = (self.par, self.rq);
        let y_s = &y[..par.ell];
        let y_key = &y[par.ell..par.ell + par.n];
        let y_eval = &y[par.ell + par.n];
        let mut out: PolyVec = (0..par.n)
            .map(|i| r.sub(&r.inner(&self.a_mat[i], y_s), &y_key[i]))
            .collect();
        out.push(r.sub(&r.inner(self.h_m, y_s), y_eval));
        out
    }

    /// The derived vector `(q_0 t + delta_{e,n}, q_0 v + delta_e)` for a
    /// **supplied** key `t`, in `R_q^{n+1}`.
    ///
    /// Takes the key rather than its index, so a caller holding a secret
    /// index can select the key obliviously and never index this list
    /// with it.  `Eval` does exactly that: `ring[j*]` is chosen by a
    /// masked pass over the whole ring, and the opening invariant is
    /// checked against *that* key.
    ///
    /// **Precondition:** `t` is `par.n` polynomials of `par.d`
    /// coefficients.  `pub(crate)` rather than `pub` for that reason —
    /// the only caller outside this module is `river::Eval`, which builds
    /// `t` at exactly that shape, and widening the public surface with a
    /// function that indexes an unvalidated slice would add a panic path
    /// for no one's benefit.  [`Self::c_i`] is the public form and takes
    /// an index into a list `OomStatement::new` has already validated.
    pub(crate) fn c_for_key(&self, t: &[Poly]) -> PolyVec {
        let (par, r) = (self.par, self.rq);
        debug_assert_eq!(t.len(), par.n, "c_for_key: wrong key shape");
        let delta = vec![par.B_e(); par.d];
        let mut out: PolyVec = (0..par.n)
            .map(|j| r.add(&r.scale(par.q0 as i64, &r.reduce(&t[j])), &delta))
            .collect();
        out.push(r.add(&r.scale(par.q0 as i64, &r.reduce(self.value)), &delta));
        out
    }

    /// The `i`-th derived vector.
    ///
    /// `i` is a *public* index here — the verifier's `combine_c` runs over
    /// all of them, and the tests name specific ones.  A caller with a
    /// secret index wants `c_for_key` instead; indexing this list
    /// with `j*` publishes it through the cache line it touches.
    pub fn c_i(&self, i: usize) -> PolyVec {
        self.c_for_key(&self.ring_pks[i])
    }

    /// `sum_i coeff_i · c_i` in `R_q^{n+1}`.
    ///
    /// Uses the shared structure of the `c_i`: the top block is
    /// `q_0 · sum_i coeff_i t_i + delta_e · sum_i coeff_i`; the bottom
    /// entry has the same offset term.
    ///
    /// **No all-zero-`coeff_i` skip**, though the reference has one.  It
    /// branches on secret data — on the commit side `coeffs` is the mask
    /// `a` — and while an all-zero `a_i` is a measure-zero accident, the
    /// argument that saved it (that `a_i` is independent of `j*`) does not
    /// hold up: `f_i = a_i + x b_i` is published, so an observer who learns
    /// something about `a_i`'s zeros can test each `i` against `f_i` and
    /// `f_i - x`.  Dropping it costs one ring multiply against a
    /// polynomial that is essentially never all-zero, which is nothing.
    pub fn combine_c(&self, coeffs: &[Poly]) -> PolyVec {
        let (par, r) = (self.par, self.rq);
        let mut out = PolyVec::with_capacity(par.n + 1);
        for j in 0..par.n {
            let mut acc = r.zero();
            for (i, coeff) in coeffs.iter().enumerate().take(par.N) {
                let t_ij = r.reduce(&self.ring_pks[i][j]);
                acc = r.add(&acc, &r.mul(coeff, &t_ij));
            }
            out.push(r.scale(par.q0 as i64, &acc));
        }
        let mut total = r.zero();
        for coeff in coeffs.iter().take(par.N) {
            total = r.add(&total, coeff);
        }
        let delta = vec![par.B_e(); par.d];
        let delta_term = r.mul(&total, &delta);
        for row in &mut out {
            *row = r.add(row, &delta_term);
        }
        out.push(r.add(
            &r.scale(par.q0 as i64, &r.mul(&total, &r.reduce(self.value))),
            &delta_term,
        ));
        out
    }
}

/// `t_OOM = (A, B, E)`: what `OM.Com` publishes.
///
/// `A` and `B` are high-bit vectors, non-negative because the
/// decomposition is taken on the canonical representative; `E` is a
/// genuine `R_q` vector.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OomCommitment {
    pub a_hi: Vec<Vec<i64>>,
    pub b_hi: Vec<Vec<i64>>,
    pub e: PolyVec,
}

/// `st_OOM`: everything `OM.Prove` needs that `t_OOM` does not carry.
///
/// One of these per attempt, not one per proof: an aborted attempt is
/// restarted from a fresh [`Oom::com`], which draws new masks and so gives
/// a different `A`, `B`, `E` and challenge.  Pairing it with the
/// [`OomCommitment`] from the *same* `com` call is the caller's job — they
/// are two halves of one draw.
pub struct OomState {
    /// `a_i`, centred integers.
    pub a: Vec<Vec<i64>>,
    /// `b_i = delta_{j*,i}`, centred integers.
    pub b: Vec<Vec<i64>>,
    pub c_sel: Vec<Vec<i64>>,
    pub d_vec: Vec<Vec<i64>>,
    pub r_a: Vec<Vec<i64>>,
    pub r_b: Vec<Vec<i64>>,
    /// `y_OM`, residues mod `q`.
    pub y_om: PolyVec,
    pub j_star: usize,
}

/// `pi_OOM = (B, x, f_1, z_b, z)`.
///
/// Field order and types match [`crate::codec::RiVeRCodec::oom_encode`],
/// which is the only thing that turns this into bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OomProof {
    pub b_hi: Vec<Vec<i64>>,
    pub x: Vec<i64>,
    pub f1: Vec<Vec<i64>>,
    pub zb: Vec<Vec<i64>>,
    /// `z = (z_s, z_m)`, whole.  It is split only on the wire — see
    /// [`crate::codec::RiVeRCodec::oom_field_values`] — because the
    /// protocol and the verifier's Euclidean check operate on all of it.
    pub z: PolyVec,
}

/// `OM.Setup / Com / Prove / Ver` for one parameter profile.
pub struct Oom {
    /// Private: `G'` and both rings are derived from it and cached, so a
    /// profile changed after construction leaves them describing a
    /// different scheme.  `pp.oom.par.K_b` was reachable and moving it made
    /// `reconstruct_u`'s shift leave range.
    par: RiVeRParams,
    seed: Vec<u8>,
    rq: Ring,
    rqhat: Ring,
    /// `G' ← SamMat(rho, qhat, n_hat, k_hat + 2N, "G'")`.
    gp: PolyMat,
    /// `G'` pre-transformed.  The matrix is fixed for the lifetime of the
    /// object and every attempt multiplies by it twice, so transforming
    /// per call — which is what [`Ring::mat_vec`] does — pays the setup
    /// on every product.  `None` when the backend declines the width, and
    /// then `mat_vec`'s schoolbook path is used, which agrees exactly.
    gp_ntt: Option<CrtNttMat>,
    sigma_a: (u64, u64),
    sigma_b: (u64, u64),
    /// The paper splits the outer response.  `z_s` answers
    /// `r_0 = s` at width `sigma_s = phi_s B_s`; `z_m` answers
    /// `r_1 = (e_key, e_eval)` at `sigma_m = phi_m eta_m`.  They are two
    /// different Gaussians drawn from one XOF stream, in this order.
    sigma_s: (u64, u64),
    sigma_m: (u64, u64),
    /// `[[.]]_K` on the *centred* representative leaves high bits in
    /// roughly `[-qhat/2^{K+1}, +qhat/2^{K+1}]`; these are the packing
    /// bounds, and they are signed.  See [`Oom::high_low`].
    hi_bound_a: i64,
    hi_bound_b: i64,
}

impl Oom {
    pub fn new(par: RiVeRParams, seed: &[u8]) -> Self {
        let rq = Ring::new(par.q(), par.d);
        let rqhat = Ring::with_backend(par.q_hat, par.d, par.gprime_cols());
        let gp = sam_mat(seed, par.q_hat, par.n_hat, par.gprime_cols(), par.d, "G'");
        let gp_ntt = rqhat.mat_to_ntt(&gp);
        // Widths come from the frozen manifest when the profile has one,
        // and are derived otherwise.  Both agree — `manifest`'s own tests
        // re-derive every entry from these same parameters — but only the
        // table is independent of the order an `f64` chain is evaluated
        // in, and a width off by one unit of `2^-20` moves every mask.
        let man = crate::manifest::for_params(&par);
        let pin = |spec: Option<crate::manifest::GaussianSpec>, sigma: f64| match spec {
            Some(g) => (g.sigma_num, g.sigma_den),
            None => rational_sigma(sigma),
        };
        Self {
            seed: seed.to_vec(),
            gp,
            gp_ntt,
            sigma_a: pin(man.map(|m| m.f1), par.sigma_a()),
            sigma_b: pin(man.map(|m| m.zb), par.sigma_b()),
            sigma_s: pin(man.map(|m| m.zs), par.sigma_s()),
            sigma_m: pin(man.map(|m| m.zm), par.sigma_m()),
            hi_bound_a: crate::ring::high_bits_bound(par.q_hat, par.K_a),
            hi_bound_b: crate::ring::high_bits_bound(par.q_hat, par.K_b),
            rq,
            rqhat,
            par,
        }
    }

    pub fn par(&self) -> &RiVeRParams {
        &self.par
    }

    pub fn rq(&self) -> &Ring {
        &self.rq
    }

    pub fn rqhat(&self) -> &Ring {
        &self.rqhat
    }

    // ---- helpers ---------------------------------------------------------

    /// Embed an integer polynomial into `R_qhat`.
    fn lift_hat(&self, p: &[i64]) -> Poly {
        self.rqhat.from_centered(p)
    }

    /// Embed an integer polynomial into `R_q`.
    fn lift_q(&self, p: &[i64]) -> Poly {
        self.rq.from_centered(p)
    }

    /// `G' · (block_0 || block_1 || block_2)` in `R_qhat`.
    fn gprime(&self, blocks: [&[Vec<i64>]; 3]) -> PolyVec {
        let mut vec: PolyVec = Vec::with_capacity(self.par.gprime_cols());
        for group in blocks {
            for p in group {
                vec.push(self.lift_hat(p));
            }
        }
        debug_assert_eq!(vec.len(), self.par.gprime_cols());
        match self.gp_ntt.as_ref() {
            Some(m) => self
                .rqhat
                .mat_vec_ntt(m, &vec)
                .unwrap_or_else(|| self.rqhat.mat_vec(&self.gp, &vec)),
            None => self.rqhat.mat_vec(&self.gp, &vec),
        }
    }

    /// `[[.]]_K` and `. mod^pm 2^K` of each coefficient.
    ///
    /// Taken on the **centred** representative
    /// `\bar a in (-qhat/2, qhat/2]`, which is what the paper's
    /// preliminaries define:
    ///
    /// ```text
    /// a mod^pm 2^K := \bar a - 2^K floor((\bar a + 2^{K-1} - 1)/2^K)
    /// [[a]]_K      := (\bar a - (a mod^pm 2^K)) / 2^K
    /// ```
    ///
    /// Both ranges are asymmetric — the low part lands in
    /// `(-2^{K-1}, 2^{K-1}]`, closed at the top — and [`power2round`]
    /// already implements that tie, so the only thing that moves here is
    /// which representative goes in.
    ///
    /// That closes it.  Through the paper this was taken on the
    /// canonical `[0, qhat)` representative, because the operator was
    /// undefined and the canonical reading let the codec encode `B`
    /// unsigned; the paper then defined it the other way and this
    /// code deliberately did not follow, since aligning moves protocol
    /// bytes.  The definition is now unambiguous and stated in the
    /// preliminaries rather than mid-appendix, so the code follows it:
    /// about half the high parts are negative, the transmitted `B` field
    /// is signed, and every vector moves.
    fn high_low(&self, v: &[Poly], k: u32) -> (Vec<Vec<i64>>, Vec<Vec<i64>>) {
        let mut highs = Vec::with_capacity(v.len());
        let mut lows = Vec::with_capacity(v.len());
        for poly in v {
            let centred = self.rqhat.centered(poly);
            let mut hi = Vec::with_capacity(centred.len());
            let mut lo = Vec::with_capacity(centred.len());
            for &c in &centred {
                let (h, l) = power2round(c as i128, k);
                hi.push(h as i64);
                lo.push(l as i64);
            }
            highs.push(hi);
            lows.push(lo);
        }
        (highs, lows)
    }

    // ---- OM.Com ----------------------------------------------------------

    /// `(t_OOM, st_OOM) ← OM.Com(pp, m, ck_r, (c_i), (j*, r))`.
    pub fn com(
        &self,
        statement: &OomStatement<'_>,
        j_star: usize,
        xof: &mut Xof,
    ) -> (OomCommitment, OomState) {
        let par = &self.par;
        let (n_ring, d) = (par.N, par.d);

        // b = (delta_{j*,0}, ..., delta_{j*,N-1})
        //
        // Written across *every* row rather than into `b[j_star]`.  `j*`
        // is the one thing this whole proof system exists to hide, and a
        // secret-indexed store publishes it through the cache line it
        // touches.  `select` turns the index into an all-ones mask and
        // the write into fixed work over the full ring; the comparison
        // `i == j_star` compiles to a `setcc`, not a jump, and its
        // *result* never chooses an address.
        let mut b = vec![vec![0i64; d]; n_ring];
        for (i, row) in b.iter_mut().enumerate() {
            row[0] = eq_mask(i, j_star) & 1;
        }

        // a_1..a_{N-1} <- D_{phi_a B_a};  a_0 = -sum_{i>=1} a_i
        let tail = gaussian_vec(
            xof,
            self.sigma_a.0,
            d,
            n_ring - 1,
            par.q_hat,
            self.sigma_a.1,
        );
        let mut a: Vec<Vec<i64>> = Vec::with_capacity(n_ring);
        a.push(vec![0i64; d]); // placeholder for a_0
        for row in &tail {
            a.push(self.rqhat.centered(row));
        }
        let mut head = vec![0i64; d];
        for row in a.iter().skip(1) {
            for k in 0..d {
                head[k] -= row[k];
            }
        }
        a[0] = head;

        // d = (-a_0^2, ..., -a_{N-1}^2)  and  c_sel = a o (1 - 2b)
        let mut d_vec = Vec::with_capacity(n_ring);
        let mut c_sel = Vec::with_capacity(n_ring);
        for (i, ai) in a.iter().enumerate() {
            let lifted = self.lift_hat(ai);
            let sq = self.rqhat.centered(&self.rqhat.mul(&lifted, &lifted));
            d_vec.push(sq.into_iter().map(|c| -c).collect::<Vec<i64>>());
            // c_sel_i = a_i (1 - 2 delta_{j*,i}): `+a_i` off the signer
            // index and `-a_i` on it.  Masked, for the reason `b` is.
            let sign = 1 - 2 * (eq_mask(i, j_star) & 1);
            c_sel.push(ai.iter().map(|&c| sign * c).collect::<Vec<i64>>());
        }

        // r_b <- U_beta^{k_hat},  r_a <- D_{phi_b B}^{k_hat}
        //
        // REPAIR.  The figure samples `r_a <- D_B`, while its
        // `Rej_2` call uses `(phi_b, B)` and the communication formula
        // charges `h(phi_b B)`.  A rejection sampler is only correct when
        // the mask width equals the sigma in its acceptance test, and
        // `phi_b B` is also the reading the paper's own size accounting
        // uses — its `k_hat d h(phi_b B)` term is what reproduces the
        // reported `|pi_OOM|`.
        let r_b: Vec<Vec<i64>> = uniform_beta_vec(xof, par.beta, d, par.k_hat, par.q_hat)
            .iter()
            .map(|p| self.rqhat.centered(p))
            .collect();
        let r_a: Vec<Vec<i64>> =
            gaussian_vec(xof, self.sigma_b.0, d, par.k_hat, par.q_hat, self.sigma_b.1)
                .iter()
                .map(|p| self.rqhat.centered(p))
                .collect();

        let u_b = self.gprime([&r_b, &b, &c_sel]);
        let (b_hi, _e_b) = self.high_low(&u_b, par.K_b);
        let u_a = self.gprime([&r_a, &a, &d_vec]);
        let (a_hi, _) = self.high_low(&u_a, par.K_a);

        // y_s <- D_{sigma_s}^{ell},  (y_key, y_eval) <- D_{sigma_m}^{n+1},
        // y_OM <- (y_s, y_key, y_eval).  Two draws, in the figure's order:
        // one XOF stream, and swapping them changes every mask.
        let mut y_om = gaussian_vec(xof, self.sigma_s.0, d, par.s_dim(), par.q(), self.sigma_s.1);
        y_om.extend(gaussian_vec(
            xof,
            self.sigma_m.0,
            d,
            par.m_dim(),
            par.q(),
            self.sigma_m.1,
        ));

        // E = ck_r y_OM - sum_i a_i c_i   (mod q)
        let a_q: PolyVec = a.iter().map(|ai| self.lift_q(ai)).collect();
        let lhs = statement.apply_ck(&y_om);
        let rhs = statement.combine_c(&a_q);
        let e: PolyVec = lhs
            .iter()
            .zip(rhs.iter())
            .map(|(l, r)| self.rq.sub(l, r))
            .collect();

        (
            OomCommitment { a_hi, b_hi, e },
            OomState {
                a,
                b,
                c_sel,
                d_vec,
                r_a,
                r_b,
                y_om,
                j_star,
            },
        )
    }

    // ---- Fiat-Shamir -----------------------------------------------------

    /// `x ← H(m, ck_r, (c_i), A, B, E; rho')`.
    ///
    /// Returns `None` only if a commitment is out of the range its own
    /// decomposition guarantees, which cannot happen for a `t_OOM` this
    /// module produced and *can* happen for one a peer sent.
    pub fn challenge(
        &self,
        commitment: &OomCommitment,
        ck_digest: &[u8],
        rho_digest: &[u8],
    ) -> Option<Vec<u64>> {
        let par = &self.par;
        let flat_a: Vec<i64> = commitment.a_hi.iter().flatten().copied().collect();
        let flat_b: Vec<i64> = commitment.b_hi.iter().flatten().copied().collect();
        let packed_a = pack_signed(
            &flat_a,
            width_for_bound(self.hi_bound_a).ok()?,
            self.hi_bound_a,
        )
        .ok()?;
        let packed_b = pack_signed(
            &flat_b,
            width_for_bound(self.hi_bound_b).ok()?,
            self.hi_bound_b,
        )
        .ok()?;
        let flat_e: Vec<u64> = commitment
            .e
            .iter()
            .flat_map(|p| self.rq.reduce(p))
            .collect();
        // `(q.bit_length() + 7) // 8`, as the reference packs it.
        let w_q = (64 - par.q().leading_zeros()).div_ceil(8) as usize;
        let packed_e = pack_unsigned(&flat_e, w_q).ok()?;
        Some(challenge_from_hash(
            par.d,
            par.w,
            par.gamma,
            par.q_hat,
            &[
                Part::Bytes(&self.seed),
                Part::Bytes(ck_digest),
                Part::Bytes(&packed_a),
                Part::Bytes(&packed_b),
                Part::Bytes(&packed_e),
                Part::Bytes(rho_digest),
            ],
        ))
    }

    // ---- OM.Prove --------------------------------------------------------

    /// One `OM.Prove` attempt.  `None` is the paper's `⊥`: the caller
    /// restarts from a fresh [`Oom::com`].
    ///
    /// Takes no statement: the reference's `prove` passes one and never
    /// reads it, because the statement reaches the challenge through
    /// `ck_digest` and through `E` inside `commitment`.  Same for `com`'s
    /// `r`, which the reference stores in `st_OOM` and then takes again as
    /// an argument here.
    pub fn prove(
        &self,
        r: &[Poly],
        commitment: &OomCommitment,
        state: &OomState,
        ck_digest: &[u8],
        rho_digest: &[u8],
        xof: &mut Xof,
    ) -> Option<OomProof> {
        let par = &self.par;
        let (n_ring, d) = (par.N, par.d);
        let j_star = state.j_star;

        let x_hat = self.challenge(commitment, ck_digest, rho_digest)?;
        let x = self.rqhat.centered(&x_hat); // integer challenge polynomial

        // f_i = x b_i + a_i.  `b` is a unit vector, so the whole of `x`
        // lands in exactly one row — masked in, rather than branched to,
        // for the reason `com` masks `b` itself.  Every row does the same
        // work whatever `j*` is.
        let f: Vec<Vec<i64>> = (0..n_ring)
            .map(|i| {
                let m = eq_mask(i, j_star);
                (0..d).map(|k| state.a[i][k] + (x[k] & m)).collect()
            })
            .collect();
        let f1 = &f[1..];

        // Rej_1(f_1, x * (delta_{j*,1}, ..., delta_{j*,N-1}), phi_a, B_a)
        //
        // `shift` is `x` at row `j* - 1` and zero elsewhere, and row 0 of
        // `f` is not in `f_1` at all — so `j* == 0` is the case where no
        // row carries it.  Written as a masked pass over every row, which
        // handles that without the `if j_star >= 1` that used to guard a
        // secret-indexed store.
        let mut shift = vec![vec![0i64; d]; n_ring - 1];
        for (i, row) in shift.iter_mut().enumerate() {
            let m = eq_mask(i + 1, j_star);
            for k in 0..d {
                row[k] = x[k] & m;
            }
        }
        if rej1(
            xof,
            &flatten(f1),
            &flatten(&shift),
            par.phi_a,
            self.sigma_a.0,
            self.sigma_a.1,
            RiVeRParams::REJ_TAU,
        ) {
            return None;
        }

        // z_b = r_a + x r_b,   z_s = y_s + x r_0,   z_m = y_m + x r_1
        let x_hat_lift = self.lift_hat(&x);
        let x_r_b: Vec<Vec<i64>> = state
            .r_b
            .iter()
            .map(|rb| {
                self.rqhat
                    .centered(&self.rqhat.mul(&x_hat_lift, &self.lift_hat(rb)))
            })
            .collect();
        let zb: Vec<Vec<i64>> = (0..par.k_hat)
            .map(|i| (0..d).map(|k| state.r_a[i][k] + x_r_b[i][k]).collect())
            .collect();

        let x_q = self.lift_q(&x);
        let x_r: PolyVec = r.iter().map(|ri| self.rq.mul(&x_q, ri)).collect();
        let z: PolyVec = (0..par.r_dim())
            .map(|i| self.rq.add(&state.y_om[i], &x_r[i]))
            .collect();

        let z_c = self.rq.vec_centered(&z);
        let x_r_c = self.rq.vec_centered(&x_r);
        let s_end = par.s_dim();
        let (z_s_c, z_m_c) = z_c.split_at(s_end);
        let (x_r0_c, x_r1_c) = x_r_c.split_at(s_end);

        // The figure's disjunction, left to right and short-circuiting, so
        // the XOF is consumed in exactly that order:
        //   Rej_1((z_s, z_key), x r_0, phi_s, B_s, tau_rej)
        //   Rej_1(z_eval,        x r_1, phi_m, eta_m, tau_rej)
        //   Rej_2(z_b,           x r_b, phi_b, B)
        if rej1(
            xof,
            &flatten(z_s_c),
            &flatten(x_r0_c),
            par.phi_s,
            self.sigma_s.0,
            self.sigma_s.1,
            RiVeRParams::REJ_TAU,
        ) {
            return None;
        }
        if rej1(
            xof,
            &flatten(z_m_c),
            &flatten(x_r1_c),
            par.phi_m,
            self.sigma_m.0,
            self.sigma_m.1,
            RiVeRParams::REJ_TAU,
        ) {
            return None;
        }
        if rej2(
            xof,
            &flatten(&zb),
            &flatten(&x_r_b),
            par.phi_b,
            self.sigma_b.0,
            self.sigma_b.1,
        ) {
            return None;
        }

        // DEFENSIVE: prover and verifier checks must not differ.
        // The prover applies **every** bound `OOM.Ver` applies, through
        // the same function — see `response_bounds_ok`.  The figure
        // comments its Euclidean check out and gives the commented form a
        // different, much smaller bound than the verifier's; a prover that
        // can return a proof its own verifier rejects is a correctness
        // bug.  It is not charged in the paper's attempt estimate and does
        // not need to be: measured at the toy profile it never fires on an
        // attempt the four infinity-norm checks let through, so it costs
        // exactly zero restarts.  That is a measurement, not a theorem.
        if !response_bounds_ok(par, f1, &zb, z_s_c, z_m_c, &z_c) {
            return None;
        }

        // g_i = f_i (x - f_i)
        let g = self.compute_g(&f, &x);
        if over(inf(&g[0]), par.B_g0()) {
            return None;
        }
        if over(inf_vec(&g[1..]), par.B_g1()) {
            return None;
        }

        // compression check on the low bits of the reconstructed A'
        let u_prime = self.reconstruct_u(&commitment.b_hi, &zb, &f, &g, &x);
        let t_cmp = par.T_cmp() as i128;
        let worst = u_prime
            .iter()
            .flat_map(|poly| poly.iter().map(|&c| mod_pm(c as i128, par.K_a).abs()))
            .max()
            .unwrap_or(0);
        if worst >= t_cmp {
            return None;
        }

        // Belt and braces: the margin above is what makes A' = A, but the
        // representative can also wrap at qhat.  That is a ~2^-20 event
        // per attempt, and the paper does not cover it either
        // — it defines the operators but still argues A' = A from the
        // decomposition margin alone, which holds over Z and not across
        // the wrap.  So detect it and restart.
        let (a_prime, _) = self.high_low(&u_prime, par.K_a);
        if a_prime != commitment.a_hi {
            return None;
        }

        Some(OomProof {
            b_hi: commitment.b_hi.clone(),
            x,
            f1: f1.to_vec(),
            zb,
            z,
        })
    }

    /// `g_i = f_i (x - f_i)`, centred.
    fn compute_g(&self, f: &[Vec<i64>], x: &[i64]) -> Vec<Vec<i64>> {
        let x_lift = self.lift_hat(x);
        f.iter()
            .map(|fi| {
                let lifted = self.lift_hat(fi);
                let diff = self.rqhat.sub(&x_lift, &lifted);
                self.rqhat.centered(&self.rqhat.mul(&lifted, &diff))
            })
            .collect()
    }

    /// `G' (z^bin || f || g) - x 2^{K_b} B  (mod qhat)`.
    fn reconstruct_u(
        &self,
        b_hi: &[Vec<i64>],
        zbin: &[Vec<i64>],
        f: &[Vec<i64>],
        g: &[Vec<i64>],
        x: &[i64],
    ) -> PolyVec {
        let par = &self.par;
        let prod = self.gprime([zbin, f, g]);
        let x_lift = self.lift_hat(x);
        (0..par.n_hat)
            .map(|i| {
                let shifted: Vec<i64> = b_hi[i]
                    .iter()
                    .map(|&c| ((c as i128) << par.K_b) as i64)
                    .collect();
                let term = self.rqhat.mul(&x_lift, &self.lift_hat(&shifted));
                self.rqhat.sub(&prod[i], &term)
            })
            .collect()
    }

    // ---- OM.Ver ----------------------------------------------------------

    /// `OM.Ver(pp, m, ck_r, (c_i), pi_OOM; rho')`.
    ///
    /// Total on `pi`: every shape and every bound is checked before the
    /// value is used, and a wrong one is `false`, never a panic.
    pub fn verify(
        &self,
        statement: &OomStatement<'_>,
        pi: &OomProof,
        ck_digest: &[u8],
        rho_digest: &[u8],
    ) -> bool {
        // Before anything: the statement has to be *this* `Oom`'s.  See
        // `OomStatement::belongs_to` — a lifetime does not encode identity,
        // and a mismatched pair panics rather than returning `false`.
        if !statement.belongs_to(self) {
            return false;
        }
        let par = &self.par;
        let (n_ring, d) = (par.N, par.d);

        if pi.f1.len() != n_ring - 1
            || pi.zb.len() != par.k_hat
            || pi.z.len() != par.r_dim()
            || pi.b_hi.len() != par.n_hat
            || pi.x.len() != d
        {
            return false;
        }
        if pi.f1.iter().any(|p| p.len() != d)
            || pi.zb.iter().any(|p| p.len() != d)
            || pi.z.iter().any(|p| p.len() != d)
            || pi.b_hi.iter().any(|p| p.len() != d)
        {
            return false;
        }
        // `z` is the one field carried as residues; a non-canonical one
        // would centre to a different integer than it encodes.
        if pi.z.iter().any(|p| p.iter().any(|&c| c >= par.q())) {
            return false;
        }
        // `x` before any arithmetic.  `f_0 = x - sum_{i>=1} f_i` is `i64`
        // subtraction, so an `x` of `i64::MIN` against an in-bound `f_1`
        // overflowed — a panic in debug and a wrap in release, either way a
        // contradiction of this function's totality.  The reference is
        // total here only because Python integers do not overflow.
        //
        // The check is challenge-space *membership*, not a magnitude bound:
        // `verify` ends by comparing `x` against a freshly derived
        // challenge, which is always in `C^d_{w,gamma}`, so no `x` outside
        // that set could ever be accepted anyway.  Rejecting it early
        // changes no decision and makes the arithmetic below total.
        if !in_challenge_space(&pi.x, par.w, par.gamma) {
            return false;
        }
        // `B`'s high parts, against the same domain the codec gives them:
        // `[[.]]_K` on the centred representative leaves them signed and
        // bounded by `high_bits_bound`, which is also the codec's field
        // bound and what `challenge` packs against.
        //
        // Written as an explicit range rather than `c.abs() > bound`.
        // `pi.b_hi` is peer-controlled `i64`, and `i64::MIN.abs()` panics
        // in debug and wraps *back to `i64::MIN`* in release — negative,
        // so the magnitude test would pass and the check would fail open.
        // This function is public and claims totality; `RiVeR::verify`
        // filtering encoded input first does not make that claim true.
        if pi.b_hi.iter().any(|row| {
            row.iter()
                .any(|&c| !(-self.hi_bound_b..=self.hi_bound_b).contains(&c))
        }) {
            return false;
        }

        // The figure's five response checks, through the same function
        // the prover calls — see `response_bounds_ok`.
        let z_c = self.rq.vec_centered(&pi.z);
        let (z_s_c, z_m_c) = z_c.split_at(par.s_dim());
        if !response_bounds_ok(par, &pi.f1, &pi.zb, z_s_c, z_m_c, &z_c) {
            return false;
        }

        // f_0 = x - sum_{i>=1} f_i
        let mut head = pi.x.clone();
        for poly in &pi.f1 {
            for k in 0..d {
                head[k] -= poly[k];
            }
        }
        let mut f = Vec::with_capacity(n_ring);
        f.push(head);
        f.extend(pi.f1.iter().cloned());

        // g_i = f_i (x - f_i), rechecked against the public thresholds
        let g = self.compute_g(&f, &pi.x);
        if over(inf(&g[0]), par.B_g0()) {
            return false;
        }
        if over(inf_vec(&g[1..]), par.B_g1()) {
            return false;
        }

        // A' and E'
        let u_prime = self.reconstruct_u(&pi.b_hi, &pi.zb, &f, &g, &pi.x);
        let (a_prime, _) = self.high_low(&u_prime, par.K_a);

        let f_q: PolyVec = f.iter().map(|fi| self.lift_q(fi)).collect();
        let lhs = statement.apply_ck(&pi.z);
        let rhs = statement.combine_c(&f_q);
        let e_prime: PolyVec = lhs
            .iter()
            .zip(rhs.iter())
            .map(|(l, r)| self.rq.sub(l, r))
            .collect();

        let recomputed = OomCommitment {
            a_hi: a_prime,
            b_hi: pi.b_hi.clone(),
            e: e_prime,
        };
        match self.challenge(&recomputed, ck_digest, rho_digest) {
            Some(expect) => self.rqhat.centered(&expect) == pi.x,
            None => false,
        }
    }
}

// ---- small helpers -------------------------------------------------------

/// `x in C^d_{w,gamma}`: exactly `w` nonzero coefficients, each in
/// `±[1, gamma]`.  Exactly what [`crate::sample::sample_challenge`]
/// produces, so it accepts every honest challenge and nothing else.
fn in_challenge_space(x: &[i64], w: usize, gamma: u64) -> bool {
    let g = gamma as i64;
    x.iter().filter(|&&c| c != 0).count() == w && x.iter().all(|&c| (-g..=g).contains(&c))
}

/// All-ones when `i == j`, zero otherwise — the mask that keeps `j*` out
/// of the addresses this module touches.
///
/// `j*` is the signer's index in the ring, which is the secret the OOM
/// proof hides; a `b[j_star] = 1` or an `if i == j_star` hands it to
/// anyone who can watch cache lines or branch history.  The comparison
/// itself is fine — it compiles to a flag, not a jump — as long as its
/// result selects a *value* rather than an address or a path.
///
/// Written on `usize` and widened, so no intermediate is signed and the
/// XOR-based form has no data-dependent shift.
#[inline(always)]
fn eq_mask(i: usize, j: usize) -> i64 {
    // `(i ^ j)` is zero exactly when they are equal; the wrapping
    // decrement then borrows through the whole word in that case only.
    let diff = (i ^ j) as u64;
    // 0 when diff == 0, else all ones
    let nonzero = ((diff | diff.wrapping_neg()) >> 63) & 1;
    (nonzero.wrapping_sub(1)) as i64
}

fn flatten(v: &[Vec<i64>]) -> Vec<i64> {
    v.iter().flatten().copied().collect()
}

fn inf(poly: &[i64]) -> i128 {
    poly.iter().map(|&c| (c as i128).abs()).max().unwrap_or(0)
}

fn inf_vec(v: &[Vec<i64>]) -> i128 {
    v.iter().map(|p| inf(p)).max().unwrap_or(0)
}

/// The figure's five response bounds, in its order.
///
/// **One function, two callers.**  `OM.Prove` applies it before returning
/// and `OM.Ver` applies it before accepting, which is what makes the
/// prover's defensive Euclidean check the *same* check the verifier runs
/// rather than a second one written to match.  It is also what makes the
/// boundary test discriminating: a test that drove `inf_over_sq` alone
/// would stay green if a call site were rewired to `over`, because it
/// would never touch a call site.
///
/// * `||f_1||_inf <= 6 phi_a B_a`
/// * `||z_b||_inf <= 6 phi_b B`
/// * `||z_s||_inf <= 6 sigma_s`
/// * `||(z_key, z_eval)||_inf <= 6 sigma_m`
/// * `||z||_2 <= 1.2 sqrt(sigma_s^2 d ell + sigma_m^2 d (n+1))`
///
/// The first four have the shape `K sqrt(M)`, so the exact form squares
/// the *norm* against a squared bound; the Euclidean one is already a
/// squared quantity on both sides.  Mixing the two conventions is not a
/// rounding difference — at `RiVeR-N8` it would compare `||f_1||_inf`
/// against `24576^2` — so they go through two differently named helpers.
///
/// `z_c` is the whole centred response and `z_s_c` / `z_m_c` are its two
/// blocks; the caller splits once and passes all three rather than having
/// this function re-derive a split it would then have to agree about.
fn response_bounds_ok(
    par: &RiVeRParams,
    f1: &[Vec<i64>],
    zb: &[Vec<i64>],
    z_s_c: &[Vec<i64>],
    z_m_c: &[Vec<i64>],
    z_c: &[Vec<i64>],
) -> bool {
    !inf_over_sq(inf_vec(f1), par.f1_inf_bound_sq())
        && !inf_over_sq(inf_vec(zb), par.zb_inf_bound_sq())
        && !inf_over_sq(inf_vec(z_s_c), par.zs_inf_bound_sq())
        && !inf_over_sq(inf_vec(z_m_c), par.zm_inf_bound_sq())
        && !over(l2_norm_sq(z_c), par.z_l2_bound_sq())
}

/// `sum c^2` over already-centred coefficients.
fn l2_norm_sq(v: &[Vec<i64>]) -> i128 {
    v.iter()
        .flat_map(|p| p.iter().map(|&c| (c as i128) * (c as i128)))
        .sum()
}

/// `value > bound`, exactly, for a non-negative integer against an exact
/// rational bound **that is already in the same units**.
///
/// Used for the two product-check thresholds `B_g0` and `B_g1`, which the
/// paper states directly rather than as `K sqrt(M)`.
///
/// The reference compares a Python `int` with a `Fraction`, which is exact
/// at any magnitude; the `value as f64 > bound` this replaced was not, and
/// these are wire-visible decisions.
fn over(value: i128, bound: Rat) -> bool {
    debug_assert!(value >= 0, "norms are non-negative");
    bound.exceeded_by(value.unsigned_abs())
}

/// `||.||_inf > sqrt(bound_sq)`, decided by squaring the norm.
///
/// The four infinity bounds have the shape `K sqrt(M)`, so the exact form
/// keeps `K^2 M` and squares the *other* side.  Squaring the bound and
/// forgetting to square the norm is not a rounding difference: at
/// `RiVeR-N8` it compares `||f_1||_inf` against `24576^2 = 603979776`
/// rather than against `24576`, which is four orders of magnitude of
/// slack, and it made both this module's verifier and the prover's own
/// restart condition wrong.  The reference squares
/// (`_inf_int(f1) ** 2 > par.f1_inf_bound_sq`), and the top-level
/// `RiVeR::verify` only caught it by accident, when re-encoding refused
/// the oversized field.
///
/// A norm past `2^64` exceeds every bound this crate has, and saying so
/// keeps the function total on a hand-built proof rather than wrapping.
fn inf_over_sq(norm: i128, bound_sq: Rat) -> bool {
    debug_assert!(norm >= 0, "norms are non-negative");
    let n = norm.unsigned_abs();
    match n.checked_mul(n) {
        Some(square) => bound_sq.exceeded_by(square),
        None => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::{RIVER_N8, RIVER_TOY};
    use crate::ring::{round_p, rounding_error};
    use crate::sample::{
        challenge_from_hash, hash_bytes, uniform_beta_vec, uniform_poly, DS_COMMIT, DS_G, DS_KEYGEN,
    };

    /// `Oom::verify` is total on an `i64::MIN` in every peer-controlled
    /// field, in **debug** as well as release.
    ///
    /// `i64::MIN.abs()` is the trap: it panics with overflow in debug and
    /// wraps back to `i64::MIN` in release, which is negative — so a
    /// magnitude test written `c.abs() > bound` fails open on exactly the
    /// value an attacker would choose.  `b_hi` had one.
    #[test]
    fn verify_is_total_on_extreme_typed_fields() {
        let par = RIVER_TOY;
        let j_star = 1;
        let (_attempts, pi, ck, rho) = run(par, j_star, b"extremes");
        let fx = fixture(&par, j_star);
        let oom = Oom::new(par, &fx.rho);
        let statement =
            OomStatement::new(&oom, &fx.a_mat, &fx.h_m, &fx.ring_pks, &fx.value).unwrap();
        assert!(
            oom.verify(&statement, &pi, &ck, &rho),
            "the fixture is honest"
        );

        for extreme in [i64::MIN, i64::MAX, i64::MIN + 1] {
            for (label, mutate) in [("b_hi", 0usize), ("f1", 1), ("zb", 2), ("x", 3)] {
                let mut bad = pi.clone();
                match mutate {
                    0 => bad.b_hi[0][0] = extreme,
                    1 => bad.f1[0][0] = extreme,
                    2 => bad.zb[0][0] = extreme,
                    _ => bad.x[0] = extreme,
                }
                assert!(
                    !oom.verify(&statement, &bad, &ck, &rho),
                    "{label} = {extreme} was accepted"
                );
            }
        }
    }

    /// The response checks cut off at exactly the encoder's cap — driven
    /// through `response_bounds_ok`, which is what production calls.
    ///
    /// What this establishes, and what it does not:
    ///
    /// * the four infinity bounds accept the cap and refuse `cap + 1`,
    ///   which is what squaring the bound without squaring the norm hid —
    ///   at `RiVeR-N8` that compared `||f_1||_inf` against
    ///   `24576^2 = 603979776` rather than against `24576`.  Rewiring one
    ///   of `response_bounds_ok`'s five lines to the wrong helper turns
    ///   this red, which a test driving `inf_over_sq` directly would not;
    /// * that `prove` and `verify` *call* it is **not** established here.
    ///   Deleting either call would leave this green.  What rules that
    ///   out is structural rather than behavioural — there is one
    ///   function and two call sites, so the two paths cannot drift into
    ///   different conventions — and no unit test can distinguish a
    ///   deleted verifier call anyway: every single-field mutation that
    ///   trips a bound also feeds the Fiat–Shamir challenge, so `verify`
    ///   rejects it either way.  Saying so beats a coverage claim that
    ///   reads stronger than it is.
    ///
    /// The cap goes in **one coefficient**, not all of them: a response
    /// every coefficient of which sits at `6 sigma` is far outside the
    /// Euclidean bound, so an all-at-cap probe would fail for the wrong
    /// reason and tell us nothing about the infinity check it names.  The
    /// Euclidean bound gets its own case below, which is the one that
    /// has to bite where the infinity bounds do not.
    #[test]
    fn the_response_checks_cut_off_at_the_encoders_cap() {
        for par in crate::params::PROFILES {
            let codec = crate::codec::RiVeRCodec::new(par);
            // One coefficient of one row set to `v`; everything else zero.
            let spike = |rows: usize, v: i64| {
                let mut out = vec![vec![0i64; par.d]; rows];
                out[0][0] = v;
                out
            };
            let zero = |rows: usize| vec![vec![0i64; par.d]; rows];

            for (label, cap) in [
                ("f1", codec.bound_f1),
                ("zb", codec.bound_zb),
                ("zs", codec.bound_zs),
                ("zm", codec.bound_zm),
            ] {
                for (delta, expected) in [(0i64, true), (1, false)] {
                    let v = cap + delta;
                    let f1 = if label == "f1" {
                        spike(par.N - 1, v)
                    } else {
                        zero(par.N - 1)
                    };
                    let zb = if label == "zb" {
                        spike(par.k_hat, v)
                    } else {
                        zero(par.k_hat)
                    };
                    let mut z = if label == "zs" {
                        spike(par.s_dim(), v)
                    } else {
                        zero(par.s_dim())
                    };
                    z.extend(if label == "zm" {
                        spike(par.m_dim(), v)
                    } else {
                        zero(par.m_dim())
                    });
                    let (zs_c, zm_c) = z.split_at(par.s_dim());
                    assert_eq!(
                        response_bounds_ok(&par, &f1, &zb, zs_c, zm_c, &z),
                        expected,
                        "{} {label} at {v} (cap {cap}) — is the norm being squared?",
                        par.name
                    );
                }
            }

            // The Euclidean bound is the one check that is *not* squared
            // on the norm side, and it has to bite where the infinity
            // bounds do not: every coefficient just inside `6 sigma` is
            // far outside `1.2 sqrt(sigma_s^2 d ell + sigma_m^2 d(n+1))`.
            let mut full = vec![vec![codec.bound_zs; par.d]; par.s_dim()];
            full.extend(vec![vec![codec.bound_zm; par.d]; par.m_dim()]);
            let (zs_c, zm_c) = full.split_at(par.s_dim());
            assert!(
                !response_bounds_ok(&par, &zero(par.N - 1), &zero(par.k_hat), zs_c, zm_c, &full),
                "{}: a full-width z passed the Euclidean bound",
                par.name
            );

            // and an all-zero response passes everything
            let z = zero(par.r_dim());
            let (zs_c, zm_c) = z.split_at(par.s_dim());
            assert!(response_bounds_ok(
                &par,
                &zero(par.N - 1),
                &zero(par.k_hat),
                zs_c,
                zm_c,
                &z
            ));

            // `B_g0` / `B_g1` are stated directly, not as `K sqrt(M)`, so
            // they are compared *unsquared*.  Mixing the two conventions
            // is the bug above, so the other convention is pinned here
            // too: the cut-off is at the bound itself, not at its root.
            for (label, bound) in [("B_g0", par.B_g0()), ("B_g1", par.B_g1())] {
                let cap = bound.floor() as i128;
                assert!(!over(cap, bound), "{} {label}: {cap} refused", par.name);
                assert!(
                    over(cap + 1, bound),
                    "{} {label}: {} accepted — is it being squared?",
                    par.name,
                    cap + 1
                );
            }
        }
    }

    /// Everything `OM.Com` needs, built the way `RiVeR.eval` builds it.
    struct Fixture {
        a_mat: PolyMat,
        h_m: PolyVec,
        ring_pks: Vec<PolyVec>,
        value: Poly,
        r: PolyVec,
        rho: Vec<u8>,
    }

    fn fixture(par: &RiVeRParams, j_star: usize) -> Fixture {
        let rq = Ring::new(par.q(), par.d);
        let rho = hash_bytes(
            32,
            &[DS_KEYGEN, b".rho"].concat(),
            &[Part::Bytes(&[7u8; 32])],
        );
        let a_mat = sam_mat(&rho, par.q(), par.n, par.ell, par.d, "RiVeR.A");

        let mut ring_pks = Vec::with_capacity(par.N);
        let mut sk_star = Vec::new();
        for i in 0..par.N {
            let mut xof = Xof::new(DS_KEYGEN, &[Part::Bytes(&[i as u8; 32])]);
            let s = uniform_beta_vec(&mut xof, par.beta, par.d, par.ell, par.q());
            let as_ = rq.mat_vec(&a_mat, &s);
            let t: PolyVec = as_.iter().map(|row| round_p(row, par.q0)).collect();
            if i == j_star {
                sk_star = s;
            }
            ring_pks.push(t);
        }

        let mut g = Xof::new(DS_G, &[Part::Bytes(b"unit")]);
        let h_m: PolyVec = (0..par.ell)
            .map(|_| uniform_poly(&mut g, par.q(), par.d))
            .collect();

        let inner = rq.inner(&h_m, &sk_star);
        let value = round_p(&inner, par.q0);
        let e_eval_canonical = rounding_error(&rq, &inner, &value, par.q0);
        let e_eval_centered: Vec<i64> = e_eval_canonical
            .iter()
            .map(|&c| c as i64 - par.B_e() as i64)
            .collect();
        let e_eval = rq.from_centered(&e_eval_centered);
        let as_star = rq.mat_vec(&a_mat, &sk_star);
        let mut r = sk_star;
        for i in 0..par.n {
            let canonical = rounding_error(&rq, &as_star[i], &ring_pks[j_star][i], par.q0);
            let centered: Vec<i64> = canonical
                .iter()
                .map(|&c| c as i64 - par.B_e() as i64)
                .collect();
            r.push(rq.from_centered(&centered));
        }
        r.push(e_eval);

        Fixture {
            a_mat,
            h_m,
            ring_pks,
            value,
            r,
            rho,
        }
    }

    /// Run attempts until one succeeds; returns `(attempts, proof)`.
    fn run(par: RiVeRParams, j_star: usize, label: &[u8]) -> (usize, OomProof, Vec<u8>, Vec<u8>) {
        let fx = fixture(&par, j_star);
        let oom = Oom::new(par, &fx.rho);
        let statement = OomStatement::new(&oom, &fx.a_mat, &fx.h_m, &fx.ring_pks, &fx.value)
            .expect("a well-formed statement");
        assert_eq!(
            statement.apply_ck(&fx.r),
            statement.c_i(j_star),
            "the honest opening must open c_j*"
        );
        let ck = hash_bytes(32, b"unit.ck", &[Part::Bytes(&fx.rho)]);
        let rho_d = hash_bytes(32, b"unit.rho", &[Part::Bytes(label)]);
        for k in 0..64u8 {
            let mut xof = Xof::new(DS_COMMIT, &[Part::Bytes(&rho_d), Part::Bytes(&[k])]);
            let (commitment, state) = oom.com(&statement, j_star, &mut xof);
            if let Some(pi) = oom.prove(&fx.r, &commitment, &state, &ck, &rho_d, &mut xof) {
                assert!(oom.verify(&statement, &pi, &ck, &rho_d));
                return (k as usize + 1, pi, ck, rho_d);
            }
        }
        panic!("no accepting attempt in 64");
    }

    /// The selector identities the whole layer rests on, checked without a
    /// proof: `sum a_i = 0`, `sum f_i = x`, and `g_i = x c_sel,i + d_i`.
    /// Port of `oom.py`'s `__main__`.
    #[test]
    fn selector_invariants_hold() {
        let par = RIVER_TOY;
        let (d, n_ring) = (par.d, par.N);
        let oom = Oom::new(par, &[2u8; 32]);
        let rqhat = oom.rqhat();
        let j_star = 2;

        let mut xof = Xof::new(DS_COMMIT, &[Part::Bytes(b"selftest")]);
        let tail = gaussian_vec(
            &mut xof,
            oom.sigma_a.0,
            d,
            n_ring - 1,
            par.q_hat,
            oom.sigma_a.1,
        );
        let mut a: Vec<Vec<i64>> = vec![vec![0i64; d]];
        for row in &tail {
            a.push(rqhat.centered(row));
        }
        for k in 0..d {
            a[0][k] = -a[1..].iter().map(|row| row[k]).sum::<i64>();
        }
        for k in 0..d {
            assert_eq!(a.iter().map(|row| row[k]).sum::<i64>(), 0, "sum a_i == 0");
        }

        let x_hat = challenge_from_hash(
            d,
            par.w,
            par.gamma,
            par.q_hat,
            &[Part::Bytes(b"x-selftest")],
        );
        let x = rqhat.centered(&x_hat);
        let f: Vec<Vec<i64>> = (0..n_ring)
            .map(|i| {
                if i == j_star {
                    (0..d).map(|k| x[k] + a[i][k]).collect()
                } else {
                    a[i].clone()
                }
            })
            .collect();
        for k in 0..d {
            assert_eq!(
                f.iter().map(|row| row[k]).sum::<i64>(),
                x[k],
                "sum f_i == x"
            );
        }

        // g_i == x c_sel,i + d_i, with d_i = -a_i^2 and c_sel,i = ±a_i
        let x_lift = rqhat.from_centered(&x);
        for i in 0..n_ring {
            let fi = rqhat.from_centered(&f[i]);
            let diff = rqhat.sub(&x_lift, &fi);
            let g = rqhat.centered(&rqhat.mul(&fi, &diff));

            let sign: i64 = if i == j_star { -1 } else { 1 };
            let c_sel: Vec<i64> = a[i].iter().map(|&c| sign * c).collect();
            let ai = rqhat.from_centered(&a[i]);
            let d_i = rqhat.mul(&ai, &ai); // +a_i^2; the identity subtracts it
            let rhs =
                rqhat.centered(&rqhat.sub(&rqhat.mul(&x_lift, &rqhat.from_centered(&c_sel)), &d_i));
            assert_eq!(g, rhs, "g_{i} != x c_sel,{i} + d_{i}");
        }

        // the challenge is in C^d_{w,gamma}
        assert_eq!(x.iter().filter(|&&c| c != 0).count(), par.w);
        assert!(x.iter().all(|&c| c.unsigned_abs() <= par.gamma));
    }

    #[test]
    fn honest_proof_verifies_at_the_toy_profile() {
        let (attempts, pi, _, _) = run(RIVER_TOY, 1, b"toy");
        assert!(attempts >= 1);
        assert_eq!(pi.f1.len(), RIVER_TOY.N - 1);
        assert_eq!(pi.zb.len(), RIVER_TOY.k_hat);
        assert_eq!(pi.z.len(), RIVER_TOY.r_dim());
    }

    /// The published profile, which is the one whose bounds are the paper's.
    #[test]
    fn honest_proof_verifies_at_a_published_profile() {
        let (_, pi, _, _) = run(RIVER_N8, 3, b"n8");
        assert_eq!(pi.f1.len(), RIVER_N8.N - 1);
    }

    /// `verify` must reject every field it checks, one at a time — otherwise
    /// a passing honest proof says nothing about the rest.
    #[test]
    fn every_transmitted_field_is_bound() {
        let par = RIVER_TOY;
        let j_star = 1;
        let fx = fixture(&par, j_star);
        let oom = Oom::new(par, &fx.rho);
        let statement = OomStatement::new(&oom, &fx.a_mat, &fx.h_m, &fx.ring_pks, &fx.value)
            .expect("a well-formed statement");
        let ck = hash_bytes(32, b"unit.ck", &[Part::Bytes(&fx.rho)]);
        let rho_d = hash_bytes(32, b"unit.rho", &[Part::Bytes(b"tamper")]);

        let mut good = None;
        for k in 0..64u8 {
            let mut xof = Xof::new(DS_COMMIT, &[Part::Bytes(&rho_d), Part::Bytes(&[k])]);
            let (c, s) = oom.com(&statement, j_star, &mut xof);
            if let Some(pi) = oom.prove(&fx.r, &c, &s, &ck, &rho_d, &mut xof) {
                good = Some(pi);
                break;
            }
        }
        let pi = good.expect("no accepting attempt");
        assert!(oom.verify(&statement, &pi, &ck, &rho_d));

        let flip = |v: &mut Vec<Vec<i64>>| {
            v[0][0] += 1;
        };
        for (label, mut bad) in [
            ("b_hi", {
                let mut p = pi.clone();
                flip(&mut p.b_hi);
                p
            }),
            ("x", {
                let mut p = pi.clone();
                let at = p.x.iter().position(|&c| c != 0).unwrap();
                p.x[at] = -p.x[at];
                p
            }),
            ("f1", {
                let mut p = pi.clone();
                flip(&mut p.f1);
                p
            }),
            ("zbin", {
                let mut p = pi.clone();
                flip(&mut p.zb);
                p
            }),
        ] {
            assert!(
                !oom.verify(&statement, &bad, &ck, &rho_d),
                "tampered {label} accepted"
            );
            let _ = &mut bad;
        }

        // `z` is residues, so move it inside the ring
        let mut bad_z = pi.clone();
        bad_z.z[0][0] = (bad_z.z[0][0] + 1) % par.q();
        assert!(!oom.verify(&statement, &bad_z, &ck, &rho_d), "tampered z");

        // A different statement must not verify, and the interesting form is
        // one that leaves `ck_digest` alone: the statement has to be bound
        // through `E'`, not only through the digest the caller supplies.
        let mut other_h = fx.h_m.clone();
        other_h[0][0] = (other_h[0][0] + 1) % par.q();
        let other_stmt =
            OomStatement::new(&oom, &fx.a_mat, &other_h, &fx.ring_pks, &fx.value).unwrap();
        assert!(!oom.verify(&other_stmt, &pi, &ck, &rho_d), "wrong h_m");

        let mut other_v = fx.value.clone();
        other_v[0] = (other_v[0] + 1) % par.p;
        let other_value =
            OomStatement::new(&oom, &fx.a_mat, &fx.h_m, &fx.ring_pks, &other_v).unwrap();
        assert!(!oom.verify(&other_value, &pi, &ck, &rho_d), "wrong value");

        let mut other_ring = fx.ring_pks.clone();
        other_ring[0][0][0] = (other_ring[0][0][0] + 1) % par.p;
        let other_pks =
            OomStatement::new(&oom, &fx.a_mat, &fx.h_m, &other_ring, &fx.value).unwrap();
        assert!(!oom.verify(&other_pks, &pi, &ck, &rho_d), "wrong ring");
        assert!(
            !oom.verify(&statement, &pi, &ck, &[9u8; 32]),
            "wrong nonce digest"
        );
    }

    /// Wrong shapes are `false`, never a panic — `verify` takes its proof
    /// from a peer.
    /// Every field is validated *before* it is used in arithmetic.
    ///
    /// `f_0 = x - sum_{i>=1} f_i` is `i64` subtraction, so an unchecked `x`
    /// of `i64::MIN` against an in-bound `f_1` overflowed: a panic in debug
    /// and a wrap in release.  The reference is total here only because
    /// Python integers do not overflow, so this is a defect the port
    /// introduced and the port has to close.
    #[test]
    fn extreme_typed_fields_are_rejected_before_any_arithmetic() {
        let par = RIVER_TOY;
        let fx = fixture(&par, 1);
        let oom = Oom::new(par, &fx.rho);
        let st = OomStatement::new(&oom, &fx.a_mat, &fx.h_m, &fx.ring_pks, &fx.value).unwrap();
        let base = OomProof {
            b_hi: vec![vec![0i64; par.d]; par.n_hat],
            x: vec![0i64; par.d],
            f1: vec![vec![1i64; par.d]; par.N - 1],
            zb: vec![vec![0i64; par.d]; par.k_hat],
            z: vec![vec![0u64; par.d]; par.r_dim()],
        };

        for extreme in [i64::MIN, i64::MAX, i64::MIN + 1] {
            let mut pi = base.clone();
            pi.x[0] = extreme;
            assert!(
                !oom.verify(&st, &pi, &[0u8; 32], &[0u8; 32]),
                "x = {extreme}"
            );
        }
        for extreme in [i64::MIN, i64::MAX, -1] {
            let mut pi = base.clone();
            pi.b_hi[0][0] = extreme;
            assert!(
                !oom.verify(&st, &pi, &[0u8; 32], &[0u8; 32]),
                "B = {extreme}"
            );
        }

        // and the membership test is exact, not a magnitude bound
        let g = par.gamma as i64;
        assert!(in_challenge_space(
            &{
                let mut v = vec![0i64; par.d];
                for c in v.iter_mut().take(par.w) {
                    *c = g;
                }
                v
            },
            par.w,
            par.gamma
        ));
        // TOY has `w == d`, so "one too many" is not expressible there;
        // "one too few" is, at every profile.
        let mut too_few = vec![0i64; par.d];
        for c in too_few.iter_mut().take(par.w - 1) {
            *c = 1;
        }
        assert!(!in_challenge_space(&too_few, par.w, par.gamma), "weight");
        let mut too_big = vec![0i64; par.d];
        for c in too_big.iter_mut().take(par.w) {
            *c = 1;
        }
        too_big[0] = g + 1;
        assert!(!in_challenge_space(&too_big, par.w, par.gamma), "magnitude");

        // every honest challenge passes it
        for k in 0..8u8 {
            let x_hat = challenge_from_hash(
                par.d,
                par.w,
                par.gamma,
                par.q_hat,
                &[Part::Bytes(b"member"), Part::Bytes(&[k])],
            );
            assert!(in_challenge_space(
                &oom.rqhat().centered(&x_hat),
                par.w,
                par.gamma
            ));
        }
    }

    /// A statement is checked once, at construction, and cannot be built
    /// wrong — two of its fields come from whoever sent the proof.
    #[test]
    fn malformed_statements_are_refused_at_construction() {
        let par = RIVER_TOY;
        let fx = fixture(&par, 1);
        let oom = Oom::new(par, &fx.rho);
        assert!(
            OomStatement::new(&oom, &fx.a_mat, &fx.h_m, &fx.ring_pks, &fx.value).is_some(),
            "the honest statement must build"
        );

        let short_a = fx.a_mat[..par.n - 1].to_vec();
        assert!(OomStatement::new(&oom, &short_a, &fx.h_m, &fx.ring_pks, &fx.value).is_none());

        let mut ragged = fx.a_mat.clone();
        ragged[0][0].pop();
        assert!(OomStatement::new(&oom, &ragged, &fx.h_m, &fx.ring_pks, &fx.value).is_none());

        let short_h = fx.h_m[..par.ell - 1].to_vec();
        assert!(OomStatement::new(&oom, &fx.a_mat, &short_h, &fx.ring_pks, &fx.value).is_none());

        let mut long_ring = fx.ring_pks.clone();
        long_ring.push(fx.ring_pks[0].clone());
        assert!(OomStatement::new(&oom, &fx.a_mat, &fx.h_m, &long_ring, &fx.value).is_none());

        // non-canonical residues, in each modulus
        let mut bad_a = fx.a_mat.clone();
        bad_a[0][0][0] = par.q();
        assert!(OomStatement::new(&oom, &bad_a, &fx.h_m, &fx.ring_pks, &fx.value).is_none());

        let mut bad_pk = fx.ring_pks.clone();
        bad_pk[0][0][0] = par.p;
        assert!(OomStatement::new(&oom, &fx.a_mat, &fx.h_m, &bad_pk, &fx.value).is_none());

        let mut bad_v = fx.value.clone();
        bad_v[0] = par.p;
        assert!(OomStatement::new(&oom, &fx.a_mat, &fx.h_m, &fx.ring_pks, &bad_v).is_none());

        // and a statement shaped for a different profile
        let other = fixture(&RIVER_N8, 1);
        assert!(
            OomStatement::new(
                &oom,
                &other.a_mat,
                &other.h_m,
                &other.ring_pks,
                &other.value
            )
            .is_none(),
            "a statement from another profile must not build against this Oom"
        );
    }

    /// A statement built against one `Oom` must not verify against another.
    ///
    /// This is the case the previous cross-profile test missed: it fed
    /// N8-shaped data to the *toy constructor*, which correctly refused it.
    /// The reachable failure is the other way — build a **valid** N8
    /// statement, then hand it to the toy verifier.  Lifetimes do not
    /// encode identity, so that type-checked, passed every shape check
    /// (they are made against the toy proof), and then sliced an
    /// 11-element `z` with N8's `ell = 56`.
    #[test]
    fn a_statement_from_another_oom_is_refused() {
        let toy = RIVER_TOY;
        let n8 = RIVER_N8;
        let fx8 = fixture(&n8, 1);
        let oom8 = Oom::new(n8, &fx8.rho);
        let st8 = OomStatement::new(&oom8, &fx8.a_mat, &fx8.h_m, &fx8.ring_pks, &fx8.value)
            .expect("a valid N8 statement");

        let fxt = fixture(&toy, 1);
        let oom_toy = Oom::new(toy, &fxt.rho);
        let pi = OomProof {
            b_hi: vec![vec![0i64; toy.d]; toy.n_hat],
            x: {
                let mut v = vec![0i64; toy.d];
                for c in v.iter_mut().take(toy.w) {
                    *c = 1;
                }
                v
            },
            f1: vec![vec![0i64; toy.d]; toy.N - 1],
            zb: vec![vec![0i64; toy.d]; toy.k_hat],
            z: vec![vec![0u64; toy.d]; toy.r_dim()],
        };
        assert!(!st8.belongs_to(&oom_toy));
        assert!(st8.belongs_to(&oom8));
        assert!(
            !oom_toy.verify(&st8, &pi, &[0u8; 32], &[0u8; 32]),
            "a statement from another Oom must be refused, not panic"
        );

        // and two `Oom`s at the *same* profile are still distinct objects,
        // because they need not share a `G'`
        let other_toy = Oom::new(toy, b"a different rho");
        let st_toy =
            OomStatement::new(&oom_toy, &fxt.a_mat, &fxt.h_m, &fxt.ring_pks, &fxt.value).unwrap();
        assert!(!st_toy.belongs_to(&other_toy));
        assert!(!other_toy.verify(&st_toy, &pi, &[0u8; 32], &[0u8; 32]));
    }

    #[test]
    fn malformed_proofs_are_rejected_without_panicking() {
        let par = RIVER_TOY;
        let fx = fixture(&par, 1);
        let oom = Oom::new(par, &fx.rho);
        let statement = OomStatement::new(&oom, &fx.a_mat, &fx.h_m, &fx.ring_pks, &fx.value)
            .expect("a well-formed statement");
        let empty = OomProof {
            b_hi: vec![],
            x: vec![],
            f1: vec![],
            zb: vec![],
            z: vec![],
        };
        assert!(!oom.verify(&statement, &empty, &[0u8; 32], &[0u8; 32]));

        let ragged = OomProof {
            b_hi: vec![vec![0i64; par.d]; par.n_hat],
            x: vec![0i64; par.d],
            f1: vec![vec![0i64; par.d - 1]; par.N - 1],
            zb: vec![vec![0i64; par.d]; par.k_hat],
            z: vec![vec![0u64; par.d]; par.r_dim()],
        };
        assert!(!oom.verify(&statement, &ragged, &[0u8; 32], &[0u8; 32]));

        // a non-canonical residue in `z` centres to a different integer than
        // it encodes, so it has to be caught before it is centred
        let mut noncanon = ragged;
        noncanon.f1 = vec![vec![0i64; par.d]; par.N - 1];
        noncanon.z[0][0] = par.q();
        assert!(!oom.verify(&statement, &noncanon, &[0u8; 32], &[0u8; 32]));
    }
}
