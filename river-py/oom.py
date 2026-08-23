"""
oom.py -- The relaxed one-out-of-many proof used by RiVeR (Figure 7).

`OM` proves knowledge of a short opening of *one* vector `c_{j*}` in a public
list `(c_i)_{i in [N]}`, without revealing `j*`:

    c_{j*} = Com^0_{ck_r}(r) = ck_r * r   (mod q)

The selector machinery lives in `R_qhat`; the opening lives in `R_q`.  The
challenge `x` is a single integer polynomial canonically embedded into both.

Representation
--------------
Selector-side quantities (`a`, `b`, `c_sel`, `d`, `f`, `g`, `z_b`, `x`) are
*integer* polynomials whose coefficients stay far below `qhat/2`, so we work
with `R_qhat` residues and recover exact integers by centring.  This is what
makes `g_i = f_i(x - f_i)` a genuine integer product and the bound checks
`||g||_inf <= B_g` meaningful.

The commitments `A` and `B` are the **high bits** of a `G'` product, taken on
the **centred** representative in `(-qhat/2, qhat/2]`, as the paper's
preliminaries define `[[.]]_K`.  The low part lands in `(-2^{K-1}, 2^{K-1}]`,
which is what the correctness argument's `||e_B||_inf <= 2^{K_b - 1}` needs.
"""

from ring import Ring, power2round, mod_pm
from sample import (XOF, DS_COMMIT, sam_mat, gaussian_vec,
                    uniform_beta_vec, rational_sigma, challenge_from_hash,
                    rej1, rej2)
from codec import pack_unsigned, pack_signed, width_for_bound


class OOMStatement:
    """The public statement `(ck_{r,m}, (c_i)_{i in [N]})`, kept structural.

    `ck_{r,m} = [A | -I_n | 0 ; h_m^T | 0 | -1]` is never materialised as a
    dense matrix: `apply()` uses the block structure directly, which turns an
    `(n+1) x (ell+n+1)` product into one `n x ell` product plus an inner
    product.  The public offsets in
    `c_i = (q_0 t_i + delta_{e,n}, q_0 v + delta_e)` are derived on use.
    """

    def __init__(self, par, Rq, A, h_m, ring_pks, value):
        self.par = par
        self.Rq = Rq
        self.A = A                     # n x ell over R_q
        self.h_m = h_m                 # ell   over R_q
        self.ring_pks = ring_pks       # N public keys, each in R_p^n
        self.value = value             # v in R_p

    def apply_ck(self, y):
        """`ck_{r,m} * y` for `y = (y_s, y_key, y_eval)`.

        Indexed by `par.ell` / `par.n`, which is the *semantic* partition of
        the opening and is unaffected by where the two response widths
        split it (`par.s_dim`).  The paper moved the latter, not this.
        """
        par, R = self.par, self.Rq
        y_s = y[:par.ell]
        y_key = y[par.ell:par.ell + par.n]
        y_eval = y[par.ell + par.n]
        top = [R.sub(R.inner(self.A[i], y_s), y_key[i]) for i in range(par.n)]
        bottom = R.sub(R.inner(self.h_m, y_s), y_eval)
        return top + [bottom]

    def c_i(self, i):
        """The i-th offset derived vector in `R_q^{n+1}`."""
        par, R = self.par, self.Rq
        t = self.ring_pks[i]
        delta = [par.B_e] * par.d
        return ([R.add(R.scale(par.q0, [c % par.q for c in t[j]]), delta)
                 for j in range(par.n)]
                + [R.add(R.scale(par.q0, [c % par.q for c in self.value]),
                         delta)])

    def combine_c(self, coeffs):
        """`sum_i coeff_i * c_i` in R_q^{n+1}.

        Uses the shared structure of the `c_i`: the top block is
        `q_0 * sum_i coeff_i t_i + delta_e * sum_i coeff_i` and the
        bottom entry has the same offset term.
        """
        par, R = self.par, self.Rq
        N = par.N
        top = []
        for j in range(par.n):
            acc = R.zero()
            for i in range(N):
                t_ij = [c % par.q for c in self.ring_pks[i][j]]
                acc = R.add(acc, R.mul(coeffs[i], t_ij))
            top.append(R.scale(par.q0, acc))
        total = R.zero()
        for i in range(N):
            total = R.add(total, coeffs[i])
        delta_term = R.mul(total, [par.B_e] * par.d)
        top = [R.add(row, delta_term) for row in top]
        bottom = R.add(
            R.scale(par.q0, R.mul(total, [c % par.q for c in self.value])),
            delta_term)
        return top + [bottom]


class OOM:
    """`OM.Setup / Com / Prove / Ver` for one parameter profile."""

    def __init__(self, par, seed):
        self.par = par
        self.seed = seed
        self.Rq = Ring(par.q, par.d)
        self.Rqhat = Ring(par.q_hat, par.d)
        #: `G' <- SamMat(rho, qhat, n_hat, k_hat + 2N, "G'")`
        self.Gp = sam_mat(seed, par.q_hat, par.n_hat, par.gprime_cols,
                          par.d, "G'")
        self.sigma_a = rational_sigma(par.sigma_a)
        self.sigma_b = rational_sigma(par.sigma_b)
        #: The outer response is split in two.  The paper puts
        #: `r_0 = (s, e_key)` -- `ell + n` ring elements -- at width
        #: `sigma_s = phi_s B_s`, and `r_1 = e_eval`, a single ring element,
        #: at `sigma_m = phi_m eta_m`. They are two different Gaussians
        #: drawn from one XOF stream, in this order.
        self.sigma_s = rational_sigma(par.sigma_s)
        self.sigma_m = rational_sigma(par.sigma_m)
        #: `[[.]]_K` on the *centred* representative leaves high bits in
        #: roughly `[-qhat/2^{K+1}, +qhat/2^{K+1}]`; these are the packing
        #: bounds, and they are signed.  See `_high_low`.
        self.hi_bound_A = high_bits_bound(par.q_hat, par.K_a)
        self.hi_bound_B = high_bits_bound(par.q_hat, par.K_b)

    # ---- helpers ---------------------------------------------------------

    def _lift(self, ring, int_poly):
        """Embed an integer polynomial into `ring`."""
        return [c % ring.q for c in int_poly]

    def _center(self, ring, poly):
        return ring.centered(poly)

    def _gprime(self, blocks):
        """`G' * (block_0 || block_1 || block_2)` in `R_qhat`."""
        vec = [self._lift(self.Rqhat, p) for group in blocks for p in group]
        assert len(vec) == self.par.gprime_cols, len(vec)
        return self.Rqhat.mat_vec(self.Gp, vec)

    def _high_low(self, vec, K):
        """`[[.]]_K` and `. mod^pm 2^K` of each coefficient.

        Taken on the **centred** representative `\bar a in (-qhat/2, qhat/2]`,
        which is what the preliminaries define:

            a mod^pm 2^K := \bar a - 2^K floor((\bar a + 2^{K-1} - 1)/2^K)
            [[a]]_K      := (\bar a - (a mod^pm 2^K)) / 2^K

        Both ranges are asymmetric -- the low part lands in
        `(-2^{K-1}, 2^{K-1}]`, closed at the top -- and `power2round` already
        implements that tie convention, so the only thing that moves here is
        which representative goes in.

        That closes it.  Through the paper this was taken on the
        canonical `[0, qhat)` representative, because the operator was
        undefined and the canonical reading let the codec encode `B`
        unsigned; the paper then defined it the other way and this
        code deliberately did not follow, since aligning moves protocol
        bytes.  The definition is now unambiguous and stated in the
        preliminaries rather than mid-appendix, so the code follows it:
        about half the high parts are negative, the transmitted `B` field is
        signed, and every vector moves.
        """
        highs, lows = [], []
        for poly in vec:
            centred = self.Rqhat.centered(poly)
            hi, lo = zip(*(power2round(c, K) for c in centred))
            highs.append(list(hi))
            lows.append(list(lo))
        return highs, lows

    # ---- OM.Com ----------------------------------------------------------

    def com(self, statement, j_star, r, xof):
        """`(t_OOM, st_OOM) <- OM.Com(pp, m, ck_r, (c_i), (j*, r))`."""
        par = self.par
        N, d = par.N, par.d

        # b = (delta_{j*,0}, ..., delta_{j*,N-1})
        b = [[0] * d for _ in range(N)]
        b[j_star][0] = 1

        # a_1..a_{N-1} <- D_{phi_a B_a};  a_0 = -sum_{i>=1} a_i
        tail = gaussian_vec(xof, self.sigma_a[0], d, N - 1, par.q_hat,
                            self.sigma_a[1])
        a = [None] * N
        for i in range(1, N):
            a[i] = self.Rqhat.centered(tail[i - 1])
        head = [0] * d
        for i in range(1, N):
            head = [head[k] - a[i][k] for k in range(d)]
        a[0] = head

        # d = (-a_0^2, ..., -a_{N-1}^2)  and  c_sel = a o (1 - 2b)
        d_vec, c_sel = [], []
        for i in range(N):
            ai = self._lift(self.Rqhat, a[i])
            sq = self.Rqhat.centered(self.Rqhat.mul(ai, ai))
            d_vec.append([-c for c in sq])
            sign = -1 if i == j_star else 1
            c_sel.append([sign * c for c in a[i]])

        # r_b <- U_beta^{k_hat},  r_a <- D_{phi_b B}^{k_hat}
        #
        # REPAIR.  The figure samples `r_a <- D_B`, while its `Rej_2`
        # call uses `(phi_b, B)` and the communication formula charges
        # `h(phi_b B)`.  A rejection sampler is only correct when the mask
        # width equals the sigma in its acceptance test, and `phi_b B` is
        # also the reading the paper's own size accounting uses -- its
        # `k_hat d h(phi_b B)` term is what reproduces the reported
        # |pi_OOM|, which `test_params.py` checks.
        r_b = [self.Rqhat.centered(p) for p in
               uniform_beta_vec(xof, par.beta, d, par.k_hat, par.q_hat)]
        r_a = [self.Rqhat.centered(p) for p in
               gaussian_vec(xof, self.sigma_b[0], d, par.k_hat, par.q_hat,
                            self.sigma_b[1])]

        u_B = self._gprime((r_b, b, c_sel))
        B, e_B = self._high_low(u_B, par.K_b)
        u_A = self._gprime((r_a, a, d_vec))
        A, _ = self._high_low(u_A, par.K_a)

        # (y_s, y_key) <- D_{sigma_s}^{ell+n},  y_eval <- D_{sigma_m},
        # y_OM <- (y_s, y_key, y_eval).  Two draws, in the figure's order.
        # The concatenation is unchanged; `par.s_dim` is where it splits.
        y_s = gaussian_vec(xof, self.sigma_s[0], d, par.s_dim, par.q,
                           self.sigma_s[1])
        y_m = gaussian_vec(xof, self.sigma_m[0], d, par.m_dim, par.q,
                           self.sigma_m[1])
        y_om = y_s + y_m

        # E = ck_r y_OM - sum_i a_i c_i   (mod q)
        a_q = [self._lift(self.Rq, ai) for ai in a]
        E = [self.Rq.sub(lhs, rhs) for lhs, rhs in
             zip(statement.apply_ck(y_om), statement.combine_c(a_q))]

        t_oom = {"A": A, "B": B, "E": E}
        st_oom = {"a": a, "b": b, "c_sel": c_sel, "d": d_vec,
                  "r_a": r_a, "r_b": r_b, "y_om": y_om,
                  "u_A": u_A, "u_B": u_B, "e_B": e_B,
                  "j_star": j_star, "r": r}
        return t_oom, st_oom

    # ---- Fiat-Shamir -----------------------------------------------------

    def challenge(self, statement, ck_digest, t_oom, rho_digest):
        """`x <- H(m, ck_r, (c_i), A, B, E; rho')`."""
        par = self.par
        parts = [
            self.seed,
            ck_digest,
            pack_signed([c for p in t_oom["A"] for c in p],
                        width_for_bound(self.hi_bound_A), self.hi_bound_A),
            pack_signed([c for p in t_oom["B"] for c in p],
                        width_for_bound(self.hi_bound_B), self.hi_bound_B),
            pack_unsigned([c for p in t_oom["E"] for c in self.Rq.reduce(p)],
                          (par.q.bit_length() + 7) // 8),
            rho_digest,
        ]
        return challenge_from_hash(par.d, par.w, par.gamma, par.q_hat, *parts)

    # ---- OM.Prove --------------------------------------------------------

    def prove(self, statement, j_star, r, t_oom, st_oom, ck_digest,
              rho_digest, xof):
        """One `OM.Prove` attempt.  Returns `pi_OOM`, or `None` for `bot`."""
        par = self.par
        N, d = par.N, par.d

        x_hat = self.challenge(statement, ck_digest, t_oom, rho_digest)
        x = self.Rqhat.centered(x_hat)           # integer challenge polynomial

        # f_i = x b_i + a_i        (b is a unit vector, so this is cheap)
        a, b = st_oom["a"], st_oom["b"]
        f = []
        for i in range(N):
            if i == j_star:
                f.append([x[k] + a[i][k] for k in range(d)])
            else:
                f.append(list(a[i]))
        f1 = f[1:]

        # Rej_1(f_1, x * (delta_{j*,1}, ..., delta_{j*,N-1}), phi_a, B_a)
        shift = [[0] * d for _ in range(N - 1)]
        if j_star >= 1:
            shift[j_star - 1] = list(x)
        if rej1(xof, _flatten(f1), _flatten(shift), par.phi_a,
                self.sigma_a[0], self.sigma_a[1], par.REJ_TAU):
            return None

        # z_b = r_a + x r_b,   z_s = y_s + x r_0,   z_m = y_m + x r_1
        r_a, r_b = st_oom["r_a"], st_oom["r_b"]
        x_r_b = [self.Rqhat.centered(self.Rqhat.mul(self._lift(self.Rqhat, x),
                                                    self._lift(self.Rqhat, rb)))
                 for rb in r_b]
        z_b = [[r_a[i][k] + x_r_b[i][k] for k in range(d)]
               for i in range(par.k_hat)]

        x_q = self._lift(self.Rq, x)
        x_r = [self.Rq.mul(x_q, ri) for ri in r]
        z = [self.Rq.add(st_oom["y_om"][i], x_r[i]) for i in range(par.r_dim)]

        z_c = [self.Rq.centered(zi) for zi in z]
        x_r_c = [self.Rq.centered(v) for v in x_r]
        s_end = par.s_dim
        z_s_c, z_m_c = z_c[:s_end], z_c[s_end:]
        x_r0_c, x_r1_c = x_r_c[:s_end], x_r_c[s_end:]

        # The figure's disjunction, left to right and short-circuiting, so
        # the XOF is consumed in exactly that order:
        #   Rej_1((z_s, z_key), x r_0, phi_s, B_s)
        #   Rej_1(z_eval,          x r_1, phi_m, eta_m)
        #   Rej_2(z_b, x r_b, phi_b, B)
        if rej1(xof, _flatten(z_s_c), _flatten(x_r0_c), par.phi_s,
                self.sigma_s[0], self.sigma_s[1], par.REJ_TAU):
            return None
        if rej1(xof, _flatten(z_m_c), _flatten(x_r1_c), par.phi_m,
                self.sigma_m[0], self.sigma_m[1], par.REJ_TAU):
            return None
        if rej2(xof, _flatten(z_b), _flatten(x_r_b), par.phi_b,
                self.sigma_b[0], self.sigma_b[1]):
            return None

        # The four infinity-norm checks, in the figure's order, decided
        # exactly: each bound is `K sqrt(M)`, so squaring removes the
        # `sqrt` and the comparison is between exact rationals.  See
        # `params.py`, "verifier bounds".
        if _inf_int(f1) ** 2 > par.f1_inf_bound_sq:
            return None
        if _inf_int(z_b) ** 2 > par.zb_inf_bound_sq:
            return None
        if _inf_int(z_s_c) ** 2 > par.zs_inf_bound_sq:
            return None
        if _inf_int(z_m_c) ** 2 > par.zm_inf_bound_sq:
            return None

        # DEFENSIVE: prover and verifier checks must not differ.
        # `OOM.Ver` applies a Euclidean bound on `z`; the corresponding
        # prover check is commented out in the figure, and the commented
        # form uses a different, much smaller bound than the verifier's.
        # A prover that can return a proof its own verifier rejects is a
        # correctness bug, so the *verifier's* bound is applied here.  It is
        # not charged in the paper's attempt estimate, and it does not need
        # to be: measured over 103 attempts at the toy profile it never
        # fires on an attempt the four infinity-norm checks let through, so
        # it costs exactly zero restarts.  That is a measurement, not a
        # theorem -- `test_e2e.py::test_the_defensive_euclidean_check_is_free`
        # re-runs it rather than trusting this comment.
        if sum(c * c for poly in z_c for c in poly) > par.z_l2_bound_sq:
            return None

        # g_i = f_i (x - f_i)
        g = []
        for i in range(N):
            fi = self._lift(self.Rqhat, f[i])
            diff = self.Rqhat.sub(self._lift(self.Rqhat, x), fi)
            g.append(self.Rqhat.centered(self.Rqhat.mul(fi, diff)))
        if _inf(g[0]) > par.B_g0:
            return None
        if max((_inf(gi) for gi in g[1:]), default=0) > par.B_g1:
            return None

        # compression check on the low bits of the reconstructed A'
        u_prime = self._reconstruct_u(t_oom["B"], z_b, f, g, x)
        w_hat = [[mod_pm(c, par.K_a) for c in poly] for poly in u_prime]
        if _inf_int(w_hat) >= par.T_cmp:
            return None

        # Belt and braces: the margin above is what makes A' = A, but the
        # representative can also wrap at qhat.  That is a ~2^-20 event per
        # attempt, and the paper does not cover it either -- it
        # defines the operators but still argues A' = A from the decomposition
        # margin alone, which holds over Z and not across the wrap.  So we
        # simply detect it and restart.
        A_prime, _ = self._high_low(u_prime, par.K_a)
        if A_prime != t_oom["A"]:
            return None

        return {"B": t_oom["B"], "x": x, "f1": f1, "zb": z_b, "z": z}

    def _reconstruct_u(self, B, z_b, f, g, x):
        """`G' (z_b || f || g) - x 2^{K_b} B  (mod qhat)`."""
        par = self.par
        prod = self._gprime((z_b, f, g))
        shift = [self.Rqhat.mul(
            self._lift(self.Rqhat, x),
            self._lift(self.Rqhat, [c << par.K_b for c in poly]))
            for poly in B]
        return [self.Rqhat.sub(prod[i], shift[i]) for i in range(par.n_hat)]

    # ---- OM.Ver ----------------------------------------------------------

    def verify(self, statement, pi, ck_digest, rho_digest):
        """`OM.Ver(pp, m, ck_r, (c_i), pi_OOM; rho')` in {0, 1}."""
        par = self.par
        N, d = par.N, par.d
        try:
            B, x, f1, z_b, z = (pi["B"], pi["x"], pi["f1"],
                                pi["zb"], pi["z"])
        except KeyError:
            return False

        if len(f1) != N - 1 or len(z_b) != par.k_hat or len(z) != par.r_dim:
            return False

        # The figure's five checks: ||f_1||_inf <= 6 phi_a B_a,
        # ||z_b||_inf <= 6 phi_b B, ||(z_s, z_key)||_inf <= 6 sigma_s,
        # ||z||_2 <= 1.2 sqrt(sigma_s^2 d(ell+n) + sigma_m^2 d), and
        # ||z_eval||_inf <= 6 sigma_m.
        if _inf_int(f1) ** 2 > par.f1_inf_bound_sq:
            return False
        if _inf_int(z_b) ** 2 > par.zb_inf_bound_sq:
            return False

        z_c = [self.Rq.centered(zi) for zi in z]
        z_s_c, z_m_c = z_c[:par.s_dim], z_c[par.s_dim:]
        if _inf_int(z_s_c) ** 2 > par.zs_inf_bound_sq:
            return False
        if _inf_int(z_m_c) ** 2 > par.zm_inf_bound_sq:
            return False
        norm_sq = sum(c * c for poly in z_c for c in poly)
        if norm_sq > par.z_l2_bound_sq:
            return False

        # f_0 = x - sum_{i>=1} f_i
        head = list(x)
        for poly in f1:
            head = [head[k] - poly[k] for k in range(d)]
        f = [head] + [list(p) for p in f1]

        # g_i = f_i (x - f_i), rechecked against the public thresholds
        g = []
        for i in range(N):
            fi = self._lift(self.Rqhat, f[i])
            diff = self.Rqhat.sub(self._lift(self.Rqhat, x), fi)
            g.append(self.Rqhat.centered(self.Rqhat.mul(fi, diff)))
        if _inf(g[0]) > par.B_g0:
            return False
        if max((_inf(gi) for gi in g[1:]), default=0) > par.B_g1:
            return False

        # A' and E'
        u_prime = self._reconstruct_u(B, z_b, f, g, x)
        A_prime, _ = self._high_low(u_prime, par.K_a)

        f_q = [self._lift(self.Rq, fi) for fi in f]
        E_prime = [self.Rq.sub(lhs, rhs) for lhs, rhs in
                   zip(statement.apply_ck(z), statement.combine_c(f_q))]

        expect = self.challenge(statement, ck_digest,
                                {"A": A_prime, "B": B, "E": E_prime},
                                rho_digest)
        return self.Rqhat.centered(expect) == list(x)


# ---- small helpers -------------------------------------------------------

def high_bits_bound(q_hat, K):
    """`max |[[a]]_K|` over `a in Z_qhat`, taken on the centred rep.

    `\bar a` runs over `(-qhat/2, qhat/2]`, and
    `[[a]]_K = floor((\bar a + 2^{K-1} - 1) / 2^K)` is monotone in `\bar a`,
    so the extremes are reached at the ends of that interval.  Computed
    rather than estimated, because it is the codec's field bound: one too
    small refuses an honest proof, one too large costs a bit per
    coefficient.
    """
    top = q_hat // 2
    bottom = -(q_hat // 2) + (0 if q_hat % 2 else 1)
    hi_top = power2round(top, K)[0]
    hi_bottom = power2round(bottom, K)[0]
    return max(abs(hi_top), abs(hi_bottom))


def _flatten(vec):
    return [c for poly in vec for c in poly]


def _inf(poly):
    return max((abs(c) for c in poly), default=0)


def _inf_int(vec):
    return max((_inf(poly) for poly in vec), default=0)


# --------------------------------------------------------------------------
if __name__ == "__main__":
    # The invariants of the selector layer, checked on their own (Section 5
    # of the implementation notes).
    import random
    from params import TOY_PARAMS

    par = TOY_PARAMS
    oom = OOM(par, b"\x02" * 32)
    Rqh = oom.Rqhat
    rng = random.Random(11)
    d, N = par.d, par.N

    xof = XOF(DS_COMMIT, b"selftest")
    j_star = 2
    b = [[0] * d for _ in range(N)]
    b[j_star][0] = 1

    tail = gaussian_vec(xof, oom.sigma_a[0], d, N - 1, par.q_hat,
                        oom.sigma_a[1])
    a = [None] * N
    for i in range(1, N):
        a[i] = Rqh.centered(tail[i - 1])
    a[0] = [-sum(a[i][k] for i in range(1, N)) for k in range(d)]

    assert all(sum(a[i][k] for i in range(N)) == 0 for k in range(d)), \
        "sum a_i == 0"

    x = Rqh.centered(challenge_from_hash(d, par.w, par.gamma, par.q_hat,
                                         b"x-selftest"))
    f = []
    for i in range(N):
        f.append([x[k] + a[i][k] for k in range(d)] if i == j_star
                 else list(a[i]))

    # sum_i f_i == x  (because sum b_i = 1 and sum a_i = 0)
    assert [sum(f[i][k] for i in range(N)) for k in range(d)] == list(x), \
        "sum f_i == x"

    # g_i == x c_sel,i + d_i
    for i in range(N):
        fi = [c % par.q_hat for c in f[i]]
        diff = Rqh.sub([c % par.q_hat for c in x], fi)
        g = Rqh.centered(Rqh.mul(fi, diff))
        sign = -1 if i == j_star else 1
        c_sel = [sign * c for c in a[i]]
        ai = [c % par.q_hat for c in a[i]]
        d_i = Rqh.centered(Rqh.mul(ai, ai))
        rhs = Rqh.centered(Rqh.sub(
            Rqh.mul([c % par.q_hat for c in x],
                    [c % par.q_hat for c in c_sel]), d_i))
        assert g == rhs, f"g_{i} != x c_sel,{i} + d_{i}"

    # challenge weight and norms
    assert sum(1 for c in x if c != 0) == par.w
    assert max(abs(c) for c in x) <= par.gamma
    assert sum(abs(c) for c in x) <= par.w * par.gamma

    print("oom.py: selector invariants hold")
