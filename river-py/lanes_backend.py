"""
lanes_backend.py -- LANES as an exact backend for RiVeR's `Pi_ex`.

Optional component.  Select it with `RiVeR(par, exact_backend="lanes")`;
the default remains the witness-revealing `OpeningBackend` in `exact.py`.

Mapping RiVeR's relation onto LANES
-----------------------------------
`R^_ex` asks for `e_eval + B_e in [0, q_0-1]^d`, a digit reconstruction, and the
link `z_eval = x e_eval + y_eval`.  LANES proves *ternary slots* plus a
*public linear system*, so the relation is expressed as:

    slots       element 0 : y_eval        (32, not ternary)
                element 1 : e_eval        (32, not ternary)
                elements 2-5 : digits     (128, ternary)

    ternary     elements [2, 6)   -- gives digits in {-1, 0, 1}

    linear      32 rows  e_eval[i] - sum_j g_j digit'_j[i] = 0
                32 rows  sum_b M[i][b] e_eval[b] + y_eval[i] = z_eval[i]

`M` is the negacyclic multiplication-by-`x` matrix, so the second block is
exactly the link equation written out as a linear map.

Two consequences of using LANES's *native* encoding:

* The digits are carried **centred**, in `{-1, 0, 1}`, because the cubic
  product proof certifies `m^3 = m`.  The paper describes digits in
  `{0, 1, 2}`; the two are the same encoding shifted by `sum_j g_j = 30`.
  Since the OOM witness now carries centred `e_eval`, those shifts cancel and
  the reconstruction row's constant is zero.  Any implementation
  of the [ESLR23] rounding relation has to centre for the same reason -- the
  product proof cannot certify `{0, 1, 2}` directly.

  Earlier drafts contradicted themselves on the direction of that shift.
  The relation and figures agree on the centred error:
  `e_eval + 30 = sum_j g_j d_j`, equivalently
  `e_eval = sum_j g_j d'_j` on centred digits.

* The link is proved **modulo `q~`**, because that is the only modulus LANES
  has.  That used to make this backend *weaker* on the link than
  `OpeningBackend`, which checks it over `Z` only because it transmits the
  witness in the clear: `||z_eval||_inf` was permitted
  up to `6 sigma_rs > q~`, so a congruence mod `q~` did not pin the integer.
  The paper closes that by splitting the response -- `z_eval` now
  sits in the error block at `sigma_m = phi_m eta_m`, and
  `q~ > 24 phi_m eta_m` (see `exact.ExactParams.q_tilde_clears`) gives every
  accepted `z_eval`, and every difference of two of them, a unique centred
  lift.

**This module runs at the paper's own parameters.**
Both the structure -- `d~ = 256`, `l = 64`, `q~ = 67107713`,
`(n~, l~) = (4, 4)`, `D = 17`, six padded 64-slot message blocks -- and the
Gaussian widths are the revision's; `lanes_params` re-derives every figure
it prints.

The supported name `"lanes"` is nevertheless still gated, on security
*evidence* rather than on parameters: `delta_MLWE = 1.0020` is not
reproducible under either reading of the paper's Gaussian convention and
[KLSS23]'s reduction loses about `2^-94.9`.

The implementation itself is ready and says so -- `exact.LANES_BACKEND_READY`
is true, and both `lanes-experimental` vector cases are re-derived
byte for byte against `river-rs`.  What still closes the gate is the security
condition alone: `exact.lanes_gate_cause()` returns
`security-evidence-pending`, so `LanesBackend(par)` refuses, and
`LanesBackend.experimental(par)` / `exact_backend="lanes-experimental"` is
the way past it -- with a *different instance name*, so a vector case
recording it reconstructs this backend and not the gated one.  See
`exact.lanes_unavailable_reason` and `lanes_security.json`.
"""

import math

from exact import (ExactBackend, ExactParams, RADIX_WEIGHTS, decompose_poly,
                   check_relation)
from codec import (BitWriter, Layout, Field, Uniform, Rice, Signed,
                   floor_sqrt, pack_signed, width_for_bound)
import lanes_proof
import lanes_ring as R
from lanes_commit import LanesCommitmentKey, commit
from lanes_params import (ALPHA, D_DROP, RESPONSE_RANK, IDENTITY_RANK,
                          N_EX, T0_HIGH_MODULUS, Z_INF_BOUND, SIGMA_Y)
from sample import hash_bytes, DS_EXACT

#: Witness layout, in message-element indices.  The paper's order,
#: `(y_eval, e_eval, d_0, ..., d_3)`; see `exact.pack_witness`.
IDX_Y = 0
IDX_E = 1
IDX_DIGITS = 2
ALPHA_LO = IDX_DIGITS
ALPHA_HI = IDX_DIGITS + len(RADIX_WEIGHTS)      # 6 == N_EX

#: sum of the radix weights; the shift between {0,1,2} and {-1,0,1} digits
WEIGHT_SUM = sum(RADIX_WEIGHTS)                 # 30


def build_linear_system(ex, x_centered, z_eval_centered):
    """`(A, u)` over `Z_q~` encoding reconstruction and the link equation."""
    q, d = R.QTILDE, ex.d
    an = N_EX * R.LSPLIT
    rows_A, rows_u = [], []

    # reconstruction: e_eval[i] - sum_j g_j centred_digit_j[i] = 0
    for i in range(d):
        row = [0] * an
        row[IDX_E * R.LSPLIT + i] = 1
        for j, weight in enumerate(RADIX_WEIGHTS):
            row[(IDX_DIGITS + j) * R.LSPLIT + i] = (-weight) % q
        rows_A.append(row)
        rows_u.append(0)

    # link: (x * e_eval)[i] + y_eval[i] = z_eval[i], negacyclic in X^d + 1
    for i in range(d):
        row = [0] * an
        for b in range(d):
            k = i - b
            coeff = x_centered[k] if k >= 0 else -x_centered[k + d]
            row[IDX_E * R.LSPLIT + b] = coeff % q
        row[IDX_Y * R.LSPLIT + i] = 1
        rows_A.append(row)
        rows_u.append(z_eval_centered[i] % q)

    # zero padding: every one of the `N_ex (l - d)` slots a message block
    # does *not* use is constrained to 0.
    #
    # Without these rows the padding is unconstrained.  The commitment has
    # `N_ex l = 384` message coordinates while the two blocks above touch
    # only the first `d = 32` of each, so 192 columns of `A` were entirely
    # zero -- exactly the padding positions.  The product proof restricts
    # the four digit blocks to `{-1,0,1}` but not to zero, and says nothing
    # at all about the `y_eval` and `e_eval` blocks, so a prover could
    # commit to any padding it liked and still satisfy the system.  The
    # honest prover pads with zero; nothing made a dishonest one.
    for element in range(N_EX):
        for slot in range(d, R.LSPLIT):
            row = [0] * an
            row[element * R.LSPLIT + slot] = 1
            rows_A.append(row)
            rows_u.append(0)

    return {"A": rows_A, "u": rows_u}


def _field_bits(field, value):
    """Bits one layout field costs, before the layout's single byte pad."""
    w = BitWriter()
    rows = [value] if field.rows is None else value
    for row in rows:
        if field.ring is not None:
            row = field.ring.centered(row)
        for coeff in row:
            field.coder.write(w, coeff)
    return w.bit_length


def statement_bytes(ex, backend, W, x_centered, z_eval_centered):
    """Canonical image of `(W, z_eval, x)`, bound into every FS challenge."""
    return hash_bytes(
        32, DS_EXACT + b".lanes.stmt",
        backend.W_encode(W),
        pack_signed(x_centered, 1, 127),
        pack_signed(z_eval_centered, width_for_bound(backend.bound_z),
                    backend.bound_z))


def _exact_coeffs(coeffs, length, where):
    """A witness polynomial as exactly `length` genuine `int` coefficients.

    Strict on both counts.  `int(c)` used to be applied here, which accepts
    `5.0`, `True` and `"5"` -- a float that happens to be integral commits
    fine and then fails to open, and a `bool` is an `int` in Python but is
    never a coefficient anyone meant.
    """
    if isinstance(coeffs, (str, bytes)) or not hasattr(coeffs, "__len__"):
        raise ValueError(f"{where}: expected a sequence of {length} ints")
    if len(coeffs) != length:
        raise ValueError(
            f"{where}: {len(coeffs)} coefficients, expected {length}")
    out = []
    for i, c in enumerate(coeffs):
        if isinstance(c, bool) or not isinstance(c, int):
            raise ValueError(
                f"{where}[{i}]: coefficient {c!r} is "
                f"{type(c).__name__}, expected int")
        out.append(c)
    return out


class LanesBackend(ExactBackend):
    """`Pi_ex` instantiated with the ported LANES prover.  Zero knowledge."""

    name = "lanes"

    @classmethod
    def unavailable_reason(cls, ex):
        """Why this backend cannot run, or `None` if it can.

        Delegates to `exact.lanes_unavailable_reason`, which is where
        readiness is decided so that every `lanes_*` module can consult it
        without importing this one.  `ex` is accepted and ignored:
        readiness is a property of the frozen manifest, the recorded
        security evidence and the audit of live constants, not of one
        `ExactParams`.
        """
        del ex
        from exact import lanes_unavailable_reason
        return lanes_unavailable_reason()

    #: What an *experimental* instance calls itself.  A different name, not
    #: a flag: `scheme.exact.name` is what `vectors.py` records in a case,
    #: and `vectors.py` rebuilds a scheme by passing that string straight
    #: back to `exact.get_backend`.  An
    #: experimental run that called itself `"lanes"` would write vectors
    #: whose verification reconstructs the *gated* backend -- which refuses
    #: -- and benchmarks that attribute experimental widths to the paper.
    EXPERIMENTAL_NAME = "lanes-experimental"

    @classmethod
    def experimental(cls, par):
        """Construct **past** the readiness gate, for tests and benchmarks.

        The gate governs the supported name: `exact_backend="lanes"`
        refuses while `exact.LANES_BACKEND_READY` is false, and must go on
        go on refusing.  But a gate nothing can get behind means
        the implementation has no regression coverage at the current
        dimensions at all, which is how an unconstrained message-block
        padding survived to be found by inspection rather than by a test.

        So this is the one way through, and it is spelled out wherever it
        is used: the instance calls itself `"lanes-experimental"`, so every
        report, manifest and vector case that records a backend name
        records *that* one.  `exact.get_backend` accepts it and returns
        this constructor, so the name round-trips.  Whatever it proves is a
        statement about an **experimental** parameter set (see
        `lanes_params`), never about the paper's.
        """
        self = cls.__new__(cls)
        self._init(par)
        self.name = cls.EXPERIMENTAL_NAME
        return self

    def __init__(self, par):
        reason = self.unavailable_reason(ExactParams(par))
        if reason is not None:
            raise NotImplementedError(reason)
        self._init(par)

    def _init(self, par):
        self.par = par
        self.ex = ExactParams(par)
        # `floor(6 sigma_m)`, from the **exact** square, not
        # `ceil` of the float.
        #
        # `(6 sigma_m)^2 = 278313880780800` exactly at every profile, whose
        # square root is `16682742.0043...` — so the largest coefficient the
        # verifier's own test admits is 16682742, and `ceil` of the float
        # gave 16682743.  That is one unit *looser* than the accept/reject
        # bound: a `z_eval` coefficient at 16682743 would encode here and be
        # rejected by `oom.verify`, and `bound_z` also enters the statement
        # hash, so the two implementations disagreed on every LANES byte
        # until this matched `river-rs`'s exact `floor_sqrt`.
        #
        # This is the rule the repository states — no float reaches an
        # accept/reject decision, and every bound has the shape `K sqrt(M)`
        # so squaring removes the root — applied at the one place that had
        # slipped it.
        self.bound_z = floor_sqrt(par.zm_inf_bound_sq)

        #: Full commitment/proof elements pack uniformly at
        #: `ceil(log2 q~)` bits -- 26 at the current ring.  `t0` has its own
        #: high-part domain, `ceil(log2 T0_HIGH_MODULUS)` bits wide after
        #: dropping `D`; `c` and the recovery hint are signed ternary
        #: fields, and `z` is Rice-coded.  The widths are read off the
        #: constants rather than written out, because this backend is gated
        #: and its dimensions and its *bounds* no longer come from the same
        #: revision -- see `exact.lanes_unavailable_reason`.
        qt = Uniform(R.QTILDE)
        t0_high = Uniform(T0_HIGH_MODULUS)
        rice_z = Rice(SIGMA_Y, Z_INF_BOUND)
        #: `c` is the transmitted challenge: ternary, so two bits a
        #: coefficient.  It replaces `w`, `v` and `v'`, which the verifier
        #: recovers -- 256 bits against 33,408.  See `lanes_proof`.
        ternary = Signed(1)
        self.W_layout = Layout(
            Field("t0", t0_high, R.DTILDE, IDENTITY_RANK),
            Field("t", qt, R.DTILDE, N_EX),
        )
        self.proof_layout = Layout(
            Field("t0", t0_high, R.DTILDE, IDENTITY_RANK),
            Field("t", qt, R.DTILDE, N_EX),
            *[Field(name, qt, R.DTILDE) for name in self._ELEMENTS],
            Field("c", ternary, R.DTILDE, ring=R),
            Field("hint", ternary, R.DTILDE, IDENTITY_RANK),
            Field("z", rice_z, R.DTILDE, RESPONSE_RANK, ring=R),
        )

    # -- Pi_ex interface ---------------------------------------------------

    def setup(self, par, seed):
        return {"ex": ExactParams(par), "ck": LanesCommitmentKey(seed),
                "seed": seed}

    @staticmethod
    def _reject_non_integer(value, where):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"{where}: coefficient {value!r} is "
                f"{type(value).__name__}, expected int")
        return value

    def com(self, pp, witness_input, xof):
        """`Pi_ex.Com`: commit before the statement exists."""
        ex = pp["ex"]
        # Strict, not coercing.  `int(c)` accepted `5.0`, `True` and `"5"`
        # here, which conflicts with the coefficient policy every other
        # entry point in this tree holds to -- and a float coefficient that
        # silently truncates is a witness the commitment does not open to.
        e_eval = _exact_coeffs(witness_input["e_eval"], ex.d, "e_eval")
        y_eval = _exact_coeffs(witness_input["y_eval"], ex.d, "y_eval")
        digits = decompose_poly([c + ex.q0 // 2 for c in e_eval])  # {0,1,2}
        centred = [[a - 1 for a in poly] for poly in digits]   # -> {-1,0,1}

        # Each block is `l = 64` slots wide and carries `d = 32`
        # coefficients, so every one of the six is *padded* -- the
        # revision's `6 outer_d != N_ex l` (192 != 384) is intentional.
        # Assigning the coefficient list directly leaves a 32-entry block
        # that `scale_blocks` and the linear system both read as 64.
        def _block(coeffs):
            # exactly `d`, not "at most `l`": a short or long semantic
            # input would silently move coefficients into slots the linear
            # system constrains to zero, or leave real ones unconstrained.
            if len(coeffs) != ex.d:
                raise ValueError(
                    f"message block carries {len(coeffs)} coefficients, "
                    f"expected d = {ex.d}")
            return list(coeffs) + [0] * (R.LSPLIT - ex.d)

        slots = [[0] * R.LSPLIT for _ in range(N_EX)]
        slots[IDX_E] = _block(e_eval)
        slots[IDX_Y] = _block(y_eval)
        for j, poly in enumerate(centred):
            slots[IDX_DIGITS + j] = _block(poly)

        message = [[v % R.QTILDE for v in s] for s in slots]
        public, secret = commit(pp["ck"], message, xof)
        state = {"message": message, "slots": slots, "secret": secret,
                 "e_eval": e_eval, "y_eval": y_eval, "digits": digits,
                 "xof": xof}
        return public, state

    def prove(self, pp, statement, witness_input, state):
        ex = pp["ex"]
        ulp = build_linear_system(ex, statement["x_centered"],
                                  statement["z_eval_centered"])
        ch = lanes_proof.Challenges(
            statement_bytes(ex, self, statement["W"],
                            statement["x_centered"],
                            statement["z_eval_centered"]))
        return lanes_proof.prove(pp["ck"], statement["W"], state["secret"],
                                 state["message"], state["slots"], ulp,
                                 ALPHA_LO, ALPHA_HI, state["xof"], ch)

    def verify(self, pp, statement, proof):
        ex = pp["ex"]
        try:
            ulp = build_linear_system(ex, statement["x_centered"],
                                      statement["z_eval_centered"])
            ch = lanes_proof.Challenges(
                statement_bytes(ex, self, statement["W"],
                                statement["x_centered"],
                                statement["z_eval_centered"]))
            return lanes_proof.verify(pp["ck"], statement["W"], proof, ulp,
                                      ALPHA_LO, ALPHA_HI, ch)
        except (KeyError, TypeError, IndexError, ValueError):
            return False

    # -- encoding ----------------------------------------------------------

    #: The uniform ring elements `sigma_ex` still carries.  `w`, `v` and
    #: `v'` are gone: each is a check target the verifier recovers, which
    #: is what the revision's `(N_ex + alpha + 1) = 10`-element uniform
    #: term already assumes.  See `lanes_proof`'s module docstring.
    _ELEMENTS = ("t_g", "t_mp1", "t_mp2", "h")

    def W_encode(self, W):
        return self.W_layout.encode(W)

    def W_decode(self, data):
        return self.W_layout.decode(data)

    @property
    def W_bytes(self):
        return self.W_layout.max_bytes          # `W` is all uniform: exact

    def proof_encode(self, pi_ex):
        flat = dict(pi_ex["sigma"])
        flat["t0"] = pi_ex["W"]["t0"]
        flat["t"] = pi_ex["W"]["t"]
        return self.proof_layout.encode(flat)

    def proof_decode(self, data):
        flat = self.proof_layout.decode(data)
        W = {"t0": flat.pop("t0"), "t": flat.pop("t")}
        return {"W": W, "sigma": flat}

    @property
    def proof_bytes(self):
        """Worst-case `|pi_ex|`.

        Rice-coding `z` makes the real length sample-dependent, so this is an
        upper bound, not the size of any particular proof.  Measure with
        `len(backend.proof_encode(pi_ex))`.
        """
        return self.proof_layout.max_bytes

    # -- size accounting ---------------------------------------------------

    def field_sizes(self, pi_ex):
        """Per-field measurement of one encoded proof.

        The paper requires the implementation to record, for
        every transmitted field, its name, coefficient count, coefficient
        distribution, encoding method and measured size, rather than quoting a
        single modelled total.  This is that record, taken from the real
        serializer: each field is encoded on its own so the bits are measured,
        not predicted.

        `bits` is the field's **payload** — `Layout` pads to a byte boundary
        once, at the end of the whole proof, and that padding belongs to no
        field.  So the rows sum to the proof's payload bits, and
        `ceil(total / 8)` is its byte length; they do not sum to `8 *
        len(proof_encode(...))` unless the total is already a multiple of 8.
        `test_lanes.py` checks the `ceil` identity rather than an equality.

        `z`'s distribution is the **response** `y + c r`, not the mask: its
        per-coefficient variance is `VAR_Z = sigma_y^2 + w_hat sigma_r^2` and
        its coefficients are correlated through `c` (see `lanes_params`).  It
        is Rice-coded at the parameter `optimal_rice_k(sigma_y)`, which is a
        coder choice sized off the dominant term, not a claim about the law.
        """
        flat = dict(pi_ex["sigma"])
        flat["t0"] = pi_ex["W"]["t0"]
        flat["t"] = pi_ex["W"]["t"]
        rows = []
        for f in self.proof_layout.fields:
            rows.append({
                "name": f.name,
                "elements": 1 if f.rows is None else f.rows,
                "coeffs": f.count,
                "dist": ("response y+cr" if isinstance(f.coder, Rice)
                         else "recovery carry in {-1,0,1}" if f.name == "hint"
                         else "ternary, weight w_hat"
                         if isinstance(f.coder, Signed)
                         else f"D={D_DROP} high part" if f.name == "t0"
                         else "uniform mod q~"),
                "coder": (f"Rice(k={f.coder.k})" if isinstance(f.coder, Rice)
                          else f"signed({f.coder.width} bits)"
                          if isinstance(f.coder, Signed)
                          else f"uniform({f.coder.width} bits)"),
                "bits": _field_bits(f, flat[f.name]),
                "max_bits": f.count * f.coder.max_bits(),
            })
        return rows

    def model_bits(self):
        """A **historical draft's** closed form, for comparison only.

        `n~ d~ (log2 q~ - D) + (N_ex + alpha + 1) d~ log2 q~
         + k_L (l~ + N_ex + alpha) d~ h(sigma_y)`, with `h(s) =
        log2(4.13 s)` and `k_L = 1`.

        **Provenance matters here.**  This formula is *not* in the current
        manuscript, which gives a closed form for `L_OOM` and, for the
        exact proof, only the serialized figure "13.5 KB".  So whatever
        this returns is a **Derived** extrapolation and cannot be used to
        check the 13.5 KB claim.

        It is idealised besides -- the concrete format carries a fixed
        recovery hint and pays integer coder widths, and this charges
        `log2 q~ - D` bits for a `t0` high part that needs one more.  The
        measured `field_sizes` is the current result.
        """
        # The paper's formula charges the declared coefficient width
        # `ceil(log2 q~) = 26`, not the entropy `log2(q~)`.
        #
        # It also charges `log2 q~ - D` bits for a `t0` high part, which is
        # nine here and one short: `power2round` leaves the high part in
        # `[0, T0_HIGH_MODULUS)` with `T0_HIGH_MODULUS = 513`, so the
        # serializer writes ten.  Reproduced as printed rather than
        # corrected -- this is the model to compare *against*, and the
        # 1,024-bit gap is part of what the comparison is for.
        log_q = R.QTILDE.bit_length()
        h_y = math.log2(4.13 * float(SIGMA_Y))
        return {
            f"t0 (compressed, D={D_DROP})":
                IDENTITY_RANK * R.DTILDE * (log_q - D_DROP),
            "uniform (N_ex+alpha+1)": (N_EX + ALPHA + 1) * R.DTILDE * log_q,
            "z (rank kappa-l~)": RESPONSE_RANK * R.DTILDE * h_y,
        }


# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Gated on security evidence, not on parameters: the widths are the
    # paper's.  See `exact.lanes_unavailable_reason`,
    # `river-py/lanes_security.json`.
    from exact import skip_if_lanes_unavailable
    skip_if_lanes_unavailable("lanes_backend.py")

    import random
    from params import TOY_PARAMS
    from ring import negacyclic_mul_int
    from sample import XOF

    par = TOY_PARAMS
    backend = LanesBackend.experimental(par)
    pp = backend.setup(par, b"\x21" * 32)
    ex = pp["ex"]
    rng = random.Random(5)

    e_eval = [rng.randrange(par.q0) - par.B_e for _ in range(par.d)]
    y_eval = [rng.randrange(-10 ** 6, 10 ** 6) for _ in range(par.d)]
    x_c = [0] * par.d
    for pos in rng.sample(range(par.d), par.w):
        x_c[pos] = rng.choice([-1, 1]) * rng.randint(1, par.gamma)
    product = negacyclic_mul_int(x_c, e_eval)
    z_c = [product[i] + y_eval[i] for i in range(par.d)]

    w_in = {"e_eval": e_eval, "y_eval": y_eval}
    W, st = backend.com(pp, w_in, XOF(DS_EXACT, b"lanes-backend"))
    stmt = {"W": W, "z_eval_centered": z_c, "x_centered": x_c}
    sigma = backend.prove(pp, stmt, w_in, st)
    assert backend.verify(pp, stmt, sigma), "honest LANES proof rejected"

    # the digits really are ternary and reconstruct
    assert all(a in (-1, 0, 1) for poly in st["slots"][ALPHA_LO:ALPHA_HI]
               for a in poly)
    for i in range(par.d):
        acc = sum(w * st["slots"][ALPHA_LO + j][i]
                  for j, w in enumerate(RADIX_WEIGHTS))
        assert acc == e_eval[i]

    # the statement is bound
    bad = dict(stmt, z_eval_centered=list(z_c))
    bad["z_eval_centered"][0] += 1
    assert not backend.verify(pp, bad, sigma), "wrong statement accepted"
    bad2 = dict(stmt, x_centered=list(x_c))
    idx = next(i for i, v in enumerate(bad2["x_centered"]) if v)
    bad2["x_centered"][idx] = -bad2["x_centered"][idx]
    assert not backend.verify(pp, bad2, sigma), "wrong challenge accepted"

    # zero knowledge: the witness is not recoverable from the transmitted proof
    blob = backend.proof_encode({"W": W, "sigma": sigma})
    assert len(blob) <= backend.proof_bytes    # Rice: variable
    again = backend.proof_decode(blob)
    assert backend.verify(pp, dict(stmt, W=again["W"]), again["sigma"])
    assert "e_eval" not in again["sigma"] and "y_eval" not in again["sigma"]

    # the field-by-field record the paper asks for
    print(f"{'field':>8} {'elts':>5} {'coeffs':>7}  {'distribution':<18} "
          f"{'encoding':<18} {'bits':>7} {'worst':>7}")
    total = 0
    for row in backend.field_sizes({"W": W, "sigma": sigma}):
        total += row["bits"]
        print(f"{row['name']:>8} {row['elements']:>5} {row['coeffs']:>7}  "
              f"{row['dist']:<18} {row['coder']:<18} "
              f"{row['bits']:>7} {row['max_bits']:>7}")
    print(f"{'total':>8} {'':>5} {'':>7}  {'':<18} {'':<18} {total:>7}"
          f"   = {total / 8192:.3f} KB")
    model = backend.model_bits()
    for name, bits in model.items():
        print(f"  model  {name:<28} {bits:10.0f}")
    print(f"  model  {'total':<28} {sum(model.values()):10.0f}"
          f"   = {sum(model.values()) / 8192:.3f} KB")

    print(f"lanes_backend.py: all self-tests passed "
          f"(|pi_ex| = {len(blob) / 1024:.3f} KB, bound {backend.proof_bytes / 1024:.3f} KB)")
