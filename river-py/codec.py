"""
codec.py -- Canonical byte encodings for RiVeR objects.

Two jobs:

  1. Serialise public keys, proofs and public parameters so they can be
     stored, transmitted and compared byte-for-byte across implementations.
  2. Provide the canonical transcript fed to the Fiat-Shamir hash `H`.

Everything goes through one bit-level codec.  A `Layout` is an ordered list of
`Field`s; each field names a coder, and the same list drives both directions,
so an encoder and a decoder cannot drift apart.  Three coders cover every
quantity in the scheme:

  `Uniform(modulus)`  values in `[0, modulus)`, at exactly
                      `(modulus - 1).bit_length()` bits -- commitments, high
                      bits, ring elements.
  `Signed(bound)`     centred values with `|v| <= bound`, offset-encoded at
                      exactly `(2 bound).bit_length()` bits -- challenges,
                      ternary randomness, anything uniform on its range.
  `Rice(sigma, ...)`  centred discrete-Gaussian values, Golomb-Rice coded --
                      every masked response, in both the OOM and exact layers.

Rice is what the paper's size estimates assume: it charges
`h(sigma) = log2(4.13 sigma)` bits per Gaussian coefficient, and Rice lands
within about half a bit of that, against the 32 bits a fixed-width field
would spend on `z`.

**Proof size is therefore data-dependent.**  A Gaussian coefficient far from
zero costs more bits than one near it, so two honest proofs at the same
profile differ in length.  `Layout.max_bytes` is the worst case -- every
coefficient at its bound -- and is what a fixed buffer must be sized to;
`proof_sizes()` reports measured lengths.

Robustness
----------
Decoding is hostile-input safe.  Every read checks: truncation, a value
outside its declared range, a unary run longer than the bound permits, a
non-canonical residue `>= modulus`, nonzero padding bits, and trailing bytes.
All of them raise `ValueError`, so a caller can treat any malformed proof
uniformly.  `test_codec.py` fuzzes random and mutated byte strings against
every layout to check that nothing else escapes.

Bit order is least-significant-first within each byte, as pinned by the
coder KATs and complete proof vectors.
"""

import math

from ring import Ring
from sample import rational_sigma


# ---- bit-level I/O -------------------------------------------------------

class BitWriter:
    """Accumulates bits, least-significant-first within each byte."""

    __slots__ = ("_out", "_pend", "_nbits")

    def __init__(self):
        self._out = bytearray()
        self._pend = 0          # bits not yet flushed, LSB-aligned
        self._nbits = 0

    def write_bits(self, value, n):
        if n == 0:
            return
        self._pend |= (value & ((1 << n) - 1)) << self._nbits
        self._nbits += n
        while self._nbits >= 8:
            self._out.append(self._pend & 0xFF)
            self._pend >>= 8
            self._nbits -= 8

    def write_unary(self, value):
        """`value` one-bits then a zero-bit.  Written in chunks so a long run
        does not cost one Python-level call per bit."""
        while value >= 32:
            self.write_bits(0xFFFFFFFF, 32)
            value -= 32
        self.write_bits((1 << value) - 1, value)
        self.write_bits(0, 1)

    def to_bytes(self):
        """Flush, zero-padding to the next byte boundary."""
        out = bytearray(self._out)
        if self._nbits:
            out.append(self._pend & 0xFF)
        return bytes(out)

    @property
    def bit_length(self):
        return 8 * len(self._out) + self._nbits


class BitReader:
    """Reads bits least-significant-first, rejecting every malformed input."""

    __slots__ = ("data", "_byte", "_pend", "_nbits", "pos")

    def __init__(self, data):
        self.data = data
        self._byte = 0          # next source byte
        self._pend = 0
        self._nbits = 0
        self.pos = 0            # bits consumed

    def read_bits(self, n):
        if n == 0:
            return 0
        while self._nbits < n:
            if self._byte >= len(self.data):
                raise ValueError("truncated: ran out of bits")
            self._pend |= self.data[self._byte] << self._nbits
            self._byte += 1
            self._nbits += 8
        value = self._pend & ((1 << n) - 1)
        self._pend >>= n
        self._nbits -= n
        self.pos += n
        return value

    def read_unary(self, max_value):
        """Read a unary run, rejecting one longer than `max_value`.

        Without the cap, a crafted input of all-ones bytes makes the decoder
        spin until it exhausts the buffer; the cap turns that into an
        immediate `ValueError`.
        """
        count = 0
        while self.read_bits(1):
            count += 1
            if count > max_value:
                raise ValueError(
                    f"unary run exceeds {max_value}: malformed input")
        return count

    def finish(self):
        """Require zero padding to the byte boundary and no trailing bytes."""
        while self.pos & 7:
            if self.read_bits(1):
                raise ValueError("nonzero padding bits")
        consumed = self.pos >> 3
        if consumed != len(self.data):
            raise ValueError(f"{len(self.data) - consumed} trailing bytes")


# ---- coders --------------------------------------------------------------

def floor_sqrt(value):
    """`floor(sqrt(value))` for a non-negative int or `Fraction`, exactly.

    `sqrt` is monotone and `k` is an integer, so `k <= sqrt(x)` iff
    `k^2 <= x` iff `k^2 <= floor(x)` -- which is why taking the floor first
    is not an approximation.  Used to turn a verifier bound of the form
    `K sqrt(M)` into the largest coefficient that can pass it.
    """
    if value < 0:
        raise ValueError("floor_sqrt of a negative value")
    return math.isqrt(math.floor(value))


#: `sqrt(2 ln 2)`, to 30 significant figures, as an exact rational.
#:
#: The Rice parameter is `k = floor(log2(sqrt(2 ln 2) sigma))`, and `k` is
#: wire-visible: one off is a different encoding, not a rounding difference.
#: The constant is irrational, so *some* rational stands in for it; what
#: matters is that the standing-in never moves `k`.
#:
#: This used to be `11774/10000`, a relative error of `8.5e-6` -- fine in
#: practice and unchecked in principle, since `k` moves whenever the true
#: and approximate products straddle a power of two.  Thirty digits makes
#: that window `1e-30` wide, and
#: `test_codec.py::test_rice_parameters_are_far_from_a_boundary` measures
#: the actual distance to the nearest boundary at every field of every
#: profile, so the margin is a checked property rather than a hope.
RICE_CONST_NUM = 1177410022515474691011569326460
RICE_CONST_DEN = 10 ** 30


def optimal_rice_k(sigma):
    """Rice parameter for a discrete Gaussian of width `sigma`.

    The classical choice `k = floor(log2(sqrt(2 ln 2) * sigma))`, evaluated
    in integers.  `sigma` is pinned to an exact rational first -- the same
    pinning the sampler uses -- so no float rounding can make two
    implementations pick different `k` and produce incompatible bytes.
    """
    num, den = sigma if isinstance(sigma, tuple) else rational_sigma(sigma)
    # floor(sqrt(2 ln 2) * sigma), in integers.
    scaled = (RICE_CONST_NUM * int(num)) // (RICE_CONST_DEN * int(den))
    return max(0, scaled.bit_length() - 1)


def exact_int(value, what="value"):
    """`value` as an `int`, refusing anything that is merely convertible.

    Coercing with `int(value)` is what made `5.0` and `True` encodable: the
    encoder would canonicalise them, so an object that `Verify` should have
    called malformed round-tripped into a well-formed proof instead.
    `river.py::_is_coefficient` is strict for exactly this reason and says so;
    the coders were not, which left the strictness true of the fields
    `Verify` checks by hand and false of everything inside a proof.

    `bool` is excluded because it is an `int` subclass and `True` is not a
    coefficient.  Nothing in `river-rs` needs this: `i64` is `i64`.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{what} must be an int, got {type(value).__name__}")
    return value


class Uniform:
    """Values in `[0, modulus)`, at exactly `(modulus - 1).bit_length()` bits."""

    __slots__ = ("modulus", "width")

    def __init__(self, modulus):
        self.modulus = int(modulus)
        if self.modulus < 1:
            raise ValueError("modulus must be positive")
        self.width = max(1, (self.modulus - 1).bit_length())

    def write(self, w, value):
        value = exact_int(value, "coefficient")
        if not 0 <= value < self.modulus:
            raise ValueError(f"{value} outside [0, {self.modulus})")
        w.write_bits(value, self.width)

    def read(self, r):
        value = r.read_bits(self.width)
        if value >= self.modulus:
            raise ValueError(f"non-canonical value {value} >= {self.modulus}")
        return value

    def max_bits(self):
        return self.width


class Signed:
    """Centred values with `|v| <= bound`, offset-encoded at a fixed width."""

    __slots__ = ("bound", "width")

    def __init__(self, bound):
        self.bound = int(bound)
        if self.bound < 0:
            raise ValueError("bound must be non-negative")
        self.width = max(1, (2 * self.bound).bit_length())

    def write(self, w, value):
        value = exact_int(value, "coefficient")
        if not -self.bound <= value <= self.bound:
            raise ValueError(f"{value} outside [-{self.bound}, {self.bound}]")
        w.write_bits(value + self.bound, self.width)

    def read(self, r):
        value = r.read_bits(self.width) - self.bound
        if not -self.bound <= value <= self.bound:
            raise ValueError(f"decoded {value} outside bound {self.bound}")
        return value

    def max_bits(self):
        return self.width


class Rice:
    """Golomb-Rice coder for centred discrete-Gaussian values.

    Each value `c` with `|c| <= bound` becomes

        low  = |c| & (2^k - 1)      k bits
        high = |c| >> k             unary
        sign                        1 bit, omitted when c == 0

    Omitting the sign of zero is what keeps the encoding canonical: `-0` and
    `+0` are the same integer and must have the same image, or `decode` after
    `encode` would not be the identity on bytes.
    """

    __slots__ = ("k", "bound", "max_high")

    def __init__(self, sigma, bound, k=None):
        self.k = optimal_rice_k(sigma) if k is None else int(k)
        self.bound = int(bound)
        self.max_high = (self.bound >> self.k) + 1

    def write(self, w, value):
        value = exact_int(value, "coefficient")
        magnitude = abs(value)
        if magnitude > self.bound:
            raise ValueError(f"|{value}| exceeds bound {self.bound}")
        w.write_bits(magnitude & ((1 << self.k) - 1), self.k)
        w.write_unary(magnitude >> self.k)
        if magnitude:
            w.write_bits(1 if value < 0 else 0, 1)

    def read(self, r):
        low = r.read_bits(self.k)
        high = r.read_unary(self.max_high)
        magnitude = (high << self.k) | low
        if magnitude > self.bound:
            raise ValueError(f"decoded |{magnitude}| exceeds {self.bound}")
        if magnitude == 0:
            return 0
        return -magnitude if r.read_bits(1) else magnitude

    def max_bits(self):
        """Worst case: a coefficient sitting exactly at the bound.

        `k` low bits, then the unary of `bound >> k`, which costs one bit
        per unit *plus* the terminating zero, then the sign.

        `max_high` is deliberately one larger than any encodable high part
        -- it is the cap `read_unary` enforces, and a permissive cap is
        the right kind, since the magnitude check behind it rejects the
        extra value anyway.  Reusing it here charged that slack as a real
        bit and overstated every layout by one bit per Rice coefficient:
        620 bytes at `RiVeR-N8`, 1648 at `RiVeR-N256`.  Only size metadata
        and the framing bound move; no encoded byte does.

        The sign bit is charged only when some encodable value is nonzero.
        At `bound == 0` the sole value is `0`, whose sign is omitted to
        keep the encoding canonical, so the worst case is `k + 1`.  No
        profile has a zero bound, but a degenerate field is a legitimate
        layout and the arithmetic should be right for it.
        """
        sign = 1 if self.bound else 0
        return self.k + (self.bound >> self.k) + 1 + sign


# ---- layouts -------------------------------------------------------------

class Field:
    """One named entry of a `Layout`.

    `rows = None` means the value is a flat list of `cols` integers; otherwise
    it is `rows` lists of `cols`.  `ring`, when given, means the value holds
    residues of that ring: they are centred on the way out and lifted back on
    the way in, so the coder only ever sees small signed integers.
    """

    __slots__ = ("name", "coder", "rows", "cols", "ring")

    def __init__(self, name, coder, cols, rows=None, ring=None):
        self.name = name
        self.coder = coder
        self.rows = rows
        self.cols = cols
        self.ring = ring

    @property
    def count(self):
        return self.cols * (1 if self.rows is None else self.rows)


class Layout:
    """An ordered field list that drives encoding and decoding alike."""

    def __init__(self, *fields):
        self.fields = fields
        names = [f.name for f in fields]
        if len(set(names)) != len(names):
            raise ValueError("duplicate field name in layout")

    def encode(self, obj):
        w = BitWriter()
        for f in self.fields:
            value = obj[f.name]
            rows = [value] if f.rows is None else value
            if f.rows is not None and len(rows) != f.rows:
                raise ValueError(f"{f.name}: expected {f.rows} rows")
            for row in rows:
                if f.ring is not None:
                    # `Ring.centered` assumes its input is already in [0, q):
                    # given `c + q` it happily returns `c`, so a non-canonical
                    # residue would encode as the canonical one and verify.
                    # Reject instead, matching what the byte decoder enforces.
                    for value_i in row:
                        if not 0 <= exact_int(value_i) < f.ring.q:
                            raise ValueError(
                                f"{f.name}: non-canonical residue {value_i}")
                    row = f.ring.centered(row)
                if len(row) != f.cols:
                    raise ValueError(
                        f"{f.name}: expected {f.cols} values, got {len(row)}")
                for value_i in row:
                    f.coder.write(w, value_i)
        return w.to_bytes()

    def decode(self, data):
        r = BitReader(data)
        out = {}
        for f in self.fields:
            rows = []
            for _ in range(1 if f.rows is None else f.rows):
                row = [f.coder.read(r) for _ in range(f.cols)]
                if f.ring is not None:
                    row = f.ring.from_centered(row)
                rows.append(row)
            out[f.name] = rows[0] if f.rows is None else rows
        r.finish()
        return out

    @property
    def max_bytes(self):
        """Upper bound on the encoding: every value at its worst case."""
        bits = sum(f.count * f.coder.max_bits() for f in self.fields)
        return (bits + 7) // 8

    @property
    def min_bytes(self):
        """Lower bound, reached when every Rice-coded value is zero."""
        bits = 0
        for f in self.fields:
            if isinstance(f.coder, Rice):
                bits += f.count * (f.coder.k + 1)
            else:
                bits += f.count * f.coder.max_bits()
        return (bits + 7) // 8


# ---- primitive packing ---------------------------------------------------

def width_for_modulus(modulus):
    """Bytes needed for an unsigned value in [0, modulus)."""
    return (modulus.bit_length() + 7) // 8


def width_for_bound(bound):
    """Bytes needed for a signed value in [-bound, bound]."""
    return ((2 * int(bound) + 1).bit_length() + 7) // 8


def pack_unsigned(values, width):
    return b"".join(exact_int(v).to_bytes(width, "little") for v in values)


def unpack_unsigned(data, width, count, modulus=None, exact=True):
    """Decode `count` unsigned values of `width` bytes.

    When `modulus` is given, values outside `[0, modulus)` are rejected.  A
    fixed-width field is wider than the modulus, so without this a decoder
    accepts non-canonical representatives: they re-encode to different bytes
    and break the "decode then encode is the identity" property that a wire
    format needs.

    `exact` requires the input to be exactly `width * count` bytes.  Checking
    only for truncation let `pk_decode(valid + b"JUNK")` succeed and return the
    original key, so two distinct byte strings decoded to one object -- the
    same malleability the modulus check above exists to prevent.
    """
    if exact and len(data) != width * count:
        raise ValueError(
            f"expected {width * count} bytes, got {len(data)}")
    if len(data) < width * count:
        raise ValueError("truncated unsigned block")
    out = [int.from_bytes(data[i * width:(i + 1) * width], "little")
           for i in range(count)]
    if modulus is not None:
        for value in out:
            if value >= modulus:
                raise ValueError(
                    f"non-canonical coefficient {value} >= {modulus}")
    return out


def pack_signed(values, width, bound):
    out = []
    for v in values:
        v = exact_int(v)
        if not -bound <= v <= bound:
            raise ValueError(f"value {v} outside [-{bound}, {bound}]")
        out.append((v + bound).to_bytes(width, "little"))
    return b"".join(out)


def unpack_signed(data, width, count, bound, exact=True):
    """Decode `count` signed values; see `unpack_unsigned` on `exact`."""
    if exact and len(data) != width * count:
        raise ValueError(
            f"expected {width * count} bytes, got {len(data)}")
    if len(data) < width * count:
        raise ValueError("truncated signed block")
    out = []
    for i in range(count):
        v = int.from_bytes(data[i * width:(i + 1) * width], "little") - bound
        if not -bound <= v <= bound:
            raise ValueError(f"decoded value {v} outside [-{bound}, {bound}]")
        out.append(v)
    return out


# ---- polynomial / vector helpers -----------------------------------------

def encode_poly_mod(ring, poly, width):
    return pack_unsigned(ring.reduce(poly), width)


def decode_poly_mod(ring, data, width):
    return unpack_unsigned(data, width, ring.d, ring.q)


def encode_vec_mod(ring, vec, width):
    return b"".join(encode_poly_mod(ring, p, width) for p in vec)


def decode_vec_mod(ring, data, width, length):
    step = width * ring.d
    if len(data) != step * length:
        raise ValueError(f"expected {step * length} bytes, got {len(data)}")
    return [decode_poly_mod(ring, data[i * step:(i + 1) * step], width)
            for i in range(length)]


def encode_vec_signed(ring, vec, bound):
    """Encode a vector by its centred representatives."""
    width = width_for_bound(bound)
    flat = [c for p in vec for c in ring.centered(p)]
    return pack_signed(flat, width, int(bound))


def decode_vec_signed(ring, data, bound, length):
    width = width_for_bound(bound)
    flat = unpack_signed(data, width, length * ring.d, int(bound))
    return [ring.from_centered(flat[i * ring.d:(i + 1) * ring.d])
            for i in range(length)]


def encode_int_vec(vec, bound):
    """Encode a list of plain integer lists (already centred)."""
    width = width_for_bound(bound)
    flat = [c for row in vec for c in row]
    return pack_signed(flat, width, int(bound))


def decode_int_vec(data, bound, rows, cols):
    width = width_for_bound(bound)
    flat = unpack_signed(data, width, rows * cols, int(bound))
    return [flat[i * cols:(i + 1) * cols] for i in range(rows)]


# ---- the RiVeR codec -----------------------------------------------------

class RiVeRCodec:
    """Encoders and decoders bound to one parameter profile."""

    def __init__(self, par):
        self.par = par
        self.Rq = Ring(par.q, par.d)
        self.Rp = Ring(par.p, par.d)
        self.Rqhat = Ring(par.q_hat, par.d)

        # field widths, all derived from the profile
        self.w_q = width_for_modulus(par.q)
        self.w_p = width_for_modulus(par.p)
        self.w_qhat = width_for_modulus(par.q_hat)

        #: High bits of the selector commitment `B`.  `[[.]]_K` is taken on
        #: the centred representative, so
        #: these are signed and about half of them are negative.
        from oom import high_bits_bound
        self.bound_b_hi = high_bits_bound(par.q_hat, par.K_b)
        self.w_b_hi = width_for_bound(self.bound_b_hi)

        # Response bounds: exactly the ones the verifier enforces, and
        # exactly -- `floor_sqrt` of the exact squared bound is the largest
        # integer that can pass, so the encoder's cap and the acceptance
        # test agree by construction rather than by a `ceil` on a float
        # that could sit either side of it.
        self.bound_f1 = floor_sqrt(par.f1_inf_bound_sq)
        self.bound_zb = floor_sqrt(par.zb_inf_bound_sq)
        self.bound_zs = floor_sqrt(par.zs_inf_bound_sq)
        self.bound_zm = floor_sqrt(par.zm_inf_bound_sq)
        self.bound_x = par.gamma

        #: `pi_OOM = (B, x, f_1, z_b, z)`.  The masked responses are Gaussian
        #: and go through Rice; `B` and `x` are uniform on their ranges and go
        #: through fixed width.
        #:
        #: `z` is split on the wire.  The paper gives its two blocks
        #: different widths -- `(z_s, z_key)` at `sigma_s`, `z_eval` at
        #: `sigma_m`, and `sigma_s / sigma_m` is between 6.9 and 12.6 across
        #: the profiles --
        #: so one Rice parameter for the whole vector would cost roughly a bit
        #: per coefficient on whichever block it did not fit.  They are two
        #: fields with their own parameters and their own bounds; `z` itself
        #: is reassembled by `oom_decode`, because that is what the protocol
        #: and the verifier's Euclidean check operate on.  Both are genuine
        #: `R_q` vectors, so they carry the ring and are centred in transit.
        self.oom_layout = Layout(
            Field("B", Signed(self.bound_b_hi), par.d, par.n_hat),
            Field("x", Signed(self.bound_x), par.d),
            Field("f1", Rice(par.sigma_a, self.bound_f1), par.d, par.N - 1),
            Field("zb", Rice(par.sigma_b, self.bound_zb),
                  par.d, par.k_hat),
            Field("zs", Rice(par.sigma_s, self.bound_zs), par.d, par.s_dim,
                  ring=self.Rq),
            Field("zm", Rice(par.sigma_m, self.bound_zm), par.d, par.m_dim,
                  ring=self.Rq),
        )

    # -- introspection, for the numeric manifest ---------------------------
    #
    # These read the parameters the layout *actually* uses rather than
    # recomputing them, which is the point: `manifest.py` freezes what the
    # encoder does, so a divergence between the two would defeat it.

    def oom_field(self, name):
        """The `Field` named `name` in the OOM layout."""
        for field in self.oom_layout.fields:
            if field.name == name:
                return field
        raise KeyError(f"no OOM field named {name!r}; "
                       f"have {[f.name for f in self.oom_layout.fields]}")

    def oom_layout_k(self, name):
        """The Rice parameter the OOM layout uses for `name`."""
        coder = self.oom_field(name).coder
        if not isinstance(coder, Rice):
            raise TypeError(f"field {name!r} is not Rice-coded")
        return coder.k

    def oom_layout_width(self, name):
        """The fixed width in bits of a `Uniform` or `Signed` OOM field."""
        coder = self.oom_field(name).coder
        if isinstance(coder, Rice):
            raise TypeError(f"field {name!r} is variable-length")
        return coder.width

    @staticmethod
    def rice_k_for(sigma):
        """The Rice parameter a field of width `sigma` would get."""
        return optimal_rice_k(sigma)

    # -- public key --------------------------------------------------------

    def pk_encode(self, pk):
        """`t` in R_p^n."""
        return encode_vec_mod(self.Rp, pk, self.w_p)

    def pk_decode(self, data):
        return decode_vec_mod(self.Rp, data, self.w_p, self.par.n)

    @property
    def pk_bytes(self):
        return self.w_p * self.par.d * self.par.n

    # -- secret key --------------------------------------------------------

    def sk_encode(self, sk):
        """`s` in S_beta^ell, one signed byte per coefficient."""
        return encode_vec_signed(self.Rq, sk, self.par.beta)

    def sk_decode(self, data):
        return decode_vec_signed(self.Rq, data, self.par.beta, self.par.ell)

    # -- VRF value ---------------------------------------------------------

    def value_encode(self, v):
        return encode_poly_mod(self.Rp, v, self.w_p)

    def value_decode(self, data):
        return decode_poly_mod(self.Rp, data, self.w_p)

    # -- challenge ---------------------------------------------------------

    def challenge_encode(self, x):
        return encode_int_vec([x], self.bound_x)

    def challenge_decode(self, data):
        return decode_int_vec(data, self.bound_x, 1, self.par.d)[0]

    # -- OOM proof ---------------------------------------------------------
    # `B`, `x`, `f_1` and `z_b` are carried as *integer* polynomials (the
    # selector layer needs their exact integer values), so they encode and
    # decode as signed integers. `(z_s, z_key)` and `z_eval` are genuine
    # R_q elements.

    def oom_encode(self, pi):
        """pi_OOM = (B, x, f_1, z_b, z), with `z` split for the wire."""
        par = self.par
        z = pi["z"]
        if len(z) != par.r_dim:
            raise ValueError(f"z: expected {par.r_dim} rows, got {len(z)}")
        obj = {k: v for k, v in pi.items() if k != "z"}
        obj["zs"] = z[:par.s_dim]
        obj["zm"] = z[par.s_dim:]
        return self.oom_layout.encode(obj)

    def oom_decode(self, data):
        obj = self.oom_layout.decode(data)
        obj["z"] = obj.pop("zs") + obj.pop("zm")
        return obj

    @property
    def oom_max_bytes(self):
        """Worst-case `|pi_OOM|`; the actual length varies with the Gaussians."""
        return self.oom_layout.max_bytes

    # -- full proof --------------------------------------------------------
    # Both components are length-prefixed.  With Rice coding neither block has
    # a length the reader can compute in advance, and a self-delimiting format
    # that guesses would be one more thing an attacker could steer.

    def proof_encode(self, pi, exact_backend):
        oom = self.oom_encode(pi["oom"])
        ex = exact_backend.proof_encode(pi["ex"])
        return (len(oom).to_bytes(4, "little") + oom
                + len(ex).to_bytes(4, "little") + ex)

    def proof_decode(self, data, exact_backend):
        def take(off, layout):
            """Read one length-prefixed block, bounded by its own layout.

            The prefix is attacker-controlled, so it is checked against what
            the profile can actually produce before any slicing: a claim of
            `0xFFFFFFFF` must not become a 4 GB allocation attempt, and a
            claim below `min_bytes` cannot be a well-formed block either.
            """
            if len(data) < off + 4:
                raise ValueError("truncated proof: missing length prefix")
            n = int.from_bytes(data[off:off + 4], "little")
            if not layout.min_bytes <= n <= layout.max_bytes:
                raise ValueError(
                    f"block length {n} outside [{layout.min_bytes}, "
                    f"{layout.max_bytes}] for this profile")
            if len(data) < off + 4 + n:
                raise ValueError("truncated proof: block shorter than prefix")
            return data[off + 4:off + 4 + n], off + 4 + n

        oom_bytes, off = take(0, self.oom_layout)
        ex_bytes, off = take(off, exact_backend.proof_layout)
        if off != len(data):
            raise ValueError(f"trailing bytes in proof: {len(data) - off}")
        return {"oom": self.oom_decode(oom_bytes),
                "ex": exact_backend.proof_decode(ex_bytes)}

    # -- sizes -------------------------------------------------------------

    def proof_sizes(self, pi=None, exact_backend=None):
        """Measured sizes next to the paper's entropy-coded estimate.

        Rice coding makes the length depend on the sample, so this needs a
        proof to measure rather than deriving a constant from the profile.
        `oom_max_bytes` is the worst case if a bound is what you need.

        **Every size here is payload, not wire.**  `pi_OOM_bytes` and
        `pi_ex_bytes` are the two encoded blocks, and `pi_RiVeR_KB` is their
        sum; `proof_encode` frames them with a 4-byte little-endian length
        prefix each, so what a peer receives is 8 bytes larger.  The
        distinction is deliberate rather than an oversight: these columns
        are compared against the paper's communication model, which has no
        framing in it, while `vectors.json`'s `proof.byte_length` records
        what goes on the wire.  `river-rs/tests/vectors.rs` asserts the
        relation between the two so the gap cannot read as a discrepancy.
        """
        out = {
            "pi_OOM_paper_KB": self.par.proof_size_oom_kb,
            "pi_OOM_max_bytes": self.oom_max_bytes,
            "pk_bytes": self.pk_bytes,
        }
        if pi is not None:
            oom = len(self.oom_encode(pi["oom"]))
            out["pi_OOM_bytes"] = oom
            out["pi_OOM_KB"] = oom / 1024
            if exact_backend is not None:
                ex = len(exact_backend.proof_encode(pi["ex"]))
                out["pi_ex_bytes"] = ex
                out["pi_ex_KB"] = ex / 1024
                out["pi_RiVeR_KB"] = (oom + ex) / 1024
        return out


# ---- Fiat-Shamir transcript ----------------------------------------------

def ring_digest(codec, ring_pks, value):
    """Digest of `rho' = (R~, v, W)`'s ring/value part.

    `W` is appended separately by the caller because its encoding belongs to
    whichever exact backend is in use.
    """
    from sample import hash_bytes, DS_CHALLENGE
    parts = [codec.pk_encode(t) for t in ring_pks]
    parts.append(codec.value_encode(value))
    return hash_bytes(32, DS_CHALLENGE + b".rho", *parts)


def statement_digest(codec, seed, h_m):
    """Digest standing for `ck_{r,m}` in the Fiat-Shamir input.

    `ck_{r,m} = [A | -I_n | 0 ; h_m^T | 0 | -1]` and `A = SamMat(rho, ...)`,
    so the pair `(rho, h_m)` determines the whole matrix.  Hashing that pair
    is therefore an injective stand-in for hashing the matrix, and avoids
    serialising `n x ell` ring elements on every call.
    """
    from sample import hash_bytes, DS_CHALLENGE
    parts = [seed] + [pack_unsigned(codec.Rq.reduce(p), codec.w_q)
                      for p in h_m]
    return hash_bytes(32, DS_CHALLENGE + b".ck", *parts)


# --------------------------------------------------------------------------
if __name__ == "__main__":
    import random
    from params import TOY_PARAMS

    par = TOY_PARAMS
    codec = RiVeRCodec(par)
    rng = random.Random(7)

    # signed round trip, including bound rejection
    assert unpack_signed(pack_signed([-5, 0, 5], 2, 5), 2, 3, 5) == [-5, 0, 5]
    try:
        pack_signed([6], 2, 5)
        raise SystemExit("bound not enforced")
    except ValueError:
        pass

    # public key round trip
    pk = [[rng.randrange(par.p) for _ in range(par.d)] for _ in range(par.n)]
    assert codec.pk_decode(codec.pk_encode(pk)) == pk
    assert len(codec.pk_encode(pk)) == codec.pk_bytes

    # secret key round trip
    sk = [[rng.choice([0, 1, par.q - 1]) for _ in range(par.d)]
          for _ in range(par.ell)]
    assert codec.sk_decode(codec.sk_encode(sk)) == sk

    print("codec.py: all self-tests passed")
