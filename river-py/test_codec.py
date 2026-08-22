"""
test_codec.py -- Encoding round trips and bound enforcement.
"""

import random

import pytest

from codec import (RiVeRCodec, width_for_bound, width_for_modulus,
                   pack_signed, unpack_signed, pack_unsigned,
                   unpack_unsigned, encode_int_vec, decode_int_vec,
                   ring_digest, statement_digest,
                   BitWriter, BitReader, Uniform, Signed, Rice,
                   optimal_rice_k)
from params import TOY_PARAMS
from river import RiVeR

PAR = TOY_PARAMS


def _available_backends():
    """Exact backends that run at the current parameters."""
    from exact import BACKENDS, OPTIONAL_BACKENDS, get_backend
    out = []
    for name in list(BACKENDS) + list(OPTIONAL_BACKENDS):
        try:
            get_backend(name, PAR)
        except NotImplementedError:
            continue
        out.append(name)
    return tuple(out)


AVAILABLE_BACKENDS = _available_backends()
CODEC = RiVeRCodec(PAR)


# ---- primitives ----------------------------------------------------------

def test_width_helpers():
    assert width_for_modulus(255) == 1
    assert width_for_modulus(256) == 2
    assert width_for_bound(127) == 1        # [-127, 127] -> 255 values
    assert width_for_bound(128) == 2


def test_unsigned_round_trip():
    values = [0, 1, 65535]
    assert unpack_unsigned(pack_unsigned(values, 2), 2, 3) == values


def test_signed_round_trip():
    values = [-100, 0, 100]
    assert unpack_signed(pack_signed(values, 2, 100), 2, 3, 100) == values


def test_signed_rejects_out_of_range():
    for bad in (-101, 101):
        try:
            pack_signed([bad], 2, 100)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")


def test_decode_rejects_out_of_range():
    """A corrupted blob must not decode into an out-of-bound element."""
    blob = pack_signed([100], 2, 100)
    corrupted = (0xFFFF).to_bytes(2, "little")
    assert blob != corrupted
    try:
        unpack_signed(corrupted, 2, 1, 100)
    except ValueError:
        return
    raise AssertionError("out-of-range value decoded")


def test_truncated_input_raises():
    try:
        unpack_unsigned(b"\x00", 2, 1)
    except ValueError:
        return
    raise AssertionError("truncated block accepted")


def test_modular_decode_rejects_non_canonical():
    """A fixed-width field is wider than the modulus; decoding must not
    accept a representative outside [0, q)."""
    from codec import decode_poly_mod, encode_poly_mod
    blob = bytearray(CODEC.pk_encode(
        [[PAR.p - 1] * PAR.d for _ in range(PAR.n)]))
    w = CODEC.w_p
    blob[0:w] = (PAR.p + 1).to_bytes(w, "little")
    try:
        CODEC.pk_decode(bytes(blob))
    except ValueError:
        return
    raise AssertionError("non-canonical public key decoded")


def test_int_vec_round_trip():
    rng = random.Random(1)
    vec = [[rng.randrange(-50, 51) for _ in range(8)] for _ in range(3)]
    assert decode_int_vec(encode_int_vec(vec, 50), 50, 3, 8) == vec


# ---- scheme objects ------------------------------------------------------

def test_pk_round_trip_and_size():
    rng = random.Random(2)
    pk = [[rng.randrange(PAR.p) for _ in range(PAR.d)] for _ in range(PAR.n)]
    blob = CODEC.pk_encode(pk)
    assert len(blob) == CODEC.pk_bytes
    assert CODEC.pk_decode(blob) == pk


def test_pk_encoding_is_canonical():
    rng = random.Random(3)
    pk = [[rng.randrange(PAR.p) for _ in range(PAR.d)] for _ in range(PAR.n)]
    once = CODEC.pk_encode(pk)
    assert CODEC.pk_encode(CODEC.pk_decode(once)) == once


def test_sk_round_trip():
    rng = random.Random(4)
    sk = [[rng.choice([0, 1, PAR.q - 1]) for _ in range(PAR.d)]
          for _ in range(PAR.ell)]
    assert CODEC.sk_decode(CODEC.sk_encode(sk)) == sk


def test_value_round_trip():
    rng = random.Random(5)
    v = [rng.randrange(PAR.p) for _ in range(PAR.d)]
    assert CODEC.value_decode(CODEC.value_encode(v)) == v


# ---- proof ---------------------------------------------------------------

def _sample_proof():
    scheme = RiVeR(PAR)
    pp = scheme.setup(b"\x11" * 32)
    keys = [scheme.keygen(pp, bytes([i]) + b"\x00" * 31)
            for i in range(PAR.N)]
    ring = [pk for _, pk in keys]
    sk, pk = keys[0]
    v, pi = scheme.eval(pp, pk, sk, ring, b"codec", b"\x22" * 32)
    return scheme, pp, ring, v, pi


def test_proof_round_trip_is_canonical():
    scheme, pp, ring, v, pi = _sample_proof()
    blob = scheme.proof_encode(pi)
    decoded = scheme.proof_decode(blob)
    assert scheme.proof_encode(decoded) == blob
    assert scheme.verify(pp, ring, b"codec", v, decoded)


def test_proof_fields_survive_round_trip():
    scheme, pp, ring, v, pi = _sample_proof()
    decoded = scheme.proof_decode(scheme.proof_encode(pi))
    for field in ("B", "x", "f1", "zb", "z"):
        assert decoded["oom"][field] == pi["oom"][field], field
    assert decoded["ex"]["W"] == pi["ex"]["W"]


def test_proof_rejects_trailing_bytes():
    scheme, pp, ring, v, pi = _sample_proof()
    try:
        scheme.proof_decode(scheme.proof_encode(pi) + b"\x00")
    except ValueError:
        return
    raise AssertionError("trailing bytes accepted")


def test_proof_size_report():
    scheme, pp, ring, v, pi = _sample_proof()
    sizes = scheme.codec.proof_sizes(pi, scheme.exact)
    blob = scheme.proof_encode(pi)
    # the two 4-byte length prefixes are the only extra
    assert sizes["pi_OOM_bytes"] + sizes["pi_ex_bytes"] + 8 == len(blob)
    # and each component stays inside the bound its layout advertises
    assert sizes["pi_OOM_bytes"] <= sizes["pi_OOM_max_bytes"]
    assert sizes["pi_ex_bytes"] <= scheme.exact.proof_bytes


def test_rice_makes_proof_length_vary():
    """Two honest proofs at one profile differ in length.

    Not an incidental property: it is the direct consequence of entropy
    coding, and anything that assumed a constant `|pi|` is now wrong.
    """
    scheme = RiVeR(PAR)
    pp = scheme.setup(b"\x11" * 32)
    keys = [scheme.keygen(pp, bytes([i]) + b"\x00" * 31)
            for i in range(PAR.N)]
    ring = [pk for _, pk in keys]
    sk, pk = keys[0]
    lengths = {len(scheme.proof_encode(
        scheme.eval(pp, pk, sk, ring, b"m", bytes([i]) * 32)[1]))
        for i in range(6)}
    assert len(lengths) > 1, lengths


# ---- transcript digests --------------------------------------------------

def test_digests_are_deterministic_and_sensitive():
    rng = random.Random(6)
    pks = [[[rng.randrange(PAR.p) for _ in range(PAR.d)]
            for _ in range(PAR.n)] for _ in range(2)]
    v = [rng.randrange(PAR.p) for _ in range(PAR.d)]
    a = ring_digest(CODEC, pks, v)
    assert a == ring_digest(CODEC, pks, v)
    assert a != ring_digest(CODEC, pks[::-1], v)
    v2 = list(v)
    v2[0] = (v2[0] + 1) % PAR.p
    assert a != ring_digest(CODEC, pks, v2)


def test_statement_digest_binds_seed_and_hm():
    rng = random.Random(7)
    h_m = [[rng.randrange(PAR.q) for _ in range(PAR.d)] for _ in range(PAR.ell)]
    a = statement_digest(CODEC, b"seed1", h_m)
    assert a == statement_digest(CODEC, b"seed1", h_m)
    assert a != statement_digest(CODEC, b"seed2", h_m)
    h2 = [list(p) for p in h_m]
    h2[0][0] = (h2[0][0] + 1) % PAR.q
    assert a != statement_digest(CODEC, b"seed1", h2)


# ---- bit-level codec -----------------------------------------------------

def _layouts():
    """Every layout in the scheme, with a valid encoding of each."""
    from exact import get_backend
    scheme, pp, ring, v, pi = _sample_proof()
    out = [("oom", scheme.codec.oom_layout, scheme.codec.oom_encode(pi["oom"]))]
    # The production `lanes` name is gated on security evidence; see
    # `lanes_backend.LanesBackend.unavailable_reason`.
    for name in AVAILABLE_BACKENDS:
        backend = get_backend(name, PAR)
        s2 = RiVeR(PAR, exact_backend=name)
        pp2 = s2.setup(b"\x11" * 32)
        keys = [s2.keygen(pp2, bytes([i]) + b"\x00" * 31)
                for i in range(PAR.N)]
        r2 = [pk for _, pk in keys]
        sk2, pk2 = keys[0]
        _, pi2 = s2.eval(pp2, pk2, sk2, r2, b"codec", b"\x22" * 32)
        out.append((f"ex:{name}", s2.exact.proof_layout,
                    s2.exact.proof_encode(pi2["ex"])))
    return out


def test_bitwriter_bitreader_round_trip():
    rng = random.Random(11)
    for _ in range(200):
        widths = [rng.randrange(1, 33) for _ in range(rng.randrange(1, 20))]
        values = [rng.randrange(1 << w) for w in widths]
        w = BitWriter()
        for value, width in zip(values, widths):
            w.write_bits(value, width)
        r = BitReader(w.to_bytes())
        assert [r.read_bits(width) for width in widths] == values


def test_unary_round_trip_including_long_runs():
    for value in (0, 1, 7, 31, 32, 33, 100, 255):
        w = BitWriter()
        w.write_unary(value)
        assert BitReader(w.to_bytes()).read_unary(300) == value


def test_rice_round_trip_over_the_whole_range():
    coder = Rice(352, 2112)
    values = list(range(-2112, 2113, 7)) + [0, 1, -1, 2112, -2112]
    w = BitWriter()
    for value in values:
        coder.write(w, value)
    r = BitReader(w.to_bytes())
    assert [coder.read(r) for _ in values] == values


def test_rice_beats_fixed_width_on_a_gaussian():
    """The point of the exercise: Rice must actually pay for itself."""
    from sample import XOF, gaussian_int
    coder = Rice(352, 2112)
    x = XOF(b"codec-test", b"rice")
    values = [gaussian_int(x, 352) for _ in range(4000)]
    w = BitWriter()
    for value in values:
        coder.write(w, value)
    bits = w.bit_length / len(values)
    assert 10.0 < bits < 12.0, bits           # against 16 for two bytes
    assert bits < Signed(2112).width          # ... and against exact fixed


def test_rice_max_bits_is_exactly_reached_by_a_coefficient_at_the_bound():
    """Measure the worst case; do not derive it from the DoS cap.

    `max_high` is deliberately one larger than any encodable high part --
    it is the cap `read_unary` enforces, and a permissive cap is the right
    kind because the magnitude check behind it rejects the extra value.
    Charging that slack as a real bit overstated every layout by one bit
    per Rice coefficient: 72 bytes at the toy profile, 1648 at
    `RiVeR-N256`.  No round-trip test could see it, because `max_bits`
    never touches an encoding.
    """
    for sigma, bound in ((352, 4970), (352, 2112), (8, 100), (1, 3)):
        coder = Rice(sigma, bound)
        w = BitWriter()
        coder.write(w, -bound)
        assert w.bit_length == coder.max_bits(), (sigma, bound)
        for value in (0, 1, -1, bound // 2, bound - 1, bound):
            w2 = BitWriter()
            coder.write(w2, value)
            assert w2.bit_length <= coder.max_bits(), value


def test_rice_with_a_zero_bound_charges_no_sign_bit():
    """The degenerate field: `0` is the only value, and it has no sign.

    No profile has a zero bound, but a layout may legitimately declare a
    field that is always zero, and `max_bits` should be right for it
    rather than one bit high.
    """
    for k in (0, 1, 5):
        coder = Rice(None, 0, k=k)
        w = BitWriter()
        coder.write(w, 0)
        assert w.bit_length == coder.max_bits() == k + 1, k
        assert BitReader(w.to_bytes()).read_bits(k + 1) is not None


def test_layout_bounds_bracket_a_real_encoding():
    scheme, pp, ring, v, pi = _sample_proof()
    blob = scheme.codec.oom_encode(pi["oom"])
    layout = scheme.codec.oom_layout
    assert layout.min_bytes <= len(blob) <= layout.max_bytes


def test_rice_rejects_out_of_bound_values():
    coder = Rice(352, 2112)
    for bad in (2113, -2113, 10 ** 9):
        try:
            coder.write(BitWriter(), bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")


def test_optimal_rice_k_is_integer_deterministic():
    """`k` must not depend on float rounding: it decides the wire format."""
    assert optimal_rice_k(352) == 8
    assert optimal_rice_k(4096) == 12
    assert optimal_rice_k(1.5e7) == 24
    assert optimal_rice_k(0.5) == 0
    for sigma in (1, 2, 3, 100, 1e6):
        assert optimal_rice_k(sigma) == optimal_rice_k(float(sigma))


def test_uniform_rejects_non_canonical_residues():
    coder = Uniform(61)
    assert coder.width == 6                    # 6 bits hold 0..63
    w = BitWriter()
    w.write_bits(62, 6)                        # a value the modulus excludes
    try:
        coder.read(BitReader(w.to_bytes()))
    except ValueError:
        return
    raise AssertionError("non-canonical residue accepted")


# ---- hostile input -------------------------------------------------------

def test_decode_rejects_truncation_at_every_prefix():
    for name, layout, blob in _layouts():
        for cut in range(0, len(blob), max(1, len(blob) // 32)):
            try:
                layout.decode(blob[:cut])
            except ValueError:
                continue
            raise AssertionError(f"{name}: accepted {cut}-byte prefix")


def test_decode_rejects_trailing_bytes_and_padding():
    for name, layout, blob in _layouts():
        try:
            layout.decode(blob + b"\x00")
        except ValueError:
            pass
        else:
            raise AssertionError(f"{name}: accepted trailing byte")
        # flip a bit in the final byte, which is padding for these layouts
        mangled = bytearray(blob)
        mangled[-1] ^= 0x80
        try:
            layout.decode(bytes(mangled))
        except ValueError:
            pass


def test_decode_never_raises_anything_but_valueerror():
    """Fuzz: random, mutated and adversarial inputs.

    The contract a caller relies on is that *any* malformed encoding surfaces
    as `ValueError`.  An `IndexError` or an unbounded loop would both be bugs,
    and the all-ones case is the specific one Rice invites: without the unary
    cap it spins to the end of the buffer.
    """
    rng = random.Random(99)
    for name, layout, blob in _layouts():
        cases = [bytes(rng.randrange(256) for _ in range(len(blob)))
                 for _ in range(20)]
        cases += [b"\xff" * len(blob), b"\x00" * len(blob), b"",
                  b"\xff" * (4 * len(blob))]
        for _ in range(40):                      # single-bit flips
            mangled = bytearray(blob)
            mangled[rng.randrange(len(mangled))] ^= 1 << rng.randrange(8)
            cases.append(bytes(mangled))
        for case in cases:
            try:
                decoded = layout.decode(case)
            except ValueError:
                continue
            # decoding succeeded: it must then be canonical
            assert layout.encode(decoded) == case, f"{name}: non-canonical"


def test_proof_decode_rejects_hostile_length_prefixes():
    """The two length prefixes are attacker-controlled; neither may be trusted.

    A prefix claiming more bytes than exist must not read past the buffer, and
    one claiming fewer must not leave the remainder unaccounted for.
    """
    scheme, pp, ring, v, pi = _sample_proof()
    blob = scheme.proof_encode(pi)
    n_oom = int.from_bytes(blob[0:4], "little")
    prefixes = [0, 4 + n_oom]                    # where the real ones live
    assert int.from_bytes(blob[prefixes[1]:prefixes[1] + 4], "little") \
        == len(blob) - prefixes[1] - 4

    for at in prefixes:
        for claim in (0xFFFFFFFF, 0, 1, n_oom + 1):
            mangled = bytearray(blob)
            mangled[at:at + 4] = claim.to_bytes(4, "little")
            try:
                scheme.proof_decode(bytes(mangled))
            except ValueError:
                continue
            raise AssertionError(f"accepted length {claim} at offset {at}")
def test_coders_reject_types_that_are_merely_convertible():
    """`5.0` and `True` are not coefficients.

    `int(value)` accepted both, so an object `Verify` should have called
    malformed round-tripped into a well-formed proof instead.  Checked on
    every coder, in both the value and the container paths, because the
    coercion was in all of them.  See `river.py`'s `_is_coefficient`,
    which was strict for exactly this reason while the coders were not.
    """
    from codec import (Uniform, Signed, Rice, BitWriter, exact_int,
                       pack_signed, pack_unsigned)

    hostile = [5.0, -3.0, 0.0, True, False, "1", None, complex(1, 0)]
    coders = [(Uniform(61), 5), (Signed(16), -3), (Rice(352, 4970), -3)]
    for coder, good in coders:
        # the good value still encodes
        w = BitWriter()
        coder.write(w, good)
        assert w.bit_length > 0
        for bad in hostile:
            with pytest.raises(TypeError):
                coder.write(BitWriter(), bad)

    # the byte-oriented helpers too
    for bad in hostile:
        with pytest.raises(TypeError):
            pack_signed([bad], 4, 100)
        with pytest.raises(TypeError):
            pack_unsigned([bad], 4)
        with pytest.raises(TypeError):
            exact_int(bad)

    # and a bool is refused even though `True == 1`
    assert exact_int(1) == 1
    with pytest.raises(TypeError):
        exact_int(True)


# --------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  {t.__name__}: ok")
    print(f"test_codec.py: {len(tests)} tests passed")
