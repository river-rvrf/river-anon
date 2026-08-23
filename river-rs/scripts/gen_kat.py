#!/usr/bin/env python3
"""gen_kat.py -- generate the cross-language KAT file for river-rs.

`vectors.json` pins whole executions, which is the acceptance test and the
worst thing to debug: one wrong byte anywhere moves every field after it and
the diff says only "proof bytes differ".  This pins the primitives in
dependency order instead -- XOF, samplers, acceptance thresholds -- so a
failing port can bisect and the first failing case names the layer.

Run from this directory:

    python3 scripts/gen_kat.py --out tests/sampler_kat.json

Every value comes from `../river-py`, so these are consistency anchors
between two implementations, not independent validation.

The `exp_threshold` block is the one that is not merely a transcription.
`river-py` evaluates `floor(scale * exp(num/den))` through `decimal` at 80
significant digits; `river-rs` computes the mathematically exact floor in
fixed point.  Those agree unless the true value sits within ~1e-21 of an
integer, which is a claim about this specific set of inputs rather than a
theorem -- so the set is large, spans the range the samplers actually reach,
and includes the exact exponents every published profile produces.

"Every profile" is load-bearing and was not true until 2026-08-01: the
Gaussian and threshold blocks covered N8, N256 and TOY, so N16, N64 and
N128 had their *widths* pinned by the parameter table and their accept /
reject *decisions* pinned by nothing.  A width that survives the table and
diverges in the sampler is exactly the failure this file exists to catch,
so any new block that iterates over profiles should iterate over all of
them.
"""

import argparse
import json
from fractions import Fraction
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "river-py"))

import codec as C                                           # noqa: E402
import params as P                                          # noqa: E402
import sample as S                                          # noqa: E402

#: Why the production `lanes` backend name is gated.  See `main`.
#:
#: No *blocks* are withheld any more: the ring, the parameters and the
#: proof layer are all current in both implementations and all three are
#: generated.  What this record carries is the gate's own cause, so a
#: consumer in the other language can check the two have not drifted.
def _withheld_record(blocks):
    """What is missing from this artifact, and why -- from the gate itself.

    Read from `exact` rather than restated: a KAT that explained the gap in
    its own words would be a second place to keep in step, and the whole
    point of the readiness gate is that there is one.

    `cause` and `constants` are language-neutral, so `river-rs` can check
    the stored record against its *own* gate and fail if the two have
    drifted.  `reason` is the human-readable form and names this language's
    API, so it is recorded but not compared.
    """
    from exact import (lanes_gate_cause, lanes_unavailable_reason,
                       live_lanes_constants)
    cause = lanes_gate_cause()
    assert cause is not None, (
        "the LANES gate has lifted -- generate the withheld blocks instead "
        "of recording why they are absent")
    found, missing = live_lanes_constants()
    assert not missing, missing
    return {
        "blocks": blocks,
        "cause": cause,
        "constants": sorted(found),
        "reason": lanes_unavailable_reason(),
    }


def _xof(domain, parts):
    return S.XOF(domain.encode(), *[p.encode() for p in parts])


def xof_cases():
    out = []
    for domain, parts, n in [
        ("KAT", ["xof"], 16),
        ("KAT", ["xof"], 200),          # spans the 136-byte block boundary
        ("RiVeR.Expand", ["rho", "A"], 64),
        ("RiVeR.Com", [], 32),
    ]:
        out.append({
            "domain": domain,
            "parts": parts,
            "n": n,
            "hex": _xof(domain, parts).read(n).hex(),
        })
    return out


def hash_cases():
    out = []
    for length, domain, parts in [
        (16, "KAT", ["a", "b"]),
        (16, "KAT", ["ab", "c"]),
        (16, "KAT", ["a", "bc"]),
        (64, "RiVeR.KeyGen", ["seed"]),
        (32, "RiVeR.dummy", []),
    ]:
        out.append({
            "length": length,
            "domain": domain,
            "parts": parts,
            "hex": S.hash_bytes(length, domain.encode(),
                                *[p.encode() for p in parts]).hex(),
        })
    return out


def uniform_cases():
    out = []
    for domain, parts, modulus, count in [
        ("KAT", ["uniform"], 61, 8),
        ("KAT", ["uniform"], 61, 64),
        ("KAT", ["u1"], 1, 16),                     # the degenerate modulus
        ("KAT", ["u2"], 2, 32),
        ("KAT", ["ubig"], P.PROFILES["RiVeR-N8"].q, 16),
        ("KAT", ["uhat"], P.PROFILES["RiVeR-N256"].q_hat, 16),
    ]:
        x = _xof(domain, parts)
        out.append({
            "domain": domain,
            "parts": parts,
            "modulus": modulus,
            "count": count,
            "values": [S.uniform_int(x, modulus) for _ in range(count)],
        })
    return out


def beta_cases():
    out = []
    par = P.PROFILES["RiVeR-N8"]
    for domain, parts, beta, length in [("KAT", ["beta"], 1, 3)]:
        x = _xof(domain, parts)
        out.append({
            "domain": domain,
            "parts": parts,
            "beta": beta,
            "d": par.d,
            "length": length,
            "modulus": par.q,
            "values": S.uniform_beta_vec(x, beta, par.d, length, par.q),
        })
    return out


def gaussian_cases():
    """Small integer widths, plus the exact rational widths of the profiles.

    Every profile, not a sample of them: the width is what decides both
    the proposal range and the acceptance threshold, so a profile whose
    width is only pinned in the parameter table has had its *arithmetic*
    checked and its *decisions* not.
    """
    cases = [
        ("KAT", ["gauss8"], 8, 1, 8),
        ("KAT", ["gauss352"], 352, 1, 8),
        ("KAT", ["gaussrat"], *S.rational_sigma(4096.0), 6),
    ]
    for name, par in P.PROFILES.items():
        for label, sigma in (("a", par.sigma_a),
                             ("b", par.sigma_b),
                             ("s", par.sigma_s),
                             ("m", par.sigma_m)):
            num, den = S.rational_sigma(sigma)
            cases.append(("KAT", [f"g-{name}-{label}"], num, den, 4))

    out = []
    for domain, parts, num, den, count in cases:
        x = _xof(domain, parts)
        out.append({
            "domain": domain,
            "parts": parts,
            "sigma_num": num,
            "sigma_den": den,
            "count": count,
            "values": [S.gaussian_int(x, num, den) for _ in range(count)],
        })
    return out


def challenge_cases():
    out = []
    for domain, parts, d, w, gamma, modulus in [
        ("KAT", ["chal"], 32, 32, 16, 61),
        ("KAT", ["chal2"], 32, 8, 16, 61),
        ("KAT", ["chalq"], 32, 32, 16, P.PROFILES["RiVeR-N8"].q_hat),
    ]:
        x = _xof(domain, parts)
        out.append({
            "domain": domain,
            "parts": parts,
            "d": d,
            "w": w,
            "gamma": gamma,
            "modulus": modulus,
            "values": S.sample_challenge(x, d, w, gamma, modulus),
        })
    return out


def exp_threshold_cases():
    """The layer where the two implementations use different mathematics."""
    pairs = [(-1, 2), (-1, 1), (0, 1), (-3, 7), (-1, 1000000),
             (-131, 1), (-133, 1), (-134, 1), (-193, 1), (-194, 1), (-195, 1)]

    # exponents a Gaussian draw actually produces, at every profile's
    # real widths -- the claim "the exact exponents each published profile
    # produces" is only true if every profile is here
    for par in P.PROFILES.values():
        for sigma in (par.sigma_a, par.sigma_b, par.sigma_s, par.sigma_m):
            num, den = S.rational_sigma(sigma)
            bound = (S.GAUSSIAN_TAILCUT * num) // den
            for z in (1, 2, bound // 3, bound // 2, bound - 1, bound):
                pairs.append((-(z * z) * den * den, 2 * num * num))

    # and a broad random sweep over the reachable range
    rng = random.Random(20260801)
    for _ in range(400):
        den = rng.randrange(1, 10 ** 7)
        pairs.append((-rng.randrange(0, 140 * den), den))

    out = []
    for num, den in pairs:
        value = S.exp_threshold(num, den)
        out.append({
            "num": str(num),
            "den": str(den),
            "scale_bits": S.PROB_BITS,
            "hex": format(value, "x"),
        })
    return out


def rej_cases():
    out = []
    specs = [
        ("rej1", "KAT", ["r1"], [1, 2, 3], [1, 0, -1], 20, 1000, 1, 10),
        ("rej2", "KAT", ["r2"], [40, 50, 60], [1, 0, 1], 3, 50, 1, 10),
        ("rej2", "KAT", ["r2neg"], [1, 2, 3], [1, 0, -1], 3, 50, 1, 4),
    ]
    # The four samplers the paper's `OM.Prove` calls, at the
    # widths and slacks it calls them with, in that order:
    #   Rej_1(f_1, ..., phi_a)   Rej_1(z_s, ..., phi_s)
    #   Rej_1(z_m, ..., phi_m)   Rej_2(z_b, ..., phi_b)
    # The first used to be absent from this block, which left `phi_a` --
    # the only per-profile slack of the four -- pinned by nothing.
    par = P.PROFILES["RiVeR-N8"]
    num_a, den_a = S.rational_sigma(par.sigma_a)
    specs.append(("rej1", "KAT", ["r1-N8-a"],
                  [20000, -8000, 4000], [512, -128, 64],
                  par.phi_a, num_a, den_a, 8))
    num_s, den_s = S.rational_sigma(par.sigma_s)
    specs.append(("rej1", "KAT", ["r1-N8-s"],
                  [2 * 10 ** 6, -10 ** 6, 5 * 10 ** 5],
                  [30720, -1024, 512],
                  par.phi_s, num_s, den_s, 8))
    num_m, den_m = S.rational_sigma(par.sigma_m)
    specs.append(("rej1", "KAT", ["r1-N8-m"],
                  [10 ** 7, -5 * 10 ** 6, 2 * 10 ** 6],
                  [30720, -1024, 512],
                  par.phi_m, num_m, den_m, 8))
    num_b, den_b = S.rational_sigma(par.sigma_b)
    specs.append(("rej2", "KAT", ["r2-N8-b"],
                  [50000, 40000, 30000], [512, 256, 128],
                  par.phi_b, num_b, den_b, 8))

    # `tau_rej` is `Rej_1`'s fifth argument since the paper
    # ; `Rej_2` has no such parameter.  It is recorded per case so
    # the KAT pins the value the *sampler* used, not only the value the
    # repetition report assumes -- those were able to disagree while both
    # looked parameterised.
    tau = P.RiVeRParams.REJ_TAU
    for kind, domain, parts, z, v, phi, num, den, count in specs:
        x = _xof(domain, parts)
        if kind == "rej1":
            values = [S.rej1(x, z, v, phi, num, den, tau) for _ in range(count)]
        else:
            values = [S.rej2(x, z, v, phi, num, den) for _ in range(count)]
        case = {
            "kind": kind,
            "domain": domain,
            "parts": parts,
            "z": [str(c) for c in z],
            "v": [str(c) for c in v],
            "phi": phi,
            "sigma_num": num,
            "sigma_den": den,
            "count": count,
            "values": values,
        }
        if kind == "rej1":
            case["tau_rej"] = tau
        out.append(case)
    return out


def sam_mat_cases():
    """`SamMat(rho, q, n, m, str)`.

    Nothing about `A` or `G'` is transmitted, so a divergence here is
    invisible until a proof fails to verify across implementations -- and
    then it looks like a bug in the proof system.  The modulus is packed
    into its *minimal* little-endian width, which is the part a port gets
    wrong.
    """
    out = []
    for seed, modulus, rows, cols, d, label in [
        (b"\x00" * 32, 61, 2, 3, 32, "RiVeR.A"),
        (b"\x01" * 32, P.PROFILES["RiVeR-TOY"].q, 2, 2, 32, "RiVeR.A"),
        (b"\x02" * 32, P.PROFILES["RiVeR-TOY"].q_hat, 1, 4, 32, "RiVeR.G"),
        # widths either side of a byte boundary: 2^16-ish and 2^24-ish
        (b"\x03" * 32, 65533, 1, 2, 32, "w2"),
        (b"\x04" * 32, 16777213, 1, 2, 32, "w3"),
        (b"\x05" * 32, P.PROFILES["RiVeR-N256"].q, 1, 2, 32, "RiVeR.A"),
    ]:
        out.append({
            "seed": seed.hex(),
            "modulus": modulus,
            "rows": rows,
            "cols": cols,
            "d": d,
            "label": label,
            "values": S.sam_mat(seed, modulus, rows, cols, d, label),
        })
    return out


def ring_cases():
    """Rounding and bit dropping.

    `round_p` is Fact 1, and `[[.]]_K` is the convention the paper
    records the paper as contradicting -- which makes it exactly the thing
    a second implementation must not quietly re-derive.
    """
    import ring as R                                        # noqa: E402

    out = []
    rng = random.Random(20260803)
    for name in ("RiVeR-TOY", "RiVeR-N8", "RiVeR-N256"):
        par = P.PROFILES[name]
        rq = R.Ring(par.q, par.d)
        rqhat = R.Ring(par.q_hat, par.d)
        a = [rng.randrange(par.q) for _ in range(par.d)]
        b = [rng.randrange(par.q) for _ in range(par.d)]
        ahat = [rng.randrange(par.q_hat) for _ in range(par.d)]
        # the boundary values, where a convention disagreement shows
        for i, c in enumerate([0, 1, par.q_hat - 1, par.q_hat // 2,
                               par.q_hat // 2 + 1, 1 << par.K_b,
                               (1 << par.K_b) - 1, (1 << par.K_b) + 1]):
            ahat[i] = c
        rounded = R.round_p(rq, a, par.q0)
        out.append({
            "profile": name,
            "q": par.q, "q_hat": par.q_hat, "q0": par.q0, "K_b": par.K_b,
            "a": a, "b": b, "a_hat": ahat,
            "mul": rq.mul(a, b),
            "centered": rq.centered(a),
            "round_p": rounded,
            "rounding_error": R.rounding_error(rq, a, rounded, par.q0),
            "high_bits": R.high_bits(rqhat, ahat, par.K_b),
            "low_bits": R.low_bits(rqhat, ahat, par.K_b),
            "mod_pm": [R.mod_pm(c, par.K_b) for c in ahat],
            "power2round": [list(R.power2round(c, par.K_b)) for c in ahat],
        })
    return out


def _frac(value):
    """An exact rational as `[numerator, denominator]`."""
    f = Fraction(value)
    return [f.numerator, f.denominator]


def profile_cases():
    """Derived columns, so a float that drifts is caught before it moves a
    mask width rather than after."""
    out = []
    for name, par in P.PROFILES.items():
        out.append({
            "name": name,
            "q": par.q,
            "q_hat": par.q_hat,
            "B_e": par.B_e,
            "T_cmp": par.T_cmp,
            "K_a_boundgen": par.K_a_boundgen,
            "sigma_a": list(S.rational_sigma(par.sigma_a)),
            "sigma_b": list(S.rational_sigma(par.sigma_b)),
            "sigma_s": list(S.rational_sigma(par.sigma_s)),
            "sigma_m": list(S.rational_sigma(par.sigma_m)),
            # Exact accept/reject bounds, as `[numerator, denominator]`.
            # Every one of these decides a wire-visible acceptance, so the
            # port has to reproduce the rational, not a float near it.
            "f1_inf_bound_sq": _frac(par.f1_inf_bound_sq),
            "zb_inf_bound_sq": _frac(par.zb_inf_bound_sq),
            "zs_inf_bound_sq": _frac(par.zs_inf_bound_sq),
            "zm_inf_bound_sq": _frac(par.zm_inf_bound_sq),
            "z_l2_bound_sq": _frac(par.z_l2_bound_sq),
            "B_g0": _frac(par.B_g0),
            "B_g1": _frac(par.B_g1),
            "mu_river": par.mu_river,
            "pi_oom_kb": par.proof_size_oom_kb,
        })
    return out


# ---- layer 5: the bit codec ---------------------------------------------
# The codec is the first layer whose output is *the wire*, so these cases
# pin bytes rather than integers.  Three levels: the coders alone, each
# profile's derived layout metadata, and one whole `pi_OOM` encoding.

def _coder_json(coder):
    if isinstance(coder, C.Uniform):
        return {"kind": "uniform", "modulus": coder.modulus,
                "width": coder.width}
    if isinstance(coder, C.Signed):
        return {"kind": "signed", "bound": coder.bound, "width": coder.width}
    return {"kind": "rice", "k": coder.k, "bound": coder.bound,
            "max_high": coder.max_high}


def _probe_values(coder):
    """Four values inside a coder's range, hitting both extremes."""
    if isinstance(coder, C.Uniform):
        return [0, 1, coder.modulus // 2, coder.modulus - 1]
    if isinstance(coder, C.Signed):
        return [0, 1, coder.bound, -coder.bound]
    return [0, 1, -1, coder.bound]


def _blob(coder, values):
    w = C.BitWriter()
    for value in values:
        coder.write(w, value)
    return w.to_bytes().hex()


def codec_coder_cases():
    """The coders on their own, independent of any profile."""
    rice_k = []
    for sigma in (0.5, 1.0, 352, 4096, 1.5e7):    # 352: the pre-1-Aug sigma_y
        num, den = S.rational_sigma(sigma)
        rice_k.append({"sigma_num": num, "sigma_den": den,
                       "k": C.optimal_rice_k(sigma)})

    widths = []
    for modulus in (1, 2, 61, 67112897, 427634113,
                    P.PROFILES["RiVeR-N256"].q_hat):
        widths.append({"modulus": modulus, "width": C.Uniform(modulus).width})
    signed = []
    for bound in (0, 1, 16, 127, 128, 2112):
        signed.append({"bound": bound, "width": C.Signed(bound).width})

    bits = []
    rng = random.Random(20260802)
    for _ in range(12):
        ws = [rng.randrange(1, 33) for _ in range(rng.randrange(1, 20))]
        vs = [rng.randrange(1 << width) for width in ws]
        w = C.BitWriter()
        for value, width in zip(vs, ws):
            w.write_bits(value, width)
        bits.append({"widths": ws, "values": vs, "bit_length": w.bit_length,
                     "hex": w.to_bytes().hex()})

    unary = []
    for value in (0, 1, 7, 31, 32, 33, 100, 255):
        w = C.BitWriter()
        w.write_unary(value)
        unary.append({"value": value, "hex": w.to_bytes().hex()})

    rice_blob = []
    for sigma, bound, values in [
        (352, 4970, [0, 1, -1, 255, -256, 4970, -4970]),
        (352, 2112, list(range(-2112, 2113, 331))),
        (8, 100, [0, 0, 0, 1, -1, 100, -100]),
        (1, 3, [0, 1, -1, 3, -3, 2, -2]),
    ]:
        num, den = S.rational_sigma(sigma)
        coder = C.Rice(sigma, bound)
        rice_blob.append({"sigma_num": num, "sigma_den": den, "bound": bound,
                          "k": coder.k, "values": values,
                          "hex": _blob(coder, values)})

    return {"rice_k": rice_k, "uniform_width": widths, "signed_width": signed,
            "bits": bits, "unary": unary, "rice_blob": rice_blob}


def codec_layout_cases():
    """Per-profile layout metadata, plus a four-value probe per field.

    The metadata alone would catch a drifted width; the probe makes the
    failure name the field that drifted.
    """
    out = []
    for name, par in P.PROFILES.items():
        codec = C.RiVeRCodec(par)
        layout = codec.oom_layout
        fields = []
        for f in layout.fields:
            values = _probe_values(f.coder)
            entry = {"name": f.name, "cols": f.cols, "rows": f.rows,
                     "ring_q": None if f.ring is None else f.ring.q,
                     "max_bits": f.coder.max_bits(),
                     "probe": values, "probe_hex": _blob(f.coder, values)}
            entry.update(_coder_json(f.coder))
            fields.append(entry)
        out.append({
            "profile": name,
            "w_q": codec.w_q, "w_p": codec.w_p, "w_qhat": codec.w_qhat,
            "bound_b_hi": codec.bound_b_hi, "w_b_hi": codec.w_b_hi,
            "bound_f1": codec.bound_f1, "bound_zb": codec.bound_zb,
            "bound_zs": codec.bound_zs, "bound_zm": codec.bound_zm,
            "bound_x": codec.bound_x,
            "pk_bytes": codec.pk_bytes,
            "max_bytes": layout.max_bytes, "min_bytes": layout.min_bytes,
            "fields": fields,
        })
    return out


def _sample_oom(codec, seed):
    """A structurally valid `pi_OOM`, deterministic in `seed`.

    Not an honest proof -- no prover has run -- but every value sits inside
    the range its coder declares, which is all the encoding depends on.
    The first row of each field is pinned to the extremes so the worst-case
    unary run and both signs are exercised.
    """
    rng = random.Random(seed)
    par = codec.par
    out = {}
    for f in codec.oom_layout.fields:
        rows = []
        n_rows = 1 if f.rows is None else f.rows
        for i in range(n_rows):
            if isinstance(f.coder, C.Uniform):
                row = [rng.randrange(f.coder.modulus) for _ in range(f.cols)]
                if i == 0:
                    row[0], row[1] = 0, f.coder.modulus - 1
            else:
                bound = f.coder.bound
                row = [rng.randrange(-bound, bound + 1) for _ in range(f.cols)]
                if i == 0:
                    row[0], row[1], row[2] = 0, bound, -bound
                if f.ring is not None:
                    row = [c % f.ring.q for c in row]
            rows.append(row)
        out[f.name] = rows[0] if f.rows is None else rows
    return out


def _oom_object(layout_fields):
    """The layout-shaped dict as `oom_encode` takes it: `z` reassembled.

    The paper splits `z` on the wire into `z_s` and `z_m`, which
    carry different Rice parameters, and `oom_encode`/`oom_decode` are
    where that split lives.  Encoding through them rather than through the
    layout is deliberate: it pins the boundary too."""
    obj = dict(layout_fields)
    obj["z"] = obj.pop("zs") + obj.pop("zm")
    return obj


def codec_oom_cases():
    """One whole `pi_OOM` encoding, at the profile small enough to inline."""
    out = []
    for name in ("RiVeR-TOY",):
        par = P.PROFILES[name]
        codec = C.RiVeRCodec(par)
        for seed in (7, 20260802):
            pi = _sample_oom(codec, seed)
            blob = codec.oom_encode(_oom_object(pi))
            assert codec.oom_decode(blob) == _oom_object(pi)
            out.append({
                "profile": name,
                "seed": seed,
                "fields": [
                    {"name": f.name,
                     "kind": "residues" if f.ring is not None else "ints",
                     "rows": [pi[f.name]] if f.rows is None else pi[f.name]}
                    for f in codec.oom_layout.fields
                ],
                "bytes": len(blob),
                "hex": blob.hex(),
            })
    return out


class _StubExactBackend:
    """Stands in for `Pi_ex` so the framing can be pinned before it exists.

    `proof_encode` frames `(pi_OOM, pi_ex)` behind two little-endian u32
    prefixes.  That is codec-layer behaviour and does not depend on what
    the exact block contains, so waiting for the exact backend to land
    would leave the one part of "byte-exact" that a source read had to
    stand in for.
    """

    def __init__(self, blob):
        self._blob = blob

    def proof_encode(self, pi):
        return self._blob


def codec_framing_cases():
    out = []
    codec = C.RiVeRCodec(P.PROFILES["RiVeR-TOY"])
    pi_oom = _oom_object(_sample_oom(codec, 7))
    oom_blob = codec.oom_encode(pi_oom)
    for label, ex_blob in [
        ("empty", b""),
        ("short", bytes(range(7))),
        ("aligned", b"\xa5" * 256),
        ("long", bytes((i * 37) % 256 for i in range(1000))),
    ]:
        framed = codec.proof_encode({"oom": pi_oom, "ex": None},
                                    _StubExactBackend(ex_blob))
        assert framed == (len(oom_blob).to_bytes(4, "little") + oom_blob
                          + len(ex_blob).to_bytes(4, "little") + ex_blob)
        out.append({
            "profile": "RiVeR-TOY",
            "label": label,
            "oom_hex": oom_blob.hex(),
            "ex_hex": ex_blob.hex(),
            "hex": framed.hex(),
        })
    return out


def codec_object_cases():
    """Public key, VRF value, challenge, and the two transcript digests."""
    out = []
    for name in ("RiVeR-TOY",):
        par = P.PROFILES[name]
        codec = C.RiVeRCodec(par)
        rng = random.Random(4242)
        pks = [[[rng.randrange(par.p) for _ in range(par.d)]
                for _ in range(par.n)] for _ in range(2)]
        value = [rng.randrange(par.p) for _ in range(par.d)]
        h_m = [[rng.randrange(par.q) for _ in range(par.d)]
               for _ in range(par.ell)]
        sk = [[rng.choice([0, 1, par.q - 1]) for _ in range(par.d)]
              for _ in range(par.ell)]
        x = [rng.randrange(-par.gamma, par.gamma + 1) for _ in range(par.d)]
        out.append({
            "profile": name,
            "pk": pks[0], "pk_hex": codec.pk_encode(pks[0]).hex(),
            "sk": sk, "sk_hex": codec.sk_encode(sk).hex(),
            "value": value, "value_hex": codec.value_encode(value).hex(),
            "challenge": x, "challenge_hex": codec.challenge_encode(x).hex(),
            "ring": pks,
            "ring_digest": C.ring_digest(codec, pks, value).hex(),
            "seed": "73656564",                    # b"seed"
            "h_m": h_m,
            "statement_digest":
                C.statement_digest(codec, b"seed", h_m).hex(),
        })
    return out


def oom_layer_cases():
    """A whole `OM.Com` / `OM.Prove` / `OM.Ver` execution, attempt by attempt.

    The layers below this are pinned value-by-value; this one is pinned as a
    *trajectory*, because that is the thing that can drift.  Every attempt
    consumes XOF bytes through three rejection samplers and can abort at one
    of six places, so a port that draws one extra byte, tests the bounds in a
    different order, or returns early from the wrong check produces a
    different sequence of aborts long before it produces a different proof.
    Pinning only a successful proof would hide all of that behind a retry.

    So: fixed seeds, no randomness outside them, and for each attempt index
    either "aborted" or the full proof.  The statement is built the way
    `RiVeR.eval` builds one -- real keys, a real rounding error, a real
    opening -- because a synthetic statement would not satisfy
    `c_{j*} = Com(0; r)` and `verify` would be testing nothing.
    """
    from river import RiVeR
    from oom import OOM, OOMStatement
    from codec import statement_digest, RiVeRCodec, pack_unsigned
    from ring import Ring, round_p, rounding_error, to_centered_error

    out = []
    for name in ("RiVeR-TOY",):
        par = P.PROFILES[name]
        scheme = RiVeR(par)
        Rq = Ring(par.q, par.d)
        codec = RiVeRCodec(par)

        seed = bytes(range(32))
        rho = S.hash_bytes(32, S.DS_KEYGEN + b".rho", seed)
        A = S.sam_mat(rho, par.q, par.n, par.ell, par.d, "RiVeR.A")

        # N keys, in index order -- no CanonPad here: this pins the OOM
        # layer, and the ring's ordering is the scheme layer's business.
        keys = []
        for i in range(par.N):
            xof = S.XOF(S.DS_KEYGEN, bytes([i]) * 32)
            s = S.uniform_beta_vec(xof, par.beta, par.d, par.ell, par.q)
            As = Rq.mat_vec(A, s)
            t = [round_p(Rq, row, par.q0) for row in As]
            keys.append((s, t))
        ring_pks = [t for _, t in keys]

        j_star = 1
        sk = keys[j_star][0]
        # one XOF, `ell` draws -- `RiVeR.hash_message` does exactly this, and
        # a fresh XOF per polynomial would be a different `G`.
        g_xof = S.XOF(S.DS_G, b"oom-kat")
        h_m = [S.uniform_poly(g_xof, par.q, par.d) for _ in range(par.ell)]

        inner = Rq.inner(h_m, sk)
        v = round_p(Rq, inner, par.q0)
        e_eval = to_centered_error(rounding_error(Rq, inner, v, par.q0),
                                   par.B_e)
        As = Rq.mat_vec(A, sk)
        e_key = [to_centered_error(
                     rounding_error(Rq, As[i], ring_pks[j_star][i], par.q0),
                     par.B_e)
                 for i in range(par.n)]
        r = list(sk) + [Rq.from_centered(e) for e in e_key] \
            + [Rq.from_centered(e_eval)]

        statement = OOMStatement(par, Rq, A, h_m, ring_pks, v)
        assert statement.apply_ck(r) == statement.c_i(j_star)

        ck_digest = statement_digest(codec, rho, h_m)
        rho_digest = S.hash_bytes(32, S.DS_COMMIT + b".nonce", b"oom-kat")
        oom = OOM(par, rho)

        attempts = []
        for k in range(8):
            # 4-byte little-endian, which is what `RiVeR.eval` uses for the
            # attempt counter.  A one-byte label would have left the
            # trajectory KAT green against a scheme port that got the width
            # wrong -- the counter is the scheme's, but this is the only
            # place it is currently pinned.
            xof = S.XOF(S.DS_COMMIT, rho_digest, k.to_bytes(4, "little"))
            t_oom, st_oom = oom.com(statement, j_star, r, xof)
            pi = oom.prove(statement, j_star, r, t_oom, st_oom,
                           ck_digest, rho_digest, xof)
            rec = {
                "attempt": k,
                "A_digest": _digest_ints(t_oom["A"]),
                "B_digest": _digest_ints(t_oom["B"]),
                "E_digest": _digest_ints([Rq.reduce(p) for p in t_oom["E"]]),
            }
            if pi is None:
                rec["aborted"] = True
            else:
                assert oom.verify(statement, pi, ck_digest, rho_digest)
                blob = codec.oom_encode(pi)
                rec.update({
                    "aborted": False,
                    "x": pi["x"],
                    "f1_digest": _digest_ints(pi["f1"]),
                    "zb_digest": _digest_ints(pi["zb"]),
                    # `z_s` and `z_m` separately as well as together: they
                    # are two Gaussians at different widths drawn from one
                    # stream, so a port that swapped the draw order would
                    # still reproduce a single `z` digest for the wrong
                    # reason only if both blocks happened to match.
                    "zs_digest": _digest_ints(
                        [Rq.reduce(p) for p in pi["z"][:par.s_dim]]),
                    "zm_digest": _digest_ints(
                        [Rq.reduce(p) for p in pi["z"][par.s_dim:]]),
                    "z_digest": _digest_ints([Rq.reduce(p) for p in pi["z"]]),
                    "pi_oom_hex": blob.hex(),
                })
                # one tampered proof, to pin that `verify` is not vacuous
                bad = dict(pi, x=list(pi["x"]))
                idx = next(i for i, c in enumerate(bad["x"]) if c)
                bad["x"][idx] = -bad["x"][idx]
                assert not oom.verify(statement, bad, ck_digest, rho_digest)
            attempts.append(rec)

        out.append({
            "profile": name,
            "seed_hex": seed.hex(),
            "rho_hex": rho.hex(),
            "ck_digest_hex": ck_digest.hex(),
            "rho_digest_hex": rho_digest.hex(),
            "j_star": j_star,
            "value": v,
            "r_digest": _digest_ints([Rq.reduce(p) for p in r]),
            "attempts": attempts,
        })
    return out


def _digest_ints(rows):
    """SHAKE-256 over a canonical decimal rendering of nested integers."""
    body = ";".join(",".join(str(int(c)) for c in row) for row in rows)
    return S.hash_bytes(16, b"KAT.digest", body.encode()).hex()


def exact_layer_cases():
    """A whole `Pi_ex.Com` / `Prove` / `Ver` round trip, plus the encodings.

    The exact layer has no rejection loop, so unlike the OOM block there is
    no trajectory to pin -- what matters instead is that the witness packing,
    the commitment and both layouts agree.  The witness is drawn from the XOF
    rather than fixed, so `e_eval` really does span `[-30, 30]` and `y_eval`
    really is Gaussian at `sigma_m`, which is what makes the Rice field
    exercise its coder rather than a corner of it.
    """
    from exact import (OpeningBackend, ExactParams, decompose_poly,
                       pack_witness, padding_is_zero, check_relation)
    from ring import negacyclic_mul_int, from_centered_error

    out = []
    for name in ("RiVeR-TOY", "RiVeR-N8"):
        par = P.PROFILES[name]
        backend = OpeningBackend(par)
        ex = ExactParams(par)
        seed = bytes([0x5A]) * 32
        pp = backend.setup(par, seed)

        # a witness the relation admits, drawn the way `Eval` would draw it
        wx = S.XOF(S.DS_EXACT, b"exact-kat", name.encode())
        e_eval = [S.uniform_int(wx, par.q0) - par.B_e for _ in range(par.d)]
        num, den = S.rational_sigma(par.sigma_m)
        y_eval = [S.gaussian_int(wx, num, den) for _ in range(par.d)]
        x_c = S.challenge_from_hash(par.d, par.w, par.gamma, par.q_hat,
                                    b"exact-kat", name.encode())
        x_c = [c - par.q_hat if c > par.q_hat // 2 else c for c in x_c]
        product = negacyclic_mul_int(x_c, e_eval)
        z_eval = [product[i] + y_eval[i] for i in range(par.d)]

        w_in = {"e_eval": e_eval, "y_eval": y_eval}
        W, st = backend.com(pp, w_in, S.XOF(S.DS_EXACT, b"exact-kat.com"))
        stmt = {"W": W, "z_eval_centered": z_eval, "x_centered": x_c}
        sigma = backend.prove(pp, stmt, w_in, st)
        assert backend.verify(pp, stmt, sigma), "honest proof rejected"
        digits = decompose_poly(from_centered_error(e_eval, par.B_e))
        assert not check_relation(ex, stmt, dict(w_in, digits=digits))

        blob = backend.proof_encode({"W": W, "sigma": sigma})
        w_blob = backend.W_encode(W)
        message = pack_witness(ex, e_eval, y_eval, digits)
        # The paper makes the padding part of the committed
        # message, so it is part of what a port has to reproduce: six
        # 64-slot blocks, 32 carried coefficients and 32 explicit zeros.
        assert padding_is_zero(ex, message)

        out.append({
            "profile": name,
            "seed_hex": seed.hex(),
            "e_eval": e_eval,
            "y_eval": y_eval,
            "x": x_c,
            "z_eval": z_eval,
            "digits_digest": _digest_ints(digits),
            "message_digest": _digest_ints(message),
            "randomness_digest": _digest_ints(st["randomness"]),
            "t0_digest": _digest_ints(W["t0"]),
            "t1_digest": _digest_ints(W["t1"]),
            "W_hex": w_blob.hex(),
            "W_bytes": backend.W_bytes,
            "pi_ex_hex": blob.hex(),
            "pi_ex_bytes": len(blob),
            "pi_ex_max_bytes": backend.proof_bytes,
            "bound_y": backend.bound_y,
            "q_tilde": ex.q_tilde,
            "q_tilde_need": ex.q_tilde_need,
            "q_tilde_clears_at_B_e": ex.q_tilde_clears(par.B_e),
            # The centred range shift is load-bearing, not presentational:
            # with the *unshifted* bound `q_0 - 1 = 60` the requirement
            # doubles and the selected modulus fails outright.
            "q_tilde_clears_at_2B_e": ex.q_tilde_clears(2 * par.B_e),
            "kappa": ex.kappa,
            "identity_rank": ex.t0_rows,
            "tail_rank": ex.roles["tail_rank"],
            "response_rank": ex.response_rank,
            "block_slots": ex.block_slots,
            "block_used": ex.block_used,
        })
    return out


def lanes_ring_cases():
    """The incomplete NTT over `R_q~`, pinned before anything builds on it.

    The transform is the foundation of the whole LANES backend: the
    commitment, the product proof and the linear proof all work in the NTT
    domain, and a twiddle in the wrong order gives a self-consistent ring
    that is not this one.  So it is checked on its own first.
    """
    import lanes_ring as LR

    rng = random.Random(20260802)
    polys = []
    for _ in range(4):
        a = [rng.randrange(LR.QTILDE) for _ in range(LR.DTILDE)]
        b = [rng.randrange(LR.QTILDE) for _ in range(LR.DTILDE)]
        polys.append({
            "a": a,
            "b": b,
            "ntt_a": LR.ntt(a),
            "mul": LR.mul(a, b),
            "ntt_mul": LR.ntt_mul(LR.ntt(a), LR.ntt(b)),
        })

    slots = [rng.randrange(LR.QTILDE) for _ in range(LR.LSPLIT)]
    scal = [rng.randrange(LR.QTILDE) for _ in range(LR.LSPLIT)]
    hat = [rng.randrange(LR.QTILDE) for _ in range(LR.DTILDE)]

    # `inner_ntt` and `add_slots_inplace` are the two the commitment layer
    # calls first, and were the two the earlier block did not cover.
    u = [LR.ntt([rng.randrange(LR.QTILDE) for _ in range(LR.DTILDE)])
         for _ in range(4)]
    v = [LR.ntt([rng.randrange(LR.QTILDE) for _ in range(LR.DTILDE)])
         for _ in range(4)]
    added = LR.add_slots_inplace(list(hat), slots)
    return {
        "inner_ntt": {
            "u": u,
            "v": v,
            "out": LR.inner_ntt(u, v),
        },
        "add_slots": {
            "hat": hat,
            "values": slots,
            "out": added,
        },
        "q_tilde": LR.QTILDE,
        "d_tilde": LR.DTILDE,
        "l_split": LR.LSPLIT,
        "psi": LR.PSI,
        "leaf_exps": list(LR.LEAF_EXPS),
        "leaf_zeta": list(LR.LEAF_ZETA),
        "polys": polys,
        "slots": {
            "values": slots,
            "to_ntt": LR.slots_to_ntt(slots),
        },
        "scale_blocks": {
            "hat": hat,
            "scalars": scal,
            "out": LR.scale_blocks(hat, scal),
        },
        "constant_coefficient": LR.constant_coefficient(hat),
    }


def lanes_params_cases():
    """The LANES samplers, which consume the XOF and so are wire-visible."""
    import lanes_params as LP
    import lanes_ring as LR

    x = S.XOF(b"KAT.lanes", b"chal")
    challenges = [LP.sample_challenge(x) for _ in range(3)]
    g = S.XOF(b"KAT.lanes", b"gauss")
    r_draw = LP.sample_gaussian_poly(g, LP.SIGMA_R)
    y_draw = LP.sample_gaussian_poly(g, LP.SIGMA_Y)
    u = S.XOF(b"KAT.lanes", b"unif")
    return {
        "kappa": LP.KAPPA,
        "response_rank": LP.RESPONSE_RANK,
        "n_tilde": LP.N_TILDE,
        "ell_tilde": LP.ELL_TILDE,
        "n_ex": LP.N_EX,
        "aux": LP.AUX,
        "w_hat": LP.W_HAT,
        "delta": LP.DELTA,
        "w_tilde": LP.W_TILDE,
        "d_drop": LP.D_DROP,
        "t0_high_modulus": LP.T0_HIGH_MODULUS,
        "recovery_buckets": LP.RECOVERY_BUCKETS,
        "recovery_error_bound": LP.RECOVERY_ERROR_BOUND,
        "sigma_r": [LP.SIGMA_R.numerator, LP.SIGMA_R.denominator],
        "sigma_y": [LP.SIGMA_Y.numerator, LP.SIGMA_Y.denominator],
        "z_norm2_bound": LP.Z_NORM2_BOUND,
        "z_inf_bound": LP.Z_INF_BOUND,
        "z_tailcut": LP.Z_TAILCUT,
        "n_z": LP.N_Z,
        "challenges": challenges,
        "gaussian_sigma_r": r_draw,
        "gaussian_sigma_y": y_draw,
        "uniform_poly": LP.sample_uniform_poly(u),
    }


def lanes_proof_cases():
    """The three layers above the ring, each pinned on its own.

    `tests/vectors.rs` already establishes byte equality for whole LANES
    proofs, which is the acceptance test.  What it does not give is a
    *local* diagnostic: a divergence in the commitment key expansion, in
    the `(t_0, t)` inner products, or in one of the six transmitted proof
    elements all surface the same way — a different proof blob, 14 KB
    downstream of the cause.  These cases are the dependency-local checks, in the order a
    failure should be read: the key, then the commitment, then the proof.

    Everything here runs off fixed XOF labels rather than a scheme
    execution, so a case can be reproduced without `Setup` or `KeyGen`.
    """
    import lanes_commit as LC
    import lanes_proof as LP
    import lanes_ring as LR
    from lanes_params import KAPPA, ELL_TILDE, N_TILDE, N_EX, RESPONSE_RANK

    seed = bytes(range(32))
    ck = LC.LanesCommitmentKey(seed)

    # the message the commitment and the proof both run against, and the
    # ternary witness the product proof covers
    alpha_lo, alpha_hi = 2, N_EX
    slots = [[0] * LR.LSPLIT for _ in range(N_EX)]
    for e in range(alpha_lo, alpha_hi):
        slots[e] = [((e * 5 + j * 3) % 3) - 1 for j in range(LR.LSPLIT)]
    an = N_EX * LR.LSPLIT
    A = [[0] * an for _ in range(LR.LSPLIT)]
    u = [0] * LR.LSPLIT
    for j in range(LR.LSPLIT):
        A[j][j] = 1
        for e in range(alpha_lo, alpha_hi):
            A[j][e * LR.LSPLIT + j] = LR.QTILDE - 1
        slots[0][j] = sum(slots[e][j] for e in range(alpha_lo, alpha_hi))
    msg = [[v % LR.QTILDE for v in row] for row in slots]

    xof = S.XOF(S.DS_EXACT, b"KAT.lanes.commit")
    pub, sec = LC.commit(ck, msg, xof)

    pi_xof = S.XOF(S.DS_EXACT, b"KAT.lanes.proof")
    pub2, sec2 = LC.commit(ck, msg, pi_xof)
    pi = LP.prove(ck, pub2, sec2, msg, slots, {"A": A, "u": u},
                  alpha_lo, alpha_hi, pi_xof,
                  LP.Challenges(b"KAT.lanes.statement"))
    assert LP.verify(ck, pub2, pi, {"A": A, "u": u}, alpha_lo, alpha_hi,
                     LP.Challenges(b"KAT.lanes.statement"))

    return {
        "seed": seed.hex(),
        "b_g": LC.B_G, "b_mp1": LC.B_MP1, "b_mp2": LC.B_MP2,
        "b_rows": LC.B_ROWS,
        # the key: first and last stored block of each half, so a wrong
        # draw order is caught without carrying 16 x 128 residues
        "key_b0_first": ck.B0[0][0],
        "key_b0_last": ck.B0[N_TILDE - 1][RESPONSE_RANK - 1],
        "key_b_first": ck.b[0][0],
        "key_b_last": ck.b[LC.B_ROWS - 1][-1],
        # the commitment, off a fixed XOF label
        "commit_r": sec["r"],
        "commit_t0": pub["t0"],
        "commit_t": pub["t"],
        # the whole proof, element by element
        "message_slots": msg,
        "ternary_slots": slots,
        "alpha_lo": alpha_lo,
        "alpha_hi": alpha_hi,
        "statement": b"KAT.lanes.statement".hex(),
        "proof_t_g": pi["t_g"],
        "proof_t_mp1": pi["t_mp1"],
        "proof_t_mp2": pi["t_mp2"],
        "proof_h": pi["h"],
        "proof_c": pi["c"],
        "proof_hint": pi["hint"],
        "proof_z": pi["z"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "tests",
                                                  "sampler_kat.json"))
    args = ap.parse_args()

    doc = {
        "generator": "river-rs/scripts/gen_kat.py",
        "source": "river-py",
        "constants": {
            "prob_bits": S.PROB_BITS,
            "gaussian_tailcut": S.GAUSSIAN_TAILCUT,
            "verifier_tailcut": S.VERIFIER_TAILCUT,
            "shake_block": S.SHAKE_BLOCK,
            "sigma_scale": S.rational_sigma(1.0)[1],
        },
        "xof": xof_cases(),
        "hash_bytes": hash_cases(),
        "uniform_int": uniform_cases(),
        "uniform_beta": beta_cases(),
        "gaussian": gaussian_cases(),
        "challenge": challenge_cases(),
        "exp_threshold": exp_threshold_cases(),
        "rej": rej_cases(),
        "sam_mat": sam_mat_cases(),
        "ring": ring_cases(),
        "profiles": profile_cases(),
        "coders": codec_coder_cases(),
        "layouts": codec_layout_cases(),
        "oom": codec_oom_cases(),
        "framing": codec_framing_cases(),
        "objects": codec_object_cases(),
        "oom_layer": oom_layer_cases(),
        "exact_layer": exact_layer_cases(),
    }

    # The **LANES layers are generated**.
    #
    # They used to be withheld: `river-rs`'s `lanes` modules carried the
    # other widths while `river-py`'s dimensions had
    # moved, so a KAT over them would have pinned the Rust against a
    # parameter set neither implementation accepted as current -- a red
    # test that says nothing.
    #
    # The paper publishes the whole Hint-MLWE chain in closed form, both
    # implementations derive the same widths from it, and both run the
    # proof layer end to end and produce byte-identical proofs.  So these
    # blocks now do what a KAT is for: bisect the two `lanes-experimental`
    # vector cases primitive by primitive, which is how the one-unit
    # `bound_z` disagreement that made every LANES byte differ was found.
    #
    # The ring block came first for the same reason: a withheld
    # ring KAT would leave the layer the exact commitment runs over pinned
    # by nothing.
    doc["lanes_ring"] = lanes_ring_cases()
    doc["lanes_params"] = lanes_params_cases()
    doc["lanes_proof"] = lanes_proof_cases()

    # What is still withheld is the *production* backend name, which is
    # gated on security evidence rather than on parameters -- so the record
    # names no blocks, and exists to carry the cause across the two
    # implementations.
    doc["withheld"] = _withheld_record([])

    with open(args.out, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=False)
        f.write("\n")
    print(f"wrote {args.out}")
    for key, value in doc.items():
        if isinstance(value, list):
            print(f"  {key:16s} {len(value)}")
        elif key == "coders":
            for sub, cases in value.items():
                print(f"  coders.{sub:9s} {len(cases)}")


if __name__ == "__main__":
    main()
