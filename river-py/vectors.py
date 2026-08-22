"""
vectors.py -- Deterministic test-vector generation for RiVeR.

Runs a fully pinned setup -> keygen -> eval -> serialize -> deserialize ->
verify workflow and emits a JSON blob with hex-encoded bytes for every object
plus selected intermediate values.  A second implementation reproduces the
file byte-for-byte or it is not compatible.

Fixed seeds
-----------
  setup seed  : 00 01 02 ... 1f
  key i seed  : bytes([i ^ 0x40]) + 00 * 31
  eval seed   : aa aa ... aa  (32 bytes)

Usage
-----
  python vectors.py                        # print JSON to stdout
  python vectors.py --out vectors.json     # write to file
  python vectors.py --verify vectors.json  # re-derive and compare
"""

import json
import sys

from params import get, PROFILES
from river import RiVeR

#: Cases covered by the shipped vectors, as `(profile, exact backend)`.
#:
#: The production `"lanes"` name is **withheld**, and the reason changed at
#: the paper.
#:
#: The parameters are not the obstacle: `lanes_params` derives every
#: published figure from the paper's closed
#: form, and both implementations run the proof end to end.
#:
#: What withholds `"lanes"` now is that the production name is gated on
#: security *evidence*: `delta_MLWE = 1.0020` is not reproducible under
#: either reading of the paper's Gaussian convention, and [KLSS23]'s
#: reduction loses about `2^-94.9`.  A vector recorded under a name whose
#: backend refuses to construct would be unverifiable by construction.
#:
#: `"lanes-experimental"` *is* shipped, and is a real cross-language
#: contract: `river-rs` re-derives both cases byte for byte.  That is what
#: makes it a vector rather than a regression test wearing the wrong name.
#:
#: See `exact.lanes_gate_cause()` and `lanes_security.json`.
CASE_PROFILES = (("RiVeR-TOY", "opening"),
                 ("RiVeR-N8", "opening"),
                 ("RiVeR-TOY", "lanes-experimental"),
                 ("RiVeR-N8", "lanes-experimental"))

#: Cases that return once the LANES security evidence does.  Listed rather
#: than deleted so the gap is visible in the artifact instead of only in a
#: commit.
WITHHELD_CASES = (("RiVeR-TOY", "lanes"),
                  ("RiVeR-N8", "lanes"))

#: The exact profile/backend set a shipped `vectors.json` must contain.
#: `verify_vectors` checks this, so dropping a case -- or shipping an empty
#: `cases` array -- fails instead of passing vacuously.
REQUIRED_CASES = frozenset(CASE_PROFILES)


def coverage():
    """What the shipped vectors cover, and what they deliberately do not.

    Returned as data so the accounting can be *tested* rather than narrated.
    A withheld case that quietly reappeared in `CASE_PROFILES`, or a shipped
    case that quietly vanished, is a difference in this dictionary.
    """
    return {
        "shipped": tuple(sorted(CASE_PROFILES)),
        "withheld": tuple(sorted(WITHHELD_CASES)),
        "profiles": tuple(sorted({p for p, _ in CASE_PROFILES})),
        "backends": tuple(sorted({b for _, b in CASE_PROFILES})),
        "withheld_backends": tuple(sorted({b for _, b in WITHHELD_CASES})),
    }

GENERATOR = "river-py"

SETUP_SEED = bytes(range(32))
EVAL_SEED = b"\xAA" * 32
MESSAGE = b"RiVeR test vector"
#: Rings are exactly `N` keys, so this is per
#: profile rather than a constant.
SIGNER = 1


def ring_size(par):
    return par.N


def key_seed(index):
    return bytes([index ^ 0x40]) + b"\x00" * 31


# ---- generation ----------------------------------------------------------

def generate_case(profile_name, exact_backend="opening"):
    """One complete pinned execution for a single profile and backend."""
    par = get(profile_name)
    scheme = RiVeR(par, exact_backend=exact_backend)
    codec = scheme.codec

    pp = scheme.setup(SETUP_SEED)

    keys, keygen_records = [], []
    for i in range(ring_size(par)):
        seed = key_seed(i)
        sk, pk = scheme.keygen(pp, seed)
        keys.append((sk, pk))
        keygen_records.append({
            "index": i,
            "seed": seed.hex(),
            "sk_bytes": codec.sk_encode(sk).hex(),
            "pk_bytes": codec.pk_encode(pk).hex(),
        })

    ring = [pk for _, pk in keys]
    sk, pk = keys[SIGNER]

    v, pi, stats = scheme.eval_deterministic(pp, pk, sk, ring, MESSAGE,
                                             EVAL_SEED, collect_stats=True)
    proof_bytes = scheme.proof_encode(pi)

    decoded = scheme.proof_decode(proof_bytes)
    verified = scheme.verify(pp, ring, MESSAGE, v, decoded)
    canonical = scheme.proof_encode(decoded) == proof_bytes

    ring = scheme.validate_ring(ring)
    j_star = scheme.ring_index(ring, pk)
    pi_oom = pi["oom"]

    return {
        "params": par.name,
        "insecure_toy": par.insecure_toy,
        "d": par.d, "N": par.N, "n": par.n, "ell": par.ell,
        "n_hat": par.n_hat, "k_hat": par.k_hat,
        "q0": par.q0, "p": par.p, "q": par.q, "q_hat": par.q_hat,
        "w": par.w, "gamma": par.gamma, "beta": par.beta,
        "K_b": par.K_b, "K_a": par.K_a,
        "setup_seed": SETUP_SEED.hex(),
        "rho": pp["rho"].hex(),
        "eval_seed": EVAL_SEED.hex(),
        "message": MESSAGE.decode(),
        "signer": SIGNER,
        "j_star": j_star,
        "keygen": keygen_records,
        "ring": [codec.pk_encode(t).hex() for t in ring],
        "value": {
            "bytes": codec.value_encode(v).hex(),
            "coefficients": list(v),
        },
        "proof": {
            "bytes": proof_bytes.hex(),
            "byte_length": len(proof_bytes),
            "attempts": stats["attempts"],
            "challenge": list(pi_oom["x"]),
            "f1_inf_norm": _inf(pi_oom["f1"]),
            "zb_inf_norm": _inf(pi_oom["zb"]),
            "z_inf_norm": _inf([scheme.Rq.centered(p) for p in pi_oom["z"]]),
            "W_bytes": scheme.exact.W_encode(pi["ex"]["W"]).hex(),
        },
        "verification": verified,
        "encoding_is_canonical": canonical,
        "sizes": codec.proof_sizes(pi, scheme.exact),
        "exact_backend": scheme.exact.name,
    }


def _inf(vec):
    return max((max(abs(c) for c in poly) for poly in vec), default=0)


def generate(profiles=CASE_PROFILES):
    return {
        "generator": GENERATOR,
        "cases": [generate_case(name, backend) for name, backend in profiles],
    }


# ---- verification --------------------------------------------------------

def verify_case(case):
    """Re-derive a case from its seeds and diff against the stored values."""
    errors = []
    name = case["params"]
    if name not in PROFILES:
        return [f"unknown profile {name}"]

    par = get(name)
    scheme = RiVeR(par, exact_backend=case["exact_backend"])
    codec = scheme.codec

    for field in ("d", "N", "n", "ell", "n_hat", "k_hat", "q0", "p", "q",
                  "q_hat", "w", "gamma", "beta", "K_b", "K_a"):
        if getattr(par, field) != case[field]:
            errors.append(f"{name}: parameter {field} differs "
                          f"({getattr(par, field)} vs {case[field]})")
    if errors:
        return errors

    pp = scheme.setup(bytes.fromhex(case["setup_seed"]))
    if pp["rho"].hex() != case["rho"]:
        errors.append(f"{name}: rho mismatch")

    keys = []
    expected_keys = ring_size(par)
    if len(case["keygen"]) != expected_keys:
        errors.append(f"{name}: expected {expected_keys} keygen records, "
                      f"got {len(case['keygen'])}")
        return errors
    for expected_index, record in enumerate(case["keygen"]):
        # `index` was recorded but never re-derived, so an altered value
        # passed; it names which seed the record belongs to, so check it.
        if record.get("index") != expected_index:
            errors.append(f"{name}: keygen index {record.get('index')} "
                          f"!= {expected_index}")
        if bytes.fromhex(record["seed"]) != key_seed(expected_index):
            errors.append(f"{name}: keygen seed at {expected_index} "
                          f"is not the pinned one")
        seed = bytes.fromhex(record["seed"])
        sk, pk = scheme.keygen(pp, seed)
        keys.append((sk, pk))
        if codec.sk_encode(sk).hex() != record["sk_bytes"]:
            errors.append(f"{name}: sk_bytes differ at key {record['index']}")
        if codec.pk_encode(pk).hex() != record["pk_bytes"]:
            errors.append(f"{name}: pk_bytes differ at key {record['index']}")
        if codec.pk_encode(codec.pk_decode(
                bytes.fromhex(record["pk_bytes"]))).hex() != record["pk_bytes"]:
            errors.append(f"{name}: pk encoding not canonical")

    ring = [pk for _, pk in keys]
    sk, pk = keys[case["signer"]]

    ring = scheme.validate_ring(ring)
    if [codec.pk_encode(t).hex() for t in ring] != case["ring"]:
        errors.append(f"{name}: CanonPad output differs")

    message = case["message"].encode()
    v, pi, stats = scheme.eval_deterministic(
        pp, pk, sk, ring, message, bytes.fromhex(case["eval_seed"]),
        collect_stats=True)

    if codec.value_encode(v).hex() != case["value"]["bytes"]:
        errors.append(f"{name}: VRF value differs")
    if list(v) != case["value"]["coefficients"]:
        errors.append(f"{name}: VRF value coefficients differ")

    # Every recorded intermediate is re-derived, not just the proof bytes: a
    # second implementation compares against these fields directly, so an
    # unchecked field is one it could disagree with silently.
    pi_oom = pi["oom"]
    recorded = case["proof"]
    for label, actual, expected in (
            ("attempts", stats["attempts"], recorded["attempts"]),
            ("challenge", list(pi_oom["x"]), recorded["challenge"]),
            ("f1_inf_norm", _inf(pi_oom["f1"]), recorded["f1_inf_norm"]),
            ("zb_inf_norm", _inf(pi_oom["zb"]), recorded["zb_inf_norm"]),
            ("z_inf_norm", _inf([scheme.Rq.centered(p) for p in pi_oom["z"]]),
             recorded["z_inf_norm"]),
            ("W_bytes", scheme.exact.W_encode(pi["ex"]["W"]).hex(),
             recorded["W_bytes"]),
            ("j_star", scheme.ring_index(ring, pk), case["j_star"]),
            ("insecure_toy", par.insecure_toy, case["insecure_toy"]),
            ("exact_backend", scheme.exact.name, case["exact_backend"]),
            ("sizes", codec.proof_sizes(pi, scheme.exact),
             case["sizes"]),
    ):
        if actual != expected:
            errors.append(f"{name}: {label} differs")
    if not case.get("encoding_is_canonical", False):
        errors.append(f"{name}: stored vector records a non-canonical encoding")

    blob = scheme.proof_encode(pi)
    expected = case["proof"]["bytes"]
    if blob.hex() != expected:
        errors.append(f"{name}: proof bytes differ")
        got = blob.hex()
        for i in range(min(len(got), len(expected)) // 2):
            if got[2 * i:2 * i + 2] != expected[2 * i:2 * i + 2]:
                errors.append(f"  first difference at byte {i}: "
                              f"got 0x{got[2*i:2*i+2]} "
                              f"expected 0x{expected[2*i:2*i+2]}")
                break
    if len(blob) != case["proof"]["byte_length"]:
        errors.append(f"{name}: proof length {len(blob)} vs "
                      f"{case['proof']['byte_length']}")

    try:
        decoded = scheme.proof_decode(bytes.fromhex(expected))
    except ValueError as exc:
        errors.append(f"{name}: proof_decode failed: {exc}")
        return errors

    if scheme.proof_encode(decoded) != bytes.fromhex(expected):
        errors.append(f"{name}: proof encoding is not canonical")
    if not scheme.verify(pp, ring, message, v, decoded):
        errors.append(f"{name}: verification failed on the decoded proof")
    if not case["verification"]:
        errors.append(f"{name}: stored vector records a failed verification")

    return errors


def verify_vectors(blob):
    """Re-derive every case, and validate the file itself.

    The file-level checks matter as much as the per-case ones: without them
    `--verify` passed on an empty `cases` array, on a changed `generator`, and
    on a case set missing entries -- so a truncated or substituted vector file
    reported success.
    """
    errors = []
    if blob.get("generator") != GENERATOR:
        errors.append(f"generator {blob.get('generator')!r} "
                      f"!= {GENERATOR!r}")

    cases = blob.get("cases")
    if not isinstance(cases, list) or not cases:
        return False, errors + ["`cases` must be a non-empty list"]

    present = set()
    well_formed = []
    for case in cases:
        try:
            key = (case["params"], case["exact_backend"])
        except (KeyError, TypeError):
            errors.append("case missing `params` or `exact_backend`")
            continue
        if key in present:
            errors.append(f"duplicate case {key}")
        present.add(key)
        well_formed.append(case)

    missing = REQUIRED_CASES - present
    extra = present - REQUIRED_CASES
    if missing:
        errors.append(f"missing required cases: {sorted(missing)}")
    if extra:
        errors.append(f"unexpected cases: {sorted(extra)}")

    # Only structurally sound cases go on to re-derivation; the rest have
    # already been recorded above, and running them would raise rather than
    # report.
    for case in well_formed:
        errors.extend(verify_case(case))
    return not errors, errors


# ---- CLI -----------------------------------------------------------------

def main(argv):
    if len(argv) >= 2 and argv[0] == "--verify":
        with open(argv[1]) as handle:
            blob = json.load(handle)
        print(f"Verifying test vectors from {argv[1]} ...", file=sys.stderr)
        ok, errors = verify_vectors(blob)
        if ok:
            cases = ", ".join(f"{c['params']}/{c['exact_backend']}"
                              for c in blob["cases"])
            print(f"ALL CHECKS PASSED ({cases})")
        else:
            print(f"FAILED ({len(errors)} errors):")
            for err in errors:
                print(f"  {err}")
        return 0 if ok else 1

    profiles = CASE_PROFILES
    if len(argv) >= 2 and argv[0] == "--profile":
        profiles = tuple((argv[1], be) for _, be in
                         dict.fromkeys(be for _, be in CASE_PROFILES))
        argv = argv[2:]

    label = ", ".join(f"{name}/{be}" for name, be in profiles)
    print(f"Generating test vectors for {label} ...", file=sys.stderr)
    blob = generate(profiles)

    if len(argv) >= 2 and argv[0] == "--out":
        with open(argv[1], "w") as handle:
            json.dump(blob, handle, indent=2)
            handle.write("\n")
        print(f"Written to {argv[1]}", file=sys.stderr)
    else:
        json.dump(blob, sys.stdout, indent=2)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
