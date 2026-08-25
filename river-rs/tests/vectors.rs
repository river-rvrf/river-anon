//! Byte-for-byte interop against `../river-py/vectors.json`.
//!
//! `sampler_kat.json` pins the primitives so a divergence names the layer
//! at fault.  This pins whole executions, which is the acceptance test:
//! setup, key generation, ring admissibility, the VRF value, the attempt
//! count,
//! every recorded intermediate, every recorded size — payload and wire,
//! which differ by the two 4-byte framing prefixes — the proof bytes, and
//! verification.  If this passes, the two implementations are the same
//! protocol.
//!
//! The tampering test flips every bit of both length prefixes — the
//! attacker-controlled part — and strides the payload; it does not exhaust
//! a 29 KB proof, which would be 236 000 decode-and-verify rounds and
//! belongs in a fuzz target.  Full-byte
//! equality is what establishes interoperability — the tampering is there
//! to show the agreement is on something that discriminates.
//!
//! ## Which cases are checked, and which are withheld
//!
//! `vectors.json` carries **four** cases:
//! `{RiVeR-TOY, RiVeR-N8}` against each of the `opening` and
//! `lanes-experimental` backends, and this test checks all four.  The
//! LANES proof layer is ported and byte-exact, so the exact layer is
//! covered here rather than deferred.
//!
//! The two production-alias `lanes` cases are withheld explicitly.  The
//! paper-derived parameters, including `delta_MLWE = 1.0040`, reproduce;
//! the tested candidate is named `lanes-experimental` because its concrete
//! compression/recovery and wire-format completion is implementation-
//! defined and this artifact supplies no reduction for that exact
//! composition.  `LanesBackend::experimental` is the same tested code under
//! the scope-accurate name.
//!
//! The accounting is **enforced rather than narrated**.
//! [`every_case_is_checked`] fails if a case names a backend this crate
//! does not have, so a case added to the reference turns this red rather
//! than shrinking the coverage in silence — and it *also* asserts that the
//! production LANES name is still gated, so the four-case set cannot
//! quietly become the permanent one after the gate lifts.

use std::collections::BTreeSet;

use river::codec::{ring_digest, RiVeRCodec};
use river::params::{self, RiVeRParams};
use river::river::{BackendKind, Proof, PublicParams, RiVeR};
use serde_json::Value;

/// How many cases `vectors.json` is expected to carry.
const CASES: usize = 4;

/// The backends those cases exercise.
const BACKENDS: [&str; 2] = ["lanes-experimental", "opening"];

/// The backend a case names, or a failure that says which one is missing.
fn backend_of(case: &Value) -> BackendKind {
    let name = case["exact_backend"].as_str().expect("exact_backend");
    BackendKind::from_name(name).unwrap_or_else(|| {
        panic!("vectors.json names exact backend {name:?}, which this crate does not have")
    })
}

fn vectors() -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../river-py/vectors.json");
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{path}: {e}\nrun `make -C ../river-py vectors` first"));
    serde_json::from_str(&text).expect("vectors.json is not valid JSON")
}

fn profile(name: &str) -> RiVeRParams {
    params::PROFILES
        .iter()
        .copied()
        .find(|p| p.name == name)
        .unwrap_or_else(|| panic!("unknown profile {name}"))
}

fn hex_of(v: &Value) -> Vec<u8> {
    hex::decode(v.as_str().expect("hex string")).expect("valid hex")
}

fn i64s(v: &Value) -> Vec<i64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect()
}

fn u64s(v: &Value) -> Vec<u64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap())
        .collect()
}

fn inf(rows: &[Vec<i64>]) -> i64 {
    rows.iter()
        .flat_map(|r| r.iter().map(|c| c.abs()))
        .max()
        .unwrap_or(0)
}

/// The file itself, before any case is re-derived.
///
/// Without these, a truncated or substituted file reports success: the
/// reference's own `--verify` gained the same checks for the same reason.
#[test]
fn the_file_is_well_formed() {
    let v = vectors();
    assert_eq!(v["generator"].as_str(), Some("river-py"), "generator");
    let cases = v["cases"].as_array().expect("cases is a list");
    assert!(!cases.is_empty(), "cases must be non-empty");

    let mut seen = BTreeSet::new();
    for case in cases {
        let key = (
            case["params"].as_str().unwrap().to_string(),
            case["exact_backend"].as_str().unwrap().to_string(),
        );
        assert!(seen.insert(key.clone()), "duplicate case {key:?}");
    }
    assert_eq!(
        seen.len(),
        CASES,
        "expected {CASES} cases, got {}",
        seen.len()
    );
}

/// The coverage accounting, enforced.
#[test]
fn every_case_is_checked() {
    let v = vectors();
    let cases = v["cases"].as_array().unwrap();
    let backends: BTreeSet<&'static str> = cases.iter().map(|c| backend_of(c).name()).collect();
    assert_eq!(
        backends,
        BTreeSet::from(BACKENDS),
        "the shipped cases must exercise exactly these backends"
    );
    assert_eq!(cases.len(), CASES);
    // The two *production* `lanes` cases are withheld because that alias is
    // reserved by the artifact's concrete-composition policy. Asserting the
    // gate here is what keeps
    // this from silently becoming a permanently smaller coverage set: when
    // the evidence lands, this fails and says so.
    //
    // The `lanes-experimental` cases are **not** withheld — the proof layer
    // is ported and byte-exact, which is the whole point of the separate
    // name: coverage behind the gate, under a name that claims no more
    // than it has.
    assert!(
        river::exact::lanes_skip_reason().is_some(),
        "the LANES backend is no longer gated — regenerate the reference \
         vectors with its production cases (make -C ../river-py vectors) \
         and raise CASES rather than leaving them withheld"
    );
    println!(
        "river-rs: all {CASES} vector cases checked, backends {backends:?}; \
         2 production lanes cases withheld (production alias reserved)"
    );
}

/// Every `opening` case, re-derived from its seeds.
#[test]
fn shipped_vectors_re_derive() {
    let v = vectors();
    let mut checked = 0usize;

    for case in v["cases"].as_array().unwrap() {
        let backend = backend_of(case);
        let name = case["params"].as_str().unwrap();
        let par = profile(name);
        let scheme = RiVeR::new_with(par, backend);
        let codec = RiVeRCodec::new(par);
        let name = &format!("{name}/{}", backend.name());

        // the profile the case names really is the profile it records
        for (field, got) in [
            ("d", par.d as u64),
            ("N", par.N as u64),
            ("n", par.n as u64),
            ("ell", par.ell as u64),
            ("n_hat", par.n_hat as u64),
            ("k_hat", par.k_hat as u64),
            ("q0", par.q0),
            ("p", par.p),
            ("q", par.q()),
            ("q_hat", par.q_hat),
            ("w", par.w as u64),
            ("gamma", par.gamma),
            ("beta", par.beta),
            ("K_b", par.K_b as u64),
            ("K_a", par.K_a as u64),
        ] {
            assert_eq!(case[field].as_u64(), Some(got), "{name}: {field}");
        }

        // ---- Setup ------------------------------------------------------
        let setup_seed = hex_of(&case["setup_seed"]);
        let pp: PublicParams = scheme.setup(&setup_seed);
        assert_eq!(
            hex::encode(pp.rho()),
            case["rho"].as_str().unwrap(),
            "{name}: rho"
        );

        // ---- KeyGen -----------------------------------------------------
        let records = case["keygen"].as_array().unwrap();
        let mut keys = Vec::with_capacity(records.len());
        for (i, rec) in records.iter().enumerate() {
            assert_eq!(
                rec["index"].as_u64(),
                Some(i as u64),
                "{name}: keygen index"
            );
            let seed = hex_of(&rec["seed"]);
            let (sk, pk) = scheme.keygen(&pp, &seed).expect("pp is this scheme's");
            assert_eq!(
                hex::encode(codec.sk_encode(&sk).unwrap()),
                rec["sk_bytes"].as_str().unwrap(),
                "{name}: sk {i}"
            );
            assert_eq!(
                hex::encode(codec.pk_encode(&pk).unwrap()),
                rec["pk_bytes"].as_str().unwrap(),
                "{name}: pk {i}"
            );
            keys.push((sk, pk));
        }

        let ring: Vec<_> = keys.iter().map(|(_, pk)| pk.clone()).collect();
        let signer = case["signer"].as_u64().unwrap() as usize;
        let (sk, pk) = &keys[signer];

        // ---- ring admissibility -----------------------------------------
        //
        // The paper removes `CanonPad`: a ring is an ordered
        // tuple of exactly `N` keys, so validation returns it unchanged
        // and the order is part of the statement.  The vector's `ring`
        // block is therefore the ring itself, not a padded rewriting of
        // it.
        let validated = scheme.validate_ring(&ring).expect("admissible ring");
        let want: Vec<&str> = case["ring"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_str().unwrap())
            .collect();
        let got: Vec<String> = validated
            .iter()
            .map(|t| hex::encode(codec.pk_encode(t).unwrap()))
            .collect();
        assert_eq!(got, want, "{name}: validated ring");
        assert_eq!(
            scheme.ring_index(&validated, pk),
            Some(case["j_star"].as_u64().unwrap() as usize),
            "{name}: j*"
        );

        // ---- Eval -------------------------------------------------------
        let message = case["message"].as_str().unwrap().as_bytes();
        let eval_seed = hex_of(&case["eval_seed"]);
        let (value, pi, stats) = scheme
            .eval_deterministic(&pp, pk, sk, &ring, message, &eval_seed)
            .expect("eval");

        assert_eq!(
            hex::encode(codec.value_encode(&value).unwrap()),
            case["value"]["bytes"].as_str().unwrap(),
            "{name}: VRF value"
        );
        assert_eq!(
            value,
            u64s(&case["value"]["coefficients"]),
            "{name}: value coefficients"
        );

        // ---- every recorded intermediate ---------------------------------
        // Not just the proof bytes: a second implementation compares against
        // these directly, so an unchecked field is one it could disagree
        // with silently.
        let rec = &case["proof"];
        assert_eq!(
            stats.attempts as u64,
            rec["attempts"].as_u64().unwrap(),
            "{name}: attempts"
        );
        assert_eq!(pi.oom.x, i64s(&rec["challenge"]), "{name}: challenge");
        assert_eq!(
            inf(&pi.oom.f1),
            rec["f1_inf_norm"].as_i64().unwrap(),
            "{name}: ||f_1||_inf"
        );
        assert_eq!(
            inf(&pi.oom.zb),
            rec["zb_inf_norm"].as_i64().unwrap(),
            "{name}: ||z_b||_inf"
        );
        let z_c: Vec<Vec<i64>> = pi
            .oom
            .z
            .iter()
            .map(|p| river::ring::Ring::new(par.q(), par.d).centered(p))
            .collect();
        assert_eq!(
            inf(&z_c),
            rec["z_inf_norm"].as_i64().unwrap(),
            "{name}: ||z||_inf"
        );
        assert_eq!(
            hex::encode(pp.exact().w_encode(&pi.w).unwrap()),
            rec["W_bytes"].as_str().unwrap(),
            "{name}: W"
        );

        // ---- the proof bytes --------------------------------------------
        let blob = scheme.proof_encode(&pp, &pi).expect("encode");
        assert_eq!(
            blob.len() as u64,
            rec["byte_length"].as_u64().unwrap(),
            "{name}: |pi|"
        );
        assert_eq!(
            hex::encode(&blob),
            rec["bytes"].as_str().unwrap(),
            "{name}: proof bytes"
        );

        // ---- Verify, and canonicality -----------------------------------
        let decoded: Proof = scheme.proof_decode(&pp, &blob).expect("decode");
        assert_eq!(decoded, pi, "{name}: decode is the inverse of encode");
        assert_eq!(
            scheme.verify(&pp, &ring, message, &value, &decoded),
            case["verification"].as_bool().unwrap(),
            "{name}: verification"
        );
        assert_eq!(
            scheme.proof_encode(&pp, &decoded).unwrap() == blob,
            case["encoding_is_canonical"].as_bool().unwrap(),
            "{name}: canonical encoding"
        );

        // ---- every reported size, not a sample of them -------------------
        let sizes = &case["sizes"];
        let (oom_bytes, ex_bytes) = {
            let (a, b) =
                river::codec::proof_unframe(&blob, &codec.oom_layout, pp.exact().proof_layout())
                    .expect("unframe");
            (a.len(), b.len())
        };
        for (field, got) in [
            ("pk_bytes", codec.pk_bytes() as u64),
            ("pi_OOM_max_bytes", codec.oom_max_bytes() as u64),
            ("pi_OOM_bytes", oom_bytes as u64),
            ("pi_ex_bytes", ex_bytes as u64),
        ] {
            assert_eq!(sizes[field].as_u64(), Some(got), "{name}: {field}");
        }
        // The KB columns are the byte counts over 1024, and `pi_RiVeR_KB`
        // is their sum — **payload**, not wire.  The framed proof carries
        // two more 4-byte length prefixes, so `proof.byte_length` is 8
        // bytes larger, and the two numbers are recorded for different
        // purposes: the KB columns compare against the paper's
        // communication model, which has no framing in it, while
        // `byte_length` is what a peer actually receives.  Asserting the
        // relation is what keeps that from reading as a discrepancy.
        let kb = |b: usize| b as f64 / 1024.0;
        for (field, got) in [
            ("pi_OOM_KB", kb(oom_bytes)),
            ("pi_ex_KB", kb(ex_bytes)),
            ("pi_RiVeR_KB", kb(oom_bytes) + kb(ex_bytes)),
            ("pi_OOM_paper_KB", par.proof_size_oom_kb()),
        ] {
            let want = sizes[field].as_f64().unwrap();
            assert!(
                (got - want).abs() <= 1e-9 * want.max(1.0),
                "{name}: {field} {got} != {want}"
            );
        }
        assert_eq!(
            oom_bytes + ex_bytes + 8,
            blob.len(),
            "{name}: payload + two 4-byte prefixes must be the wire length"
        );
        assert_eq!(
            oom_bytes + ex_bytes + 8,
            rec["byte_length"].as_u64().unwrap() as usize,
            "{name}: and that is what byte_length records"
        );

        // ---- and the ring digest the nonce is built from -----------------
        assert!(ring_digest(&codec, &validated, &value).is_ok());

        checked += 1;
    }
    assert_eq!(checked, CASES, "expected to check every case");
}

/// A vector case must not verify against anything it was not made for.
///
/// Re-deriving the same bytes says the two implementations agree; this says
/// the agreement is on something that discriminates.
#[test]
fn shipped_vectors_reject_what_they_should() {
    let v = vectors();
    for case in v["cases"].as_array().unwrap() {
        let backend = backend_of(case);
        let name = case["params"].as_str().unwrap();
        let par = profile(name);
        let scheme = RiVeR::new_with(par, backend);
        let pp = scheme.setup(&hex_of(&case["setup_seed"]));
        let name = &format!("{name}/{}", backend.name());

        let keys: Vec<_> = case["keygen"]
            .as_array()
            .unwrap()
            .iter()
            .map(|rec| scheme.keygen(&pp, &hex_of(&rec["seed"])).unwrap())
            .collect();
        let ring: Vec<_> = keys.iter().map(|(_, pk)| pk.clone()).collect();
        let signer = case["signer"].as_u64().unwrap() as usize;
        let (sk, pk) = &keys[signer];
        let message = case["message"].as_str().unwrap().as_bytes();

        let (value, pi, _) = scheme
            .eval_deterministic(&pp, pk, sk, &ring, message, &hex_of(&case["eval_seed"]))
            .unwrap();
        assert!(scheme.verify(&pp, &ring, message, &value, &pi));

        // Every bit of both 4-byte length prefixes — the attacker-controlled
        // part, and the one place a wrong answer is a buffer read rather
        // than a failed check — then a stride across the payload.  The
        // stride is coprime with 8, so it walks every bit position within a
        // byte rather than always the same one.
        //
        // Not "anywhere": exhausting a 29 KB proof is 236 000
        // decode-and-verify rounds, which belongs in a fuzz target.
        let blob = scheme.proof_encode(&pp, &pi).unwrap();
        let total_bits = blob.len() * 8;
        let oom_len = u32::from_le_bytes(blob[..4].try_into().unwrap()) as usize;
        let prefixes: Vec<usize> = (0..32)
            .chain((4 + oom_len) * 8..(4 + oom_len) * 8 + 32)
            .filter(|&b| b < total_bits)
            .collect();
        let strided: Vec<usize> = (0..total_bits).step_by(1291).collect();
        let dense = prefixes;
        let mut flipped = 0usize;
        for bit in dense.into_iter().chain(strided) {
            let mut bad = blob.clone();
            bad[bit / 8] ^= 1 << (bit % 8);
            if bad == blob {
                continue;
            }
            flipped += 1;
            match scheme.proof_decode(&pp, &bad) {
                None => {}
                Some(p) => assert!(
                    !scheme.verify(&pp, &ring, message, &value, &p),
                    "{name}: a proof with bit {bit} flipped verified"
                ),
            }
        }
        assert!(flipped > 80, "{name}: only {flipped} positions exercised");
        // a different message, and a different value
        assert!(!scheme.verify(&pp, &ring, b"not the message", &value, &pi));
        let mut moved = value.clone();
        moved[0] = (moved[0] + 1) % par.p;
        assert!(!scheme.verify(&pp, &ring, message, &moved, &pi));
    }
}
