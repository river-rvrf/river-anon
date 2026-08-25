//! Cross-language known-answer tests against the Python reference.
//!
//! `sampler_kat.json` is produced by `scripts/gen_kat.py` from
//! `../river-py`.  It pins the primitives in dependency order — XOF,
//! samplers, acceptance thresholds — so a divergence names the layer at
//! fault instead of surfacing much later as "proof bytes differ".
//!
//! The `exp_threshold` block is the one that is not a transcription.
//! `river-py` evaluates `floor(scale · exp(num/den))` through `decimal`
//! at 80 significant digits; this crate computes the mathematically
//! exact floor in fixed point.  Those two agree unless the true value
//! sits within about `1e-21` of an integer, which is a property of the
//! inputs rather than a theorem — so the block is large and spans the
//! range the samplers actually reach, including the exact exponents every
//! published profile produces — all six, not a sample of them.

use std::collections::HashMap;

use river::codec::{
    optimal_rice_k, proof_frame, ring_digest, statement_digest, BitWriter, Coder, FieldValue,
    RiVeRCodec,
};
use river::exact::{ExactStatement, ExactWitness, OpeningBackend};
use river::fixed::{exp_threshold, Int, Nat};
use river::params::{self, Rat, RiVeRParams};
use river::ring::{high_bits, low_bits, mod_pm, power2round, round_p, rounding_error, Ring};
use river::sample::{
    absorb, gaussian_int, hash_bytes, rational_sigma, rej1, rej2, sam_mat, sample_challenge,
    uniform_beta_vec, uniform_int, Part, Xof, GAUSSIAN_TAILCUT, PROB_BITS, SHAKE_BLOCK,
    SIGMA_SCALE, VERIFIER_TAILCUT,
};
use serde_json::Value;

fn kat() -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/sampler_kat.json");
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{path}: {e}\nrun `make kat-regen` to produce it"));
    serde_json::from_str(&text).expect("sampler_kat.json is not valid JSON")
}

/// Rebuild the XOF a case names.  Every part is an ASCII string on the
/// Python side, absorbed as bytes.
fn xof_of(case: &Value) -> Xof {
    let domain = case["domain"].as_str().unwrap().as_bytes().to_vec();
    let parts: Vec<Vec<u8>> = case["parts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p.as_str().unwrap().as_bytes().to_vec())
        .collect();
    let refs: Vec<Part<'_>> = parts.iter().map(|p| Part::Bytes(p.as_slice())).collect();
    Xof::new(&domain, &refs)
}

fn u128s(v: &Value) -> Vec<u128> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|c| {
            c.as_u64()
                .map(u128::from)
                .or_else(|| c.as_str().and_then(|s| s.parse().ok()))
                .expect("u128 entry")
        })
        .collect()
}

fn u64s(v: &Value) -> Vec<u64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap())
        .collect()
}

fn i64s(v: &Value) -> Vec<i64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect()
}

fn hex_of(v: &Value) -> Vec<u8> {
    hex::decode(v.as_str().unwrap()).unwrap()
}

fn profiles_by_name() -> HashMap<&'static str, RiVeRParams> {
    params::PROFILES.into_iter().map(|p| (p.name, p)).collect()
}

/// Whether a `lanes_*` KAT block is absent from the reference file.
///
/// All three blocks are present, so this returns `false` throughout; it is
/// kept because a block that stopped being generated would otherwise turn
/// into a silently smaller KAT rather than a visible skip.
fn lanes_kat_withheld(name: &str) -> bool {
    let k = kat();
    if k.get(name).is_some() {
        return false;
    }
    println!(
        "SKIPPED ({name}): {}",
        k["withheld"]["reason"].as_str().unwrap_or("withheld")
    );
    true
}

// ---- layer 0: wire-visible constants -------------------------------------

#[test]
fn pinned_constants_match() {
    let k = kat();
    let c = &k["constants"];
    assert_eq!(c["prob_bits"].as_u64().unwrap() as u32, PROB_BITS);
    assert_eq!(c["gaussian_tailcut"].as_u64().unwrap(), GAUSSIAN_TAILCUT);
    assert_eq!(c["verifier_tailcut"].as_u64().unwrap(), VERIFIER_TAILCUT);
    assert_eq!(c["shake_block"].as_u64().unwrap() as usize, SHAKE_BLOCK);
    assert_eq!(c["sigma_scale"].as_u64().unwrap(), SIGMA_SCALE);

    // The `lanes_*` blocks are withheld, and the artifact says so.  The
    // gap is asserted rather than left implicit: a KAT that quietly
    // stopped covering a layer is indistinguishable from one that never
    // did.
    let withheld = &k["withheld"];
    // The stored record has to describe the state this crate is *in*.
    // Checking only that the reason is non-empty would let a changed
    // blocker leave the committed JSON describing something else — and
    // that record is the artifact's only account of why a whole layer is
    // missing from it.
    //
    // The comparison is on `cause` and `constants`, which are
    // language-neutral, rather than on the prose: each implementation's
    // reason names its own API (`exact.LANES_*` against
    // `exact::LANES_*`), so they are not the same string and should not
    // be forced to be.  `make kat-regen` brings the record back into step.
    let cause = withheld["cause"].as_str().expect("withheld.cause");
    assert!(
        river::exact::LANES_GATE_CAUSES.contains(&cause),
        "{cause} is not one of the shared gate-cause tokens"
    );

    // The record describes the *generator's* gate, and the two now have
    // to **agree**.
    //
    // They did not have to before: `river-py` had a frozen manifest and
    // this crate had none, so it reported `no-parameter-manifest` for a
    // table it had simply never been given.  That is fixed —
    // `src/lanes_manifest.rs` is generated from
    // `../river-py/lanes_manifest.json` by
    // `scripts/gen_lanes_manifest.py` and checked by `make manifest-check`
    // — so both implementations are gated on the *same* table and any
    // difference in cause is now a real divergence rather than an artefact
    // of one side lacking an input.
    //
    // The prose `reason` is still not compared: each names its own API
    // (`exact.LANES_*` against `exact::LANES_*`).  The token is the shared
    // vocabulary, and that is what this compares.
    assert_eq!(
        river::exact::lanes_gate_cause(),
        Some(cause),
        "the two gates report different causes; run `make -C ../river-py \
         lanes-manifest-regen` and `make lanes-manifest-regen`, then \
         `make kat-regen`"
    );
    let recorded: Vec<&str> = withheld["constants"]
        .as_array()
        .expect("withheld.constants")
        .iter()
        .map(|c| c.as_str().unwrap())
        .collect();
    let mut ours: Vec<&str> = river::exact::live_lanes_constants()
        .into_iter()
        .map(|(n, _, _)| n)
        .collect();
    ours.sort_unstable();
    assert_eq!(recorded, ours, "the gated constant sets have drifted");
    assert!(
        withheld["reason"].as_str().is_some_and(|r| !r.is_empty()),
        "the withheld record carries no human-readable reason"
    );
    let blocks: Vec<&str> = withheld["blocks"]
        .as_array()
        .expect("withheld.blocks")
        .iter()
        .map(|b| b.as_str().unwrap())
        .collect();
    // **No blocks are withheld any more.**  The ring, the parameters and
    // the proof layer are all current in both implementations, and all
    // three are generated and driven — which is what makes the two
    // `lanes-experimental` vector cases bisectable primitive by primitive
    // rather than pass-or-fail as a whole.
    //
    // The ring block came first, for that reason; the parameter and proof
    // blocks joined it once the widths were pinned.
    assert!(blocks.is_empty(), "no KAT block is withheld: {blocks:?}");
    for block in ["lanes_ring", "lanes_params", "lanes_proof"] {
        assert!(k.get(block).is_some(), "the {block} KAT must be active");
    }
    // What the record still carries is the gate's *cause*, which is about
    // the production backend name rather than about any block.
    assert!(
        river::exact::lanes_skip_reason().is_some(),
        "the LANES production name is no longer gated — the withheld \
         record has nothing left to say; drop it (make kat-regen)"
    );
}

// ---- layer 1: the XOF ----------------------------------------------------

#[test]
fn xof_stream_matches() {
    let k = kat();
    for case in k["xof"].as_array().unwrap() {
        let n = case["n"].as_u64().unwrap() as usize;
        let got = hex::encode(xof_of(case).read(n));
        assert_eq!(got, case["hex"].as_str().unwrap(), "xof case {case}");
    }
}

#[test]
fn hash_bytes_matches() {
    let k = kat();
    for case in k["hash_bytes"].as_array().unwrap() {
        let domain = case["domain"].as_str().unwrap().as_bytes().to_vec();
        let parts: Vec<Vec<u8>> = case["parts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p.as_str().unwrap().as_bytes().to_vec())
            .collect();
        let refs: Vec<Part<'_>> = parts.iter().map(|p| Part::Bytes(p.as_slice())).collect();
        let n = case["length"].as_u64().unwrap() as usize;
        let got = hex::encode(hash_bytes(n, &domain, &refs));
        assert_eq!(got, case["hex"].as_str().unwrap(), "hash case {case}");
    }
}

#[test]
fn absorption_is_injective_in_the_part_boundaries() {
    // The KAT carries ("ab","c") and ("a","bc") precisely so this cannot
    // silently regress to a plain concatenation.
    let a = absorb(b"KAT", &[Part::Bytes(b"ab"), Part::Bytes(b"c")]);
    let b = absorb(b"KAT", &[Part::Bytes(b"a"), Part::Bytes(b"bc")]);
    assert_ne!(a, b);
}

// ---- layer 2: samplers ---------------------------------------------------

#[test]
fn uniform_int_matches() {
    let k = kat();
    for case in k["uniform_int"].as_array().unwrap() {
        let modulus = case["modulus"].as_u64().unwrap();
        let count = case["count"].as_u64().unwrap() as usize;
        let mut x = xof_of(case);
        let got: Vec<u64> = (0..count).map(|_| uniform_int(&mut x, modulus)).collect();
        assert_eq!(got, u64s(&case["values"]), "uniform modulus {modulus}");
    }
}

#[test]
fn uniform_beta_matches() {
    let k = kat();
    for case in k["uniform_beta"].as_array().unwrap() {
        let beta = case["beta"].as_u64().unwrap();
        let d = case["d"].as_u64().unwrap() as usize;
        let len = case["length"].as_u64().unwrap() as usize;
        let modulus = case["modulus"].as_u64().unwrap();
        let mut x = xof_of(case);
        let got = uniform_beta_vec(&mut x, beta, d, len, modulus);
        let want: Vec<Vec<u64>> = case["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(u64s)
            .collect();
        assert_eq!(got, want);
    }
}

#[test]
fn gaussian_matches() {
    let k = kat();
    for case in k["gaussian"].as_array().unwrap() {
        let num = case["sigma_num"].as_u64().unwrap();
        let den = case["sigma_den"].as_u64().unwrap();
        let count = case["count"].as_u64().unwrap() as usize;
        let mut x = xof_of(case);
        let got: Vec<i64> = (0..count)
            .map(|_| gaussian_int(&mut x, num, den, GAUSSIAN_TAILCUT))
            .collect();
        assert_eq!(got, i64s(&case["values"]), "gaussian sigma {num}/{den}");
    }
}

#[test]
fn challenge_matches() {
    let k = kat();
    for case in k["challenge"].as_array().unwrap() {
        let d = case["d"].as_u64().unwrap() as usize;
        let w = case["w"].as_u64().unwrap() as usize;
        let gamma = case["gamma"].as_u64().unwrap();
        let modulus = case["modulus"].as_u64().unwrap();
        let mut x = xof_of(case);
        let got = sample_challenge(&mut x, d, w, gamma, modulus);
        assert_eq!(got, u64s(&case["values"]), "challenge w={w}");
    }
}

// ---- layer 3: acceptance thresholds --------------------------------------

#[test]
fn exp_threshold_matches_the_decimal_reference() {
    let k = kat();
    let cases = k["exp_threshold"].as_array().unwrap();
    assert_eq!(
        cases.len(),
        555,
        "the frozen threshold sweep changed; update its documented coverage deliberately"
    );
    for case in cases {
        let num_s = case["num"].as_str().unwrap();
        let (neg, mag_s) = match num_s.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, num_s),
        };
        let mag = Nat::from_dec_str(mag_s).unwrap();
        let num = Int { neg, mag };
        let den = Nat::from_dec_str(case["den"].as_str().unwrap()).unwrap();
        let scale = Nat::pow2(case["scale_bits"].as_u64().unwrap() as u32);
        let got = exp_threshold(&num, &den, &scale);
        assert_eq!(
            got.to_hex_string(),
            case["hex"].as_str().unwrap(),
            "exp_threshold({num_s} / {})",
            case["den"].as_str().unwrap()
        );
    }
}

#[test]
fn rej_decisions_match() {
    let k = kat();
    for case in k["rej"].as_array().unwrap() {
        let kind = case["kind"].as_str().unwrap();
        let z: Vec<i64> = case["z"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c.as_str().unwrap().parse().unwrap())
            .collect();
        let v: Vec<i64> = case["v"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c.as_str().unwrap().parse().unwrap())
            .collect();
        let phi = case["phi"].as_u64().unwrap();
        let num = case["sigma_num"].as_u64().unwrap();
        let den = case["sigma_den"].as_u64().unwrap();
        let count = case["count"].as_u64().unwrap() as usize;
        let mut x = xof_of(case);
        // Python returns 1 to reject and 0 to accept; the port returns a
        // bool with `true` = reject.
        let got: Vec<i64> = (0..count)
            .map(|_| {
                let rejected = match kind {
                    "rej1" => rej1(&mut x, &z, &v, phi, num, den),
                    "rej2" => rej2(&mut x, &z, &v, phi, num, den),
                    other => panic!("unknown rejection kind {other}"),
                };
                rejected as i64
            })
            .collect();
        assert_eq!(got, i64s(&case["values"]), "{kind} phi={phi}");
    }
}

// ---- layer 3b: derived-but-untransmitted values --------------------------
// Neither `A` nor the bit-dropped high parts travel on the wire, so a
// divergence here surfaces only as "the other implementation's proof does
// not verify" — which reads as a bug in the proof system.

#[test]
fn sam_mat_matches() {
    let k = kat();
    for case in k["sam_mat"].as_array().unwrap() {
        let seed = hex_of(&case["seed"]);
        let got = sam_mat(
            &seed,
            case["modulus"].as_u64().unwrap(),
            case["rows"].as_u64().unwrap() as usize,
            case["cols"].as_u64().unwrap() as usize,
            case["d"].as_u64().unwrap() as usize,
            case["label"].as_str().unwrap(),
        );
        let want: Vec<Vec<Vec<u64>>> = case["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row.as_array().unwrap().iter().map(u64s).collect())
            .collect();
        assert_eq!(got, want, "sam_mat modulus {}", case["modulus"]);
    }
}

#[test]
fn ring_rounding_and_bit_dropping_match() {
    let k = kat();
    for case in k["ring"].as_array().unwrap() {
        let name = case["profile"].as_str().unwrap();
        let q = case["q"].as_u64().unwrap();
        let q_hat = case["q_hat"].as_u64().unwrap();
        let q0 = case["q0"].as_u64().unwrap();
        let kb = case["K_b"].as_u64().unwrap() as u32;
        let rq = Ring::new(q, 32);
        let rqhat = Ring::new(q_hat, 32);

        let a = u64s(&case["a"]);
        let b = u64s(&case["b"]);
        let a_hat = u64s(&case["a_hat"]);

        assert_eq!(rq.mul(&a, &b), u64s(&case["mul"]), "{name} mul");
        assert_eq!(rq.centered(&a), i64s(&case["centered"]), "{name} centered");

        let rounded = round_p(&a, q0);
        assert_eq!(rounded, u64s(&case["round_p"]), "{name} round_p");
        assert_eq!(
            rounding_error(&rq, &a, &rounded, q0),
            u64s(&case["rounding_error"]),
            "{name} rounding_error"
        );

        // `[[·]]_K` is a convention the paper states inconsistently, which
        // is exactly why it is pinned here: a second implementation must
        // not quietly re-derive it.
        let want_hi: Vec<i128> = i64s(&case["high_bits"])
            .into_iter()
            .map(i128::from)
            .collect();
        let want_lo: Vec<i128> = i64s(&case["low_bits"])
            .into_iter()
            .map(i128::from)
            .collect();
        assert_eq!(high_bits(&rqhat, &a_hat, kb), want_hi, "{name} high_bits");
        assert_eq!(low_bits(&rqhat, &a_hat, kb), want_lo, "{name} low_bits");

        for (i, &c) in a_hat.iter().enumerate() {
            let want = case["mod_pm"].as_array().unwrap()[i].as_i64().unwrap();
            assert_eq!(mod_pm(c as i128, kb), want as i128, "{name} mod_pm[{i}]");
            let pair = i64s(&case["power2round"].as_array().unwrap()[i]);
            assert_eq!(
                power2round(c as i128, kb),
                (pair[0] as i128, pair[1] as i128),
                "{name} power2round[{i}]"
            );
        }
    }
}

// ---- layer 4: the parameter layer ----------------------------------------

#[test]
fn profile_derived_values_match() {
    let k = kat();
    let by_name = profiles_by_name();
    for case in k["profiles"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let par = by_name
            .get(name)
            .unwrap_or_else(|| panic!("profile {name} missing from the Rust table"));

        assert_eq!(par.q(), case["q"].as_u64().unwrap(), "{name} q");
        assert_eq!(par.q_hat, case["q_hat"].as_u64().unwrap(), "{name} q_hat");
        assert_eq!(par.B_e(), case["B_e"].as_u64().unwrap(), "{name} B_e");
        assert_eq!(par.T_cmp(), case["T_cmp"].as_i64().unwrap(), "{name} T_cmp");
        assert_eq!(
            par.K_a_boundgen() as u64,
            case["K_a_boundgen"].as_u64().unwrap(),
            "{name} K_a_boundgen"
        );

        // The widths are the ones the samplers consume, so they have to
        // agree as exact rationals, not approximately: a unit of 2^-20
        // moves every mask.
        for (label, got, want) in [
            ("sigma_a", rational_sigma(par.sigma_a()), &case["sigma_a"]),
            ("sigma_b", rational_sigma(par.sigma_b()), &case["sigma_b"]),
            ("sigma_s", rational_sigma(par.sigma_s()), &case["sigma_s"]),
            ("sigma_m", rational_sigma(par.sigma_m()), &case["sigma_m"]),
        ] {
            let want = u64s(want);
            assert_eq!(got.0, want[0], "{name} {label} numerator");
            assert_eq!(got.1, want[1], "{name} {label} denominator");
        }

        // The exact accept/reject bounds, as rationals.  Each decides a
        // wire-visible acceptance, so agreeing to a float's precision is
        // not agreeing: a coefficient on the boundary would be encoded by
        // one implementation and refused by the other.
        for (label, got, want) in [
            (
                "f1_inf_bound_sq",
                par.f1_inf_bound_sq(),
                &case["f1_inf_bound_sq"],
            ),
            (
                "zb_inf_bound_sq",
                par.zb_inf_bound_sq(),
                &case["zb_inf_bound_sq"],
            ),
            (
                "zs_inf_bound_sq",
                par.zs_inf_bound_sq(),
                &case["zs_inf_bound_sq"],
            ),
            (
                "zm_inf_bound_sq",
                par.zm_inf_bound_sq(),
                &case["zm_inf_bound_sq"],
            ),
            ("z_l2_bound_sq", par.z_l2_bound_sq(), &case["z_l2_bound_sq"]),
            ("B_g0", par.B_g0(), &case["B_g0"]),
            ("B_g1", par.B_g1(), &case["B_g1"]),
        ] {
            let want = u128s(want);
            assert_eq!(got, Rat::new(want[0], want[1]), "{name} {label}");
        }

        let mu = case["mu_river"].as_f64().unwrap();
        assert!(
            (par.mu_river() - mu).abs() <= 1e-9 * mu.abs(),
            "{name} mu_river: {} vs {mu}",
            par.mu_river()
        );
        let size = case["pi_oom_kb"].as_f64().unwrap();
        assert!(
            (par.proof_size_oom_kb() - size).abs() <= 1e-9 * size.abs(),
            "{name} |pi_OOM|: {} vs {size}",
            par.proof_size_oom_kb()
        );
    }
}

// ---- layer 5: the bit codec ----------------------------------------------
// The first layer whose output *is* the wire, so these cases pin bytes.
// Three levels: the coders alone, each profile's derived layout, and one
// whole `pi_OOM` encoding.

/// Rebuild a coder from the JSON description of one.
fn coder_of(case: &Value) -> Coder {
    match case["kind"].as_str().unwrap() {
        "uniform" => Coder::uniform(case["modulus"].as_u64().unwrap()),
        "signed" => Coder::signed(case["bound"].as_i64().unwrap()),
        "rice" => Coder::rice_with_k(
            case["k"].as_u64().unwrap() as u32,
            case["bound"].as_i64().unwrap(),
        ),
        other => panic!("unknown coder kind {other}"),
    }
}

fn blob_of(coder: &Coder, values: &[i64]) -> String {
    let mut w = BitWriter::new();
    for &v in values {
        coder.write(&mut w, v).unwrap();
    }
    hex::encode(w.to_bytes())
}

#[test]
fn rice_parameter_matches() {
    // `k` is wire-visible: a different choice is a different encoding, so
    // it is computed in integers over the exact rational on both sides.
    let k = kat();
    for case in k["coders"]["rice_k"].as_array().unwrap() {
        let num = case["sigma_num"].as_u64().unwrap();
        let den = case["sigma_den"].as_u64().unwrap();
        assert_eq!(
            optimal_rice_k(num, den) as u64,
            case["k"].as_u64().unwrap(),
            "optimal_rice_k({num}/{den})"
        );
    }
}

#[test]
fn field_widths_match() {
    let k = kat();
    for case in k["coders"]["uniform_width"].as_array().unwrap() {
        let modulus = case["modulus"].as_u64().unwrap();
        assert_eq!(
            Coder::uniform(modulus).max_bits() as u64,
            case["width"].as_u64().unwrap(),
            "Uniform({modulus})"
        );
    }
    for case in k["coders"]["signed_width"].as_array().unwrap() {
        let bound = case["bound"].as_i64().unwrap();
        assert_eq!(
            Coder::signed(bound).max_bits() as u64,
            case["width"].as_u64().unwrap(),
            "Signed({bound})"
        );
    }
}

#[test]
fn bit_writer_output_matches() {
    let k = kat();
    for case in k["coders"]["bits"].as_array().unwrap() {
        let widths = u64s(&case["widths"]);
        let values = u64s(&case["values"]);
        let mut w = BitWriter::new();
        for (value, width) in values.iter().zip(&widths) {
            w.write_bits(*value, *width as u32);
        }
        assert_eq!(w.bit_length() as u64, case["bit_length"].as_u64().unwrap());
        assert_eq!(hex::encode(w.to_bytes()), case["hex"].as_str().unwrap());
    }
    for case in k["coders"]["unary"].as_array().unwrap() {
        let mut w = BitWriter::new();
        w.write_unary(case["value"].as_u64().unwrap());
        assert_eq!(
            hex::encode(w.to_bytes()),
            case["hex"].as_str().unwrap(),
            "unary {}",
            case["value"]
        );
    }
}

#[test]
fn rice_encoding_matches() {
    let k = kat();
    for case in k["coders"]["rice_blob"].as_array().unwrap() {
        let num = case["sigma_num"].as_u64().unwrap();
        let den = case["sigma_den"].as_u64().unwrap();
        let bound = case["bound"].as_i64().unwrap();
        let coder = Coder::rice(num, den, bound);
        match coder {
            Coder::Rice { k: got, .. } => {
                assert_eq!(got as u64, case["k"].as_u64().unwrap())
            }
            _ => unreachable!(),
        }
        let values = i64s(&case["values"]);
        assert_eq!(
            blob_of(&coder, &values),
            case["hex"].as_str().unwrap(),
            "Rice({num}/{den}, {bound})"
        );
    }
}

#[test]
fn oom_layout_derivation_matches() {
    // Every width, Rice parameter and bound in `pi_OOM` is derived from
    // the profile.  A float that drifts here moves the wire format, not
    // just a size estimate — so the metadata is compared field by field
    // and each field then encodes a four-value probe.
    let k = kat();
    let by_name = profiles_by_name();
    for case in k["layouts"].as_array().unwrap() {
        let name = case["profile"].as_str().unwrap();
        let par = by_name
            .get(name)
            .unwrap_or_else(|| panic!("profile {name} missing from the Rust table"));
        let codec = RiVeRCodec::new(*par);

        for (label, got, want) in [
            ("w_q", codec.w_q as u64, &case["w_q"]),
            ("w_p", codec.w_p as u64, &case["w_p"]),
            ("w_qhat", codec.w_qhat as u64, &case["w_qhat"]),
            ("bound_b_hi", codec.bound_b_hi as u64, &case["bound_b_hi"]),
            ("w_b_hi", codec.w_b_hi as u64, &case["w_b_hi"]),
            ("bound_f1", codec.bound_f1 as u64, &case["bound_f1"]),
            ("bound_zb", codec.bound_zb as u64, &case["bound_zb"]),
            ("bound_zs", codec.bound_zs as u64, &case["bound_zs"]),
            ("bound_zm", codec.bound_zm as u64, &case["bound_zm"]),
            ("bound_x", codec.bound_x as u64, &case["bound_x"]),
            ("pk_bytes", codec.pk_bytes() as u64, &case["pk_bytes"]),
            (
                "max_bytes",
                codec.oom_layout.max_bytes() as u64,
                &case["max_bytes"],
            ),
            (
                "min_bytes",
                codec.oom_layout.min_bytes() as u64,
                &case["min_bytes"],
            ),
        ] {
            assert_eq!(got, want.as_u64().unwrap(), "{name} {label}");
        }

        let fields = case["fields"].as_array().unwrap();
        assert_eq!(fields.len(), codec.oom_layout.fields.len(), "{name} fields");
        for (f, want) in codec.oom_layout.fields.iter().zip(fields) {
            let label = format!("{name}.{}", f.name);
            assert_eq!(f.name, want["name"].as_str().unwrap(), "{label} name");
            assert_eq!(
                f.cols as u64,
                want["cols"].as_u64().unwrap(),
                "{label} cols"
            );
            assert_eq!(
                f.rows.map(|r| r as u64),
                want["rows"].as_u64(),
                "{label} rows"
            );
            assert_eq!(f.ring_q, want["ring_q"].as_u64(), "{label} ring");
            assert_eq!(
                f.coder.max_bits() as u64,
                want["max_bits"].as_u64().unwrap(),
                "{label} max_bits"
            );
            // The coder rebuilt from the JSON and the one the Rust profile
            // derived have to be the same object, not merely agree on the
            // probe.
            assert_eq!(f.coder, coder_of(want), "{label} coder");
            let probe = i64s(&want["probe"]);
            assert_eq!(
                blob_of(&f.coder, &probe),
                want["probe_hex"].as_str().unwrap(),
                "{label} probe"
            );
        }
    }
}

#[test]
fn whole_oom_encoding_matches() {
    let k = kat();
    let by_name = profiles_by_name();
    let cases = k["oom"].as_array().unwrap();
    assert!(!cases.is_empty(), "no whole-proof cases");
    for case in cases {
        let name = case["profile"].as_str().unwrap();
        let codec = RiVeRCodec::new(by_name[name]);
        let pi: Vec<FieldValue> = case["fields"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| {
                let rows = f["rows"].as_array().unwrap();
                match f["kind"].as_str().unwrap() {
                    "ints" => FieldValue::Ints(rows.iter().map(i64s).collect()),
                    "residues" => FieldValue::Residues(rows.iter().map(u64s).collect()),
                    other => panic!("unknown field kind {other}"),
                }
            })
            .collect();

        let blob = codec.oom_encode(&pi).unwrap();
        assert_eq!(
            blob.len() as u64,
            case["bytes"].as_u64().unwrap(),
            "{name} |pi_OOM|"
        );
        assert_eq!(hex::encode(&blob), case["hex"].as_str().unwrap(), "{name}");
        // and the decoder is the inverse of the reference's encoder
        assert_eq!(codec.oom_decode(&blob).unwrap(), pi, "{name} round trip");
    }
}

#[test]
fn scheme_objects_and_transcript_digests_match() {
    let k = kat();
    let by_name = profiles_by_name();
    for case in k["objects"].as_array().unwrap() {
        let name = case["profile"].as_str().unwrap();
        let codec = RiVeRCodec::new(by_name[name]);

        let pk: Vec<Vec<u64>> = case["pk"].as_array().unwrap().iter().map(u64s).collect();
        assert_eq!(
            hex::encode(codec.pk_encode(&pk).unwrap()),
            case["pk_hex"].as_str().unwrap(),
            "{name} pk"
        );
        assert_eq!(codec.pk_decode(&hex_of(&case["pk_hex"])).unwrap(), pk);

        let sk: Vec<Vec<u64>> = case["sk"].as_array().unwrap().iter().map(u64s).collect();
        assert_eq!(
            hex::encode(codec.sk_encode(&sk).unwrap()),
            case["sk_hex"].as_str().unwrap(),
            "{name} sk"
        );
        assert_eq!(codec.sk_decode(&hex_of(&case["sk_hex"])).unwrap(), sk);

        let value = u64s(&case["value"]);
        assert_eq!(
            hex::encode(codec.value_encode(&value).unwrap()),
            case["value_hex"].as_str().unwrap(),
            "{name} value"
        );

        let x = i64s(&case["challenge"]);
        assert_eq!(
            hex::encode(codec.challenge_encode(&x).unwrap()),
            case["challenge_hex"].as_str().unwrap(),
            "{name} challenge"
        );

        // The two Fiat–Shamir digests: a byte of disagreement here is a
        // different challenge and so a different proof, everywhere above.
        let ring: Vec<Vec<Vec<u64>>> = case["ring"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t.as_array().unwrap().iter().map(u64s).collect())
            .collect();
        assert_eq!(
            hex::encode(ring_digest(&codec, &ring, &value).unwrap()),
            case["ring_digest"].as_str().unwrap(),
            "{name} ring_digest"
        );

        let h_m: Vec<Vec<u64>> = case["h_m"].as_array().unwrap().iter().map(u64s).collect();
        let seed = hex_of(&case["seed"]);
        assert_eq!(
            hex::encode(statement_digest(&codec, &seed, &h_m).unwrap()),
            case["statement_digest"].as_str().unwrap(),
            "{name} statement_digest"
        );
    }
}

#[test]
fn full_proof_framing_matches() {
    // The two little-endian u32 prefixes.  Source inspection had shown
    // the two implementations agreed; this is the part of "byte-exact"
    // that was still resting on a read rather than on a vector.  The
    // reference frames a real `pi_OOM` against a stub exact backend, so
    // this does not wait on the exact layer.
    let k = kat();
    let by_name = profiles_by_name();
    let cases = k["framing"].as_array().unwrap();
    assert!(!cases.is_empty(), "no framing cases");
    for case in cases {
        let name = case["profile"].as_str().unwrap();
        let codec = RiVeRCodec::new(by_name[name]);
        let oom = hex_of(&case["oom_hex"]);
        let ex = hex_of(&case["ex_hex"]);
        let want = case["hex"].as_str().unwrap();
        assert_eq!(hex::encode(proof_frame(&oom, &ex)), want, "{name} framing");

        // and the prefixes the reference wrote are the ones this reads
        // back: the block boundary is where a length-prefix disagreement
        // would surface, and `oom` is at the profile's real size.
        let framed = hex_of(&case["hex"]);
        assert_eq!(&framed[4..4 + oom.len()], oom.as_slice());
        assert_eq!(
            u32::from_le_bytes(framed[0..4].try_into().unwrap()) as usize,
            oom.len()
        );
        assert_eq!(
            u32::from_le_bytes(framed[4 + oom.len()..8 + oom.len()].try_into().unwrap()) as usize,
            ex.len()
        );
        assert!(oom.len() <= codec.oom_layout.max_bytes());
        assert!(oom.len() >= codec.oom_layout.min_bytes());
    }
}

// ---- the OOM layer -------------------------------------------------------

/// SHAKE-256 over a canonical decimal rendering, matching
/// `gen_kat.py::_digest_ints`.
fn digest_ints<T: std::fmt::Display>(rows: &[Vec<T>]) -> String {
    let body = rows
        .iter()
        .map(|row| {
            row.iter()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join(",")
        })
        .collect::<Vec<_>>()
        .join(";");
    hex::encode(hash_bytes(
        16,
        b"KAT.digest",
        &[Part::Bytes(body.as_bytes())],
    ))
}

/// Rebuild the statement `RiVeR.eval` would build, from seeds alone.
///
/// A synthetic statement would not satisfy `c_{j*} = Com(0; r)`, and then
/// `verify` would be testing nothing — so this runs real key generation and
/// takes the real rounding errors, which is also what makes it a check on
/// `round_p` and `rounding_error` in situ rather than on their own.
struct OomFixture {
    a_mat: Vec<Vec<Vec<u64>>>,
    h_m: Vec<Vec<u64>>,
    ring_pks: Vec<Vec<Vec<u64>>>,
    value: Vec<u64>,
    r: Vec<Vec<u64>>,
    rho: Vec<u8>,
}

fn oom_fixture(par: &RiVeRParams, seed: &[u8], j_star: usize, msg: &[u8]) -> OomFixture {
    let rq = Ring::new(par.q(), par.d);
    let rho = hash_bytes(
        32,
        &[river::sample::DS_KEYGEN, b".rho"].concat(),
        &[Part::Bytes(seed)],
    );
    let a_mat = sam_mat(&rho, par.q(), par.n, par.ell, par.d, "RiVeR.A");

    let mut ring_pks = Vec::with_capacity(par.N);
    let mut sk_star = Vec::new();
    for i in 0..par.N {
        let mut xof = Xof::new(river::sample::DS_KEYGEN, &[Part::Bytes(&[i as u8; 32])]);
        let s = uniform_beta_vec(&mut xof, par.beta, par.d, par.ell, par.q());
        let as_ = rq.mat_vec(&a_mat, &s);
        let t: Vec<Vec<u64>> = as_.iter().map(|row| round_p(row, par.q0)).collect();
        if i == j_star {
            sk_star = s;
        }
        ring_pks.push(t);
    }

    let mut g_xof = Xof::new(river::sample::DS_G, &[Part::Bytes(msg)]);
    let h_m: Vec<Vec<u64>> = (0..par.ell)
        .map(|_| river::sample::uniform_poly(&mut g_xof, par.q(), par.d))
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
    let e_key: Vec<Vec<u64>> = (0..par.n)
        .map(|i| {
            let canonical = rounding_error(&rq, &as_star[i], &ring_pks[j_star][i], par.q0);
            let centered: Vec<i64> = canonical
                .iter()
                .map(|&c| c as i64 - par.B_e() as i64)
                .collect();
            rq.from_centered(&centered)
        })
        .collect();

    let mut r = sk_star;
    r.extend(e_key);
    r.push(e_eval);
    assert_eq!(r.len(), par.r_dim());

    OomFixture {
        a_mat,
        h_m,
        ring_pks,
        value,
        r,
        rho,
    }
}

/// The whole `OM.Com` / `OM.Prove` / `OM.Ver` trajectory, attempt by attempt.
///
/// Pinned as a trajectory rather than as one successful proof: each attempt
/// consumes XOF bytes through **four** rejection samplers and can abort at
/// one of eleven places, so an extra draw, a reordered bound check or an
/// early return produces a different sequence of aborts long before it
/// produces different proof bytes.  A retry loop would hide all of it.
#[test]
fn oom_layer_trajectory_matches() {
    let k = kat();
    let by_name = profiles_by_name();
    let cases = k["oom_layer"].as_array().expect("no oom_layer block");
    assert!(!cases.is_empty());

    for case in cases {
        let name = case["profile"].as_str().unwrap();
        let par = by_name[name];
        let codec = RiVeRCodec::new(par);
        let seed = hex_of(&case["seed_hex"]);
        let j_star = case["j_star"].as_u64().unwrap() as usize;

        let fx = oom_fixture(&par, &seed, j_star, b"oom-kat");
        assert_eq!(hex::encode(&fx.rho), case["rho_hex"].as_str().unwrap());

        let oom = river::oom::Oom::new(par, &fx.rho);
        let statement =
            river::oom::OomStatement::new(&oom, &fx.a_mat, &fx.h_m, &fx.ring_pks, &fx.value)
                .expect("a well-formed statement");

        // the honest opening really does open `c_{j*}`
        assert_eq!(
            statement.apply_ck(&fx.r),
            statement.c_i(j_star),
            "{name}: c_j* != Com(0; r)"
        );
        assert_eq!(
            digest_ints(&fx.r.iter().map(|p| oom.rq().reduce(p)).collect::<Vec<_>>()),
            case["r_digest"].as_str().unwrap(),
            "{name} r"
        );

        let ck_digest = statement_digest(&codec, &fx.rho, &fx.h_m).unwrap();
        assert_eq!(
            hex::encode(&ck_digest),
            case["ck_digest_hex"].as_str().unwrap()
        );
        let rho_digest = hex_of(&case["rho_digest_hex"]);

        for att in case["attempts"].as_array().unwrap() {
            let idx = att["attempt"].as_u64().unwrap() as u32;
            // 4-byte little-endian: `RiVeR.eval`'s attempt counter.
            let mut xof = Xof::new(
                river::sample::DS_COMMIT,
                &[Part::Bytes(&rho_digest), Part::Bytes(&idx.to_le_bytes())],
            );
            let (commitment, state) = oom.com(&statement, j_star, &mut xof);
            assert_eq!(
                digest_ints(&commitment.a_hi),
                att["A_digest"].as_str().unwrap(),
                "{name} attempt {idx}: A"
            );
            assert_eq!(
                digest_ints(&commitment.b_hi),
                att["B_digest"].as_str().unwrap(),
                "{name} attempt {idx}: B"
            );
            assert_eq!(
                digest_ints(
                    &commitment
                        .e
                        .iter()
                        .map(|p| oom.rq().reduce(p))
                        .collect::<Vec<_>>()
                ),
                att["E_digest"].as_str().unwrap(),
                "{name} attempt {idx}: E"
            );

            let pi = oom.prove(
                &fx.r,
                &commitment,
                &state,
                &ck_digest,
                &rho_digest,
                &mut xof,
            );
            if att["aborted"].as_bool().unwrap() {
                assert!(pi.is_none(), "{name} attempt {idx}: expected abort");
                continue;
            }
            let pi = pi.unwrap_or_else(|| panic!("{name} attempt {idx}: expected a proof"));

            assert_eq!(pi.x, i64s(&att["x"]), "{name} attempt {idx}: x");
            assert_eq!(digest_ints(&pi.f1), att["f1_digest"].as_str().unwrap());
            assert_eq!(digest_ints(&pi.zb), att["zb_digest"].as_str().unwrap());
            // `z_s` and `z_m` separately as well as together: they are two
            // Gaussians at different widths drawn from one stream, so a
            // port that swapped the draw order has to fail here rather
            // than coincidentally agree on the concatenation.
            let z_res: Vec<Vec<u64>> = pi.z.iter().map(|p| oom.rq().reduce(p)).collect();
            let (zs, zm) = z_res.split_at(par.s_dim());
            assert_eq!(digest_ints(zs), att["zs_digest"].as_str().unwrap());
            assert_eq!(digest_ints(zm), att["zm_digest"].as_str().unwrap());
            assert_eq!(digest_ints(&z_res), att["z_digest"].as_str().unwrap());

            assert!(
                oom.verify(&statement, &pi, &ck_digest, &rho_digest),
                "{name} attempt {idx}: honest proof rejected"
            );

            // the bytes, which is the whole point
            let fields = codec
                .oom_field_values(&pi.b_hi, &pi.x, &pi.f1, &pi.zb, &pi.z)
                .unwrap();
            let blob = codec.oom_encode(&fields).unwrap();
            assert_eq!(
                hex::encode(&blob),
                att["pi_oom_hex"].as_str().unwrap(),
                "{name} attempt {idx}: |pi_OOM| bytes"
            );

            // and `verify` is not vacuous
            let mut bad = pi.clone();
            let at = bad.x.iter().position(|&c| c != 0).unwrap();
            bad.x[at] = -bad.x[at];
            assert!(
                !oom.verify(&statement, &bad, &ck_digest, &rho_digest),
                "{name} attempt {idx}: tampered proof accepted"
            );
        }
    }
}

// ---- the exact layer -----------------------------------------------------

/// `Pi_ex.Com` / `Prove` / `Ver`, and both encodings.
///
/// No rejection loop here, so unlike the OOM block there is no trajectory:
/// what has to agree is the witness packing, the commitment and the two
/// layouts.  The witness comes from the reference so the two sides cannot
/// agree by both deriving it the same wrong way.
#[test]
fn exact_layer_round_trip_matches() {
    let k = kat();
    let by_name = profiles_by_name();
    let cases = k["exact_layer"].as_array().expect("no exact_layer block");
    assert!(!cases.is_empty());

    for case in cases {
        let name = case["profile"].as_str().unwrap();
        let par = by_name[name];
        let seed = hex_of(&case["seed_hex"]);
        let backend = OpeningBackend::new(par, &seed).expect("shipped profile");
        let ex = backend.ex();

        // the parameters themselves
        assert_eq!(ex.q_tilde(), case["q_tilde"].as_u64().unwrap(), "{name} q~");
        assert_eq!(
            ex.kappa() as u64,
            case["kappa"].as_u64().unwrap(),
            "{name} kappa"
        );
        assert_eq!(
            backend.bound_y,
            case["bound_y"].as_i64().unwrap(),
            "{name} bound_y"
        );
        let need = case["q_tilde_need"].as_f64().unwrap();
        assert!(
            (ex.q_tilde_need() - need).abs() / need < 1e-12,
            "{name} q~ need"
        );
        assert!(ex.check().is_empty(), "{name}: {:?}", ex.check());

        let e_eval = i64s(&case["e_eval"]);
        let y_eval = i64s(&case["y_eval"]);
        let x = i64s(&case["x"]);
        let z_eval = i64s(&case["z_eval"]);

        // packing, before anything is committed
        let canonical: Vec<i64> = e_eval.iter().map(|&c| c + par.B_e() as i64).collect();
        let digits = river::exact::decompose_poly(&canonical).expect("e_eval in range");
        assert_eq!(
            digest_ints(&digits),
            case["digits_digest"].as_str().unwrap()
        );
        let message = river::exact::pack_witness(ex, &e_eval, &y_eval, &digits).unwrap();
        assert_eq!(
            digest_ints(&message),
            case["message_digest"].as_str().unwrap()
        );
        // Six blocks of `l = 64` slots, of which `d = 32` carry
        // coefficients and 32 are explicit zero padding.  The old
        // `6 d == N_ex l` identity is gone — 192 != 384 — so what comes
        // back is the *carried* count, and the padding is a separate,
        // checked property.
        assert_eq!(
            river::exact::unpack_witness(ex, &message).len(),
            ex.n_ex() * ex.block_used()
        );
        assert_ne!(ex.n_ex() * ex.block_used(), ex.n_ex() * ex.block_slots());
        assert!(
            river::exact::padding_is_zero(ex, &message),
            "{name}: padding slots are not zero"
        );

        let witness = ExactWitness {
            e_eval: e_eval.clone(),
            y_eval: y_eval.clone(),
        };
        let mut com_xof = Xof::new(river::sample::DS_EXACT, &[Part::Bytes(b"exact-kat.com")]);
        let (w, state) = backend.com(&witness, &mut com_xof).expect("com");
        assert_eq!(
            digest_ints(&state.randomness),
            case["randomness_digest"].as_str().unwrap(),
            "{name} randomness"
        );
        assert_eq!(
            digest_ints(&w.t0),
            case["t0_digest"].as_str().unwrap(),
            "{name} t0"
        );
        assert_eq!(
            digest_ints(&w.t1),
            case["t1_digest"].as_str().unwrap(),
            "{name} t1"
        );

        let statement = ExactStatement {
            w: &w,
            z_eval: &z_eval,
            x: &x,
        };
        assert!(
            river::exact::check_relation(ex, &statement, &witness, &digits).is_empty(),
            "{name}: the relation should hold"
        );

        let sigma = backend.prove(&witness, &state);
        assert!(
            backend.verify(&statement, &sigma),
            "{name}: honest proof rejected"
        );

        // the bytes
        assert_eq!(
            hex::encode(backend.w_encode(&w).unwrap()),
            case["W_hex"].as_str().unwrap(),
            "{name} W"
        );
        assert_eq!(backend.w_bytes() as u64, case["W_bytes"].as_u64().unwrap());
        let blob = backend.proof_encode(&w, &sigma).unwrap();
        assert_eq!(
            hex::encode(&blob),
            case["pi_ex_hex"].as_str().unwrap(),
            "{name} pi_ex"
        );
        assert_eq!(blob.len() as u64, case["pi_ex_bytes"].as_u64().unwrap());
        assert_eq!(
            backend.proof_bytes() as u64,
            case["pi_ex_max_bytes"].as_u64().unwrap()
        );

        // both decoders invert the reference's encoder
        assert_eq!(backend.w_decode(&hex_of(&case["W_hex"])).unwrap(), w);
        let (w2, sigma2) = backend.proof_decode(&blob).unwrap();
        assert_eq!(w2, w);
        assert_eq!(sigma2, sigma);
        assert!(backend.verify(&statement, &sigma2));

        // and `verify` is not vacuous: every clause of the relation
        let mut bad_e = sigma.clone();
        bad_e.e_eval[0] = if bad_e.e_eval[0] < par.B_e() as i64 {
            bad_e.e_eval[0] + 1
        } else {
            bad_e.e_eval[0] - 1
        };
        assert!(
            !backend.verify(&statement, &bad_e),
            "{name} tampered e_eval"
        );

        let mut bad_y = sigma.clone();
        bad_y.y_eval[0] += 1;
        assert!(
            !backend.verify(&statement, &bad_y),
            "{name} tampered y_eval"
        );

        let mut bad_d = sigma.clone();
        bad_d.digits[0][0] = if bad_d.digits[0][0] == 0 { 1 } else { 0 };
        assert!(
            !backend.verify(&statement, &bad_d),
            "{name} tampered digits"
        );

        let mut bad_r = sigma.clone();
        bad_r.randomness[0][0] = (bad_r.randomness[0][0] + 1) % ex.q_tilde();
        assert!(
            !backend.verify(&statement, &bad_r),
            "{name} tampered randomness"
        );

        let mut moved_z = z_eval.clone();
        moved_z[0] += 1;
        let other = ExactStatement {
            w: &w,
            z_eval: &moved_z,
            x: &x,
        };
        assert!(!backend.verify(&other, &sigma), "{name} wrong z_eval");
    }
}

// ---- the LANES ring ------------------------------------------------------

/// The incomplete NTT over `R_q~`, before anything builds on it.
///
/// The commitment, the product proof and the linear proof all work in the
/// NTT domain, so a twiddle applied in the wrong order gives a
/// self-consistent ring that is not this one — and every later block would
/// agree with itself while disagreeing with the reference.
#[test]
fn lanes_ring_matches() {
    if lanes_kat_withheld("lanes_ring") {
        return;
    }
    use river::lanes::ring as lr;

    let coeff = |v: &Value| lr::CoeffPoly::new(&u64s(v)).expect("canonical d~-vector");
    let nttp = |v: &Value| lr::NttPoly::new(&u64s(v)).expect("canonical d~-vector");

    let k = kat();
    let c = &k["lanes_ring"];
    assert_eq!(c["q_tilde"].as_u64().unwrap(), lr::QTILDE);
    assert_eq!(c["d_tilde"].as_u64().unwrap() as usize, lr::DTILDE);
    assert_eq!(c["l_split"].as_u64().unwrap() as usize, lr::LSPLIT);

    let exps: Vec<usize> = c["leaf_exps"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect();
    assert_eq!(lr::leaf_exps(), exps.as_slice(), "leaf exponents");
    assert_eq!(
        lr::leaf_zeta(),
        u64s(&c["leaf_zeta"]).as_slice(),
        "leaf zetas"
    );

    for (i, case) in c["polys"].as_array().unwrap().iter().enumerate() {
        let a = coeff(&case["a"]);
        let b = coeff(&case["b"]);
        assert_eq!(lr::ntt(&a).to_vec(), u64s(&case["ntt_a"]), "case {i}: ntt");
        assert_eq!(lr::intt(&lr::ntt(&a)), a, "case {i}: round trip");
        assert_eq!(
            lr::mul(&a, &b).to_vec(),
            u64s(&case["mul"]),
            "case {i}: mul"
        );
        assert_eq!(
            lr::ntt_mul(&lr::ntt(&a), &lr::ntt(&b)).to_vec(),
            u64s(&case["ntt_mul"]),
            "case {i}: ntt_mul"
        );
        // and the schoolbook reference agrees, which is what says the
        // transform is the right one rather than merely reproducible
        assert_eq!(
            lr::mul_schoolbook(&a, &b).to_vec(),
            u64s(&case["mul"]),
            "case {i}: schoolbook"
        );
    }

    let slots = lr::Slots::new(&u64s(&c["slots"]["values"])).unwrap();
    assert_eq!(
        lr::slots_to_ntt(&slots).to_vec(),
        u64s(&c["slots"]["to_ntt"]),
        "slots_to_ntt"
    );
    assert_eq!(lr::ntt_to_slots(&lr::slots_to_ntt(&slots)), slots);

    let hat = nttp(&c["scale_blocks"]["hat"]);
    let scal = lr::Slots::new(&u64s(&c["scale_blocks"]["scalars"])).unwrap();
    assert_eq!(
        lr::scale_blocks(&hat, &scal).to_vec(),
        u64s(&c["scale_blocks"]["out"]),
        "scale_blocks"
    );
    assert_eq!(
        lr::constant_coefficient(&hat),
        c["constant_coefficient"].as_u64().unwrap(),
        "constant_coefficient"
    );

    // the two the commitment layer calls first
    let ip = &c["inner_ntt"];
    let u: Vec<lr::NttPoly> = ip["u"].as_array().unwrap().iter().map(nttp).collect();
    let v: Vec<lr::NttPoly> = ip["v"].as_array().unwrap().iter().map(nttp).collect();
    assert_eq!(
        lr::inner_ntt(&u, &v).unwrap().to_vec(),
        u64s(&ip["out"]),
        "inner_ntt"
    );
    assert!(lr::inner_ntt(&u[..2], &v).is_none(), "unequal lengths");

    let sl = &c["add_slots"];
    let mut target = nttp(&sl["hat"]);
    let values = lr::Slots::new(&u64s(&sl["values"])).unwrap();
    lr::add_slots_inplace(&mut target, &values);
    assert_eq!(target.to_vec(), u64s(&sl["out"]), "add_slots_inplace");
}

/// The LANES samplers, which consume the XOF and so are wire-visible.
#[test]
fn lanes_params_match() {
    if lanes_kat_withheld("lanes_params") {
        return;
    }
    use river::lanes::params as lp;

    let k = kat();
    let c = &k["lanes_params"];
    for (field, got) in [
        ("kappa", lp::KAPPA as u64),
        ("response_rank", lp::RESPONSE_RANK as u64),
        ("n_tilde", lp::N_TILDE as u64),
        ("ell_tilde", lp::ELL_TILDE as u64),
        ("n_ex", lp::N_EX as u64),
        ("aux", lp::AUX as u64),
        ("w_hat", lp::W_HAT as u64),
        ("delta", lp::DELTA as u64),
        ("w_tilde", lp::W_TILDE as u64),
        ("d_drop", lp::D_DROP as u64),
        ("t0_high_modulus", lp::T0_HIGH_MODULUS),
        ("recovery_buckets", lp::RECOVERY_BUCKETS),
        ("recovery_error_bound", lp::RECOVERY_ERROR_BOUND),
        ("n_z", lp::N_Z as u64),
    ] {
        assert_eq!(c[field].as_u64(), Some(got), "{field}");
    }
    assert_eq!(u64s(&c["sigma_r"]), vec![lp::SIGMA_R.0, lp::SIGMA_R.1]);
    assert_eq!(u64s(&c["sigma_y"]), vec![lp::SIGMA_Y.0, lp::SIGMA_Y.1]);
    // the derived bounds, which `river-py/dgs.py` computes in `Decimal`
    assert_eq!(
        c["z_norm2_bound"].as_i64().unwrap() as i128,
        lp::Z_NORM2_BOUND
    );
    assert_eq!(c["z_inf_bound"].as_i64().unwrap(), lp::Z_INF_BOUND);
    assert_eq!(c["z_tailcut"].as_i64().unwrap(), lp::Z_TAILCUT);

    // the challenge sampler, draw for draw
    let mut x = Xof::new(b"KAT.lanes", &[Part::Bytes(b"chal")]);
    for (i, want) in c["challenges"].as_array().unwrap().iter().enumerate() {
        assert_eq!(
            lp::sample_challenge(&mut x).to_vec(),
            u64s(want),
            "challenge {i}"
        );
    }

    let mut g = Xof::new(b"KAT.lanes", &[Part::Bytes(b"gauss")]);
    assert_eq!(
        lp::sample_gaussian_poly(&mut g, lp::SIGMA_R).to_vec(),
        u64s(&c["gaussian_sigma_r"]),
        "D_{{sigma_r}}"
    );
    assert_eq!(
        lp::sample_gaussian_poly(&mut g, lp::SIGMA_Y).to_vec(),
        u64s(&c["gaussian_sigma_y"]),
        "D_{{sigma_y}}"
    );

    let mut u = Xof::new(b"KAT.lanes", &[Part::Bytes(b"unif")]);
    assert_eq!(
        lp::sample_uniform_poly(&mut u).to_vec(),
        u64s(&c["uniform_poly"]),
        "uniform"
    );

    // and the challenge really is in the space the proof assumes
    let mut x = Xof::new(b"KAT.lanes", &[Part::Bytes(b"space")]);
    for _ in 0..20 {
        let cc = lp::sample_challenge(&mut x);
        assert_eq!(lp::challenge_l1_norm(&cc), lp::W_HAT as i64);
        assert!(cc.centered().iter().all(|&v| (-1..=1).contains(&v)));
    }
}

/// The three layers above the ring, each pinned on its own.
///
/// `tests/vectors.rs` establishes byte equality for whole LANES proofs,
/// which is the acceptance test.  What it does not give is a *local*
/// diagnostic: a divergence in the commitment key expansion, in the
/// `(t_0, t)` inner products, or in one of the six transmitted proof
/// elements all surface the same way — a different 14 KB blob, with
/// nothing saying which of the three moved.  This is that diagnostic, in the order a
/// failure should be read.
#[test]
fn lanes_proof_matches() {
    if lanes_kat_withheld("lanes_proof") {
        return;
    }
    use river::lanes::commit::{commit, CommitmentKey, B_G, B_MP1, B_MP2, B_ROWS};
    use river::lanes::params::{ELL_TILDE, KAPPA, N_EX, N_TILDE, RESPONSE_RANK};
    use river::lanes::proof::{prove, verify, Challenges, LinearSystem, AN};
    use river::lanes::ring::{self as lr, LSPLIT, QTILDE};
    use river::sample::DS_EXACT;

    let k = kat();
    let c = &k["lanes_proof"];
    let seed = hex::decode(c["seed"].as_str().unwrap()).unwrap();
    let ck = CommitmentKey::new(&seed);

    for (field, got) in [
        ("b_g", B_G as u64),
        ("b_mp1", B_MP1 as u64),
        ("b_mp2", B_MP2 as u64),
        ("b_rows", B_ROWS as u64),
    ] {
        assert_eq!(c[field].as_u64(), Some(got), "{field}");
    }

    // ---- the key: a wrong draw order shows up here and nowhere later ----
    // `B_0` and the `b` rows come off one XOF in that order, so the first
    // and last stored block of each half pins both the order and the split.
    let probe = |row: usize, col: usize, is_b0: bool| -> Vec<u64> {
        let mut r_hat = vec![lr::NttPoly::zero(); KAPPA];
        let mut one = vec![0u64; lr::DTILDE];
        one[0] = 1;
        let unit = lr::ntt(&lr::CoeffPoly::new(&one).unwrap());
        // `e_k` reads column `k` of the key straight out of the structured
        // inner product, since the identity part sits at a different index
        r_hat[if is_b0 {
            N_TILDE + col
        } else {
            KAPPA - ELL_TILDE + col
        }] = unit;
        if is_b0 {
            ck.apply_b0(row, &r_hat).unwrap().to_vec()
        } else {
            ck.apply_b(row, &r_hat).unwrap().to_vec()
        }
    };
    assert_eq!(probe(0, 0, true), u64s(&c["key_b0_first"]), "B_0[0][0]");
    assert_eq!(
        probe(N_TILDE - 1, RESPONSE_RANK - 1, true),
        u64s(&c["key_b0_last"]),
        "B_0 last"
    );
    assert_eq!(probe(0, 0, false), u64s(&c["key_b_first"]), "b[0][0]");
    assert_eq!(
        probe(B_ROWS - 1, ELL_TILDE - 1, false),
        u64s(&c["key_b_last"]),
        "b last"
    );

    // ---- the commitment -------------------------------------------------
    let msg: Vec<lr::Slots> = c["message_slots"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| lr::Slots::new(&u64s(row)).expect("l canonical slots"))
        .collect();
    let mut xof = Xof::new(DS_EXACT, &[Part::Bytes(b"KAT.lanes.commit")]);
    let (pub_, sec) = commit(&ck, &msg, &mut xof).expect("commit");
    let rows = |v: &Value| -> Vec<Vec<u64>> { v.as_array().unwrap().iter().map(u64s).collect() };
    assert_eq!(
        sec.r().iter().map(|p| p.to_vec()).collect::<Vec<_>>(),
        rows(&c["commit_r"]),
        "commitment randomness"
    );
    assert_eq!(
        pub_.t0.iter().map(|p| p.to_vec()).collect::<Vec<_>>(),
        rows(&c["commit_t0"]),
        "t_0 = B_0 r"
    );
    assert_eq!(
        pub_.t.iter().map(|p| p.to_vec()).collect::<Vec<_>>(),
        rows(&c["commit_t"]),
        "t = <b_i, r> + m_i"
    );

    // ---- the proof, element by element ----------------------------------
    let slots: Vec<Vec<i64>> = c["ternary_slots"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_i64().unwrap())
                .collect()
        })
        .collect();
    let (lo, hi) = (
        c["alpha_lo"].as_u64().unwrap() as usize,
        c["alpha_hi"].as_u64().unwrap() as usize,
    );
    let mut a = vec![vec![0u64; AN]; LSPLIT];
    let u = vec![0u64; LSPLIT];
    for (j, row) in a.iter_mut().enumerate() {
        row[j] = 1;
        for e in lo..hi {
            row[e * LSPLIT + j] = QTILDE - 1;
        }
    }
    let ulp = LinearSystem::new(a, u).expect("well-formed system");
    let statement = hex::decode(c["statement"].as_str().unwrap()).unwrap();

    let mut pi_xof = Xof::new(DS_EXACT, &[Part::Bytes(b"KAT.lanes.proof")]);
    let (pub2, sec2) = commit(&ck, &msg, &mut pi_xof).expect("commit");
    let pi = prove(
        &ck,
        &pub2,
        &sec2,
        &msg,
        &slots,
        &ulp,
        lo,
        hi,
        &mut pi_xof,
        &mut Challenges::new(&statement),
    )
    .expect("honest prover");

    for (field, got) in [
        ("proof_t_g", &pi.t_g),
        ("proof_t_mp1", &pi.t_mp1),
        ("proof_t_mp2", &pi.t_mp2),
        ("proof_h", &pi.h),
    ] {
        assert_eq!(got.to_vec(), u64s(&c[field]), "{field}");
    }
    assert_eq!(pi.c.to_vec(), u64s(&c["proof_c"]), "c");
    let signed_rows = |v: &Value| -> Vec<Vec<i64>> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_i64().unwrap())
                    .collect()
            })
            .collect()
    };
    assert_eq!(pi.hint, signed_rows(&c["proof_hint"]), "recovery hint");
    assert_eq!(
        pi.z.iter().map(|p| p.to_vec()).collect::<Vec<_>>(),
        rows(&c["proof_z"]),
        "z = y + c r"
    );
    assert_eq!(pi.z.len(), RESPONSE_RANK);
    assert_eq!(msg.len(), N_EX);
    let _ = ELL_TILDE;

    // and the pinned proof verifies, so the KAT is of an accepting one
    assert!(verify(
        &ck,
        &pub2,
        &pi,
        &ulp,
        lo,
        hi,
        &mut Challenges::new(&statement)
    ));
}
