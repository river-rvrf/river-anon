//! `river-rs` micro-benchmarks — `make bench`, or `make bench-lanes` for
//! the LANES ring, exact backend and wire format only.
//!
//! Self-contained: `std::time::Instant`, no harness crate, matching the
//! crate's one-dependency posture.  Each figure is a median of repeated
//! batches, which is enough to separate 10 ns from 500 ns and is not
//! trying to be more than that.
//!
//! Both layers are measured end to end, and so is the composition: one
//! `Eval` is `mu_RiVeR` OOM attempts plus one exact proof.  The exact
//! layer appears twice, once per backend — `opening` is the cheap
//! witness-revealing mock and `lanes-experimental` is the candidate LANES
//! prover, which does not transmit the witness, so the gap between their
//! two `pi_ex` lines is what hiding it costs here.  The per-primitive
//! numbers above are what explain the ones below.

use std::{env, process, time::Instant};

use river::codec::RiVeRCodec;
use river::exact::{ExactStatement, ExactWitness, OpeningBackend};
use river::lanes::backend::{LanesBackend, LanesStatement};
use river::lanes::params::{
    AUX, D_DROP, ELL_TILDE, KAPPA, N_EX, N_TILDE, RESPONSE_RANK, SIGMA_Y, T0_HIGH_MODULUS,
};
use river::lanes::ring::{self as lanes_ring, DTILDE, QTILDE};
use river::oom::{Oom, OomStatement};
use river::params::{RiVeRParams, PROFILES, RIVER_N256, RIVER_N8, RIVER_TOY};
use river::ring::{round_p, rounding_error, to_centered_error, Poly, PolyMat, PolyVec, Ring};
use river::river::{BackendKind, RiVeR};
use river::sample::{
    challenge_from_hash, gaussian_int, gaussian_int_ctx, hash_bytes, rational_sigma, sam_mat,
    uniform_beta_vec, uniform_int, uniform_poly, GaussCtx, Part, Xof, GAUSSIAN_TAILCUT,
};

/// Median wall time of `reps` batches of `n` iterations, in ns per
/// iteration.  The median rather than the mean, because a stray
/// scheduler preemption otherwise dominates a short batch.
///
/// Every iteration's result crosses [`std::hint::black_box`].  The
/// observable accumulator alone is **not** enough: it stops dead-code
/// elimination, but for a pure function of loop-invariant inputs LLVM is
/// still free to hoist the call out and multiply — reporting the cost of
/// one evaluation spread over `n` iterations.  That is a real hazard for
/// the cheapest closures here; a 1 ns reduction over a fixed array is
/// exactly the shape that hoists.
///
/// `black_box` on the *output* stops the hoist.  A closure whose *inputs*
/// are also loop-invariant has to black-box those at its own call site,
/// or it is still timing a constant.
fn bench<F: FnMut() -> u64>(n: usize, reps: usize, mut f: F) -> f64 {
    let mut sink = 0u64;
    let mut times = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        for _ in 0..n {
            sink = sink.wrapping_add(std::hint::black_box(f()));
        }
        times.push(t.elapsed().as_nanos() as f64 / n as f64);
    }
    // keep the accumulator observable so nothing is optimized away
    if sink == u64::MAX {
        println!("(unreachable {sink})");
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[reps / 2]
}

/// What the numbers below depend on besides the code.
///
/// A timing with no CPU, OS and compiler beside it is not reproducible, so
/// they are printed with every run rather than left for the reader to
/// record.  The CPU model is read from `/proc/cpuinfo` where that exists
/// and omitted otherwise; nothing here fails if it is absent.
fn print_environment() {
    let cpu = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|v| v.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".to_string());
    println!("  cpu     {cpu}");
    println!("  target  {}", env!("RIVER_TARGET"));
    println!("  rustc   {}", env!("RIVER_RUSTC_VERSION"));
    println!("  NOTE    sizes are exact and reproduce byte for byte; timings do not.");
}

fn row(label: &str, ns: f64, note: &str) {
    let rate = if ns > 0.0 { 1e9 / ns } else { 0.0 };
    println!("  {label:<38} {ns:>10.1} ns   {:>12.0}/s   {note}", rate);
}

fn section(title: &str) {
    println!("\n{title}\n{}", "-".repeat(title.len()));
}

/// Independent seeds each `Eval` is run over.
///
/// Small on purpose — an `Eval` at `RiVeR-N256` is most of a second — but
/// more than one, because a single retry loop says almost nothing about the
/// per-attempt cost.
const EVAL_SEEDS: usize = 5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Selection {
    All,
    Lanes,
    Sizes,
}

fn usage() {
    println!(
        "Usage: river-bench [--lanes | --sizes]\n\n\
         With no option, benchmark the complete implementation.\n\
         --lanes  benchmark only the LANES ring, backend and codec\n\
         --sizes  generate one proof per published profile and report bytes\n\
         -h, --help  show this help"
    );
}

fn selection() -> Selection {
    let mut args = env::args().skip(1);
    match (args.next().as_deref(), args.next()) {
        (None, None) => Selection::All,
        (Some("--lanes"), None) => Selection::Lanes,
        (Some("--sizes"), None) => Selection::Sizes,
        (Some("-h" | "--help"), None) => {
            usage();
            process::exit(0);
        }
        _ => {
            usage();
            process::exit(2);
        }
    }
}

fn main() {
    let selection = selection();
    println!("river-rs benchmarks  (single-threaded, release)");
    print_environment();

    if selection == Selection::Lanes {
        bench_lanes_ring();
        bench_lanes_backend();
        return;
    }
    if selection == Selection::Sizes {
        communication_sizes();
        return;
    }

    section("XOF — SHAKE-256 counter mode");
    let mut x = Xof::new(b"bench", &[Part::Bytes(b"xof")]);
    let ns = bench(20_000, 7, || x.uint(8));
    row("read 8 bytes", ns, "");
    let mut x = Xof::new(b"bench", &[Part::Bytes(b"blk")]);
    let ns = bench(5_000, 7, || x.read(136)[0] as u64);
    row(
        "read one 136-byte block",
        ns,
        &format!("{:.2} ns/byte", ns / 136.0),
    );

    section("Gaussian sampler");
    for par in [RIVER_N8, RIVER_N256] {
        for (lab, sigma) in [
            ("sigma_a", par.sigma_a()),
            ("sigma_b", par.sigma_b()),
            ("sigma_s", par.sigma_s()),
            ("sigma_m", par.sigma_m()),
        ] {
            let (num, den) = rational_sigma(sigma);
            let ctx = GaussCtx::new(num, den, GAUSSIAN_TAILCUT);
            let mut x = Xof::new(b"bench", &[Part::Bytes(lab.as_bytes())]);
            let ns = bench(20_000, 5, || gaussian_int_ctx(&mut x, &ctx) as u64);
            row(
                &format!("{} {}", par.name, lab),
                ns,
                &format!("sigma = {sigma:.0}"),
            );
        }
    }
    println!(
        "  (~11 proposals per accepted coefficient, each consuming a\n   \
         {}-bit uniform — the sampler is XOF-bound by specification)",
        river::sample::PROB_BITS
    );

    section("Ring arithmetic  (d = 32)");
    for par in [RIVER_N8, RIVER_N256] {
        let q = par.q();
        let ring = Ring::new(q, par.d);
        let a: Vec<u64> = (0..par.d).map(|i| (i as u64 * 7919 + 13) % q).collect();
        let b: Vec<u64> = (0..par.d).map(|i| (i as u64 * 104_729 + 7) % q).collect();
        let ns = bench(20_000, 5, || ring.mul(&a, &b)[0]);
        row(&format!("{} schoolbook mul", par.name), ns, "");
        let ns = bench(200_000, 5, || ring.add(&a, &b)[0]);
        row(&format!("{} add", par.name), ns, "");

        // The matrix path.  `G'` and `A` are fixed by `rho` forever, so
        // the transform is paid once and the products amortise it; the
        // one-shot form is shown too because it is the case the design
        // note says loses, and it does.
        let rows = 8;
        let m = par.gprime_cols().min(64);
        let ring_bk = Ring::with_backend(q, par.d, m);
        let mat: Vec<Vec<Vec<u64>>> = (0..rows).map(|_| vec![a.clone(); m]).collect();
        let vec_: Vec<Vec<u64>> = vec![b.clone(); m];
        let pre = ring_bk.mat_to_ntt(&mat).expect("backend sized for this m");
        let ns = bench(100, 5, || ring_bk.mat_vec_ntt(&pre, &vec_).unwrap()[0][0]);
        row(
            &format!("{} mat_vec {rows}x{m} pre-transformed", par.name),
            ns,
            &format!("{:.0} ns per ring product", ns / (rows * m) as f64),
        );
        let ns = bench(100, 5, || ring.mat_vec(&mat, &vec_)[0][0]);
        row(
            &format!("{} mat_vec {rows}x{m} schoolbook", par.name),
            ns,
            &format!("{:.0} ns per ring product", ns / (rows * m) as f64),
        );
        let ns = bench(30, 5, || {
            ring_bk.mat_to_ntt(&mat).unwrap();
            0
        });
        row(
            &format!("{} mat_to_ntt {rows}x{m} (one-off)", par.name),
            ns,
            "amortised over every product with this matrix",
        );
    }

    section("SamMat — public matrix expansion");
    for par in [RIVER_N8, RIVER_N256] {
        let seed = [0u8; 32];
        let ns = bench(3, 5, || {
            sam_mat(&seed, par.q(), 2, 4, par.d, "RiVeR.A")[0][0][0]
        });
        row(
            &format!("{} A 2x4", par.name),
            ns,
            &format!("{:.0} ns per ring element", ns / 8.0),
        );
    }

    section("Codec");
    for par in PROFILES {
        let codec = RiVeRCodec::new(par);
        let pk: Vec<Vec<u64>> = (0..par.n)
            .map(|i| {
                (0..par.d)
                    .map(|j| ((i * j) as u64 * 31 + 7) % par.p)
                    .collect()
            })
            .collect();
        let blob = codec.pk_encode(&pk).unwrap();
        let ns = bench(2_000, 5, || codec.pk_encode(&pk).unwrap()[0] as u64);
        row(
            &format!("{} pk_encode", par.name),
            ns,
            &format!("{} B", blob.len()),
        );
        let ns = bench(2_000, 5, || codec.pk_decode(&blob).unwrap()[0][0]);
        row(&format!("{} pk_decode", par.name), ns, "");
    }

    section("OOM layer");
    println!("  One Com, one Prove attempt (accepted or not) and one Ver.");
    for par in [RIVER_TOY, RIVER_N8] {
        let fx = oom_fixture(&par);
        let oom = Oom::new(par, &fx.rho);
        let statement = OomStatement::new(&oom, &fx.a_mat, &fx.h_m, &fx.ring_pks, &fx.value)
            .expect("a well-formed statement");
        let ck = [1u8; 32];
        let rho_d = [2u8; 32];

        let ns = bench(3, 5, || {
            let mut xof = Xof::new(river::sample::DS_COMMIT, &[Part::Bytes(&rho_d)]);
            oom.com(&statement, fx.j_star, &mut xof).0.e[0][0]
        });
        row(&format!("{} OM.Com", par.name), ns, "two G' products");

        // one accepted proof, kept for the Ver line
        let mut accepted = None;
        let mut prove_ns = 0.0;
        for k in 0..64u8 {
            let mut xof = Xof::new(
                river::sample::DS_COMMIT,
                &[Part::Bytes(&rho_d), Part::Bytes(&[k])],
            );
            let (c, s) = oom.com(&statement, fx.j_star, &mut xof);
            let t0 = Instant::now();
            let pi = oom.prove(&fx.r, &c, &s, &ck, &rho_d, &mut xof);
            let elapsed = t0.elapsed().as_nanos() as f64;
            if let Some(pi) = pi {
                prove_ns = elapsed;
                accepted = Some(pi);
                break;
            }
        }
        let pi = accepted.expect("no accepting attempt in 64");
        row(
            &format!("{} OM.Prove (accepting)", par.name),
            prove_ns,
            "single sample",
        );
        let ns = bench(3, 5, || u64::from(oom.verify(&statement, &pi, &ck, &rho_d)));
        row(&format!("{} OM.Ver", par.name), ns, "");
    }

    section("Exact layer (opening backend)");
    println!("  Com, Prove, Ver and the two encodings.  Not zero knowledge.");
    for par in [RIVER_TOY, RIVER_N8] {
        let backend = OpeningBackend::new(par, &[0x11; 32]).expect("shipped profile");
        let (witness, x, z_eval) = exact_witness(&par);
        let mut xof = Xof::new(river::sample::DS_EXACT, &[Part::Bytes(b"bench")]);
        let (w, state) = backend.com(&witness, &mut xof).unwrap();
        let statement = ExactStatement {
            w: &w,
            z_eval: &z_eval,
            x: &x,
        };
        let sigma = backend.prove(&witness, &state);
        assert!(backend.verify(&statement, &sigma));

        let ns = bench(20, 5, || {
            let mut x = Xof::new(river::sample::DS_EXACT, &[Part::Bytes(b"b")]);
            backend.com(&witness, &mut x).unwrap().0.t0[0][0]
        });
        let ex = backend.ex();
        row(
            &format!("{} Pi_ex.Com", par.name),
            ns,
            &format!(
                "BDLOP over R_q~, d~ = {}, kappa = {}",
                ex.d_tilde(),
                ex.kappa()
            ),
        );
        let ns = bench(20, 5, || u64::from(backend.verify(&statement, &sigma)));
        row(&format!("{} Pi_ex.Ver", par.name), ns, "");
        let blob = backend.proof_encode(&w, &sigma).unwrap();
        let ns = bench(200, 5, || {
            backend.proof_encode(&w, &sigma).unwrap().len() as u64
        });
        row(
            &format!("{} pi_ex encode", par.name),
            ns,
            &format!("{} B ({} B worst case)", blob.len(), backend.proof_bytes()),
        );
    }

    bench_lanes_ring();
    bench_lanes_backend();

    section("Scheme");
    println!("  Eval and Verify per backend, at every published profile.");
    println!("  `Eval` is run over {EVAL_SEEDS} independent seeds.  The per-attempt");
    println!("  figure is TOTAL time over TOTAL attempts across those seeds, not");
    println!("  one run divided by its own attempt count: attempts abort at");
    println!("  different points and so do not cost the same, and the count");
    println!("  itself is geometric.  Even aggregated this is a small sample --");
    println!("  treat it as indicative, not as a converged mean.");
    println!("  `Verify` and `proof_decode` are medians of repeated batches.");
    for (par, kind) in [BackendKind::Opening, BackendKind::LanesExperimental]
        .into_iter()
        .flat_map(|k| {
            std::iter::once(RIVER_TOY)
                .chain(PROFILES.into_iter().filter(|p| !p.insecure_toy))
                .map(move |p| (p, k))
        })
    {
        // A ring is exactly `N` keys, with no
        // padding, so the fixture builds all of them.
        let Ok(scheme) = RiVeR::build(par, kind) else {
            continue; // gated backend; the section above says why
        };
        let pp = scheme.setup(&[0u8; 32]);
        let keys = ring_keys(&scheme, &pp, par.N);
        let ring: Vec<_> = keys.iter().map(|(_, pk)| pk.clone()).collect();
        let (sk, pk) = &keys[1];

        // Aggregate over independent seeds: sum the wall time and sum the
        // attempts, then divide once.  Dividing each run by its own attempt
        // count and averaging those quotients would weight a lucky
        // single-attempt run as heavily as a twenty-attempt one.
        let mut total_ns = 0.0f64;
        let mut total_attempts = 0u64;
        let mut last = None;
        for seed in 0..EVAL_SEEDS {
            let mut nonce = [0xAAu8; 32];
            nonce[0] = seed as u8;
            let t0 = Instant::now();
            let (v, pi, stats) = scheme
                .eval_deterministic(&pp, pk, sk, &ring, b"bench", &nonce)
                .expect("eval");
            total_ns += t0.elapsed().as_nanos() as f64;
            total_attempts += stats.attempts as u64;
            last = Some((v, pi));
        }
        let (v, pi) = last.expect("at least one seed");
        let blob = scheme.proof_encode(&pp, &pi).unwrap();
        row(
            &format!("{} Eval/attempt [{}]", par.name, kind.name()),
            total_ns / total_attempts as f64,
            &format!(
                "{total_attempts} attempts over {EVAL_SEEDS} seeds, \
                 {:.0} ms/proof mean, {} B proof",
                total_ns / EVAL_SEEDS as f64 / 1e6,
                blob.len()
            ),
        );
        let ns = bench(3, 5, || {
            u64::from(scheme.verify(&pp, &ring, b"bench", &v, &pi))
        });
        row(&format!("{} Verify [{}]", par.name, kind.name()), ns, "");
        let ns = bench(50, 5, || {
            scheme.proof_decode(&pp, &blob).unwrap().oom.x[0] as u64
        });
        row(
            &format!("{} proof_decode [{}]", par.name, kind.name()),
            ns,
            "",
        );
    }

    communication_sizes();

    section("Rejection-sampling budget");
    println!("  Coefficients drawn per attempted proof, and attempts per proof:");
    for par in PROFILES {
        let codec = RiVeRCodec::new(par);
        let gauss = coefficients(&par);
        let mu = par.mu_river();
        println!(
            "  {:<12} {:>7} Gaussian coeffs  x {:>5.2} attempts  = {:>9.0} draws \
             ({:>5} B worst-case proof)",
            par.name,
            gauss,
            mu,
            gauss as f64 * mu,
            codec.oom_max_bytes()
        );
    }
}

/// One deterministic, fully encoded LANES proof for every published profile.
///
/// Rice fields are variable length, so these are measurements, not constants.
/// The adjacent paper columns make the distinction explicit and keep framing
/// (eight bytes) from being mistaken for a model discrepancy.
fn communication_sizes() {
    section("Key sizes (measured, every published profile)");
    println!(
        "  {:<12} {:>6} {:>8} {:>8} {:>6}",
        "profile", "N", "pk B", "sk B", "ring B"
    );
    for par in PROFILES.into_iter().filter(|p| !p.insecure_toy) {
        let scheme = RiVeR::new_with(par, BackendKind::Opening);
        let pp = scheme.setup(&[0xC0; 32]);
        let (sk, pk) = scheme.keygen(&pp, &[0xC1; 32]).expect("keygen");
        let pk_len = scheme.codec().pk_encode(&pk).unwrap().len();
        let sk_len = scheme.codec().sk_encode(&sk).unwrap().len();
        println!(
            "  {:<12} {:>6} {:>8} {:>8} {:>6}",
            par.name,
            par.N,
            pk_len,
            sk_len,
            pk_len * par.N
        );
    }
    println!("  a ring is N public keys verbatim: no padding, no compression");

    // Both backends, every published profile.  `opening` is a mock whose
    // `|pi_ex|` is the cost of revealing the witness; `lanes-experimental`
    // is the candidate LANES layer at the paper's own widths.  It does not
    // transmit the witness, and is the one comparable to the paper's
    // stated 13.5 KB.
    for kind in [BackendKind::Opening, BackendKind::LanesExperimental] {
        section(&format!(
            "Communication (measured, one deterministic {} proof)",
            kind.name()
        ));
        if kind == BackendKind::Opening {
            println!(
                "  `opening` transmits the witness: its ex KB is the cost of \
                 the leak, NOT comparable to the paper's 13.5 KB."
            );
        } else {
            println!(
                "  candidate LANES layer at the paper's own widths; does not \
                 transmit the witness.  Its ex KB is what the paper's 13.5 KB \
                 should be compared against."
            );
        }
        println!(
            "  {:<12} {:>8} {:>8} {:>8} {:>10} {:>10} {:>5}",
            "profile", "OOM KiB", "ex KiB", "wire KiB", "ideal OOM", "ideal tot", "tries"
        );
        for par in PROFILES.into_iter().filter(|p| !p.insecure_toy) {
            let Ok(scheme) = RiVeR::build(par, kind) else {
                continue;
            };
            let pp = scheme.setup(&[0xC0; 32]);
            let keys = ring_keys(&scheme, &pp, par.N);
            let ring: Vec<_> = keys.iter().map(|(_, pk)| pk.clone()).collect();
            let (sk, pk) = &keys[0];
            let (value, proof, stats) = scheme
                .eval_deterministic(&pp, pk, sk, &ring, b"bench.sizes", &[0xC2; 32])
                .expect("deterministic size sample");
            assert!(scheme.verify(&pp, &ring, b"bench.sizes", &value, &proof));
            let blob = scheme.proof_encode(&pp, &proof).expect("proof encodes");
            let oom_len = u32::from_le_bytes(blob[0..4].try_into().unwrap()) as usize;
            let ex_at = 4 + oom_len;
            let ex_len = u32::from_le_bytes(blob[ex_at..ex_at + 4].try_into().unwrap()) as usize;
            assert_eq!(blob.len(), oom_len + ex_len + 8);
            println!(
                "  {:<12} {:>8.3} {:>8.3} {:>8.3} {:>10.3} {:>10.3} {:>5}",
                par.name,
                oom_len as f64 / 1024.0,
                ex_len as f64 / 1024.0,
                blob.len() as f64 / 1024.0,
                par.proof_size_oom_kb(),
                par.proof_size_total_kb(),
                stats.attempts
            );
        }
    }

    section("Ideal versus measured");
    println!(
        "  `ideal` columns are the paper's own size model, not this encoder: an\n  \
         entropy cost of h(sigma) = log2(4.13 sigma) bits per Gaussian\n  \
         coefficient with no coder named.  Golomb-Rice is this implementation's\n  \
         concrete approximation and lands about half a bit per coefficient above\n  \
         it, so measured OOM exceeds ideal OOM by a few tenths of a percent."
    );
    println!(
        "  `ideal tot` adds the paper's fixed |pi_ex| = {:.1} KB, stated for\n  \
         every profile with no field-by-field derivation, so it is a claim\n  \
         rather than something this encoder can reproduce.",
        RiVeRParams::EXACT_PROOF_KB
    );
    println!(
        "  The final table uses this same response split, so `ideal OOM` is\n  \
         the manuscript's displayed model directly."
    );
    println!("  every proof length is data-dependent: Rice coding moves the last few bytes");
}

fn bit_width(modulus: u64) -> usize {
    (u64::BITS - (modulus - 1).leading_zeros()) as usize
}

/// The ideal `L_ex` model printed by the final table.  This is kept next
/// to the measured encoding so a bandwidth change cannot leave the benchmark
/// comparing against the pre-optimisation 11.9 KB format again.
fn lanes_model_bits() -> f64 {
    // The paper charges the declared fixed coefficient width
    // ceil(log2(q~)) = 26, not the entropy log2(q~).
    let log_q = bit_width(QTILDE) as f64;
    let sigma_y = SIGMA_Y.0 as f64 / SIGMA_Y.1 as f64;
    let h_y = (4.13 * sigma_y).log2();
    N_TILDE as f64 * DTILDE as f64 * (log_q - D_DROP as f64)
        + (N_EX + AUX + 1) as f64 * DTILDE as f64 * log_q
        + RESPONSE_RANK as f64 * DTILDE as f64 * h_y
}

fn print_lanes_profile() {
    let q_width = bit_width(QTILDE);
    let t0_width = bit_width(T0_HIGH_MODULUS);
    let t0_before = N_TILDE * DTILDE * q_width;
    let t0_now = N_TILDE * DTILDE * t0_width;
    let challenge = DTILDE * 2;
    let hint = N_TILDE * DTILDE * 2;
    let model = lanes_model_bits();

    println!(
        "  executable set: d~={DTILDE}, split={}, (n~,l~,N_ex)=({N_TILDE},{ELL_TILDE},{N_EX}), \
         kappa={KAPPA}, response rank={RESPONSE_RANK}, q~={QTILDE}",
        river::lanes::ring::LSPLIT
    );
    println!(
        "  closed-form model: {model:.3} bits = {:.6} KB; paper states {} KB \
         with no field list",
        model / 8192.0,
        river::exact::LANES_STATED_BITS as f64 / river::exact::BITS_PER_KB as f64
    );
    println!(
        "  D={D_DROP} t0: {t0_before} -> {t0_now} bits; recovery metadata: \
         c {challenge} + hint {hint} bits"
    );
    println!("  parameters: the paper's own, the paper; byte-exact against river-py");
}

fn bench_lanes_ring() {
    // Never gated: the ring is what `crate::exact` commits over under
    // every backend, including `opening`.
    section(&format!(
        "LANES ring  (d~ = {DTILDE}, l = {}, q~ = {QTILDE} = 2^26 - 1151)",
        river::lanes::ring::LSPLIT
    ));
    let a_values: Vec<u64> = (0..DTILDE)
        .map(|i| (i as u64 * 7919 + 13) % QTILDE)
        .collect();
    let b_values: Vec<u64> = (0..DTILDE)
        .map(|i| (i as u64 * 104_729 + 7) % QTILDE)
        .collect();
    let a = lanes_ring::CoeffPoly::from_reduced(&a_values).unwrap();
    let b = lanes_ring::CoeffPoly::from_reduced(&b_values).unwrap();
    // Warm the lazily-built twiddle tables before starting the clock.
    let a_hat = lanes_ring::ntt(&a);
    let b_hat = lanes_ring::ntt(&b);

    let ns = bench(2_000, 5, || lanes_ring::ntt(&a).as_slice()[0]);
    row(
        "forward incomplete NTT",
        ns,
        &format!(
            "{DTILDE} -> {} degree-{} blocks",
            river::lanes::ring::LSPLIT,
            river::lanes::ring::SUBDEG
        ),
    );
    let ns = bench(2_000, 5, || lanes_ring::intt(&a_hat).as_slice()[0]);
    row("inverse incomplete NTT", ns, "");
    let ns = bench(5_000, 5, || {
        lanes_ring::ntt_mul(&a_hat, &b_hat).as_slice()[0]
    });
    row(
        "NTT-domain product",
        ns,
        &format!(
            "{} negacyclic degree-{} products",
            river::lanes::ring::LSPLIT,
            river::lanes::ring::SUBDEG
        ),
    );
    let ns = bench(1_000, 5, || lanes_ring::mul(&a, &b).as_slice()[0]);
    row(
        "coefficient product via NTT",
        ns,
        "includes both transforms",
    );
    let ns = bench(1_000, 5, || {
        lanes_ring::mul_schoolbook(&a, &b).as_slice()[0]
    });
    row(
        "coefficient product schoolbook",
        ns,
        "correctness reference",
    );

    // The reduction itself, measured rather than argued.  `q~ = 2^26 -
    // 1151` is pseudo-Mersenne, so a canonical product reduces with two
    // base-2^26 folds and one masked subtraction; the generic 128-bit
    // Barrett is what that replaced.  Both are driven over the same
    // inputs by `lanes::ring::tests`, so this is a cost comparison
    // between two things already known to agree.
    let products: Vec<u64> = (0..1024)
        .map(|i| {
            let x = (i as u64 * 2_654_435_761) % QTILDE;
            let y = (i as u64 * 40_503 + 7) % QTILDE;
            x * y
        })
        .collect();
    // Inputs *and* outputs cross `black_box`.  Without the input barrier
    // the whole fold is a pure function of a fixed array and may be
    // hoisted out of the timing loop — one evaluation divided by `n`;
    // without the output barrier it may vanish entirely.  The ratio below
    // is only meaningful with both.
    //
    // These are throughput numbers, not latency: the fold is a chain of
    // independent reductions and a superscalar core overlaps them, which
    // is what a stream of reductions costs — and a stream is what the
    // transforms actually issue.
    let reduce_bench = |label: &str, note: &str, f: fn(u64) -> u64| {
        let ns = bench(200, 5, || {
            let src = std::hint::black_box(&products);
            src.iter()
                .fold(0u64, |acc, &v| acc ^ f(std::hint::black_box(v)))
        });
        row(label, ns, &format!("{:.2} ns each; {note}", ns / 1024.0));
    };
    reduce_bench(
        "reduce x1024 (pseudo-Mersenne)",
        "two folds + a masked subtract",
        lanes_ring::reduce_product,
    );
    reduce_bench("reduce x1024 (Barrett)", "128-bit multiply-high", |v| {
        lanes_ring::barrett_reduce(v as u128)
    });
    reduce_bench(
        "reduce x1024 (pseudo-Mersenne, u128 entry)",
        "the wide path, five extra folds",
        |v| lanes_ring::reduce(v as u128),
    );
}

fn bench_lanes_backend() {
    // The production name is **gated** on security evidence.  Printing the reason and
    // returning keeps `make bench-lanes` a truthful report of what exists
    // rather than a set of numbers for an unported proof system.
    section("Exact layer (LANES backend)");
    // Measured through `LanesBackend::experimental`, the ungated name.
    //
    // The production name is gated on security *evidence*, not on
    // parameters — those are the paper's — so refusing to benchmark would
    // report nothing about a layer that runs, is byte-exact against
    // `river-py`, and is exactly the thing the paper's 13.5 KB figure
    // should be compared against.  The reason is printed alongside so the
    // numbers are not mistaken for a passing backend.
    if let Some(cause) = river::exact::lanes_gate_cause() {
        println!(
            "  NOTE: the production `lanes` name is gated ({cause}); measured \
             below through `lanes-experimental`, which is the same code."
        );
    }
    print_lanes_profile();

    for par in [RIVER_TOY, RIVER_N8] {
        let seed = [0x11; 32];
        let backend = LanesBackend::experimental(par, &seed).expect("experimental builds");
        let (witness, x, z_eval) = exact_witness(&par);
        let mut xof = Xof::new(river::sample::DS_EXACT, &[Part::Bytes(b"bench")]);
        let (w, state) = backend.com(&witness, &mut xof).unwrap();
        let statement = LanesStatement {
            w: &w,
            z_eval: &z_eval,
            x: &x,
        };
        let sigma = backend.prove(&statement, &state, &mut xof).unwrap();
        assert!(backend.verify(&statement, &sigma));

        let ns = bench(1, 3, || {
            // `proof_bytes()` does not itself read the commitment key. Make
            // the constructed backend opaque so LTO cannot discard key
            // expansion and leave this measuring only layout construction.
            std::hint::black_box(LanesBackend::experimental(par, &seed).unwrap()).proof_bytes()
                as u64
        });
        row(
            &format!("{} backend setup", par.name),
            ns,
            "commitment-key expansion",
        );
        let ns = bench(5, 3, || {
            let mut xf = Xof::new(river::sample::DS_EXACT, &[Part::Bytes(b"com")]);
            backend.com(&witness, &mut xf).unwrap().0.t0[0].as_slice()[0]
        });
        row(
            &format!("{} Pi_ex.Com", par.name),
            ns,
            &format!("BDLOP over R_q~, kappa = {KAPPA}"),
        );
        // The state contains the XOF continuation and is intentionally not
        // cloneable.  Rebuild Com for each repetition and name the timed
        // operation accordingly; the old label said Prove while timing both.
        let ns = bench(5, 3, || {
            let mut xf = Xof::new(river::sample::DS_EXACT, &[Part::Bytes(b"bench")]);
            let (w2, state2) = backend.com(&witness, &mut xf).unwrap();
            let statement2 = LanesStatement {
                w: &w2,
                z_eval: &z_eval,
                x: &x,
            };
            u64::from(backend.prove(&statement2, &state2, &mut xf).is_some())
        });
        row(
            &format!("{} Pi_ex.Com+Prove", par.name),
            ns,
            "one complete prover transcript",
        );
        let ns = bench(5, 3, || u64::from(backend.verify(&statement, &sigma)));
        row(&format!("{} Pi_ex.Ver", par.name), ns, "");

        let w_blob = backend.w_encode(&w).unwrap();
        let ns = bench(100, 3, || backend.w_encode(&w).unwrap().len() as u64);
        row(
            &format!("{} W encode", par.name),
            ns,
            &format!("{} B", w_blob.len()),
        );
        let ns = bench(100, 3, || {
            backend.w_decode(&w_blob).unwrap().t0[0].as_slice()[0]
        });
        row(&format!("{} W decode", par.name), ns, "");

        let blob = backend.proof_encode(&w, &sigma).unwrap();
        let ns = bench(50, 3, || {
            backend.proof_encode(&w, &sigma).unwrap().len() as u64
        });
        row(
            &format!("{} pi_ex encode", par.name),
            ns,
            &format!(
                "{} B = {:.3} KB ({} B worst case)",
                blob.len(),
                blob.len() as f64 / 1024.0,
                backend.proof_bytes()
            ),
        );
        let ns = bench(50, 3, || {
            backend.proof_decode(&blob).unwrap().1.c.as_slice()[0]
        });
        row(&format!("{} pi_ex decode", par.name), ns, "");
    }
}

/// Gaussian coefficients in one `pi_OOM`: `f_1`, `z^bin` and `z`.
fn coefficients(par: &RiVeRParams) -> usize {
    par.d * ((par.N - 1) + par.k_hat + par.r_dim())
}

/// A statement the OOM layer will accept, built the way `Eval` builds one.
struct OomFixture {
    a_mat: PolyMat,
    h_m: PolyVec,
    ring_pks: Vec<PolyVec>,
    value: Poly,
    r: PolyVec,
    rho: Vec<u8>,
    j_star: usize,
}

/// `n` distinct key pairs, which is what an exact-`N` ring needs.
fn ring_keys(scheme: &RiVeR, pp: &river::river::PublicParams, n: usize) -> Vec<(PolyVec, PolyVec)> {
    (0..n)
        .map(|i| {
            let mut seed = [0u8; 32];
            seed[0] = i as u8;
            seed[1] = (i >> 8) as u8;
            scheme.keygen(pp, &seed).expect("pp is this scheme's")
        })
        .collect()
}

fn oom_fixture(par: &RiVeRParams) -> OomFixture {
    let rq = Ring::new(par.q(), par.d);
    let j_star = 1;
    let rho = hash_bytes(32, b"bench.rho", &[Part::Bytes(&[7u8; 32])]);
    let a_mat = sam_mat(&rho, par.q(), par.n, par.ell, par.d, "RiVeR.A");

    let mut ring_pks = Vec::with_capacity(par.N);
    let mut sk_star = Vec::new();
    for i in 0..par.N {
        let mut xof = Xof::new(b"bench.kg", &[Part::Bytes(&[i as u8; 32])]);
        let s = uniform_beta_vec(&mut xof, par.beta, par.d, par.ell, par.q());
        let as_ = rq.mat_vec(&a_mat, &s);
        let t: PolyVec = as_.iter().map(|row| round_p(row, par.q0)).collect();
        if i == j_star {
            sk_star = s;
        }
        ring_pks.push(t);
    }

    let mut g = Xof::new(b"bench.G", &[Part::Bytes(b"m")]);
    let h_m: PolyVec = (0..par.ell)
        .map(|_| uniform_poly(&mut g, par.q(), par.d))
        .collect();
    let inner = rq.inner(&h_m, &sk_star);
    let value = round_p(&inner, par.q0);
    let e_eval_canonical = rounding_error(&rq, &inner, &value, par.q0);
    let e_eval_centered =
        to_centered_error(&e_eval_canonical, par.B_e()).expect("honest rounding error");
    let e_eval = rq.from_centered(&e_eval_centered);
    let as_star = rq.mat_vec(&a_mat, &sk_star);
    let mut r = sk_star;
    for i in 0..par.n {
        let canonical = rounding_error(&rq, &as_star[i], &ring_pks[j_star][i], par.q0);
        let centered = to_centered_error(&canonical, par.B_e()).expect("honest rounding error");
        r.push(rq.from_centered(&centered));
    }
    r.push(e_eval);

    OomFixture {
        a_mat,
        h_m,
        ring_pks,
        value,
        r,
        rho,
        j_star,
    }
}

/// A witness `R^_ex` admits, drawn the way `Eval` would draw it.
fn exact_witness(par: &RiVeRParams) -> (ExactWitness, Vec<i64>, Vec<i64>) {
    let mut x = Xof::new(river::sample::DS_EXACT, &[Part::Bytes(b"bench.w")]);
    let e_eval: Vec<i64> = (0..par.d)
        .map(|_| uniform_int(&mut x, par.q0) as i64 - par.B_e() as i64)
        .collect();
    let (num, den) = rational_sigma(par.sigma_m());
    let y_eval: Vec<i64> = (0..par.d)
        .map(|_| gaussian_int(&mut x, num, den, GAUSSIAN_TAILCUT))
        .collect();
    let x_hat = challenge_from_hash(
        par.d,
        par.w,
        par.gamma,
        par.q_hat,
        &[Part::Bytes(b"bench.x")],
    );
    let half = par.q_hat / 2;
    let x_c: Vec<i64> = x_hat
        .iter()
        .map(|&c| {
            if c > half {
                c as i64 - par.q_hat as i64
            } else {
                c as i64
            }
        })
        .collect();
    let prod = Ring::mul_int(
        &x_c.iter().map(|&c| c as i128).collect::<Vec<_>>(),
        &e_eval.iter().map(|&c| c as i128).collect::<Vec<_>>(),
    );
    let z_eval: Vec<i64> = (0..par.d)
        .map(|i| (prod[i] + y_eval[i] as i128) as i64)
        .collect();
    (ExactWitness { e_eval, y_eval }, x_c, z_eval)
}
