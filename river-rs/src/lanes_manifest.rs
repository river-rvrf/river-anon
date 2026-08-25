//! LANES parameter manifest -- **generated**, do not edit.
//!
//! `scripts/gen_lanes_manifest.py` writes this from
//! `../river-py/lanes_manifest.json`, the frozen parameter table.
//! `make lanes-manifest-check` requires the two to agree.
//!
//! This is a **typed projection**, not a copy: the source carries
//! prose (`how` strings, provenance notes, the security summary)
//! that has no place in a Rust const.  So `LANES_MANIFEST` also
//! carries `source_sha256`, the digest of the canonical JSON it was
//! projected from -- which is what makes projection *drift*
//! detectable: if the source moves and this file does not, the
//! digest differs and `make lanes-manifest-check` fails.
//!
//! [`crate::exact::validate_lanes_manifest`] compares each constant
//! against what [`crate::lanes::params`] consumes, so a table that
//! described a different parameter set would fail there rather than
//! as "proof bytes differ" in a cross-language vector.
//!
//! Possessing this table is **not** permission to run the backend;
//! see [`crate::exact::lanes_unavailable_reason`].

use crate::exact::{
    DimensionSpec, EstimatorSpec, LanesManifest, ManifestConstant, RankRoleSpec, RecoverySpec,
    ResponseBoundSpec, SamplerSpec, TranscriptField, TranscriptRound, WireField, WireSpec,
};

/// The absorbed fields, flattened, in hash order.
pub static TRANSCRIPT: [TranscriptField; 10] = [
    TranscriptField {
        name: "statement",
        domain_separator: "RiVeR.Exact.lanes.fs",
        hashed_form: "statement digest",
    },
    TranscriptField {
        name: "t0",
        domain_separator: "RiVeR.Exact.lanes.fs",
        hashed_form: "transmitted",
    },
    TranscriptField {
        name: "t",
        domain_separator: "RiVeR.Exact.lanes.fs",
        hashed_form: "transmitted",
    },
    TranscriptField {
        name: "w_high",
        domain_separator: "RiVeR.Exact.lanes.fs",
        hashed_form: "reconstructed",
    },
    TranscriptField {
        name: "t_g",
        domain_separator: "RiVeR.Exact.lanes.fs",
        hashed_form: "transmitted",
    },
    TranscriptField {
        name: "t_mp1",
        domain_separator: "RiVeR.Exact.lanes.fs",
        hashed_form: "transmitted",
    },
    TranscriptField {
        name: "t_mp2",
        domain_separator: "RiVeR.Exact.lanes.fs",
        hashed_form: "transmitted",
    },
    TranscriptField {
        name: "v",
        domain_separator: "RiVeR.Exact.lanes.fs",
        hashed_form: "reconstructed",
    },
    TranscriptField {
        name: "h",
        domain_separator: "RiVeR.Exact.lanes.fs",
        hashed_form: "transmitted",
    },
    TranscriptField {
        name: "v_prime",
        domain_separator: "RiVeR.Exact.lanes.fs",
        hashed_form: "reconstructed",
    },
];

static ROUND_0: [&str; 5] = ["statement", "t0", "t", "w_high", "t_g"];
static ROUND_1: [&str; 3] = ["t_mp1", "t_mp2", "v"];
static ROUND_2: [&str; 2] = ["h", "v_prime"];

/// The three Fiat-Shamir rounds, each with what precedes its
/// challenge.  A `|`-joined name is one `absorb` argument: the parts
/// are concatenated before hashing, and a port that hashed them
/// separately would derive different challenges.
pub static ROUNDS: [TranscriptRound; 3] = [
    TranscriptRound {
        challenge: "alpha",
        separator: ".alpha",
        absorbs: &ROUND_0,
    },
    TranscriptRound {
        challenge: "gamma",
        separator: ".gamma",
        absorbs: &ROUND_1,
    },
    TranscriptRound {
        challenge: "c",
        separator: ".c",
        absorbs: &ROUND_2,
    },
];

/// The serialized proof layout, in wire order, with coder
/// parameters.  `bits: None` marks the one variable-length field.
pub static WIRE_FIELDS: [WireField; 9] = [
    WireField {
        name: "t0",
        rows: 4,
        cols: 256,
        coder: "Uniform",
        bits: Some(10240),
        modulus: Some(513),
        bound: None,
        width_bits: Some(10),
        rice_k: None,
    },
    WireField {
        name: "t",
        rows: 6,
        cols: 256,
        coder: "Uniform",
        bits: Some(39936),
        modulus: Some(67107713),
        bound: None,
        width_bits: Some(26),
        rice_k: None,
    },
    WireField {
        name: "t_g",
        rows: 1,
        cols: 256,
        coder: "Uniform",
        bits: Some(6656),
        modulus: Some(67107713),
        bound: None,
        width_bits: Some(26),
        rice_k: None,
    },
    WireField {
        name: "t_mp1",
        rows: 1,
        cols: 256,
        coder: "Uniform",
        bits: Some(6656),
        modulus: Some(67107713),
        bound: None,
        width_bits: Some(26),
        rice_k: None,
    },
    WireField {
        name: "t_mp2",
        rows: 1,
        cols: 256,
        coder: "Uniform",
        bits: Some(6656),
        modulus: Some(67107713),
        bound: None,
        width_bits: Some(26),
        rice_k: None,
    },
    WireField {
        name: "h",
        rows: 1,
        cols: 256,
        coder: "Uniform",
        bits: Some(6656),
        modulus: Some(67107713),
        bound: None,
        width_bits: Some(26),
        rice_k: None,
    },
    WireField {
        name: "c",
        rows: 1,
        cols: 256,
        coder: "Signed",
        bits: Some(512),
        modulus: None,
        bound: Some(1),
        width_bits: Some(2),
        rice_k: None,
    },
    WireField {
        name: "hint",
        rows: 4,
        cols: 256,
        coder: "Signed",
        bits: Some(2048),
        modulus: None,
        bound: Some(1),
        width_bits: Some(2),
        rice_k: None,
    },
    WireField {
        name: "z",
        rows: 13,
        cols: 256,
        coder: "Rice",
        bits: None,
        modulus: None,
        bound: Some(3448),
        width_bits: None,
        rice_k: Some(8),
    },
];

/// The gated constants, each selected *by value*.
///
/// A **Paper** label on a retained value does not make the paper
/// have chosen it, so the gate compares these against what
/// [`crate::lanes::params`] consumes.
pub static CONSTANTS: [ManifestConstant; 6] = [
    ManifestConstant {
        name: "RECOVERY_BUCKETS",
        value: (16, 1),
        provenance: "Repair",
    },
    ManifestConstant {
        name: "RECOVERY_ERROR_BOUND",
        value: (2886972, 1),
        provenance: "Repair",
    },
    ManifestConstant {
        name: "SIGMA_R",
        value: (2901189, 524288),
        provenance: "Derived",
    },
    ManifestConstant {
        name: "SIGMA_Y",
        value: (255304631, 1048576),
        provenance: "Derived",
    },
    ManifestConstant {
        name: "Z_INF_BOUND",
        value: (3448, 1),
        provenance: "Repair",
    },
    ManifestConstant {
        name: "Z_NORM2_BOUND",
        value: (1578304756, 1),
        provenance: "Derived",
    },
];

/// The frozen table itself.
pub static LANES_MANIFEST: LanesManifest = LanesManifest {
    source_sha256: "64856e7e459faa74243547bb613e6d88cc17f9dc9131b59c83813cd7a52ba388",
    dimensions: DimensionSpec {
        d_tilde: 256,
        l_split: 64,
        sub_degree: 4,
        q_tilde: 67107713,
        q_tilde_bits: 26,
        n_tilde: 4,
        ell_tilde: 4,
        n_ex: 6,
        alpha: 3,
        d_drop: 17,
        w_hat: 44,
        w_tilde: 11,
        delta_stride: 4,
        n_lwe: 1024,
        m_lwe: 3328,
        block_slots: 64,
        block_payload: 32,
        message_blocks: 6,
    },
    rank_roles: RankRoleSpec {
        identity_rank: 4,
        tail_rank: 4,
        kappa: 17,
        response_rank: 13,
    },
    sampler: SamplerSpec {
        sigma_r: (2901189, 524288),
        sigma_y: (255304631, 1048576),
        epsilon_exponent: 100,
        convention: "standard deviation",
        tail_cut_r: 14,
        tail_cut_y: 14,
        prob_bits: 192,
    },
    response_bounds: ResponseBoundSpec {
        inf: 3448,
        l2: 1578304756,
        comparison: "<",
        population: "3328",
    },
    recovery: RecoverySpec {
        d_drop: 17,
        rounding: "power2round with a centred low part in (-2^(D-1), 2^(D-1)]",
        ties: "a tie at exactly 2^(D-1) goes to the low part, so high is not incremented",
        omitted_response_rows: 4,
        omitted_response_coefficients: 1024,
        omitted_t0_low_bits: 17408,
        recovery_carries: 1024,
        hint_alphabet: "{-1, 0, 1}, one per t_0 coefficient",
        limit: 2886972,
        failure_rule: "none: the bound is unconditional over the sampler's support, so recovery cannot fail for an honest prover",
        verification_rule: "apply the carry, then require the recovered w to match the challenge equation exactly",
        encoding: "signed 2-bit per coefficient, n~ d~ coefficients, byte-aligned once with the whole layout",
    },
    transcript: &TRANSCRIPT,
    rounds: &ROUNDS,
    wire: WireSpec {
        fields: &WIRE_FIELDS,
        total_bits: None,
        fixed_bits: 79360,
        kb_convention: "1 KB = 8192 bits",
        discrepancy: Some("13.5 KB is the paper's entropy estimate; this implementation reports the concrete Rice-coded payload field by field via LanesBackend.field_sizes(), so a small coding overhead is expected."),
    },
    estimator: EstimatorSpec {
        hint_mlwe_inputs: "{\"identity\": \"equals s_0^2 before the independent 2^-20 rounding of s_1 and s_2; the stored rational is derived from the rounded widths\", \"m\": 3328, \"n\": 1024, \"q\": 67107713, \"reduction\": \"1/sigma_MLWE^2 = 2(1/s_1^2 + w_hat^2/s_2^2)  [KLSS23]\", \"sigma_mlwe_sq\": \"548617212868547486114556975081/71666668028181948762926612480\"}",
        hint_mlwe_outputs: "{\"bits_by_reading\": {\"gaussian-parameter-as-stddev\": 134.32, \"standard-deviation\": 116.216}, \"delta_by_reading\": {\"gaussian-parameter-as-stddev\": 1.003611261724709, \"standard-deviation\": 1.0039959885516858}, \"paper_reports\": \"delta_MLWE = 1.0040\", \"status\": \"REPRODUCED by the standard-deviation reading; the alternate estimator-API conversion is retained as a sensitivity diagnostic.\"}",
        msis_inputs: "{\"length_bound\": \"B_MSIS = 8 w_hat beta'\", \"m\": 4352, \"q\": 67107713, \"rank\": 1024}",
        msis_outputs: "{\"B_MSIS\": \"15991561.7451824826921161023461938787121515276134079965128786\", \"bits\": 128.188, \"delta_closed_form\": 1.0037343664586467, \"paper_reports\": \"delta_MSIS = 1.0037\", \"published_B_MSIS_bits\": 128.188, \"status\": \"REPRODUCED, both by the closed form and by the estimator run\"}",
        challenge: "{\"paper_lanes_noninvertibility\": \"2^-90.5\", \"paper_outer\": \"2^-91.5\", \"status\": \"reported paper values; the optional large-table reproduction is outside the core implementation tests\"}",
    },
    constants: &CONSTANTS,
};
