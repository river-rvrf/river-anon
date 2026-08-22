#!/usr/bin/env python3
"""gen_manifest.py -- turn `../river-py/manifest.json` into `src/manifest.rs`.

The Python reference freezes every wire-visible numeric choice in
`manifest.json`: the pinned Gaussian rationals, the Rice parameters, the
exact integer bounds, the fixed field widths, and the layout accounting.
This script transcribes that file into a Rust module so that

  * an ordinary `cargo build` needs no Python and no float derivation --
    production reads the pinned table rather than recomputing
    `round(sigma * 2^20)` from an `f64` chain, which is where two
    implementations diverge by one unit in the last place;
  * `make manifest-check` regenerates and requires an empty diff, so the
    table cannot drift from the reference it was copied from;
  * `src/manifest.rs::tests` re-derives every entry from `params.rs` and
    compares, so the table cannot drift from *this* crate either.

Deliberate act, like `make kat-regen`: it moves the frozen description the
port is built against.  Run `make manifest-regen`.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IN = os.path.normpath(os.path.join(HERE, "..", "..", "river-py",
                                           "manifest.json"))
DEFAULT_OUT = os.path.normpath(os.path.join(HERE, "..", "src", "manifest.rs"))

#: Emitted in this order, so the generated table reads like the paper's.
PROFILE_ORDER = ["RiVeR-N8", "RiVeR-N16", "RiVeR-N64", "RiVeR-N128",
                 "RiVeR-N256", "RiVeR-TOY"]

#: Rust identifiers for the profile constants.
CONST_NAME = {
    "RiVeR-N8": "N8", "RiVeR-N16": "N16", "RiVeR-N64": "N64",
    "RiVeR-N128": "N128", "RiVeR-N256": "N256", "RiVeR-TOY": "TOY",
}


#: Emitted verbatim at the end of the generated module.  Static, so it is
#: reviewable here rather than in the output: these are the checks that
#: make the table a *cross-check* of the Rust rather than a second,
#: independent source of truth for it.
TESTS = r"""// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{optimal_rice_k, Coder};
    use crate::exact::ExactParams;
    use crate::params::{self, RiVeRParams};
    use crate::sample::rational_sigma;

    fn params_for(m: &ProfileManifest) -> RiVeRParams {
        params::get(m.profile).unwrap_or_else(|| panic!("no profile {}", m.profile))
    }

    #[test]
    fn every_profile_has_a_manifest_entry() {
        let mut names: Vec<&str> = params::PROFILES.iter().map(|p| p.name).collect();
        let mut listed: Vec<&str> = MANIFEST.iter().map(|m| m.profile).collect();
        names.sort_unstable();
        listed.sort_unstable();
        assert_eq!(names, listed);
    }

    /// The pinned rationals are what `rational_sigma` produces from the
    /// profile's own float width, in the reference's operation order.
    ///
    /// This is the check that lets production read the table instead of
    /// re-deriving `round_ties_even(sigma * 2^20)` from an `f64` chain:
    /// if the chain ever moves, this fails here rather than as a byte
    /// difference in a cross-language vector.
    #[test]
    fn gaussian_widths_rederive_from_the_parameters() {
        for m in MANIFEST {
            let p = params_for(&m);
            for (label, spec, sigma) in [
                ("f1", m.f1, p.sigma_a()),
                ("zb", m.zb, p.sigma_b()),
                ("zs", m.zs, p.sigma_s()),
                ("zm", m.zm, p.sigma_m()),
                ("y_eval", m.exact.y_eval, p.sigma_m()),
            ] {
                assert_eq!(
                    rational_sigma(sigma),
                    (spec.sigma_num, spec.sigma_den),
                    "{} {label}",
                    m.profile
                );
                assert_eq!(spec.sigma_den, SIGMA_DEN, "{} {label}", m.profile);
            }
        }
    }

    /// `k = floor(log2(sqrt(2 ln 2) sigma))`, over the exact rational.
    #[test]
    fn rice_parameters_rederive_from_the_pinned_widths() {
        for m in MANIFEST {
            for (label, spec) in [
                ("f1", m.f1),
                ("zb", m.zb),
                ("zs", m.zs),
                ("zm", m.zm),
                ("y_eval", m.exact.y_eval),
            ] {
                assert_eq!(
                    optimal_rice_k(spec.sigma_num, spec.sigma_den),
                    spec.rice_k,
                    "{} {label}",
                    m.profile
                );
            }
        }
    }

    /// No field's Rice parameter sits near a power-of-two boundary, so
    /// the 30-digit stand-in for `sqrt(2 ln 2)` cannot move one.
    ///
    /// Measured rather than assumed: the margin is the distance from
    /// `sqrt(2 ln 2) sigma` to the nearest power of two, relative.
    #[test]
    fn rice_parameters_are_far_from_a_boundary() {
        let c = (2.0f64 * 2.0f64.ln()).sqrt();
        for m in MANIFEST {
            for (label, spec) in [
                ("f1", m.f1),
                ("zb", m.zb),
                ("zs", m.zs),
                ("zm", m.zm),
                ("y_eval", m.exact.y_eval),
            ] {
                let sigma = spec.sigma_num as f64 / spec.sigma_den as f64;
                let v = c * sigma;
                let lo = (2f64).powi(spec.rice_k as i32);
                let hi = 2.0 * lo;
                let margin = ((v - lo) / lo).min((hi - v) / hi);
                assert!(
                    margin > 1e-3,
                    "{} {label}: sqrt(2 ln 2) sigma = {v} is {margin} from a \\
                     boundary at k = {}",
                    m.profile,
                    spec.rice_k
                );
            }
        }
    }

    /// Each bound is the largest coefficient the verifier can accept —
    /// `floor(sqrt(bound_sq))` — so the encoder's cap and the acceptance
    /// test agree by construction.
    #[test]
    fn bounds_are_the_largest_value_that_can_pass() {
        for m in MANIFEST {
            let p = params_for(&m);
            for (label, bound, sq) in [
                ("f1", m.f1.bound, p.f1_inf_bound_sq()),
                ("zb", m.zb.bound, p.zb_inf_bound_sq()),
                ("zs", m.zs.bound, p.zs_inf_bound_sq()),
                ("zm", m.zm.bound, p.zm_inf_bound_sq()),
            ] {
                assert_eq!(sq.floor_sqrt() as i64, bound, "{} {label}", m.profile);
                let b = bound as u128;
                assert!(!sq.exceeded_by(b * b), "{} {label}: bound rejected", m.profile);
                let n = b + 1;
                assert!(sq.exceeded_by(n * n), "{} {label}: bound+1 accepted", m.profile);
            }
            // `x` is the challenge, bounded by `gamma` rather than by a
            // Gaussian tail.
            assert_eq!(m.x.bound, p.gamma as i64, "{} x", m.profile);
            // `y_eval` is *not* bounded by `6 sigma_m`: it is
            // `z_eval - x e_eval`, so an accepted transcript reaches
            // `6 sigma_m + ||x||_1 ||e_eval||_inf`.
            assert_eq!(
                m.exact.y_eval.bound,
                m.zm.bound + (p.w as u64 * p.gamma * p.B_e()) as i64,
                "{} y_eval",
                m.profile
            );
        }
    }

    /// Every fixed-width field is exactly as wide as its bound needs.
    #[test]
    fn fixed_widths_rederive_from_their_bounds() {
        for m in MANIFEST {
            for (label, spec) in [("B", m.b_hi), ("x", m.x)] {
                match Coder::signed(spec.bound) {
                    Coder::Signed { width, bound } => {
                        assert_eq!(width, spec.width, "{} {label} width", m.profile);
                        assert_eq!(bound, spec.bound, "{} {label} bound", m.profile);
                    }
                    other => panic!("{} {label}: {other:?}", m.profile),
                }
            }
        }
    }

    /// The longest unary run a decoder may accept, `(bound >> k) + 1`.
    #[test]
    fn max_high_matches_the_coder() {
        for m in MANIFEST {
            for (label, spec) in [
                ("f1", m.f1),
                ("zb", m.zb),
                ("zs", m.zs),
                ("zm", m.zm),
                ("y_eval", m.exact.y_eval),
            ] {
                assert_eq!(
                    spec.max_high,
                    (spec.bound >> spec.rice_k) as u64 + 1,
                    "{} {label}",
                    m.profile
                );
            }
        }
    }

    /// The exact layer's frozen dimensions are the ones `ExactParams`
    /// derives — including the two ranks whose roles have been reversed
    /// twice, which are both 4 here and so cannot be told apart by
    /// a numeric comparison of the constants alone.
    #[test]
    fn exact_dimensions_match_the_parameters() {
        for m in MANIFEST {
            let ex = ExactParams::new(&params_for(&m)).expect("exact params");
            assert_eq!(ex.d_tilde(), m.exact.d_tilde, "{}", m.profile);
            assert_eq!(ex.l_split(), m.exact.l_split, "{}", m.profile);
            assert_eq!(ex.q_tilde(), m.exact.q_tilde, "{}", m.profile);
            assert_eq!(ex.kappa(), m.exact.kappa, "{}", m.profile);
            assert_eq!(ex.t0_rows(), m.exact.identity_rank, "{}", m.profile);
            assert_eq!(ex.tail_rank(), m.exact.tail_rank, "{}", m.profile);
            assert_eq!(ex.response_rank(), m.exact.response_rank, "{}", m.profile);
            assert_eq!(ex.n_ex(), m.exact.n_ex, "{}", m.profile);
            assert_eq!(ex.block_slots(), m.exact.block_slots, "{}", m.profile);
            assert_eq!(ex.block_used(), m.exact.block_used, "{}", m.profile);
            assert_eq!(crate::exact::RADIX_WEIGHTS, m.exact.radix_weights);
        }
    }

    /// A profile that merely *says* it is `RiVeR-N8` does not get N8's
    /// frozen widths.
    ///
    /// `for_profile` matches on the name alone, which is fine for a test
    /// holding a canonical profile and is a hazard in production: a
    /// modified profile would sample at the frozen widths while every
    /// bound came from its own fields — a sampler and an acceptance test
    /// derived from different parameter sets.  `for_params` is what the
    /// codec, the OOM layer and the exact backend call.
    #[test]
    fn a_modified_profile_does_not_inherit_its_namesakes_manifest() {
        for canonical in params::PROFILES {
            assert!(
                for_params(&canonical).is_some(),
                "{} lost its own manifest",
                canonical.name
            );
            // One field at a time, each of them wire-visible.
            let mutations: [fn(&mut RiVeRParams); 6] = [
                |p| p.ell += 1,
                |p| p.n += 1,
                |p| p.k_hat += 1,
                |p| p.n_hat += 1,
                |p| p.phi_s += 1,
                |p| p.gamma += 1,
            ];
            for mutate in mutations {
                let mut modified = canonical;
                mutate(&mut modified);
                assert_eq!(modified.name, canonical.name, "the name is unchanged");
                assert!(
                    for_params(&modified).is_none(),
                    "{}: a modified profile kept the frozen manifest",
                    canonical.name
                );
                // and the name-keyed lookup still answers, which is why
                // production must not use it
                assert!(for_profile(modified.name).is_some());
            }
        }
    }

}
"""


def gaussian(field, layout_field):
    """A `GaussianSpec` literal from a manifest field entry."""
    return (f"GaussianSpec {{ sigma_num: {field['sigma_num']}, "
            f"sigma_den: {field['sigma_den']}, rice_k: {field['rice_k']}, "
            f"bound: {field['bound']}, max_high: {layout_field['max_high']} }}")


def signed(field, layout_field):
    assert field["bound"] == layout_field["bound"], field
    assert field["width_bits"] == layout_field["width"], field
    return f"SignedSpec {{ bound: {field['bound']}, width: {field['width_bits']} }}"


def render(manifest):
    g = manifest["global"]
    out = []
    w = out.append

    w('//! Frozen wire-visible numeric choices, per profile and field.')
    w('//!')
    w('//! **Generated by `scripts/gen_manifest.py` from')
    w('//! `../river-py/manifest.json`.  Do not edit by hand** -- run')
    w('//! `make manifest-regen`, and `make manifest-check` to verify the')
    w('//! checked-in copy still matches the reference.')
    w('//!')
    w('//! Everything here is a value the two implementations must agree on')
    w('//! *exactly* or they produce different bytes, and none of it is')
    w('//! stated by the paper:')
    w('//!')
    w('//! * **`(sigma_num, sigma_den)`** per Gaussian field.  The paper\'s')
    w('//!   widths are irrational, so every implementation pins some')
    w('//!   rational; the reference pins `round_ties_even(sigma * 2^20)`.')
    w('//!   The rounding removes only the *final* float error, so the input')
    w('//!   has to be computed in the same operation order -- which is')
    w('//!   exactly the fragility this table removes from the build.')
    w('//! * **`rice_k`** per Gaussian field.  `k = floor(log2(sqrt(2 ln 2)')
    w('//!   sigma))` is wire-visible, and one off is a different encoding')
    w('//!   rather than a rounding difference.')
    w('//! * **`bound`** per response field: the largest coefficient that can')
    w('//!   pass verification, `floor(sqrt(bound_sq))` -- exact, so the')
    w('//!   encoder\'s cap and the acceptance test cannot disagree.')
    w('//! * **`width`** per fixed-width field, in bits.')
    w('//!')
    w('//! The module\'s own tests re-derive every entry from')
    w('//! [`crate::params`] and the coders, so a divergence between this')
    w('//! table and the code that consumes it fails here rather than as')
    w('//! "proof bytes differ" in a cross-language vector.')
    w('')
    w('#![allow(non_upper_case_globals)]')
    w('')
    w('/// Denominator every pinned Gaussian width shares: `2^20`.')
    w(f'pub const SIGMA_DEN: u64 = {g["sigma_den"]};')
    w('/// Fixed-point width of the rejection-sampling acceptance test.')
    w(f'pub const PROB_BITS: u32 = {g["prob_bits"]};')
    w('/// Where the Gaussian proposal is truncated, in units of sigma.')
    w(f'pub const GAUSSIAN_TAILCUT: u64 = {g["gaussian_tailcut"]};')
    w('/// Verifier bound multiplier: `6 sigma`.')
    w(f'pub const VERIFIER_TAILCUT: u64 = {g["verifier_tailcut"]};')
    w('/// `sqrt(2 ln 2)` to 30 significant figures, as an exact rational.')
    w('///')
    w('/// The Rice parameter is `floor(log2(sqrt(2 ln 2) sigma))` and `k` is')
    w('/// wire-visible, so *some* rational stands in for the irrational')
    w('/// constant; what matters is that the standing-in never moves `k`.')
    w('/// Thirty digits makes the window in which it could `1e-30` wide, and')
    w('/// a test measures the actual distance to the nearest boundary at')
    w('/// every field of every profile.')
    w(f'pub const RICE_CONST_NUM_DEC: &str = "{g["rice_const_num"]}";')
    w(f'pub const RICE_CONST_DEN_DEC: &str = "{g["rice_const_den"]}";')
    w('')
    w('/// One Rice-coded Gaussian field.')
    w('#[derive(Debug, Clone, Copy, PartialEq, Eq)]')
    w('pub struct GaussianSpec {')
    w('    /// Numerator of the pinned width; the denominator is [`SIGMA_DEN`].')
    w('    pub sigma_num: u64,')
    w('    pub sigma_den: u64,')
    w('    /// `floor(log2(sqrt(2 ln 2) sigma))`.')
    w('    pub rice_k: u32,')
    w('    /// Largest coefficient that can pass verification.')
    w('    pub bound: i64,')
    w('    /// Longest unary run a decoder may accept: `(bound >> k) + 1`.')
    w('    pub max_high: u64,')
    w('}')
    w('')
    w('/// One fixed-width signed field.')
    w('#[derive(Debug, Clone, Copy, PartialEq, Eq)]')
    w('pub struct SignedSpec {')
    w('    pub bound: i64,')
    w('    pub width: u32,')
    w('}')
    w('')
    w('/// The exact layer\'s frozen dimensions and its one Gaussian field.')
    w('#[derive(Debug, Clone, Copy, PartialEq, Eq)]')
    w('pub struct ExactSpec {')
    w('    /// Backend the sizes below were measured against.')
    w('    pub backend: &\'static str,')
    w('    pub d_tilde: usize,')
    w('    pub l_split: usize,')
    w('    pub q_tilde: u64,')
    w('    /// `n~ + l~ + N_ex + alpha`.')
    w('    pub kappa: usize,')
    w('    /// `l~`: rows of `t_0` and width of `B_0`\'s identity block.')
    w('    pub identity_rank: usize,')
    w('    /// `n~`: the shared random tail every `b_i` draws from.')
    w('    pub tail_rank: usize,')
    w('    /// `kappa - l~`: the part of the opening actually transmitted.')
    w('    pub response_rank: usize,')
    w('    /// Message ring elements, one per 64-slot block.')
    w('    pub n_ex: usize,')
    w('    pub block_slots: usize,')
    w('    pub block_used: usize,')
    w('    pub radix_weights: [i64; 4],')
    w('    /// `y_eval`, the one Gaussian field in the exact proof.')
    w('    pub y_eval: GaussianSpec,')
    w('    /// `|W|`, which is all uniform and therefore exact.')
    w('    pub w_bytes: usize,')
    w('    /// Worst-case `|pi_ex|`; Rice makes the real length vary.')
    w('    pub proof_bytes_max: usize,')
    w('}')
    w('')
    w('/// Every wire-visible numeric choice for one profile.')
    w('#[derive(Debug, Clone, Copy, PartialEq, Eq)]')
    w('pub struct ProfileManifest {')
    w('    pub profile: &\'static str,')
    w('    /// High bits of the selector commitment `B`.  Signed since the')
    w('    /// The paper takes `[[.]]_K` on the centred representative.')
    w('    pub b_hi: SignedSpec,')
    w('    pub x: SignedSpec,')
    w('    pub f1: GaussianSpec,')
    w('    pub zb: GaussianSpec,')
    w('    pub zs: GaussianSpec,')
    w('    pub zm: GaussianSpec,')
    w('    /// Row counts of the OOM layout, in wire order after `x`.')
    w('    pub rows: OomRows,')
    w('    /// Worst-case and best-case `|pi_OOM|`.')
    w('    pub oom_max_bytes: usize,')
    w('    pub oom_min_bytes: usize,')
    w('    pub exact: ExactSpec,')
    w('}')
    w('')
    w('/// Row counts of every multi-row OOM field.')
    w('#[derive(Debug, Clone, Copy, PartialEq, Eq)]')
    w('pub struct OomRows {')
    w('    pub b_hi: usize,')
    w('    pub f1: usize,')
    w('    pub zb: usize,')
    w('    pub zs: usize,')
    w('    pub zm: usize,')
    w('}')
    w('')
    w('/// The OOM layout\'s field order on the wire.')
    order = manifest["profiles"][PROFILE_ORDER[0]]["layouts"]["oom"]["order"]
    w(f'pub const OOM_FIELD_ORDER: [&str; {len(order)}] = [')
    w('    ' + ', '.join(f'"{name}"' for name in order) + ',')
    w('];')
    w('')
    w('/// The framing every proof carries: two length-prefixed blocks.')
    fr = manifest["profiles"][PROFILE_ORDER[0]]["framing"]
    w(f'pub const FRAMING_PREFIX_BYTES: usize = {fr["length_prefix_bytes"]};')
    w(f'pub const FRAMING_TOTAL_BYTES: usize = {fr["total_framing_bytes"]};')
    w('pub const FRAMING_BLOCK_ORDER: [&str; %d] = [%s];'
      % (len(fr["block_order"]),
         ', '.join(f'"{b}"' for b in fr["block_order"])))
    w('')

    for name in PROFILE_ORDER:
        prof = manifest["profiles"][name]
        f = prof["fields"]
        lay = prof["layouts"]["oom"]["fields"]
        ex = prof["exact"]
        exlay = prof["layouts"]
        rows = prof["rows"]
        ident = CONST_NAME[name]
        w(f'/// Frozen wire manifest for `{name}`.')
        w(f'pub const {ident}: ProfileManifest = ProfileManifest {{')
        w(f'    profile: "{name}",')
        w(f'    b_hi: {signed(f["B"], lay["B"])},')
        w(f'    x: {signed(f["x"], lay["x"])},')
        for key in ("f1", "zb", "zs", "zm"):
            w(f'    {key}: {gaussian(f[key], lay[key])},')
        w('    rows: OomRows {')
        w(f'        b_hi: {rows["B"]}, f1: {rows["f1"]}, zb: {rows["zb"]},')
        w(f'        zs: {rows["zs"]}, zm: {rows["zm"]},')
        w('    },')
        w(f'    oom_max_bytes: {prof["oom_max_bytes"]},')
        w(f'    oom_min_bytes: {prof["layouts"]["oom"]["min_bytes"]},')
        w('    exact: ExactSpec {')
        w(f'        backend: "{ex["backend"]}",')
        w(f'        d_tilde: {ex["d_tilde"]},')
        w(f'        l_split: {ex["l_split"]},')
        w(f'        q_tilde: {ex["q_tilde"]},')
        w(f'        kappa: {ex["kappa"]},')
        w(f'        identity_rank: {ex["identity_rank"]},')
        w(f'        tail_rank: {ex["tail_rank"]},')
        w(f'        response_rank: {ex["response_rank"]},')
        w(f'        n_ex: {ex["N_ex"]},')
        w(f'        block_slots: {ex["block_slots"]},')
        w(f'        block_used: {ex["block_used"]},')
        w('        radix_weights: [%s],'
          % ', '.join(str(v) for v in ex["radix_weights"]))
        exf = exlay["exact_opening"]["fields"]["y_eval"]
        w('        y_eval: %s,'
          % gaussian(f["y_eval"], exf))
        w(f'        w_bytes: {exlay["exact_W"]["max_bytes"]},')
        w(f'        proof_bytes_max: {ex["proof_bytes_max"]},')
        w('    },')
        w('};')
        w('')

    w(f'/// Every profile, in the order [`crate::params::PROFILES`] lists them.')
    w(f'pub const MANIFEST: [ProfileManifest; {len(PROFILE_ORDER)}] = [')
    w('    ' + ', '.join(CONST_NAME[n] for n in PROFILE_ORDER) + ',')
    w('];')
    w('')
    w('/// The frozen manifest for a set of parameters, or `None`.')
    w('///')
    w('/// **This is the one production may use.**  It matches the *whole*')
    w('/// canonical profile, not its name: a caller can build a')
    w('/// `RiVeRParams` that says `"RiVeR-N8"` and carries a different')
    w('/// `ell`, and handing that one N8\'s frozen widths would make it')
    w('/// sample at N8 while `check()` validated the modified fields — a')
    w('/// sampler and a bound derived from different profiles, with')
    w('/// nothing to notice.')
    w('///')
    w('/// A profile that does not match exactly gets `None`, and the')
    w('/// caller derives instead: slower to reason about, but always')
    w('/// consistent with the parameters in hand.')
    w('pub fn for_params(par: &crate::params::RiVeRParams) -> Option<&\'static ProfileManifest> {')
    w('    let canonical = crate::params::get(par.name)?;')
    w('    if canonical != *par {')
    w('        return None;')
    w('    }')
    w('    MANIFEST.iter().find(|m| m.profile == par.name)')
    w('}')
    w('')
    w('/// The frozen manifest for a profile *name*.')
    w('///')
    w('/// For tests and tooling that already hold a canonical profile.')
    w('/// Production wants [`for_params`], which checks that the name and')
    w('/// the parameters agree.')
    w('pub fn for_profile(name: &str) -> Option<&\'static ProfileManifest> {')
    w('    MANIFEST.iter().find(|m| m.profile == name)')
    w('}')
    w('')
    w(TESTS)
    return "\n".join(out) + "\n"


def rustfmt(text):
    """Normalise through `rustfmt`, so `--check` is stable under `cargo fmt`.

    Without this the first `cargo fmt` after a regeneration reflows the
    generated file and `--check` reports a difference that is not one.
    `rustfmt` is part of every Rust toolchain that can build this crate, so
    its absence is worth saying out loud rather than silently skipping.
    """
    exe = shutil.which("rustfmt")
    if exe is None:
        print("warning: rustfmt not found; emitting unformatted output",
              file=sys.stderr)
        return text
    out = subprocess.run([exe, "--edition", "2021", "--emit", "stdout"],
                         input=text, capture_output=True, text=True)
    if out.returncode != 0:
        print(f"warning: rustfmt failed ({out.returncode}); emitting "
              f"unformatted output\n{out.stderr}", file=sys.stderr)
        return text
    # `--emit stdout` prefixes a `// rustfmt-...` banner line.
    body = out.stdout
    if body.startswith("//"):
        body = body.split("\n", 1)[1]
    return body.lstrip("\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=DEFAULT_IN)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the checked-in file differs")
    args = ap.parse_args()

    with open(args.input) as fh:
        manifest = json.load(fh)
    text = rustfmt(render(manifest))

    if args.check:
        try:
            with open(args.out) as fh:
                current = fh.read()
        except FileNotFoundError:
            print(f"{args.out}: missing", file=sys.stderr)
            return 1
        if current != text:
            print(f"{args.out} differs from {args.input}; "
                  f"run `make manifest-regen`", file=sys.stderr)
            return 1
        print(f"{args.out}: up to date with {args.input}")
        return 0

    with open(args.out, "w") as fh:
        fh.write(text)
    print(f"wrote {args.out} from {args.input}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
