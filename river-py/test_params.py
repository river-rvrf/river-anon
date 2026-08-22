"""
test_params.py -- Check the parameter profiles against the paper.

The reference values below are transcribed from Table
`tab:river-final-all-params` (Appendix "Detailed Parameter Setting") and
Table `tab:river-d32-concrete` (Section "Parameter Setting") of the
The paper revision.

Scope, precisely.  The *input* columns -- `(phi_a, phi_s)`, `(n, ell)`,
`(n_hat, k_hat)`, `(tau_g0, tau_g1)`, the moduli bit lengths -- are profile
literals in `params.py`; they are inputs, not predictions.  What is checked
here is that every *derived* column follows from them through `BoundGen` and
the paper's own formulas: `B`, `B_s`, `B_g0`, `B_g1`, `beta_SIS,1`,
`beta_SIS,2`, `beta_SIS`, `beta_sel,inf`, the repeat bound, `|pi_OOM|` and
the total.

The table rounds ordinary non-integral values, so the comparison here is
"rounds to the printed value" rather than a relative tolerance: two
significant figures for the bound columns, one decimal for the repeat bound
and the sizes.  That is stricter than a tolerance and does not silently
accept a value that would print differently.

Not reproduced: the root-Hermite-factor figures (they need a lattice
estimator), the Groebner-basis estimates, and the empirical non-abort data
behind `epsilon_g^U`.

Some tests pin *disagreements* on purpose -- places where the draft
contradicts itself.  They are named `test_paper_*_is_*` and are not defects
in this implementation; normalising them away would hide the finding.
"""

import dataclasses
import math
import re
from fractions import Fraction

from params import (BOUNDGEN_ORDER, BOUNDGEN_ORDER_SWAPPED, PROFILES,
                    TOY_PARAMS, _TAU, _TAU_DISPLAYED, get, is_prime,
                    largest_prime_below, verify_moduli)

PUBLISHED = [8, 16, 64, 128, 256]


#: N -> (B, B_s, (B_g0, B_g1), beta_SIS1, beta_SIS2, beta_SIS, beta_sel_inf)
#: exactly as printed, to two significant figures.
PAPER_BOUNDS = {
    8:   (2.0e4, 8.6e5, (3.9e9, 7.2e8), 3.0e9, 3.0e9, 6.2e12, 8.1e12),
    16:  (2.0e4, 8.7e5, (1.3e10, 1.3e9), 2.6e9, 2.6e9, 5.3e12, 2.7e13),
    64:  (2.1e4, 8.6e5, (3.9e10, 1.0e9), 2.8e9, 2.8e9, 5.7e12, 8.0e13),
    128: (2.1e4, 8.6e5, (4.0e10, 5.4e8), 4.0e9, 4.0e9, 8.1e12, 8.1e13),
    256: (2.1e4, 8.7e5, (6.6e10, 4.9e8), 4.8e9, 4.8e9, 9.8e12, 1.4e14),
}

#: N -> (repeat bound, |pi_OOM| KB, total KB), to one decimal place.
#:
#: The two size columns are the paper's, and the paper's are computed with a
#: communication formula that charges a *different* response split; see
#: `test_the_published_proof_sizes_use_a_different_response_split`.  They
#: are therefore compared against `proof_size_oom_kb_paper`, not against what
#: this implementation transmits.
PAPER_SIZES = {
    8:   (8.3, 19.6, 33.1),
    16:  (8.4, 21.0, 34.5),
    64:  (8.6, 25.0, 38.5),
    128: (8.6, 28.5, 42.0),
    256: (8.5, 35.6, 49.1),
}

#: What the transmitted layout actually costs, `(|pi_OOM| KB, total KB)`.
MEASURED_SIZES = {
    8:   (20.1, 33.6),
    16:  (21.4, 34.9),
    64:  (25.5, 39.0),
    128: (29.1, 42.6),
    256: (36.2, 49.7),
}

#: The main table's per-profile module ranks and modulus bit lengths.
PAPER_INPUTS = {
    8:   ((32, 26), (44, 54), 44, (42, 46, 44)),
    16:  ((40, 22), (41, 59), 48, (43, 49, 46)),
    64:  ((34, 24), (44, 54), 44, (50, 51, 48)),
    128: ((24, 34), (45, 54), 44, (50, 51, 48)),
    256: ((22, 40), (42, 59), 48, (49, 52, 49)),
}

#: The concrete moduli, `N -> (p, q, q_hat)`, from
#: `tab:river-concrete-moduli`.  **Paper**; before it the
#: paper printed only bit lengths and this tree derived them.
PAPER_MODULI = {
    8:   (17592186043877, 1073123348676497, 8796093022237),
    16:  (281474976710597, 17169973579346417, 35184372088997),
    64:  (17592186043877, 1073123348676497, 140737488355333),
    128: (17592186043877, 1073123348676497, 140737488355333),
    256: (281474976710597, 17169973579346417, 281474976710677),
}


# ---- helpers -------------------------------------------------------------

def sig(x, digits=2):
    """`x` rounded to `digits` significant figures, as a (mantissa, exp) pair."""
    if x == 0:
        return (0.0, 0)
    e = math.floor(math.log10(abs(x)))
    return (round(x / 10 ** e, digits - 1), e)


def round1_half_up(fr):
    """Exact round-half-up to one decimal place, as the table does."""
    return Fraction(math.floor(fr * 10 + Fraction(1, 2)), 10)


def published():
    return [(N, get(f"RiVeR-N{N}")) for N in PUBLISHED]


# ---- provenance ----------------------------------------------------------



def test_moduli_are_rederivable():
    assert verify_moduli() == []


def test_the_concrete_moduli_match_the_published_table():
    """the paper prints `p`, `q` and `q_hat` outright.

    Before it the paper gave only bit lengths and this tree derived all
    three by one rule -- largest prime below `2^bits` that is 5 mod 8.  That
    rule is right for `p` and wrong for `q_hat`, which the paper takes as
    the smallest prime *above* `2^{bits-1}`, roughly half the value.  Since
    `q_hat` enters `b_B` and hence the wire, the difference was not
    cosmetic.  Both rules are re-derived in `verify_moduli`.
    """
    for N, par in published():
        p, q, q_hat = PAPER_MODULI[N]
        assert (par.p, par.q, par.q_hat) == (p, q, q_hat), par.name
        assert par.q == 61 * par.p


def test_moduli_are_prime_and_split_in_two():
    for _, par in published() + [(TOY_PARAMS.N, TOY_PARAMS)]:
        for label, value in (("p", par.p), ("q_0", par.q0),
                             ("q_hat", par.q_hat)):
            assert is_prime(value), f"{par.name}: {label} not prime"
            assert value % 8 == 5, f"{par.name}: {label} not 5 mod 8"


def test_modulus_bit_lengths_match_the_table():
    for N, par in published():
        _, _, log_p, (_, _, log_qhat) = PAPER_INPUTS[N]
        assert par.p.bit_length() == log_p
        assert par.q_hat.bit_length() == log_qhat
        # The table prints log2 q to one decimal.
        assert round(math.log2(par.q), 1) == round(log_p + math.log2(61), 1)


def test_profile_inputs_match_the_table():
    for N, par in published():
        (phi_a, phi_s), (n, ell), _, (n_hat, k_hat, _) = PAPER_INPUTS[N]
        assert (par.phi_a, par.phi_s) == (phi_a, phi_s)
        assert (par.n, par.ell) == (n, ell)
        assert (par.n_hat, par.k_hat) == (n_hat, k_hat)


# ---- common parameters ---------------------------------------------------

def test_common_parameters_are_shared():
    """`(d,q_0,w,gamma,B_e,beta) = (32,61,32,16,30,1)`, `r'=1`, `phi_m=32`,
    `phi_b=2`, `K_b=5`, `K_a=28` -- the table's caption."""
    for _, par in published():
        assert (par.d, par.q0, par.w, par.gamma, par.B_e, par.beta) == \
            (32, 61, 32, 16, 30, 1)
        assert par.r_prime == 1
        assert par.phi_m == 32 and par.phi_b == 2
        assert (par.K_b, par.K_a) == (5, 28)


def test_challenge_space_has_160_bits():
    for _, par in published():
        assert round(par.challenge_entropy) == 160
        assert par.challenge_entropy >= 128


def test_the_papers_own_soundness_figures_are_not_the_challenge_space():
    """`2^-91.5` is not `1/|C|`, so the footnote means something else.

    `5.Parameter.tex`'s footnote says the outer RVRF challenge space "remains
    approximately `2^-91.5` in both constructions", and gives the LANES
    challenge-difference noninvertibility probability as `2^-93.5`
    re-optimized (`2^-70` originally).  Neither reaches the 128-bit level the
    paragraph above the footnote selects against.

    This pins the arithmetic that says the `2^-91.5` cannot be the challenge
    space's own size, so it is a noninvertibility or soundness-slack term the
    footnote does not name.  Nothing here is normalised away: the disagreement
    is the finding.
    """
    for _, par in published():
        # |C| = binom(d, w) (2 gamma)^w, and d == w, so binom == 1
        assert par.d == par.w
        size = math.comb(par.d, par.w) * (2 * par.gamma) ** par.w
        assert size == 32 ** 32
        assert math.log2(size) == 160.0
        assert round(par.challenge_entropy) == 160

        # the footnote's figures, against what this parameter set gives
        for stated in (93.5, 91.5, 70.0):
            assert stated < 128, "a figure at or above 128 would settle this"
            assert stated != math.log2(size)


def test_B_a_is_exactly_128():
    """`B_a = gamma sqrt(2w) = 128`, and exact, so `phi_a B_a` is an integer."""
    for _, par in published():
        assert par.B_a == 128
        assert isinstance(par.B_a, int)
        assert float(par.sigma_a).is_integer()


def test_eta_m_is_the_same_for_every_profile():
    """`eta_m = w gamma B_e sqrt(d) = 86889.3`, independent of the profile."""
    values = {par.eta_m for _, par in published()}
    assert len(values) == 1
    assert round(values.pop(), 1) == 86889.3


def test_K_a_is_28_because_the_log_term_is_20():
    """`ceil(log2(w gamma n_hat d)) = 20` for every selected `n_hat`."""
    for _, par in published():
        term = math.ceil(math.log2(par.w * par.gamma * par.n_hat * par.d))
        assert term == 20, (par.name, par.n_hat, term)
        assert par.K_a == par.K_a_boundgen == 28


def test_embedded_key_noise_is_zero_with_negligible_probability():
    """`(2B_e+1)^{-d r'} = 61^-32`, which is below the `2^-128` target.

    DISCREPANCY (cosmetic).  The appendix writes this as `< 2^-190`.  It is
    `2^-189.78`, so the stated inequality is false by 0.22 bits.  Nothing
    depends on it -- the conclusion it supports, "below the `2^-128`
    target", holds with 60 bits to spare -- but the printed bound is not
    reproducible and is pinned rather than rounded away.
    """
    for _, par in published():
        p_zero = (2 * par.B_e + 1) ** (-par.d * par.r_prime)
        assert p_zero == 61.0 ** -32
        assert math.log2(p_zero) < -128
        assert not math.log2(p_zero) < -190
        assert -189.79 < math.log2(p_zero) < -189.78


# ---- derived bound columns ----------------------------------------------

def test_bound_columns_reproduce():
    for N, par in published():
        B, B_s, (B_g0, B_g1), s1, s2, sis, sel = PAPER_BOUNDS[N]
        assert sig(par.cal_B) == sig(B), (N, "B", par.cal_B)
        assert sig(par.B_s) == sig(B_s), (N, "B_s", par.B_s)
        assert sig(float(par.B_g0)) == sig(B_g0), (N, "B_g0", float(par.B_g0))
        assert sig(float(par.B_g1)) == sig(B_g1), (N, "B_g1", float(par.B_g1))
        assert sig(par.beta_sis_1) == sig(s1), (N, "beta_SIS,1", par.beta_sis_1)
        assert sig(par.beta_sis_2) == sig(s2), (N, "beta_SIS,2", par.beta_sis_2)
        assert sig(par.beta_sis) == sig(sis), (N, "beta_SIS", par.beta_sis)
        assert sig(par.beta_sel_inf) == sig(sel), (N, "beta_sel", par.beta_sel_inf)


def test_first_beta_sis_term_dominates_as_the_paper_states():
    """The paper: "for all five parameter sets, `4 w gamma beta_SIS,1` is
    the larger of the two terms"."""
    for _, par in published():
        first = 4 * par.w * par.gamma * par.beta_sis_1
        assert first > par.beta_sis_embedded
        assert par.beta_sis == first


def test_beta_sel_inf_is_the_g0_entry():
    """`B_g0` dominates the six selector bounds in every profile."""
    for _, par in published():
        blocks = par.beta_sel
        assert par.beta_sel_inf == blocks[3]
        # The merged five-entry estimate is an upper bound on the six.
        assert max(par.beta_sel_merged) >= par.beta_sel_inf


def test_size_columns_reproduce():
    """The repeat bound reproduces outright; the two size columns reproduce
    through the paper's own display formula."""
    for N, par in published():
        repeat, oom, total = PAPER_SIZES[N]
        assert round(par.mu_river, 1) == repeat, (N, par.mu_river)
        assert round(par.proof_size_oom_kb_paper, 1) == oom, \
            (N, par.proof_size_oom_kb_paper)
        assert round(par.proof_size_total_kb_paper, 1) == total, \
            (N, par.proof_size_total_kb_paper)


def test_the_transmitted_layout_costs_what_this_tree_reports():
    for N, par in published():
        oom, total = MEASURED_SIZES[N]
        assert round(par.proof_size_oom_kb, 1) == oom, (N, par.proof_size_oom_kb)
        assert round(par.proof_size_total_kb, 1) == total, \
            (N, par.proof_size_total_kb)


def test_the_published_proof_sizes_use_a_different_response_split():
    """DISCREPANCY, medium.  The paper regroups the OOM response but
    leaves the communication display charging the previous grouping.

    The algorithm transmits `ell + n` ring elements at `sigma_s` and one at
    `sigma_m`.  The display (`7.suppl.tex`, the `|pi_OOM|` align block)
    charges `ell d h(sigma_s) + (n+1) d h(sigma_m)`, which is a different
     split.  Since `sigma_s > sigma_m` at every profile now -- the revision
    requires it -- moving `n` elements to the cheaper width under-counts.

    Pinned three ways, so this cannot be mistaken for a rounding gap:
      * the paper's printed column is reproduced by the *stale* formula;
      * it is not reproduced by the transmitted one;
      * the gap is 0.4-0.6 KB, always in the same direction.

    The appendix describes the LANES parameter search as minimising
    communication, so the objective was evaluated on the wrong layout.
    Question 8 to the authors.
    """
    for N, par in published():
        printed = PAPER_SIZES[N][1]
        assert round(par.proof_size_oom_kb_paper, 1) == printed
        assert round(par.proof_size_oom_kb, 1) != printed
        gap = par.proof_size_oom_kb - par.proof_size_oom_kb_paper
        assert 0.4 <= gap <= 0.7, (N, gap)

        # It is exactly the `n` elements moved between the two widths.
        expect = (par.n * par.d
                  * (par._h(par.sigma_s) - par._h(par.sigma_m)) / 8192)
        assert abs(gap - expect) < 1e-9, (N, gap, expect)


def test_repeat_bound_is_below_the_design_target():
    """The design requirement is `mu-tilde_RiVeR <= 10`; the table says the
    five profiles land between 8.3 and 8.6."""
    values = [par.mu_river for _, par in published()]
    assert all(v <= 10 for v in values)
    assert 8.3 <= round(min(values), 1) and round(max(values), 1) <= 8.6


def test_epsilon_g_upper_bound_is_in_the_stated_range():
    """The paper: "between 0.78% and 0.91%, and hence below the required 1%".

    The exported values span 0.7793% to 0.9060%, which is that range to the
    two decimals it is quoted to.
    """
    pct = sorted(par.epsilon_g_u * 100 for _, par in published())
    assert round(pct[0], 2) == 0.78 and round(pct[-1], 2) == 0.91
    for _, par in published():
        assert par.epsilon_g_u < 0.01


def test_four_gaussian_samplers_are_charged():
    """`mu_OOM` charges `f_1`, `z_b`, `z_s` and `z_m`, and `mu_b` carries the
    half-space factor 2 that the infinity-norm check must not charge again."""
    for _, par in published():
        assert par.mu_gaussian == par.mu_a * par.mu_b * par.mu_s * par.mu_m
        assert par.mu_b == 2 * math.exp(1 / (2 * par.phi_b ** 2))
        assert par.mu_m == par.mu_a if par.phi_a == par.phi_m else True


# ---- modulus conditions --------------------------------------------------

def test_every_published_profile_passes_check():
    for _, par in published():
        assert par.check() == [], (par.name, par.check())


def test_outer_modulus_conditions():
    """`q > beta_SIS` and `q > max{beta_SIS,2, 12 sigma_s, 12 sigma_m}`."""
    for _, par in published():
        assert par.q > par.beta_sis
        assert par.q > max(par.beta_sis_2, 12 * par.sigma_s, 12 * par.sigma_m)


def test_selector_modulus_condition():
    """`q_hat > max{2(2 gamma + 12 phi_a B_a)^2, 2N^2, 2^{K_a+1},
    beta_sel,inf}`."""
    for _, par in published():
        need = max(2 * (2 * par.gamma + 12 * par.sigma_a) ** 2,
                   2 * par.N ** 2,
                   2 ** (par.K_a + 1),
                   par.beta_sel_inf)
        assert par.q_hat > need, (par.name, par.q_hat, need)


def test_check_is_total_and_fail_closed():
    """`check()` never raises, and never blesses a degenerate profile.

    It used to do both: `d = 0`, `w = 0` and `gamma = 0` raised out of a
    derived property it evaluated before validating anything, while
    `beta = 0`, `N = 0`, `max_attempts = 0`, `phi_a = 0` and a NaN width all
    came back clean.  A validation function that can throw is not one, and
    one that passes on a zero secret-key bound is worse.
    """
    degenerate = [
        ("d", 0), ("d", 3), ("d", -8), ("q0", 1), ("p", 0), ("q_hat", 1),
        ("n", 0), ("n", -1), ("ell", 0), ("n_hat", 0), ("k_hat", 0),
        ("N", 0), ("N", 1), ("w", 0), ("w", 64), ("gamma", 0), ("beta", 0),
        ("r_prime", 0), ("phi_a", 0), ("phi_s", -1), ("phi_m", 0),
        ("phi_b", float("nan")), ("phi_s", float("inf")),
        ("tau_g0", 0), ("tau_g1", -1),
        ("K_b", 0), ("K_a", 0), ("K_b", 30), ("s_cmp", -1),
        ("lam", 0), ("max_attempts", 0),
        ("epsilon_g_u", 1.0), ("epsilon_g_u", -0.1),
        ("name", ""), ("insecure_toy", "yes"),
        ("d", 32.0), ("w", True), ("N", "8"),
        # Integers too large to convert to a double.  `math.isfinite`
        # converts first, so these used to raise `OverflowError` out of the
        # domain pass itself -- the totality check reintroducing the
        # failure it exists to remove.
        ("epsilon_g_u", 10 ** 400), ("epsilon_g_u", -10 ** 400),
        ("phi_a", 10 ** 400), ("phi_s", 10 ** 400), ("phi_m", 10 ** 400),
        ("tau_g0", Fraction(10 ** 400)),
        ("tau_g1", Fraction(1, 10 ** 400)),
        ("d", 10 ** 400), ("N", 10 ** 400), ("q_hat", 10 ** 400),
        ("p", 10 ** 400), ("q0", 10 ** 400), ("beta", 10 ** 400),
        ("gamma", 10 ** 400), ("lam", 10 ** 400),
        ("max_attempts", 10 ** 400), ("K_a", 10 ** 6), ("s_cmp", 10 ** 400),
        ("n", 10 ** 400), ("ell", 10 ** 400), ("n_hat", 10 ** 400),
        ("k_hat", 10 ** 400), ("r_prime", 10 ** 400), ("w", 10 ** 400),
        # Past 4300 digits Python refuses int->str, so *formatting* the
        # diagnostic raised `ValueError` and a domain finding surfaced as
        # the outer guard.  The report must not be able to fail where the
        # check cannot.
        ("epsilon_g_u", 10 ** 10000), ("epsilon_g_u", -10 ** 8000),
        ("d", 10 ** 5000), ("N", 10 ** 10000), ("phi_s", 10 ** 6000),
        ("tau_g1", Fraction(1, 10 ** 9000)),
    ]
    for field, value in degenerate:
        par = dataclasses.replace(TOY_PARAMS, **{field: value})
        try:
            errors = par.check()
        except Exception as exc:                      # noqa: BLE001
            raise AssertionError(
                f"check() raised on {field}={value!r}: "
                f"{type(exc).__name__}: {exc}") from None
        assert errors, f"check() accepted {field}={value!r}"
        # ... and it is a *named* finding, not the outer guard catching an
        # exception.  The guard is a backstop; a domain rule reaching it
        # means the rule is missing, not that totality is satisfied.
        assert not any("raised" in e for e in errors), (field, value, errors)

        # The rendering is bounded: no diagnostic embeds an unbounded
        # integer, so none can be longer than the value is informative.
        for message in errors:
            assert len(message) < 400, (field, len(message))

    # ... and a sound profile still passes, so the gate is not just "always
    # complain".
    assert TOY_PARAMS.check() == []
    for _, par in published():
        assert par.check() == []


def test_domain_errors_short_circuit_the_derived_conditions():
    """A domain failure returns before any derived property is evaluated.

    That ordering is the whole point: `K_a_boundgen` takes `log2(w gamma
    n_hat d)`, so a zero in any of them is a `ValueError` rather than a
    finding, unless the domain pass runs first and stops.
    """
    par = dataclasses.replace(TOY_PARAMS, d=0, q0=1, p=0)
    errors = par.check()
    assert errors == par._domain()
    assert all("is not prime" not in e for e in errors)


def test_toy_profile_is_structurally_identical_but_insecure():
    par = TOY_PARAMS
    assert par.check() == []
    assert par.insecure_toy
    assert (par.d, par.q0, par.w, par.gamma, par.beta) == (32, 61, 32, 16, 1)
    assert (par.r_prime, par.phi_m, par.phi_b) == (1, 32, 2)
    # It is only the *security* conditions it fails.
    relaxed = dataclasses.replace(par, insecure_toy=False)
    assert relaxed.check() != []


# ---- provenance of the rounded inputs -----------------------------------

def test_one_decimal_tau_would_not_reproduce_the_table():
    """Why the table's second decimal is load-bearing.

    The paper prints `(tau_g0, tau_g1)` to two decimals and adds a note
    saying it does so "to make the corresponding bounds reproducible from
    the table".  This is the check behind that note: rounded to one
    decimal, the same values reproduce only 8 of the 10 `(B_g0, B_g1)`
    entries -- `N = 256` fails both -- while the printed two-decimal values
    reproduce all ten.

    So `_TAU` is **Paper**, not a reconstruction, and this test is what
    would catch a regression to the coarser figures.
    """
    def hits(table):
        n = 0
        for N, par in published():
            t0, t1 = table[N]
            alt = dataclasses.replace(par, tau_g0=t0, tau_g1=t1)
            want_0, want_1 = PAPER_BOUNDS[N][2]
            n += (sig(float(alt.B_g0)) == sig(want_0))
            n += (sig(float(alt.B_g1)) == sig(want_1))
        return n

    assert hits(_TAU) == 10
    assert hits(_TAU_DISPLAYED) == 8

    for N in PUBLISHED:
        for exact, shown in zip(_TAU[N], _TAU_DISPLAYED[N]):
            assert round1_half_up(exact) == shown, (N, exact, shown)


def test_tau_values_are_exact_rationals():
    """Not binary floats: a tie in `||g_0||_inf <= B_g0` is wire-visible."""
    for _, par in published():
        assert isinstance(par.tau_g0, Fraction)
        assert isinstance(par.tau_g1, Fraction)
        assert isinstance(par.B_g0, Fraction)
        assert isinstance(par.B_g1, Fraction)


# ---- pinned disagreements ------------------------------------------------

def test_the_tail_count_paragraph_now_agrees_with_its_own_definition():
    """**closed by the paper**, pinned as a closure.

    The `z_s` tail was `2d(ell+1)exp(-18)` in the display and
    `2d(n+ell)exp(-18)` in the definition two paragraphs later.  The
    algorithm responds to `ell + n` elements at `sigma_s`, so the
    definition was the right one and `eps_tail` took it; the display now
    agrees.

    Asserted against the *algorithm's* dimension rather than against a
    literal, so a future response regrouping moves both together.
    """
    for name, par in published():
        _, _, eps_s, eps_m = par.eps_tail
        t = 2 * par.d * math.exp(-18)
        assert math.isclose(eps_s, t * par.s_dim, rel_tol=1e-15), name
        assert math.isclose(eps_m, t * par.m_dim, rel_tol=1e-15), name
        assert par.s_dim == par.ell + par.n
        # The other split would have given this, and does not:
        assert not math.isclose(eps_s, t * (par.ell + 1), rel_tol=1e-9)


def test_the_repetition_denominator_now_is_the_product_of_its_components():
    """**closed in the paper**, and pinned as a closure.

    Under the table the appendix's own denominator --
    `(1-eps_a)(1-eps_b)(1-eps_s)(1-eps_m)(1-eps_g^U)(1-eps_c^U)` -- did not
    reproduce the printed "Repeat bound" column, and this tree carried a
    per-profile `c_pub_model` backsolved from that column.  Against the
    The paper table the components multiply out to the printed value at
    every profile, so the backsolve is gone.

    The assertion is on the *printed column*, not on an internal identity:
    `mu_river` is defined as the component product, so comparing it to
    itself would pass no matter what.  What is checked is that the product
    lands on the number the paper prints.
    """
    printed = {8: 8.3, 16: 8.4, 64: 8.6, 128: 8.6, 256: 8.5}
    for name, par in published():
        assert round(par.mu_river, 1) == printed[par.N], (name, par.mu_river)
        # And the backsolve really is gone, not merely unused.
        assert not hasattr(par, "c_pub_model")


def test_the_rejection_constant_is_the_hard_coded_twelve():
    """`tau_rej` is a parameter now, and its concrete value is the
    constant it replaced -- which is why that rendering moved no bytes.

    The paper writes `M_1 = exp(tau_rej/phi + 1/(2 phi^2))`
    and fixes `tau_rej = 12` for the concrete sets.  Pinned in the direction
    that can fail: if a future revision moves it, every repetition factor
    moves and this says so, rather than the change surfacing as a quietly
    different "Repeat bound" column.
    """
    for name, par in published():
        assert par.REJ_TAU == 12, name
        # The three `Rej_1` factors are all built from it...
        for phi, mu in ((par.phi_a, par.mu_a), (par.phi_s, par.mu_s),
                        (par.phi_m, par.mu_m)):
            assert math.isclose(
                mu, math.exp(par.REJ_TAU / phi + 1 / (2 * phi ** 2)),
                rel_tol=0, abs_tol=0)
        # ...and `mu_b` is Lemma grs(2)'s, which has no `tau_rej`.
        assert math.isclose(par.mu_b, 2 * math.exp(1 / (2 * par.phi_b ** 2)),
                            rel_tol=1e-15)

    # The printed column is unchanged by the parameterisation, which is
    # what "wire-neutral" means here.
    printed = {8: 8.3, 16: 8.4, 64: 8.6, 128: 8.6, 256: 8.5}
    for name, par in published():
        assert round(par.mu_river, 1) == printed[par.N], name


def test_the_euclidean_restart_term_is_negligible_as_the_paper_claims():
    """the paper adds `eps_2` to the denominator and says it is below
    `2^-150` at every final profile.  It is, by a wide margin -- so the
    term is carried for correctness, not because it moves the estimate.
    """
    for name, par in published():
        assert 0 < par.eps_euclidean < 2.0 ** -150, (name, par.eps_euclidean)
        # It does not move the reported bound at the printed precision.
        eps_a, eps_b, eps_s, eps_m = par.eps_tail
        without = ((1 - eps_a) * (1 - eps_b) * (1 - eps_s) * (1 - eps_m)
                   * (1 - par.epsilon_g_u) * par.p_cmp_uniform)
        assert round(par.mu_gaussian / without, 1) == round(par.mu_river, 1)


def test_the_boundgen_tuple_now_has_one_order():
    """`BoundGen`'s output tuple has exactly one order.

    The figure defining `BoundGen` and the three OOM algorithms that parse
    its output must agree on positions 4 and 5, `phi_s` and `phi_b`.  They
    are the same multiset, so a positional reader of the wrong order gets a
    well-formed tuple and only wrong numbers.

    The swapped order is kept in `params.py` and asserted *not* to be the
    live one.  Dropping it would make a regression to the old parse
    indistinguishable from correctness, which is the failure this guards.
    """
    assert BOUNDGEN_ORDER[2:6] == ("phi_a", "phi_b", "phi_s", "phi_m")
    assert BOUNDGEN_ORDER != BOUNDGEN_ORDER_SWAPPED
    differing = [i for i, (a, b) in
                 enumerate(zip(BOUNDGEN_ORDER, BOUNDGEN_ORDER_SWAPPED))
                 if a != b]
    assert differing == [3, 4]
    # The two are the same multiset, so a positional reader of the wrong one
    # gets a well-formed tuple and no error -- only wrong numbers.  That is
    # why the swap could survive a draft in the first place.
    assert sorted(BOUNDGEN_ORDER) == sorted(BOUNDGEN_ORDER_SWAPPED)

    # The size of the mistake it would still be, at every profile.
    for _, par in published():
        assert par.phi_b == 2
        assert 22 <= par.phi_s <= 40
        assert par.phi_s / par.phi_b >= 11


def test_named_fields_make_the_boundgen_order_unreachable():
    """Whichever order is right, `params.py` cannot be affected by it."""
    for name in BOUNDGEN_ORDER:
        for _, par in published():
            assert hasattr(par, name), (par.name, name)


def test_the_abstract_and_the_tables_now_agree():
    """**closed in the paper**.  The abstract said 35.5 KB and
    roughly 30x at `N = 64` while its own tables said 37.3 KB and 28.5x.
    The revision states 38.5 KB and 28x in both places.

    What this pins is the paper's *internal* consistency, so it uses the
    paper's own (stale-formula) size.  That the stale formula is itself
    wrong is a separate, still-open item -- see
    `test_the_published_proof_sizes_use_a_different_response_split`.
    """
    par = get("RiVeR-N64")
    assert round(par.proof_size_total_kb_paper, 1) == 38.5
    baseline_kb = 1.04 * 1024
    # The abstract now says "roughly 28x"; the intro table says 27.6x.
    assert round(baseline_kb / par.proof_size_total_kb_paper, 1) == 27.6
    assert round(baseline_kb / par.proof_size_total_kb_paper) == 28


def test_paper_embedded_key_noise_bound_is_stated_twice():
    """DISCREPANCY (medium).  Sub-Assumption `rvp-embedded-key-binding`
    requires the embedded-key noise `nz` to be nonzero and bounded by
    `beta = 1`, while the concrete parameter discussion samples
    `nz <- U_{B_e}^{r'}` with `B_e = 30` to obtain the stated `61^-32`
    probability.  Those are different distributions and different bounds;
    the implementation cannot resolve it, so it is pinned here.

    The `61^-32` figure is only reachable from the `U_{B_e}` reading: the
    `beta = 1` reading would give `3^-32`.
    """
    for _, par in published():
        from_B_e = (2 * par.B_e + 1) ** (-par.d * par.r_prime)
        from_beta = (2 * par.beta + 1) ** (-par.d * par.r_prime)
        assert from_B_e == 61.0 ** -32
        assert from_beta == 3.0 ** -32
        assert from_B_e != from_beta


def test_the_prover_now_applies_the_verifiers_euclidean_bound():
    """**Closed in the paper**, and pinned as a closure.

    In the paper `OOM.Ver` checked
    `||z||_2 <= 1.2 sqrt(sigma_s^2 d ell + sigma_m^2 d(n+1))` while the
    corresponding prover check sat *commented out* in the figure, in a
    different form -- `1.2 phi_s B_s sqrt((ell+n+1) d)`.  A prover that can
    return a proof its own verifier rejects is a defect regardless of which
    form is meant, so this tree applied the verifier's bound on the prover
    path as a named repair (measured to cost zero restarts).

    The paper puts that same bound in `OOM.Prove`'s reject list, in the
    verifier's form.  The repair is now the specification.

    The formerly-commented form is still worth pinning, because the
    regrouping flipped its direction: `sigma_s > sigma_m` now, so charging
    `sigma_s` to all `ell+n+1` coefficients is an *upper* bound on the
    verifier's, not the much smaller one it used to be.  That is the same
    domination the revision's own `eps_2` argument uses.
    """
    for _, par in published():
        dominating = 1.2 * par.sigma_s * math.sqrt(par.r_dim * par.d)
        assert dominating > par.z_l2_bound
        # Loose by under 1%: the two differ only in the one `z_eval`
        # element, which is `sigma_m` rather than `sigma_s`.
        assert 1.0 < dominating / par.z_l2_bound < 1.01
        # And the revision's premise for that domination holds.
        assert par.sigma_s >= par.sigma_m


# --------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  {t.__name__}: ok")
    print(f"test_params.py: {len(tests)} tests passed")
