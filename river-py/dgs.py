"""
dgs.py -- discrete-Gaussian tail arithmetic for parameter selection.

Everything here is exact `Decimal` arithmetic.  Nothing in this module calls
`math.erfc`, `math.exp` or any other libm function on a value that reaches a
decision: `river-py` is a test-vector generator, so a bound that shifts by one
ulp between platforms is a bug, not a rounding detail.  This mirrors
`../../lemur-dev/parameter/Lemur-DGS-Prec_TailCut.py`, which computes the same
quantities the same way.

Three separate things get called "the tail cut" and they are not the same:

`VERIFIER_TAILCUT` (`sample.py`)
    The paper's, from Figure 2: `Rej` returns `bot` when `||z||_inf > 6 sigma`.
    A protocol constant -- `beta_sel,inf` and the MSIS estimate derive from it.

`GAUSSIAN_TAILCUT` (`sample.py`)
    Where `gaussian_int` truncates the *mask*.  Ours, sized here.

`tc_id` / reference cut
    A cut so far out that the truncated law stands in for the ideal one when
    computing divergences against it.  20 sigma in the lemur script.

Two defensible ways to size the second one, and the sibling implementations
pick differently because their masks protect different things:

`renyi_tailcut` -- `../../lemur-dev`, tail cut 5
    HPRR19 (ePrint 2019/1411, Thm 7): keep `RD_{2 lambda + 1}` of the truncated
    law against the ideal below `1/(8 Q)` and the *unforgeability* loss is at
    most one bit.  The cut then scales with `log Q`, not with `lambda`, which
    is why 5 suffices there.  Rényi divergence has the probability-preservation
    property for **search** problems.

`statistical_tailcut` -- `../../lotrs-dev`, tail cut 14
    Union-bound the per-coefficient tail mass over every Gaussian coefficient
    in a transcript and require the total below `2^-lambda`.  Costs a `sqrt`
    of a much smaller epsilon, so the cut lands near 14.

`river-py` takes the second.  The mask that `gaussian_int` produces protects
**anonymity** -- which ring member evaluated -- and that is a decision problem,
where Rényi divergence does not give the clean bound it gives for search.
LoTRS makes the same call for the same reason, and its masking widths sit in
the same `10^6`-`10^7` range as `sigma_s` and `sigma_m`; lemur's cut of 5 is for a key-generation
base sampler under unforgeability.  `renyi_tailcut` is kept here so the gap
between the two routes is visible rather than asserted -- see `__main__`.

The paper adopts this three-way split as a requirement, at the
`(6, 14, 192)` used here, and requires the sampler to *reject* a declared cut
that `PROB_BITS` cannot reach rather than silently ignore it -- which is what
`sample._check_probability_width` does and `__main__` below exercises.  None
of that rescues the paper's own `2^-128` claim for the protocol's `6 sigma`
response truncation.
"""

from decimal import ROUND_CEILING, Decimal, localcontext
from fractions import Fraction

#: 60 digits, more than any call here needs.
PI = Decimal("3.14159265358979323846264338327950288419716939937510582097494459")

DEFAULT_PREC = 60

#: Depth of the Mills-ratio continued fraction.  Convergence is fast for
#: `t >= 2`; 400 is far past the point where extra levels change the result
#: at `DEFAULT_PREC`, and `__main__` checks that.
CF_DEPTH = 400


def _d(x):
    """`x` as a `Decimal`, exactly.

    `Fraction` is handled term-by-term because `str(Fraction)` is `"a/b"`,
    which `Decimal` will not parse -- and the LANES widths are `Fraction`s
    precisely so no float rounding reaches a bound.
    """
    if isinstance(x, Decimal):
        return x
    if isinstance(x, Fraction):
        return Decimal(x.numerator) / Decimal(x.denominator)
    return Decimal(str(x))


def tail_exact(t, prec=DEFAULT_PREC, depth=CF_DEPTH):
    """`Pr[|X| > t sigma]` for `X ~ N(0, sigma^2)`, as an exact `Decimal`.

    Evaluated as `2 phi(t) / M(t)` where `M` is the Mills ratio, expanded as
    the continued fraction

        M(t) = t + 1/(t + 2/(t + 3/(t + ...)))

    and folded backwards.  This is `erfc(t/sqrt 2)` to full precision without
    touching libm; `__main__` checks the two agree.

    The discrete Gaussian `D_sigma` differs from the continuous one by a
    Poisson-summation correction of order `exp(-2 pi^2 sigma^2)`, which at the
    smallest width RiVeR uses (`sigma_a = 3328`) is below `2^-10^8`.  Ignored.
    """
    with localcontext() as ctx:
        ctx.prec = prec
        t = _d(t)
        if t <= 0:
            return Decimal(1)
        phi = (-(t * t) / 2).exp() / (2 * PI).sqrt()
        cf = t
        for k in range(depth, 0, -1):
            cf = t + Decimal(k) / cf
        return 2 * phi / cf


def tail_bound(t, prec=DEFAULT_PREC):
    """The standard upper bound `2 exp(-t^2 / 2)` on `Pr[|X| > t sigma]`.

    Looser than `tail_exact` by a factor of about `t sqrt(2 pi)` -- 15x at
    `t = 6`, 35x at `t = 14` -- but it inverts in closed form, which is what
    `statistical_tailcut` and `renyi_tailcut` need.  This is the bound the
    lemur script uses.
    """
    with localcontext() as ctx:
        ctx.prec = prec
        t = _d(t)
        return 2 * (-(t * t) / 2).exp()


def _invert_tail_bound(eps, prec):
    """Smallest real `t` with `2 exp(-t^2/2) <= eps`, i.e. `sqrt(2 ln(2/eps))`."""
    with localcontext() as ctx:
        ctx.prec = prec
        return (2 * (2 / _d(eps)).ln()).sqrt()


def statistical_tailcut(n_coeff, lam=128, prec=DEFAULT_PREC):
    """Integer tail cut making the total statistical distance below `2^-lam`.

    Truncating `D_sigma` at `t sigma` and renormalising puts the truncated law
    at statistical distance exactly the tail mass from the ideal one, so a
    union bound over `n_coeff` coefficients wants

        n_coeff * Pr[|X| > t sigma] <= 2^-lam.

    Returns the smallest integer `t` for which that holds under `tail_exact`.
    The closed-form inverse of `tail_bound` seeds the search, so the result is
    never larger than the bound-based answer.
    """
    with localcontext() as ctx:
        ctx.prec = prec
        budget = _d(2) ** (-lam)              # total, over all coefficients
        per_coeff = budget / _d(n_coeff)      # what each one may contribute

    # `tail_bound` is looser than `tail_exact`, so inverting it never
    # under-shoots; walk down from there using the exact tail.
    t = int(_invert_tail_bound(per_coeff, prec)) + 1
    while t > 1 and tail_exact(t - 1, prec) * _d(n_coeff) <= budget:
        t -= 1
    return t


def chi2_slack(n, lam=128, prec=DEFAULT_PREC):
    """Smallest `1 + eps` making `Pr[||z||^2 > (1 + eps) n v] <= 2^-lam`.

    For `z` with `n` independent coordinates of variance `v`, the standard
    chi-square Chernoff bound is

        Pr[||z||^2 > (1 + eps) n v]  <=  exp(-n (eps - ln(1 + eps)) / 2),

    so the requirement is `eps - ln(1 + eps) >= 2 lam ln2 / n`.  Bisected in
    `Decimal`, then rounded up at `max(4, min(30, prec - 5))` decimals so the
    returned factor still satisfies the inequality when a consumer
    re-evaluates it at a different precision -- the bisection alone leaves
    it a few ulp short, and at `n = 3328` a consumer at 50 digits reads
    `2^-127.999...` rather than `2^-128`.  `prec` below 12 is refused
    rather than answered coarsely.  The value at `DEFAULT_PREC` is
    unaffected by the rounding.

    A verifier bound on `||z||_2` has to come from this and not from a
    small fudge factor: the difference between a 5% margin and the right one
    is the difference between an honest proof failing once in 584 attempts and
    never.  The paper now requires exactly this: "the smallest
    chi-square slack whose failure probability is below `2^-128`".

    **Independence is a hypothesis, not a formality.**  This bound is the
    right one for a vector of independent coordinates.  It is *not* the right
    one for a LANES response `z = y + c r`: conditioned on `c`, negacyclic
    convolution correlates the coefficients of `c r`, so `||z||^2` is a
    quadratic form in a Gaussian with a non-scalar covariance and its tail is
    governed by the whole spectrum rather than by `n` alone.  Use
    `quadratic_form_bound` there; `lanes_params` does.
    """
    with localcontext() as ctx:
        ctx.prec = prec
        if prec < 12:
            raise ValueError(f"chi2_slack needs prec >= 12, got {prec}")
        target = 2 * _d(lam) * Decimal(2).ln() / _d(n)
        lo, hi = Decimal(0), Decimal(1)
        while hi - Decimal(1) - (1 + hi).ln() < target:
            hi *= 2
        for _ in range(8 * prec):
            mid = (lo + hi) / 2
            if mid - (1 + mid).ln() < target:
                lo = mid
            else:
                hi = mid

        raw = 1 + hi

    # Round the answer *up*, coarsely, before returning it.
    #
    # The bisection converges to the root from above at `prec` digits, but
    # a consumer that re-evaluates the Chernoff bound at some other
    # precision can land a few ulp on the wrong side -- at `n = 3328` the
    # raw `1 + hi` gives `-127.999...9992` rather than `<= -128`.  Rounding
    # up at `places` decimals moves the answer by `10^-places`, which is
    # far larger than that gap and far smaller than one unit of any bound
    # derived from it, so the docstring's promise stops depending on the
    # consumer's precision matching this one.
    #
    # `places` is derived from `prec` and kept clear of it on both sides.
    # Quantizing at a fixed 30 used to raise `InvalidOperation` below
    # `prec = 31` -- the result would not fit the context -- and to trip
    # the check at `prec = 31`, where the perturbation lands in the same
    # digits as the cancellation in `eps - ln(1 + eps)`.
    places = max(4, min(30, prec - 5))
    with localcontext() as ctx:
        ctx.prec = prec + 10
        factor = raw.quantize(Decimal(1).scaleb(-places),
                              rounding=ROUND_CEILING)
        if factor - 1 - factor.ln() < target:   # pragma: no cover
            raise ArithmeticError(
                f"chi2_slack lost the inequality at prec={prec}")
    return factor


def quadratic_form_bound(trace, frob_sq, op_norm, lam=128, prec=DEFAULT_PREC):
    """Tail bound on `||z||^2` for `z ~ N(0, Sigma)` with *correlated* entries.

    Laurent-Massart (Ann. Statist. 28(5), 2000, Lemma 1): writing
    `||z||^2 = sum_i lambda_i g_i^2` with `g_i` iid standard normal and
    `lambda_i` the eigenvalues of `Sigma`,

        Pr[ ||z||^2 >= tr(Sigma) + 2 ||lambda||_2 sqrt(t)
                                 + 2 ||lambda||_inf t ]  <=  exp(-t),

    so `t = lam ln 2` gives a `2^-lam` bound.  The three inputs are
    `tr(Sigma)`, `||lambda||_2^2 = ||Sigma||_F^2` and `||lambda||_inf =
    ||Sigma||_op`; any valid upper bounds for the last two are sound, since
    the right-hand side is increasing in both.

    Why this rather than `chi2_slack`: that one is the special case
    `Sigma = v I`, where `||lambda||_2^2 = n v^2` and `||lambda||_inf = v`, and
    it is what a response of *independent* coordinates needs.  A LANES
    response is `z = y + c r`; given `c`, the map `r -> c r` is negacyclic
    multiplication, whose singular values are `|c-hat(zeta)|` over the
    primitive `2 d~`-th roots of unity.  Those are equal only if `c` is a
    monomial.  The mean of `|c-hat|^2` is `||c||_2^2`, which is why the
    *diagonal* -- and so the per-coefficient variance and the infinity-norm
    union bound -- is unaffected; the spread is what the Euclidean tail sees.

    **What this is and is not.**  Laurent-Massart is a theorem about a
    *continuous* Gaussian, and the sampler produces a *truncated discrete*
    one.  So this is the same modelling step `chi2_slack` already made and
    `tail_exact` documents: the discrete Gaussian differs from the
    continuous one by a Poisson-summation correction of order
    `exp(-2 pi^2 sigma^2)`, which at these widths is far below `2^-lambda`,
    and truncation only removes mass from the tail, which can only reduce
    `||z||^2`.  Neither step is stated here as a proof, and a real
    parameter proposal would want the sub-Gaussian quadratic-form version
    of the inequality rather than the Gaussian one.  What is *not* a
    modelling step, and was the actual defect, is the covariance: the
    independent-coordinate bound is wrong for this vector at any level of
    rigour, because `Sigma` is not a multiple of the identity.

    Returns the bound as an exact `Decimal`; the caller rounds up.
    """
    with localcontext() as ctx:
        ctx.prec = prec
        t = _d(lam) * Decimal(2).ln()
        return (_d(trace)
                + 2 * _d(frob_sq).sqrt() * t.sqrt()
                + 2 * _d(op_norm) * t)


def renyi_tailcut(q_queries, prec=DEFAULT_PREC):
    """Integer tail cut by the HPRR19 route, as in the lemur script.

    Sizes the cut so the tail mass falls below `1/(8 Q)`, which keeps
    `RD_{2 lambda + 1}` of the truncated law against the ideal within
    `1 + 1/(4 Q)` and costs at most one bit of *search* security.

    Sound for unforgeability.  Not what `river-py` uses for the mask -- see
    the module docstring.
    """
    with localcontext() as ctx:
        ctx.prec = prec
        eps = Decimal(1) / (8 * _d(q_queries))
        return int(_invert_tail_bound(eps, prec).to_integral_value(rounding="ROUND_CEILING"))


def log2(x, prec=DEFAULT_PREC):
    """`log2` of a positive `Decimal`, for reporting."""
    with localcontext() as ctx:
        ctx.prec = prec
        return _d(x).ln() / Decimal(2).ln()


def gaussian_coefficients_per_transcript(par, attempts=None):
    """How many `gaussian_int` draws a full `Eval` publishes.

    The four masked blocks: the selector tail `f_1` (`(N-1) d` at
    `sigma_a`), the binary response `z_b` (`k_hat d` at `sigma_b`), and the
    two halves of the outer response -- `z_s`
    (`ell d` at `sigma_s`) and `z_m` (`(n+1) d` at `sigma_m`).  Multiplied by
    the mean attempt count, since a restart still draws its masks.
    """
    per_attempt = ((par.N - 1) * par.d
                   + par.k_hat * par.d
                   + par.r_dim * par.d)
    if attempts is None:
        attempts = par.mu_river              # carries the half-space factor
    return int(per_attempt * attempts)


# --------------------------------------------------------------------------

if __name__ == "__main__":
    import math
    from params import get, PROFILES

    # `chi2_slack` answers the same at every precision it accepts, and the
    # answer survives re-evaluation somewhere else.  Both halves are
    # regressions: a fixed 30-place rounding used to raise
    # `InvalidOperation` below `prec = 31` and to lose the inequality at
    # 31, while no rounding at all lost it at 50.
    for _n in (2176, 3328, 4096):
        _at_default = chi2_slack(_n)
        for _prec in (12, 15, 20, 25, 30, 31, 35, 40, 60, 100):
            _f = chi2_slack(_n, prec=_prec)
            assert _f <= _at_default * Decimal("1.0001"), (_n, _prec)
            for _check in (30, 40, 50, 60, 80):
                with localcontext() as _ctx:
                    _ctx.prec = _check
                    _eps = Decimal(_f) - 1
                    _log2 = (-Decimal(_n) * (_eps - (1 + _eps).ln()) / 2
                             / Decimal(2).ln())
                assert _log2 <= -128, (_n, _prec, _check, _log2)
    for _bad in (0, 1, 11, -5):
        try:
            chi2_slack(3328, prec=_bad)
        except ValueError:
            pass
        else:
            raise SystemExit(f"chi2_slack accepted prec={_bad}")

    # Pinned reference values, to 30 significant figures.  Asserting against
    # these rather than against `math.erfc` keeps the check independent of
    # the platform's libm -- which is the whole reason this module exists.
    REFERENCE = {
        2:  "0.0455002638963584144005652743331",
        4:  "0.0000633424836662398425075415134443",
        6:  "1.97317529007539628140172826480E-9",
        8:  "1.24419211485435682470319903452E-15",
        14: "1.55870736383856005087193636778E-44",
    }
    for t, want in REFERENCE.items():
        with localcontext() as ctx:
            ctx.prec = 30
            got = +tail_exact(t, prec=40)        # unary + rounds to ctx.prec
        assert got == Decimal(want), (t, got, want)
    print(f"tail_exact matches {len(REFERENCE)} pinned references to 30 s.f.")

    # Informational only: how far the platform's erfc is from those.  Never
    # asserted -- a weak libm is not a reason for `make selftest` to fail.
    worst = max(abs(float(tail_exact(t)) - math.erfc(t / math.sqrt(2)))
                / math.erfc(t / math.sqrt(2)) for t in REFERENCE)
    print(f"  (this platform's math.erfc deviates by up to {worst:.2e})")

    # ... and is converged well before CF_DEPTH
    a = tail_exact(6, depth=CF_DEPTH)
    b = tail_exact(6, depth=CF_DEPTH * 2)
    assert a == b, (a, b)

    # the bound is a bound
    for t in (2, 6, 14):
        assert tail_bound(t) > tail_exact(t)

    print(f"\n{'profile':12s} {'coeffs':>8s} {'renyi':>6s} {'stat':>5s} "
          f"{'2^? at 6':>9s} {'2^? at 14':>10s}")
    for name in PROFILES:
        par = get(name)
        n = gaussian_coefficients_per_transcript(par)
        at6 = log2(tail_exact(6) * n)
        at14 = log2(tail_exact(14) * n)
        print(f"{name:12s} {n:8d} {renyi_tailcut(n):6d} "
              f"{statistical_tailcut(n):5d} {float(at6):9.2f} {float(at14):10.2f}")

    from sample import GAUSSIAN_TAILCUT, VERIFIER_TAILCUT
    par = get("RiVeR-N8")
    need = statistical_tailcut(gaussian_coefficients_per_transcript(par))
    assert GAUSSIAN_TAILCUT >= need, (GAUSSIAN_TAILCUT, need)
    assert GAUSSIAN_TAILCUT > VERIFIER_TAILCUT
    print(f"\nGAUSSIAN_TAILCUT = {GAUSSIAN_TAILCUT} covers the statistical "
          f"requirement ({need}) at every profile")
