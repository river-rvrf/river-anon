#!/usr/bin/env sage -python
"""Regenerate the OOM product-threshold validation aggregates.

The experiment is deterministic for the recorded seed, validation grouping,
chunk size, and product block size.  Only aggregate failure counts are
written; raw samples are intentionally not retained.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_distribution


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SECURITY = ROOT / "data" / "final_oom_security.json"
DEFAULT_OUTPUT = ROOT / "data" / "product_tau_validation.csv"
DEFAULT_METADATA = ROOT / "data" / "product_tau_validation_metadata.json"
DEFAULT_SEED = 0x5249564552
DEFAULT_RUNS = (
    ("validation run for N8_64_128_256", (8, 64, 128, 256)),
    ("validation run for N16 (shares its initial seed with N8)", (16,)),
)


@dataclass(frozen=True)
class ProductParams:
    N: int
    d: int
    w: int
    gamma: int
    sigma_a: float
    B_g0: float
    B_g1: float
    tau_g0: float
    tau_g1: float


def challenge_sample(
    rng: np.random.Generator, trials: int, d: int, w: int, gamma: int
) -> np.ndarray:
    """Sample the concrete challenge distribution."""
    if w == d:
        magnitudes = rng.integers(1, gamma + 1, size=(trials, d), dtype=np.int64)
        signs = rng.integers(0, 2, size=(trials, d), dtype=np.int64) * 2 - 1
        return magnitudes * signs

    out = np.zeros((trials, d), dtype=np.int64)
    for row in range(trials):
        support = rng.choice(d, size=w, replace=False)
        magnitudes = rng.integers(1, gamma + 1, size=w, dtype=np.int64)
        signs = rng.integers(0, 2, size=w, dtype=np.int64) * 2 - 1
        out[row, support] = magnitudes * signs
    return out


class DiscreteGaussianSampler:
    """Centered discrete Gaussian sampler with symmetric 16-sigma support."""

    def __init__(self, sigma: float, tail_sigma: float = 16.0):
        self.sigma = float(sigma)
        self.bound = int(math.ceil(tail_sigma * self.sigma))
        xs = np.arange(-self.bound, self.bound + 1, dtype=np.float64)
        weights = np.exp(-(xs * xs) / (2.0 * self.sigma * self.sigma))
        cdf = np.cumsum(weights)
        cdf /= cdf[-1]
        self.values = np.arange(-self.bound, self.bound + 1, dtype=np.int64)
        self.cdf = cdf

    def sample(self, rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
        uniforms = rng.random(shape)
        indices = np.searchsorted(self.cdf, uniforms, side="left")
        return self.values[indices]


def negacyclic_mul_fft(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Batched negacyclic product using an FFT followed by integer rounding."""
    d = a.shape[-1]
    fft_len = 2 * d
    conv = np.fft.irfft(
        np.fft.rfft(a, n=fft_len, axis=-1)
        * np.fft.rfft(b, n=fft_len, axis=-1),
        n=fft_len,
        axis=-1,
    )
    conv = np.rint(conv).astype(np.int64)
    return conv[..., :d] - conv[..., d : 2 * d]


def negacyclic_mul_exact(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Batched schoolbook reference product over the integers."""
    a, b = np.broadcast_arrays(a, b)
    d = a.shape[-1]
    out = np.zeros(a.shape, dtype=np.int64)
    for i in range(d):
        for j in range(d):
            if i + j < d:
                out[..., i + j] += a[..., i] * b[..., j]
            else:
                out[..., i + j - d] -= a[..., i] * b[..., j]
    return out


def check_fft_against_exact() -> None:
    """Catch a convolution, fold, or rounding change before a long replay."""
    rng = np.random.default_rng(0x5249564552)
    for shape in ((7, 32), (5, 3, 32)):
        # The largest final sigma_a is 5120 and the sampler cuts at 16 sigma,
        # so 90,000 exceeds every coefficient magnitude reached in the run.
        a = rng.integers(-90_000, 90_001, size=shape, dtype=np.int64)
        b = rng.integers(-90_000, 90_001, size=shape, dtype=np.int64)
        if not np.array_equal(negacyclic_mul_fft(a, b), negacyclic_mul_exact(a, b)):
            raise SystemExit(f"FFT negacyclic product disagrees with exact product for {shape}")


def estimate_product_norm_failures(
    params: ProductParams,
    trials: int,
    chunk_size: int,
    block_size: int,
    seed: int,
    multiplication: str,
) -> int:
    rng = np.random.default_rng(seed)
    gaussian = DiscreteGaussianSampler(params.sigma_a)
    multiply = negacyclic_mul_fft if multiplication == "fft" else negacyclic_mul_exact
    failures = 0
    completed = 0
    started = time.time()

    while completed < trials:
        chunk = min(chunk_size, trials - completed)
        x = challenge_sample(rng, chunk, params.d, params.w, params.gamma)
        failed = np.zeros(chunk, dtype=bool)
        alive = np.ones(chunk, dtype=bool)
        sum_f = np.zeros((chunk, params.d), dtype=np.int64)

        remaining = params.N - 1
        while remaining > 0 and np.any(alive):
            take = min(block_size, remaining)
            alive_idx = np.flatnonzero(alive)
            x_alive = x[alive_idx]
            f = gaussian.sample(rng, (alive_idx.size, take, params.d))
            sum_f[alive_idx] += np.sum(f, axis=1)
            g = multiply(f, x_alive[:, None, :] - f)
            block_failed = np.max(np.abs(g), axis=(1, 2)) > params.B_g1
            if np.any(block_failed):
                failed_indices = alive_idx[block_failed]
                failed[failed_indices] = True
                alive[failed_indices] = False
            remaining -= take

        if np.any(alive):
            alive_idx = np.flatnonzero(alive)
            f0 = x[alive_idx] - sum_f[alive_idx]
            g0 = multiply(f0, x[alive_idx] - f0)
            g0_failed = np.max(np.abs(g0), axis=1) > params.B_g0
            failed[alive_idx[g0_failed]] = True

        failures += int(np.count_nonzero(failed))
        completed += chunk
        print(
            f"[N={params.N}] product trials {completed}/{trials}, "
            f"failures={failures}, elapsed={time.time() - started:.1f}s",
            flush=True,
        )

    return failures


def clopper_pearson_upper(failures: int, trials: int, alpha: float) -> float:
    if failures == trials:
        return 1.0
    return float(beta_distribution.ppf(1.0 - alpha, failures + 1, trials - failures))


def load_params(path: Path) -> list[ProductParams]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    constants = payload["global_constants"]
    d = int(constants["d"])
    w = int(constants["w"])
    gamma = int(constants["gamma"])
    out = []
    for row in payload["rows"]:
        B_a = gamma * math.sqrt(2.0 * w)
        bounds = row["selector_asis_bounds"]
        out.append(
            ProductParams(
                N=int(row["N"]),
                d=d,
                w=w,
                gamma=gamma,
                sigma_a=float(row["phi_a"]) * B_a,
                B_g0=float(bounds["B_g_0"]),
                B_g1=float(bounds["B_g_1"]),
                tau_g0=float(row["tau_g0"]),
                tau_g1=float(row["tau_g1"]),
            )
        )
    return sorted(out, key=lambda item: item.N)


def validation_plan(seed: int) -> list[dict[str, object]]:
    runs = []
    for name, row_order in DEFAULT_RUNS:
        seeds = {
            str(N): seed + 1_000_003 + 32_452_843 * index
            for index, N in enumerate(row_order)
        }
        runs.append({"name": name, "row_order": list(row_order), "seeds": seeds})
    return runs


def make_metadata(args: argparse.Namespace) -> dict[str, object]:
    return {
        "purpose": "Provenance for data/product_tau_validation.csv.",
        "product_tau_csv": "data/product_tau_validation.csv",
        "source_script": "scripts/estimate_public_restart_bounds.sage.py",
        "source_code_included": True,
        "source_function": "estimate_product_norm_failures",
        "sampler_model": "sample x from C_{w,gamma}^d and independent f_i from a centered discrete Gaussian truncated at 16 sigma, without conditioning on the protocol's preceding 6-sigma infinity check; count whether either product bound B_g,0 or B_g,1 fails",
        "statistical_model_note": "This exactly reproduces the supplied aggregate experiment. It is an unconditioned, wider-support proxy for the conditional event used in the protocol analysis; no claim of equality with that conditional probability is made.",
        "scale_note": "The experiment reads each final profile's phi_a and recomputed B_g bounds. The normalized failure event is scale-invariant; scale-dependent B_g columns are intentionally omitted from the aggregate CSV.",
        "validation_trials_per_row": args.trials,
        "chunk_size": args.chunk_size,
        "product_block_size": args.block_size,
        "multiplication": args.multiplication,
        "fft_self_test": "FFT negacyclic products are compared with exact integer schoolbook products before sampling.",
        "alpha_cell": args.alpha_cell,
        "confidence_bound": "one-sided Clopper-Pearson upper bound from validation_failures and validation_trials",
        "default_seed_decimal": args.seed,
        "default_seed_hex": hex(args.seed),
        "seed_rule": "seed = default_seed + 1000003 + 32452843 * row_index within the validation run",
        "cross_run_seed_note": "The two supplied validation runs restart the row index, so N=8 and N=16 share the same initial seed and challenge prefix. Their streams diverge after Gaussian sampling, and no cross-profile independence is assumed or used.",
        "validation_runs": validation_plan(args.seed),
        "raw_vectors": "Not stored; aggregate counts are deterministically regenerated by the source script.",
        "reproducibility_scope": "The artifact validates the recorded aggregates quickly. Running the included generator with its defaults replays all five one-million-trial experiments and reproduces the integer failure counts exactly; the floating Clopper-Pearson quantile is compared with a 5e-15 tolerance because its last digits may depend on the numerical library.",
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def check_outputs(
    rows: list[dict[str, object]], metadata: dict[str, object], output: Path, metadata_output: Path
) -> None:
    recorded = {int(row["N"]): row for row in read_csv(output)}
    generated = {int(row["N"]): row for row in rows}
    if set(recorded) != set(generated):
        raise SystemExit("generated and recorded product-threshold row sets differ")
    for N, row in generated.items():
        old = recorded[N]
        for key in ("validation_failures", "validation_trials", "count_source"):
            if str(row[key]) != old[key]:
                raise SystemExit(f"N={N}: {key} differs: {row[key]!r} != {old[key]!r}")
        for key in (
            "tau_g0_fixed", "tau_g1_fixed", "epsilon_g_validation_hat",
            "epsilon_g_validation_upper", "alpha_cell",
        ):
            got = float(row[key])
            want = float(old[key])
            if abs(got - want) > 5e-15 * max(1.0, abs(got), abs(want)):
                raise SystemExit(f"N={N}: {key} differs: {got!r} != {want!r}")
    recorded_metadata = json.loads(metadata_output.read_text(encoding="utf-8"))
    if metadata != recorded_metadata:
        raise SystemExit("generated product-threshold metadata differs")


def write_outputs(
    rows: list[dict[str, object]], metadata: dict[str, object], output: Path, metadata_output: Path
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "N", "tau_g0_fixed", "tau_g1_fixed", "validation_failures",
        "validation_trials", "epsilon_g_validation_hat",
        "epsilon_g_validation_upper", "alpha_cell", "count_source",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    metadata_output.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--security", type=Path, default=DEFAULT_SECURITY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--trials", type=int, default=1_000_000)
    parser.add_argument("--chunk-size", type=int, default=20_000)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--alpha-cell", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--multiplication", choices=("fft", "exact"), default="fft")
    parser.add_argument("--only-N", type=int, choices=(8, 16, 64, 128, 256))
    parser.add_argument("--check", action="store_true", help="replay without writing and compare with the recorded aggregate")
    parser.add_argument("--self-test-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.trials <= 0 or args.chunk_size <= 0 or args.block_size <= 0:
        raise SystemExit("trials, chunk size, and block size must be positive")
    check_fft_against_exact()
    if args.self_test_only:
        print(json.dumps({"product_self_test": "PASS"}, sort_keys=True))
        return

    params_by_N = {param.N: param for param in load_params(args.security)}
    rows = []
    for run in validation_plan(args.seed):
        for N in run["row_order"]:
            if args.only_N is not None and N != args.only_N:
                continue
            param = params_by_N[N]
            failures = estimate_product_norm_failures(
                param, args.trials, args.chunk_size, args.block_size,
                int(run["seeds"][str(N)]), args.multiplication,
            )
            rows.append({
                "N": N,
                "tau_g0_fixed": param.tau_g0,
                "tau_g1_fixed": param.tau_g1,
                "validation_failures": failures,
                "validation_trials": args.trials,
                "epsilon_g_validation_hat": failures / args.trials,
                "epsilon_g_validation_upper": clopper_pearson_upper(
                    failures, args.trials, args.alpha_cell
                ),
                "alpha_cell": args.alpha_cell,
                "count_source": run["name"],
            })
    rows.sort(key=lambda row: int(row["N"]))
    metadata = make_metadata(args)
    if args.check:
        if args.only_N is not None:
            recorded = {int(row["N"]): row for row in read_csv(args.output)}
            generated = rows[0]
            for key in ("validation_failures", "validation_trials"):
                if str(generated[key]) != recorded[args.only_N][key]:
                    raise SystemExit(f"N={args.only_N}: {key} differs")
        else:
            check_outputs(rows, metadata, args.output, args.metadata_output)
        action = "checked"
    else:
        if args.only_N is not None:
            raise SystemExit("--only-N is diagnostic and requires --check")
        write_outputs(rows, metadata, args.output, args.metadata_output)
        action = "written"
    print(json.dumps({"action": action, "rows": len(rows), "status": "PASS"}, sort_keys=True))


if __name__ == "__main__":
    main()
