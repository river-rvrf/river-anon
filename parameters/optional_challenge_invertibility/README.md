# Optional Challenge-Difference Invertibility Scripts

These are optional MatRiCT+ challenge-difference invertibility scripts.  They are included only for
reference and are not needed for the main RiVeR OOM parameter-table reproduction.

The large generated `STable_*.csv` files are intentionally not included in this clean artifact.
The bundled d=256 scripts are configured for the current LANES modulus
`q_hat=67107713`, with `d=256`, `L=64`, `q_hat mod 256 = 129`, and `w=11`.
`d256_current_q_result.txt` records the current-modulus postcompute result
`logp=-90.5`. The fast table check pins that recorded value and modulus, but
does not rederive them: reproduction requires regenerating the omitted large
`STable` intermediate, and no checksum ties the recorded result to that
generated intermediate.

## Scripts

| Script | Source role |
|---|---|
| `d128_precompute.sage` | Precompute the S-table for the hard-coded d=128 profile. |
| `d128_postcompute.sage` | Read the d=128 S-table and compute the non-invertibility probability bound. |
| `d256_precompute.sage` | Precompute the S-table for the current d=256, `q_hat=67107713` LANES profile. |
| `d256_postcompute.sage` | Read the d=256 S-table and compute the current-modulus non-invertibility probability bound. |
| `d256_current_q_result.txt` | Recorded output summary for the current d=256, `q_hat=67107713` run. |
| `heuristic_model_d256.sage` | Heuristic model script from the MatRiCT+ source. |

## Usage

For the bundled d=128 profile:

```bash
sage d128_precompute.sage
sage d128_postcompute.sage
```

For the current d=256 LANES profile:

```bash
sage d256_precompute.sage
sage d256_postcompute.sage
```

For a different `q` or `w`, edit the parameter block at the top of the relevant Sage script before
running it.  The postcompute script expects the generated `STable_gen_q=<q>_n.csv` file in the same
directory.
