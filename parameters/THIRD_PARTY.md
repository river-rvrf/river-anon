# Third-Party Source Provenance

This artifact bundles a local copy of `malb/lattice-estimator` under
`external/lattice-estimator/` and a local copy of the selector A-MSIS estimator
helpers under `external/msis-security/`.  They are included so the parameter
checks can be rerun without fetching code from the network.

The bundled `lattice-estimator` files `README.rst`, `requirements.txt`, and
`estimator/` match the upstream tree
`https://github.com/malb/lattice-estimator.git` at commit
`66771ec3d331e2021eccf17331a5ed1ff71f3ddb`
(`Merge pull request #184 from TabOg/edit_beta_range`, dated
2026-01-12).  This provenance is recorded in
`external/lattice-estimator/UPSTREAM.txt`.

The upstream README states that lattice-estimator is licensed under LGPLv3+.
A copy of the LGPLv3 license text is included as
`external/lattice-estimator/COPYING.LESSER-3.0.txt`.

`external/lattice-estimator/LANES.sage` is RiVeR-specific glue code for this
artifact and is not part of upstream lattice-estimator.

## Selector A-MSIS helper provenance

The three Python files under `external/msis-security/` are based on the
`asymmetric_sis_estimate/` scripts from
`https://gitlab.com/raykzhao/matrict_plus` at commit
`b24f3176d2db15ca55d91c8c9cbe1cef201c5d2d`. The upstream project uses the
BSD-0-Clause license; its license text is included as
`external/msis-security/LICENSE-BSD-0-Clause.txt`.

`external/msis-security/UPSTREAM.txt` records file hashes and distinguishes the
one byte-identical file from the two artifact-local derived copies. The latter
contain compatibility and runtime adaptations. The artifact driver fixes its
search steps explicitly and cross-checks the accelerated cost evaluator against
the bundled reference calculation at sampled points.
