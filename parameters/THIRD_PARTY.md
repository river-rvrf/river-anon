# Third-Party Source Provenance

This artifact bundles a local copy of `malb/lattice-estimator` under
`external/lattice-estimator/` and a local copy of the selector A-MSIS estimator
helpers under `external/msis-security/`.  They are included so the reviewer can
rerun the parameter checks without fetching code from the network.

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

The three files under `external/msis-security/` are required by the selector
A-MSIS rerun, but this package does not currently contain their upstream URL,
commit identifier, copyright notice, or license. Their numerical inclusion is
therefore reproducible, but their provenance and redistribution permission are
not established by this artifact. Those details must be supplied before a
public artifact release; they must not be inferred from the unrelated
`lattice-estimator` license above.
