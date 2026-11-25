# carbon_management

PyPSA's `carbon_management` example network, as PyPSA hands it to specsolve.
`capture.py` writes `spec.yaml`, `sources/` and `scalars.json` from it.

## Source and license

The network is 20 days of the hybrid scenario of Hofmann, F., Tries, C.,
Neumann, F. et al. H2 and CO2 network strategies for the European energy
system. *Nat Energy* 10, 715–724 (2025),
[doi:10.1038/s41560-025-01752-6](https://doi.org/10.1038/s41560-025-01752-6),
built with PyPSA-Eur. PyPSA distributes it as
`pypsa.examples.carbon_management()`, from `data.pypsa.org`.

PyPSA licenses its example network data under
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/), copyright PyPSA
contributors ([PyPSA's `REUSE.toml`](https://github.com/PyPSA/PyPSA/blob/master/REUSE.toml)).
The data here is under the same license. `spec.yaml` is written by PyPSA's
code, which is MIT.

## Changes from the source

- Each source is a table: an array as one row per cell, its NaN cells dropped.
- A label column is cast to its dimension's `dtype`: a datetime under a `str`
  dimension is written `YYYY-MM-DD HH:MM:SS`, an `int` dimension is `Int64`.
- The four `*_nom_max` parameters read a dropped row as `missing: absent`.
- Every table is written as zstd-compressed parquet.
