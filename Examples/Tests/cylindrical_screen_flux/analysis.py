#!/usr/bin/env python3

import csv
import math
from pathlib import Path


path = Path("diags/reducedfiles/screen.csv")
assert path.exists(), f"missing diagnostic output {path}"

rows = []
with path.open() as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue
        rows.append(next(csv.reader([line], delimiter=" ", skipinitialspace=True)))

assert len(rows) == 1

(
    step,
    _time,
    species,
    _pid,
    x,
    y,
    z,
    r,
    theta,
    px,
    py,
    pz,
    ke_ev,
    ux,
    uy,
    uz,
    weight,
) = rows[0]

assert int(step) == 2
assert species == "electrons"
assert float(r) <= 0.5
assert float(z) == 0.0
assert math.isclose(float(theta), 0.0, abs_tol=1.0e-14)
assert math.isclose(float(x), float(r), rel_tol=0.0, abs_tol=1.0e-14)
assert math.isclose(float(y), 0.0, abs_tol=1.0e-14)
assert float(ux) < 0.0
assert math.isclose(float(uy), 0.0, abs_tol=1.0e-14)
assert math.isclose(float(uz), 0.0, abs_tol=1.0e-14)
assert float(px) < 0.0
assert math.isclose(float(py), 0.0, abs_tol=1.0e-40)
assert math.isclose(float(pz), 0.0, abs_tol=1.0e-40)
assert float(ke_ev) > 0.0
assert math.isclose(float(weight), 4.0, rel_tol=0.0, abs_tol=1.0e-14)
