#!/usr/bin/env python3

import argparse
import csv
import math
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--dim", choices=["2d", "3d"], required=True)
args = parser.parse_args()

path = Path("diags/reducedfiles/screen.csv")
assert path.exists(), f"missing diagnostic output {path}"

rows = []
with path.open() as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue
        rows.append(next(csv.reader([line], delimiter=" ", skipinitialspace=True)))

if args.dim == "3d":
    assert len(rows) == 1
    row = rows[0]
    assert int(row[0]) == 2
    assert row[2] == "electrons"
    assert float(row[7]) <= 0.5
    assert math.isclose(float(row[4]), float(row[7]), rel_tol=0.0, abs_tol=1.0e-14)
    assert math.isclose(float(row[5]), 0.0, abs_tol=1.0e-14)
    assert math.isclose(float(row[6]), 0.0, abs_tol=1.0e-14)
    assert math.isclose(float(row[8]), 0.0, abs_tol=1.0e-14)
    assert float(row[13]) < 0.0
    assert math.isclose(float(row[16]), 4.0, rel_tol=0.0, abs_tol=1.0e-14)
else:
    assert len(rows) == 2
    rows.sort(key=lambda row: float(row[4]))
    left, right = rows

    assert int(left[0]) == 2
    assert int(right[0]) == 2
    assert left[2] == "electrons"
    assert right[2] == "electrons"

    assert float(left[4]) < 0.0
    assert float(right[4]) > 0.0
    assert math.isclose(float(left[5]), 0.0, abs_tol=1.0e-14)
    assert math.isclose(float(right[5]), 0.0, abs_tol=1.0e-14)
    assert math.isclose(float(left[6]), 0.0, abs_tol=1.0e-14)
    assert math.isclose(float(right[6]), 0.0, abs_tol=1.0e-14)
    assert float(left[7]) <= 0.5
    assert float(right[7]) <= 0.5
    assert math.isclose(float(left[8]), -math.pi, abs_tol=1.0e-14)
    assert math.isclose(float(right[8]), 0.0, abs_tol=1.0e-14)
    assert float(left[13]) > 0.0
    assert float(right[13]) < 0.0
    assert math.isclose(float(left[16]), 6.0, rel_tol=0.0, abs_tol=1.0e-14)
    assert math.isclose(float(right[16]), 4.0, rel_tol=0.0, abs_tol=1.0e-14)
