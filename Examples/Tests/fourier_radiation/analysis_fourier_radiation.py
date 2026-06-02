#!/usr/bin/env python3

import sys
from pathlib import Path


def main():
    case = sys.argv[1]
    data_path = Path("diags/reducedfiles/rad.txt")
    if not data_path.exists():
        raise RuntimeError(f"Missing Fourier radiation diagnostic: {data_path}")

    rows = []
    with data_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            values = [float(v) for v in line.split()]
            if len(values) != 6:
                raise RuntimeError(f"Unexpected Fourier radiation row: {line}")
            rows.append(values)

    if not rows:
        raise RuntimeError("Fourier radiation diagnostic did not write data rows")

    intensity = [row[-1] for row in rows]
    if not all(value == value and abs(value) != float("inf") for value in intensity):
        raise RuntimeError("Fourier radiation intensity contains non-finite values")

    final_intensity = intensity[-1]
    if case == "accelerated":
        assert final_intensity > 0.0, final_intensity
    elif case == "uniform":
        assert final_intensity == 0.0, final_intensity
    else:
        raise RuntimeError(f"Unknown Fourier radiation test case: {case}")


if __name__ == "__main__":
    main()
