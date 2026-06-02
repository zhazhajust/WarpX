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
    elif case == "undulator":
        final_step = max(row[0] for row in rows)
        final_rows = [row for row in rows if row[0] == final_step]
        intensity = [row[-1] for row in final_rows]

        assert all(value >= 0.0 for value in intensity), intensity
        assert max(intensity) > 0.0, intensity

        nfreq = 31
        freq_min = 1.5e14
        freq_max = 3.3e14
        peak_index = max(range(len(intensity)), key=lambda i: intensity[i])
        peak_freq = freq_min + (freq_max - freq_min) * peak_index / (nfreq - 1)

        c = 299792458.0
        qe = 1.602176634e-19
        me = 9.1093837139e-31
        gamma = (1.0 + 20.0**2) ** 0.5
        lambda_u = 1.0e-3
        b0 = 0.05715
        pi = 3.141592653589793
        k_undulator = qe * b0 * lambda_u / (2.0 * pi * me * c)
        expected_freq = (
            2.0
            * gamma**2
            * c
            / lambda_u
            / (1.0 + 0.5 * k_undulator**2)
        )

        relative_error = abs(peak_freq - expected_freq) / expected_freq
        assert relative_error < 0.04, (peak_freq, expected_freq, relative_error)
    else:
        raise RuntimeError(f"Unknown Fourier radiation test case: {case}")


if __name__ == "__main__":
    main()
