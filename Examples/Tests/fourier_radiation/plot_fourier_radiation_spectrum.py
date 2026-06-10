#!/usr/bin/env python3

import argparse
import math
from pathlib import Path


def read_final_spectrum(rad_path):
    rows = []
    for line in rad_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append([float(value) for value in line.split()])

    if not rows:
        raise RuntimeError(f"No Fourier radiation rows found in {rad_path}")

    final_step = max(row[0] for row in rows)
    final_rows = [row for row in rows if row[0] == final_step]
    nfreq = max(int(row[2]) for row in final_rows) + 1
    frequency = [0.0] * nfreq
    spectrum = [0.0] * nfreq

    for row in final_rows:
        iomega = int(row[2])
        frequency[iomega] = row[3]
        spectrum[iomega] += row[6]

    return final_step, frequency, spectrum


def expected_peak_frequencies():
    c = 299792458.0
    qe = 1.602176634e-19
    me = 9.1093837139e-31
    lambda_u = 1.0e-3
    b0 = 0.05715
    k_undulator = qe * b0 * lambda_u / (2.0 * math.pi * me * c)
    return [
        2.0 * (1.0 + uz * uz) * c / lambda_u / (1.0 + 0.5 * k_undulator**2)
        for uz in (18.0, 19.0, 20.0, 21.0, 22.0)
    ]


def write_svg(output_path, final_step, frequency, spectrum):
    positive = [value for value in spectrum if value > 0.0]
    if not positive:
        raise RuntimeError("Spectrum contains no positive values")

    ymin = min(positive) * 0.6
    ymax = max(positive) * 1.8
    xmin = min(frequency)
    xmax = max(frequency)

    width, height = 900, 540
    left, right, top, bottom = 92, 34, 52, 76
    plot_width = width - left - right
    plot_height = height - top - bottom

    log_xmin = math.log10(xmin)
    log_xmax = math.log10(xmax)
    log_ymin = math.log10(ymin)
    log_ymax = math.log10(ymax)

    def sx(x):
        return left + (math.log10(x) - log_xmin) / (log_xmax - log_xmin) * plot_width

    def sy(y):
        return top + plot_height - (math.log10(y) - log_ymin) / (log_ymax - log_ymin) * plot_height

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="sans-serif" '
            f'font-size="18">Fourier radiation energy-spread beam spectrum, step {int(final_step)}</text>'
        ),
    ]

    for power in range(math.floor(log_xmin), math.ceil(log_xmax) + 1):
        for mantissa in range(1, 10):
            value = mantissa * (10**power)
            if xmin <= value <= xmax:
                x = sx(value)
                major = mantissa == 1
                svg.append(
                    f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" '
                    f'stroke="#e5e7eb" stroke-width="{1.1 if major else 0.6}"/>'
                )
                if major:
                    svg.append(
                        f'<text x="{x:.2f}" y="{top + plot_height + 24}" text-anchor="middle" '
                        f'font-family="sans-serif" font-size="11">1e{power}</text>'
                    )

    for power in range(math.floor(log_ymin), math.ceil(log_ymax) + 1):
        for mantissa in range(1, 10):
            value = mantissa * (10**power)
            if ymin <= value <= ymax:
                y = sy(value)
                major = mantissa == 1
                svg.append(
                    f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" '
                    f'stroke="#e5e7eb" stroke-width="{1.1 if major else 0.6}"/>'
                )
                if major:
                    svg.append(
                        f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" '
                        f'font-family="sans-serif" font-size="11">1e{power}</text>'
                    )

    for expected_frequency in expected_peak_frequencies():
        if xmin <= expected_frequency <= xmax:
            x = sx(expected_frequency)
            svg.append(
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" '
                f'stroke="#dc2626" stroke-width="1.2" opacity="0.35"/>'
            )

    svg.append(
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" '
        'fill="none" stroke="#111827" stroke-width="1.2"/>'
    )
    points = " ".join(
        f"{sx(freq):.2f},{sy(max(value, ymin)):.2f}" for freq, value in zip(frequency, spectrum)
    )
    svg.append(f'<polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="2.2"/>')

    for freq, value in zip(frequency, spectrum):
        if value > 0.0:
            svg.append(f'<circle cx="{sx(freq):.2f}" cy="{sy(value):.2f}" r="2.5" fill="#2563eb"/>')

    svg.extend(
        [
            (
                f'<text x="{left + plot_width / 2}" y="{height - 22}" text-anchor="middle" '
                'font-family="sans-serif" font-size="14">Frequency [Hz]</text>'
            ),
            (
                f'<text transform="translate(22 {top + plot_height / 2}) rotate(-90)" '
                'text-anchor="middle" font-family="sans-serif" font-size="14">'
                'Summed d2I/domega/dOmega [SI]</text>'
            ),
            (
                f'<text x="{left + plot_width - 8}" y="{top + 18}" text-anchor="end" '
                'font-family="sans-serif" font-size="12" fill="#7f1d1d">'
                'red lines: expected peak frequencies</text>'
            ),
            "</svg>",
        ]
    )
    output_path.write_text("\n".join(svg) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Plot the final-step FourierRadiation spectrum from rad.txt as SVG."
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        default=".",
        help="WarpX run directory containing diags/reducedfiles/rad.txt.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="fourier_radiation_energy_spread_beam_spectrum.svg",
        help="Output SVG path. Relative paths are resolved under run_dir.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    rad_path = run_dir / "diags" / "reducedfiles" / "rad.txt"
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = run_dir / output_path

    final_step, frequency, spectrum = read_final_spectrum(rad_path)
    write_svg(output_path, final_step, frequency, spectrum)
    print(output_path)


if __name__ == "__main__":
    main()
