#!/usr/bin/env python3

import math
import sys
from pathlib import Path


def read_input_grid_bounds(path, name):
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, values = line.split("=", 1)
        if key.strip() == name:
            values = values.split("#", 1)[0]
            parts = [float(v) for v in values.split()]
            if len(parts) != 3:
                raise RuntimeError(f"Unexpected grid specification for {name}: {line}")
            return parts[0], parts[1]
    raise RuntimeError(f"Could not find {name} in {path}")


def save_energy_spread_beam_slice(final_rows):
    nfreq = max(int(row[2]) for row in final_rows) + 1
    ntheta = max(int(row[4]) for row in final_rows) + 1
    theta_min, theta_max = read_input_grid_bounds(
        Path("warpx_used_inputs"), "warpx.fourier_radiation_theta")
    phi_index = 0

    intensity_slice = [[0.0 for _ in range(nfreq)] for _ in range(ntheta)]
    frequency = [0.0 for _ in range(nfreq)]
    for row in final_rows:
        iomega = int(row[2])
        frequency[iomega] = row[3]
        itheta = int(row[4])
        iphi = int(row[5])
        if iphi == phi_index:
            intensity_slice[itheta][iomega] = row[-1]

    max_intensity = max(max(row) for row in intensity_slice)
    assert max_intensity > 0.0, intensity_slice
    log_slice = [
        [
            -12.0 if value <= 0.0 else max(-12.0, math.log10(value / max_intensity))
            for value in row
        ]
        for row in intensity_slice
    ]

    def color(value):
        t = min(1.0, max(0.0, (value + 12.0) / 12.0))
        stops = (
            (15, 23, 42),
            (37, 99, 235),
            (20, 184, 166),
            (250, 204, 21),
        )
        scaled = t * (len(stops) - 1)
        index = min(len(stops) - 2, int(scaled))
        frac = scaled - index
        rgb = tuple(
            round(stops[index][component] * (1.0 - frac) + stops[index + 1][component] * frac)
            for component in range(3)
        )
        return "#{:02x}{:02x}{:02x}".format(*rgb)

    c = 299792458.0
    qe = 1.602176634e-19
    me = 9.1093837139e-31
    lambda_u = 1.0e-3
    b0 = 0.05715
    k_undulator = qe * b0 * lambda_u / (2.0 * math.pi * me * c)
    expected_freqs = [
        2.0 * (1.0 + uz * uz) * c / lambda_u / (1.0 + 0.5 * k_undulator**2)
        for uz in (18.0, 19.0, 20.0, 21.0, 22.0)
    ]
    freq_min = min(frequency)
    freq_max = max(frequency)

    width = 760
    height = 460
    left = 86
    right = 28
    top = 48
    bottom = 72
    plot_width = width - left - right
    plot_height = height - top - bottom
    cell_height = plot_height / ntheta
    log_freq_min = math.log10(freq_min)
    log_freq_max = math.log10(freq_max)

    def frequency_x(freq):
        return left + (math.log10(freq) - log_freq_min) / (log_freq_max - log_freq_min) * plot_width

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        (
            '<text x="380" y="24" text-anchor="middle" font-family="sans-serif" '
            'font-size="17">Fourier radiation energy-spread beam, phi index 0</text>'
        ),
    ]
    for itheta, row in enumerate(log_slice):
        for iomega, value in enumerate(row):
            if iomega == 0:
                x0 = left
            else:
                x0 = 0.5 * (frequency_x(frequency[iomega - 1]) + frequency_x(frequency[iomega]))
            if iomega == nfreq - 1:
                x1 = left + plot_width
            else:
                x1 = 0.5 * (frequency_x(frequency[iomega]) + frequency_x(frequency[iomega + 1]))
            y = top + (ntheta - 1 - itheta) * cell_height
            svg.append(
                f'<rect x="{x0:.3f}" y="{y:.3f}" width="{x1 - x0 + 0.4:.3f}" '
                f'height="{cell_height + 0.4:.3f}" fill="{color(value)}"/>'
            )
    for expected_freq in expected_freqs:
        x = frequency_x(expected_freq)
        if left <= x <= left + plot_width:
            svg.append(
                f'<line x1="{x:.3f}" y1="{top}" x2="{x:.3f}" y2="{top + plot_height}" '
                'stroke="white" stroke-width="1.2" stroke-dasharray="4 4" opacity="0.8"/>'
            )
    svg.extend(
        [
            (
                f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" '
                'fill="none" stroke="#111827" stroke-width="1"/>'
            ),
            (
                '<text x="410" y="438" text-anchor="middle" font-family="sans-serif" '
                'font-size="14">frequency [1e14 Hz]</text>'
            ),
            (
                '<text x="20" y="218" text-anchor="middle" font-family="sans-serif" '
                'font-size="14" transform="rotate(-90 20 218)">theta [rad]</text>'
            ),
        ]
    )
    for tick in range(5):
        frac = tick / 4.0
        x = left + frac * plot_width
        freq = 10.0 ** (log_freq_min + frac * (log_freq_max - log_freq_min)) * 1.0e-14
        svg.append(
            f'<line x1="{x:.3f}" y1="{top + plot_height}" '
            f'x2="{x:.3f}" y2="{top + plot_height + 5}" stroke="#111827"/>'
        )
        svg.append(
            f'<text x="{x:.3f}" y="{top + plot_height + 23}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="12">{freq:.2f}</text>'
        )
    for tick in range(4):
        frac = tick / 3.0
        y = top + plot_height - frac * plot_height
        theta = theta_min + frac * (theta_max - theta_min)
        svg.append(
            f'<line x1="{left - 5}" y1="{y:.3f}" '
            f'x2="{left}" y2="{y:.3f}" stroke="#111827"/>'
        )
        svg.append(
            f'<text x="{left - 10}" y="{y + 4:.3f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="12">{theta:.3f}</text>'
        )
    svg.append(
        '<text x="615" y="42" font-family="sans-serif" font-size="12" fill="#374151">'
        'white dashed: single-particle fundamentals</text>'
    )
    svg.append(
        '<text x="615" y="438" font-family="sans-serif" font-size="12" fill="#374151">'
        'color: log10(normalized intensity)</text>'
    )
    svg.append("</svg>")
    Path("fourier_radiation_energy_spread_beam_phi0.svg").write_text("\n".join(svg) + "\n")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    image = ax.imshow(
        log_slice,
        aspect="auto",
        extent=[freq_min * 1.0e-14, freq_max * 1.0e-14, theta_min, theta_max],
        origin="lower",
        cmap="viridis",
        vmin=-12.0,
        vmax=0.0,
    )
    ax.set_xlabel("frequency [1e14 Hz]")
    ax.set_ylabel("theta [rad]")
    ax.set_title("Fourier radiation energy-spread beam, phi index 0")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("log10(normalized intensity)")
    fig.tight_layout()
    fig.savefig("fourier_radiation_energy_spread_beam_phi0.png", dpi=200)
    plt.close(fig)


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
            if len(values) != 7:
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

        peak_index = max(range(len(intensity)), key=lambda i: intensity[i])
        peak_freq = final_rows[peak_index][3]

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
    elif case == "energy_spread_beam":
        final_step = max(row[0] for row in rows)
        final_rows = [row for row in rows if row[0] == final_step]

        nfreq = max(int(row[2]) for row in final_rows) + 1
        ntheta = max(int(row[4]) for row in final_rows) + 1
        nphi = max(int(row[5]) for row in final_rows) + 1
        expected_rows = nfreq * ntheta * nphi
        assert len(final_rows) == expected_rows, (len(final_rows), expected_rows)

        spectrum = [0.0] * nfreq
        frequency = [0.0] * nfreq
        angular = [0.0] * (ntheta * nphi)
        for row in final_rows:
            iomega = int(row[2])
            frequency[iomega] = row[3]
            itheta = int(row[4])
            iphi = int(row[5])
            value = row[-1]
            assert value >= 0.0, value
            spectrum[iomega] += value
            angular[itheta + ntheta * iphi] += value

        peak_index = max(range(nfreq), key=lambda i: spectrum[i])
        peak_freq = frequency[peak_index]
        assert max(spectrum) > 0.0, spectrum

        c = 299792458.0
        qe = 1.602176634e-19
        me = 9.1093837139e-31
        lambda_u = 1.0e-3
        b0 = 0.05715
        pi = 3.141592653589793
        k_undulator = qe * b0 * lambda_u / (2.0 * pi * me * c)
        expected_freqs = [
            2.0 * (1.0 + uz * uz) * c / lambda_u / (1.0 + 0.5 * k_undulator**2)
            for uz in (18.0, 19.0, 20.0, 21.0, 22.0)
        ]
        assert min(expected_freqs) * 0.94 <= peak_freq <= max(expected_freqs) * 1.06, (
            peak_freq,
            expected_freqs,
        )

        nonzero_angles = sum(value > 0.01 * max(angular) for value in angular)
        assert nonzero_angles > 1, angular

        save_energy_spread_beam_slice(final_rows)
    else:
        raise RuntimeError(f"Unknown Fourier radiation test case: {case}")


if __name__ == "__main__":
    main()
