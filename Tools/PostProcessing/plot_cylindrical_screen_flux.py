#!/usr/bin/env python3
#
# Copyright 2026
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

import argparse
import csv
import re
import struct
import zlib
from pathlib import Path


HEADER_RE = re.compile(r"\[(\d+)\]([^()\s]+)")


def parse_header(line):
    """Return column names from a CylindricalScreenFlux header line."""
    columns = []
    for match in HEADER_RE.finditer(line):
        index = int(match.group(1))
        name = match.group(2)
        if index != len(columns):
            raise ValueError(f"unexpected column index {index} in header")
        columns.append(name)
    return columns


def read_screen_csv(path):
    """Read a CylindricalScreenFlux CSV file into a dict of arrays."""
    header = None
    rows = []

    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            if line.startswith("#"):
                header = parse_header(line)
                continue
            rows.append(next(csv.reader([line], delimiter=" ", skipinitialspace=True)))

    if header is None:
        raise ValueError(f"{path} does not contain a CylindricalScreenFlux header")
    if not rows:
        raise ValueError(f"{path} does not contain any screen crossing rows")

    if len(rows[0]) != len(header):
        raise ValueError(
            f"{path} has {len(rows[0])} data columns, but the header has {len(header)}"
        )

    data = {}
    string_columns = {"species"}
    int_columns = {"step", "id", "theta_bin", "z_bin", "energy_bin"}
    for i, name in enumerate(header):
        values = [row[i] for row in rows]
        if name in string_columns:
            data[name] = values
        elif name in int_columns:
            data[name] = [int(value) for value in values]
        else:
            data[name] = [float(value) for value in values]
    return data


def filter_data(data, species):
    if species is None:
        return data
    mask = [value == species for value in data["species"]]
    if not any(mask):
        raise ValueError(f"no rows found for species {species!r}")
    return {
        name: [value for value, keep in zip(values, mask) if keep]
        for name, values in data.items()
    }


def weighted_mean(values, weights):
    weight_sum = sum(weights)
    if weight_sum == 0.0:
        return sum(values) / len(values)
    return sum(value * weight for value, weight in zip(values, weights)) / weight_sum


def weighted_histogram2d(x, y, weights, xbins, ybins, xrange=None, yrange=None):
    if xrange is None:
        xrange = (min(x), max(x))
    if yrange is None:
        yrange = (min(y), max(y))

    grid = [[0.0 for _ in range(xbins)] for _ in range(ybins)]
    xmin, xmax = xrange
    ymin, ymax = yrange
    if xmax <= xmin:
        xmax = xmin + 1.0
    if ymax <= ymin:
        ymax = ymin + 1.0

    for xv, yv, weight in zip(x, y, weights):
        if xv < xmin or xv > xmax or yv < ymin or yv > ymax:
            continue
        ix = min(xbins - 1, int((xv - xmin) * xbins / (xmax - xmin)))
        iy = min(ybins - 1, int((yv - ymin) * ybins / (ymax - ymin)))
        grid[iy][ix] += weight
    return grid, xrange, yrange


def make_matplotlib_plot(data, output_path, title, plot_type, bins):
    import matplotlib.pyplot as plt
    import numpy as np

    if "energy_bin" in data:
        make_matplotlib_histogram_plot(data, output_path, title, plot_type)
        return

    z = data["z"]
    r = data["r"]
    theta = data["theta"]
    ke_ev = data["KE_eV"]
    weights = data["weight"]
    species = sorted(set(data["species"]))

    if plot_type == "heatmap":
        hist, z_edges, theta_edges = np.histogram2d(
            z,
            theta,
            bins=bins,
            range=[[min(z), max(z)], [-np.pi, np.pi]],
            weights=weights,
        )
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        image = ax.imshow(
            hist.T,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=[z_edges[0], z_edges[-1], theta_edges[0], theta_edges[-1]],
            cmap="magma",
        )
        ax.set_xlabel("z [m]")
        ax.set_ylabel("theta [rad]")
        ax.set_title("Unwrapped cylindrical screen weighted electron signal")
        fig.colorbar(image, ax=ax, label="Sum of particle weights")
        fig.suptitle(title)
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    scatter = axes[0, 0].scatter(z, r, c=ke_ev, s=28, cmap="viridis", edgecolors="none")
    axes[0, 0].set_xlabel("z [m]")
    axes[0, 0].set_ylabel("r [m]")
    axes[0, 0].set_title("Screen hit positions")
    fig.colorbar(scatter, ax=axes[0, 0], label="Kinetic energy [eV]")

    axes[0, 1].scatter(z, theta, c=ke_ev, s=28, cmap="viridis", edgecolors="none")
    axes[0, 1].set_xlabel("z [m]")
    axes[0, 1].set_ylabel("theta [rad]")
    axes[0, 1].set_title("Unwrapped cylindrical screen")

    bins = min(50, max(5, int(np.sqrt(len(ke_ev)))))
    axes[1, 0].hist(ke_ev, bins=bins, weights=weights, color="#4477aa", alpha=0.85)
    axes[1, 0].set_xlabel("Kinetic energy [eV]")
    axes[1, 0].set_ylabel("Weighted counts")
    axes[1, 0].set_title("Energy spectrum")

    axes[1, 1].axis("off")
    summary = [
        f"Rows: {len(ke_ev)}",
        f"Species: {', '.join(species)}",
        f"Steps: {min(data['step'])} - {max(data['step'])}",
        f"Total weight: {sum(weights):.6g}",
        f"Mean KE: {weighted_mean(ke_ev, weights):.6g} eV",
        f"Mean r: {weighted_mean(r, weights):.6g} m",
        f"z range: {min(z):.6g} - {max(z):.6g} m",
    ]
    axes[1, 1].text(0.02, 0.98, "\n".join(summary), va="top", family="monospace")

    fig.suptitle(title)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_matplotlib_histogram_plot(data, output_path, title, plot_type):
    import matplotlib.pyplot as plt
    import numpy as np

    theta_bins = max(data["theta_bin"]) + 1
    z_bins = max(data["z_bin"]) + 1
    energy_bins = max(data["energy_bin"]) + 1
    theta_min = min(data["theta_min"])
    theta_max = max(data["theta_max"])
    z_min = min(data["z_min"])
    z_max = max(data["z_max"])
    energy_min = min(data["energy_min"])
    energy_max = max(data["energy_max"])

    projection = np.zeros((theta_bins, z_bins))
    spectrum = np.zeros(energy_bins)
    counts = np.zeros(energy_bins)
    for itheta, iz, ie, weight, count in zip(
        data["theta_bin"],
        data["z_bin"],
        data["energy_bin"],
        data["sum_weight"],
        data["count"],
    ):
        projection[itheta, iz] += weight
        spectrum[ie] += weight
        counts[ie] += count

    if plot_type == "heatmap":
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        image = ax.imshow(
            projection,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=[z_min, z_max, theta_min, theta_max],
            cmap="magma",
        )
        ax.set_xlabel("z [m]")
        ax.set_ylabel("theta [rad]")
        ax.set_title("Accumulated theta-z signal, summed over energy")
        fig.colorbar(image, ax=ax, label="Sum of particle weights")
        fig.suptitle(title)
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    image = axes[0].imshow(
        projection,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[z_min, z_max, theta_min, theta_max],
        cmap="magma",
    )
    axes[0].set_xlabel("z [m]")
    axes[0].set_ylabel("theta [rad]")
    axes[0].set_title("Unwrapped screen projection")
    fig.colorbar(image, ax=axes[0], label="Sum of particle weights")

    energy_edges = np.linspace(energy_min, energy_max, energy_bins + 1)
    axes[1].stairs(spectrum, energy_edges, fill=True, color="#4477aa", alpha=0.85)
    axes[1].set_xlabel("Kinetic energy [eV]")
    axes[1].set_ylabel("Weighted counts")
    axes[1].set_title("Window-integrated energy spectrum")

    species = sorted(set(data["species"]))
    window = ""
    if "interval_start" in data and "interval_stop" in data:
        window = (
            f", interval=[{min(data['interval_start']):.6g}, "
            f"{max(data['interval_stop']):.6g}] s"
        )
    fig.suptitle(
        f"{title}\nSpecies: {', '.join(species)}, "
        f"rows: {len(data['sum_weight'])}, total weight: {sum(data['sum_weight']):.6g}"
        f"{window}"
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_png(path, width, height, pixels):
    def chunk(name, data):
        return (
            struct.pack(">I", len(data))
            + name
            + data
            + struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + bytes(row) for row in pixels)
    with path.open("wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(raw, 9)))
        f.write(chunk(b"IEND", b""))


def basic_color(value, vmin, vmax):
    if vmax <= vmin:
        t = 0.5
    else:
        t = max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
    if t < 0.5:
        u = t * 2.0
        return (int(45 * (1 - u) + 40 * u), int(70 * (1 - u) + 150 * u), 170)
    u = (t - 0.5) * 2.0
    return (int(40 * (1 - u) + 245 * u), int(150 * (1 - u) + 185 * u), int(170 * (1 - u) + 55 * u))


def draw_rect(pixels, x0, y0, x1, y1, color):
    height = len(pixels)
    width = len(pixels[0]) // 3
    x0, x1 = sorted((max(0, x0), min(width - 1, x1)))
    y0, y1 = sorted((max(0, y0), min(height - 1, y1)))
    for y in range(y0, y1 + 1):
        row = pixels[y]
        for x in range(x0, x1 + 1):
            i = 3 * x
            row[i : i + 3] = color


def draw_circle(pixels, cx, cy, radius, color):
    rr = radius * radius
    for y in range(cy - radius, cy + radius + 1):
        for x in range(cx - radius, cx + radius + 1):
            if (x - cx) * (x - cx) + (y - cy) * (y - cy) <= rr:
                draw_rect(pixels, x, y, x, y, color)


def map_value(value, vmin, vmax, lo, hi):
    if vmax <= vmin:
        return (lo + hi) // 2
    return int(round(lo + (value - vmin) * (hi - lo) / (vmax - vmin)))


def draw_heatmap(pixels, grid, panel, empty_color=(255, 255, 255)):
    x0, y0, x1, y1 = panel
    values = [value for row in grid for value in row]
    vmax = max(values) if values else 0.0
    ybins = len(grid)
    xbins = len(grid[0]) if ybins else 0
    if vmax <= 0.0 or xbins == 0 or ybins == 0:
        draw_rect(pixels, x0, y0, x1, y1, empty_color)
        return

    for iy, row in enumerate(grid):
        for ix, value in enumerate(row):
            px0 = x0 + ix * (x1 - x0 + 1) // xbins
            px1 = x0 + (ix + 1) * (x1 - x0 + 1) // xbins - 1
            py1 = y1 - iy * (y1 - y0 + 1) // ybins
            py0 = y1 - (iy + 1) * (y1 - y0 + 1) // ybins + 1
            color = empty_color if value <= 0.0 else basic_color(value, 0.0, vmax)
            draw_rect(pixels, px0, py0, px1, py1, color)


def make_basic_png(data, output_path, plot_type, bins):
    if "energy_bin" in data:
        make_basic_histogram_png(data, output_path)
        return

    width, height = 1000, 720
    pixels = [bytearray([248, 248, 246] * width) for _ in range(height)]

    z = data["z"]
    r = data["r"]
    theta = data["theta"]
    ke_ev = data["KE_eV"]
    weights = data["weight"]
    ke_min, ke_max = min(ke_ev), max(ke_ev)

    if plot_type == "heatmap":
        panel = (90, 70, 930, 610)
        draw_rect(pixels, panel[0], panel[1], panel[2], panel[3], (255, 255, 255))
        grid, _, _ = weighted_histogram2d(
            z, theta, weights, bins[0], bins[1], (min(z), max(z)), (-3.141592653589793, 3.141592653589793)
        )
        draw_heatmap(pixels, grid, panel)
        draw_rect(pixels, panel[0], panel[3] - 1, panel[2], panel[3] + 1, (50, 50, 50))
        draw_rect(pixels, panel[0] - 1, panel[1], panel[0] + 1, panel[3], (50, 50, 50))
        write_png(output_path, width, height, pixels)
        return

    panels = [(70, 60, 470, 330), (560, 60, 930, 330), (70, 420, 470, 660)]
    for x0, y0, x1, y1 in panels:
        draw_rect(pixels, x0, y0, x1, y1, (255, 255, 255))
        draw_rect(pixels, x0, y1 - 1, x1, y1 + 1, (50, 50, 50))
        draw_rect(pixels, x0 - 1, y0, x0 + 1, y1, (50, 50, 50))

    for zv, rv, ev in zip(z, r, ke_ev):
        x = map_value(zv, min(z), max(z), panels[0][0] + 12, panels[0][2] - 12)
        y = map_value(rv, min(r), max(r), panels[0][3] - 12, panels[0][1] + 12)
        draw_circle(pixels, x, y, 5, basic_color(ev, ke_min, ke_max))

    for zv, tv, ev in zip(z, theta, ke_ev):
        x = map_value(zv, min(z), max(z), panels[1][0] + 12, panels[1][2] - 12)
        y = map_value(tv, min(theta), max(theta), panels[1][3] - 12, panels[1][1] + 12)
        draw_circle(pixels, x, y, 5, basic_color(ev, ke_min, ke_max))

    nbins = min(40, max(5, int(len(ke_ev) ** 0.5)))
    hist = [0.0] * nbins
    for ev, weight in zip(ke_ev, weights):
        if ke_max <= ke_min:
            idx = 0
        else:
            idx = min(nbins - 1, int((ev - ke_min) * nbins / (ke_max - ke_min)))
        hist[idx] += weight
    hist_max = max(hist) if hist else 1.0
    x0, y0, x1, y1 = panels[2]
    for i, value in enumerate(hist):
        bx0 = x0 + 8 + i * (x1 - x0 - 16) // nbins
        bx1 = x0 + 8 + (i + 1) * (x1 - x0 - 16) // nbins - 2
        by0 = map_value(value, 0.0, hist_max, y1 - 8, y0 + 12)
        draw_rect(pixels, bx0, by0, bx1, y1 - 8, (70, 119, 170))

    write_png(output_path, width, height, pixels)


def make_basic_histogram_png(data, output_path):
    width, height = 1000, 720
    pixels = [bytearray([248, 248, 246] * width) for _ in range(height)]
    theta_bins = max(data["theta_bin"]) + 1
    z_bins = max(data["z_bin"]) + 1
    grid = [[0.0 for _ in range(z_bins)] for _ in range(theta_bins)]
    for itheta, iz, weight in zip(
        data["theta_bin"], data["z_bin"], data["sum_weight"]
    ):
        grid[itheta][iz] += weight
    panel = (90, 70, 930, 610)
    draw_rect(pixels, panel[0], panel[1], panel[2], panel[3], (255, 255, 255))
    draw_heatmap(pixels, grid, panel)
    draw_rect(pixels, panel[0], panel[3] - 1, panel[2], panel[3] + 1, (50, 50, 50))
    draw_rect(pixels, panel[0] - 1, panel[1], panel[0] + 1, panel[3], (50, 50, 50))
    write_png(output_path, width, height, pixels)


def make_plot(data, output_path, title, plot_type, bins):
    try:
        make_matplotlib_plot(data, output_path, title, plot_type, bins)
    except ModuleNotFoundError:
        make_basic_png(data, output_path, plot_type, bins)


def main():
    parser = argparse.ArgumentParser(
        description="Plot electrons collected by a CylindricalScreenFlux reduced diagnostic."
    )
    parser.add_argument(
        "csv",
        nargs="?",
        default="diags/reducedfiles/screen.csv",
        type=Path,
        help="CylindricalScreenFlux CSV file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=Path("cylindrical_screen_flux.png"),
        type=Path,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--species",
        help="Only plot one species, for example electrons.",
    )
    parser.add_argument(
        "--plot",
        choices=("summary", "heatmap"),
        default="summary",
        help="Plot style. Use heatmap for the unwrapped theta-z pseudocolor image.",
    )
    parser.add_argument(
        "--bins",
        nargs=2,
        default=(120, 120),
        metavar=("Z_BINS", "THETA_BINS"),
        type=int,
        help="Number of z and theta bins for --plot heatmap.",
    )
    args = parser.parse_args()

    data = filter_data(read_screen_csv(args.csv), args.species)
    title = f"Cylindrical screen flux: {args.csv}"
    make_plot(data, args.output, title, args.plot, args.bins)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
