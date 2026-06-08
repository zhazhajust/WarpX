#!/usr/bin/env python3
#
# Copyright 2026
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

import argparse
import math
import random
from pathlib import Path


def gaussian_offsets(n, sigma, extent):
    if n == 1:
        return [(0.0, 1.0)]
    offsets = []
    for i in range(n):
        x = -extent + 2.0 * extent * i / (n - 1)
        weight = math.exp(-0.5 * (x / sigma) ** 2)
        offsets.append((x, weight))
    return offsets


def format_values(values):
    return " ".join(f"{value:.16e}" for value in values)


def build_particles():
    # Keep the four beams away from the theta = +/- pi unwrap seam so the
    # pseudocolor plot shows four intact spots.
    beam_theta = [-0.75 * math.pi, -0.25 * math.pi, 0.25 * math.pi, 0.75 * math.pi]
    beam_z = [0.0, 0.0, 0.0, 0.0]
    r_start = 0.82
    radial_u_mean = 0.22
    radial_u_std = 0.035
    rng = random.Random(20260608)
    theta_offsets = gaussian_offsets(15, sigma=0.09, extent=0.24)
    z_offsets = gaussian_offsets(21, sigma=0.11, extent=0.30)

    x = []
    y = []
    z = []
    ux = []
    uy = []
    uz = []
    weight = []

    for theta0, z0 in zip(beam_theta, beam_z):
        for dtheta, wtheta in theta_offsets:
            theta = theta0 + dtheta
            for dz, wz in z_offsets:
                particle_weight = 1.0e4 * wtheta * wz
                if particle_weight < 50.0:
                    continue
                radial_u = max(0.08, rng.gauss(radial_u_mean, radial_u_std))
                x.append(r_start * math.cos(theta))
                y.append(r_start * math.sin(theta))
                z.append(z0 + dz)
                ux.append(-radial_u * math.cos(theta))
                uy.append(-radial_u * math.sin(theta))
                uz.append(0.0)
                weight.append(particle_weight)

    return x, y, z, ux, uy, uz, weight


def write_input(path):
    x, y, z, ux, uy, uz, weight = build_particles()
    text = f"""# Copyright 2026
#
# Four Gaussian electron beams crossing an inward cylindrical screen.
# Generate this file with:
#   python Examples/Tests/cylindrical_screen_flux/generate_four_beam_example.py

max_step = 6

amr.n_cell = 32 64
amr.max_grid_size = 32
amr.max_level = 0

geometry.dims = RZ
geometry.prob_lo = 0.0 -1.0
geometry.prob_hi = 1.0  1.0

boundary.field_lo = none periodic
boundary.field_hi = none periodic
boundary.particle_lo = none periodic
boundary.particle_hi = absorbing periodic

warpx.const_dt = 1.2e-9
warpx.serialize_initial_conditions = 1
warpx.verbose = 1

algo.current_deposition = direct
algo.field_gathering = energy-conserving
algo.particle_shape = 1

particles.species_names = electrons

electrons.charge = -q_e
electrons.mass = m_e
electrons.injection_style = MultipleParticles
electrons.multiple_particles_pos_x = {format_values(x)}
electrons.multiple_particles_pos_y = {format_values(y)}
electrons.multiple_particles_pos_z = {format_values(z)}
electrons.multiple_particles_ux = {format_values(ux)}
electrons.multiple_particles_uy = {format_values(uy)}
electrons.multiple_particles_uz = {format_values(uz)}
electrons.multiple_particles_weight = {format_values(weight)}
electrons.do_not_deposit = 1

warpx.reduced_diags_names = screen
screen.type = CylindricalScreenFlux
screen.intervals = 1
screen.r0 = 0.5
screen.species = electrons
screen.write_particles = 0
screen.bins_energy = 80
screen.bins_theta = 120
screen.bins_z = 120
screen.energy_max = 30000.0
screen.energy_min = 0.0
screen.z_max =  1.0
screen.z_min = -1.0
screen.time_interval = 2.4e-9
screen.histogram_file_name = diags/reducedfiles/screen_hist.csv
"""
    path.write_text(text)
    print(f"Wrote {path} with {len(x)} electrons")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a four-beam CylindricalScreenFlux example input."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).with_name(
            "inputs_test_rz_cylindrical_screen_flux_four_beams"
        ),
        help="Output WarpX input file.",
    )
    args = parser.parse_args()
    write_input(args.output)


if __name__ == "__main__":
    main()
