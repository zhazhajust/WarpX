#!/usr/bin/env python3

from math import isfinite
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


path = Path("diags/reducedfiles/rad.txt")
if not path.exists():
    raise RuntimeError(f"Missing TimeDomainRadiation diagnostic: {path}")

rows = []
for line in path.read_text().splitlines():
    if not line or line.startswith("#"):
        continue
    parts = line.split()
    if len(parts) != 9:
        raise RuntimeError(f"Unexpected diagnostic row: {line}")
    rows.append([float(v) for v in parts])

if not rows:
    raise RuntimeError("TimeDomainRadiation diagnostic did not write data rows")

field_values = [value for row in rows for value in row[6:9]]
if not all(isfinite(value) for value in field_values):
    raise RuntimeError("TimeDomainRadiation field contains non-finite values")

if max(abs(value) for value in field_values) <= 0.0:
    raise RuntimeError("TimeDomainRadiation field is identically zero")

data = np.array(rows)
data = data[data[:, 0] == np.max(data[:, 0])]
times = np.unique(data[:, 3])
thetas = np.unique(data[:, 4])
phis = np.unique(data[:, 5])

phi_index = int(np.argmin(np.abs(phis)))
components = ("Ex", "Ey", "Ez")
theta_signals = []
for component_index in range(3):
    theta_signal = np.zeros((thetas.size, times.size))
    for row in data:
        if row[5] != phis[phi_index]:
            continue
        it = np.where(times == row[3])[0][0]
        ith = np.where(thetas == row[4])[0][0]
        theta_signal[ith, it] = row[6 + component_index]
    theta_signals.append(theta_signal)

component_index = int(np.argmax([np.max(np.abs(values)) for values in theta_signals]))
component = components[component_index]
theta_signal = theta_signals[component_index]

vmax = np.max(np.abs(theta_signal))
theta_index = np.argmax(np.max(np.abs(theta_signal), axis=1))

fig, axes = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)
image = axes[0].imshow(
    theta_signal,
    aspect="auto",
    origin="lower",
    cmap="bwr",
    vmin=-vmax,
    vmax=vmax,
    extent=[times[0] * 1.0e15, times[-1] * 1.0e15, thetas[0], thetas[-1]],
)
axes[0].set_xlabel("radiation time (fs)")
axes[0].set_ylabel("theta (rad)")
axes[0].set_title(f"Time-domain radiation {component}, phi = {phis[phi_index]:.3f} rad")
fig.colorbar(image, ax=axes[0], label=f"{component} (V/m)")

axes[1].plot(times * 1.0e15, theta_signal[theta_index, :], color="black")
axes[1].set_xlabel("radiation time (fs)")
axes[1].set_ylabel(f"{component} (V/m)")
axes[1].set_title(f"Strongest theta: {thetas[theta_index]:.3f} rad")
fig.savefig("time_domain_radiation.png", dpi=150)
