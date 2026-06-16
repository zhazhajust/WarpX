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

signal = np.zeros((thetas.size * phis.size, times.size))
for row in data:
    it = np.where(times == row[3])[0][0]
    ith = np.where(thetas == row[4])[0][0]
    iph = np.where(phis == row[5])[0][0]
    signal[ith * phis.size + iph, it] = row[7]

vmax = np.max(np.abs(signal))
angle_index = np.argmax(np.max(np.abs(signal), axis=1))

fig, axes = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)
image = axes[0].imshow(
    signal,
    aspect="auto",
    origin="lower",
    cmap="bwr",
    vmin=-vmax,
    vmax=vmax,
    extent=[times[0] * 1.0e15, times[-1] * 1.0e15, 0, signal.shape[0] - 1],
)
axes[0].set_xlabel("radiation time (fs)")
axes[0].set_ylabel("observation angle index")
axes[0].set_title("Time-domain radiation Ey")
fig.colorbar(image, ax=axes[0], label="Ey (V/m)")

axes[1].plot(times * 1.0e15, signal[angle_index, :], color="black")
axes[1].set_xlabel("radiation time (fs)")
axes[1].set_ylabel("Ey (V/m)")
axes[1].set_title(f"Strongest observation angle index: {angle_index}")
fig.savefig("time_domain_radiation.png", dpi=150)
