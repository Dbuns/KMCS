# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 13:47:09 2026

@author: MonteiroCunhaD1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data transcribed from your table
data = [
    # frequency, temperature, density
    (0.5, 600, 0.111958),
    (0.5, 700, 0.0341091),
    (0.5, 750, 0.0164526),
    (0.5, 800, 0.00922951),
    (0.5, 850, 0.00280898),

    (2, 600, 0.127608),
    (2, 700, 0.0678168),
    (2, 750, 0.0409309),
    (2, 800, 0.018459),
    (2, 850, 0.00922951),
    (2, 900, 0.00521668),

    (5, 600, 0.136436),
    (5, 700, 0.102327),
    (5, 750, 0.063804),
    (5, 800, 0.0357142),
    (5, 850, 0.0140449),
    (5, 900, 0.00682181),

    (10, 600, 0.15008),
    (10, 700, 0.120786),
    (10, 750, 0.0814605),
    (10, 800, 0.0533706),
    (10, 850, 0.0280898),
    (10, 900, 0.0136436),

    (20, 600, 0.14687),
    (20, 700, 0.132022),
    (20, 750, 0.0951041),
    (20, 800, 0.0714284),
    (20, 850, 0.0413322),
    (20, 900, 0.0196629),

    (35, 600, 0.141252),
    (35, 700, 0.132022),
    (35, 750, 0.110754),
    (35, 800, 0.0914925),
    (35, 850, 0.0521668),
    (35, 900, 0.0325039),

    (50, 600, 0.144863),
    (50, 700, 0.148073),
    (50, 750, 0.118379),
    (50, 800, 0.104735),
    (50, 850, 0.0605937),
    (50, 900, 0.0429373),
]
df = pd.DataFrame(data, columns=["frequency", "temperature", "density"])

# Grid for fitted surface
freq_grid = np.linspace(df["frequency"].min(), df["frequency"].max(), 200)
temp_grid = np.linspace(df["temperature"].min(), df["temperature"].max(), 200)
F, T = np.meshgrid(freq_grid, temp_grid)

# DOE response-surface fit:
# y = b0 + b1*x1 + b2*x2 + b3*x1*x2 + b4*x1^2 + b5*x2^2
freq_min, freq_max = df["frequency"].min(), df["frequency"].max()
temp_min, temp_max = df["temperature"].min(), df["temperature"].max()

x1 = 2 * (df["frequency"].to_numpy() - freq_min) / (freq_max - freq_min) - 1
x2 = 2 * (df["temperature"].to_numpy() - temp_min) / (temp_max - temp_min) - 1
density = df["density"].to_numpy()

X1 = 2 * (F - freq_min) / (freq_max - freq_min) - 1
X2 = 2 * (T - temp_min) / (temp_max - temp_min) - 1


def doe_matrix(x1_values, x2_values):
    return np.column_stack([
        np.ones_like(x1_values),
        x1_values,
        x2_values,
        x1_values * x2_values,
        x1_values ** 2,
        x2_values ** 2,
    ])


A = doe_matrix(x1, x2)
coeffs, *_ = np.linalg.lstsq(A, density, rcond=None)

grid_A = doe_matrix(X1.ravel(), X2.ravel())
Z = (grid_A @ coeffs).reshape(F.shape)


def predict_density(frequency_values, temperature_values):
    frequency_values = np.asarray(frequency_values)
    temperature_values = np.asarray(temperature_values)
    x1_values = 2 * (frequency_values - freq_min) / (freq_max - freq_min) - 1
    x2_values = 2 * (temperature_values - temp_min) / (temp_max - temp_min) - 1
    return doe_matrix(x1_values, x2_values) @ coeffs

fitted_density = A @ coeffs
ss_res = np.sum((density - fitted_density) ** 2)
ss_tot = np.sum((density - density.mean()) ** 2)
r_squared = 1 - ss_res / ss_tot

print("DOE fit in coded variables:")
print("x1 = coded frequency, x2 = coded temperature")
for name, value in zip(["b0", "b1", "b2", "b3", "b4", "b5"], coeffs):
    print(f"{name} = {value:.6g}")
print(f"R^2 = {r_squared:.4f}")

# OVAT slices to show beside the response surface
ovat_temperature = 800
ovat_frequency = 20
freq_slice = freq_grid
temp_slice = temp_grid
freq_slice_fit = predict_density(
    freq_slice,
    np.full_like(freq_slice, ovat_temperature),
)
temp_slice_fit = predict_density(
    np.full_like(temp_slice, ovat_frequency),
    temp_slice,
)
freq_slice_data = df[df["temperature"] == ovat_temperature].sort_values("frequency")
temp_slice_data = df[df["frequency"] == ovat_frequency].sort_values("temperature")

# Plot
fig = plt.figure(figsize=(11, 7.5))
gs = fig.add_gridspec(
    2,
    3,
    width_ratios=[4.8, 1.45, 0.18],
    height_ratios=[1.25, 4.2],
    wspace=0.12,
    hspace=0.08,
)
ax_top = fig.add_subplot(gs[0, 0])
ax_main = fig.add_subplot(gs[1, 0], sharex=ax_top)
ax_side = fig.add_subplot(gs[1, 1], sharey=ax_main)
cax = fig.add_subplot(gs[1, 2])
ax_blank = fig.add_subplot(gs[0, 1:])
ax_blank.axis("off")

levels = np.linspace(density.min(), Z.max(), 10)
lines = ax_main.contour(F, T, Z, levels=levels, cmap="viridis", linewidths=1.0)
ax_main.clabel(lines, inline=True, fontsize=8, fmt="%.3f")

ax_main.scatter(
    df["frequency"],
    df["temperature"],
    s=38,
    facecolors="white",
    edgecolors="black",
    linewidths=0.8,
    zorder=3,
    clip_on=False,
)
ax_main.axhline(
    ovat_temperature,
    color="tab:blue",
    linestyle="--",
    linewidth=1.2,
)
ax_main.axvline(
    ovat_frequency,
    color="tab:orange",
    linestyle="--",
    linewidth=1.2,
)
ax_main.set_xlabel("Frequency (Hz)")
ax_main.set_ylabel("Temperature (Celsius)")
ax_main.grid(True, color="0.9", linewidth=0.6)

ax_top.scatter(
    freq_slice_data["frequency"],
    freq_slice_data["density"],
    s=32,
    facecolors="white",
    edgecolors="tab:blue",
    linewidths=1.0,
    zorder=3,
    clip_on=False,
)
ax_top.set_ylabel("Density")
ax_top.set_title(f"OVAT: T = {ovat_temperature} C")
ax_top.grid(True, color="0.9", linewidth=0.6)
ax_top.tick_params(labelbottom=False)
ax_top.margins(y=0.15)

ax_side.scatter(
    temp_slice_data["density"],
    temp_slice_data["temperature"],
    s=32,
    facecolors="white",
    edgecolors="tab:orange",
    linewidths=1.0,
    zorder=3,
    clip_on=False,
)
ax_side.set_xlabel("Density")
ax_side.set_title(f"OVAT: f = {ovat_frequency} Hz")
ax_side.grid(True, color="0.9", linewidth=0.6)
ax_side.tick_params(labelleft=False)
ax_side.margins(x=0.15)

fig.suptitle("Number density per nm^2", y=0.98)
fig.colorbar(lines, cax=cax, label="Number density / nm^2")
fig.subplots_adjust(top=0.91)

out = "density_surface_doe.png"
fig.savefig(out, dpi=300)
plt.show()
