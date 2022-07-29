import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
import matplotlib.colors as colors

theta1 = 80
theta2 = 80

ds = 0.51
dw = 0.80
R = 6

df = pd.read_csv("potential_80_80_R6_HR_080.csv")

test = df.pivot(index = "z", columns = "x", values = "energy")

test2 = test.to_numpy()

f, ax = plt.subplots(figsize=(8, 8))

norm = colors.SymLogNorm(vmin=-1000, vmax=1000, linthresh=0.00001, base=10)

# plt.imshow(np.flip(test2, 0), aspect = "equal", cmap='viridis')
plt.imshow(np.flip(test2, 0), aspect = "equal", cmap='nipy_spectral', extent=[-2*R,2*R,-2*R,4*R + ds], interpolation = 'none', norm = norm)

half_angle = -theta1

half1 = patches.Wedge((0, 0), R, half_angle, 180 + half_angle, linewidth=0, facecolor='tab:blue')
half2 = patches.Wedge((0, 0), R, 180 + half_angle, half_angle, linewidth=0, facecolor='tab:red')

half_angle = 180 + theta2

half3 = patches.Wedge((0, 2*R + ds), R, half_angle, 180 + half_angle, linewidth=0, facecolor='tab:blue')
half4 = patches.Wedge((0, 2*R + ds), R, 180 + half_angle, half_angle, linewidth=0, facecolor='tab:red')

# Add the patch to the Axes
ax.add_patch(half1)
ax.add_patch(half2)
ax.add_patch(half3)
ax.add_patch(half4)

plt.xlabel(r"x $[\sigma]$")
plt.ylabel(r"z $[\sigma]$")

ax.set_title(f"LJ Potential with $d_s = {ds}, d_w = {dw}, R = {R}$")

plt.colorbar().set_label("energy potential")

plt.tight_layout()
plt.savefig("plots/plot_potential2_HR_080.png")

plt.show()