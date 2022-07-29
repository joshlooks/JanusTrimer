import os
import re
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.ticker as plticker
import matplotlib.colors as colors
import glob

from scipy.interpolate import interp2d

kBT = 1.1

fig, axs = plt.subplots(nrows=3, ncols=6, figsize=(20, 12),
                        subplot_kw={'xticks': [], 'yticks': []})

files = glob.glob("data/vary_patch_R100/*.csv")

def sort_temp(val):
    return int(val.split("\\")[1].split("_")[0])

files.sort(key = sort_temp)

# print(files)

theta1s = np.linspace(-2.5, 92.5, 20)
theta2s = np.linspace(-2.5, 92.5, 20)

for ax, filename in zip(axs.flat, files[1:]):
    df = pd.read_csv(filename)

    df['probability'] = np.sin(df['theta1'] * np.pi / 180) * np.sin(df['theta2'] * np.pi / 180) * np.exp(-(df['energy'] - df['energy'].min()) / kBT)

    df['normalised_probability'] = df['probability'] / df['probability'].sum()

    test = df.pivot(index='theta1', columns='theta2', values='normalised_probability')
    test2 = test.to_numpy()

    ax.pcolormesh(theta1s, theta2s, test2, edgecolors='k', linewidth=0.003, cmap = 'viridis')
    ax.axis('square')

    # ax.imshow(np.flip(test2, 0), interpolation = None, cmap='viridis', norm=colors.SymLogNorm(vmin=test2.min(), vmax=test2.max(), linthresh=0.001, base=0.1))
    # ax.imshow(np.flip(test2, 0), interpolation = 'lanczos', cmap='viridis')
    ax.set_title("$\gamma = " + str(int(filename.split("\\")[1].split("_")[0])) + "\degree$")

plt.tight_layout()
plt.savefig("plots/plot_all_vary_patch_R100.png")

plt.show()