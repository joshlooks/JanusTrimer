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

fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(12, 6),
                        subplot_kw={'xticks': [], 'yticks': []})

files = glob.glob("data/vary_patch/*.csv")

def sort_temp(val):
    return int(val.split("\\")[1].split("_")[0][1:])

files.sort(key = sort_temp)

files_grouped = [files[10*i:10*i+10] for i in range(17)]

print(files_grouped)

theta1s = np.linspace(2.5, 87.5, 18)
theta2s = np.linspace(2.5, 87.5, 18)

# df_bench = pd.read_csv('data/vary_patch/J90_P00.csv')

for ax, filename in zip(axs.flat, files_grouped[2:]):
    added_probability = np.zeros((18, 18))

    for i in range(10):
        # print(filename[i])
        df = pd.read_csv(filename[i])

        df['probability'] = np.sin(df['theta1'] * np.pi / 180) * np.sin(df['theta2'] * np.pi / 180) * np.exp(-(df['energy'] - df['energy'].min()) / kBT)

        df['normalised_probability'] = df['probability'] / df['probability'].sum()

        test = df.pivot(index='theta1', columns='theta2', values='normalised_probability')
        added_probability += test.to_numpy()

    added_probability /= 10

    # if int(filename[0].split("\\")[1].split("_")[0][1:]) == 90:
    #     # print(added_probability)
    #     df_save = pd.DataFrame()

    #     df_save['theta1'] = df['theta1']
    #     df_save['theta2'] = df['theta2']

    #     df_save['probability'] = np.flip(added_probability, 1).flatten()

    #     # df = pd.read_csv('data/vary_patch/J90_P00.csv')

    #     # df['probability'] = np.flip(added_probability, 1).flatten()

    #     df_save.to_csv("probability_LJ126_A05_T11_J90_R32_PA.csv")

    # df = 

    out = ax.pcolormesh(theta1s, theta2s, added_probability * 100, edgecolors='k', linewidth=0.0, cmap = 'jet', shading = "nearest", norm=colors.SymLogNorm(vmin=0.0, vmax=20, linthresh=0.01))
    ax.axis('square')

    # ax.imshow(np.flip(test2, 0), interpolation = None, cmap='viridis', norm=colors.SymLogNorm(vmin=test2.min(), vmax=test2.max(), linthresh=0.001, base=0.1))
    # ax.imshow(np.flip(test2, 0), interpolation = 'lanczos', cmap='viridis')
    ax.set_title("$\gamma = " + str(int(filename[0].split("\\")[1].split("_")[0][1:])) + "\degree$")

axs.flat[-1].axis('off')

plt.tight_layout()

fig.subplots_adjust(right=0.875)
cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
cbar = fig.colorbar(out, cax=cbar_ax, extend='max')
cbar.set_label("probability [%]")

# plt.savefig("plots/figure1.png")

plt.show()