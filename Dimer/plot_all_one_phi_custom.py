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

fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))

files = glob.glob("data/vary_patch/J90*.csv")
files += glob.glob("data/vary_patch_J90_R4/*.csv")
files += glob.glob("data/vary_patch_J90_R6/*.csv")
files += glob.glob("data/vary_patch_J90_R8/*.csv")
files += glob.glob("data/vary_patch_J90_R14/*.csv")
files += glob.glob("data/vary_patch_J90_R20/*.csv")
files += glob.glob("data/vary_patch_J90_R26/*.csv")
files += glob.glob("data/vary_patch_J90_R32/*.csv")
files += glob.glob("data/vary_patch_J90_R48/*.csv")
files += glob.glob("data/vary_patch_J90_R64/*.csv")

different_radius = [10.0, 4.0, 6.0, 8.0, 14.0, 20.0, 26.0, 32.0, 48.0, 64.0]

def sort_temp(val):
    return int(val.split("\\")[1].split("_")[0][1:])

files.sort(key = sort_temp)

files_grouped = [files[10*i:10*i+10] for i in range(10)]

print(files_grouped)

theta1s = np.linspace(2.5, 87.5, 18)
theta2s = np.linspace(2.5, 87.5, 18)

theta12s = np.linspace(2.5, 87.5, 18)


# df_bench = pd.read_csv('data/vary_patch/J90_P00.csv')

for (j, filename) in enumerate(files_grouped):
    added_probability = np.zeros((18, 18))

    for i in range(10):
        # print(filename[i])
        df = pd.read_csv(filename[i])

        df['probability'] = np.sin(df['theta1'] * np.pi / 180) * np.sin(df['theta2'] * np.pi / 180) * np.exp(-(df['energy'] - df['energy'].min()) / kBT)

        df['normalised_probability'] = df['probability'] / df['probability'].sum()

        test = df.pivot(index='theta1', columns='theta2', values='normalised_probability')
        added_probability += test.to_numpy()

    added_probability /= 10

    # test3 = np.diagonal(added_probability)
    # test3 = test3 / np.sum(test3)

    test3 = np.sum(added_probability, axis = 1)

    print(str(theta12s[np.argmax(test3)]) + "," + str(np.max(test3)))

    plt.plot(theta12s, 100 * test3 / np.sum(test3), marker='o', linestyle='dashed', linewidth=2, markersize=4, label="$" + str(different_radius[j]) + "$")

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

    # ax.pcolormesh(theta1s, theta2s, added_probability, edgecolors='k', linewidth=0.0, cmap = 'viridis', shading = "nearest")
    # ax.axis('square')

    # ax.imshow(np.flip(test2, 0), interpolation = None, cmap='viridis', norm=colors.SymLogNorm(vmin=test2.min(), vmax=test2.max(), linthresh=0.001, base=0.1))
    # ax.imshow(np.flip(test2, 0), interpolation = 'lanczos', cmap='viridis')
    # ax.set_title("$\gamma = " + str(int(filename[0].split("\\")[1].split("_")[0][1:])) + "\degree$")

# axs.flat[-1].axis('off')

# plt.title(f"probability distribution for $d_s = 0.51$, $d_w = 0.80$, averaged $\phi$")

plt.xlabel(r"$\theta_i, \theta_j$")
plt.ylabel(r"probability %")

plt.gca().xaxis.set_ticks(np.arange(0, 90.01, 15))

plt.grid()

plt.legend(loc="upper right")

plt.tight_layout()
plt.savefig("plots/figure2_test.png")

plt.show()