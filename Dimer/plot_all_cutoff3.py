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
import math

from scipy.interpolate import interp2d

kBT = 1.1

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

# files += glob.glob("data/vary_patch_J90_DF50/*.csv")
# files += glob.glob("data/vary_patch_J90_DF60/*.csv")
files = glob.glob("data/vary_patch_J90_DF70/*.csv")
files += glob.glob("data/vary_patch/J90*.csv")
files += glob.glob("data/vary_patch_J90_DF90/*.csv")
files += glob.glob("data/vary_patch_J90_DF100/*.csv")


# def sort_temp(val):
#     return int(val.split("\\")[1].split("_")[0][1:])

# files.sort(key = sort_temp)

files_grouped = [files[10*i:10*i+10] for i in range(4)]

print(files_grouped)

theta1s = np.linspace(2.5, 87.5, 18)
theta2s = np.linspace(2.5, 87.5, 18)

theta12s = np.linspace(2.5, 87.5, 18)

different_df = [0.70, 0.80, 0.90, 1.00]


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

    # print(str(theta12s[np.argmax(test3)]) + "," + str(np.max(test3)))

    # cutoff_index = 0

    # for i in range(len(test3)):
    #     if test3[i] < 0.001:
    #         cutoff_index = i
    #         break

    # # print(test3)
    # print(theta12s[cutoff_index])
    # print(test3[cutoff_index])

    # print(int(filename[0].split("\\")[0].split("_")[0][1:]))

    # axs.errorbar(different_radius[j], theta12s[cutoff_index], yerr=5, color="tab:blue", marker="o")
    # axs[1].errorbar(different_radius[j], theta12s[np.argmax(test3)], yerr=5, color="tab:blue", marker="o")

    axs[0].plot(theta12s, 100 * test3 / np.sum(test3), marker='o', linestyle='dashed', linewidth=2, markersize=4, label=different_df[j])

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

# values = np.linspace(0.0, 100.0, 1000)

# formula = 90 - np.degrees(np.arccos((values + 0.51/2.0) / (values + 0.8)))

# axs.plot(values, formula, color="black", linestyle="dashed")

# formula = 0.0 * values + 45

# axs[1].plot(values, formula, color="black", linestyle="dashed")

# print(values)
# print(formula)

files = glob.glob("data/vary_patch_J90_DS41/*.csv")
files += glob.glob("data/vary_patch/J90*.csv")
files += glob.glob("data/vary_patch_J90_DS61/*.csv")
files += glob.glob("data/vary_patch_J90_DS71/*.csv")
files += glob.glob("data/vary_patch_J90_DS81/*.csv")
files += glob.glob("data/vary_patch_J90_DS91/*.csv")


# def sort_temp(val):
#     return int(val.split("\\")[1].split("_")[0][1:])

# files.sort(key = sort_temp)

files_grouped = [files[10*i:10*i+10] for i in range(6)]

print(files_grouped)

theta1s = np.linspace(2.5, 87.5, 18)
theta2s = np.linspace(2.5, 87.5, 18)

theta12s = np.linspace(2.5, 87.5, 18)

different_df = [0.41, 0.51, 0.61, 0.71, 0.81, 0.91]


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

    # print(str(theta12s[np.argmax(test3)]) + "," + str(np.max(test3)))

    # cutoff_index = 0

    # for i in range(len(test3)):
    #     if test3[i] < 0.001:
    #         cutoff_index = i
    #         break

    # # print(test3)
    # print(theta12s[cutoff_index])
    # print(test3[cutoff_index])

    # print(int(filename[0].split("\\")[0].split("_")[0][1:]))

    # axs.errorbar(different_radius[j], theta12s[cutoff_index], yerr=5, color="tab:blue", marker="o")
    # axs[1].errorbar(different_radius[j], theta12s[np.argmax(test3)], yerr=5, color="tab:blue", marker="o")

    axs[1].plot(theta12s, 100 * test3 / np.sum(test3), marker='o', linestyle='dashed', linewidth=2, markersize=4, label=different_df[j])





axs[0].set_xlabel(r"$\theta_i$ or $\theta_j$ (deg)")
axs[0].set_ylabel(r"Occurence Probability (%)")
axs[0].xaxis.set_ticks(np.arange(0, 100.01, 20))
# axs.yaxis.set_ticks(np.arange(0, 90.01, 15))
axs[0].set_xlim([0, 90])
axs[0].set_ylim([0, 14])

axs[1].set_xlabel(r"$\theta_i$ or $\theta_j$ (deg)")
axs[1].set_ylabel(r"Occurence Probability (%)")
axs[1].xaxis.set_ticks(np.arange(0, 100.01, 20))
# axs.yaxis.set_ticks(np.arange(0, 90.01, 15))
axs[1].set_xlim([0, 90])
axs[1].set_ylim([0, 14])


# axs[1].set_xlabel(r"R")
# axs[1].set_ylabel(r"most likely angle")
# axs[1].xaxis.set_ticks(np.arange(0, 80.01, 10))
# axs[1].yaxis.set_ticks(np.arange(0, 90.01, 15))
# axs[1].set_xlim([0, 80])
# axs[1].set_ylim([0, 90])


axs[0].grid(alpha=0.2)
axs[1].grid(alpha=0.2)

axs[0].legend(labels=["$d_f=0.70$", "$d_f=0.80$", "$d_f=0.90$", "$d_f=1.00$"], loc="upper right")
axs[1].legend(labels=["$d_s=0.41$", "$d_s=0.51$", "$d_s=0.61$", "$d_s=0.71$", "$d_s=0.81$", "$d_s=0.91$"], loc="upper right")

plt.tight_layout()
plt.savefig("plots/figure1000.png")

plt.show()