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

# filename = 'data/vary_patch_96/J90_P00.csv'

files = glob.glob("data/vary_patch_96/J90*.csv")

df = pd.read_csv(files[0])

energy = ['e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8', 'e9', 'e10']



for file in files[1:]:
    df2 = pd.read_csv(file)

    df['energy'] += df2['energy']

    for e in energy:
        df[e] += df2[e]

for e in energy:
    df[e] /= len(files)

df['energy'] /= len(files)


# df2 = pd.read_csv(filename)

kBT = 1.1
# Ptot = 0.0

# df['probability'] = np.sin(df2['theta1'] * np.pi / 180) * np.sin(df2['theta2'] * np.pi / 180) * np.exp(-(df['energy'] - df['energy'].min()) / kBT)

# df['probability'] = df['probability'] / df['probability'].sum()

# df['theta1'] = df2['theta1']
# df['theta2'] = df2['theta2']

# df = pd.read_csv('data/vary_patch/J90_P00.csv')

# df['probability'] = np.flip(added_probability, 1).flatten()

# df.to_csv("test.csv")



# test = df.pivot(index='theta1', columns='theta2', values='probability')
# test2 = test.to_numpy()

# # print(test2)

# # font = {'size'   : 14}
# # matplotlib.rc('font', **font)

theta1s = np.linspace(2.5, 87.5, 18)
theta2s = np.linspace(2.5, 87.5, 18)

# fig = plt.figure(figsize = (8, 5))
# plt.pcolormesh(theta1s, theta2s, test2, edgecolors='k', linewidth=0.01, shading = "nearest")
# # plt.imshow(np.flip(test2, 0), interpolation='none', cmap='viridis', extent=[df['theta1'].min(),df['theta1'].max() + 5,df['theta2'].min(),df['theta2'].max() + 5])

# plt.colorbar().set_label("probability")
# plt.xlabel(r"$\theta_i$")
# plt.ylabel(r"$\theta_j$")
# plt.gca().xaxis.set_ticks(np.arange(0, 90.01, 10))
# plt.gca().yaxis.set_ticks(np.arange(0, 90.01, 10))

# # plt.title(f"probability distribution for $d_s = 0.51$, $d_w = 0.80$, $\phi = 0$")# for $\phi = {phi}$")
# plt.axis('square')
# # plt.show()

# plt.savefig("plots/" + (filename.split("/")[-1]).split(".")[0] + ".png")
# plt.show()
# plt.close()
good_df = pd.read_csv("data/vary_patch/J90_P00.csv")

fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(12, 6),
                        subplot_kw={'xticks': [], 'yticks': []})

energy = ['e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8', 'e9', 'e10']
energy.reverse()

for ax, energy_type in zip(axs.flat, energy):
    df['probability'] = np.sin(good_df['theta1'] * np.pi / 180) * np.sin(good_df['theta2'] * np.pi / 180) * np.exp(-(df[energy_type] - df[energy_type].min()) / kBT)

    df['normalised_probability'] = df['probability'] / df['probability'].sum()

    test = df.pivot(index='theta1', columns='theta2', values=energy_type)
    test2 = test.to_numpy()

    ax.pcolormesh(theta1s, theta2s, test2, edgecolors='k', linewidth=0.01, shading = "nearest" , norm=colors.SymLogNorm(vmin=test2.min(), vmax=test2.max(), linthresh=0.001, base=0.1))
    # ax.pcolormesh(theta1s, theta2s, test2, norm=colors.SymLogNorm(vmin=test2.min(), vmax=test2.max(), linthresh=0.001, base=0.1), edgecolors='k', linewidth=0.01)
    ax.axis('square')

    # ax.imshow(np.flip(test2, 0), interpolation = None, cmap='viridis', norm=colors.SymLogNorm(vmin=test2.min(), vmax=test2.max(), linthresh=0.001, base=0.1))
    # ax.imshow(np.flip(test2, 0), interpolation = None, cmap='viridis')
    ax.set_title(str(energy_type))

plt.tight_layout()
plt.savefig("plots/potential_testing/1_96.png")

plt.show()
plt.close()


fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(12, 6),
                        subplot_kw={'xticks': [], 'yticks': []})

energy = ['e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8', 'e9', 'e10']
energy.reverse()

df['test'] = df['energy']

for ax, energy_type in zip(axs.flat, energy):
    df['test'].values[:] = 0

    index = energy.index(energy_type)

    for i in range(0, index + 1):
        df['test'] += df[energy[i]]

    # df['probability'] = np.exp(-(df['test'] - df['test'].min()) / kBT)
    df['probability'] = np.sin(good_df['theta1'] * np.pi / 180) * np.sin(good_df['theta2'] * np.pi / 180) * np.exp(-(df['test'] - df['test'].min()) / kBT)

    df['normalised_probability'] = df['probability'] / df['probability'].sum()

    test = df.pivot(index='theta1', columns='theta2', values='normalised_probability')
    test2 = test.to_numpy()

    ax.pcolormesh(theta1s, theta2s, test2, edgecolors='k', linewidth=0.01, norm=colors.SymLogNorm(vmin=test2.min(), vmax=test2.max(), linthresh=0.001, base=0.1))
    ax.axis('square')

    # ax.imshow(np.flip(test2, 0), interpolation = None, cmap='viridis', norm=colors.SymLogNorm(vmin=test2.min(), vmax=test2.max(), linthresh=0.001, base=10))
    ax.set_title(str(energy_type))

plt.tight_layout()
plt.savefig("plots/potential_testing/2_96.png")

plt.show()
