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

files = glob.glob("data/vary_patch_J90_extra_phi/J90*.csv")


def sort_temp(val):
    return int(val.split("\\")[1].split("_")[0][1:])

files.sort(key = sort_temp)

# files_grouped = [files[10*i:10*i+10] for i in range(1)]

# print(files_grouped)

theta1s = np.linspace(2.5, 87.5, 18)
theta2s = np.linspace(2.5, 87.5, 18)

theta12s = np.linspace(2.5, 87.5, 18)


df_bench = pd.read_csv('data/vary_patch/J90_P00.csv')

# for filename in files_grouped:
min_energy = 100000000;

for i in range(19):
    df = pd.read_csv(files[i])

    if df['energy'].min() < min_energy:
        min_energy = df['energy'].min()


for i in range(19):
    print(files[i])
    df = pd.read_csv(files[i])

    df['probability'] = np.sin(df_bench['theta1'] * np.pi / 180) * np.sin(df_bench['theta2'] * np.pi / 180) * np.exp(-(df['energy'] - min_energy) / kBT)

    df['normalised_probability'] = df['probability'] / df['probability'].sum()

    test = df.pivot(index='theta1', columns='theta2', values='normalised_probability')

    df_save = pd.DataFrame()

    df_save['theta1'] = df['theta1']
    df_save['theta2'] = df['theta2']

    df_save['probability'] = df['probability']

    # df = pd.read_csv('data/vary_patch/J90_P00.csv')

    # df['probability'] = np.flip(added_probability, 1).flatten()

    df_save.to_csv("temp/extra_phi/probability_" + files[i].split('\\')[1])
        