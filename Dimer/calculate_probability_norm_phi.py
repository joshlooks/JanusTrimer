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

files = glob.glob("data/vary_patch/J90*.csv")

print(files)

def sort_temp(val):
    return int(val.split("\\")[1].split(".")[0].split("_")[1][1:])

files.sort(key = sort_temp)

files_grouped = [files[10*i:10*i+10] for i in range(1)]

print(files_grouped)

theta1s = np.linspace(2.5, 87.5, 18)
theta2s = np.linspace(2.5, 87.5, 18)

theta12s = np.linspace(2.5, 87.5, 18)


df_bench = pd.read_csv('data/vary_patch/J90_P00.csv')

for filename in files_grouped:
    added_probability = np.zeros((18, 18))

    for i in range(10):
        print(filename[i])
        df = pd.read_csv(filename[i])

        df['energy'] = df['e1'] + df['e2'] + df['e3'] + df['e4'] + 0.974025974026 * df['e5'] + 0.974025974026 * df['e6'] + 0.974025974026 * df['e7'] + 0.974025974026 * df['e8'] + 0.974025974026 * df['e9'] + 0.974025974026 * df['e10']

        df['probability'] = np.sin(df_bench['theta1'] * np.pi / 180) * np.sin(df_bench['theta2'] * np.pi / 180) * np.exp(-(df['energy'] - df['energy'].min()) / kBT)

        df['normalised_probability'] = df['probability'] / df['probability'].sum()

        test = df.pivot(index='theta1', columns='theta2', values='normalised_probability')
        added_probability += test.to_numpy()

    added_probability /= 10

    df_save = pd.DataFrame()

    df_save['theta1'] = df_bench['theta1']
    df_save['theta2'] = df_bench['theta2']

    df_save['probability'] = np.flip(added_probability, 1).flatten()

    df_save.to_csv("temp/probability_126_J90_PA_MOD.csv")
        