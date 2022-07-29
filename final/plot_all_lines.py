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

df = pd.read_csv("trimer_05/trimer_05.csv")
theta1s = df['theta1'].unique()+30
theta2s = df['theta2'].unique()
theta3s = df['theta3'].unique()
phi1s = df['phi1'].unique()
phi2s = df['phi2'].unique()
phi3s = df['phi3'].unique()

# theta 1 plots
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))
#fig.delaxes(axs[1,2])
axslist = axs.flat
# t1 p1
ax = axslist[0]
for i in phi1s:
    df_current = df[df['phi1'].values==i].copy()
    df_current['probability'] = np.abs(np.sin((df['theta1'] - 30) * np.pi / 180) * np.sin((df['theta1'] + 30) * np.pi / 180)) * np.abs(np.sin((df['theta2'] - 30) * np.pi / 180) * np.sin((df['theta2'] + 30) * np.pi / 180)) * np.abs(np.sin((df['theta3'] - 30) * np.pi / 180) * np.sin((df['theta3'] + 30) * np.pi / 180)) * np.exp(-(df['energy'] - df['energy'].min()) / kBT)
    prob = df_current.pivot_table(index='theta1',values='probability',aggfunc=np.sum)
    prob = prob.to_numpy()/np.sum(prob.to_numpy()+1e-200)
    ax.plot(theta1s,prob,label=f"{i}")
ax.set_title("$\\theta_1 vs. \phi_1$")
# t1 p2
ax = axslist[1]
for i in phi2s:
    df_current = df[df['phi2'].values==i].copy()
    df_current['probability'] = np.abs(np.sin((df['theta1'] - 30) * np.pi / 180) * np.sin((df['theta1'] + 30) * np.pi / 180)) * np.abs(np.sin((df['theta2'] - 30) * np.pi / 180) * np.sin((df['theta2'] + 30) * np.pi / 180)) * np.abs(np.sin((df['theta3'] - 30) * np.pi / 180) * np.sin((df['theta3'] + 30) * np.pi / 180)) * np.exp(-(df['energy'] - df['energy'].min()) / kBT)
    prob = df_current.pivot_table(index='theta1',values='probability',aggfunc=np.sum)
    prob = prob.to_numpy()/np.sum(prob.to_numpy()+1e-200)
    ax.plot(theta1s,prob,label=f"{i}")
ax.set_title("$\\theta_1 vs. \phi_2$")
# t1 p3
ax = axslist[2]
for i in phi3s:
    df_current = df[df['phi3'].values==i].copy()
    df_current['probability'] = np.abs(np.sin((df['theta1'] - 30) * np.pi / 180) * np.sin((df['theta1'] + 30) * np.pi / 180)) * np.abs(np.sin((df['theta2'] - 30) * np.pi / 180) * np.sin((df['theta2'] + 30) * np.pi / 180)) * np.abs(np.sin((df['theta3'] - 30) * np.pi / 180) * np.sin((df['theta3'] + 30) * np.pi / 180)) * np.exp(-(df['energy'] - df['energy'].min()) / kBT)
    prob = df_current.pivot_table(index='theta1',values='probability',aggfunc=np.sum)
    prob = prob.to_numpy()/np.sum(prob.to_numpy()+1e-200)
    ax.plot(theta1s,prob,label=f"{i}")
ax.set_title("$\\theta_1 vs. \phi_3$")
# # t1 t2
# ax = axslist[3]
# for i in theta2s:
#     df_current = df[df['theta2'].values==i].copy()
#     df_current['probability'] = np.abs(np.sin((df['theta1'] - 30) * np.pi / 180) * np.sin((df['theta1'] + 30) * np.pi / 180)) * np.abs(np.sin((df['theta2'] - 30) * np.pi / 180) * np.sin((df['theta2'] + 30) * np.pi / 180)) * np.abs(np.sin((df['theta3'] - 30) * np.pi / 180) * np.sin((df['theta3'] + 30) * np.pi / 180)) * np.exp(-(df['energy'] - df['energy'].min()) / kBT)
#     prob = df_current.pivot_table(index='theta1',values='probability',aggfunc=np.sum)
#     prob = prob.to_numpy()/np.sum(prob.to_numpy()+1e-200)
#     ax.plot(theta1s,prob,label=f"{i}")
# ax.set_title("$\\theta_1 vs. \\theta_2$")
# ax.legend()
# # t1 t3
# ax = axslist[4]
# for i in theta2s:
#     df_current = df[df['theta3'].values==i].copy()
#     df_current['probability'] = np.abs(np.sin((df['theta1'] - 30) * np.pi / 180) * np.sin((df['theta1'] + 30) * np.pi / 180)) * np.abs(np.sin((df['theta2'] - 30) * np.pi / 180) * np.sin((df['theta2'] + 30) * np.pi / 180)) * np.abs(np.sin((df['theta3'] - 30) * np.pi / 180) * np.sin((df['theta3'] + 30) * np.pi / 180)) * np.exp(-(df['energy'] - df['energy'].min()) / kBT)
#     prob = df_current.pivot_table(index='theta1',values='probability',aggfunc=np.sum)
#     prob = prob.to_numpy()/np.sum(prob.to_numpy()+1e-200)
#     ax.plot(theta1s,prob,label=f"{i}")
# ax.set_title("$\\theta_1 vs. \\theta_3$")
# ax.legend()
axs.flat[-1].axis('off')
plt.tight_layout()
fig.subplots_adjust(right=0.875)
plt.savefig("trimer_plots/t1_linemaps.png")

# # theta 2 plots
# fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))
# fig.delaxes(axs[1,2])
# axslist = axs.flat
# # t2 p1
# ax = axslist[0]
# test = df.pivot_table(index='phi1', columns='theta2', values='probability',aggfunc=np.sum)
# added_probability = test.to_numpy()/np.sum(test.to_numpy())
# out = ax.pcolormesh(theta2s, phi1s, added_probability * 100, edgecolors='k', linewidth=0.0, cmap = 'jet', shading = "nearest", norm=colors.SymLogNorm(vmin=0.0, vmax=20, linthresh=0.01))
# ax.set_title("$\\theta_2 vs. \phi_1$")
# # t2 p2
# ax = axslist[1]
# test = df.pivot_table(index='phi2', columns='theta2', values='probability',aggfunc=np.sum)
# added_probability = test.to_numpy()/np.sum(test.to_numpy())
# out = ax.pcolormesh(theta2s, phi2s, added_probability * 100, edgecolors='k', linewidth=0.0, cmap = 'jet', shading = "nearest", norm=colors.SymLogNorm(vmin=0.0, vmax=20, linthresh=0.01))
# ax.set_title("$\\theta_2 vs. \phi_2$")
# # t2 p3
# ax = axslist[2]
# test = df.pivot_table(index='phi3', columns='theta2', values='probability',aggfunc=np.sum)
# added_probability = test.to_numpy()/np.sum(test.to_numpy())
# out = ax.pcolormesh(theta2s, phi3s, added_probability * 100, edgecolors='k', linewidth=0.0, cmap = 'jet', shading = "nearest", norm=colors.SymLogNorm(vmin=0.0, vmax=20, linthresh=0.01))
# ax.set_title("$\\theta_2 vs. \phi_3$")
# # t2 t1
# ax = axslist[3]
# test = df.pivot_table(index='theta1', columns='theta2', values='probability',aggfunc=np.sum)
# added_probability = test.to_numpy()/np.sum(test.to_numpy())
# out = ax.pcolormesh(theta2s, theta2s, added_probability * 100, edgecolors='k', linewidth=0.0, cmap = 'jet', shading = "nearest", norm=colors.SymLogNorm(vmin=0.0, vmax=20, linthresh=0.01))
# ax.set_title("$\\theta_2 vs. \\theta_1$")
# # t2 t3
# ax = axslist[4]
# test = df.pivot_table(index='theta3', columns='theta2', values='probability',aggfunc=np.sum)
# added_probability = test.to_numpy()/np.sum(test.to_numpy())
# out = ax.pcolormesh(theta2s, theta3s, added_probability * 100, edgecolors='k', linewidth=0.0, cmap = 'jet', shading = "nearest", norm=colors.SymLogNorm(vmin=0.0, vmax=20, linthresh=0.01))
# ax.set_title("$\\theta_2 vs. \\theta_3$")

# axs.flat[-1].axis('off')
# plt.tight_layout()
# fig.subplots_adjust(right=0.875)
# cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
# cbar = fig.colorbar(out, cax=cbar_ax, extend='max')
# cbar.set_label("probability [%]")
# plt.savefig("timer_plots/t2_heatmaps.png")

# # theta 3 plots
# fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))
# fig.delaxes(axs[1,2])
# axslist = axs.flat
# # t3 p1
# ax = axslist[0]
# test = df.pivot_table(index='phi1', columns='theta3', values='probability',aggfunc=np.sum)
# added_probability = test.to_numpy()/np.sum(test.to_numpy())
# out = ax.pcolormesh(theta3s, phi1s, added_probability * 100, edgecolors='k', linewidth=0.0, cmap = 'jet', shading = "nearest", norm=colors.SymLogNorm(vmin=0.0, vmax=20, linthresh=0.01))
# ax.set_title("$\\theta_3 vs. \phi_1$")
# # t3 p2
# ax = axslist[1]
# test = df.pivot_table(index='phi2', columns='theta3', values='probability',aggfunc=np.sum)
# added_probability = test.to_numpy()/np.sum(test.to_numpy())
# out = ax.pcolormesh(theta3s, phi2s, added_probability * 100, edgecolors='k', linewidth=0.0, cmap = 'jet', shading = "nearest", norm=colors.SymLogNorm(vmin=0.0, vmax=20, linthresh=0.01))
# ax.set_title("$\\theta_3 vs. \phi_2$")
# # t3 p3
# ax = axslist[2]
# test = df.pivot_table(index='phi3', columns='theta3', values='probability',aggfunc=np.sum)
# added_probability = test.to_numpy()/np.sum(test.to_numpy())
# out = ax.pcolormesh(theta3s, phi3s, added_probability * 100, edgecolors='k', linewidth=0.0, cmap = 'jet', shading = "nearest", norm=colors.SymLogNorm(vmin=0.0, vmax=20, linthresh=0.01))
# ax.set_title("$\\theta_3 vs. \phi_3$")
# # t3 t1
# ax = axslist[3]
# test = df.pivot_table(index='theta1', columns='theta3', values='probability',aggfunc=np.sum)
# added_probability = test.to_numpy()/np.sum(test.to_numpy())
# out = ax.pcolormesh(theta3s, theta1s, added_probability * 100, edgecolors='k', linewidth=0.0, cmap = 'jet', shading = "nearest", norm=colors.SymLogNorm(vmin=0.0, vmax=20, linthresh=0.01))
# ax.set_title("$\\theta_3 vs. \\theta_1$")
# # t3 t2
# ax = axslist[4]
# test = df.pivot_table(index='theta2', columns='theta3', values='probability',aggfunc=np.sum)
# added_probability = test.to_numpy()/np.sum(test.to_numpy())
# out = ax.pcolormesh(theta3s, theta2s, added_probability * 100, edgecolors='k', linewidth=0.0, cmap = 'jet', shading = "nearest", norm=colors.SymLogNorm(vmin=0.0, vmax=20, linthresh=0.01))
# ax.set_title("$\\theta_3 vs. \\theta_2$")

# axs.flat[-1].axis('off')
# plt.tight_layout()
# fig.subplots_adjust(right=0.875)
# cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
# cbar = fig.colorbar(out, cax=cbar_ax, extend='max')
# cbar.set_label("probability [%]")
# plt.savefig("timer_plots/t3_heatmaps.png")


# # phi 1 plots
# fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))
# fig.delaxes(axs[1,2])
# axslist = axs.flat
# # p1 t1
# ax = axslist[0]
# test = df.pivot_table(index='theta1', columns='phi1', values='probability',aggfunc=np.sum)
# added_probability = test.to_numpy()/np.sum(test.to_numpy())
# out = ax.pcolormesh(phi1s, theta1s, added_probability * 100, edgecolors='k', linewidth=0.0, cmap = 'jet', shading = "nearest", norm=colors.SymLogNorm(vmin=0.0, vmax=20, linthresh=0.01))
# ax.set_title("$\phi_1 vs. \\theta_1$")
# # p1 t2
# ax = axslist[1]
# test = df.pivot_table(index='theta2', columns='phi1', values='probability',aggfunc=np.sum)
# added_probability = test.to_numpy()/np.sum(test.to_numpy())
# out = ax.pcolormesh(phi1s, theta2s, added_probability * 100, edgecolors='k', linewidth=0.0, cmap = 'jet', shading = "nearest", norm=colors.SymLogNorm(vmin=0.0, vmax=20, linthresh=0.01))
# ax.set_title("$\phi_1 vs. \\theta_2$")
# # p1 t3
# ax = axslist[2]
# test = df.pivot_table(index='theta3', columns='phi1', values='probability',aggfunc=np.sum)
# added_probability = test.to_numpy()/np.sum(test.to_numpy())
# out = ax.pcolormesh(phi1s, theta3s, added_probability * 100, edgecolors='k', linewidth=0.0, cmap = 'jet', shading = "nearest", norm=colors.SymLogNorm(vmin=0.0, vmax=20, linthresh=0.01))
# ax.set_title("$\phi_1 vs. \\theta_3$")
# # p1 p2
# ax = axslist[3]
# test = df.pivot_table(index='phi2', columns='phi1', values='probability',aggfunc=np.sum)
# added_probability = test.to_numpy()/np.sum(test.to_numpy())
# out = ax.pcolormesh(phi1s, phi2s, added_probability * 100, edgecolors='k', linewidth=0.0, cmap = 'jet', shading = "nearest", norm=colors.SymLogNorm(vmin=0.0, vmax=20, linthresh=0.01))
# ax.set_title("$\phi_1 vs. \phi_2$")
# # p1 p3
# ax = axslist[4]
# test = df.pivot_table(index='phi3', columns='phi1', values='probability',aggfunc=np.sum)
# added_probability = test.to_numpy()/np.sum(test.to_numpy())
# out = ax.pcolormesh(phi1s, phi3s, added_probability * 100, edgecolors='k', linewidth=0.0, cmap = 'jet', shading = "nearest", norm=colors.SymLogNorm(vmin=0.0, vmax=20, linthresh=0.01))
# ax.set_title("$\phi_1 vs. \phi_3$")

# axs.flat[-1].axis('off')
# plt.tight_layout()
# fig.subplots_adjust(right=0.875)
# cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
# cbar = fig.colorbar(out, cax=cbar_ax, extend='max')
# cbar.set_label("probability [%]")
# plt.savefig("timer_plots/p1_heatmaps.png")

# # phi 2 plots
# fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))
# fig.delaxes(axs[1,2])
# axslist = axs.flat
# # p2 t1
# ax = axslist[0]
# test = df.pivot_table(index='theta1', columns='phi2', values='probability',aggfunc=np.sum)
# added_probability = test.to_numpy()/np.sum(test.to_numpy())
# out = ax.pcolormesh(phi2s, theta1s, added_probability * 100, edgecolors='k', linewidth=0.0, cmap = 'jet', shading = "nearest", norm=colors.SymLogNorm(vmin=0.0, vmax=20, linthresh=0.01))
# ax.set_title("$\phi_2 vs. \\theta_1$")
# # p2 t2
# ax = axslist[1]
# test = df.pivot_table(index='theta2', columns='phi2', values='probability',aggfunc=np.sum)
# added_probability = test.to_numpy()/np.sum(test.to_numpy())
# out = ax.pcolormesh(phi2s, theta2s, added_probability * 100, edgecolors='k', linewidth=0.0, cmap = 'jet', shading = "nearest", norm=colors.SymLogNorm(vmin=0.0, vmax=20, linthresh=0.01))
# ax.set_title("$\phi_2 vs. \\theta_2$")
# # p2 t3
# ax = axslist[2]
# test = df.pivot_table(index='theta3', columns='phi2', values='probability',aggfunc=np.sum)
# added_probability = test.to_numpy()/np.sum(test.to_numpy())
# out = ax.pcolormesh(phi2s, theta3s, added_probability * 100, edgecolors='k', linewidth=0.0, cmap = 'jet', shading = "nearest", norm=colors.SymLogNorm(vmin=0.0, vmax=20, linthresh=0.01))
# ax.set_title("$\phi_2 vs. \\theta_3$")
# # p2 p1
# ax = axslist[3]
# test = df.pivot_table(index='phi1', columns='phi2', values='probability',aggfunc=np.sum)
# added_probability = test.to_numpy()/np.sum(test.to_numpy())
# out = ax.pcolormesh(phi2s, phi1s, added_probability * 100, edgecolors='k', linewidth=0.0, cmap = 'jet', shading = "nearest", norm=colors.SymLogNorm(vmin=0.0, vmax=20, linthresh=0.01))
# ax.set_title("$\phi_2 vs. \phi_1$")
# # p2 p3
# ax = axslist[4]
# test = df.pivot_table(index='phi3', columns='phi2', values='probability',aggfunc=np.sum)
# added_probability = test.to_numpy()/np.sum(test.to_numpy())
# out = ax.pcolormesh(phi2s, phi3s, added_probability * 100, edgecolors='k', linewidth=0.0, cmap = 'jet', shading = "nearest", norm=colors.SymLogNorm(vmin=0.0, vmax=20, linthresh=0.01))
# ax.set_title("$\phi_2 vs. \phi_3$")

# axs.flat[-1].axis('off')
# plt.tight_layout()
# fig.subplots_adjust(right=0.875)
# cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
# cbar = fig.colorbar(out, cax=cbar_ax, extend='max')
# cbar.set_label("probability [%]")
# plt.savefig("timer_plots/p2_heatmaps.png")

# # phi 3 plots
# fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))
# fig.delaxes(axs[1,2])
# axslist = axs.flat
# # p3 t1
# ax = axslist[0]
# test = df.pivot_table(index='theta1', columns='phi3', values='probability',aggfunc=np.sum)
# added_probability = test.to_numpy()/np.sum(test.to_numpy())
# out = ax.pcolormesh(phi3s, theta1s, added_probability * 100, edgecolors='k', linewidth=0.0, cmap = 'jet', shading = "nearest", norm=colors.SymLogNorm(vmin=0.0, vmax=20, linthresh=0.01))
# ax.set_title("$\phi_3 vs. \\theta_1$")
# # p3 t2
# ax = axslist[1]
# test = df.pivot_table(index='theta2', columns='phi3', values='probability',aggfunc=np.sum)
# added_probability = test.to_numpy()/np.sum(test.to_numpy())
# out = ax.pcolormesh(phi3s, theta2s, added_probability * 100, edgecolors='k', linewidth=0.0, cmap = 'jet', shading = "nearest", norm=colors.SymLogNorm(vmin=0.0, vmax=20, linthresh=0.01))
# ax.set_title("$\phi_3 vs. \\theta_2$")
# # p3 t3
# ax = axslist[2]
# test = df.pivot_table(index='theta3', columns='phi3', values='probability',aggfunc=np.sum)
# added_probability = test.to_numpy()/np.sum(test.to_numpy())
# out = ax.pcolormesh(phi3s, theta3s, added_probability * 100, edgecolors='k', linewidth=0.0, cmap = 'jet', shading = "nearest", norm=colors.SymLogNorm(vmin=0.0, vmax=20, linthresh=0.01))
# ax.set_title("$\phi_3 vs. \\theta_3$")
# # p3 p1
# ax = axslist[3]
# test = df.pivot_table(index='phi1', columns='phi3', values='probability',aggfunc=np.sum)
# added_probability = test.to_numpy()/np.sum(test.to_numpy())
# out = ax.pcolormesh(phi3s, phi1s, added_probability * 100, edgecolors='k', linewidth=0.0, cmap = 'jet', shading = "nearest", norm=colors.SymLogNorm(vmin=0.0, vmax=20, linthresh=0.01))
# ax.set_title("$\phi_3 vs. \phi_1$")
# # p3 p2
# ax = axslist[4]
# test = df.pivot_table(index='phi2', columns='phi3', values='probability',aggfunc=np.sum)
# added_probability = test.to_numpy()/np.sum(test.to_numpy())
# out = ax.pcolormesh(phi3s, phi2s, added_probability * 100, edgecolors='k', linewidth=0.0, cmap = 'jet', shading = "nearest", norm=colors.SymLogNorm(vmin=0.0, vmax=20, linthresh=0.01))
# ax.set_title("$\phi_3 vs. \phi_2$")

# axs.flat[-1].axis('off')
# plt.tight_layout()
# fig.subplots_adjust(right=0.875)
# cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
# cbar = fig.colorbar(out, cax=cbar_ax, extend='max')
# cbar.set_label("probability [%]")
# plt.savefig("timer_plots/p3_heatmaps.png")