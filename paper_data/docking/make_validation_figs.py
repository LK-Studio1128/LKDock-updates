#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成论文验证图：Fig11 引擎耗时 / Fig12 RMSD / Fig13 口袋预测（英文 SCI 300dpi）"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np, os

_FONT = '/System/Library/Fonts/Supplemental/Arial.ttf'
if os.path.exists(_FONT):
    fm.fontManager.addfont(_FONT)
    plt.rcParams['font.family'] = fm.FontProperties(fname=_FONT).get_name()
plt.rcParams['axes.unicode_minus'] = False

OUT = '/Users/luoxiaowen/Downloads/LKDock软件介绍论文/01_论文图表'
BLUE, TEAL, AMBER = '#185FA5', '#0F6E56', '#B2571B'

def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved', name)

# ---- Fig 11 · Engine timing (log scale) ----
fig, ax = plt.subplots(figsize=(5.2, 3.6))
labels = ['3PTB', '1HVR']
v1 = [6.2, np.nan]
v4 = [2.5, 119.5]
ud = [4.0, np.nan]
x = np.arange(2); w = 0.26
b1 = ax.bar(x - w, v1, w, label='Vina 1-thread', color='#B5D4F4', edgecolor=BLUE, lw=0.6)
b2 = ax.bar(x, v4, w, label='Vina 4-thread', color=BLUE, edgecolor=BLUE, lw=0.6)
b3 = ax.bar(x + w, ud, w, label='Uni-Dock CPU 4-thread', color=TEAL, edgecolor=TEAL, lw=0.6)
ax.set_yscale('log')
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel('Time (s, log scale)')
ax.set_title('Engine timing (exhaustiveness 32, 25 Å box)')
for bars in (b1, b2, b3):
    for r in bars:
        h = r.get_height()
        if not np.isnan(h):
            ax.text(r.get_x() + r.get_width()/2, h*1.08, f'{h:.1f}', ha='center', va='bottom', fontsize=7.5)
ax.legend(fontsize=7, frameon=False)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
save(fig, 'Fig11_engine_timing.png')

# ---- Fig 12 · Docking RMSD ----
fig, ax = plt.subplots(figsize=(5.2, 3.6))
sys_lab = ['3PTB\n(Vina)', '3PTB\n(Uni-Dock)', '1HVR\n(Vina)']
rmsd = [1.5, 1.5, 5.85]
cols = [BLUE, TEAL, BLUE]
bars = ax.bar(sys_lab, rmsd, 0.5, color=cols, edgecolor=cols, lw=0.6)
ax.axhline(2.0, color=AMBER, ls='--', lw=1.0)
ax.text(0.35, 2.05, '2 Å criterion', color=AMBER, fontsize=7.5, ha='left')
ax.set_ylabel('Heavy-atom RMSD (Å)')
ax.set_ylim(0, 6.6)
ax.set_title('Docking accuracy (top-scored pose, all heavy atoms)')
for r, v in zip(bars, rmsd):
    ax.text(r.get_x() + r.get_width()/2, v + 0.12, f'{v:.2f}', ha='center', fontsize=8)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
save(fig, 'Fig12_docking_rmsd.png')

# ---- Fig 13 · Pocket prediction (fpocket, 3PTB) ----
fig, ax = plt.subplots(figsize=(5.2, 3.6))
pockets = ['P1', 'P6', 'P3', 'P9', 'P7', 'P4', 'P2', 'P5', 'P8', 'P10']
dists = [0.39, 9.91, 12.10, 13.47, 14.42, 15.77, 16.10, 16.55, 18.40, 19.20]
cols2 = [AMBER if i == 0 else '#B5D4F4' for i in range(len(pockets))]
bars = ax.bar(pockets, dists, 0.6, color=cols2, edgecolor='#888780', lw=0.4)
ax.axhline(8.0, color=TEAL, ls='--', lw=1.0)
ax.text(9.3, 8.3, '8 Å criterion', color=TEAL, fontsize=7.5, ha='right')
ax.set_ylabel('Pocket centroid – ligand-site distance (Å)')
ax.set_xlabel('fpocket-predicted pocket (3PTB)')
ax.set_title('Pocket prediction: top-1 pocket recovers the benzamidine site (0.39 Å)')
for r, v in zip(bars, dists):
    ax.text(r.get_x() + r.get_width()/2, v + 0.3, f'{v:.1f}', ha='center', fontsize=6.5)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
save(fig, 'Fig13_pocket_prediction.png')
