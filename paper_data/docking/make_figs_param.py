#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LKDock 论文 · 新增测试可视化（Fig 26-27）
Fig 26: 参数稳健性（exhaustiveness / 盒尺寸 / 重复性）
Fig 27: 评分函数对比 + 新案例（病毒/蛋白位点）
600 DPI 规范化
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

BASE = '/Users/luoxiaowen/Downloads/LKDock软件介绍论文/04_测试结果'
FIG = '/Users/luoxiaowen/Downloads/LKDock软件介绍论文/01_论文图表'
R = json.load(open(os.path.join(BASE, 'test_extra/param_scan_results.json')))
NC = json.load(open(os.path.join(BASE, 'test_extra/new_cases_results.json')))

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 8.5, 'axes.titlesize': 9,
    'legend.fontsize': 7.5, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'axes.linewidth': 0.7, 'lines.linewidth': 1.0,
    'savefig.dpi': 600, 'savefig.bbox': None,
    'axes.spines.top': False, 'axes.spines.right': False,
})
C_RED, C_GREEN, C_BLUE, C_PURPLE, C_ORANGE, C_GRAY = '#C62828', '#2E7D32', '#1565C0', '#6A1B9A', '#EF6C00', '#757575'

# ================= Fig 26 参数稳健性 =================
fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.4))

# (A) exhaustiveness vs score
ax = axes[0]
exs = [4, 8, 16, 32]
s3 = [R['exhaustiveness'][f'3PTB_ex{e}']['mean_score'] for e in exs]
sd3 = [R['exhaustiveness'][f'3PTB_ex{e}']['std_score'] for e in exs]
ax.plot(exs, s3, 'o-', color=C_BLUE, label='3PTB (trypsin)')
ax.fill_between(exs, np.array(s3)-np.array(sd3), np.array(s3)+np.array(sd3), color=C_BLUE, alpha=0.15)
ax.plot([8, 16], [R['exhaustiveness']['1HVR_ex8']['mean_score'], R['exhaustiveness']['1HVR_ex16']['mean_score']],
        's--', color=C_RED, label='1HVR (HIV-1 PR)')
ax.set_xlabel('Exhaustiveness'); ax.set_ylabel('Top-1 score (kcal/mol)')
ax.set_title('(A) Score vs. search effort', fontweight='bold')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.legend(loc='lower right', frameon=False)
ax.grid(alpha=0.25, lw=0.4)

# (B) exhaustiveness vs RMSD
ax = axes[1]
r3 = [R['exhaustiveness'][f'3PTB_ex{e}']['mean_rmsd'] for e in exs]
ax.plot(exs, r3, 'o-', color=C_BLUE, label='3PTB')
ax.plot([8, 16], [R['exhaustiveness']['1HVR_ex8']['mean_rmsd'], R['exhaustiveness']['1HVR_ex16']['mean_rmsd']],
        's--', color=C_RED, label='1HVR')
ax.axhline(2.0, ls=':', lw=0.7, color=C_GRAY)
ax.text(4.3, 2.05, '2 Å criterion', fontsize=6.5, color=C_GRAY)
ax.set_xlabel('Exhaustiveness'); ax.set_ylabel('Top-1 heavy-atom RMSD (Å)')
ax.set_title('(B) RMSD vs. search effort', fontweight='bold')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_ylim(0, 6.5)
ax.legend(loc='upper right', frameon=False)
ax.grid(alpha=0.25, lw=0.4)

# (C) 盒尺寸 RMSD（双 y 轴：RMSD 左 + score 副轴）
ax = axes[2]
boxes = [20, 25, 30]
rb = [R['boxsize'][f'box{b}']['rmsd'] for b in boxes]
sb_ = [R['boxsize'][f'box{b}']['score'] for b in boxes]
ax.plot(boxes, rb, 'o-', color=C_GREEN, label='RMSD', linewidth=1.2)
ax.axhline(2.0, ls=':', lw=0.7, color=C_GRAY, alpha=0.7)
ax.set_xlabel('Box size (Å)'); ax.set_ylabel('RMSD (Å)', color=C_GREEN)
ax.tick_params(axis='y', labelcolor=C_GREEN)
ax.set_ylim(0, 2.5); ax.set_xticks(boxes)
ax2 = ax.twinx()
ax2.plot(boxes, sb_, 's--', color=C_ORANGE, label='Score', linewidth=1.2)
ax2.set_ylabel('Score (kcal/mol)', color=C_ORANGE)
ax2.tick_params(axis='y', labelcolor=C_ORANGE)
ax2.set_ylim(-6.5, -5.8)
ax.set_title('(C) Box size & run-to-run', fontweight='bold')
ax.grid(alpha=0.25, lw=0.4, axis='x')
# 重复性（文字标注）
ax.text(0.98, 0.97, f'3PTB ex16 repeatability (n=3):\n'
                  f'  score range = {R["repeatability"]["score_range"]:.3f} kcal/mol\n'
                  f'  RMSD range  = {R["repeatability"]["rmsd_range"]:.2f} Å',
        transform=ax.transAxes, fontsize=6.5, va='top', ha='right', color=C_PURPLE)

plt.tight_layout()
fig.savefig(os.path.join(FIG, 'Fig26_param_robustness.png'))
print('saved Fig26_param_robustness.png')

# ================= Fig 27 评分函数 + 新案例 =================
fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.4))

# (A) vina vs vinardo 打分
ax = axes[0]
sys_names = ['3PTB', '1HVR', '6LU7']
vina_s = [R['scoring'][f'{s}_vina']['score'] for s in sys_names]
vino_s = [R['scoring'][f'{s}_vinardo']['score'] for s in sys_names]
x = np.arange(3); w = 0.35
b1 = ax.bar(x-w/2, vina_s, w, label='vina', color=C_BLUE, edgecolor='black', lw=0.4)
b2 = ax.bar(x+w/2, vino_s, w, label='vinardo', color=C_ORANGE, edgecolor='black', lw=0.4)
ax.set_xticks(x); ax.set_xticklabels(['3PTB\n(trypsin)', '1HVR\n(HIV-1 PR)', '6LU7\n(Mpro)'])
ax.set_ylabel('Top-1 score (kcal/mol)')
ax.set_title('(A) Scoring functions', fontweight='bold')
ax.legend(frameon=False)
ax.grid(alpha=0.25, lw=0.4, axis='y')
for b in list(b1)+list(b2):
    ax.annotate(f'{b.get_height():.2f}', (b.get_x()+b.get_width()/2, b.get_height()),
                ha='center', va='bottom', fontsize=6.5)

# (B) 新案例 RMSD
ax = axes[1]
cases = ['1REV\nHIV-1 RT + TIBO', '4HJO\nEGFR + erlotinib']
rmsds = [NC['1REV_HIV1RT_TIBO']['rmsd_a'], NC['4HJO_EGFR_Erlotinib']['rmsd_a']]
scores = [NC['1REV_HIV1RT_TIBO']['score'], NC['4HJO_EGFR_Erlotinib']['score']]
colors = [C_GREEN, C_ORANGE]
bars = ax.bar(cases, rmsds, 0.5, color=colors, edgecolor='black', lw=0.4)
ax.axhline(2.0, ls=':', lw=0.7, color=C_GRAY)
ax.text(0.35, 2.05, '2 Å criterion', fontsize=6.5, color=C_GRAY)
for b, r, s_ in zip(bars, rmsds, scores):
    ax.annotate(f'{r:.2f} Å\n(score {s_:.2f})', (b.get_x()+b.get_width()/2, b.get_height()),
                ha='center', va='bottom', fontsize=6.5)
ax.set_ylabel('Top-1 heavy-atom RMSD (Å)')
ax.set_title('(B) New viral & kinase cases', fontweight='bold')
ax.set_ylim(0, 6.2)
ax.grid(alpha=0.25, lw=0.4, axis='y')

# (C) 新案例打分 + 成功率标注
ax = axes[2]
ax.bar(cases, scores, 0.5, color=[C_GREEN, C_ORANGE], edgecolor='black', lw=0.4)
for i, (c, s_) in enumerate(zip(cases, scores)):
    ax.annotate(f'{s_:.2f} kcal/mol', (i, s_), ha='center', va='bottom', fontsize=7)
ax.set_ylabel('Top-1 score (kcal/mol)')
ax.set_title('(C) New case scores', fontweight='bold')
ax.grid(alpha=0.25, lw=0.4, axis='y')

plt.tight_layout()
fig.savefig(os.path.join(FIG, 'Fig27_scoring_newcases.png'))
print('saved Fig27_scoring_newcases.png')
