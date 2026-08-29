#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig 18: 补充测试验证图（4 面板）
(a) 打分函数一致性 vina vs vinardo  (b) 引擎版本一致性 1.2.7 vs 1.1.2
(c) 筛选排序稳定性 (Spearman rho=1.0)  (d) 盲对接 vs 定向对接 RMSD
SCI 规范：全英文、Arial、300 dpi、无中文
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['font.size'] = 9

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, '..', '01_论文图表')
os.makedirs(OUT, exist_ok=True)

res = json.load(open(os.path.join(BASE, 'extra_tests_results.json')))

BLUE, ORANGE, GREEN, GRAY = '#185FA5', '#D85A30', '#1D9E75', '#888780'

fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))

# (a) scoring functions
ax = axes[0, 0]
sf = res['test9_scoring']
names = ['vina', 'vinardo']
scores = [sf['vina']['score'], sf['vinardo']['score']]
rmsds = [sf['vina']['rmsd_a'], sf['vinardo']['rmsd_a']]
bars = ax.bar(names, scores, width=0.55, color=[BLUE, ORANGE], edgecolor='black', linewidth=0.5)
ax.axhline(0, color='black', linewidth=0.6)
for b, s, r in zip(bars, scores, rmsds):
    ax.text(b.get_x() + b.get_width()/2, s - 0.25, f'{s:.2f}\nRMSD {r:.2f} A',
            ha='center', va='top', fontsize=8)
ax.set_ylabel('Top-1 score (kcal/mol)')
ax.set_title('(a) Scoring functions', fontsize=9, fontweight='bold')
ax.set_ylim(-7.5, 0)

# (b) engine versions
ax = axes[0, 1]
ev = res['test10_engine_versions']
names = ['Vina 1.2.7\n(LKDock)', 'Vina 1.1.2\n(upstream)']
scores = [ev['v127']['score'], ev['v112']['score']]
rmsds = [ev['v127']['rmsd_a'], ev['v112']['rmsd_a']]
bars = ax.bar(names, scores, width=0.55, color=[BLUE, GREEN], edgecolor='black', linewidth=0.5)
ax.axhline(0, color='black', linewidth=0.6)
for b, s, r in zip(bars, scores, rmsds):
    ax.text(b.get_x() + b.get_width()/2, s - 0.25, f'{s:.2f}\nRMSD {r:.2f} A',
            ha='center', va='top', fontsize=8)
ax.set_ylabel('Top-1 score (kcal/mol)')
ax.set_title('(b) Engine versions', fontsize=9, fontweight='bold')
ax.set_ylim(-7.5, 0)

# (c) screening ranking stability
ax = axes[1, 0]
st = res['test11_screening_stability']
s8 = st['exhaust8']['scores']
s16 = st['exhaust16']['scores']
ligs = list(s8.keys())
v8 = [s8[k] for k in ligs]
v16 = [s16[k] for k in ligs]
ax.scatter(v8, v16, s=30, color=BLUE, edgecolor='black', linewidth=0.5, zorder=3)
lim = [min(min(v8), min(v16)) - 0.3, max(max(v8), max(v16)) + 0.3]
ax.plot(lim, lim, 'k--', linewidth=0.8)
ax.set_xlabel('Score at exhaustiveness 8 (kcal/mol)')
ax.set_ylabel('Score at exhaustiveness 16 (kcal/mol)')
ax.set_title(f'(c) Ranking stability  (Spearman rho = {st["spearman_rho"]:.2f})',
             fontsize=9, fontweight='bold', pad=2)
ax.set_xlim(lim); ax.set_ylim(lim)
texts = []
for k in ligs:
    t = ax.annotate(k.replace('_', ' ').replace('-', ' '), (s8[k], s16[k]),
                fontsize=6.5, ha='left', va='bottom', xytext=(0, 0),
                textcoords='offset points')
    texts.append(t)
from adjustText import adjust_text
adjust_text(texts, ax=ax, only_move={'points':'xy', 'text':'xy'},
           arrowprops=dict(arrowstyle='-', color='gray', lw=0.4, shrinkA=0, shrinkB=2),
           maxmove=15, force_text=(0.02, 0.15, 0.98, 0.82))

# (d) blind vs targeted
ax = axes[1, 1]
bd = res['test12_blind']
names = ['Blind\n(whole-protein box)', 'Targeted\n(25 A box)']
rmsds = [bd['blind']['rmsd_a'], bd['targeted']['rmsd_a']]
scores = [bd['blind']['score'], bd['targeted']['score']]
bars = ax.bar(names, rmsds, width=0.5, color=[ORANGE, GREEN], edgecolor='black', linewidth=0.5)
ax.axhline(2.0, color='gray', linestyle=':', linewidth=0.8)
ax.text(1.42, 2.02, '2 A success\ncriterion', fontsize=7, color='gray', ha='right', va='bottom')
for b, r, s in zip(bars, rmsds, scores):
    ax.text(b.get_x() + b.get_width()/2, r + 0.06, f'{r:.2f} A\n({s:.2f} kcal/mol)',
            ha='center', va='bottom', fontsize=8)
ax.set_ylabel('Heavy-atom RMSD (A)')
ax.set_title('(d) Blind vs targeted docking', fontsize=9, fontweight='bold')
ax.set_ylim(0, 1.6)

fig.tight_layout(pad=1.2)
fig.savefig(os.path.join(OUT, 'Fig18_extra_validation.png'), dpi=300,
            bbox_inches='tight', facecolor='white')
print('saved Fig18_extra_validation.png')
