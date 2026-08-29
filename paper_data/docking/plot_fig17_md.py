#!/usr/bin/env python
"""Fig17: 案例3 MD 轨迹稳定性 (Mpro-5FU, 1ns 显式水, RTX 2080)"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'Arial', 'font.size': 9, 'axes.linewidth': 0.8})
BASE = '/Users/luoxiaowen/Desktop/LKDock/LKDock软件介绍论文'
d = np.loadtxt(BASE + '/04_测试结果/case3_md/rmsd_series.txt', skiprows=1)
t, prot, lig = d[:, 0]/1000.0, d[:, 1], d[:, 2]   # ps -> ns

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), dpi=300)

ax = axes[0]
ax.plot(t, prot, color='#1f77b4', lw=1.0)
ax.set_xlabel('Time (ns)'); ax.set_ylabel('Protein C$\\alpha$ RMSD (Å)')
ax.set_title('(a) Mpro backbone stability', fontsize=9)
ax.set_xlim(0, 1.0); ax.set_ylim(bottom=0)

ax = axes[1]
ax.plot(t, lig, color='#d62728', lw=1.0)
# 平台期均值线(后500ps)
m = t >= 0.5
ax.axhline(lig[m].mean(), color='gray', ls='--', lw=0.8)
ax.text(0.98, lig[m].mean()+0.12, 'mean %.2f Å (t ≥ 0.5 ns)' % lig[m].mean(),
        ha='right', fontsize=8, color='gray', transform=ax.get_yaxis_transform())
ax.set_xlabel('Time (ns)'); ax.set_ylabel('5-FU ligand RMSD (Å)')
ax.set_title('(b) Ligand stability in pocket', fontsize=9)
ax.set_xlim(0, 1.0); ax.set_ylim(bottom=0)

for ax in axes:
    ax.tick_params(direction='in', top=True, right=True, width=0.8)

fig.tight_layout()
out = BASE + '/01_论文图表/Fig17_MD_stability.png'
fig.savefig(out, dpi=300, bbox_inches='tight')
print('saved:', out)
print('prot mean=%.2f final=%.2f | lig mean(>=0.5ns)=%.2f' % (prot.mean(), prot[-1], lig[m].mean()))
