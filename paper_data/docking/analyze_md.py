#!/usr/bin/env python
"""案例3 RMSD分析 (独立脚本): DCD轨迹 -> Kabsch RMSD"""
import os, warnings
warnings.filterwarnings('ignore')
import numpy as np
from openmm import unit
from openmm.app import PDBFile, DCDFile

BASE = '/root/case3_md'
topo = PDBFile(os.path.join(BASE, 'topology.pdb')).topology
ca_idx = np.array([a.index for a in topo.atoms() if a.name == 'CA' and a.residue.name != 'MOL'])
li_idx = np.array([a.index for a in topo.atoms()
                   if a.residue.name == 'MOL' and not (a.element is None) and a.element.symbol != 'H'])
print('CA:', len(ca_idx), '配体重原子:', len(li_idx), flush=True)

import mdtraj as md
traj = md.load(os.path.join(BASE, 'traj.dcd'), top=os.path.join(BASE, 'topology.pdb'))
print('帧数:', traj.n_frames, flush=True)
ref = traj[0]
ca_sel = traj.topology.select('name CA and not resname MOL')
li_sel = traj.topology.select('resname MOL and element != H')
prot_rmsd = md.rmsd(traj, ref, atom_indices=ca_sel) * 10.0  # Å
lig_rmsd = []
for i in range(traj.n_frames):
    fr = traj[i].superpose(ref, atom_indices=ca_sel)
    lig_rmsd.append(np.sqrt(np.mean(np.sum((fr.xyz[0][li_sel]-ref.xyz[0][li_sel])**2, axis=1)))*10.0)
lig_rmsd = np.array(lig_rmsd)
print('蛋白Cα RMSD(Å): mean %.2f | final %.2f | max %.2f' % (prot_rmsd.mean(), prot_rmsd[-1], prot_rmsd.max()), flush=True)
print('配体 RMSD(Å): mean %.2f | final %.2f | max %.2f' % (lig_rmsd.mean(), lig_rmsd[-1], lig_rmsd.max()), flush=True)

np.save(os.path.join(BASE, 'prot_rmsd.npy'), prot_rmsd)
np.save(os.path.join(BASE, 'lig_rmsd.npy'), lig_rmsd)
# RMSD时间序列文本
t_ps = np.arange(traj.n_frames) * 25.0
np.savetxt(os.path.join(BASE, 'rmsd_series.txt'),
           np.column_stack([t_ps, prot_rmsd, lig_rmsd]),
           header='time_ps\tprot_CA_rmsd_A\tlig_rmsd_A', fmt='%.2f')
with open(os.path.join(BASE, 'summary.txt'), 'w') as fo:
    fo.write('frames=%d\ninterval_ps=25\nduration_ns=1.0\n'
             'prot_ca_rmsd_A_mean=%.2f\nprot_ca_rmsd_A_final=%.2f\nprot_ca_rmsd_A_max=%.2f\n'
             'lig_rmsd_A_mean=%.2f\nlig_rmsd_A_final=%.2f\nlig_rmsd_A_max=%.2f\n'
             'gpu=RTX2080\nprod_time_s=534\nspeed_ns_per_day=%.1f\n' %
             (traj.n_frames, prot_rmsd.mean(), prot_rmsd[-1], prot_rmsd.max(),
              lig_rmsd.mean(), lig_rmsd[-1], lig_rmsd.max(),
              (1000/534)*86400/1000))
print('ANALYSIS_DONE', flush=True)
