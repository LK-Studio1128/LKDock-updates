#!/usr/bin/env python
"""Prepare dry complex trajectory for MM-GBSA:
1) strip water+ions from traj.dcd -> dry.nc (mdtraj)
2) write complex.pdb (dry, frame0) + receptor.pdb + ligand.pdb (ligand kept with H)
"""
import warnings
warnings.filterwarnings('ignore')
import mdtraj as md
import numpy as np

BASE = '/root/case3_md'
traj = md.load(f'{BASE}/traj.dcd', top=f'{BASE}/topology.pdb')
print('loaded frames:', traj.n_frames, 'atoms:', traj.n_atoms)

# 干燥: 去水与离子 (保留蛋白+MOL)
keep = traj.topology.select('not water and not resname NA CL K MG CA ZN')
dry = traj.atom_slice(keep)
print('dry atoms:', dry.n_atoms, 'frames:', dry.n_frames)

# 抽稀: 后 1ns 中每 50ps 一帧 -> 20帧 (GBSA 计算量适中)
sel = list(range(0, dry.n_frames, 5))  # every 25ps*5=125? frames are 25ps apart -> every5 =125ps
# 实际帧间隔25ps, 40帧; 取每2帧=50ps, 共20帧
sel = list(range(0, dry.n_frames, 2))[1:]  # skip t=0 (起点偏差)
dry2 = dry.slice(sel)
print('selected frames:', dry2.n_frames)
dry2.save_amberrst7(f'{BASE}/dry.rst7')  # 仅最后帧坐标用途
dry2.save_netcdf(f'{BASE}/dry.nc')
dry2[0].save_pdb(f'{BASE}/complex_dry.pdb')

# 受体/配体拆分
lig = dry2.atom_slice(dry2.topology.select('resname MOL'))
rec = dry2.atom_slice(dry2.topology.select('not resname MOL'))
lig[0].save_pdb(f'{BASE}/lig_dry.pdb')
rec[0].save_pdb(f'{BASE}/rec_dry.pdb')
print('lig atoms:', lig.n_atoms, 'rec atoms:', rec.n_atoms)
print('PREP_DONE')
