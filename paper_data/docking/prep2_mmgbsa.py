#!/usr/bin/env python
"""Build REC/LIG prmtops consistent with COM_full, convert aligned xyz to mdcrd, run MMPBSA.py"""
import warnings
warnings.filterwarnings('ignore')
import subprocess, os

BASE = '/root/case3_md'
os.chdir(BASE)

# ---- 1) REC 和 LIG 的 prmtop: 用与 COM 相同流程 ----
# REC: rec_heavy.pdb -> tleap
leap_rec = '''source leaprc.protein.ff14SB
REC = loadpdb rec_heavy.pdb
saveamberparm REC REC_full.prmtop REC_full.inpcrd
quit
'''
open('leap_rec.in','w').write(leap_rec)
subprocess.run('tleap -f leap_rec.in > leap_rec.log 2>&1', shell=True)

# LIG: mol2 -> tleap
leap_lig = '''source leaprc.gaff2
loadamberparams lig.frcmod
LIG = loadmol2 lig_ac.mol2
saveamberparm LIG LIG_full.prmtop LIG_full.inpcrd
quit
'''
open('leap_lig.in','w').write(leap_lig)
subprocess.run('tleap -f leap_lig.in > leap_lig.log 2>&1', shell=True)

import parmed as pmd
for f in ('REC_full.prmtop','LIG_full.prmtop'):
    p = pmd.load_file(f)
    print(f, len(p.atoms), 'atoms')

# ---- 2) aligned xyz -> mdcrd via pytraj ----
import numpy as np
import pytraj as pt
xyz = np.load(f'{BASE}/aligned_xyz.npy')  # (200, 4657, 3) in nm (mdtraj)
# pytraj: 构造 Trajectory, 单位 Angstrom
tr = pt.Trajectory(xyz=(xyz.astype(np.float64)*10.0), top=f'{BASE}/COM_full.prmtop')
pt.write_traj(f'{BASE}/dry_amber.mdcrd', tr)
print('mdcrd written, frames:', pt.iterload(f'{BASE}/dry_amber.mdcrd', f'{BASE}/COM_full.prmtop').n_frames)
print('PREP2_DONE')
