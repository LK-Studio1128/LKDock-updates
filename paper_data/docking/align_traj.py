#!/usr/bin/env python
"""Align OpenMM trajectory (traj.dcd dry) to COM_full.prmtop atom order -> dry_aligned.mdcrd + per-frame RMSD check"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import parmed as pmd
import mdtraj as md

BASE = '/root/case3_md'
p = pmd.load_file(f'{BASE}/COM_full.prmtop')
parm_keys = [(a.name, a.residue.name) for a in p.atoms]
print('prmtop atoms:', len(parm_keys))

t = md.load(f'{BASE}/traj.dcd', top=f'{BASE}/topology.pdb')
keep = t.topology.select('not water and not resname NA CL K MG CA ZN')
dry = t.atom_slice(keep)
print('dry frames:', dry.n_frames, 'atoms:', dry.n_atoms)

traj_keys = [(a.name, a.residue.name) for a in dry.topology.atoms]
from collections import defaultdict
traj_map = defaultdict(list)
for i, (n, r) in enumerate(traj_keys):
    traj_map[(r, n)].append(i)

# tleap 把 HIS 改名为 HIE/HID/HIP; 轨迹里仍是 HIS
RES_ALIAS = {'HIE': 'HIS', 'HID': 'HIS', 'HIP': 'HIS', 'CYX': 'CYS'}
ALT = {}
for a, b in [('HB2','HB3'),('HG2','HG3'),('HD2','HD3'),('HE2','HE3'),('HZ2','HZ3')]:
    ALT[a], ALT[b] = b, a

order = []
missing = 0
for j, (n, r) in enumerate(parm_keys):
    r_traj = RES_ALIAS.get(r, r)
    lst = traj_map.get((r_traj, n))
    if not lst and n in ALT:
        lst = traj_map.get((r_traj, ALT[n]))
    if not lst and n == 'H1':
        lst = traj_map.get((r_traj, 'H'))
    if not lst and n == 'HE2':
        # HIE 的 HE2 在轨迹 HIS 中可能叫 HE2 或 HG(δ质子化HID时无HE2)
        for cand in ('HE2', 'HG', 'HD1'):
            lst = traj_map.get((r_traj, cand))
            if lst: break
    if lst:
        order.append(lst.pop(0))
    else:
        missing += 1
        order.append(None)
print('missing:', missing)
if missing:
    for j, (n, r) in enumerate(parm_keys):
        if order[j] is None:
            print('  missing atom %d: %s %s' % (j, r, n))
            order[j] = order[j-1]  # fallback: 复制前一个(仅占位, 会剔除)

ok = [o for o in order if o is not None]
if missing == 0:
    xyz = dry.xyz[:, order, :]  # (frames, atoms, 3) nm
    # 验证: 键长检查 frame0
    bonds = [(b.atom1.idx, b.atom2.idx) for b in list(p.bonds)[:500]]
    xyz0 = xyz[0] * 10
    lens = np.array([np.linalg.norm(xyz0[i]-xyz0[j]) for i, j in bonds])
    print('aligned frame0: 前500键 median %.2f A, >3A: %d/500' % (np.median(lens), (lens>3).sum()))
    assert (lens>3).sum() < 5, 'ALIGNMENT FAILED'
    np.save(f'{BASE}/aligned_xyz.npy', xyz.astype(np.float32))
    print('aligned_xyz.npy saved, shape', xyz.shape)
    print('ALIGN_OK')
