#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LKDock 论文 · 参数扫描测试（测试 13-15）
- exhaustiveness 4/8/16/32 扫描（3PTB ×3 次重复 + 1HVR）
- 评分函数 vina vs vinardo（3PTB / 1HVR / 6LU7）
- 盒尺寸 20/25/30 Å（3PTB）
- 随机重复性（3 次 ex16）
输出：test_extra/param_scan_results.json
"""
import subprocess, json, time, os, re
import numpy as np
from collections import defaultdict

BASE = '/Users/luoxiaowen/Downloads/LKDock软件介绍论文/04_测试结果'
OUT = os.path.join(BASE, 'test_extra')
os.makedirs(OUT, exist_ok=True)
VINA = '/Users/luoxiaowen/Desktop/LKDock_v2.0_Mac/engine/vina127'
OBABEL = '/opt/homebrew/bin/obabel'

def read_atoms_pdbqt(path):
    """PDBQT: 77-79 列是 AutoDock 类型(A/C/N/OA/HD...)，映射为元素；跳过 H 型；只读 MODEL 1"""
    type2elem = {'A': 'C', 'C': 'C', 'N': 'N', 'OA': 'O', 'NA': 'N', 'SA': 'S',
                 'F': 'F', 'CL': 'Cl', 'BR': 'Br', 'I': 'I', 'MG': 'Mg', 'ZN': 'Zn',
                 'P': 'P', 'S': 'S'}
    atoms = []
    for ln in open(path):
        if ln.startswith('ENDMDL'):
            break
        if ln.startswith(('ATOM', 'HETATM')):
            atype = ln[76:79].strip()
            if atype.startswith('H') or atype in ('HD', 'H'):
                continue
            el = type2elem.get(atype)
            if el is None:
                name = ln[12:16].strip()
                el = name[0] if name else 'C'
            try:
                atoms.append((el, float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
            except ValueError:
                continue
    return atoms
def read_atoms_pdb(path):
    atoms = []
    for ln in open(path):
        if ln.startswith(('ATOM', 'HETATM')):
            el = ln[76:79].strip() or ln[12:16].strip()
            if el == 'H':
                continue
            try:
                atoms.append((el, float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
            except ValueError:
                continue
    return atoms

def centroid(atoms):
    a = np.array([[x, y, z] for _, x, y, z in atoms])
    return a.mean(0)

def greedy_match_rmsd(cryst, docked):
    c = defaultdict(list)
    for e, x, y, z in cryst:
        c[e].append([x, y, z])
    d = defaultdict(list)
    for e, x, y, z in docked:
        d[e].append([x, y, z])
    pairs_c, pairs_d = [], []
    for e in c:
        if e not in d:
            continue
        ca, da = c[e], d[e]
        if len(ca) != len(da):
            n = min(len(ca), len(da))
            ca, da = ca[:n], da[:n]
        pairs_c.append(np.array(ca)); pairs_d.append(np.array(da))
    if not pairs_c:
        return None
    A = np.vstack(pairs_c); B = np.vstack(pairs_d)
    ca_ = A.mean(0); cb_ = B.mean(0)
    A0 = A - ca_; B0 = B - cb_
    H = B0.T @ A0
    U, S, Vt = np.linalg.svd(H)
    dd = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, dd]) @ U.T
    B_rot = (R @ B0.T).T + ca_
    return float(np.sqrt(np.mean(np.sum((A - B_rot) ** 2, axis=1))))

def top1_score(out_pdbqt):
    """Vina 1.2.7 输出: REMARK VINA RESULT:    -6.195      0.000      0.000"""
    for ln in open(out_pdbqt):
        if ln.startswith('REMARK VINA RESULT'):
            try:
                return float(ln.split(':')[1].split()[0])
            except (IndexError, ValueError):
                pass
    return None
def run_vina(rec, lig, cx, cy, cz, sx, sy, sz, ex, scoring='vina', out=None, timeout=900):
    out = out or f'/tmp/ps_{os.path.basename(lig)}_{ex}_{scoring}.pdbqt'
    cmd = [VINA, '--receptor', rec, '--ligand', lig,
           '--center_x', str(cx), '--center_y', str(cy), '--center_z', str(cz),
           '--size_x', str(sx), '--size_y', str(sy), '--size_z', str(sz),
           '--exhaustiveness', str(ex), '--out', out]
    if scoring != 'vina':
        cmd += ['--scoring', scoring]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    dt = time.time() - t0
    score = top1_score(out) if os.path.exists(out) else None
    return score, dt, r.returncode

# ============ 体系定义 ============
T3 = os.path.join(BASE, 'test1_rmsd/3PTB')
T1 = os.path.join(BASE, 'test1_rmsd/1HVR')
rec3 = os.path.join(T3, 'BEN_rec_fixed.pdbqt')
lig3 = os.path.join(T3, 'BEN_lig.pdbqt')
cry3 = read_atoms_pdb(os.path.join(T3, 'BEN_lig.pdb'))
rec1 = os.path.join(T1, 'XK2_rec.pdbqt')
lig1 = os.path.join(T1, 'XK2_lig.pdbqt')
cry1 = read_atoms_pdb(os.path.join(T1, 'XK2_lig.pdb'))
c3 = centroid(cry3); c1 = centroid(cry1)

# 6LU7 配体（obabel 生成）
rec6 = os.path.join(BASE, 'case1_mpro/6LU7_prot.pdbqt')
lig6 = '/tmp/5fu_lig.pdbqt'
if not os.path.exists(lig6):
    subprocess.run([OBABEL, '-:Fc1c[nH]c(=O)[nH]c1=O', '--gen3d', '-O', lig6],
                   capture_output=True)
c6 = (-14.043, 17.445, 66.228)  # Cys145 中心（case1 一键流程）

results = {'exhaustiveness': {}, 'scoring': {}, 'boxsize': {}}

# ============ 1. exhaustiveness 扫描 ============
print('== exhaustiveness 扫描 ==')
for sysname, rec, lig, cry, c in [('3PTB', rec3, lig3, cry3, c3),
                                  ('1HVR', rec1, lig1, cry1, c1)]:
    exs = [4, 8, 16, 32] if sysname == '3PTB' else [8, 16]
    nrep = 3 if sysname == '3PTB' else 1
    for ex in exs:
        scores, rmsds, times = [], [], []
        for rep in range(nrep):
            out = f'/tmp/ps_{sysname}_ex{ex}_r{rep}.pdbqt'
            sc, dt, rc = run_vina(rec, lig, *c, 25, 25, 25, ex, out=out)
            if sc is not None:
                rmsd = greedy_match_rmsd(cry, read_atoms_pdbqt(out))
                scores.append(sc); rmsds.append(rmsd); times.append(dt)
        results['exhaustiveness'][f'{sysname}_ex{ex}'] = {
            'scores': scores, 'rmsds': rmsds, 'times': times,
            'mean_score': float(np.mean(scores)) if scores else None,
            'mean_rmsd': float(np.mean([r for r in rmsds if r is not None])) if rmsds else None,
            'std_score': float(np.std(scores)) if len(scores) > 1 else 0.0}
        print(f'  {sysname} ex{ex}: score={results["exhaustiveness"][f"{sysname}_ex{ex}"]["mean_score"]} '
              f'RMSD={results["exhaustiveness"][f"{sysname}_ex{ex}"]["mean_rmsd"]} t={dt:.1f}s')

# ============ 2. 评分函数对比 ============
print('== 评分函数对比 ==')
for sysname, rec, lig, cry, c, ex in [('3PTB', rec3, lig3, cry3, c3, 16),
                                      ('1HVR', rec1, lig1, cry1, c1, 16),
                                      ('6LU7', rec6, lig6, None, c6, 16)]:
    for sc_name in ['vina', 'vinardo']:
        out = f'/tmp/ps_{sysname}_{sc_name}.pdbqt'
        sc, dt, rc = run_vina(rec, lig, *c, 25, 25, 25, ex, scoring=sc_name, out=out)
        rmsd = greedy_match_rmsd(cry, read_atoms_pdbqt(out)) if cry else None
        results['scoring'][f'{sysname}_{sc_name}'] = {
            'score': sc, 'rmsd': rmsd, 'time': dt}
        print(f'  {sysname} {sc_name}: score={sc} RMSD={rmsd} t={dt:.1f}s')

# ============ 3. 盒尺寸影响（3PTB）============
print('== 盒尺寸影响 ==')
for sz in [20, 25, 30]:
    out = f'/tmp/ps_3PTB_box{sz}.pdbqt'
    sc, dt, rc = run_vina(rec3, lig3, *c3, sz, sz, sz, 16, out=out)
    rmsd = greedy_match_rmsd(cry3, read_atoms_pdbqt(out))
    results['boxsize'][f'box{sz}'] = {'score': sc, 'rmsd': rmsd, 'time': dt}
    print(f'  box {sz}: score={sc} RMSD={rmsd}')

# 汇总统计
ex = results['exhaustiveness']
rep3 = [ex[f'3PTB_ex16']['scores'], ex[f'3PTB_ex16']['rmsds']]
results['repeatability'] = {
    '3PTB_ex16_scores': ex['3PTB_ex16']['scores'],
    '3PTB_ex16_rmsds': ex['3PTB_ex16']['rmsds'],
    'score_range': max(ex['3PTB_ex16']['scores']) - min(ex['3PTB_ex16']['scores']),
    'rmsd_range': max([r for r in ex['3PTB_ex16']['rmsds'] if r]) - min([r for r in ex['3PTB_ex16']['rmsds'] if r]),
}

with open(os.path.join(OUT, 'param_scan_results.json'), 'w') as f:
    json.dump(results, f, indent=1, ensure_ascii=False)
print('\nDONE → test_extra/param_scan_results.json')
print(json.dumps(results['repeatability'], indent=1))
