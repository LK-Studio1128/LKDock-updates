#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LKDock 论文 · 新案例测试（测试 16-17）
- 案例 A：4HJO EGFR 激酶域 + 厄洛替尼（AQ4）—— 肿瘤蛋白位点研究
- 案例 B：1REV HIV-1 逆转录酶 + TIBO（TB9）—— 抗病毒位点研究
每案例：受体准备（obabel）→ 对接（Vina 1.2.7 ex16）→ RMSD vs 晶体 → 打分
输出：test_extra/new_cases_results.json
"""
import subprocess, json, time, os, re
import numpy as np
from collections import defaultdict

BASE = '/Users/luoxiaowen/Downloads/LKDock软件介绍论文/04_测试结果'
OUT = os.path.join(BASE, 'test_extra')
os.makedirs(OUT, exist_ok=True)
VINA = '/Users/luoxiaowen/Desktop/LKDock_v2.0_Mac/engine/vina127'
OBABEL = '/opt/homebrew/bin/obabel'

def read_atoms_pdb(path, resname=None, chain=None):
    atoms = []
    for ln in open(path):
        if ln.startswith(('ATOM', 'HETATM')):
            el = ln[76:79].strip() or ln[12:16].strip()
            if el == 'H':
                continue
            rn = ln[17:20].strip()
            ch = ln[21:22].strip()
            if resname and rn != resname:
                continue
            if chain and ch != chain:
                continue
            try:
                atoms.append((el, float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
            except ValueError:
                continue
    return atoms

def centroid(atoms):
    a = np.array([[x, y, z] for _, x, y, z in atoms])
    return a.mean(0)

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
def prepare_receptor(pdb_in, pdbqt_out, remove_res=None):
    """PDB → 去配体/水 → PDBQT（obabel）"""
    lines = []
    for ln in open(pdb_in):
        if ln.startswith(('ATOM', 'HETATM', 'TER', 'END')):
            rn = ln[17:20].strip()
            if remove_res and rn in remove_res:
                continue
            lines.append(ln)
    clean = pdb_in.replace('.pdb', '_clean.pdb')
    with open(clean, 'w') as f:
        f.writelines(lines)
    r = subprocess.run([OBABEL, clean, '-xr', '-O', pdbqt_out],
                       capture_output=True, text=True)
    return r.returncode == 0

def extract_ligand(pdb_in, resname, chain, pdbqt_out):
    """提取配体残基 → PDB → PDBQT"""
    lig_pdb = pdb_in.replace('.pdb', f'_{resname}.pdb')
    lines = [ln for ln in open(pdb_in) if ln.startswith('HETATM') and ln[17:20].strip() == resname
             and (not chain or ln[21:22].strip() == chain)]
    with open(lig_pdb, 'w') as f:
        f.writelines(lines)
        f.write('END\n')
    r = subprocess.run([OBABEL, lig_pdb, '-O', pdbqt_out], capture_output=True, text=True)
    return r.returncode == 0, lig_pdb

def run_vina(rec, lig, cx, cy, cz, sz=25, ex=16, out='/tmp/nc_out.pdbqt', timeout=900):
    cmd = [VINA, '--receptor', rec, '--ligand', lig,
           '--center_x', str(cx), '--center_y', str(cy), '--center_z', str(cz),
           '--size_x', str(sz), '--size_y', str(sz), '--size_z', str(sz),
           '--exhaustiveness', str(ex), '--out', out]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    dt = time.time() - t0
    return top1_score(out) if os.path.exists(out) else None, dt, r.returncode

results = {}

# ============ 案例 A：4HJO EGFR 激酶 + 厄洛替尼 ============
print('== 案例 A：4HJO EGFR 激酶域 + Erlotinib (AQ4) ==')
pdb4 = '/tmp/4HJO.pdb'
rec4 = '/tmp/4HJO_rec.pdbqt'
lig4 = '/tmp/4HJO_AQ4.pdbqt'
prepare_receptor(pdb4, rec4, remove_res={'HOH', 'AQ4'})
ok, lig4_pdb = extract_ligand(pdb4, 'AQ4', 'A', lig4)
cry4 = read_atoms_pdb(lig4_pdb, resname='AQ4')
c4 = centroid(cry4)
print(f'  配体 AQ4: {len(cry4)} 重原子, 质心 {c4.round(2)}')
sc4, dt4, rc4 = run_vina(rec4, lig4, *c4)
rmsd4 = greedy_match_rmsd(cry4, read_atoms_pdbqt('/tmp/nc_out.pdbqt'))
results['4HJO_EGFR_Erlotinib'] = {'score': sc4, 'rmsd_a': rmsd4, 'time_s': dt4,
                                   'ligand_atoms': len(cry4)}
print(f'  对接完成: score={sc4} RMSD={rmsd4:.2f}Å t={dt4:.1f}s')

# ============ 案例 B：1REV HIV-1 RT + TIBO (TB9) ============
print('== 案例 B：1REV HIV-1 逆转录酶 + TIBO (TB9) ==')
pdb1 = '/tmp/1REV.pdb'
rec1 = '/tmp/1REV_rec.pdbqt'
lig1 = '/tmp/1REV_TB9.pdbqt'
prepare_receptor(pdb1, rec1, remove_res={'HOH', 'TB9', 'CSD', 'MG'})
ok, lig1_pdb = extract_ligand(pdb1, 'TB9', 'A', lig1)
cry1 = read_atoms_pdb(lig1_pdb, resname='TB9')
c1 = centroid(cry1)
print(f'  配体 TB9: {len(cry1)} 重原子, 质心 {c1.round(2)}')
sc1, dt1, rc1 = run_vina(rec1, lig1, *c1)
rmsd1 = greedy_match_rmsd(cry1, read_atoms_pdbqt('/tmp/nc_out.pdbqt'))
results['1REV_HIV1RT_TIBO'] = {'score': sc1, 'rmsd_a': rmsd1, 'time_s': dt1,
                                'ligand_atoms': len(cry1)}
print(f'  对接完成: score={sc1} RMSD={rmsd1:.2f}Å t={dt1:.1f}s')

with open(os.path.join(OUT, 'new_cases_results.json'), 'w') as f:
    json.dump(results, f, indent=1, ensure_ascii=False)
print('\nDONE → test_extra/new_cases_results.json')
print(json.dumps(results, indent=1))
