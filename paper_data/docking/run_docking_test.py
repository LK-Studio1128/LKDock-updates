#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LKDock 论文测试 1/2 自动化：PDB 下载→配体/受体提取→PDBQT→Vina 对接（计时）→RMSD"""
import subprocess, os, sys, time, json
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
PDB_DIR = os.path.join(BASE, 'pdbs')
OUT_DIR = os.path.join(BASE, 'test1_rmsd')
OBABEL = '/opt/homebrew/bin/obabel'
VINA = '/Users/luoxiaowen/Desktop/LKDock/打包/LKDock_v1.0_Mac/vina'

SYSTEMS = [
    {'code': '3PTB', 'lig': 'BEN', 'label': 'Trypsin–benzamidine', 'cpu1': True},
    {'code': '1HVR', 'lig': 'XK2', 'label': 'HIV-1 protease–XK263', 'cpu1': False},
    {'code': '4YXO', 'lig': '4JC', 'label': 'Carbonic anhydrase II–benzenesulfonamide', 'cpu1': False},
]

def run(cmd, timeout=600):
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr, time.time() - t0

def download(code):
    path = os.path.join(PDB_DIR, f'{code}.pdb')
    if not os.path.exists(path):
        run(['curl', '-s', '-f', f'https://files.rcsb.org/download/{code}.pdb', '-o', path])
    return path

def split_pdb(pdb_path, lig_resn, out_dir):
    """分离配体与受体（去水），返回 (配体pdb, 受体pdb, 质心)"""
    lig_atoms, rec_atoms = [], []
    for line in open(pdb_path):
        if not line.startswith(('ATOM', 'HETATM')):
            continue
        rn = line[17:20].strip()
        if rn == 'HOH':
            continue
        elem = (line[76:78].strip() or line[12:16].strip())[0].upper()
        x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        if rn == lig_resn:
            lig_atoms.append((elem, x, y, z))
        else:
            rec_atoms.append((elem, x, y, z))
    if not lig_atoms:
        raise ValueError(f'ligand {lig_resn} not found in {pdb_path}')
    center = np.mean(np.array([[a[1], a[2], a[3]] for a in lig_atoms]), axis=0)
    # 写配体 PDB
    lig_pdb = os.path.join(out_dir, f'{lig_resn}_lig.pdb')
    with open(lig_pdb, 'w') as f:
        for i, (e, x, y, z) in enumerate(lig_atoms, 1):
            f.write(f'HETATM{i:5d} {e:>4s} {lig_resn:>3s} A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00          {e:>2s}\n')
        f.write('END\n')
    # 写受体 PDB（含蛋白原子 + 非水杂原子）
    rec_pdb = os.path.join(out_dir, f'{lig_resn}_rec.pdb')
    with open(rec_pdb, 'w') as f:
        for line in open(pdb_path):
            if line.startswith(('ATOM', 'HETATM')) and line[17:20].strip() != 'HOH' and line[17:20].strip() != lig_resn:
                f.write(line)
        f.write('END\n')
    return lig_pdb, rec_pdb, center

def to_pdbqt(in_pdb, out_pdbqt, receptor=False):
    args = [OBABEL, '-ipdb', in_pdb, '-opdbqt', '-O', out_pdbqt, '-p', '7.4', '--partialcharge', 'gasteiger']
    if receptor:
        args += ['-xr']
    return run(args)[0] == 0

def dock(rec_pdbqt, lig_pdbqt, center, out_pdbqt, cpu):
    cmd = [VINA, '--receptor', rec_pdbqt, '--ligand', lig_pdbqt,
           '--center_x', f'{center[0]:.3f}', '--center_y', f'{center[1]:.3f}', '--center_z', f'{center[2]:.3f}',
           '--size_x', '25', '--size_y', '25', '--size_z', '25',
           '--exhaustiveness', '32', '--cpu', str(cpu), '--out', out_pdbqt]
    rc, so, se, dt = run(cmd, timeout=900)
    return rc, so, se, dt

def parse_pdbqt_ligand(pdbqt_path, model=1):
    """取 MODEL 指定模型的配体原子坐标（重原子）"""
    atoms = []
    in_model, cur = False, 0
    for line in open(pdbqt_path):
        if line.startswith('MODEL'):
            cur += 1
            in_model = (cur == model)
            continue
        if line.startswith('ENDMDL'):
            in_model = False
            continue
        if in_model and line.startswith('ATOM'):
            e = line[76:78].strip() or 'C'
            if e == 'H':
                continue
            atoms.append((e, float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return atoms

def read_pdb_coords(pdb):
    atoms = []
    for line in open(pdb):
        if line.startswith(('ATOM', 'HETATM')):
            e = line[76:78].strip() or line[12:16].strip()[0]
            if e == 'H':
                continue
            atoms.append((e, float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return atoms

def greedy_match_rmsd(cryst, docked):
    """同元素贪心配对（最小距离），Kabsch 对齐后算重原子 RMSD"""
    from collections import defaultdict
    def by_elem(atoms):
        d = defaultdict(list)
        for e, x, y, z in atoms:
            d[e].append([x, y, z])
        return {k: np.array(v, dtype=float) for k, v in d.items()}
    c = by_elem(cryst); d = by_elem(docked)
    pairs_c, pairs_d = [], []
    for e in c:
        if e not in d:
            continue
        ca, da = c[e], d[e]
        if len(ca) != len(da):
            n = min(len(ca), len(da))
            ca, da = ca[:n], da[:n]
        pairs_c.append(ca); pairs_d.append(da)
    if not pairs_c:
        return None
    A = np.vstack(pairs_c); B = np.vstack(pairs_d)
    # Kabsch: align B onto A
    ca_ = A.mean(0); cb_ = B.mean(0)
    A0 = A - ca_; B0 = B - cb_
    H = B0.T @ A0
    U, S, Vt = np.linalg.svd(H)
    d_ = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d_]) @ U.T
    B_rot = (R @ B0.T).T + ca_
    return float(np.sqrt(np.mean(np.sum((A - B_rot) ** 2, axis=1))))

def score_from_log(stdout, stderr):
    txt = stdout + stderr
    m = re.search(r'^\s*1\s+([-\d.]+)', txt, re.MULTILINE)
    return float(m.group(1)) if m else None

import re

def main():
    results = []
    for sys_ in SYSTEMS:
        code, lig, label = sys_['code'], sys_['lig'], sys_['label']
        print(f'\n===== {code} ({label}) =====', flush=True)
        d = os.path.join(OUT_DIR, code)
        os.makedirs(d, exist_ok=True)
        pdb = download(code)
        lig_pdb, rec_pdb, center = split_pdb(pdb, lig, d)
        print(f'  ligand atoms: {sum(1 for l in open(lig_pdb) if l.startswith("HETATM"))}, center: {center.round(2)}', flush=True)
        lig_pdbqt = os.path.join(d, f'{lig}_lig.pdbqt')
        rec_pdbqt = os.path.join(d, f'{lig}_rec.pdbqt')
        if not to_pdbqt(lig_pdb, lig_pdbqt):
            print('  ERROR: ligand pdbqt conversion failed', flush=True); continue
        if not to_pdbqt(rec_pdb, rec_pdbqt, receptor=True):
            print('  ERROR: receptor pdbqt conversion failed', flush=True); continue
        # 单线程对接（仅小体系，避免大体系超时）
        t_cpu1 = None
        if sys_.get('cpu1'):
            out1 = os.path.join(d, f'{lig}_out_cpu1.pdbqt')
            rc, so, se, dt1 = dock(rec_pdbqt, lig_pdbqt, center, out1, 1)
            t_cpu1 = round(dt1, 1)
            print(f'  vina cpu1: rc={rc} time={dt1:.1f}s', flush=True)
        # 多线程对接（4 核）
        out4 = os.path.join(d, f'{lig}_out_cpu4.pdbqt')
        rc2, so2, se2, dt4 = dock(rec_pdbqt, lig_pdbqt, center, out4, 4)
        print(f'  vina cpu4: rc={rc2} time={dt4:.1f}s', flush=True)
        if rc2 != 0:
            print('  ERROR: docking failed', flush=True); print((so2 or se2)[-400:], flush=True); continue
        # RMSD（用 cpu4 输出，即默认最高分构象）
        docked = parse_pdbqt_ligand(out4)
        cryst = read_pdb_coords(lig_pdb)
        rmsd = greedy_match_rmsd(cryst, docked)
        s1 = score_from_log(so2, se2)
        print(f'  RMSD(heavy, Kabsch) = {rmsd:.2f} A ; top score = {s1}', flush=True)
        results.append({'system': code, 'ligand': lig, 'rmsd_a': round(rmsd, 2) if rmsd else None,
                        'score': s1, 't_cpu1_s': t_cpu1, 't_cpu4_s': round(dt4, 1)})

    print('\n===== 汇总 =====', flush=True)
    for r in results:
        print(r, flush=True)
    with open(os.path.join(BASE, 'docking_results.json'), 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print('saved docking_results.json', flush=True)

if __name__ == '__main__':
    main()
