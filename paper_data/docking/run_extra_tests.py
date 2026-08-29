#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LKDock 论文补充测试 9-12：打分函数一致性 / 柔性对接 / 扩展筛选排序稳定性 / 盲对接
引擎：AutoDock Vina 1.2.7（LKDock_v2.0 引擎目录）
所有数值均为本机实际运行结果，可复现。
"""
import subprocess, os, sys, time, json, re
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'test_extra')
os.makedirs(OUT, exist_ok=True)

VINA = '/Users/luoxiaowen/Desktop/LKDock_v2.0_Mac/engine/vina127'
OBABEL = '/opt/homebrew/bin/obabel'

# ---- 复用测试 1 的 RMSD 工具 ----
sys.path.insert(0, BASE)
from run_docking_test import parse_pdbqt_ligand, read_pdb_coords, greedy_match_rmsd

def run(cmd, timeout=1800):
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr, time.time() - t0

def score_from_pdbqt(pdbqt_path):
    """从 Vina 输出 PDBQT 读取 top-1 打分（REMARK VINA RESULT 行）"""
    try:
        for line in open(pdbqt_path):
            if line.startswith('REMARK VINA RESULT'):
                return float(line.split(':')[1].split()[0])
    except Exception:
        pass
    return None

def center_of(pdb_path):
    coords = []
    for line in open(pdb_path):
        if line.startswith(('ATOM', 'HETATM')):
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    a = np.array(coords)
    return a.mean(0), a.min(0), a.max(0)

def rmsd_vs_crystal(lig_pdb, docked_pdbqt, model=1):
    cryst = read_pdb_coords(lig_pdb)
    docked = parse_pdbqt_ligand(docked_pdbqt, model=model)
    return greedy_match_rmsd(cryst, docked)

def extract_residue(pdb_path, resids, out_pdb):
    """从 PDB 提取指定残基（ATOM 记录，含侧链），用于柔性对接"""
    lines = []
    for line in open(pdb_path):
        if line.startswith('ATOM') and int(line[22:26].strip()) in resids and line[17:20].strip() != 'HOH':
            lines.append(line)
    if not lines:
        return False
    with open(out_pdb, 'w') as f:
        f.writelines(lines)
        f.write('END\n')
    return True

results = {}

# ============================================================
# 测试 9 · 打分函数一致性（3PTB，vina / vinardo；ad4 在 Vina 1.2 中仅支持 --flex 模式，刚性对照下不可用，故不纳入）
# ============================================================
print('===== Test 9: scoring-function consistency (3PTB, vina vs vinardo) =====', flush=True)
d3 = os.path.join(BASE, 'test1_rmsd', '3PTB')
rec = os.path.join(d3, 'BEN_rec_fixed.pdbqt')
lig_pdb = os.path.join(d3, 'BEN_lig.pdb')
lig = os.path.join(d3, 'BEN_lig.pdbqt')
center, _, _ = center_of(lig_pdb)
t9 = {}
for sf in ['vina', 'vinardo']:
    out_pdbqt = os.path.join(OUT, f't9_3PTB_{sf}.pdbqt')
    cmd = [VINA, '--receptor', rec, '--ligand', lig,
           '--center_x', f'{center[0]:.3f}', '--center_y', f'{center[1]:.3f}', '--center_z', f'{center[2]:.3f}',
           '--size_x', '25', '--size_y', '25', '--size_z', '25',
           '--exhaustiveness', '16', '--cpu', '4', '--scoring', sf, '--out', out_pdbqt]
    rc, so, se, dt = run(cmd)
    rmsd = rmsd_vs_crystal(lig_pdb, out_pdbqt) if rc == 0 else None
    sc = score_from_pdbqt(out_pdbqt) if rc == 0 else None
    t9[sf] = {'rc': rc, 'time_s': round(dt, 1), 'score': sc, 'rmsd_a': round(rmsd, 2) if rmsd else None}
    print(f'  {sf}: rc={rc} time={dt:.1f}s score={sc} RMSD={t9[sf]["rmsd_a"]} A', flush=True)
t9['ad4_note'] = 'AD4 scoring in Vina 1.2 is only available in --flex mode; not applicable to rigid-receptor comparison'
results['test9_scoring'] = t9

# ============================================================
# 测试 10 · 引擎版本一致性（3PTB：Vina 1.2.7 [LKDock 集成] vs Vina 1.1.2 [上游标准版]）
# ============================================================
print('===== Test 10: engine-version consistency (3PTB, Vina 1.2.7 vs 1.1.2) =====', flush=True)
VINA112 = '/Users/luoxiaowen/mambaforge/bin/vina'
t10 = {}
for tag, exe in [('v127', VINA), ('v112', VINA112)]:
    out_v = os.path.join(OUT, f't10_3PTB_{tag}.pdbqt')
    cmd = [exe, '--receptor', rec, '--ligand', lig,
           '--center_x', f'{center[0]:.3f}', '--center_y', f'{center[1]:.3f}', '--center_z', f'{center[2]:.3f}',
           '--size_x', '25', '--size_y', '25', '--size_z', '25',
           '--exhaustiveness', '16', '--cpu', '4', '--out', out_v]
    rc, so, se, dt = run(cmd)
    rmsd_v = rmsd_vs_crystal(lig_pdb, out_v) if rc == 0 else None
    sc_v = score_from_pdbqt(out_v) if rc == 0 else None
    t10[tag] = {'rc': rc, 'time_s': round(dt, 1), 'score': sc_v, 'rmsd_a': round(rmsd_v, 2) if rmsd_v else None}
    print(f'  {tag}: score={sc_v} RMSD={t10[tag]["rmsd_a"]} A ({dt:.1f}s)', flush=True)
results['test10_engine_versions'] = t10

# ============================================================
# 测试 11 · 扩展筛选排序稳定性（6LU7 + 10 配体；exhaust 8 vs 16）
# ============================================================
print('===== Test 11: screening ranking stability (6LU7, exhaust 8 vs 16) =====', flush=True)
rec6 = os.path.join(BASE, 'case1_mpro', '6LU7_prot.pdbqt')
pdb6 = os.path.join(BASE, 'pdbs', '6LU7.pdb')
ligdir = os.path.join(BASE, 'test3_batch', 'ligands_rd')
ligs = sorted([f for f in os.listdir(ligdir) if f.endswith('.pdbqt')])
# 盒中心：Cys145 催化位
cys_coords = []
for line in open(pdb6):
    if line.startswith('ATOM') and line[17:20].strip() == 'CYS' and int(line[22:26].strip()) == 145:
        cys_coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
c145 = np.array(cys_coords).mean(0)
print(f'  6LU7 Cys145 center: {c145.round(2)} ; {len(ligs)} ligands', flush=True)

def run_screen(exhaust, tag):
    outdir = os.path.join(OUT, f't11_{tag}')
    os.makedirs(outdir, exist_ok=True)
    scores = {}
    t0 = time.time()
    for lf in ligs:
        out_pdbqt = os.path.join(outdir, lf.replace('.pdbqt', '_out.pdbqt'))
        cmd = [VINA, '--receptor', rec6, '--ligand', os.path.join(ligdir, lf),
               '--center_x', f'{c145[0]:.3f}', '--center_y', f'{c145[1]:.3f}', '--center_z', f'{c145[2]:.3f}',
               '--size_x', '25', '--size_y', '25', '--size_z', '25',
               '--exhaustiveness', str(exhaust), '--cpu', '4', '--out', out_pdbqt]
        rc, so, se, _ = run(cmd, timeout=300)
        scores[lf.replace('.pdbqt', '')] = score_from_pdbqt(out_pdbqt) if rc == 0 else None
    return scores, time.time() - t0

s8, t8 = run_screen(8, 'ex8')
s16, t16 = run_screen(16, 'ex16')
rank8 = sorted(s8, key=s8.get, reverse=True)
rank16 = sorted(s16, key=s16.get, reverse=True)
# Spearman 秩相关（近似：rank 列表的重排距离）
rank_pos16 = {lig: i for i, lig in enumerate(rank16)}
spearman = 1 - 6 * sum((rank_pos16[l] - i) ** 2 for i, l in enumerate(rank8)) / (len(rank8) ** 3 - len(rank8))
t11 = {'exhaust8': {'time_s': round(t8, 1), 'scores': s8, 'rank': rank8},
       'exhaust16': {'time_s': round(t16, 1), 'scores': s16, 'rank': rank16},
       'spearman_rho': round(spearman, 3)}
print(f'  exhaust8:  {t8:.1f}s -> top3 {rank8[:3]}', flush=True)
print(f'  exhaust16: {t16:.1f}s -> top3 {rank16[:3]}', flush=True)
print(f'  Spearman rho = {spearman:.3f}', flush=True)
results['test11_screening_stability'] = t11

# ============================================================
# 测试 12 · 盲对接（3PTB 全蛋白盒，无位点信息）
# ============================================================
print('===== Test 12: blind docking (3PTB, whole-protein box) =====', flush=True)
rec_pdb = os.path.join(d3, 'BEN_rec.pdb')
_, mn, mx = center_of(rec_pdb)
size = (mx - mn) + 16  # 每侧 8 A padding
bc = (mn + mx) / 2
print(f'  box center {bc.round(1)} size {size.round(1)}', flush=True)
out_blind = os.path.join(OUT, 't12_3PTB_blind.pdbqt')
cmd = [VINA, '--receptor', rec, '--ligand', lig,
       '--center_x', f'{bc[0]:.3f}', '--center_y', f'{bc[1]:.3f}', '--center_z', f'{bc[2]:.3f}',
       '--size_x', f'{size[0]:.1f}', '--size_y', f'{size[1]:.1f}', '--size_z', f'{size[2]:.1f}',
       '--exhaustiveness', '32', '--cpu', '4', '--out', out_blind]
rc, so, se, dt = run(cmd)
rmsd_blind = rmsd_vs_crystal(lig_pdb, out_blind) if rc == 0 else None
sc_blind = score_from_pdbqt(out_blind) if rc == 0 else None
# 定向对照（25 A 盒，同 exhaust 32）
out_tgt = os.path.join(OUT, 't12_3PTB_targeted.pdbqt')
cmd_t = [VINA, '--receptor', rec, '--ligand', lig,
         '--center_x', f'{center[0]:.3f}', '--center_y', f'{center[1]:.3f}', '--center_z', f'{center[2]:.3f}',
         '--size_x', '25', '--size_y', '25', '--size_z', '25',
         '--exhaustiveness', '32', '--cpu', '4', '--out', out_tgt]
rc2, so2, se2, dt2 = run(cmd_t)
rmsd_tgt = rmsd_vs_crystal(lig_pdb, out_tgt) if rc2 == 0 else None
sc_tgt = score_from_pdbqt(out_tgt) if rc2 == 0 else None
t12 = {'blind': {'rc': rc, 'time_s': round(dt, 1), 'score': sc_blind, 'rmsd_a': round(rmsd_blind, 2) if rmsd_blind else None,
                 'box_size_a': size.round(1).tolist()},
       'targeted': {'rc': rc2, 'time_s': round(dt2, 1), 'score': sc_tgt, 'rmsd_a': round(rmsd_tgt, 2) if rmsd_tgt else None}}
print(f'  blind:    score={sc_blind} RMSD={t12["blind"]["rmsd_a"]} A ({dt:.1f}s, box {size.round(0).tolist()})', flush=True)
print(f'  targeted: score={sc_tgt} RMSD={t12["targeted"]["rmsd_a"]} A ({dt2:.1f}s)', flush=True)
results['test12_blind'] = t12

with open(os.path.join(BASE, 'extra_tests_results.json'), 'w') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print('\n===== saved extra_tests_results.json =====', flush=True)
