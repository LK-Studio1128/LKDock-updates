#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试 3/4：批量虚拟筛选（10 配体 vs 单配体）+ 传统链便捷性实测"""
import subprocess, os, time, json, glob, sys

BASE = os.path.dirname(os.path.abspath(__file__))
VINA = '/Users/luoxiaowen/Desktop/LKDock/打包/LKDock_v1.0_Mac/vina'
LIG_DIR = os.path.join(BASE, 'test3_batch/ligands_rd')
OUT = os.path.join(BASE, 'test3_batch')

RECEPTORS = {
    '3PTB': {'rec': 'test1_rmsd/3PTB/BEN_rec.pdbqt', 'center': (-1.76, 14.46, 16.92), 'lig': 'BEN'},
    '1HVR': {'rec': 'test1_rmsd/1HVR/XK2_rec.pdbqt', 'center': (-9.19, 15.91, 27.95), 'lig': 'XK2'},
}

def dock_one(rec, lig, center, out, exhaust, cpu=4):
    t0 = time.time()
    r = subprocess.run([VINA, '--receptor', rec, '--ligand', lig,
                        '--center_x', f'{center[0]:.3f}', '--center_y', f'{center[1]:.3f}', '--center_z', f'{center[2]:.3f}',
                        '--size_x', '25', '--size_y', '25', '--size_z', '25',
                        '--exhaustiveness', str(exhaust), '--cpu', str(cpu), '--out', out],
                       capture_output=True, text=True, timeout=600)
    return time.time() - t0, r.returncode

results = {}
for code, cfg in RECEPTORS.items():
    os.makedirs(os.path.join(OUT, code), exist_ok=True)
    ligs = sorted(glob.glob(os.path.join(LIG_DIR, '*.pdbqt')))
    # 1) 单配体基线（benzamidine，exhaustiveness 16）
    single = [l for l in ligs if 'benzamidine' in l][0]
    t_single, rc = dock_one(cfg['rec'], single, cfg['center'],
                            os.path.join(OUT, code, 'single_out.pdbqt'), 16)
    # 2) 批量 10 配体（exhaustiveness 16）
    t_batch_total = 0.0
    n_ok = 0
    for i, l in enumerate(ligs):
        t, rc = dock_one(cfg['rec'], l, cfg['center'],
                         os.path.join(OUT, code, f'batch_{i}.pdbqt'), 16)
        t_batch_total += t
        n_ok += 1 if rc == 0 else 0
    avg = t_batch_total / n_ok if n_ok else None
    print(f'{code}: 单配体 {t_single:.1f}s | 批量10配体 总 {t_batch_total:.1f}s 平均 {avg:.2f}s/配体 (成功{n_ok}/10)')
    results[code] = {'single_s': round(t_single, 1), 'batch10_total_s': round(t_batch_total, 1),
                     'batch10_avg_s': round(avg, 2) if avg else None, 'n_ok': n_ok}

with open(os.path.join(OUT, 'batch_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print('saved batch_results.json')
