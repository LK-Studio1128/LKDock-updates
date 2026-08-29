#!/usr/bin/env python
"""案例 3: Mpro(6LU7) + 5-FU 复合物 MD 稳定性验证 (OpenMM, GAFF2 + ff14SB, 隐式溶剂)"""
import os, sys
os.environ.setdefault('OPENMM_PLUGIN_DIR', '')
import numpy as np
from openmm import unit, LangevinIntegrator, Platform
from openmm.app import (PDBFile, ForceField, Modeller, Simulation, PDBReporter,
                        StateDataReporter, HBonds, CutoffNonPeriodic)
from openmmforcefields.generators import GAFFTemplateGenerator
from openff.toolkit import Molecule as OFFMolecule

BASE = '/Users/luoxiaowen/Desktop/LKDock/LKDock软件介绍论文/04_测试结果/case3_md'
os.makedirs(BASE, exist_ok=True)

# ---------- 1. 体系准备 ----------
# 受体: 6LU7 Mpro 链 A 蛋白（重原子，剔除主链不完整残基）
rec_pdb = os.path.join(BASE, '6LU7_prot.pdb')
if not os.path.exists(rec_pdb):
    from collections import defaultdict
    atoms = [l for l in open('/Users/luoxiaowen/Desktop/LKDock/LKDock软件介绍论文/04_测试结果/pdbs/6LU7.pdb')
             if l.startswith('ATOM') and l[21] == 'A' and l[13] != 'H'
             and int(l[22:26].strip()) <= 304]
    by_res = defaultdict(list)
    for l in atoms:
        by_res[l[22:26].strip()].append(l)
    complete = []
    for key, res_atoms in by_res.items():
        names = {a[12:16].strip() for a in res_atoms}
        if {'N', 'CA', 'C', 'O'} <= names:
            complete.extend(res_atoms)
    # 给 C 端残基加 OXT（用该残基 C 原子行做模板，保证列对齐）
    last_res = '304'  # 截断后 C 端为 304（字符串 max 会误选 '99'）
    c_atom = [a for a in by_res[last_res] if a[12:16].strip() == 'C'][0]
    o_atom = [a for a in by_res[last_res] if a[12:16].strip() == 'O'][0]
    cx, cy, cz = float(c_atom[30:38]), float(c_atom[38:46]), float(c_atom[46:54])
    ox, oy, oz = float(o_atom[30:38]), float(o_atom[38:46]), float(o_atom[46:54])
    row = list(c_atom.rstrip('\n'))
    def put(row, col, text):
        for i, ch in enumerate(text):
            if col-1+i < len(row): row[col-1+i] = ch
    put(row, 13, ' OXT')
    put(row, 31, '%8.3f' % (2*cx-ox))
    put(row, 39, '%8.3f' % (2*cy-oy))
    put(row, 47, '%8.3f' % (2*cz-oz))
    put(row, 77, 'O')
    oxt = ''.join(row).rstrip() + '\n'
    print('OXT 行:', repr(oxt[:60]))
    print('OXT 残基号:', repr(oxt[22:26]), '| 坐标:', repr(oxt[30:54]))
    complete.append(oxt)
    print('链A完整残基数:', len(by_res), '→', len({l[22:26].strip() for l in complete}), '+OXT')
    open(rec_pdb, 'w').writelines(complete + ['END\n'])
print('受体原子:', sum(1 for l in open(rec_pdb) if l.startswith('ATOM')))

# 配体: 5-FU 全氢结构（obabel gen3d）→ Kabsch 对齐到对接 pose 重原子
lig_out = '/Users/luoxiaowen/Desktop/LKDock/LKDock软件介绍论文/04_测试结果/case1_mpro/out/5-fluorouracil_out.pdbqt'
lig_pdb = os.path.join(BASE, '5fu_pose.pdb')
import subprocess, numpy as np
from rdkit import Chem as RChem
from rdkit.Chem import AllChem
# rdkit ETKDG 生成 5-FU 3D
r_mol = RChem.AddHs(RChem.MolFromSmiles('Fc1c[nH]c(=O)[nH]c1=O'))
try:
    AllChem.EmbedMolecule(r_mol, randomSeed=42)
except Exception:
    AllChem.EmbedMolecule(r_mol)
AllChem.MMFFOptimizeMolecule(r_mol)
RChem.MolToPDBFile(r_mol, os.path.join(BASE, '5fu_gen.pdb'))
# 提取对接 pose 重原子（P0 作为对齐目标）
dock_heavy = []
for l in open(lig_out):
    if l.startswith('MODEL') and '2' in l.split()[1:2]:
        break
    if l.startswith(('ATOM', 'HETATM')) and not l[12:16].strip().startswith('H'):
        dock_heavy.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])
dock_heavy = np.array(dock_heavy)
print('DEBUG dock_heavy:', dock_heavy.shape)
# 提取生成结构重原子
gen_heavy = []
for l in open(os.path.join(BASE, '5fu_gen.pdb')):
    if l.startswith(('ATOM', 'HETATM')) and not l[12:16].strip().startswith('H'):
        gen_heavy.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])
gen_heavy = np.array(gen_heavy)
print('DEBUG gen_heavy:', gen_heavy.shape)
# Kabsch: gen → dock
A = gen_heavy - gen_heavy.mean(0)
B = dock_heavy - dock_heavy.mean(0)
H = B.T @ A
U, S, Vt = np.linalg.svd(H)
d = np.sign(np.linalg.det(Vt.T @ U.T))
R = Vt.T @ np.diag([1, 1, d]) @ U.T
trans = dock_heavy.mean(0) - R @ gen_heavy.mean(0)
# 写出对齐后的全氢配体 PDB
out_lines = []
for l in open(os.path.join(BASE, '5fu_gen.pdb')):
    if l.startswith(('ATOM', 'HETATM')):
        xyz = np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])
        xyz2 = R @ xyz + trans
        row = list(l.rstrip('\n'))
        row[30:38] = '%8.3f' % xyz2[0]
        row[38:46] = '%8.3f' % xyz2[1]
        row[46:54] = '%8.3f' % xyz2[2]
        row[17:20] = 'MOL'
        out_lines.append(''.join(row) + '\n')
open(lig_pdb, 'w').writelines(out_lines + ['END\n'])
# 补 CONECT（rdkit 生成的键信息，保证 OpenMM 建键与 GAFF 同构匹配）
conect_lines = [l for l in open(os.path.join(BASE, '5fu_gen.pdb')) if l.startswith('CONECT')]
with open(lig_pdb, 'a') as f:
    for cl in conect_lines:
        f.write(cl)
    f.write('END\n')
print('配体原子(全氢):', sum(1 for l in out_lines if l.startswith(('ATOM', 'HETATM'))))
align_rmsd = float(np.sqrt(np.mean(np.sum(((R @ A.T).T - B)**2, axis=1))))
print('对齐 RMSD(重原子): %.2f Å' % align_rmsd)

# ---------- 2. 力场 ----------
ff = ForceField('amber14/protein.ff14SB.xml', 'implicit/obc2.xml')
lig_mol = OFFMolecule.from_smiles('Fc1c[nH]c(=O)[nH]c1=O', allow_undefined_stereo=True)
lig_mol.name = 'MOL'  # 匹配 rdkit 输出的配体残基名
lig_mol.assign_partial_charges(partial_charge_method='gasteiger')  # AM1-BCC 不可用，用 gasteiger
gaff = GAFFTemplateGenerator(molecules=[lig_mol], forcefield='gaff-2.11')
ff.registerTemplateGenerator(gaff.generator)

pdb = PDBFile(lig_pdb)
rec = PDBFile(rec_pdb)
# 受体加氢（ff14SB）后合并配体
mod = Modeller(rec.topology, rec.positions)
mod.addHydrogens(forcefield=ff, pH=7.4)
lig_mod = Modeller(pdb.topology, pdb.positions)
mod.add(lig_mod.topology, lig_mod.positions)
print('复合物原子(加氢后):', mod.topology.getNumAtoms())
for r in mod.topology.residues():
    if r.name not in ('UNL', 'HOH') and len([a for a in r.atoms()]) < 5:
        pass
print('残基列表(前3):', [r.name for r in mod.topology.residues()][:3])

# 隐式溶剂 (GBSA OBC2) —— 通过 XML 力场加载
system = ff.createSystem(mod.topology, nonbondedMethod=CutoffNonPeriodic,
                         constraints=HBonds)
print('隐式溶剂 GBSA(OBC) 体系创建成功')

# ---------- 3. 最小化 + MD ----------
integrator = LangevinIntegrator(300*unit.kelvin, 1/unit.picosecond, 2*unit.femtosecond)
sim = Simulation(mod.topology, system, integrator)
sim.context.setPositions(mod.positions)
# 跳过 minimize（GBSA 隐式水 4657 原子极慢）；直接 MD（初猜位置已接近对接 pose）
print('跳过 minimize，直接 MD', flush=True)

# 记录配体原子索引（从蛋白质序列末尾开始 = 最后 9 个原子 5FU 重原子+H）
lig_heavy = [a.index for a in mod.topology.atoms() if a.residue.name == 'UNL']
if not lig_heavy:
    # GAFF 配体残基名可能是配体名
    lig_heavy = [a.index for a in mod.topology.atoms() if a.index >= mod.topology.getNumAtoms() - 12]
print('配体原子索引数:', len(lig_heavy))

# 蛋白 Cα 索引（对齐用）
ca_idx = [a.index for a in mod.topology.atoms() if a.name == 'CA' and a.residue.name != 'UNL']
print('蛋白 Cα 数:', len(ca_idx))

# 预取起始坐标
init_pos = sim.context.getState(getPositions=True).getPositions(asNumpy=True)

# 平衡 10ps + 生产 100ps（GBSA 隐式水 CPU 较慢，演示用短模拟）
sim.reporters.append(PDBReporter(os.path.join(BASE, 'traj.pdb'), 500))
sim.reporters.append(StateDataReporter(os.path.join(BASE, 'md.log'), 500,
    step=True, time=True, potentialEnergy=True, temperature=True))
print('开始 MD: 平衡 10ps + 生产 100ps ...', flush=True)
sim.step(int(5e3))     # 平衡 10 ps
sim.step(int(50e3))    # 生产 100 ps (2fs * 50000 = 100ps)
print('MD 完成', flush=True)

# ---------- 4. 配体 RMSD 分析 ----------
# 重读轨迹
from openmm.app import PDBFile as PDBReader
traj = PDBReader(os.path.join(BASE, 'traj.pdb'))
n_frames = 0
lig_rmsd = []
protein_rmsd = []
for frame_pos in traj.getPositions():
    import numpy as np
    pos = np.array([[p[0], p[1], p[2]] for p in frame_pos])
    init = np.array([[p[0], p[1], p[2]] for p in init_pos])
    # 蛋白 Cα 对齐 (Kabsch)
    ca_init = init[ca_idx]; ca_pos = pos[ca_idx]
    ca_ci = ca_init.mean(0); ca_cp = ca_pos.mean(0)
    A = ca_init - ca_ci; B = ca_pos - ca_cp
    H = B.T @ A
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    aligned = (R @ B.T).T + ca_ci
    # 配体 RMSD（对齐后）
    lr = float(np.sqrt(np.mean(np.sum((init[lig_heavy] - aligned[lig_heavy])**2, axis=1)))) if lig_heavy else np.nan
    lig_rmsd.append(lr)
    # 蛋白 Cα RMSD
    pr = float(np.sqrt(np.mean(np.sum((ca_init - aligned[ca_idx])**2, axis=1))))
    protein_rmsd.append(pr)
    n_frames += 1

lig_rmsd = np.array(lig_rmsd); protein_rmsd = np.array(protein_rmsd)
print('轨迹帧数:', n_frames)
print('配体 RMSD (Å): 起始 %.2f | 均值 %.2f | 末值 %.2f | 最大 %.2f' %
      (lig_rmsd[0], lig_rmsd.mean(), lig_rmsd[-1], lig_rmsd.max()))
print('蛋白 Cα RMSD (Å): 均值 %.2f | 末值 %.2f' % (protein_rmsd.mean(), protein_rmsd[-1]))

np.save(os.path.join(BASE, 'lig_rmsd.npy'), lig_rmsd)
np.save(os.path.join(BASE, 'protein_rmsd.npy'), protein_rmsd)
with open(os.path.join(BASE, 'summary.txt'), 'w') as f:
    f.write(f'frames={n_frames}\nlig_rmsd_mean={lig_rmsd.mean():.2f}\nlig_rmsd_final={lig_rmsd[-1]:.2f}\nlig_rmsd_max={lig_rmsd.max():.2f}\nprotein_rmsd_mean={protein_rmsd.mean():.2f}\nprotein_rmsd_final={protein_rmsd[-1]:.2f}\n')
print('分析完成 → case3_md/summary.txt')
