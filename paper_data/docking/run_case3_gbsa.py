#!/usr/bin/env python
"""案例3补充: GBSA 单点结合自由能（MD 最后帧，OpenMM 隐式溶剂）"""
import os, warnings
warnings.filterwarnings('ignore')
import numpy as np
from openmm import unit, LangevinIntegrator, Platform
from openmm.app import (PDBFile, ForceField, Modeller, Simulation, HBonds,
                        CutoffNonPeriodic)
from openmmforcefields.generators import GAFFTemplateGenerator
from openff.toolkit import Molecule as OFFMolecule

BASE = '/Users/luoxiaowen/Desktop/LKDock/LKDock软件介绍论文/04_测试结果/case3_md'
rec_pdb = os.path.join(BASE, '6LU7_prot.pdb')
lig_pdb = os.path.join(BASE, '5fu_pose.pdb')

ff = ForceField('amber14/protein.ff14SB.xml', 'implicit/obc2.xml')
lig_mol = OFFMolecule.from_smiles('Fc1c[nH]c(=O)[nH]c1=O', allow_undefined_stereo=True)
lig_mol.name = 'MOL'
lig_mol.assign_partial_charges(partial_charge_method='gasteiger')
gaff = GAFFTemplateGenerator(molecules=[lig_mol], forcefield='gaff-2.11')
ff.registerTemplateGenerator(gaff.generator)

def system_energy(pdb_path, n_lig_atoms=0, lig_only=False, prot_only=False):
    """计算一个 PDB 的 GBSA 总能量（-1 帧）"""
    pdb = PDBFile(pdb_path)
    mod = Modeller(pdb.topology, pdb.positions)
    if not (lig_only or prot_only):
        # 复合物: 需要加氢（受体）——直接读 5fu_pose 是配体；复合物用重建
        pass
    if lig_only:
        pass
    sys = ff.createSystem(mod.topology, nonbondedMethod=CutoffNonPeriodic, constraints=HBonds)
    integrator = LangevinIntegrator(300*unit.kelvin, 1/unit.picosecond, 2*unit.femtosecond)
    sim = Simulation(mod.topology, sys, integrator)
    sim.context.setPositions(mod.positions)
    st = sim.context.getState(getEnergy=True)
    return st.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)

# 1) 复合物最后帧: 从 traj.pdb 取最后 frame
# PDBReporter 输出的 traj.pdb 是 multi-model PDB（MODEL/ENDMDL）
frames = []
cur = []
for l in open(os.path.join(BASE, 'traj.pdb')):
    if l.startswith('MODEL'):
        cur = []
    elif l.startswith('ENDMDL'):
        if cur: frames.append(cur)
    elif l.startswith(('ATOM', 'HETATM')):
        cur.append(l)
print('轨迹帧数:', len(frames))
if not frames:
    print('无轨迹帧，跳过')
    raise SystemExit
last = frames[-1]
open(os.path.join(BASE, 'complex_last.pdb'), 'w').writelines(last + ['END\n'])

# 2) 蛋白-only（最后帧的蛋白原子）: 蛋白原子数 = 复合物 - 配体12
# 配体是最后 12 个原子（MOL 残基）
prot_atoms = [l for l in last if not (l[17:20].strip() == 'MOL')]
lig_atoms = [l for l in last if l[17:20].strip() == 'MOL']
open(os.path.join(BASE, 'prot_last.pdb'), 'w').writelines(prot_atoms + ['END\n'])
open(os.path.join(BASE, 'lig_last.pdb'), 'w').writelines(lig_atoms + ['END\n'])
print('蛋白原子:', len(prot_atoms), '| 配体原子:', len(lig_atoms))

# 3) 分别算能量（复合物/蛋白/配体，各自 createSystem）
# 注意: 蛋白-only 拓扑末端 OXT 已在受体；配体-only 直接读
# 复合物: 从最后帧重建（蛋白+配体）——但蛋白需要保持加氢后状态，直接读 complex_last.pdb
E_cplx = system_energy(os.path.join(BASE, 'complex_last.pdb'))
E_prot = system_energy(os.path.join(BASE, 'prot_last.pdb'))
E_lig  = system_energy(os.path.join(BASE, 'lig_last.pdb'))
dG = E_cplx - E_prot - E_lig
print('E(complex) = %.1f kcal/mol' % E_cplx)
print('E(protein) = %.1f kcal/mol' % E_prot)
print('E(ligand)  = %.1f kcal/mol' % E_lig)
print('ΔG_GBSA(单点) = %.1f kcal/mol' % dG)
with open(os.path.join(BASE, 'gbsa_dG.txt'), 'w') as f:
    f.write('E_complex=%.1f\nE_protein=%.1f\nE_ligand=%.1f\ndG_gbsa=%.1f\n' % (E_cplx, E_prot, E_lig, dG))
print('saved gbsa_dG.txt')
