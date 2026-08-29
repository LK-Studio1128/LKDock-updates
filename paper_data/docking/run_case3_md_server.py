#!/usr/bin/env python
"""案例3 服务器版: Mpro(6LU7)+5-FU 显式水 MD 1ns (OpenMM GPU)"""
import os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import openmm
from openmm import unit, LangevinIntegrator, Platform
from openmm.app import (PDBFile, ForceField, Modeller, Simulation, PDBReporter,
                        StateDataReporter, HBonds, CutoffPeriodic, DCDReporter)
from openmmforcefields.generators import GAFFTemplateGenerator
from openff.toolkit import Molecule as OFFMolecule

BASE = '/root/case3_md'
os.makedirs(BASE, exist_ok=True)

# ---------- 1. 力场（显式水） ----------
ff = ForceField('amber14/protein.ff14SB.xml', 'amber14/tip3p.xml')
lig_mol = OFFMolecule.from_smiles('Fc1c[nH]c(=O)[nH]c1=O', allow_undefined_stereo=True)
lig_mol.name = 'MOL'
try:
    lig_mol.assign_partial_charges(partial_charge_method='gasteiger')
except Exception:
    pass
gaff = GAFFTemplateGenerator(molecules=[lig_mol], forcefield='gaff-2.11')
ff.registerTemplateGenerator(gaff.generator)

# ---------- 2. 加载体系 ----------
rec = PDBFile(os.path.join(BASE, '6LU7_prot.pdb'))
lig = PDBFile(os.path.join(BASE, '5fu_pose.pdb'))
mod = Modeller(rec.topology, rec.positions)
mod.addHydrogens(forcefield=ff, pH=7.4)
print('加氢后原子:', mod.topology.getNumAtoms(), flush=True)
lig_mod = Modeller(lig.topology, lig.positions)
mod.add(lig_mod.topology, lig_mod.positions)
print('复合物原子:', mod.topology.getNumAtoms(), flush=True)

# 显式水
mod.addSolvent(ff, model='tip3p', padding=1.0*unit.nanometer)
print('加水后原子:', mod.topology.getNumAtoms(), flush=True)

# ---------- 3. 体系 + 最小化（GPU） ----------
platform = Platform.getPlatformByName('CUDA')
system = ff.createSystem(mod.topology, nonbondedMethod=CutoffPeriodic,
                         nonbondedCutoff=1.0*unit.nanometer, constraints=HBonds)
integrator = LangevinIntegrator(300*unit.kelvin, 1/unit.picosecond, 2*unit.femtosecond)
sim = Simulation(mod.topology, system, integrator, platform)
sim.context.setPositions(mod.positions)
sim.minimizeEnergy(maxIterations=5000, tolerance=10*unit.kilojoules_per_mole/unit.nanometer)
print('最小化完成', flush=True)

# ---------- 3b. 约束平衡（蛋白 Cα + 配体重原子 restrain） ----------
restraint = openmm.CustomExternalForce('0.5*k*periodicdistance(x, y, z, x0, y0, z0)^2')
restraint.addGlobalParameter('k', 10.0*unit.kilocalories_per_mole/unit.angstrom**2)
restraint.addPerParticleParameter('x0'); restraint.addPerParticleParameter('y0'); restraint.addPerParticleParameter('z0')
for a in mod.topology.atoms():
    if a.name == 'CA' or a.residue.name == 'MOL':
        restraint.addParticle(a.index, mod.positions[a.index])
system.addForce(restraint)
integrator2 = LangevinIntegrator(300*unit.kelvin, 1/unit.picosecond, 2*unit.femtosecond)
sim2 = Simulation(mod.topology, system, integrator2, platform)
sim2.context.setPositions(mod.positions)
sim2.context.setVelocitiesToTemperature(300*unit.kelvin)
print('约束平衡 20ps ...', flush=True)
sim2.step(int(10e3))   # 20 ps 约束平衡
print('约束平衡完成', flush=True)

# ---------- 4. 生产（去约束，新 system + 新 integrator） ----------
system_prod = ff.createSystem(mod.topology, nonbondedMethod=CutoffPeriodic,
                              nonbondedCutoff=1.0*unit.nanometer, constraints=HBonds)
integrator_prod = LangevinIntegrator(300*unit.kelvin, 1/unit.picosecond, 2*unit.femtosecond)
sim = Simulation(mod.topology, system_prod, integrator_prod, platform)
sim.context.setPositions(sim2.context.getState(getPositions=True).getPositions())
sim.context.setVelocitiesToTemperature(300*unit.kelvin)
sim.reporters.append(DCDReporter(os.path.join(BASE, 'traj.dcd'), 1000))
sim.reporters.append(StateDataReporter(os.path.join(BASE, 'md.log'), 1000,
    step=True, time=True, potentialEnergy=True, temperature=True))
sim.step(int(500e3))  # 1 ns 生产
print('MD 完成', flush=True)

# ---------- 5. RMSD 分析 ----------
print('开始分析...', flush=True)
# 保存起始坐标
init = sim.context.getState(getPositions=True).getPositions(asNumpy=True)
# 蛋白 Cα 索引 + 配体原子索引
prot_ca = [a.index for a in mod.topology.atoms() if a.name == 'CA' and a.residue.name != 'MOL']
lig_atoms = [a.index for a in mod.topology.atoms() if a.residue.name == 'MOL']
print('Cα 数:', len(prot_ca), '配体原子:', len(lig_atoms), flush=True)

from openmm.app import PDBFile as PDBReader
lig_rmsd, prot_rmsd, times = [], [], []
f = 0
for pos in PDBReader(os.path.join(BASE, 'traj_frame0.pdb'), load_all_models=False):
    pass
# 用 DCD 读轨迹
from openmm.app import DCDFile
dcd = DCDFile(os.path.join(BASE, 'traj.dcd'))
for frame_pos in dcd.getPositions():
    pos = np.array([[p[0], p[1], p[2]] for p in frame_pos]) * 10.0  # nm→Å
    init_ = np.array([[p[0], p[1], p[2]] for p in init]) * 10.0
    ca_i = init_[prot_ca]; ca_p = pos[prot_ca]
    ci = ca_i.mean(0); cp = ca_p.mean(0)
    A = ca_i - ci; B = ca_p - cp
    H = B.T @ A
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    aligned = (R @ B.T).T + ci
    lr = float(np.sqrt(np.mean(np.sum((init_[lig_atoms] - aligned[lig_atoms])**2, axis=1))))
    pr = float(np.sqrt(np.mean(np.sum((ca_i - aligned[prot_ca])**2, axis=1))))
    lig_rmsd.append(lr); prot_rmsd.append(pr)
    f += 1

lig_rmsd = np.array(lig_rmsd); prot_rmsd = np.array(prot_rmsd)
print('轨迹帧:', f, flush=True)
print('配体 RMSD: mean %.2f | final %.2f | max %.2f Å' % (lig_rmsd.mean(), lig_rmsd[-1], lig_rmsd.max()), flush=True)
print('蛋白 Cα RMSD: mean %.2f | final %.2f Å' % (prot_rmsd.mean(), prot_rmsd[-1]), flush=True)
np.save(os.path.join(BASE, 'lig_rmsd.npy'), lig_rmsd)
np.save(os.path.join(BASE, 'prot_rmsd.npy'), prot_rmsd)
with open(os.path.join(BASE, 'summary.txt'), 'w') as fout:
    fout.write('frames=%d\nlig_mean=%.2f\nlig_final=%.2f\nlig_max=%.2f\nprot_mean=%.2f\nprot_final=%.2f\n' %
               (f, lig_rmsd.mean(), lig_rmsd[-1], lig_rmsd.max(), prot_rmsd.mean(), prot_rmsd[-1]))
print('DONE_ALL', flush=True)
