#!/usr/bin/env python
"""案例3 服务器版 v5: Mpro(6LU7)+5-FU 显式水 MD 1ns (OpenMM CUDA)
v4修复: 加大最小化 -> 1fs步长 -> 温度梯度平衡(50/150/300K) -> 高约束容差 -> 单context复用
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import openmm as mm
from openmm import unit, Platform
from openmm.app import (PDBFile, ForceField, Modeller, Simulation, DCDReporter,
                        StateDataReporter, HBonds, CutoffPeriodic)

BASE = '/root/case3_md'
os.makedirs(BASE, exist_ok=True)
LOG = open(os.path.join(BASE, 'md_run.log'), 'a', buffering=1)
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); LOG.write(s+'\n')

# ---------- 1. 力场 ----------
ff = ForceField('amber14/protein.ff14SB.xml', 'amber14/tip3p.xml')
from openmmforcefields.generators import GAFFTemplateGenerator
from openff.toolkit import Molecule as OFFMolecule
lig_mol = OFFMolecule.from_smiles('Fc1c[nH]c(=O)[nH]c1=O', allow_undefined_stereo=True)
lig_mol.name = 'MOL'
try:
    lig_mol.assign_partial_charges(partial_charge_method='gasteiger')
except Exception:
    pass
gaff = GAFFTemplateGenerator(molecules=[lig_mol], forcefield='gaff-2.11')
ff.registerTemplateGenerator(gaff.generator)

# ---------- 2. 构建体系 ----------
rec = PDBFile(os.path.join(BASE, '6LU7_prot.pdb'))
lig = PDBFile(os.path.join(BASE, '5fu_pose.pdb'))
mod = Modeller(rec.topology, rec.positions)
mod.addHydrogens(forcefield=ff, pH=7.4)
log('加氢后原子:', mod.topology.getNumAtoms())
lig_mod = Modeller(lig.topology, lig.positions)
mod.add(lig_mod.topology, lig_mod.positions)
log('复合物原子:', mod.topology.getNumAtoms())
mod.addSolvent(ff, model='tip3p', padding=1.0*unit.nanometer)
log('加水后原子:', mod.topology.getNumAtoms())

platform = Platform.getPlatformByName('CUDA')
# 冗长日志只打一次初始化信息
system = ff.createSystem(mod.topology, nonbondedMethod=CutoffPeriodic,
                         nonbondedCutoff=1.0*unit.nanometer,
                         constraints=HBonds, rigidWater=True)

# ---------- 3. 位置约束力（全程保留，k 值阶段递减） ----------
restraint = mm.CustomExternalForce('0.5*k*periodicdistance(x, y, z, x0, y0, z0)^2')
restraint.addGlobalParameter('k', 10.0*unit.kilocalorie_per_mole/unit.angstrom**2)
for p in ('x0', 'y0', 'z0'):
    restraint.addPerParticleParameter(p)
n_rest = 0
for a in mod.topology.atoms():
    if (a.name == 'CA' and a.residue.name != 'MOL') or a.residue.name == 'MOL':
        if not a.name.startswith('H'):
            restraint.addParticle(a.index, mod.positions[a.index])
            n_rest += 1
system.addForce(restraint)
log('约束粒子数:', n_rest)

integrator = mm.LangevinMiddleIntegrator(50*unit.kelvin, 1/unit.picosecond, 1*unit.femtosecond)
integrator.setConstraintTolerance(1e-7)
sim = Simulation(mod.topology, system, integrator, platform)
sim.context.setPositions(mod.positions)
log('平台:', platform.getPlatformValueType('CUDA') if hasattr(platform,'getPlatformValueType') else 'CUDA',
    '| 设备:', os.environ.get('CUDA_VISIBLE_DEVICES','default'))
log('最小化中(最多20000次迭代)...')
sim.minimizeEnergy(maxIterations=20000, tolerance=5*unit.kilojoule_per_mole/unit.nanometer)
st = sim.context.getState(getEnergy=True)
log('最小化完成 PE=%.1f kJ/mol' % (st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)))

# ---------- 4. 温度梯度平衡: 50K->150K->300K 各50ps, k 逐级减半 ----------
def equilibrate(temp_k, ps, k_val):
    integrator.setTemperature(temp_k*unit.kelvin)
    sim.context.setParameter('k', k_val)
    sim.context.setVelocitiesToTemperature(temp_k*unit.kelvin)
    log('平衡 %.0fK, k=%.1f, %d ps ...' % (temp_k, k_val, ps))
    # 每10ps检查一次NaN/能量
    chunk = int(ps*1000/10)
    for i in range(10):
        sim.step(chunk)
        stq = sim.context.getState(getEnergy=True)
        pe = stq.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        if not np.isfinite(pe):
            raise RuntimeError('NaN detected at temp=%dK chunk %d' % (temp_k, i))
        log('  t=%.0fps PE=%.0f' % ((i+1)*ps, pe))

equilibrate(50, 50, 10.0)
equilibrate(150, 50, 5.0)
equilibrate(300, 50, 2.0)
log('约束平衡全部完成')

# ---------- 5. 生产 1ns (2fs, 松约束k=0.5) ----------
integrator.setTemperature(300*unit.kelvin)
integrator.setStepSize(2*unit.femtosecond)
sim.context.setParameter('k', 0.5)
sim.context.setVelocitiesToTemperature(300*unit.kelvin)
sim.reporters.append(DCDReporter(os.path.join(BASE, 'traj.dcd'), 2500))  # 每帧25ps, 共40帧
sim.reporters.append(StateDataReporter(LOG, 2500, step=True, time=True,
    potentialEnergy=True, temperature=True, separator=','))
log('生产MD 1ns 开始')
sim.step(500000)
log('MD 完成')

# ---------- 6. RMSD 分析（Kabsch, 帧对帧首） ----------
log('开始分析...')
import mdtraj as md
try:
    traj = md.load(os.path.join(BASE, 'traj.dcd'), top=os.path.join(BASE, 'topology.pdb'))
    ca = traj.topology.select('name CA and not resname MOL')
    lig_sel = traj.topology.select('resname MOL')
    ref = traj[0]
    lig_rmsd, prot_rmsd = [], []
    for i in range(traj.n_frames):
        fr = traj[i]
        pr = md.rmsd(fr, ref, atom_indices=ca)[0]
        # 配体RMSD: 先按CA叠合再算
        fr.superpose(ref, atom_indices=ca)
        lr = np.sqrt(np.mean(np.sum((fr.xyz[0][lig_sel]-ref.xyz[0][lig_sel])**2, axis=1)))*10.0
        prot_rmsd.append(pr*10.0); lig_rmsd.append(lr)
except ImportError:
    log('mdtraj不可用, 用纯numpy分析')
    from openmm.app import DCDFile
    dcdf = DCDFile(open(os.path.join(BASE,'traj.dcd'),'rb'), sim.topology, 2*unit.femtosecond)
    frames = [np.array(f.value_in_unit(unit.nanometer)) for f in dcd.readModel()]
    pos0 = np.array(mod.positions.value_in_unit(unit.nanometer))
    ca_idx = [a.index for a in mod.topology.atoms() if a.name=='CA' and a.residue.name!='MOL']
    li_idx = [a.index for a in mod.topology.atoms() if a.residue.name=='MOL' and not a.element is None and a.element.symbol!='H']
    def kabsch_rmsd(P, Q):
        pc, qc = P.mean(0), Q.mean(0); A, B = P-pc, Q-qc
        U,S,Vt = np.linalg.svd(A.T@B); d = np.sign(np.linalg.det(Vt.T@U.T))
        R = Vt.T@np.diag([1,1,d])@U.T
        return float(np.sqrt(np.mean(np.sum((A@R - B)**2, axis=1))))
    ref_ca, ref_li = pos0[ca_idx], pos0[li_idx]
    lig_rmsd, prot_rmsd = [], []
    for fpos in frames:
        prot_rmsd.append(kabsch_rmsd(fpos[ca_idx], ref_ca))
        lig_rmsd.append(kabsch_rmsd(fpos[li_idx], ref_li))

lig_rmsd = np.array(lig_rmsd); prot_rmsd = np.array(prot_rmsd)
log('轨迹帧:', len(lig_rmsd))
log('配体 RMSD: mean %.2f | final %.2f | max %.2f Å' % (lig_rmsd.mean(), lig_rmsd[-1], lig_rmsd.max()))
log('蛋白 Cα RMSD: mean %.2f | final %.2f Å' % (prot_rmsd.mean(), prot_rmsd[-1]))
with open(os.path.join(BASE, 'summary.txt'), 'w') as fo:
    fo.write('frames=%d\nlig_mean=%.2f\nlig_final=%.2f\nlig_max=%.2f\nprot_mean=%.2f\nprot_final=%.2f\n' %
             (len(lig_rmsd), lig_rmsd.mean(), lig_rmsd[-1], lig_rmsd.max(), prot_rmsd.mean(), prot_rmsd[-1]))
np.save(os.path.join(BASE, 'lig_rmsd.npy'), lig_rmsd)
np.save(os.path.join(BASE, 'prot_rmsd.npy'), prot_rmsd)
# 也保存第一帧拓扑用于mdtraj加载
from openmm.app import PDBFile as PF
with open(os.path.join(BASE,'topology.pdb'),'w') as fo:
    PF.writeFile(sim.topology, mod.positions, fo)
log('DONE_ALL')
