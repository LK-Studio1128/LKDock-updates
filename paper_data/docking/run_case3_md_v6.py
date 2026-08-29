#!/usr/bin/env python
"""案例3 服务器版 v6 (最终): Mpro(6LU7)+5-FU 显式水 MD 1ns (RTX 2080 CUDA)
根因: OpenMM CUDA mixed精度下能量最小化欠收敛(蛋白C端304残基Fmax~4e6) -> 动力学NaN
方案: double精度最小化(Fmax<1e4) -> 保存XML -> mixed精度重启跑平衡+生产(速度)
流程: double最小化 -> 50K/1fs 50ps(k=10) -> 150K/1fs 50ps(k=5) -> 300K/1fs 50ps(k=2)
      -> 生产300K/2fs 1ns(k=0.5) -> DCD轨迹 + Kabsch RMSD分析
"""
import os, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import openmm as mm
from openmm import unit, Platform, XmlSerializer
from openmm.app import (PDBFile, ForceField, Modeller, Simulation, DCDReporter,
                        StateDataReporter, HBonds, CutoffPeriodic)

BASE = '/root/case3_md'
T0 = time.time()
LOG = open(os.path.join(BASE, 'md_run.log'), 'a', buffering=1)
def log(*a):
    s = '[%6.0fs] ' % (time.time()-T0) + ' '.join(str(x) for x in a)
    print(s, flush=True); LOG.write(s+'\n')

def top_force_atoms(sim, mod_topo, n=5):
    atom_list = list(mod_topo.atoms())
    f = sim.context.getState(getForces=True).getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
    fn = np.linalg.norm(f, axis=1); w = np.argsort(-fn)[:n]
    return float(fn.max()), [(atom_list[int(i)].name, int(i), atom_list[int(i)].residue.id) for i in w]

# ---------- 1. 力场 + 配体GAFF ----------
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

# ---------- 2. 构建体系（缓存，避免重复加溶剂） ----------
cache_sys = os.path.join(BASE, 'system_cache.xml')
cache_pos = os.path.join(BASE, 'positions.npy')
if os.path.exists(cache_sys) and os.path.exists(cache_pos):
    log('加载缓存体系...')
    system = XmlSerializer.deserialize(open(cache_sys).read())
    positions = unit.Quantity(np.load(cache_pos), unit.nanometer)
    topology = PDBFile(os.path.join(BASE, 'topology.pdb')).topology
else:
    rec = PDBFile(os.path.join(BASE, '6LU7_prot.pdb'))
    lgd = PDBFile(os.path.join(BASE, '5fu_pose.pdb'))
    mod = Modeller(rec.topology, rec.positions)
    mod.addHydrogens(forcefield=ff, pH=7.4)
    log('加氢后原子:', mod.topology.getNumAtoms())
    lm = Modeller(lgd.topology, lgd.positions)
    mod.add(lm.topology, lm.positions)
    log('复合物原子:', mod.topology.getNumAtoms())
    mod.addSolvent(ff, model='tip3p', padding=1.0*unit.nanometer)
    log('加水后原子:', mod.topology.getNumAtoms())
    # 约束力模板（k稍后按阶段设置）
    restraint = mm.CustomExternalForce('0.5*k*periodicdistance(x, y, z, x0, y0, z0)^2')
    restraint.addGlobalParameter('k', 10.0*unit.kilocalorie_per_mole/unit.angstrom**2)
    for p in ('x0','y0','z0'):
        restraint.addPerParticleParameter(p)
    n_rest = 0
    for a in mod.topology.atoms():
        if (a.name == 'CA' and a.residue.name != 'MOL') or (a.residue.name == 'MOL' and not a.name.startswith('H')):
            restraint.addParticle(a.index, mod.positions[a.index]); n_rest += 1
    log('约束粒子数:', n_rest)
    system = ff.createSystem(mod.topology, nonbondedMethod=CutoffPeriodic,
                             nonbondedCutoff=1.0*unit.nanometer,
                             constraints=HBonds, rigidWater=True)
    system.addForce(restraint)
    positions = mod.positions
    topology = mod.topology
    with open(cache_sys,'w') as fo:
        fo.write(XmlSerializer.serialize(system))
    np.save(cache_pos, positions.value_in_unit(unit.nanometer))
    with open(os.path.join(BASE,'topology.pdb'),'w') as fo:
        PDBFile.writeFile(topology, positions, fo)

platform = Platform.getPlatformByName('CUDA')

# ---------- 3a. DOUBLE 精度最小化 ----------
min_cache = os.path.join(BASE, 'minimized_double.npy')
if os.path.exists(min_cache):
    log('加载已最小化坐标(double缓存)...')
    positions_min = unit.Quantity(np.load(min_cache), unit.nanometer)
else:
    integ_d = mm.LangevinMiddleIntegrator(0*unit.kelvin, 1/unit.picosecond, 1*unit.femtosecond)
    integ_d.setConstraintTolerance(1e-10)
    sim_d = Simulation(topology, system, integ_d, platform, {'Precision':'double'})
    sim_d.context.setPositions(positions)
    e0 = sim_d.context.getState(getEnergy=True).getPotentialEnergy()
    log('double最小化前 PE=%.0f kJ/mol' % e0.value_in_unit(unit.kilojoule_per_mole))
    sim_d.minimizeEnergy(maxIterations=50000, tolerance=1*unit.kilojoule_per_mole/unit.nanometer)
    st = sim_d.context.getState(getEnergy=True, getPositions=True)
    fmax, top5 = top_force_atoms(sim_d, topology)
    log('double最小化后 PE=%.0f Fmax=%.0f top=%s' % (st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole), fmax, top5))
    positions_min = st.getPositions(asNumpy=True)
    np.save(min_cache, positions_min.value_in_unit(unit.nanometer))
    del sim_d, integ_d

# ---------- 3b. MIXED 精度动力学（用double最小化的高质量坐标） ----------
integ = mm.LangevinMiddleIntegrator(50*unit.kelvin, 1/unit.picosecond, 1*unit.femtosecond)
integ.setConstraintTolerance(1e-8)
sim = Simulation(topology, system, integ, platform, {'Precision':'mixed'})
sim.context.setPositions(positions_min)
# 将约束参考坐标(x0/y0/z0)更新为最小化后的位置，避免初始张力
force_rest = None
for f in system.getForces():
    if isinstance(f, mm.CustomExternalForce):
        force_rest = f; break
if force_rest is not None:
    for i in range(force_rest.getNumParticles()):
        pi = int(force_rest.getParticleParameters(i)[0])
        xyz = positions_min[pi].value_in_unit(unit.nanometer)
        force_rest.setParticleParameters(i, pi, [xyz[0], xyz[1], xyz[2]])
    force_rest.updateParametersInContext(sim.context)
log('约束参考坐标已更新为最小化后位置')

# 阶段化温度梯度
def equilibrate(temp_k, ps, k_val):
    integ.setTemperature(temp_k*unit.kelvin)
    sim.context.setParameter('k', k_val)
    sim.context.setVelocitiesToTemperature(temp_k*unit.kelvin)
    log('== 平衡 %.0fK k=%.1f %dps ==' % (temp_k, k_val, ps))
    chunk = int(ps * 1000 / 10)
    for c in range(10):
        sim.step(chunk)
        pe = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        if not np.isfinite(pe):
            raise RuntimeError('NaN at %.0fK chunk %d' % (temp_k, c))
        log(' chunk %d/10 PE=%.0f (%.0fs elapsed)' % (c+1, pe, time.time()-T0))

log('开始温度梯度平衡')
equilibrate(50, 50, 10.0)
equilibrate(150, 50, 5.0)
equilibrate(300, 50, 2.0)
log('平衡完成')

# ---------- 4. 生产 1ns ----------
integ.setTemperature(300*unit.kelvin)
integ.setStepSize(2*unit.femtosecond)
sim.context.setParameter('k', 0.5)
sim.context.setVelocitiesToTemperature(300*unit.kelvin)
sim.reporters.append(DCDReporter(os.path.join(BASE, 'traj.dcd'), 2500))
sim.reporters.append(StateDataReporter(LOG, 5000, step=False, time=True,
    potentialEnergy=True, temperature=True, separator=', '))
t_prod = time.time()
sim.step(500000)
log('生产MD 1ns 完成, 耗时 %.0f s' % (time.time()-t_prod))

with open(os.path.join(BASE, 'MINIMIZED_OK.flag'), 'w') as fo:
    fo.write('production done at %s\n' % time.strftime('%Y-%m-%d %H:%M:%S'))
print('PRODUCTION_DONE', flush=True)

# ---------- 5. RMSD 分析（纯numpy Kabsch, 帧vs首帧, 按蛋白CA叠合） ----------
log('开始RMSD分析...')
from openmm.app import DCDFile
topo_pdb = PDBFile(os.path.join(BASE, 'topology.pdb'))
ca_idx = [a.index for a in topo_pdb.topology.atoms() if a.name == 'CA' and a.residue.name != 'MOL']
li_idx = [a.index for a in topo_pdb.topology.atoms()
          if a.residue.name == 'MOL' and not a.element is None and a.element.symbol != 'H']
with open(os.path.join(BASE, 'traj.dcd'), 'rb') as f_dcd:
    dcd = DCDFile(f_dcd, topo_pdb.topology, 2*unit.femtosecond)
    frames = [fr.copy() for fr in dcd.readModel()]
pos0 = np.array(positions_min.value_in_unit(unit.nanometer))

def kabsch_aligned_rmsd(P, Q):
    """P,Q: (N,3) nm; 返回最优叠合后RMSD(nm)"""
    pc, qc = P.mean(0), Q.mean(0)
    A, B = P - pc, Q - qc
    U, S, Vt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    aligned = (R @ A.T).T + qc
    return float(np.sqrt(np.mean(np.sum((aligned - B)**2, axis=1))))

ref_ca, ref_li = pos0[ca_idx], pos0[li_idx]
prot_rmsd, lig_rmsd = [], []
for fr in frames:
    fr = np.array(fr)
    prot_rmsd.append(kabsch_aligned_rmsd(fr[ca_idx], ref_ca) * 10.0)  # nm->Å
    # 配体: 先按CA叠合该帧到参考帧再算配体位移
    pc, qc = fr[ca_idx].mean(0), ref_ca.mean(0)
    A, B = fr[ca_idx]-pc, ref_ca-qc
    U, S, Vt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0,1.0,d]) @ U.T
    fr_aligned = ((R @ (fr-pc).T).T + qc)
    lig_rmsd.append(float(np.sqrt(np.mean(np.sum((fr_aligned[li_idx]-ref_li)**2,axis=1)))) * 10.0)

lig_rmsd = np.array(lig_rmsd); prot_rmsd = np.array(prot_rmsd)
n_fr = len(lig_rmsd)
log('轨迹帧数(每25ps):', n_fr)
log('配体 RMSD(Å): mean %.2f | final %.2f | max %.2f' % (lig_rmsd.mean(), lig_rmsd[-1], lig_rmsd.max()))
log('蛋白Cα RMSD(Å): mean %.2f | final %.2f' % (prot_rmsd.mean(), prot_rmsd[-1]))
np.save(os.path.join(BASE,'lig_rmsd.npy'), lig_rmsd)   # 单位Å
np.save(os.path.join(BASE,'prot_rmsd.npy'), prot_rmsd)
with open(os.path.join(BASE,'summary.txt'),'w') as fo:
    fo.write('frames=%d\nframe_interval_ps=25\n'
             'lig_rmds_A_mean=%.2f\nlig_rmsd_A_final=%.2f\nlig_rmsd_A_max=%.2f\n'
             'prot_ca_rmsd_A_mean=%.2f\nprot_ca_rmsd_A_final=%.2f\n'
             'gpu=RTX2080\nproduction_ns=1.0\n' %
             (n_fr, lig_rmsd.mean(), lig_rmsd[-1], lig_rmsd.max(),
              prot_rmsd.mean(), prot_rmsd[-1]))
print('ANALYSIS_DONE', flush=True)
log('DONE_ALL')
