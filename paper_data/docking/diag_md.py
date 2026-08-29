#!/usr/bin/env python
"""诊断: 定位NaN粒子 + 测试不同CUDA精度"""
import os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import openmm as mm
from openmm import unit
from openmm.app import (PDBFile, ForceField, Modeller, Simulation, HBonds, CutoffPeriodic)

BASE='/root/case3_md'
ff = ForceField('amber14/protein.ff14SB.xml','amber14/tip3p.xml')
from openmmforcefields.generators import GAFFTemplateGenerator
from openff.toolkit import Molecule as OFFMolecule
lig = OFFMolecule.from_smiles('Fc1c[nH]c(=O)[nH]c1=O', allow_undefined_stereo=True); lig.name='MOL'
try: lig.assign_partial_charges(partial_charge_method='gasteiger')
except Exception: pass
ff.registerTemplateGenerator(GAFFTemplateGenerator(molecules=[lig], forcefield='gaff-2.11').generator)

rec=PDBFile(f'{BASE}/6LU7_prot.pdb'); lgd=PDBFile(f'{BASE}/5fu_pose.pdb')
mod=Modeller(rec.topology,rec.positions); mod.addHydrogens(forcefield=ff,pH=7.4)
lm=Modeller(lgd.topology,lgd.positions); mod.add(lm.topology,lm.positions)
mod.addSolvent(ff,model='tip3p',padding=1.0*unit.nanometer)
print('atoms:',mod.topology.getNumAtoms(),flush=True)

platform=mm.Platform.getPlatformByName('CUDA')
for prec in ('mixed','double'):
    print(f'===== CUDA Precision={prec} =====',flush=True)
    system=ff.createSystem(mod.topology,nonbondedMethod=CutoffPeriodic,
                           nonbondedCutoff=1.0*unit.nanometer,constraints=HBonds,rigidWater=True)
    rest=mm.CustomExternalForce('0.5*k*periodicdistance(x,y,z,x0,y0,z0)^2')
    rest.addGlobalParameter('k',10*unit.kilocalorie_per_mole/unit.angstrom**2)
    for p in ('x0','y0','z0'): rest.addPerParticleParameter(p)
    for a in mod.topology.atoms():
        if (a.name=='CA' and a.residue.name!='MOL') or (a.residue.name=='MOL' and not a.name.startswith('H')):
            rest.addParticle(a.index,mod.positions[a.index])
    system.addForce(rest)
    integ=mm.LangevinMiddleIntegrator(50*unit.kelvin,1/unit.picosecond,1*unit.femtosecond)
    integ.setConstraintTolerance(1e-8)
    sim=Simulation(mod.topology,system,integ,platform,{'Precision':prec})
    sim.context.setPositions(mod.positions)
    e0=sim.context.getState(getEnergy=True).getPotentialEnergy()
    print('E_min=',e0,flush=True)
    sim.minimizeEnergy(maxIterations=20000,tolerance=1*unit.kilojoule_per_mole/unit.nanometer)
    st=sim.context.getState(getEnergy=True,getForces=True)
    fmax=np.max(np.linalg.norm(st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer),axis=1))
    w=np.argsort(-np.linalg.norm(st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer),axis=1))[:5]
    atom_list=list(mod.topology.atoms())
    print('after min E=',st.getPotentialEnergy(),' Fmax=%.1f'%fmax,' top-force atoms:',[ (atom_list[int(i)].name,int(i),atom_list[int(i)].residue.id) for i in w],flush=True)
    sim.context.setVelocitiesToTemperature(50*unit.kelvin)
    ok=True
    try:
        for c in range(15):
            integ.step(1000)  # 1ps每段
            pe=sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            print(' t=%dps PE=%.0f'%(c+1,pe),flush=True)
            if not np.isfinite(pe): raise RuntimeError('NaN')
    except Exception as ex:
        # 找NaN粒子
        pos=sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        bad=np.where(~np.isfinite(pos).all(axis=1))[0]
        print('FAIL:',ex,' bad atoms:',len(bad))
        for i in bad[:10]:
            at=atom_list[int(i)]
            print('  ',at,at.residue,flush=True)
        ok=False
    if ok:
        print(f'PRECISION {prec}: STABLE_15PS',flush=True)
        break
