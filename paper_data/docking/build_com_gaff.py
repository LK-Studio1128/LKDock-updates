#!/usr/bin/env python
"""Build COM with gaff2-typed ligand; verify no protein-parameter pollution"""
import subprocess, sys

leap = '''source leaprc.protein.ff14SB
source leaprc.gaff2
loadamberparams lig_gaff.frcmod
LIG = loadmol2 lig_gaff.mol2
REC = loadpdb rec_heavy.pdb
COM = combine { REC LIG }
saveamberparm COM COM_g.prmtop COM_g.inpcrd
quit
'''
open('leap_com4.in','w').write(leap)
r1 = subprocess.run('tleap -f leap_com4.in > leap_com4.log 2>&1', shell=True)
log = open('leap_com4.log').read()
for line in log.splitlines():
    if 'Errors' in line or 'FATAL' in line:
        print(line)

import parmed as pmd
from collections import Counter
c = pmd.load_file('COM_g.prmtop')
r = pmd.load_file('REC_full.prmtop')
co_c = [b.type.k for b in c.bonds if {b.atom1.name,b.atom2.name}=={'C','O'} and b.atom1.residue.name!='MOL'][:5]
co_r = [b.type.k for b in r.bonds if {b.atom1.name,b.atom2.name}=={'C','O'}][:5]
print('COM protein C=O k:', [round(x,1) for x in co_c])
print('REC         C=O k:', [round(x,1) for x in co_r])

def bs(p, mol):
    s = Counter()
    for b in p.bonds:
        has_mol = 'MOL' in (b.atom1.residue.name, b.atom2.residue.name)
        if has_mol != mol: continue
        key = tuple(sorted([(b.atom1.residue.idx,b.atom1.name),(b.atom2.residue.idx,b.atom2.name)])) + (round(b.type.k,2), round(b.type.req,3))
        s[key]+=1
    return s
sc, sr = bs(c,False), bs(r,False)
print('protein bonds diff: COM-only', sum((sc-sr).values()), '| REC-only', sum((sr-sc).values()))
print('atoms:', len(c.atoms))
print('BUILD_OK' if sum((sc-sr).values())==0 else 'STILL_POLLUTED')
