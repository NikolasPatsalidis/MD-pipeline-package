# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 10:50:42 2023

@author: n.patsalidis
"""

import sys
sys.path.append('../')
import md_pipeline as mdp  
import numpy as np
import os
from time import perf_counter
import random

def Rg(coords):
    rm = np.mean(coords,axis=0)
    r = coords -rm
    r3 = 0
    for i in range(r.shape[0]):
        r3 += np.dot(r[i],r[i])
    rg = np.sqrt(r3/r.shape[0])
    return rg
def write_bash(fname,sysname):
    lines=['#!/bin/bash',
    'rm tp.tpr',
    'gmx grompp -c {0:s}.gro  -f steep.mdp -p {0:s}.top -o tp.tpr -maxwarn 3'.format(sysname) ,
    'gmx mdrun -s tp.tpr -ntmpi 8 -ntomp 1' ,
    'cp confout.gro minimization.gro',
    '#rm tp.tpr',
    '#gmx grompp -c confout.gro  -f gromp.mdp -p {:s}.top -o tp.tpr'.format(sysname) ,
    '#gmx mdrun -s tp.tpr',] 
    with open(fname,'w') as f:
        for line in lines:
            f.write('{:s}\n'.format(line))
        f.close()
    return
def write_n_run(mol,num):
    '''
    writes the topology and force field paramters and then runs gromacs minimization
    '''

    if True:
        #sil.element_based_matching([('O02R','SiR'),('Si3', 'O02R', 'SiR'),('O02R', 'SiR', 'Ob'),('O02R', 'SiR', 'Onb')])
        sil.element_based_matching(sil.nonexisting_types('bond')['inff'])
        sil.element_based_matching(sil.nonexisting_types('angle')['inff'])
    else:
        if mol in ['MPTES','TESPD']:
            sil.match_types([ ('O02R','SiR'), ('Si3', 'O02R', 'SiR'), ('O02R', 'SiR', 'Ob'),('O02R', 'SiR', 'Onb') ],
                             [ ('Ob','Si'),  ('Si', 'Ob', 'Si') ,      ('Ob', 'Si', 'Ob'),    ('Ob', 'Si', 'Ob') ] )
            #sil.ff.bondtypes[('O02R','SiR')] = sil.ff.bondtypes[('Ob','Si')]
            #sil.ff.angletypes[('Si3', 'O02R', 'SiR')] = sil.ff.angletypes[('Si', 'Ob', 'Si')]
            #sil.ff.angletypes[('O02R', 'SiR', 'Ob')] = sil.ff.angletypes[('Ob', 'Si', 'Ob')]
            #sil.ff.angletypes[('O02R', 'SiR', 'Onb')] = sil.ff.angletypes[('Ob', 'Si', 'Ob')]  
        else:
            sil.ff.bondtypes[('O0KR','SiR')] = sil.ff.bondtypes[('Ob','Si')]
            sil.ff.angletypes[('SiD', 'O0KR', 'SiR')] = sil.ff.angletypes[('Si', 'Ob', 'Si')]
            sil.ff.angletypes[('O0KR', 'SiR', 'Ob')] = sil.ff.angletypes[('Ob', 'Si', 'Ob')]
            sil.ff.angletypes[('O0KR', 'SiR', 'Onb')] = sil.ff.angletypes[('Ob', 'Si', 'Ob')]
    sil.filter_ff()
    sil.clean_dihedrals_from_topol_based_on_ff()
    
    sysname = 'S{:s}{:d}'.format(mol[0],num)
    sil.exclusions_map[sysname] = 2
    sil.exclusions = sil.pairs # this is a trick to not count twice the 1-4 interactions
    sil.mol_names[:] = sysname
    sil.mol_ids[:] = 1
    
    
    tw = perf_counter()
    sil.write_topfile('{:s}/{:s}'.format(savepath,sysname),opls_convection=True)
    sil.write_gro_file('{:s}/{:s}.gro'.format(savepath,sysname))
    
    write_bash('{:s}/{:s}.sh'.format(savepath,sysname),sysname)
    print('writing time = {:.3e} sec'.format(perf_counter()-tw))
    com1 = 'cp setup_eq_functionilization/* {:s}'.format(savepath)
    com2 = 'cd {:s}'.format(savepath)
    com3 = 'bash {:s}.sh'.format(sysname)
    
    
    tr = perf_counter()
    command = ' ; '.join([com1, com2, com3])
    print(command)
    os.system(command)
    #os.system('cp visu1.vmd {:s}/'.format(savepath) )
    print('Running time = {:.3e} sec'.format(perf_counter()-tr))
    return 


####### Main code ###
#Settings
########################################################################################
####################################    
# Bond types that react per molecule
#silc = 'slabs/particle.gro' # silica initial gro file
#silitp='slabs/particle.itp' # silica initial itp file
#siltop = 'slabs/particle.top' # silica initial top file
storing_prefix='newmix' # just for storing purposes
silc = 'slabs/smallslab.gro' # silica initial gro file
silitp='slabs/smallslab.top' # silica initial itp file
siltop = 'slabs/smallslab.top' # silica initial top file
substrate_shape = 'flat'  # flat or shperical
bonds = {'MPTES':('C01','O02'),'TESPD':('C01','O02'),'NXT':('C0M','O0K')}
# Bond on Silica to react
c = ('Onb','Si')
# method to graft with a sufficiently uniform distribution (finding the prober reaction cites)
cite_method = 'uniform'  # 'random'
grid=(3,3) # grid imposing uniformity
# every how much molecules to write the topology and structure
nwrite=1 

# keyward arguments per method of finding the reaction cites
cite_method_kwargs = {'random':dict(),
                      'separation_distance':dict(separation_type='SiR',separation_distance=0.8),
                      'uniform':dict(separation_type='SiR',separation_distance=0.8,grid=grid)
                      } 
# Sudo potentials used for initial placement (this is to avoid overlaps)
morse_bond={'NXT':(100,0.16,2), # De,re ,alpha
            'TESPD':(100,0.16,2),# De,re ,alpha
            'MPTES':(100,0.16,2)}# De,re ,alpha
morse_overlaps = {'NXT':(0.2,5),
                'TESPD':(0.2,5),
                'MPTES':(0.2,5),
                 }
# the maximum target grafting density in molecules/nm^2
mols_rho = {'MPTES':0.7,'NXT':0.5,'TESPD':0.3}
###############################################################################
###################################################################################
####################################################################################

############################# Main code #########################################
sil = mdp.Analysis(silc,silitp,fftop=siltop)
sil.read_file(sil.topol_file)

if substrate_shape =='flat':
    box = sil.get_box(0)
    surf_area = 2*box[0]*box[1]
elif substate_shape=='spherical':
    cds = sil.get_coords(0)
    radius = np.mean(np.max(cds,axis=0) -np.min(cds,axis=0))/2.0
    surf_area = 4*np.pi*radius**2
else:
    raise NotImplementedError('Only "spherical" and "flat" shapes are supported at the moment')
mols_num = {k:int(round(surf_area*v,0)) for k,v in mols_rho.items()}
mollist = []
for m,v in mols_num.items():
    mollist += [m]*v
random.shuffle(mollist)

savepath = 'SIL-{:s}'.format(storing_prefix)
mdp.ass.make_dir(savepath)
print(mollist)

for i,mol in enumerate(mollist):

    
    
    
    
    
    coupling_agent = mdp.Analysis('ligpargen/{:s}.gro'.format(mol),
                                  'ligpargen/{:s}.itp'.format(mol))
    #raise
    coupling_agent.read_file(coupling_agent.topol_file)
    
    rcut = 4*Rg(coupling_agent.get_coords(0))
    
    print('Cutoff used for overlap calculations = {:4.6f}'.format(rcut))

    t0 = perf_counter()
    print('Adding molecule {:s} {:d}/{:d}'.format(mol,i+1,len(mollist)))
    
    coupling_agent = mdp.Analysis('ligpargen/{:s}.gro'.format(mol),
                              'ligpargen/{:s}.itp'.format(mol))
    coupling_agent.read_file(coupling_agent.topol_file)

    a = mdp.React_two_systems(sil,coupling_agent,
                              c,bonds[mol],
                              react1=1,react2=1, # which type participates in the reaction, 0 or 1
                              seed1=None,seed2=None,  # for random generations
                              rcut=rcut,  # cut of radius to compute sudo potentials
                              shape=substrate_shape, #shape of particle, for now just adding "flat" will make any difference
                              morse_bond = morse_bond[mol], # sudopotential for the new bond created
                              morse_overlaps = morse_overlaps[mol], # Repulsive sudopotential for  the rest
                              cite_method = cite_method, # for now separation distance is good
                              cite_method_kwargs=cite_method_kwargs[cite_method],# keyward arguments for the method of choosing cites
                              iapp=mol[0]) # append on atom types and names to ensure they don't overlap 
    tf = perf_counter() - t0
    print(' added {:d} molecules time --> {:.3e}  sec  time/mol = {:.3e} sec/mol'.format(i+1,tf,tf/(i+1)))
    if (i+1) % nwrite ==0:
        sysname = 'S{:s}{:d}'.format(mol[0],i+1) #giving a name to the system
        write_n_run(mol,i+1)  # write and run the system 
        gro_file = '{:s}/confout.gro'.format(savepath)  # read the new silica
        del sil.timeframes[0]
        sil.read_file(gro_file)
        subsys = '{:s}/{:s}'.format(savepath,sysname)
        mdp.ass.make_dir(subsys)
        os.system('mv {:s}/*.* {:s}'.format(savepath,subsys))

#wrn_run(mol,num)
    

################################################################################################
################################################################################################
