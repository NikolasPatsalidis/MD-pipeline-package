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
#from time import perf_counter
#setup

import argparse

def main():
    parser = argparse.ArgumentParser(description='Process concentration and accelerator arguments.')
    
    parser.add_argument('-a','--acc', type=str,nargs='+', help='Specify the accelerator type.')
    parser.add_argument('-c','--conc', type=float,nargs='+', help='Specify the accelerator concentration')
    parser.add_argument('-n','--numbers', type=int,nargs='+', help='Specify the number of  accelerators')
    parser.add_argument('-notvoid','--notvoid', type=int,default=0, help='avoid part 1. Default is 0, which means do not avoid part 1 (creating the voids)')
    parser.add_argument('-ipath','--init_path', type=str,default='inits/cPI30--bare', help='path of initial system')
    parser.add_argument('-p','--positions', type=float,nargs='+', help='positions to inset; must be equal to 3n where n is the number of molecules')
    parser.add_argument('-r','--restrict', type=str,default='no', help='method to restrict, available is center and group names')
    parser.add_argument('-rd','--restrict_direction', type=str,default='z', help='restrict direction to measure the distance')
    parser.add_argument('-rl','--restrict_distance_low', type=float,default=0, help='restrict distance low')
    parser.add_argument('-rh','--restrict_distance_high', type=float,default=0, help='restrict distance high')
    args = parser.parse_args()
    if args.numbers is None and args.conc is None:
        raise Exception('give either concentrations or numbers')
    if args.numbers is None:
        numbers  = [0 for j in range(len(args.conc)) ]
    else:
        numbers = args.numbers
    if args.conc is None:
        conc = [0.0 for j in range(len(numbers)) ]
    else:
        conc = args.conc
    try:
        s = '-'.join(  ['{:s}{:3.2f}'.format(a,c) for a,c in zip(args.acc,args.conc) ] )
    except:

        s = '-'.join(  ['{:s}n{:d}'.format(a,n) for a,n in zip(args.acc,args.numbers) ] )
    init_path = args.init_path
    createpath ='{:s}/{:s}'.format(init_path.split('/')[-1],s)
    initgro = '{:s}/ceq.gro'.format(init_path)
    itpfiles = ['{:s}/{:s}'.format(init_path,f) for f in os.listdir(init_path) if '.itp' in f ] 
    for i in itpfiles:
        if 'cPI30.itp' not in i:
            silname= i.split('/')[-1].split('.')[0]
    #print(itpfiles)
    ##################################
    bublepath = '{:s}/buble'.format(createpath)
    mdp.ass.make_dir(createpath)
    mdp.ass.make_dir(bublepath)
    
    trajf = initgro
    
    init = mdp.Analysis(initgro,itpfiles,fftop='{:s}/topol.top'.format(init_path))
    init.read_file(trajf)
    box = init.get_box(0)
    cm2sil = init.get_coords(0)[:,2][init.mol_names==silname].mean()
    mass_cpi = init.atom_mass [ init.mol_names =='cPI30'].sum()
    
    numacc=0
    numperacc = dict()
    objsacc = dict()
    for a,c,n in zip(args.acc,conc,numbers):
        
        gro_acc = 'ligpargen/{0:s}.gro'.format(a)
        itp_acc= 'ligpargen/{0:s}.itp'.format(a)
        
        accelerator = mdp.Analysis(gro_acc,itp_acc)         
    
        accelerator.read_file(gro_acc)
        
        objsacc[a] = accelerator

        macc = accelerator.total_mass
        num = max(int(round(c*mass_cpi/macc/100 ,0)),n)
        numacc += num
        numperacc[a] = num
        ce = num*macc/mass_cpi*100
        print('adding {:d} {:s} molecules --> exact concentrations = {:4.3f}'.format(num,a,ce))
    
    p = args.positions
    if p is  None:
        pass
    else:
        if len(p)*3 != numacc:
            raise Exception('You specified positions {:d} but the total number of accelerators is {:d},\
                    position should be 3n where n is the molecule number'.format(len(p),numacc ))
    #add sudo atoms
    if args.notvoid==0:
        print(numacc)
        sudo = mdp.add_sudo_atoms(init,numacc,0.001,positions=p,
                r = args.restrict,
                rd=args.restrict_direction,
                rl=args.restrict_distance_low,
                rh=args.restrict_distance_high)
    
        init.ff.add_posres({'by':'at_types','val':'SUDO','r':0.,'k':1000000})
    
        print('Writing file with sudo-atoms')
        init.write_topfile('{:s}/topol'.format(bublepath),includes=['oplsaa.ff/forcefield.itp'])
        init.write_gro_file('{:s}/piS.gro'.format(bublepath))
        print('Executing void creation via gromacs')
        commands = ['cd ' + bublepath,
            'cp -r ../../../setupbuble/* . ',
            'bash makebuble.sh ',
            ]
        os.system(' ; '.join(commands))
    print('Reading file with voids')
    #read again and remove the sudos
    su = mdp.Analysis('{:s}/piS.gro'.format(bublepath),
                      itpfiles+ ['{:s}/MTOP.itp'.format(bublepath)],fftop='{:s}/topol.top'.format(bublepath))
    su.read_file('{:s}/confout.gro'.format(bublepath))
    sudo_coords = su.get_coords(0)[su.at_types=='SUDO']
    
    alphabet={0:'' , 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i',
            10: 'j', 11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r',
            19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z'}

    print('Inserting atoms in the voids')
    j=0
    for l,(a,num) in enumerate(numperacc.items()):
        obj = objsacc[a]
        b = alphabet[l]
        for i in range(num):
            cj = sudo_coords[j]
            ca = obj.get_coords(0)
            cm = np.mean(ca,axis=0)
            obj.timeframes[0]['coords'] = ca-cm + cj
            su.merge_system(obj,add=b)
            j+=1
    su.filter_system(su.at_types != 'SUDO')
    
    eqpath = '{:s}/wacc'.format(createpath)
    mdp.ass.make_dir(eqpath)
    su.write_topfile('{:s}/topol.top'.format(eqpath),includes=['oplsaa.ff/forcefield.itp'])
    su.write_gro_file('{:s}/wacc.gro'.format(eqpath))
    #
    com = commands = ['cd ' + eqpath,
                'cp -r ../../../setupequil/* . ',
                'bash loc.sh ',
                ]
    print('executing',com)
    os.system(' ; '.join(com))
if __name__ == '__main__':
    main()
