# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 13:55:40 2023

@author: n.patsalidis
"""
import argparse
import sys
sys.path.append('../')
import md_pipeline as mdp 
import numpy as np

def main():
    help_trajf = "trajectory file(s). set --> average, list/tuple --> wrapping, dictionary --> key:result pairs"
    help_property = "The property to be computed, e.g. density_profile, pair_distribution, Sq, segmental_dynamics etc .."
    help_topol = "Topology file (.gro,.ltop (lammps .dat file))"
    help_memory_demanding= "If True then each frame is read from the disk, necessary computations are done and then is deleted"
    help_connectivity=".itp or list of itp(s) / .ltop file"
    help_keep_frames = "2 integers. First is the frame number to start and second the frame number to end"
    help_conftype = "Type of confinment. Providing this the algorithm assumes we have a confined system"
    help_polymer_method = "Method to understand which is polymer. Default is 'molname'"
    help_particle_method = "Method to understand which is particle. Defauslt is 'molname'"
    help_polymer = "Polymer name or molecule ids or atom type or atom ids"
    help_particle = "Particle name or molecule ids or atom type or atom ids"
    help_adsorption_interval = "tuple or list of tuples. Denotes the distance intervale between particle and polymer in which the polymer is considered adsorbed"
    help_cylinder_length = "The length of the (finite) cylinder."
    help_ztrain = "when there is a (finite) cylindrical confinment this denotes the adsorption distance at the zdirection (above and below the cylinder)"
    help_property_args = "tuple: arguments for the 'property' to be computed"
    help_property_kwargs = "dictionary: keyword arguments for the property to be computed" 
    help_result_file = "prefix of the name of the file that the data will be stored"
    adddef = " [  default: %(default)s ]"
    
    argparser = argparse.ArgumentParser(description="analyze your system(s)")
    argparser.add_argument('-p',"--property",metavar=None,
            type=str, required=True, help=help_property)
    argparser.add_argument('-f',"--trajf",metavar=None,
            type=str, required=True, help=help_trajf)
    argparser.add_argument('-t',"--topol",metavar=None,
            type=str, required=True, help=help_topol)
    argparser.add_argument('-c',"--connectivity",metavar=None,
            type=str, required=True, help=help_connectivity)
    argparser.add_argument('-mem',"--memory_demanding",metavar=None,
            type=bool, required=False, help=help_memory_demanding)
    argparser.add_argument('-kf',"--keep_frames",metavar=None,
            type=int, required=False,nargs=2, help=help_keep_frames)

    argparser.add_argument('-ct',"--conftype",metavar=None,
            type=str, required=False, help=help_conftype)
    argparser.add_argument('-a',"--adsorption_interval",metavar=None,
            type=str, required=False, help=help_adsorption_interval)
    argparser.add_argument('-pam',"--particle_method",metavar=None,
            type=str, required=False, help=help_particle_method)
    argparser.add_argument('-pom',"--polymer_method",metavar=None,
            type=str, required=False, help=help_polymer_method)
    argparser.add_argument('-particle',"--particle",metavar=None,
            type=str, required=False, help=help_particle)
    argparser.add_argument('-polymer',"--polymer",metavar=None,
            type=str, required=False, help=help_polymer)
    argparser.add_argument('-cl',"--cylinder_length",metavar=None,
            type=float, required=False, help=help_cylinder_length)
    argparser.add_argument('-ztrain',"--ztrain",metavar=None,
            type=float, required=False, help=help_ztrain)
    argparser.add_argument('-pargs',"--property_args",metavar=None,
            type=str, required=False,default='tuple()', help=help_property_args)
    argparser.add_argument('-pkwargs',"--property_kwargs",metavar=None,
            type=str, required=False,default='dict()', help=help_property_kwargs)
    
    argparser.add_argument('-rf',"--result_file",metavar=None,
             required=False,default=None, help=help_result_file)
    
    parsed_args = argparser.parse_args()
    
    known_init_kwargs = ['memory_demanding','keep_frames','conftype',
                         'adsorption_interval','particle_method','polymer_method',
                        'train_custom_method','polymer','particle']
    
    evaluated_strings = ['trajf','connectivity','polymer','particle',
                         'train_costum_method','adsorption_interval',
                         'property_args','property_kwargs']
    margs = dict()
    init_kwargs = dict()
    for s in vars(parsed_args):
        
        attr = getattr( parsed_args,s)
        if attr is None:
            continue
        if s in evaluated_strings:
            try:
                margs[s] = eval(attr)
            except NameError:
                margs[s] = attr
            except SyntaxError:
                margs[s] = attr
        else:
            margs[s] = attr
        if s in known_init_kwargs:
            if s in evaluated_strings:
                try:
                    init_kwargs[s] = eval(attr)
                except NameError:
                    init_kwargs[s] = attr
            else:
                init_kwargs[s] = attr
        
    
    for k,m in margs.items():
        print(k,type(m),m)
    
    supra = mdp.supraClass(margs['topol'],margs['connectivity'],**init_kwargs)
    data = supra.get_property(margs['trajf'],margs['property'],
                              *margs['property_args'],
                              **margs['property_kwargs'])
    
    if parsed_args.result_file is not None:
        fname = parsed_args.result_file.split('.pickle')[0]
        fname = '{:s}-{:s}.pickle'.format(fname,margs['property'])
    else:
        fname = '{:s}.pickle'.format(margs['property'])
    mdp.ass.save_data(data,fname)
    return

if __name__=='__main__':
    main()
