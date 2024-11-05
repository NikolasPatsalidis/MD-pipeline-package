import numpy as np
import sys
sys.path.append('../')
import md_pipeline as mdp
import argparse

def main():
    ht1='gro file bulk'
    ht2 = 'gro file substrate'
    hitp1='itp files 1 as list'
    hitp2 ='itp  files 2 as list'
    htop1='top1 file'
    htop2 ='top2  file'
    hincl='inclusion files as list'
    hd='distance of bulk from substrate'
    hpath='path to write the files'
    hmult='multiply system'
    argparser = argparse.ArgumentParser(description="Merge bulk to substrate. Bulk must be in unwrapped coordinates")
    
    
    argparser.add_argument('-c1',"--gro1",metavar=None,
            type=str, required=True, help=ht1)
    argparser.add_argument('-c2',"--gro2",metavar=None,
            type=str, required=True, help=ht2)
    argparser.add_argument('-itp1',"--itp1",metavar=None,
            type=str, required=True, help=hitp1)
    argparser.add_argument('-itp2',"--itp2",metavar=None,
            type=str, required=True, help=hitp2)
    argparser.add_argument('-top1',"--top1",metavar=None,
            type=str, required=True, help=htop1)
    argparser.add_argument('-top2',"--top2",metavar=None,
            type=str, required=True, help=htop2)
    argparser.add_argument('-i',"--incl",metavar=None,
            type=str, required=False,default="[]", help=hincl)
    argparser.add_argument('-d',"--distance",metavar=None,
            type=float, default = 0.5, required=False, help=hd)
    argparser.add_argument('-p',"--path",metavar=None,
            type=str, required=True, help=hpath)
    
    argparser.add_argument('-m',"--mult",metavar=None,
            type=int, required=False,default=1, help=hmult)

    parsed_args = argparser.parse_args()
    
    
    grofile1 = parsed_args.gro1 
    
    itp1 = eval(parsed_args.itp1)

    grofile2= parsed_args.gro2   
    
    itp2= eval(parsed_args.itp2)
    
    path = parsed_args.path 
    
    
    incl = eval(parsed_args.incl)

    top1=parsed_args.top1
    top2=parsed_args.top2
    
    merge_n_translate( path, grofile1,grofile2,itp1,itp2,top1,top2,incl,
            bulk_translation=parsed_args.distance,
            mult=(0,0,parsed_args.mult) )
    
    return

def  merge_n_translate(path,gro1,gro2,itp1,itp2,top1,top2,incl,
        bulk_translation=0.5,
        mult=(0,0,1)):
    import copy
    global bulk
    global substrate
    wpath = '{:s}'.format(path)
    mdp.ass.make_dir(wpath)
    
    bulk = mdp.Analysis(gro1,itp1,fftop=top1)
    substrate = mdp.Analysis(gro2,itp2,fftop=top2)
    
    bulk.read_file(gro1)

    bulk.multiply_periodic(mult)

    substrate.read_file(gro2)
    
    boxa = substrate.get_box(0)
    boxb = bulk.get_box(0)
    box =  np.maximum(boxa,boxb)
    
    
    substratecm = mdp.CM(substrate.get_coords(0),substrate.atom_mass)
    bulkcm = mdp.CM(bulk.get_coords(0),bulk.atom_mass)
    bulk.timeframes[0]['coords']+=box/2-bulkcm
    substrate.timeframes[0]['coords']+=box/2-substratecm
    substratecm = mdp.CM(substrate.get_coords(0),substrate.atom_mass)
    
    merged = copy.deepcopy(bulk)
    merged.timeframes[0]['coords'][:,2]+=-bulk.get_coords(0)[:,2].min()+substrate.get_coords(0)[:,2].max()+bulk_translation
    
    
    merged.merge_system(substrate,add='s')
    box[2] = merged.get_coords(0)[:,2].max()-merged.get_coords(0)[:,2].min()+bulk_translation
    merged.timeframes[0]['boxsize'] = box
    
    merged.timeframes[0]['coords'][:,2] += box[2]/2-substratecm[2]
    
    merged.write_gro_file('{:s}/merged.gro'.format(wpath))
    merged.write_topfile('{:s}/topol.top'.format(wpath),
                         includes=incl)
    
    return merged

if __name__=='__main__':
    main()

