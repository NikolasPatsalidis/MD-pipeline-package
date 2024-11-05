import numpy as np
from matplotlib import pyplot as plt

from time import perf_counter
from scipy.optimize import minimize #,dual_annealing
from numba import jit,prange
from numba.experimental import jitclass
import os 
import inspect
import collections
import six
from scipy.integrate import simpson
from pytrr import GroTrrReader
from scipy.optimize import dual_annealing,differential_evolution
import pandas as pd
import logging
import coloredlogs
import matplotlib

import lammpsreader 

import pickle
@jit(nopython=True,fastmath=True)
def compute_residual(pars,A,y):
    w = pars[:-1]
    bias = pars[-1]
    r = np.dot(A,w) + bias - y
    return np.sqrt(np.sum(r*r)/y.size)

@jit(nopython=True,fastmath=True)
def dCRdw(pars,A,y):
    dw = np.zeros(pars.shape)
    
    n = y.size
    w = pars[:-1]
    bias = pars[-1]
  
    r = np.dot(A,w)+ bias - y
    c = np.sqrt(np.sum(r*r)/y.size)

    for j in range(w.size):
        for i in range(n):
            dw[j] += r[i]*A[i,j]

    dw[-1] = np.sum(r) 
    dw *= 1.0/n/c
    
    return dw

@jit(nopython=True,fastmath=True)
def constraint(pars,A,y,residual):
    return residual - compute_residual(pars,A,y)   


@jit(nopython=True,fastmath=True)
def dCdw(pars,A,y,residual):
    dw = - dCRdw(pars,A,y)
    return dw

@jit(nopython=True,fastmath=True)
def smoothness(pars,dlogtau):
    w = pars[:-1]
    x = np.empty_like(w)
    n = w.size-1
    for j in range(1,n):
        x[j] = w[j+1]-2.0*w[j] +w[j-1]
    x[0] = w[2]-2*w[1]+w[0]
    
    x[n] = w[n]-2*w[n-1]+w[n-2]
    
    return  np.sum(x*x)/dlogtau**3 

@jit(nopython=True,fastmath=True)
def dSdw(pars,dlogtau):
    
    w = pars[:-1]
    #print(pars.size,w.size)
    dw = np.empty_like(pars)
    
    for j in range(2,w.size-2):
        dw[j] = 12*w[j]-8*(w[j+1] + w[j-1])+2*(w[j+2]+w[j-2])
    dw[0] = 2*(w[2]-2*w[1]+w[0])
    dw[1] = 18*w[1]-12*w[2]-8*w[0]+2*w[3]
    
    n = w.size-1
    dw[n] = 2*(w[n-2]-2*w[n-1]+w[n])
    dw[n-1] = 18*w[n-1]-12*w[n-2]-8*w[n]+2*w[n-3]
    
    dw[:-1] = dw[:-1]/dlogtau**3 
    dw[-1] = 0
    return dw

@jit(nopython=True,fastmath=True)
def Trelax(w,fi):
    return np.sum(w*fi)

@jit(nopython=True,fastmath=True)
def Frelax(w,fi):
    return np.sum(w/fi)

@jit(nopython=True,fastmath=True)
def FrelaxCost(w,fi):
    contr = Frelax(w[:-1],fi)
    
    return contr

@jit(nopython=True,fastmath=True)
def dFdw(w,fi):
    d  = np.empty_like(w)
    d[:-1] = 1.0/fi
    d[-1] = 0
    return d
@jit(nopython=True,fastmath=True)
def FrelaxCon(w,fi,target):
    return target - np.sum(w[:-1]/fi)
@jit(nopython=True,fastmath=True)
def dFCdw(w,fi,target):
    d  = np.empty_like(w)
    d[:-1] = -1.0/fi
    d[-1] = 0
    return d

@jit(nopython=True,fastmath=True)
def L2(w):
    return np.sum(w*w)/w.size
@jit(nopython=True,fastmath=True)
def dL2dw(w):
    return 2*w/w.size
@jit(nopython=True,fastmath=True)
def FitCost(pars,dlogtau):
    sm = smoothness(pars,dlogtau)
    return  sm
 


@jit(nopython=True,fastmath=True)
def dFitCostdw(pars,dlogtau):
    dL = dSdw(pars,dlogtau)   
    return  dL 




class logs():
    '''
    A class for modifying the loggers
    '''
    def __init__(self):
        self.logger = self.get_logger()
        
    def get_logger(self):
    
        LOGGING_LEVEL = logging.CRITICAL
        
        logger = logging.getLogger(__name__)
        logger.setLevel(LOGGING_LEVEL)
        logFormat = '%(asctime)s\n[ %(levelname)s ]\n[%(filename)s -> %(funcName)s() -> line %(lineno)s]\n%(message)s\n --------'
        formatter = logging.Formatter(logFormat)
        

        if not logger.hasHandlers():
            logfile_handler = logging.FileHandler('piplib.log',mode='w')
            logfile_handler.setFormatter(formatter)
            logger.addHandler(logfile_handler)
            self.log_file = logfile_handler
            stream = logging.StreamHandler()
            stream.setLevel(logging.CRITICAL)
            stream.setFormatter(formatter)
            
            logger.addHandler(stream)
         
        fieldstyle = {'asctime': {'color': 'magenta'},
                      'levelname': {'bold': True, 'color': 'green'},
                      'filename':{'color':'green'},
                      'funcName':{'color':'green'},
                      'lineno':{'color':'green'}}
                                           
        levelstyles = {'critical': {'bold': True, 'color': 'red'},
                       'debug': {'color': 'blue'}, 
                       'error': {'color': 'red'}, 
                       'info': {'color':'cyan'},
                       'warning': {'color': 'yellow'}}
        
        coloredlogs.install(level=LOGGING_LEVEL,
                            logger=logger,
                            fmt=logFormat,
                            datefmt='%H:%M:%S',
                            field_styles=fieldstyle,
                            level_styles=levelstyles)
        return logger
    
    def __del__(self):
        self.logger.handlers.clear()
        try:
            self.log_file.close()
        except AttributeError:
            pass
        

logobj = logs()        
logger = logobj.logger


class gromacsTop():
    def __init__(self,analysis_system,funmap=dict()):
        asys = analysis_system
        self.asys = asys
        self.mol_info = dict()
        for mol_name in np.unique(asys.mol_names):
            self.mol_info[mol_name] = dict()
            mol_args =asys.molecule_args[asys.mol_ids[asys.mol_names==mol_name][0]]
            
            molcon = {bid:t 
                      for bid,t in asys.connectivity.items()
                      if bid[0] in mol_args
                      }
            molang = {aid:t 
                      for aid,t in asys.angles.items()
                      if aid[0] in mol_args
                      }
            moldih =  {did:t 
                      for did,t in asys.dihedrals.items()
                      if did[0] in mol_args
                      }
            self.mol_info[mol_name]['atoms'] = mol_args
            self.mol_info[mol_name]['connectivity'] = molcon
            self.mol_info[mol_name]['angles'] = molang
            self.mol_info[mol_name]['dihedrals'] = moldih
            self.funmap = funmap
        return
    def get_fun(self,k,bestrict=False):
        if len(k) == 2:
            ty = self.asys.connectivity[k]
        elif len(k) ==3:
            ty = self.asys.angles[k]
        elif len(k) ==4:
            ty = self.asys.dihedrals[k]
        if ty in self.funmap:
            return self.funmap[ty]
        else:
            if bestrict:
                raise Exception('You need to provide me the function for interaction type {}\n Add all the interaction types and functions in the funmap dictionary when you initialize'.format(ty))
            else:
                return 1
            
    def write_itp(self,mol_name,nexl=3,fname=None,bestrict=False,**kwargs):
        if fname is None:
            fname = mol_name+'.itp'
        info = self.mol_info[mol_name]
        first_id = self.asys.at_ids[info['atoms'][0]]
        subtr = first_id -1
        with open(fname,'w') as f:
            f.write('[moleculetype] \n ; Name      nrexcl \n')
            f.write('{:5s}   {:1d} \n'.format(mol_name,nexl))
            f.write('[atoms]\n')
            f.write('; nr    type   resnr  residu    atom  cgnr  charge mass\n')
            
            for i in info['atoms']:
                aid = self.asys.at_ids[i] - subtr
                ty = self.asys.at_types[i]
                mass = self.asys.atom_mass[i]
                charge = self.asys.atom_charge[i]
                f.write('{:5d}  {:5s}  {:5d}  {:5s}  {:5s}  {:5d}  {:4.5f}  {:4.5f} \n'.format(aid,
                                ty,aid,mol_name,ty,1,charge,mass))
            
            f.write('[bonds]\n;         ai      aj     funct\n')
            for k,b in info['connectivity'].items():
                ids = [i-subtr for i in k ]
                fun = self.get_fun(k,bestrict)
                f.write('{:5d}   {:5d}   {:2d} \n'.format(*ids,fun))
            f.write('[angles]\n;         ai      aj     ak    funct\n')
            for k,an in info['angles'].items():
                ids = [i-subtr for i in k ]
                fun = self.get_fun(k,bestrict)
                f.write('{:5d}   {:5d}   {:5d}   {:2d} \n'.format(*ids,fun))
            f.write('[dihedrals]\n;         ai      aj     ak    funct\n')
            for k,di in info['dihedrals'].items():
                ids = [i-subtr for i in k ]
                fun = self.get_fun(k,bestrict)
                f.write('{:5d}   {:5d}   {:5d}   {:5d}   {:2d} \n'.format(*ids,fun))
            if 'k' in kwargs and 'r' in kwargs:
                self.write_posres(f,info['atoms'].shape[0],k=kwargs['k'],r=kwargs['r'])
                
        return
    
    def write_posres(self,file,natoms,k=10000,r=0):
        file.write('[position_restraints]\n')
        for i in range(natoms):
            file.write('{:d}  {:d}  {:d}  {:8.5f}  {:8.5f} \n'.format(i+1,2,1,r,k))
        return
    
class supraClass():
    def __init__(self,topol_file,connectivity_info,
                 memory_demanding=False,keep_frames=(None,None),**kwargs):
        if 'conftype' not in kwargs:
            systemClass = Analysis
        else:
            systemClass = Analysis_Confined
            
        self.mdobj = systemClass(topol_file,connectivity_info,memory_demanding,**kwargs)
        self.keep_frames = keep_frames
        return
    
    def set_keep_frames(self,num_start=None,num_end=None):
        self.keep_frames = (num_start,num_end)
        return 
    
    def get_property(self,trajf,funcname,*func_args,**func_kwargs):
        
        self.traj_files = trajf
        
        func = getattr(self,funcname)
        
        data = multy_traj.multiple_trajectory(trajf,func,*func_args,**func_kwargs)
        
        return data
    
    def dealloc_timeframes(self):
        try:
            if type(self.traj_files) is not str:
                del self.mdobj.timeframes
                self.mdobj.timeframes = dict()
        except:
            pass
            logger.warning('WARNING: timeframes not been able to deallocated. Unless you have multiple trajectories you should not have any problem other than filling the memory')
        return
    
    def read_timeframes(self,trajf):

        if not self.mdobj.memory_demanding:
            if self.keep_frames[1] is None:
                num_end = 1e16
            else:
                num_end = self.keep_frames[1]
            self.mdobj.read_file(trajf,num_end)
        else:
            self.mdobj.setup_reading(trajf)
        if not self.keep_frames == (None,None):
            self.mdobj.cut_timeframes(*self.keep_frames)
        return
    def handleCharge(self,appendhydro=[]):
        if len(appendhydro)>0:
            add_atoms.add_ghost_hydrogens(self.mdobj,appendhydro)
            add_atoms.append_atoms(self.mdobj,'ghost')
        
        return 
    
    def handleWeights(self,ft,dynOptions):
        if 'degree' in ft:
            weights = ft['degree'] 
            ft = { t:v for t,v in ft.items() if t != 'degree' }
        
        if 'w' in dynOptions:
            if 'degree' in ft:
                dynOptions['weights_t'] = weights      
            del dynOptions['w']
        
        return ft, dynOptions
    
    @staticmethod
    def check_direction(direction):
    
        if len(direction)!=3:
            raise Exception('direction must be a 3D vector (list) or np.array')
        td = type(direction)
        if not (td is list or td is type(np.ones(3)) or td is tuple):
            raise Exception('direction is not the prober type')
        direction = np.array(direction)
        return direction
    
    def computeTACF(self,prop,xt,ft,dynOptions):
        
        ft,  dynOptions = self.handleWeights(ft,dynOptions)
        
        if len(ft) >0:    
            dyn = {k : self.mdobj.TACF(prop,xt,fs,**dynOptions) 
                   for k,fs in ft.items()
                   }
            dyn['system'] = self.mdobj.TACF(prop,xt)
        else:
            dyn = self.mdobj.TACF(prop,xt)
        return dyn  
          
    
    def computeDynamics(self,prop,xt,ft,dynOptions):
        
        ft,  dynOptions = self.handleWeights(ft,dynOptions)
        
        if len(ft) >0:    
            dyn = {k : self.mdobj.Dynamics(prop,xt,fs,**dynOptions) 
                   for k,fs in ft.items()
                   }
            dyn['system'] = self.mdobj.Dynamics(prop,xt,**dynOptions)
        else:
            dyn = self.mdobj.Dynamics(prop,xt,**dynOptions)
        return dyn 

    def computeKinetics(self,filt_t):
        kinetics = dict()
        for key, ft in filt_t.items():
            ftads = {t:np.logical_not(v) for t,v in ft.items()}
            kinet = self.mdobj.Kinetics(ftads)
            kinetics['{}-res'.format(key)] = kinet['K'].copy()

            kinet = self.mdobj.Kinetics(ft)
            kinetics['{}-nonres'.format(key)] = kinet['K'].copy()
        
        kinetics['time'] = kinet['time'].copy()

        return  kinetics

    def residence_dynamics(self, trajf, segmol, segbond=None,spaces=None ):
        if spaces is None:
            spaces = self.mdobj.adsorption_interval
        filters = {'seg_space' : spaces}
        segmol_ids = self.mdobj.find_general_ids(segmol,segbond)

        self.read_timeframes(trajf)
        cm_t, filt_t = self.mdobj.calc_segmolCM_t(segmol, segbond, filters, option='')
        self.dealloc_timeframes()

        kinetics = self.computeKinetics(filt_t)
        for key, ft in filt_t.items():
            fraction = np.array( [ np.count_nonzero(f)/f.shape[0] for f in ft.values() ]) 
            kinetics['{}-fraction'.format(key)] = fraction
        return kinetics

    def cluster_dynamics(self,trajf,mol,rcut,method='min'):
        
        self.read_timeframes(trajf)
        data = self.mdobj.calc_cluster_size_t(mol,rcut,method)
        self.dealloc_timeframes()
        
        #identify isolated ids
        n = np.unique(self.mdobj.mol_ids [ self.mdobj.mol_names == mol] ).shape[0]

        clusters_time = data['clusters']

        
        isol = dict()
        # molecule ids are stored from 0 in these data
        for k,clusters in clusters_time.items():
            isisolated = np.zeros(n,dtype=bool)
            for c in clusters:
                ln = len(c)
                if ln == 1:
                    mid = list(c)[0] 
                    isisolated[ mid ] = True
            isol[k] = isisolated
        
        notisol = {t:np.logical_not(v) for t,v in isol.items()}
        kinet = dict()
        
        
        det = self.mdobj.Kinetics(isol)
        
    
        kinet['isolated'] = det['K']
        
        att = self.mdobj.Kinetics(notisol)

        kinet['in cluster'] = att['K']
        
        kinet.update(data)
        
        return kinet



    def cluster_size(self,trajf,mol,rcut,method='min'):
        
        self.read_timeframes(trajf)
        dyn = self.mdobj.calc_cluster_size_t(mol,rcut,method)
        self.dealloc_timeframes()
        
        return dyn
    
    def dynamic_structure_factor(self,trajf,q,filters=dict(),
                           dynOptions=dict()):
        
        self.read_timeframes(trajf)
        
        dynOptions["q"] = q
        try:
            ids = self.mdobj.polymer_ids
        except:
            ids = self.mdobj.at_ids
        coords_t,ft = self.mdobj.calc_coords_t( ids , filters = filters )
        
        dyn = self.computeDynamics('Fs',coords_t,ft,dynOptions)
        
        self.dealloc_timeframes()
        
        return dyn
    def segmental_dynamics(self,trajf,topol_vec=4,filters=dict(),
                           prop='P1',dynOptions=dict()):
        
        self.read_timeframes(trajf)
        
        seg_t,fseg = self.mdobj.calc_segmental_vectors_t( topol_vec , filters = filters )
        
        dyn = self.computeDynamics(prop,seg_t,fseg,dynOptions)
        
        self.dealloc_timeframes()
        
        return dyn
    
    def chain_dynamics(self,trajf,filters=dict(),
                           prop='P1',dynOptions=dict()):
        
        self.read_timeframes(trajf)
        
        seg_t,fseg = self.mdobj.calc_Ree_t( )
        chain_cm_t,ft = self.mdobj.calc_chainCM_t( filters = filters)
        dyn = self.computeDynamics(prop,seg_t,ft,dynOptions)
        
        self.dealloc_timeframes()
        
        return dyn
    
    def segmental_dipole_dynamics(self,trajf,topol_vec,segbond,
                                  appendhydro=[],filters=dict(),
                                  prop='P1',dynOptions=dict()):
        
        self.read_timeframes(trajf)
        
        self.handleCharge(appendhydro)

        dm_t,fdm = self.mdobj.calc_segmental_dipole_moment_t(topol_vec,segbond=segbond,filters=filters)

        dyn = self.computeDynamics(prop,dm_t,fdm,dynOptions)        
        
        self.dealloc_timeframes()
        
        return dyn
    
    def chain_dipole_dynamics(self,trajf,
                                  appendhydro=[],filters=dict(),
                                  prop='P1',dynOptions=dict(),**options):
        
        self.read_timeframes(trajf)
        
        chain_cm_t,ft = self.mdobj.calc_chainCM_t( filters = filters )
        
        del chain_cm_t
        
        self.handleCharge(appendhydro)
        
        dm_t,fdm = self.mdobj.calc_chain_dipole_moment_t(**options)

        dyn = self.computeDynamics(prop,dm_t,ft,dynOptions)        
        
        self.dealloc_timeframes()
        
        return dyn
    
    def segmental_desorption(self,trajf,topol_vec,kin=['des','ads'],method='space'):
        
        if method =='space':
            filt = {'space': self.mdobj.adsorption_interval}
        elif method == 'conf':
            filt = {'conformations':['train']}
        else:
            raise ValueError('method should be equal to "space" or "conf", you gave {}'.format(method))
        
        valuerror = 'acceptable values are "des","ads", ["des","ads"], you gave {}'.format(kin)
        if kin=='des' or kin=='ads' : 
            kin =[kin]
        else:
            try:
                for k in kin:
                    if k not in ['des','ads']:
                        raise ValueError(valuerror)
            except:
                raise ValueError(valuerror)

        self.read_timeframes(trajf)
        
        seg_t,fseg = self.mdobj.calc_segmental_vectors_t( topol_vec , filters = filt )
        
        #looping over the multpliple adsorption interavls
        keys = list(fseg.keys())
        ft = {t: v for t,v in fseg[keys[0]].items()}
        for key in keys[1:]:
            for t in ft:
                ft[t] = np.logical_or(ft[t],fseg[key][t]) 
        kinetics = dict()
        for k in kin:    
            if k == 'ads':
                ftads = {t:np.logical_not(v) for t,v in ft.items()}
                kinet = self.mdobj.Kinetics(ftads)
                kinetics['ads'] = kinet['K'].copy()
            if k == 'des':
                kinet = self.mdobj.Kinetics(ft)
                kinetics['des'] = kinet['K'].copy()
        
            kinetics['time'] = kinet['time'].copy()

        self.dealloc_timeframes()
        return kinetics

    def chain_desorption(self,trajf,kin='des',w=True):
        
        self.read_timeframes(trajf)
      
        filt = {'adsorption':None}
        
        
        chain_cm_t,ft = self.mdobj.calc_chainCM_t( filters = filt)
        
        del chain_cm_t
        
        wt = ft['degree']
        ft = ft['ads']
                 
        if kin =='ads':
            ft = {t:np.logical_not(v) for t,v in ft.items()}
        
        if w:
            kinet = self.mdobj.Kinetics(ft,wt)
        else:
            kinet = self.mdobj.Kinetics(ft)
        
        self.dealloc_timeframes()
        return kinet
    
    def segmental_msd(self,trajf,seg_vec,segbond,direction=[1,1,1],
                      filters=dict(),dynOptions=dict()):
        
        direction = self.check_direction(direction)
        
        self.read_timeframes(trajf)
 
        cmt, ft = self.mdobj.calc_segCM_t(seg_vec,segbond,filters = filters)
        
        if not (direction == np.ones(3)).all():
            cmt = {t:c*direction for t,c in cmt.items()}
        
        dyn = self.computeDynamics('MSD',cmt,ft,dynOptions)
 
        self.dealloc_timeframes()
        return dyn
    
    def chain_msd(self,trajf,direction=[1,1,1],
                      filters=dict(),dynOptions=dict()):
        
        direction = self.check_direction(direction)
        
        self.read_timeframes(trajf)
 
        cmt, ft = self.mdobj.calc_chainCM_t(filters = filters)
        
        if not (direction == np.ones(3)).all():
            cmt = {t:c*direction for t,c in cmt.items()}
        
        dyn = self.computeDynamics('MSD',cmt,ft,dynOptions)
 
        self.dealloc_timeframes()
        return dyn
    
    def dihedral_dynamics(self,trajf,phi,prop='sin',filters=dict(),dynOptions=dict()):
        
        
        if phi not in self.mdobj.dihedral_types:
            raise ValueError('{} is not in dihedrals'.format(phi))
        
        self.read_timeframes(trajf)
 
        phit, ft = self.mdobj.calc_dihedrals_t(phi,filters = filters)
        
        dyn = self.computeTACF(prop,phit,ft,dynOptions)
 
        self.dealloc_timeframes()
        return dyn
    
    def dihedral_distribution(self,trajf,phi,filters=dict(),degrees=True):
    
        if phi not in self.mdobj.dihedral_types:
            raise ValueError('{} is not in dihedrals'.format(phi))
        
        self.read_timeframes(trajf)
 
        phit, ft = self.mdobj.calc_dihedrals_t(phi,filters = filters)
        
        distr = {'system':[]}
        distr.update({k:[] for k in ft})
        for t,dih in phit.items():
            distr['system'].extend(dih)
            for k in ft.keys():
                 distr[k].extend( dih[ft[k][t]] )
        if degrees:
            scale = 180/np.pi
            distr = {k:np.array(distr[k])*scale for k in distr}
        else:
            distr = {k:np.array(distr[k])*scale for k in distr}
            
        self.dealloc_timeframes()
        
        if len(distr)==1:
            distr = distr[ list(distr.keys())[0] ]
        
        return distr
    
    def conformation_evolution(self,trajf,option=''):
        
        self.read_timeframes(trajf)
        confs = self.mdobj.calc_conformations_t(option)
        self.dealloc_timeframes()
        
        return confs
    def density_profile(self,trajf,binl,dmax,offset=0,option='',mode='mass',flux=None,types=None):
        self.read_timeframes(trajf)
        densprof = self.mdobj.calc_density_profile(binl,dmax,offset,option,mode,flux,types)
        self.dealloc_timeframes()
        return densprof
    
    def orientation(self,trajf,topol_vec,binl,dmax,offset=0,option=''):
        self.read_timeframes(trajf)
        p2 = self.mdobj.calc_P2(topol_vec,binl,dmax,offset,option)
        self.dealloc_timeframes()
        return p2
    
    def Rg(self,trajf,option=''):
        self.read_timeframes(trajf)
        cha = self.mdobj.calc_Rg(option)
        self.dealloc_timeframes()
        return cha
    
    def chain_structure(self,trajf,binl,dmax,offset=0,option=''):
        self.read_timeframes(trajf)
        cha = self.mdobj.calc_chain_characteristics(binl,dmax,offset)
        self.dealloc_timeframes()
        return cha
    
    def static_dipole_correlations(self,trajf,topol_vec,segbond,appendhydro=[],filters=dict()):
        self.read_timeframes(trajf)
        self.handleCharge(appendhydro)
        corrs = self.mdobj.calc_segmental_dipole_moment_correlation(topol_vec,segbond,filters)
        self.dealloc_timeframes()
        return corrs
    
    def segmental_pair_distribution(self,trajf,binl,dmax,topol_vector,segbond,far_region=0.8):
        self.read_timeframes(trajf)
        paird = self.mdobj.calc_segmental_pair_distribution(binl,dmax,topol_vector,segbond,far_region)
        self.dealloc_timeframes()
        return paird
    
    def pair_distribution(self,trajf,binl,dmax,type1=None,type2=None,
                              intra=False,inter=False,
                               far_region=0.8):
        self.read_timeframes(trajf)
        pd = self.mdobj.calc_pair_distribution(binl,dmax,type1,type2,intra,inter,far_region)
        self.dealloc_timeframes()
        return pd
    
    def Sq(self,trajf,dq,qmax,method='inverse',qmin=2,dmin=0,dr=None,dmax=None,ids=None,direction=None):
        self.read_timeframes(trajf)
        if method =='inverse':
            if dr is None: dr =dq
            if dmax is None: dmax = qmax/10.0
            dat = self.mdobj.calc_Sq_byInverseGr(dr,dmax,dq,qmax,qmin,ids,direction)
        else:
            dat = self.mdobj.calc_Sq(qmin,dq,qmax,direction=None,ids=None)
        
        q = dat['q'] ; Sq = dat['Sq'] 
        
        dat.update({'qmax':q[q>1][Sq[q>1].argmax()],'Sqmax':Sq.max()})
        
        self.dealloc_timeframes()
        
        return dat
    
    def get_dirs_for_stress_relaxation(self,u):
    
        valueErr = 'u (direction of stress) must be one of the following {shear, normal, x, y, z, xy, xz, yz, yx, zx, zy, 0, 1, 2, 3, 4, 5, 6, 7 }'
        
        if u == 'shear': dirs = [1,2,3,5,6,7]
        elif u == 'normal': dirs = [0,4,8]
        elif u =='z': dirs = [8,]
        elif u == 'y': dirs = [4,]
        elif u == 'x': dirs = [0]
        elif u =='xy' or u =='yx': dirs = [1,3]
        elif u =='xz' or u =='zx': dirs = [2,6]
        elif u =='yz'or u =='zy': dirs = [5,8]
        else:
            try:
                dirs = [int(u),]
            except:
                raise ValueError(valueErr)
            else:
                if dirs[0]<0 or dirs[0]>8:
                    raise ValueError(valueErr)
        return dirs
    
    def average_the_inner(self,srel,n,key):
        average_s = dict()
        for k in srel:
            average_s[k] = np.zeros(n,dtype=float)
            for i in srel[k]:
                average_s[k]+=srel[k][i][key]            
            average_s[k] /= len(srel[k])
        
        for k in srel:
            i0 = list(srel[k].keys())[0]
            average_s[k] = {'time':srel[k][i0]['time'],'g':average_s[k]}
        return average_s
    
    def region_stress_relaxation(self,trajf,u='shear',filters=dict()):
        dirs = self.get_dirs_for_stress_relaxation(u)
       
        self.read_timeframes(trajf)
        stress_t,ft = self.mdobj.stress_per_t(filters=filters)
        vregion = {'system':{i:{t:np.mean(stress_t[t],axis=0)[i] for t in stress_t} for i in dirs} }
        if len(ft) != 0:
            vregion.update( {k: {t:np.mean(stress_t[t][f[t]] ,axis=0) for t in stress_t} for k,f in ft.items() } )    
       #return vregion
        srel = {k:{i:self.mdobj.multy_tau_average(v[i])
                  for i in dirs}
               for k,v in vregion.items() }
        average_s = self.average_the_inner(srel,len(stress_t),'corr')
        
        
        self.dealloc_timeframes()
        
        return average_s
         
    
    def atom_stress_relaxation(self,trajf,u='shear',filters=dict(),dynOptions=dict()):
        
        dirs = self.get_dirs_for_stress_relaxation(u)
       
        self.read_timeframes(trajf)
        
        stress_t,ft = self.mdobj.stress_per_t(filters=filters)
        
        #dynOptions = {'block_average':True,}
        
        sti = { i : {t:v[:,i] for t,v in stress_t.items()} for i in dirs }
        
        srel = { i: self.computeDynamics('scalar',st,ft,dynOptions) for i,st in sti.items() }
        
        if len(ft)==0:
            srel = {'system':srel}
        else:
            srel = ass.rearrange_dict_keys(srel)
        
        average_s = self.average_the_inner(srel,len(stress_t),'scalar')
        if len(ft) == 0:
            return average_s['system']
        
        self.dealloc_timeframes()
        
        return average_s
        
class multy_traj():
    def __init__(self):
        return

    @staticmethod
    def average_data(files,function,*fargs, **fkwargs):
       
        def ave1():
            ave_data = dict()
            ldata = {k:[data[k]] for k in data}
            for i in range(1,len(files)): 
                file = files[i]
                di = function(file,*fargs,**fkwargs)
                for k in ldata:
                    ldata[k].append( di[k])
            for k in ldata:
                ave_data[k] = np.mean( ldata[k], axis=0 )
                ave_data[k+'(std)'] =  np.std( ldata[k], axis=0 )
            return ave_data
        def ave2():
            ave_data = {k1:dict() for k1 in data}
            ldata = {k1:{k2: [data[k1][k2] ] for k2 in data[k1]} for k1 in data}
            for i in range(1,len(files)): 
                file = files[i]
                di = function(file,*fargs,**fkwargs)
                for k1 in ldata:
                    for k2 in ldata[k1]:
                        ldata[k1][k2].append(di[k1][k2])
            for k1 in ldata:
                for k2 in ldata[k1]:
                    ave_data[k1][k2] = np.mean( ldata[k1][k2], axis=0 )
                    ave_data[k1][k2+'_STD'] =  np.std( ldata[k1][k2], axis=0 )
            return ave_data
        
        files = list(files)
        
        data = function(files[0],*fargs,**fkwargs)

        if ass.is_dict_of_dicts(data):
            ave_method =  ave2
        elif type(data) is dict:
            ave_method = ave1
        else:
            raise Exception('Type of data and arreangement not regognised')
        
        averaged_data = ave_method()        

        return averaged_data
    
    @staticmethod
    def wrap_the_data(data_to_wrap,wrapon):
        wraped_data = dict()
        print(data_to_wrap.keys())
        for i,(file,data) in enumerate(data_to_wrap.items()):
            if i==0:
                wraped_data = data
                continue
            if ass.is_dict_of_dicts(data):
                print(data.keys())
                varw_old =  {k: wraped_data[k][wrapon] for k in wraped_data}
                for k,d  in data.items():
                    jp0 = d[wrapon][0]
                    jp1 = d[wrapon][1]
                    joint_point = jp0 if jp0 !=0 else jp1
                    fk = varw_old[k]<joint_point
                    for ik in d: 
                        try:
                            wraped_data[k][ik] = wraped_data[k][ik][fk] # cutting edge points of the old
                        except KeyError:
                            pass
                    fnk = d[wrapon]>=joint_point
                    for ik in d:
                        if  ik in wraped_data[k]:
                            ctupl = (wraped_data[k][ik], d[ik][fnk])
                        else:
                            ctupl =(d[ik][fnk],)
                        wraped_data[k][ik] = np.concatenate(ctupl)
            elif type(data) is dict:
                varw_old = wraped_data[wrapon]
                jp0 = data[wrapon][0]
                jp1 = data[wrapon][1]
                joint_point = jp0 if jp0 !=0 else jp1
                fk = varw_old<joint_point
                for ik in data:
                    try:
                        wraped_data[ik] = wraped_data[ik][fk]
                    except KeyError:
                        pass
                fnk = data[wrapon]>=joint_point
                for ik in data:
                    if ik in wraped_data:
                        ctupl = (wraped_data[ik], data[ik][fnk])
                    else:
                        ctupl = (data[ik][fnk],)
                    wraped_data[ik] = np.concatenate(ctupl)
            else: 
                raise Exception('Unregognized type of data for wrapping')
        return wraped_data
    
    @staticmethod
    def multiple_trajectory(files,function,*fargs,**fkwargs):
       
        if 'wrapon' not in fkwargs:
            wrapon = 'time'
        else:
            wrapon = fkwargs['wrapon']
            
        type_files = type(files)
        if type_files is dict:
            
            changing_args = ass.is_tuple_of_samesized_tuples(fargs)
                
            if changing_args:
                if len(fargs) != len(files):
                    raise Exception('if you pass tuple of tuples then they should be the same size as your dictionary')
                return {k:
                        multy_traj.multiple_trajectory(f, function, *fargs[i], **fkwargs)
                        for i,(k,f) in enumerate(files.items())
                        }
            else:
                return {k:
                        multy_traj.multiple_trajectory(f, function, *fargs, **fkwargs)
                        for k,f in files.items()
                        }
        if type_files is set:
            mult_data = multy_traj.average_data(files,function,*fargs,**fkwargs)
        elif type_files is list or type_files is tuple:
            data_to_wrap = dict()
            for file in files:
                type_file  = type(file)
                if type_file is set:
                    data = multy_traj.average_data(file,function,*fargs,**fkwargs)
                elif type_file is str:
                    data  =  function(file,*fargs,**fkwargs)
                if type_file is set:
                    key = '-'.join(file)
                else:
                    key = file
                data_to_wrap[key] = data
                
            mult_data = multy_traj.wrap_the_data(data_to_wrap,wrapon)
        elif type_files is str:
             # single file
            mult_data = function(files,*fargs,**fkwargs)
        else:
            raise Exception('type {} is not Regognized as file or files to read'.format(type_files))
        return mult_data
    
    
    
class ass():
    '''
    The ASSISTANT class
    Contains everything that is needed to assist in the data analysis including
    1) functions for manipulating data in dictionaries
    2) The wrapper for using the same function and multyple trajectories on the Dynamic analysis
    3) printing and logger functions e.g. print_time, clear_logs 
    
    '''
    @staticmethod
    def update_dict_in_object(obj,name,di):
        if not hasattr(obj,name):
            setattr(obj,name,di)
        else:
            getattr(obj,name).update(di)
        return
    @staticmethod
    def list_ifint(i):
        if type(i) is int:
            return [i,]
        else:
            return i
    
    @staticmethod
    def list_ifstr(i):
        if type(i) is str:
            return [i,]
        else:
            return i
    @staticmethod
    def list_iffloat(i):
        if type(i) is float:
            return [i,]
        else:
            return i
        
    @staticmethod
    def make_dir(name):
        name = name.replace('\\','/')
        n = name.split('/')
        lists = ['/'.join(n[:i+1]) for i in range(len(n))]  
        a = 0 
        for l in lists:
            if not os.path.exists(l):
                a = os.system('mkdir {:s}'.format(l))
                if a!=0:
                    s = l.replace('/','\\')
                    a = os.system('mkdir {:s}'.format(s))
        
        return a
    @staticmethod
    def numerical_derivative(x,y):
        d = np.empty_like(x)
        d[1:-1] = (y[2:] -y[:-2])/(x[2:]-x[:-2])
        d[0] = (y[1] -y[0])/(x[1]-x[0])
        d[-1] = (y[-1]-y[-2])/(x[-2]-x[-1])
        return d
    
    beebbeeb = True

    @staticmethod
    def is_tuple_of_samesized_tuples(x):
        if type(x) is not tuple:
            return False
        for i in x:
            if type(i) is not tuple:
                return False
        try:
            x[0]
        except IndexError:
            return False
        else:
            l = len(x[0])
            for i in x:
                if l != len(i):
                    return False
        return True
    @staticmethod
    def rename_key(d,oldname,newname):
        d[newname] = d[oldname]
        del d[oldname]
        return
    @staticmethod
    def rename_keys_via_keyvalue(new_names,data_dict):
        new_dict = dict()
        for k,v in new_names.items():
            new_dict[v] = data_dict[k]
        return new_dict
    @staticmethod
    def rename_keys_via_enumeration(new_names,data_dict):
        new_dict = dict()
        l = list(data_dict.keys())
        for i,k in enumerate(new_names):
            new_dict[k] = data_dict[l[i]]
        return new_dict
    @staticmethod
    def rename_keys(new_names,data_dict):
        if type(new_names) is dict:
            data_dict = ass.rename_keys_via_keyvalue(new_names,data_dict)
        elif type(new_names) is list or type(new_names) is tuple:
            data_dict = ass.rename_keys_via_enumeration(new_names,data_dict)
        else:
            raise ValueError('new_names must be dict,list or tuple ')
        return data_dict
    
    @staticmethod
    def write_pickle(data,data_file):
        with open(data_file,'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return
    
    @staticmethod
    def read_pickle(data_file):
        with open(data_file, 'rb') as handle:
            data = pickle.load(handle)
        return data
    @staticmethod
    def save_data(data,fname,method='pickle'):
        if not ass.iterable(method):
            methods = [method,]
        for m in methods:
            if m =='pickle':
                ass.write_pickle(data,fname)
            else:
                raise NotImplementedError
                ass.write_txt(data,fname)        
        return
    
    
    @staticmethod
    def try_beebbeeb():
        if ass.beebbeeb:
            try:
                import winsound
                winsound.Beep(500, 1000)
                import time
                time.sleep(0.5)
                winsound.Beep(500, 1000)
            except:
                pass
        return
    
    @staticmethod
    def change_key(d,keyold,keynew):
        try:
            value = d[keyold]
        except KeyError as e:
            raise Exception('{} This key "{}" does not belong to the dictionary'.format(e,keyold))
        d[keynew] = value
        del d[keyold]
        return 
    
    @staticmethod
    def dict_slice(d,i1,i2):
        return {k:v for i,(k,v) in enumerate(d.items()) if i1<=i<i2 }
    
    @staticmethod
    def numpy_keys(d):
        return np.array(list(d.keys()))
    
    @staticmethod
    def numpy_values(d):
        return np.array(list(d.values()))
    @staticmethod
    def common_keys(d1,d2):
        k1 = ass.numpy_keys(d1) ; k2 = ass.numpy_keys(d2)
        return np.intersect1d(k1,k2)
    @staticmethod
    def trunc_at(dold,dnew):
        
        new_key_first = list(dnew.keys())[1]
        dko = ass.numpy_keys(dold)
        if len(dold) ==0:
            return 0
        try:
            fa = dko< new_key_first
        except:
            return 0
        else:
            dl = dko[fa]
            itrunc = np.where(dl[-1] == dko)[0][0]
            return itrunc
    

    @staticmethod
    def is_dict_of_dicts(data):
        ks = list(data.keys())
        try:
            v0 = data[ks[0]]
            if type(v0) is dict:
                return True
            else:
                return False
        except IndexError:
            return False
        

    @staticmethod
    def print_time(tf,name,nf=None):
        s1 = ass.readable_time(tf)
        if nf is None:
            s2 =''
        else:
            s2 = ' Time/frame --> {:s}\n'.format( ass.readable_time(tf/nf))
        logger.info('Function "{:s}"\n{:s} Total time --> {:s}'.format(name,s2,s1))
        return
    
    @staticmethod
    def print_stats( stats):
        print('ads chains  = {:4.4f} %'.format(stats['adschains_perc']*100))
        x = [stats[k] for k in stats if '_perc' in k and k.split('_')[0] in ['train','tail','loop','bridge'] ]
        tot = np.sum(x)
        for k in ['train','loop','tail','bridge']:
            print('{:s} = {:4.2f}%'.format(k,stats[k+'_perc']/tot*100))
        return
    
    @staticmethod
    def stay_True(dic):
        keys = list(dic.keys())
        stayTrue = {keys[0]:dic[keys[0]]}
        for i in range(1,len(dic)):
            stayTrue[keys[i]] = np.logical_and(stayTrue[keys[i-1]],dic[keys[i]])
        return stayTrue
    
    @staticmethod
    def become_False(dic):
        keys = list(dic.keys())
        bFalse = {keys[0]:dic[keys[0]]}
        for i in range(1,len(dic)):
            bFalse[keys[i]] = np.logical_and(bFalse[keys[0]],np.logical_not(dic[keys[i]]))
        return bFalse

    @staticmethod
    def iterable(arg):
        return type(arg) is list or type(arg) is tuple or type(arg) is dict
    
    @staticmethod
    def readable_time(tf):
        hours = int(tf/3600)
        minutes = int((tf-3600*hours)/60)
        sec = tf-3600*hours - 60*minutes
        dec = sec - int(sec)
        sec = int(sec)
        return '{:d}h : {:d}\' : {:d}" : {:0.3f}"\''.format(hours,minutes,sec,dec)
    
    @staticmethod
    def rearrange_dict_keys(dictionary):
        '''
        Changes the order of the keys to access data
        Parameters
        ----------
        dictionary : Dictionary of dictionaries with the same keys.
        Returns
        -------
        x : Dictionary with the second set of keys being now first.
    
        '''
        x = {k2 : {k1:None for k1 in dictionary} for k2 in dictionary[list(dictionary.keys())[0]]}
        for k1 in dictionary:
            for k2 in dictionary[k1]:
                x[k2][k1] = dictionary[k1][k2]
        return x
    
    @staticmethod  
    def check_occurances(a):
        x = set()
        for i in a:
            if i not in x:
                x.add(i)
            else:
                raise Exception('{} is more than once in the array'.format(i))
        return     
    @jit(nopython=True,fastmath=True)
    def running_average(X,every=1):
        # curnel to find running average
        n = X.shape[0]
        xrun_mean = np.zeros(n)
        for j in range(0,len(X),every):
            y = X[:j+1]
            n = y.shape[0]
            xrun_mean[j] = np.sum(y)/n
        return xrun_mean
    
    def moving_average(a, n=10) :
        #moving average kernel
        
        mov = np.empty_like(a)
        
        n2 = int(n/2)
        if n2%2 ==1: n2+=1
        up = a.shape[0]-n2
        for i in range(n2):
            mov[i] = a[:2*i+1].mean()
        for i in range(n2,up):
            mov[i] = a[i-n2:i+n2+1].mean()
        for i in range(up,a.shape[0]):
            j = (a.shape[0]-i)-1
            mov[i] = a[up-j:].mean()
            
        return mov
    
    def block_average(a, n=100) :
        #block average kernel
     
        bv = np.empty(int(a.shape[0]/n)+1,dtype=float)
        for i in range(bv.shape[0]):
            x = a[i*n:(i+1)*n]
            bv[i] = x.mean()
        return bv
    
    def block_std(a, n=100) :
        #block standard diviation
        bstd = np.empty(int(a.shape[0]/n),dtype=float)
        for i in range(bstd.shape[0]):
            x = a[i*n:(i+1)*n]
            bstd[i] = x.std()
        return bstd

@jit(nopython=True,fastmath=True,parallel=False)
def distance_kernel(d,coords,c):
    #Kernels for finding distances
    nd = coords.shape[1]
    for i in prange(d.shape[0]):
        d[i] = 0
        for j in range(nd):
            rel = coords[i][j]-c[j]
            d[i] += rel*rel
        d[i] = d[i]**0.5
    return 


@jit(nopython=True,fastmath=True,parallel=True)
def smaller_distance_kernel(d1,d2,c1,c2):
    #Kernel for finding the minimum distance between two coords
    for i in prange(c1.shape[0]):
        distance_kernel(d2,c2,c1[i])
        d1[i] = 1e16
        for j in range(d2.shape[0]):
            if d2[j]<d1[i]: 
                d1[i] = d2[j]
    return




class Energetic_Analysis():
    '''
    class to analyze gromacs energy files
    Currently needs improvement
    '''    
    def __init__(self,file):
        self.data_file = file
        if self.data_file[-4:] =='.xvg':
            self.xvg_reader()
        else:
            raise NotImplementedError('Currently only accepting xvg files')
    
    def xvg_reader(self):
        
        with open(self.data_file) as f:
            lines = f.readlines()
            f.closed
            
        columns = ['time']
        for i,line in enumerate(lines):
            l = line.split()
            
            if l[0]=='@' and l[1][0]=='s' and l[2] =='legend':
                columns.append(l[3].strip('"'))
                last_legend = i
            elif line[0] =='@' or line[0]=='#':
                continue
            else:
                break
            
        data = np.array([line.split() for line in lines[last_legend+1:]],dtype=float)
        self.data = pd.DataFrame(data,columns=columns)
        return
    
    def simple_plot(self,ycols,xcol='time',size = 3.5,dpi = 300, 
             title='',func=None,
             xlabel=['time (ps)'],save_figs=False,fname=None,path=None):
        figsize = (size,size)
        plt.figure(figsize=figsize,dpi=dpi)
        plt.minorticks_on()
        plt.tick_params(direction='in', which='minor',length=1.5*size)
        plt.tick_params(direction='in', which='major',length=1.5*size)
        plt.xlabel('time (ps)',fontsize=size*3)
        x = self.data[xcol].to_numpy()
        if func is not None:
            funcs = globals()[func]
        for ycol in ycols:
            y = self.data[ycol].to_numpy()
            plt.plot(x,y,lab=ycol)
            if funcs is not None:
                plt.plot(x,funcs(y),lab ='{} - {}'.format(ycol,func))
        plt.legend(frameon=False)
        if fname is not None:
            if path is not None:
                plt.savefig('{}\{}'.format(path,fname),bbox_inches='tight')
            else:
                plt.savefig('{}'.format(fname),bbox_inches='tight')
        plt.show()
        
class XPCS_communicator():
    @staticmethod
    def ReadPlot_XPCSdistribution(files,**plot_kwargs):
        datadict = dict()
        for file in files:
            key = file.split('/')[-1].split('.XPCSCONTINbatch')[0].replace('xpcs_','')
            datadict[key] = XPCS_Reader(file)
        plotter.relaxation_time_distributions(datadict,**plot_kwargs)
        return datadict
    
    @staticmethod
    def write_xy_forXPCS(fname,x,y):
        data = np.zeros((x.shape[0],3),dtype=float)
        data[:,0] = x
        data[:,1] = y
        data[:,2]+= np.random.uniform(0,1e-9,x.shape[0])
        np.savetxt(fname,data)
        return
    
    @staticmethod
    def write_data_forXPCS(datadict,path='XPCS_data',cutf=None,midtime=None,num=100):
        
        if cutf is None:
            cutf ={k:10**10 for k in datadict}
        for k,dy in datadict.items():
            
            fn = '{:s}\\xpcs_{:}.txt'.format(path,k)
            
            x = ass.numpy_keys(dy)/1000
            y = ass.numpy_values(dy)
            t = x<=cutf[k]
            x = x[t]
            y = y[t]
            
            args = plotter.sample_logarithmically_array(x,midtime=midtime,num=num)
            xw = x[args]
            yw = y[args]
            XPCS_communicator.write_xy_forXPCS(fn, xw,yw+1)
        return
    
class XPCS_Reader():
    def __init__(self,fname,fitfunc='freq'):
        self.relaxation_modes = []
        self.params = []
        self.params_std = []
        self.func = getattr(fitFuncs,fitfunc)
        with open( fname,'r') as f:
            lines = f.readlines()
            f.closed
            
        for i,line in enumerate(lines):
            l = line.strip() 
            if 'Background' in l:
                self.background= float(l.split(':')[1])
            if 'log Lagrange Multiplier' in l:
                self.reg = float(l.split(':')[1])
            if 'log(Upsilon)' in line:
                linum = i+2
            if 'Kohlrausch Exponent' in l:
                self.ke = float(l.split(':')[1])
        
        for i,line in enumerate(lines[linum:]):
            l = line.strip().split()
            if '----' in line: break
            self.relaxation_modes.append(float(l[0]))
            self.params.append(float(l[1]))
            self.params_std.append(float(l[2]))
        
        self.params = np.array(self.params)
        self.params_std= np.array(self.params_std)
        
        dlogtau = self.relaxation_modes[1]-self.relaxation_modes[0]
        
        
        self.smoothness = smoothness(self.params,dlogtau)
        self.relaxation_modes = 10**np.array(self.relaxation_modes)
        w = self.params
        f = self.relaxation_modes
        self.taus = self.relax()
        self.bounds=(f[0],f[-1])
        self.contributions  = w*self.taus
        self.taur = self.tau_relax()
        
        self.t = np.logspace(-4,np.log10(self.taur)+2,num=10000)
        self.curve =self.fitted_curve(self.t) 
        return
    
    def tau_relax(self):
        return np.sum(self.contributions)
    def relax(self):
        if self.ke == 1.0:
            return 1.0/self.relaxation_modes
        elif self.ke == 2.0:
            return 0.5*np.sqrt(np.pi)/np.sqrt(self.relaxation_modes)
    
    def plot_distribution(self,size=3.5,xlim=(-6,8),title=None):
        fig,ax=plt.subplots(figsize=(size,size),dpi=300)
        plt.minorticks_on()
        plt.tick_params(direction='in', which='minor',length=size*1.5)
        plt.tick_params(direction='in', which='major',length=size*3)
        plt.yscale('log')
        plt.xscale('log')
        if title is not None:
            plt.title(title)
        xticks = [10**x for x in range(xlim[0],xlim[1]+1) ]
        plt.xticks(xticks,fontsize=min(2.5*size,2.5*size*8/len(xticks)))
        plt.yticks(fontsize=2.5*size)
        plt.xlabel(r'$f$ / $ns^{-1}$',fontsize=2*size)
        #plt.ylabel(r'$w$',fontsize=3*size)
        locmin = matplotlib.ticker.LogLocator(base=10.0,subs=np.arange(0.1,1,0.1),numticks=12)
        ax.xaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        
        x = self.relaxation_modes
        w = self.params
        c = self.contributions
        
        plt.plot(x,w,ls='--',lw=0.25*size,marker = 'o',
            markersize=size*1.2,markeredgewidth=0.15*size,
            fillstyle='none',label='w')
        plt.plot(x,c,ls='--',lw=0.25*size,marker = 's',
            markersize=size*1.2,markeredgewidth=0.15*size,
            fillstyle='none',label='c')
            #plt.plot(f.relaxation_modes,f.params,ls='--',label=k,color=cmap[k])
        plt.legend(frameon=False,fontsize=2.3*size)
        plt.show()
        return 
    
    def fitted_curve(self,x):
        fc = self.func(x,self.bounds[0],self.bounds[1],self.params)
        try:
            fc += self.background
        except AttributeError:
            pass
        
        return fc
    
class plotter():
    def __init__(self):
        return
    
    class colors():
        qualitative = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999']
        diverging = ['#d73027','#f46d43','#fdae61','#fee08b','#ffffbf','#d9ef8b','#a6d96a','#66bd63','#1a9850']
        div6 = ['#d73027','#fc8d59','#fee090','#e0f3f8','#91bfdb','#4575b4']
        sequential = ['#fee0d2','#fc9272','#de2d26']
        safe = ['#1b9e77','#d95f02','#7570b3']
        semisafe = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02','#a6761d','#666666']
        safe2 = ['#a6cee3','#1f78b4','#b2df8a']
        qual6 = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02']
        
    class linestyles():
        lst_map = {
            'loosely dotted':      (0, (1, 3)),
            'dotted':               (0, (1, 1)),
            'densely dotted':       (0, (2, 1)),
            
            'loosely dashed':       (0, (5, 3)),
            'dashed':               (0, (4, 2)),
            'densely dashed':       (0, (3, 1)),

         'loosely dashdotted':   (0, (5, 3, 1, 3)),
         'dashdotted':           (0, (4, 2, 1, 2)),
         'densely dashdotted':    (0, (3, 1, 1, 1)),

         'dashdotdotted':         (0, (5, 2, 1, 2, 1, 2)),
         'loosely dashdotdotted': (0, (4, 3, 1,3, 1, 3)),
         'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1))
         }
        lst3 = [lst_map['densely dotted'],lst_map['loosely dashed'],
              lst_map['densely dashed']]
        lst4 = [lst_map['densely dotted'],lst_map['loosely dotted'],
              lst_map['densely dashed'],lst_map['loosely dashed']]
        lst6 = [lst_map['loosely dotted'],
                lst_map['dotted'],
                lst_map['dashed'],
                lst_map['loosely dashdotted'],
                lst_map['dashdotted'],
                lst_map['densely dashdotted']]
        lst7 = ['-','-.','--',
                lst_map['loosely dotted'],
                lst_map['dotted'],
                lst_map['dashed'],
                lst_map['loosely dashdotted'],
                lst_map['dashdotted'],
                lst_map['densely dashdotted'],
                lst_map['densely dotted']]
    @staticmethod
    def boldlabel(label):
        label = label.split(' ')
        boldl = ' '.join([r'$\mathbf{'+l+'}$' for l in label])
        return boldl
    
    @staticmethod
    def relaxation_time_distributions(datadict,fitobject='params',yscale=None,
                                      size=3.5,fname=None,title=None,
                                      cmap=None,xlim=(-6,6),pmap=None,ylim=(1e-6,1.0),
                                      units='ns',mode='tau'):
        
        if cmap is None:
            c = plotter.colors.semisafe
            try:
                cmap = { k : c[i] for i,k in enumerate(datadict.keys()) }
            except IndexError:
                color_map = matplotlib.cm.get_cmap('viridis')
                cmap = {k: color_map(i/len(datadict)) for i,k in enumerate(datadict.keys())}
        if pmap is None:
            pmap = {k:'o' for k in datadict}
        figsize = (size,size)
        dpi = 300
        fig,ax=plt.subplots(figsize=figsize,dpi=dpi)
        plt.minorticks_on()
        plt.tick_params(direction='in', which='minor',length=size*1.5)
        plt.tick_params(direction='in', which='major',length=size*3)
        plt.xscale('log')
        if yscale is not None:
            plt.yscale(yscale)
            plt.ylim(ylim)
        if title is not None: 
            plt.title(title)
        xticks = [10**x for x in range(xlim[0],xlim[1]+1) ]
        plt.xticks(xticks,fontsize=min(2.5*size,2.5*size*8/len(xticks)))
        plt.yticks(fontsize=2.5*size)
        if mode =='freq': 
            units = '{:s}^{:s}'.format(units,'{-1}')
            lab = 'freq'
        else:
            lab='\tau'
        plt.xlabel(r'${:s}$ / ${:s}$'.format(lab,units),fontsize=2*size)
        #plt.ylabel(r'$w$',fontsize=3*size)
        locmin = matplotlib.ticker.LogLocator(base=10.0,subs=np.arange(0.1,1,0.1),numticks=12)
        ax.xaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        for i,(k,f) in enumerate(datadict.items()):
            if fitobject=='params':
                x = f.relaxation_modes
                y = f.params
            elif fitobject=='eps_imag':
                x = f.omega
                y = f.eps_imag
                
            plt.plot(x,y,ls='--',lw=0.25*size,marker = pmap[k],
                markersize=size*1.2,markeredgewidth=0.15*size,fillstyle='none',color=cmap[k],label=k)
            #plt.plot(f.relaxation_modes,f.params,ls='--',label=k,color=cmap[k])
        plt.legend(frameon=False,fontsize=2.3*size)
        if fname is not None:
            plt.savefig(fname,bbox_inches='tight')
        plt.show()
        return 
    
    @staticmethod
    def compare_data(data1,data2,plot_args=[],**kwargs):
        if type(data1) is dict:
            if type(data2) is dict:
                multiple_data=True
            else:
                raise Exception('Either give both data as objects or as dicts')
            if len(data1) != len(data2):
                raise Exception('Data dicts must be of the same size')
        else:
            multiple_data = False

            
        if multiple_data == True:
            for i in range(len(data1)):
                if i>=len(plot_args):
                    plot_args.append(dict())
                
                plot_args[i] = {**kwargs,**plot_args[i]}
                
            for (k1,d1),(k2,d2),pargs in zip(data1.items(),data2.items(),plot_args):
                dc = {k1:d1,k2:d2}
                plotter.relaxation_time_distributions(dc,**pargs)
        else:
           dc = {'data1':data1,'data2':data2}
           plotter.relaxation_time_distributions(dc,**kwargs) 
        return                

        
    @staticmethod
    def sample_logarithmically_array(x,midtime=None,num=100,first_ten=True):
        
        
        if midtime is not None:
            if not ass.iterable(midtime):
                if type(midtime) is not float:
                    if type(midtime) is not int: 
                        raise Exception('midtime must be float,int or iterable of floats,ints')
                midtime = [float(midtime)]
            else:
                midtime = [float(m) for m in midtime]
            midtime.append(float(x[-1]))
        else:
            midtime = [ float(x[-1]) ]
    
        nm = len(midtime)

        args = np.array([0])
        if first_ten: 
            num-= nm*10
        num /= nm
        num = int(num)
        i=0
        for midt in midtime:
   
            if type(midt) is not float: 
                    raise Exception('midtime must be float or iterable of floats')
            mid = x[x<=midt].shape[0]
            if first_ten:
                fj = int(round(10**i,0))
                args = np.concatenate((args,[j for j in range(fj,fj+10)]+[fj-1]))
            try:
                lgsample = np.logspace(i,np.log10(mid),num=num).round(0) 
            
                args = np.concatenate( (args, np.array(lgsample,dtype=int)) )
            except ValueError as e:
                logger.warning('Excepted ValueError{}\nConsider increasing number of sumpling points'.format(e))
            i=np.log10(mid)
        args = np.unique(args[args<x.shape[0]])
        return np.array(args,dtype=int)
    
    @staticmethod
    def colormaps():
        return sorted(m for m in plt.cm.datad)     
    
    @staticmethod
    def plotDynamics(datadict,xaxis='time',yaxis=None,compare=None,
                     style='points',comp_style='lines', fits = None,
             fname =None,title=None,size=3.5,add_point=None,
             ylabel=None,xlabel=None, ticksfont=2.0,
             xlim=None,ylim=None, xscale='log',yscale=None,
             pmap=None, cmap = None, labmap=None,lstmap=None,cutf=None,
             midtime=None,num=100,write_plot_data=False,ls=None,lw=0.5,
             edgewidth=0.3,markersize=1.5,legend=True,moving_average=None,
             legkwargs=dict(),legfont=2.0,extra_line=False,fillstyle='none',
             xticks=None,yticks=None,minoryticks=None,minorxticks=None,majorticks=1.5,
             saveformats=[],):
        
        if not ass.is_dict_of_dicts(datadict):
            datadict = {'key':datadict}
            legend=False
        if labmap is None:
            labmap  = {k:k for k in datadict}
        if cmap is None:
            c = plotter.colors.semisafe
            try:
                cmap = { k : c[i] for i,k in enumerate(datadict.keys()) }
            except:
                c = plotter.colors.qualitative*3
                cmap = { k : c[i] for i,k in enumerate(datadict.keys()) }
        elif cmap in plotter.colormaps():
            cm = matplotlib.cm.get_cmap(cmap)
            n = len(datadict)
            cmap = {k: cm((i+0.5)/n) for i,k in enumerate(datadict.keys())}
        
        if lstmap is None:
            if ls is None:
                lst = plotter.linestyles.lst7*10
                lstmap = {k:lst[i] for i,k in enumerate(datadict.keys())}
            else:
                lstmap = {k:'-' for i,k in enumerate(datadict.keys())}
        elif type(lstmap) is str:
            lstmap = {k:lstmap for k in datadict}
        if ylabel is not  None:
            ylabel=ylabel
        if pmap is None:
            pmap = {k:'o' for k in datadict}
        if cutf is None:
            cutf ={k:10**10 for k in datadict}
        
       
        
        figsize = (size,size)
        dpi = 300
        fig,ax=plt.subplots(figsize=figsize,dpi=dpi)
        plt.minorticks_on()
        plt.tick_params(direction='in', which='minor',length=size*1.0)
        if minorxticks is not None:
            plt.tick_params(direction='in',axis='x', which='minor',length=size*minorxticks)
        if minoryticks is not None:
            plt.tick_params(direction='in',axis='y', which='minor',length=size*minoryticks)
            
        plt.tick_params(direction='in', which='major',length=size*majorticks)
        plt.xticks(fontsize=ticksfont*size) 
        plt.yticks(fontsize=ticksfont*size) 
        if xticks is not None:
            plt.xticks(xticks,fontsize=2.0*size) 
        if yticks is not None:
            plt.yticks(yticks,fontsize=2.0*size) 
        if xscale =='log':
            plt.xscale('log')
        if yscale is not None:
            plt.yscale(yscale)
        if xlim is not None:
            if xscale =='log':
                xticks = [10**x for x in range(xlim[0],xlim[1]+1) ]
                plt.xticks(xticks,fontsize=min(ticksfont*size,ticksfont*size*8/len(xticks)))
                locmin = matplotlib.ticker.LogLocator(base=10.0,subs=np.arange(0.1,1,0.1),numticks=12)
                ax.xaxis.set_minor_locator(locmin)
                ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                plt.xlim([min(xticks),max(xticks)])
            else:
                plt.xlim(xlim[0],xlim[1])
                
        if ylim is None:
            plt.yticks(fontsize=2.0*size)
        else:
            if yscale =='log':
                yticks = [10**y for y in range(ylim[0],ylim[1]+1)]
                plt.yticks(yticks,fontsize=min(2.5*size,2.5*size*8/len(yticks)))
                locmin = matplotlib.ticker.LogLocator(base=10.0,subs=np.arange(0.1,1,0.1),numticks=12)
                ax.yaxis.set_minor_locator(locmin)
                ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                plt.ylim([min(yticks),max(yticks)])
            else:
                plt.ylim(ylim)
                
        if title is not None:
            plt.title(r'{:s}'.format(title),fontsize=3.2*size)
        plt.xlabel(xlabel,fontsize=2.3*size)
        plt.ylabel(ylabel,fontsize=2.3*size)
        
        for i,(k,dy) in enumerate(datadict.items()):
            x = np.array(dy[xaxis])
            y = np.array(dy[yaxis])
            t = x<=cutf[k]
            x = x[t]
            y = y[t]
            if moving_average is not None:
                if extra_line == False:
                    y = ass.moving_average(y,moving_average)
            if num == 'all':
                args = np.arange(0,x.shape[0],dtype='i')
            else:
                args = plotter.sample_logarithmically_array(x,midtime=midtime,num=num)
            
            if style == 'points':
                plt.plot(x[args],y[args],ls='none',
                marker = pmap[k],markeredgewidth=edgewidth*size,
                label=labmap[k], markersize=size*markersize,fillstyle=fillstyle,
                color=cmap[k])
            elif style=='lines':
                plt.plot(x[args],y[args],ls=lstmap[k],lw=size*lw,
                label=labmap[k],color=cmap[k])
            elif style=='linepoints':
                plt.plot(x[args],y[args],ls=lstmap[k],lw=size*lw,
                marker = pmap[k],markeredgewidth=edgewidth*size,
                label=labmap[k], markersize=size*markersize,fillstyle=fillstyle,
                color=cmap[k])
            else:
                raise NotImplementedError('Implement here you own plotting style. Use elif statement')
            if extra_line and moving_average is not None:
                y = ass.moving_average(y,moving_average)
                plt.plot(x[args],y[args],ls='-',lw=size*lw,color=cmap[k])
            if fits is not None:
                dy = fits[k]
                xf = dy[xaxis]
                yf = dy[yaxis]
                plt.plot(xf,yf,color=cmap[k],ls='-',lw=0.3*size)
            if add_point is not None:
                xp = dy[add_point]
                plt.plot([x[0],x[-1]],[xp,xp],ls='-.',lw=lw*size*0.4,color=cmap[k])
        if legend:
            plt.legend(frameon=False,fontsize=legfont*size,**legkwargs)
        if fname is not None:
            saveformats  = set(saveformats)
            saveformats.add(fname.split('.')[-1])
            for ext in saveformats:
                fn = '.'.join(fname.split('.')[:-1] + [ext])
                plt.savefig(fn,bbox_inches='tight')
        plt.show()
        return

    
    @staticmethod
    def fits(datadict,fitteddict,
             fname =None,title=None,size=3.5,ylim=None,xlim=(-6,6),pmap=None,
             cmap = None,ylabel=None,units='ns',midtime=None,show_leg=True,
             legkwargs=dict(frameon=False,ncol=2,fontsize=11),
             num=100,cutf=None,write_plot_data=False,marksize=1.2):

        if write_plot_data:
            plotter.write_data_forXPCS(datadict,cutf=cutf,midtime=midtime)
        
        if cmap is None:
            c = plotter.colors.semisafe
            try:
                cmap = { k : c[i] for i,k in enumerate(datadict.keys()) }
            except:
                c = plotter.colors.qualitative*3
                cmap = { k : c[i] for i,k in enumerate(datadict.keys()) }
        
        if pmap is None:
            pmap = {k:'o' for k in datadict}
        
        if ylabel is None:
            ylabel =r'$P_1(t)$'
        else:
            ylabel=ylabel
        if cutf is None:
            cutf ={k:10**10 for k in fitteddict}
        
        figsize = (size,size)
        dpi = 300
        fig,ax=plt.subplots(figsize=figsize,dpi=dpi)
        plt.minorticks_on()
        plt.tick_params(direction='in', which='minor',length=size*1.5)
        plt.tick_params(direction='in', which='major',length=size*3)
        plt.xscale('log')
        if ylim is not None:
            plt.ylim(ylim)
        xticks = [10**x for x in range(xlim[0],xlim[1]+1) ]
        plt.xticks(xticks,fontsize=min(2.5*size,2.5*size*8/len(xticks)))
        plt.yticks(fontsize=2.5*size)
        plt.xlabel(r'$t ({})$'.format(units),fontsize=3*size)
        plt.ylabel(ylabel,fontsize=3*size)
        locmin = matplotlib.ticker.LogLocator(base=10.0,subs=np.arange(0.1,1,0.1),numticks=12)
        ax.xaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        for i,(k,fitd) in enumerate(fitteddict.items()):
            dy = datadict[k]
            x = ass.numpy_keys(dy)/1000
            y = ass.numpy_values(dy)
            t = x<=cutf[k]
            x = x[t][1:]
            y = y[t][1:]
            
            args = plotter.sample_logarithmically_array(x,midtime=midtime,num=num)

 
            xf = np.logspace(xlim[0],xlim[1],base=10,num=10000)
            yee =fitd.fitted_curve(xf)
            plt.plot(xf,yee,ls ='-.',color=cmap[k],label ='fit {}'.format(k))
            
            plt.plot(x[args],y[args],ls='none',marker = pmap[k],markeredgewidth=0.2*size,
        markersize=size*marksize,fillstyle='none',color=cmap[k],label=k)
        if show_leg:
            plt.legend(frameon=False,**legkwargs)
        if fname is not None:plt.savefig(fname,bbox_inches='tight')
        plt.show()
        return

class inverseFourier():
    def __init__(self,t,ft,omega,omega_0=1e-16,omega_oo=1e16):
        self.t = t
        self.ft = ft
        self.omega = omega
        self.omega_0 = omega_0
        self.omega_oo = omega_oo
        return
    @staticmethod
    def derft(ft,t):
        d = np.empty(ft.shape[0],dtype=float)
        d[0] = (ft[1]-ft[0])/(t[1]-t[0])
        d[-1] = (ft[-2]-ft[-1])/(t[-1]-t[-2])
        for i in range(1,ft.shape[0]-1):
            d[i] = (ft[i+1]-ft[i-1])/(t[i+1]-t[i-1])
        return d
    
    def find_epsilon(self,normalize=True):
        

        def ep_epp(t,dft,o):
            I = -simpson(dft*np.exp(-1j*o*t),x=t)
            return I
        
        dft = self.derft(self.ft,self.t)
        if ass.iterable(self.omega):
            eps = [ ep_epp(self.t,dft,o) for o in self.omega]
            eps = np.array(eps)
        else:
            eps = ep_epp(self.t,dft,self.omega)
        if normalize:
            eps0 = ep_epp(self.t,dft,self.omega_0)
            epsoo = ep_epp(self.t,dft,self.omega_oo)
            eps = eps*(eps0-epsoo)+epsoo
        return eps
    

    
class fitLinear():

    def __init__(self,xdata,ydata,nlines=1):
        self.xdata = np.array(xdata)
        self.ydata = np.array(ydata)
        self.nlines = nlines
        self.fitlines()
    @staticmethod
    def piecewise_linear(x, x0, y0, k1, k2):
            return np.piecewise(x, [x < x0], 
                                [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])
    @staticmethod
    def costF(p,x,y):
        yp = fitLinear.piecewise_linear(x, *p)
        return np.sum((y-yp)**2/y**2)
    def find_slopes(self):
        slopes = [] 
        intersects = []
        xf = self.xyline[0]
        yf = self.xyline[1]
        a = xf.argsort()
        xf = xf[a]
        yf=yf[a]
        
        breaks = self.breaks[self.breaks.argsort()]
        for i,b in enumerate(breaks[1:]):
            bm = breaks[i]
            
            fb = np.logical_and(bm<xf,xf<b)
           
            x = xf[fb]
            y = yf[fb]
            
            slope = (y[-1]-y[0])/(x[-1]-x[0])
            ise = y[-1] -slope*x[-1]
            slopes.append(slope)
            intersects.append(ise)
        self.slopes = slopes
        self.intersections = intersects
        return

    def fitlines(self):
        import pwlf
        x = self.xdata
        y = self.ydata
        my_pwlf = pwlf.PiecewiseLinFit(x, y)
        breaks = my_pwlf.fit(self.nlines)
        self.breaks = breaks
        xd = np.arange(x.min(),x.max(),0.01)
        self.xyline =  (xd,my_pwlf.predict(xd))
        self.find_slopes()
        
        
        
        return 
class xifit():
    def __init__(self,tau,d,xirho=0.2):
        self.lntau = np.log(tau)
        self.tau = tau
        self.d = d
        self.xirho = xirho
        self.fit()
    @staticmethod
    def func(xi,c,lntau0,xirho,d):
        return c*np.tanh((d-xirho)/xi) + lntau0
    @staticmethod
    def dfunc(xi,c,lntau0,xirho,d):
        return c/np.cosh((d-xirho)/xi)**2/xi
    @staticmethod
    def costf(pars,d,xirho,lntau):
        pred = xifit.func(*pars,xirho,d)
        r = (((lntau-pred)/lntau)**2)
        return r.sum()/r.shape[0]
    def fit(self,):
        bounds = [(0,3),(-5,5),(-15,15)]
        
        from scipy.optimize import  differential_evolution
        opt_res = differential_evolution(self.costf, bounds,
                    args = (self.d,self.xirho,self.lntau) ,
                    maxiter = 10000)
        self.opt_res = opt_res
        self.p = opt_res.x
        self.d = np.arange(self.d.min(),self.d.max()+0.01,0.01)
        self.curve = np.exp(self.func(*self.p,self.xirho,self.d))
        self.dcurve = self.curve*self.dfunc(*self.p,self.xirho,self.d)
        return 
class Arrheniusfit():
    
    
    def __init__(self,temp,tau):
        self.tau = tau
        self.temp =  temp
        
        self.fit()
        t = np.arange(temp.min(),temp.max()+0.01,0.01)
        self.t = t
        self.curve = self.exp(t,*self.opt_res.x)
        return       
    @staticmethod
    def exp(temp,A,Ea):
        tau = A*np.exp(-Ea/temp)
        return tau
    @staticmethod
    def costf(pars,temp,tau):
        tau_pred = Arrheniusfit.exp(temp,*pars)
        return np.sum((np.log10(tau_pred)-np.log10(tau))**2)/tau.size
    
    def fit(self):
      #  pars = np.array([100,-1000])
        bounds = [(0,1e4),(-10e4,0)]
        
        from scipy.optimize import  differential_evolution
        opt_res = differential_evolution(self.costf, bounds,
                    args = (self.temp,self.tau) ,
                    maxiter = 1000)
        self.opt_res = opt_res
        self.p = opt_res.x
        self.t = np.arange(self.temp.min(),self.temp.max()+0.01,0.01)
        self.curve = self.exp(self.t,*self.p)
        return 

# lalala 

class VFTfit():
    def __init__(self,temp,tau,pars=[],pmin=None,pmax=None):
        self.tau = tau
        self.temp =  temp
        if len(pars)==0:    
            self.fitVFT()
        elif len(pars)==3:
            self.p = np.array(pars)
        if pmin is None:
            pmin = temp.min()
        if pmax is None:
            pmax = temp.max()
        t = np.arange(pmin,pmax+0.01,0.01)
        self.t =t
        self.curve = self.vft(t,*self.p)
        return       
    @staticmethod
    def vft(temp,A,D,t0):
        tau = A*np.exp(D/(temp-t0))
        return tau
    @staticmethod
    def costf(pars,temp,tau):
        tau_pred = VFTfit.vft(temp,*pars)
        return np.sum((np.log10(tau_pred)/np.log10(tau)-1)**2/tau**0.5)/tau.size
    
    def fitVFT(self):
        from scipy.optimize import differential_evolution#,dual_annealing
     #   pars = np.array([1,1,50])
        bounds = [(0,1e-3),(1e2,1e4),(0,180)]
        opt_res = differential_evolution(self.costf, bounds,
                    args = (self.temp,self.tau) ,
                    maxiter = 10000)
        self.opt_res = opt_res
        self.p = opt_res.x
        return 
class fitData():
    
    from scipy.optimize import minimize
    
    def __init__(self,xdata,ydata,func,method='distribution',bounds=None,
                 search_grid=None,reg_bounds=None,keep_factor=2,bias_factor=1.0,
                 nw=50,bound_res=1e-9,maxiter=200,sigmF=[10,20,30,40,50,60],
                 show_plots=False,**options):
        
        self.show_plots = show_plots
        
        
        self.method = method
        self.ydata = ydata
        self.xdata = xdata
        self.mode = func
        self.func = getattr(fitFuncs,func)
        
        if method == 'distribution':
            self.kernel = getattr(fitKernels,func)
        
        if bounds is not None:
            self.bounds = bounds
        else:
            self.bounds = (1e-7,1e7)
        
        if search_grid is not None:
            self.search_grid= search_grid
        else:
            self.search_grid= (12,4,4)
        
        if reg_bounds is not None:
            self.reg_bounds = reg_bounds
        else:
            self.reg_bounds = (1e-20,1e8)
        self.keep_factor = keep_factor
        self.bias_factor = bias_factor
        self.minimum_res=bound_res
        self.nw = nw
        self.maxiter=maxiter
        self.sigmF = sigmF
        self.sigma_factor = 30
        if 'weights' in options:
            self.weights = options['weights']
        if 'weighting_method' in options:
            self.weighting_method = options['weighting_method']
        else:
            self.weighting_method = 'xy'
        if 'is_distribution' in options:
            self.is_distribution = options['is_distribution']
        else:
            self.is_distribution = True
            
        if 'zeroEdge_distribution' in options:
            self.zeroEdge_distribution = options['zeroEdge_distribution']
        else:
            self.zeroEdge_distribution = True
        
        if 'show_report' in options:
            self.show_report = options['show_report']
        else:
            self.show_report = True
        if 'init_method' in options:
            self.init_method = options['init_method']

        else:
            self.init_method = 'normal'
        if 'p0' in options:
            self.p0 = options['p0']    
        if 'opt_method' in options:
            self.opt_method = options['opt_method']
        else:
            self.opt_method = 'SLSQP'
            
        
        return
    
    @property
    def keep_res(self):
        return self.keep_factor*self.minimum_res
    
    def clean_positive_derivative_data(self):
        der = np.empty_like(self.ydata)
        der[1:] = (self.ydata[1:]-self.ydata[:-1])/(self.xdata[1:]-self.xdata[:-1])
        der[0]=der[1]
        f = der< 0
        self.xdata= self.xdata[f]
        self.ydata = self.ydata[f]
        return
    def clean_non_monotonic_data(self):
        x = self.xdata
        y = self.ydata
        f = np.ones(x.size,dtype=bool)
        for i in range(x.size):
            d = y[i]<y[i+1:]
            if d.any():
                f[i] = False
        self.xdata= self.xdata[f]
        self.ydata = self.ydata[f]
        return
    
    def clean_data(self):
        if True:
            self.clean_positive_derivative_data()
        if True:
            self.clean_non_monotonic_data()
        if True:
            izero = self.get_arg_t0()
            if  izero != self.ydata.size:
                self.ydata = self.ydata[:izero]
                self.xdata = self.xdata[:izero]
        return
    def fitTheModel(self):
        self.clean_data()
            
        self.justFit()
        self.estimate_minimum_residual()

        self.taulow = self.estimate_taulow()
        self.tauhigh = self.estimate_tauhigh()


        self.refine_bounds(self.tauhigh)
        self.search_best()
        return
    
    def save_for_XPCS(self,fname):
        x = self.xdata
        y = self.ydata
        data = np.zeros((x.shape[0],3),dtype=float)
        data[:,0] = x
        data[:,1] = y +1
        data[:,2]+= np.random.uniform(0,1e-9,x.shape[0])
        np.savetxt(fname,data)
        return
    def estimate_tauhigh(self):
        smfittau = self.smootherFit()
        if self.get_arg_t0() == self.ydata.size:
            # Need to extrapolate
            tau = smfittau
        else:
            tau = self.estimate_taudata()
        return 1e2*tau
    
    def estimate_taulow(self):
        smalltau = self.smallerTauRelaxFit()
        taudata = self.estimate_taudata()
        tau = min(smalltau,taudata) if smalltau !=0 else taudata
        return 10**(-2*(1-self.lasty))*tau
    
    def estimate_taudata(self):
        x = self.xdata
        y = self.ydata
        izero = self.get_arg_t0()
        dt = x[1:izero] - x[:izero-1]
        yi = y[:izero-1]
        return np.sum(yi*dt)
        
        
    def estimate_minimum_residual(self):
        mres = max(self.minimum_res,1*self.data_res)
        self.minimum_res = mres
        
        print('estimated data residual = {:.4e}'.format(mres))
        return
    def get_arg_t0(self):
        f = self.ydata <= 1e-2
        smallys = np.where(f)[0]
        
        if len(smallys) ==0:
            return self.ydata.size
        izero = smallys[0]
        return izero
    
    def refine_bounds(self,th):
        if self.mode =='freq':
            bh = self.bounds[1]
            bl = 1/(th*1e5)
        else:
            bl = self.bounds[0]
            bh = th*1e5
        self.bounds = (bl,bh)
        return 
    def get_weights(self):
        if self.weighting_method == 'high-y':
            w = np.abs(self.ydata)+0.3
        elif self.weighting_method=='xy':
            w=1#self.xdata**0.1 +self.ydata
            w = w
        else:
            w = np.ones(self.ydata.shape[0])
        return w  
    
    @property
    def bestreg(self):

        a = self.bestcrit() 

        regbest =  self.regs[a]
            
        return regbest
    def search_best(self):
        self.crit = []
        self.storing_list=  ['relaxation_modes','con_res','data_res',
                   'params','prob_distr','loss','smoothness','tau_relax',
                   'bias','sigma_factor']
        self.storing_dict = {k:[] for k in self.storing_list}
        
        for st in self.storing_list:
            setattr(self,'best_'+st,None)
            

        tl = self.taulow
        th = self.tauhigh
        print('trelax bounds: [{:.3e} , {:.3e}]'.format(tl,th))        
        if tl > th:
            raise Exception('Minimizing the relaxation time gives higher tau than minimizing the smoothness')
        

        self.bestcr = float('inf')
        
        tau_bounds = (tl,th)
        for numreg in self.search_grid:
            #print(tau_bounds)
            dilog = (np.log10(tau_bounds[1])-np.log10(tau_bounds[0]))/numreg
            
            self.search_reg(numreg,tau_bounds)
            
            f = np.array(self.storing_dict['data_res']) < self.keep_res
            c = np.array(self.crit)[f]
            re = np.array(self.storing_dict['tau_relax'])[f]
            rem = re[c.argmin()]
            th = rem*10**dilog
            tl = rem*10**(-dilog)
            tau_bounds = (tl,th)
            
            
        for st in self.storing_list:
            setattr(self,st,getattr(self,'best_'+st))
        #rgbest = self.bestreg


        return
    
    def search_reg(self,numreg,taub):
        
        for attr in ['crit','smv','drv','regs']:
            try:
                getattr(self,attr)
            except AttributeError:
                setattr(self,attr,[])
                
        target_taus = np.logspace(np.log10(taub[0]),np.log10(taub[1]),base=10,num=numreg)       
        #print(target_taus)
        for tartau in target_taus:
            
            self.target_tau = tartau
            self.refine_bounds(tartau)
            for sigmF in self.sigmF:
                self.sigma_factor = sigmF
                self.exactFit(tartau)
            
                s = self.smoothness ; t = self.tau_relax ; d = self.data_res 
                for k in self.storing_list:
                    self.storing_dict[k].append(getattr(self,k))
                

                crt = self.criterium(d,s,t)

                self.crit.append(crt)
    
                self.nsearches = len(self.crit)

        self.refine_keep_res()
        
        self.select_best_solution()
        
        return
    @property
    def lasty(self):
        return self.ydata[-1]
    
    def criterium(self,d,s,t):
        return d**2*s*t**(1-self.lasty)
    
    def refine_keep_res(self):
        dres = np.array(self.storing_dict['data_res'])
        fd = dres < self.keep_res
        i=0
        factor = 1.1
        
        while fd.any() == False:
            self.keep_factor=self.keep_factor*factor
            i+=1
            fd =  dres < self.keep_res            
            if self.keep_res>0.1: 
                raise Exception('Increased keep_res too much and still no solution satysfies it')
        #print('Increased keep_res by {:4.3f} times'.format(factor**i)) 
        return   
    def select_best_solution(self):
        f = np.array(self.storing_dict['data_res']) < self.keep_res
        a = np.array(self.crit)[f].argmin()
        best = dict()
        for st in self.storing_list:
            attr = np.array(self.storing_dict[st])[f][a]
            if type(attr) is type(np.ones(3)): 
                attr = attr.copy()
            best[st] = attr
            setattr(self,'best_'+st,attr)
        self.best_sol = best
        return 
    
    @staticmethod
    def get_logtimes(a,b,n):
        tau = np.logspace(np.log10(a),np.log10(b),base=10,num=n)
        return tau
    
    def initial_params(self):
        if self.init_method =='uniform':
            pars = np.ones(self.nw)/self.nw 
        elif self.init_method =='from_previous':
            try:
                pars = self.params
            except:
                pars =  np.ones(self.nw)/self.nw
        elif self.init_method =='ones':
            pars = np.ones(self.nw)
        elif self.init_method=='zeros':
            pars = np.zeros(self.nw) +1e-15
        elif self.init_method=='normal':
            x = np.arange(0,self.nw,1)
            mu = self.nw/2
            sigma = mu/self.sigma_factor
            pars = np.exp(-(x-mu)**2/(sigma)**2)
            pars /=pars.sum()
        return pars
    
    def distribution_constraints(self):
        constraints = []
        
        if self.is_distribution:
            cdistr = lambda w: 1 - np.sum(w[:-1])
            def dcddw(w):
                dw = np.zeros(w.shape)
                dw[:-1] = -1.0
                return dw
            constraints.append({'type':'eq','fun':cdistr,'jac': dcddw})
        
        if self.zeroEdge_distribution:
            w0 = lambda w:  -w[0] 
            wn = lambda w:  -w[-2]
            def dw0dw(w):
                x = np.zeros(w.shape)
                x[0] = -1
                return x
            def dwndw(w):
                x = np.zeros(w.shape)
                x[-2] = -1
                return x
            constraints.append({'type':'eq','fun':w0,'jac':dw0dw})
            constraints.append({'type':'eq','fun':wn,'jac':dwndw})
        return constraints
    
    def get_them(self):
        n = self.nw
        tau = fitData.get_logtimes(self.bounds[0],self.bounds[1],n)
        dlogtau = np.log10(tau[1]/tau[0]) 
        x = self.xdata.copy()
        y = self.ydata.copy()
        A = self.kernel(x,tau)
        whs = self.get_weights()
        for i in range(A.shape[1]):
            A[:,i] = A[:,i]*whs
        return n,tau,dlogtau,  whs*y, A
    
    def get_params(self):
        p0 = self.initial_params()
        bounds = [(0,1) for i in range(self.nw)]
        
        p0 = np.concatenate((p0,[0.0]))
        ms = self.minimum_res
        
        bounds.append( ( -1*ms*self.bias_factor,1*ms*self.bias_factor ) )
        
        return p0,bounds
    
    def evaluateNstore(self,opt_res):
        n, tau, dlogtau, y, A = self.get_them()

        w = opt_res.x[:-1]
        self.bias = opt_res.x[-1]

        isd = 1-w.sum() ; bl = w[0] ; bu = w[-1]
        self.prob_distr = {'isd':isd,'blow':bl,'bup':bu}
        
        self.con_res = compute_residual(opt_res.x,A,y)
        self.data_res = self.con_res
        
        self.params = w
        self.opt_res = opt_res
        self.relaxation_modes = tau
        self.smoothness = smoothness(opt_res.x,dlogtau)
        self.loss = opt_res.fun
        self.tau_relax = self.trelax
        if self.show_plots:
            self.show_relaxation_modes(prefix='',)
            self.show_relaxation_modes(prefix='',show_contributions=True,yscale='log')
            self.show_fit()

        if self.show_report:
            self.report()
        return
    def justFit(self):
        
        n, tau, dlogtau, y, A = self.get_them()

        constraints = self.distribution_constraints()
        
        p0,bounds = self.get_params()
        
        costf = compute_residual
        
        opt_res = minimize(costf,p0,
                          args=(A,y),method = self.opt_method,
                          constraints=constraints,
                          bounds=bounds,
                          jac=dCRdw,
                          options = {'maxiter':int(self.maxiter/2),'disp':self.show_report}, 
                          tol=1e-16)
        
        self.evaluateNstore(opt_res)
        return opt_res
    
    def smallerTauRelaxFit(self):
        n, tau, dlogtau, y, A = self.get_them()
        constraints = [   {'type':'ineq',
                         'fun': constraint,
                         'jac': dCdw,
                         'args':(A,y,self.minimum_res)
                        } 
                      ]
        
        constraints.extend(self.distribution_constraints())
        
        p0,bounds = self.get_params()
        
        costf = FrelaxCost
              
        opt_res = minimize(costf,p0,
                          args=(tau,),method = self.opt_method,
                          constraints=constraints,
                          bounds=bounds,
                          #jac=dFdw,
                          options = {'maxiter':self.maxiter,'disp':self.show_report},
                          tol=1e-6)
        
        self.evaluateNstore(opt_res)
        return self.trelax
    
    def smootherFit(self):
        n, tau, dlogtau, y, A = self.get_them()
        constraints = [   {'type':'ineq',
                         'fun': constraint,
                         'jac': dCdw,
                         'args':(A,y,self.minimum_res)
                        } 
                      ]
        
        constraints.extend(self.distribution_constraints())
        
        
        p0,bounds = self.get_params()
        
        costf = smoothness
              
        opt_res = minimize(costf,p0,
                          args=(dlogtau,),method = self.opt_method,
                          constraints=constraints,
                          bounds=bounds,
                          jac=dSdw,
                          options = {'maxiter':self.maxiter,'disp':self.show_report},
                          tol=1e-6)
        
        self.evaluateNstore(opt_res)
        return self.trelax
    
    
    def exactFit(self,target_tau):

        n, tau, dlogtau, y, A = self.get_them()
        
        constraints = [   {'type':'ineq',
                         'fun': constraint,
                         'jac': dCdw,
                         'args':(A,y,self.minimum_res)
                        } ,
                       {'type':'eq',
                         'fun': FrelaxCon,
                         'jac': dFCdw,
                         'args':(tau,target_tau)
                        }
                      ]
        
        
        constraints.extend(self.distribution_constraints())
        constraints.extend(self.contribution_constraints(target_tau,tau))    
        p0,bounds = self.get_params()
        
        costf = FitCost
              
        opt_res = minimize(costf,p0,
                          args=(dlogtau,),method = self.opt_method,
                          constraints=constraints,
                          bounds=bounds,
                          jac=dFitCostdw,
                          options = {'maxiter':self.maxiter,'disp':self.show_report},
                          tol=1e-6)
        
        self.evaluateNstore(opt_res)
        return opt_res
    
    def contribution_constraints(self,target_tau,tau):
        constraints = []
        return constraints
    def show_residual_distribution(self,fname=None,size=3.5,title=None):
        figsize = (size,size)
        dpi = 300
        fig,ax=plt.subplots(figsize=figsize,dpi=dpi)
        plt.xscale('log')
        if title is not None:
            plt.title(title)
        plt.ylabel(r'number of occurances')
        plt.xlabel(r"$residual$")
        plt.minorticks_on()
        plt.tick_params(direction='in', which='minor',length=size*1.5)
        plt.tick_params(direction='in', which='major',length=size*3)
        plt.hist(self.storing_dict['data_res'],bins=100,color='k')
        plt.show()
        return 
    def show_tstar(self,tmax,n=1000,size=3.5,title=None):
        figsize = (size,size)
        dpi = 300
        fig,ax=plt.subplots(figsize=figsize,dpi=dpi)
        
        dt = tmax/n
        t = np.arange(0,tmax,dt)
        tstar = [ self.tstar(ts) for ts in t]
        plt.yscale('log')
        plt.xscale('log')
        if title is not None:
            plt.title(title)
        plt.xlabel(r'$t^*$')
        plt.ylabel(r"$t^{*}_{relax}$")
        plt.minorticks_on()
        plt.tick_params(direction='in', which='minor',length=size*1.5)
        plt.tick_params(direction='in', which='major',length=size*3)
        plt.plot(t,tstar,color='k',marker='.')
        plt.show()
        return 
    def get_wrm(self):
        try:
            w=self.params
        except AttributeError as err:
             raise err
        try:
            rm = self.relaxation_modes
        except AttributeError as e:
            raise e
        return w,rm
    def phit(self,ts):
        w, rm = self.get_wrm()
        f = lambda w,rm : np.sum(w*np.exp(-ts*rm))
        t = lambda w,rm : np.sum(w*np.exp(-ts/rm))
        p = f(w,rm)  if self.mode =='freq' else t(w,rm)
        
        return p 
    
    def dertstar(self,ts):
        dts = 1 - self.phit(ts)
        
        return dts
    def dtstar(self,ts):
        w, rm = self.get_wrm()
        f = lambda w,rm : np.sum((1.0-np.exp(-ts*rm))*w)
        t = lambda w,rm : np.sum(w*(1.0-np.exp(-ts/rm)))
        tr = f(w,rm)  if self.mode =='freq' else t(w,rm)
        
        return tr
    def tstar(self,ts):
        w, rm = self.get_wrm()
        f = lambda w,rm : np.sum((1.0-np.exp(-ts*rm))*w/rm)
        t = lambda w,rm : np.sum(rm*w*(1.0-np.exp(-ts/rm)))
        tr = f(w,rm)  if self.mode =='freq' else t(w,rm)
        
        return tr
    
    @property
    def tmax(self):
        w, rm = self.get_wrm()
        a = w.argmax()
        tr = 1/rm[a]  if self.mode =='freq' else rm[a]
        return tr
    
    @property
    def trelax(self):
        w, rm = self.get_wrm()
        tr = Frelax(w,rm) if self.mode =='freq' else Trelax(w,rm)
        return tr
    
    @property
    def print_trelax(self):
        print('For bounds {}: --> trelax = {:.4e} ns'.format(self.bounds,self.trelax))
   
    def report(self):
        for k in ['prob_distr','con_res','data_res',
                  'smoothness','loss','trelax',
                  'target_tau','nsearches','bias']:
            try:
                a = getattr(self,k)
            except:
                pass
            else:
                print('{:s} = {}'.format(k,a))
        return
    

      
    def print_reg(self):
        print('best reg = {:.6e}'.format(self.bestreg))
        return
    
    def show_Pareto_front(self,size=3.5,
                              title=None,color='magenta',fname=None):
        
        figsize = (size,size)
        dpi = 300
        fig,ax=plt.subplots(figsize=figsize,dpi=dpi)
        plt.minorticks_on()
        plt.tick_params(direction='in', which='minor',length=size*1.5)
        plt.tick_params(direction='in', which='major',length=size*3)
        d = np.array(self.storing_dict['data_res'])
        si = np.array(self.storing_dict['smoothness'])
        ti = np.array(self.storing_dict['tau_relax'])
        
        filt = d <= self.keep_res
        s = si[filt]
        t = ti[filt]
        
        nf = np.logical_not(filt)
        ns = si[nf]
        ts = ti[nf]
        plt.xscale('log')
        plt.yscale('log')
        plt.yticks(fontsize=2.5*size)
        plt.xlabel(r'smoothness',fontsize=3*size)
        plt.ylabel(r'$\tau_{relax}$',fontsize=3*size)
        if title is None:
            plt.title('Pareto Front')
        else:
            plt.title(title)
        plt.plot(s,t,label='accepted',
                 ls='none',marker='o',color='green',markersize=1.7*size,fillstyle='none')
        
        pareto = []
        for i,(si,ti) in enumerate(zip(s,t)):
            fs = si > s 
            ft = ti > t
            f = np.logical_and(fs,ft)
            if f.any(): continue
            pareto.append(i)
        p = np.array(pareto,dtype=int)
        sp = s[p] ; tp =t[p]
        ser = sp.argsort()
        sp = sp[ser] ; tp = tp[ser]
        #plt.ylim([tp.min()/3,tp.max()*3.0])
        #plt.xlim([sp.min()/3,sp.max()*3.0])
        plt.plot(ns,ts,color = 'red',label='rejected',ls='none',
                 marker='o',fillstyle='none',markersize=1.7*size)
        plt.plot(sp,tp,ls='--',color='blue',lw=size/5,label='Opt. front')
        
        plt.plot([self.best_smoothness],[self.best_tau_relax],marker='o',
                 color='blue',markersize=1.7*size,label='selected')

        plt.xticks(fontsize=2.5*size)
        plt.legend(frameon=False,fontsize=1.5*size)
        if fname is not None:
            plt.savefig(fname,bbox_inches='tight')
        plt.show()
        return
    def eps_omega(self,omega=[]):
        n = len(omega)
        eps_real = np.empty(len(omega),dtype=float)
        eps_imag = np.empty(len(omega),dtype=float)
        w,f = self.get_wrm()
        if self.mode != 'freq':
            f =1/f
        for i in range(n):
            eps_real[i] = np.sum(w*f/(omega[i]**2+f**2))
            eps_imag[i] = -np.sum(-w*omega[i]/(omega[i]**2+f**2))
        tr = self.trelax
        eps_real /= tr
        eps_imag /= tr
        self.omega = omega
        self.eps_real = eps_real
        self.eps_imag = eps_imag
        return eps_real,eps_imag
    def omega_peak(self,omega):
        eps_real, eps_imag = self.eps_omega(omega)
        der = np.empty_like(eps_imag)
        der[0] = eps_imag[1]-eps_imag[0]
        der[-1] = eps_imag[-2] - eps_imag[-1]
        der[1:]= eps_imag[1:]-eps_imag[:-1]
        sign_change = []
        for i in range(1,der.size):
            if der[i-1]*der[i]<0:
                sign_change.append(i)
        if self.mode =='freq': arg = 0
        else: arg = -1
        if len(sign_change)==0:
            return omega[0]
        
        return 1/omega[sign_change[arg]]
    def show_eps_omega(self,omega,e='imag',size=3.5,units='ns',yscale=None,
                              title=None,color='red',fname=None,
                              prefix='best_'):
        

        eps_real,eps_imag = self.eps_omega(omega)
        if e=='imag':
            eps = eps_imag
        elif e=='real':
            eps = eps_real
            yscale=None
        else:
            raise ValueError('option e = "{:s}" not known'.format(e))
        
        figsize = (size,size)
        dpi = 300
        fig,ax=plt.subplots(figsize=figsize,dpi=dpi)
        plt.minorticks_on()
        plt.tick_params(direction='in', which='minor',length=size*1.5)
        plt.tick_params(direction='in', which='major',length=size*3)
        plt.xscale('log')
        if yscale is not None:
            plt.yscale(yscale)
        if yscale =='log':
            y0 = -5
            ym = int(np.log10(max(eps))+1)    
            plt.yticks([10**y for y in range(y0,ym )])
            plt.ylim(10**y0,10**ym)
        plt.yticks(fontsize=2.5*size)
        locmin = matplotlib.ticker.LogLocator(base=10.0,subs=np.arange(0.1,1,0.1),numticks=12)
        ax.xaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        if e=='imag':
            ylabel=r'$e^{\prime\prime}(\omega)$'
        elif e =='real':
            ylabel = r"$e^{\prime}(\omega)$"
        plt.ylabel(ylabel,fontsize=3*size)
        if self.mode=='freq':
            units = r'${:s}^{:s}$'.format(units,'{-1}')
            lab = r'$f$ / {:s}'.format(units)
        elif self.mode =='tau':
            lab = r'$\tau$ / {:s}'.format(units)
        plt.xlabel(lab,fontsize=3*size)
        if title is not None:
            plt.title(title)
        
        plt.plot(omega,eps,
                 ls='-',marker='o',color=color,markersize=1.3*size,lw=0.2*size,fillstyle='none')
        xticks = [10**x for x in range(-10,20) if omega[0]<=10**x<=omega[-1] ]
        plt.xticks(xticks,fontsize=min(2.5*size,2.5*size*8/len(xticks)))
        if fname is not None:
            plt.savefig(fname,bbox_inches='tight')
        plt.show()
        return
    def show_fit(self,size=3.5,units='ns',yscale=None,
                              title=None,color='red',fname=None,
                              prefix='best_'):
        
        xlim = 10
        xf = np.logspace(-4,xlim,base=10,num=10000)
        yee = self.fitted_curve(xf)
        figsize = (size,size)
        dpi = 300
        fig,ax=plt.subplots(figsize=figsize,dpi=dpi)
        plt.minorticks_on()
        plt.tick_params(direction='in', which='minor',length=size*1.5)
        plt.tick_params(direction='in', which='major',length=size*3)
        plt.xscale('log')
        
        xticks = [10**x for x in range(-4,xlim+1) ]
        plt.xticks(xticks,fontsize=2.5*size)
        plt.yticks(fontsize=2.5*size)
        plt.xlabel(r'$t (ns)$',fontsize=3*size)
        plt.ylabel(r'$P_1(t)$',fontsize=3*size)
        locmin = matplotlib.ticker.LogLocator(base=10.0,subs=np.arange(0.1,1,0.1),numticks=12)
        ax.xaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        
        plt.plot(xf,yee,ls ='-.',color=color,label ='fit')
        plt.ylim((-0.05,1))
        plt.plot(self.xdata,self.ydata,ls='none',marker = 'o',
            markersize=size*0.8,fillstyle='none',color=color,label='data')
        plt.legend(frameon=False,bbox_to_anchor=(1,1),fontsize=2.3*size)
        if fname is not None:
            plt.savefig(fname,bbox_inches='tight')
        plt.show()
        return
        
    def show_relaxation_modes(self,size=3.5,units='ns',yscale=None,
                              title=None,color='red',fname=None,
                              show_contributions=False,prefix='best_'):
        
        rm = getattr(self,prefix+'relaxation_modes')
        pars = getattr(self,prefix+'params')
        
        figsize = (size,size)
        dpi = 300
        fig,ax=plt.subplots(figsize=figsize,dpi=dpi)
        plt.minorticks_on()
        plt.tick_params(direction='in', which='minor',length=size*1.5)
        plt.tick_params(direction='in', which='major',length=size*3)
        plt.xscale('log')
        if yscale is not None:
            plt.yscale(yscale)
        if yscale =='log':
            y0 = -5
            if show_contributions:
                contr = pars/rm if self.mode == 'freq' else pars*rm
                ym = int(np.log10(max(contr))+1)    
            plt.yticks([10**y for y in range(y0,ym )])
            plt.ylim(10**y0,10**ym)
        plt.yticks(fontsize=2.5*size)
        locmin = matplotlib.ticker.LogLocator(base=10.0,subs=np.arange(0.1,1,0.1),numticks=12)
        ax.xaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        plt.ylabel('w',fontsize=3*size)
        if self.mode=='freq':
            units = r'${:s}^{:s}$'.format(units,'{-1}')
            lab = r'$f$ / {:s}'.format(units)
        elif self.mode =='tau':
            lab = r'$\tau$ / {:s}'.format(units)
        plt.xlabel(lab,fontsize=3*size)
        if title is None:
            plt.title('Relaxation times distribution')
        else:
            plt.title(title)
        
        plt.plot(rm,pars,
                 ls='-',marker='o',color=color,markersize=1.3*size,lw=0.2*size,fillstyle='none')
        if show_contributions:
            contr = pars/rm if self.mode == 'freq' else pars*rm
            plt.plot(rm,contr,label='contribution',color='k',
                     ls='-',lw=0.1*size,marker='.',fillstyle='none',markersize=1.5*size,
                     markeredgewidth=size*0.5)
        xticks = [10**x for x in range(-10,20) if self.bounds[0]<=10**x<=self.bounds[1] ]
        plt.xticks(xticks,fontsize=min(2.5*size,2.5*size*8/len(xticks)))
        if fname is not None:
            plt.savefig(fname,bbox_inches='tight')
        plt.show()
        return
    
    def fitted_data(self):
        return self.fitted_curve(self.xdata)
    
    def fitted_curve(self,x):
        if self.method =='distribution':
            return self.func(x,self.bounds[0],self.bounds[1],self.params) + self.bias
        else:
            return self.func(x,*self.params)
        
    
class fitKernels():
    @staticmethod
    def freq(t,f):
        return np.exp(-np.outer(t,f))
    @staticmethod
    def tau(t,tau):
        return np.exp(-np.outer(t,(1/tau)))

class fitFuncs():
    @staticmethod
    def gauss(t,a,b,w):
        frqs = fitData.get_logtimes(a,b,len(w))
        s=0
        for i,fr in enumerate(frqs):
            s+= w[i]*np.exp(-t*t*fr)
        return s
    @staticmethod
    def freq(t,a,b,w):
        frqs = fitData.get_logtimes(a,b,len(w))
        s=0
        for i,fr in enumerate(frqs):
            s+= w[i]*np.exp(-t*fr)
        return s
    @staticmethod
    def tau(t,a,b,w):
        taus = fitData.get_logtimes(a,b,len(w))
        s=0
        for i,ta in enumerate(taus):
            s+= w[i]*np.exp(-t*ta)
        return s
      
    @staticmethod
    def KWW(t,tww,beta,A=1):
        #Kohlrausch–Williams–Watts
        #beta changes the shape of the curve. 
        #very small beta makes the curve linear.
        #the larger the beta the sharpest the curve
        #all curves regardless beta pass through the same point
        phi = A*np.exp( -(t/tww )**beta )
        return phi
    
    @staticmethod
    def KWW2(t,tw1,tw2,b1,b2,A1=1,A2=1):
        phi1 = fitFuncs.KWW(t,tw1,b1,A1)
        phi2 = fitFuncs.KWW(t,tw2,b2,A2)
        return phi1 + phi2
    
    @staticmethod
    def exp(t,t0,A=1):
        #A parameter shifts the starting point to A
        phi = A*np.exp(-t/t0)
        return phi
    
class Analytical_Expressions():

    
    @staticmethod
    def expDecay_sum(t,t0v):
        s = np.zeros(t.shape[0])
        t0v = np.array(t0v)
        for i,t0 in enumerate(t0v):
            s+=Analytical_Expressions.expDecay_simple(t,t0)
        return s/t0v.shape[0]
    @staticmethod
    def expDecay_simple(t,t0):
        phi =  np.exp(-t/t0)
        return phi
    @staticmethod
    def expDecay(t,A,t0):
        #A paramater shifts the end point of the curve up to minus A, see exersize 2
        phi = 1+A*( np.exp(-t/t0) - 1 )
        return phi
    @staticmethod
    def expDecay2(t,A,t0):
        #A parameter shifts the starting point to A
        phi = A*np.exp(-t/t0)
        return phi
    @staticmethod
    def expDecay3(t,A,t0):
        #A parameter shifts and the end point to point to -A
        phi = A*(np.exp(-t/t0)-1)
        return phi  
    @staticmethod
    def expDecay4(t,As,Ae,t0):
        #As shifts initial point to As
        #Ae shifts the whole curve by Ae
        phi = As*np.exp(-t/t0)-Ae
        return phi 
    @staticmethod
    def expDecay_KWW(t,A1,A2,tc,t0,beta,tww):
        tl =  t[t<tc]
        tup = t[t>=tc]
        phil = Analytical_Expressions.expDecay(tl,A1,t0)
        A2 = Analytical_Expressions.expDecay(tc,A1,t0) #continuity
        phiup = Analytical_Expressions.KWW(tup,A2,tc,beta,tww)
        return np.concatenate( (phil,phiup) )
    
@jitclass
class Analytical_Functions():
    '''
    This class is used exclucively to add hydrogens on correct locations
    and be able to compute all atom properties like dipole momement vectors (dielectric data)
    Currently works well for PB. It is NOT tested for anything else. 
    It needs attention to the geometry calculations and should be used
    with the add_atoms class.
    '''
    def __init__(self):
        pass

    @staticmethod
    def Rz(th):
        R = np.array([[1, 0 ,0], 
            [0, np.cos(th), -np.sin(th)], 
            [0, np.sin(th), np.cos(th)]])
        return R
    @staticmethod
    def Ry(th):
        R = np.array([[np.cos(th), 0, np.sin(th)] ,
                       [0, 1, 0], 
                       [-np.sin(th), 0, np.cos(th)]])
        return R
    @staticmethod
    def Rx(th):
        R = np.array([[np.cos(th), -np.sin(th), 0],
                     [np.sin(th), np.cos(th), 0],
                     [0,0,1]])
        return R
        
 
    
    @staticmethod
    #@jit(nopython=True,fastmath=True)
    def q_mult(q1,q2):
        w1 = q1[0] ; x1 = q1[1] ; y1 = q1[2] ; z1 = q1[3]
        w2 = q2[0] ; x2 = q2[1] ; y2 = q2[2] ; z2 = q2[3]
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
        return np.array((w, x, y, z))
    
    @staticmethod
    #@jit(nopython=True,fastmath=True)
    def quaternionConjugate(q):
        w = q[0] ; x = q[1] ; y = q[2] ; z = q[3]
        return np.array((w, -x, -y, -z))
    
    @staticmethod
    #@jit(nopython=True,fastmath=True)
    def rotate_around_an_axis(axis,r,theta):
        q1 = (0,r[0],r[1],r[2])
        th2 = 0.5*theta
        c = np.cos(th2)
        s = np.sin(th2)
        
        q2 = np.array((c,s*axis[0],s*axis[1],s*axis[2]))
        
        q2c = Analytical_Functions().quaternionConjugate(q2)
        q = Analytical_Functions().q_mult(q1,q2c)
        q3 = Analytical_Functions().q_mult(q2,q)
        return np.array((q3[1],q3[2],q3[3]))
    
    @staticmethod
    #@jit(nopython=True,fastmath=True)
    def rotate_to_theta_target(rp,r0,rrel,theta_target):
        th0 = calc_angle(rp,r0,r0+rrel)
        theta = theta_target - th0
        naxis = np.cross((rp-r0)/norm2(rp-r0),rrel/norm2(rrel))
        rrot = Analytical_Functions().rotate_around_an_axis(naxis, rrel, theta)
        newth = calc_angle(rp,r0,r0+rrot)
        #logger.debug('th0 = {:6.5f}, new th = {:6.5f} '.format(th0*180/np.pi,newth*180/np.pi))
        return rrot,newth
    
    @staticmethod
    #@jit(nopython=True,fastmath=True)
    def position_hydrogen_analytically_endgroup(bondl,theta,r1,r0,nh=1):
        r01 = r1-r0
        ru01 = r01/norm2(r01)
        rp = r0 + bondl*ru01
        dhalf = bondl/np.sqrt(3)
        s = -np.sign(r01)
        
        rrel = np.array([s[0]*dhalf,s[1]*dhalf,-s[2]*dhalf])
        #logger.debug('theta target = {:4.3f}'.format(theta*180/np.pi))
        
        newth = theta +1    
        af = Analytical_Functions()
        while np.abs(newth - theta)>1.e-4:
            rrel,newth = af.rotate_to_theta_target(rp, r0, rrel, theta)
        
        r2 = r0 +rrel
       
        if nh == 1 :
            return r2
            
        rn = af.position_hydrogen_analytically(bondl,theta,rp,r0,r2,nh-1)

        return rn
    @staticmethod
    #@jit(nopython=True,fastmath=True)
    def position_hydrogen_analytically_cis(bondl,theta,r1,r0,r2,nh=1):
        
        r01 = r1-r0
        r02 = r2-r0
        r1 = r0 +bondl*(r01)/norm2(r01)
        r2 = r0 +bondl*(r02)/norm2(r02)
        
        rm = 0.5*(r1+r2)
        #ru1 = r2 - r1 ; u1 = ru1/norm2(ru1)
        ru2 = r0 - rm ; u2 = ru2/norm2(ru2)
        rn = r0 + bondl*u2
        return rn
    @staticmethod
    #@jit(nopython=True,fastmath=True)
    def position_hydrogen_analytically(bondl_h,theta,r1,r0,r2,nh=1):
        '''
        Parameters
        ----------
        Works for CH2 groups, can be used for CH3 if one hydrogen is added properly
        bondl_h :Float (length units)
            length of the hydrogen bond.
        theta : float (radians)
            angle between r1-r0-newH or r2-r0-newH. It should be equivalent 
        r1 : float with shape (3,)
            coordinates atom to the left
        r0 : float with shape (3,)
            coordinates atom to the middle
        r2 : float with shape (3,)
            coordinates atom to the right
        nh : int 
            Solution 1 or 2
            The default is 1.
    
        Returns
        -------
        rh : float with shape (3,)
            coordinates of added atom
    
        '''
        r01 = r1-r0
        r02 = r2-r0
        r1 = r0 +bondl_h*(r01)/norm2(r01)
        r2 = r0 +bondl_h*(r02)/norm2(r02)
        
        rm = 0.5*(r1+r2)
        
        ru1 = r2 - r1 ; u1 = ru1/norm2(ru1)
        ru2 = r0 - rm ; u2 = ru2/norm2(ru2)
        
        u3 = np.cross(u1,u2)
        a = theta/2
        if nh ==1:
            rh = r0 + bondl_h*(np.cos(a)*u2 + np.sin(a)*u3)
        elif nh ==2:
            rh = r0 + bondl_h*(np.cos(a)*u2 - np.sin(a)*u3)
        return rh

class add_atoms():
    '''
    This class is used to add atoms to the system
    Currently works well to add hydrogens to PB
    It needs modification to achieve generality
    '''
    def __init__(self):
        pass
    
    hydrogen_map = {'CD':[1,0.11,116,(1,),'_cis'],
                    'C':[2,0.11,109.47,(1,2),''],
                    'CE':[3,0.109,109.47,(1,2,3),'_endgroup']}
    
    @staticmethod
    def add_ghost_atoms(self,system2,gconnectivity=dict()):
        
        for s in ['at_ids','at_types','mol_names','mol_ids']:
            a1 = getattr( system2, s )
            setattr(self,'_'.join(('ghost',s)),a1)
        for frame in system2.timeframes:
            c1 = system2.timeframes[frame]['coords']
            self.timeframes[frame]['_'.join(('ghost','coords'))] = c1
        self.ghost_connectivity = gconnectivity
        self.mass_map.update(system2.mass_map)
        return
            
    @staticmethod
    def append_atoms(self,k='ghost'):
        t0 = perf_counter()
        for s in ['at_ids','at_types','mol_names','mol_ids']:
            a1 = getattr( self, s )
            a2 = getattr( self,'_'.join([k,s]) )
            n12 = np.concatenate((a1,a2))
            setattr(self,s,n12)
        for frame in self.timeframes:
            c1 = self.timeframes[frame]['coords']
            c2 = self.timeframes[frame]['_'.join((k,'coords'))]
            c12 = np.concatenate((c1,c2))
            self.timeframes[frame]['coords'] = c12
        
        self.connectivity.update(self.ghost_connectivity)
        self.topology_initialization()
        
        if self.__class__.__name__ == 'Analysis_Confined':
            self.confined_system_initialization()
        ass.print_time(perf_counter()-t0,
                   inspect.currentframe().f_code.co_name,frame+1)
        return 
    
    @staticmethod
    def add_ghost_hydrogens(self,types,noise=None,pickleframes=False):
        t0 = perf_counter()
        
        new_atoms_info = add_atoms.get_new_atoms_info(self,'h',types)
        
        self.ghost_atoms_info = new_atoms_info
        
        add_atoms.set_ghost_connectivity(self, new_atoms_info)
        
        add_atoms.assign_ghost_topol(self, new_atoms_info)
        
        
        #for frame in self.timeframes:    
         #   add_atoms.set_ghost_coords(self, frame, new_atoms_info)
        add_atoms.set_all_ghost_coords(self,new_atoms_info,noise,pickleframes=pickleframes) 
        
        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name)
    
        return 
    
    @staticmethod
    def set_ghost_connectivity(self,info):
        gc = dict()
        for j,v in info.items():
            gc[(v['bw'],j)] = (self.at_types[v['bw']],v['ty'])   
        self.ghost_connectivity = gc
        return
    
    @staticmethod
    def set_all_ghost_coords(self,info,noise=None,pickleframes=False):
        
        f,l,th,s,ir1,ir0,ir2 = add_atoms.serialize_info(info)
        #self.unwrap_all()
        run = False
        fpickle = '{:s}_N{:d}_.pickle'.format(self.trajectory_file,self.nframes)
        try:
            with open(fpickle, 'rb') as handle:
                timeframes = pickle.load(handle)
                logger.info('Done: Read from {} '.format(fpickle))
                self.timeframes = timeframes
        except:
            run = True
            
        if run or pickleframes==False:

            for frame in self.timeframes:      
                ghost_coords = np.empty((len(info),3),dtype=float)
                coords = self.get_coords(frame)
                add_atoms.set_ghost_coords_parallel(f,l,th,s,ir1,ir0,ir2,
                                                coords,ghost_coords)
                self.timeframes[frame]['ghost_coords'] = ghost_coords
                if frame%1000==0:  logger.info('Done: setting ghost coords frame {}'.format(frame))
            if pickleframes:
                logger.info('Done: pickling to {} '.format(fpickle))
                with open(fpickle,'wb') as handle:
                    try:
                        pickle.dump(self.timeframes, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    except:
                        os.system('rm {}'.format(fpickle))
        return
    
    @staticmethod
    def serialize_info(info):
        n = len(info)
        f = np.empty(n,dtype=int)
        l = np.empty(n,dtype=float)
        th = np.empty(n,dtype=float)
        s = np.empty(n,dtype=int)
        ir1 = np.empty(n,dtype=int)
        ir0 = np.empty(n,dtype=int)
        ir2 = np.zeros(n,dtype=int)
        for j,(k,v) in enumerate(info.items()):
            f[j] = v['f']
            l[j] = v['l']
            th[j] = v['th']
            s[j] = v['s']
            ir1[j] = v['ir'][0]
            ir0[j] = v['ir'][1]
            if len(v['ir'])==3:
                ir2[j] = v['ir'][2]
        return f,l,th,s,ir1,ir0,ir2
    
    @staticmethod
    @jit(nopython=True,fastmath=True,parallel=True)
    def set_ghost_coords_parallel(f,l,th,s,ir1,ir0,ir2,
                                  coords,ghost_coords):
       
        N = f.shape[0]
        af = Analytical_Functions()
        for j in prange(N):
            cr1 = coords[ir1[j]]
            cr0 = coords[ir0[j]]
            cr2 = coords[ir2[j]]
            if   f[j] == 1:
                rn = af.position_hydrogen_analytically_cis(l[j],th[j],cr1,cr0,cr2,s[j])
            elif f[j] == 2:
                rn = af.position_hydrogen_analytically(l[j],th[j],cr1,cr0,cr2,s[j])
            elif f[j] == 3:
                rn = af.position_hydrogen_analytically_endgroup(l[j],th[j],cr1,cr0,s[j])
            ghost_coords[j] = rn
                 
        return

    @staticmethod
    def set_ghost_coords(self,info):
       
        N = len(info)
        frame = self.current_frame
        ghost_coords = np.empty((N,3),dtype=float)
        coords = self.get_whole_coords(frame)    
        
        for j,(k,v) in enumerate(info.items()):
            cr = coords[v['ir']]
            rn = v['func'](v['l'],v['th'],*cr,v['s'])
            ghost_coords[j] = rn
            
        self.timeframes[frame]['ghost_coords'] = ghost_coords
        return
    
    @staticmethod
    def assign_ghost_topol(self,info):
        n = len(info)
        
        gtypes = np.empty(n,dtype=object)
        gmol_names = np.empty(n,dtype=object)
        gat_ids = np.empty(n,dtype=int)
        gmol_ids = np.empty(n,dtype=int)
        #gatom_mass = np.empty(n,dtype=int)
        
        for i,(k,v) in enumerate(info.items()):
            gtypes[i] = v['ty']
            gmol_names[i] = v['res_nm']
            gmol_ids[i] = v['res']
            gat_ids[i] = k
            #gatom_mass[i] = v['mass']
             
            
        self.ghost_at_types = gtypes
        self.ghost_mol_names = gmol_names
        self.ghost_at_ids = gat_ids
        self.ghost_mol_ids = gmol_ids
        #self.ghost_atom_mass = gatom_mass
        
        return 
    
    @staticmethod
    def get_new_atoms_info(self,m,types):
        
        if m =='h':
            type_map = add_atoms.hydrogen_map
            self.mass_map.update({m+ty:maps.elements_mass['H'] for ty in types})
        else:
            raise NotImplementedError('Currently the only option is to add hydrogens')
        
        bond_ids = np.array(list(self.connectivity.keys()))
        ang_ids = np.array(list(self.angles.keys()))
        
        at_types = self.at_types
        residues = self.mol_ids
        res_nms = self.mol_names
        
        
        jstart = self.at_types.shape[0]
        new_atoms_info = dict()
        
        fu = 'position_hydrogen_analytically'
        for t,ty in enumerate(at_types):
            if ty  not in types:
                continue
            tm = type_map[ty]
            i = self.at_ids[t]
            
            try:
                ir = ang_ids[np.where(ang_ids[:,1] == i)][0]
            except IndexError:
                try:
                    ir = bond_ids[np.where(bond_ids[:,0]==i)][0]
                    ir = np.array([ir[1],ir[0]])
                except IndexError:
                    try:
                        ir = bond_ids[np.where(bond_ids[:,1]==i)][0]
                    except IndexError as exc:
                        raise Exception('{}: Could not find bonds for atom {:d}, type = {:s}'.format(exc,i,ty))
                   
            finally:
                func = getattr(Analytical_Functions(),fu+tm[4])
            if tm[4] == '_cis': f = 1
            elif tm[4] =='' :  f = 2
            elif tm[4] =='_endgroup': f = 3
            for j in range(tm[0]):    
                new_atoms_info[jstart] = {'ir':ir,'l':tm[1],'th':tm[2]*np.pi/180.0,'s':tm[3][j], 'bw': i,
                                     'ty':m+ty,'res':residues[i],'res_nm':res_nms[i],
                                     'func':func,'f':f}
                jstart+=1
            
        return new_atoms_info

class Box_Additions():
    '''
    Depending on the confinemnt type one of these functions
    will be called.
    These class functions are used to calculate 
    the minimum image distance
    '''
    @staticmethod
    def zdir(box):
        return [box[2],0,-box[2]]
    @staticmethod
    def ydir(box):
        return [box[1],0,-box[1]]
    @staticmethod
    def xdir(box):
        return [box[0],0,-box[0]]
    @staticmethod
    def minimum_distance(box):
        zd = Box_Additions.zdir(box)
        yd = Box_Additions.ydir(box)
        xd = Box_Additions.xdir(box)
        lst_L = [np.array([x,y,z]) for x in xd for y in yd for z in zd]
        return lst_L
    @staticmethod
    def spherical_particle(box):
        zd = Box_Additions.zdir(box)
        yd = Box_Additions.ydir(box)
        xd = Box_Additions.xdir(box)
        lst_L = np.array([np.array([x,y,z]) for x in xd for y in yd for z in zd])
        return lst_L
    @staticmethod
    def zcylindrical(box):
        return [0]
        
class Distance_Functions():
    '''
    Depending on the confinemnt type one of these functions
    will be called.
    These class functions are used to calculate 
    the Distance between coords and a center position (usually particle center of mass is passed)
    '''
    @staticmethod
    def zdir(self,coords,c):
        return np.abs(coords[:,2] - c[2])
    @staticmethod
    def ydir(self,coords,c):
        return np.abs(coords[:,1] - c[1])
    @staticmethod
    def xdir(self,coords,c):
        return np.abs(coords[:,0] - c[0])
    

    
    @staticmethod
    def zdir__2side(self,coords,zc):
        return coords[:,2]-zc
    @staticmethod
    def ydir__2side(self,coords,yc):
        return coords[:,1]-yc
    @staticmethod
    def xdir__2side(self,coords,xc):
        return coords[:,0]-xc
    
    @staticmethod
    def spherical_particle(self,coords,c):
        d = np.zeros(coords.shape[0],dtype=float)
        distance_kernel(d,coords,c)
        #r = coords -c
        #d = np.sqrt(np.sum(r*r,axis=1))
        return d
    
    @staticmethod
    def minimum_distance(self,coords1,coords2):
        d1 = np.empty(coords1.shape[0])
        d2 = np.empty(coords2.shape[0])
        smaller_distance_kernel(d1,d2,coords1,coords2)
        return d1

    @staticmethod
    def zcylindrical(self,coords,c):
         r = coords[:,0:2]-c[0:2]
         d = np.sqrt(np.sum(r*r,axis=1))

         return d
     
class bin_Volume_Functions():
    '''
    Depending on the confinemnt type one of these functions
    will be called.
    These class functions are used to calculate 
    the volume of each bin when needed (e.g. for density profile calculations) 
    '''
    @staticmethod
    def zdir(self,bin_low,bin_up):
        box = self.get_box(self.current_frame)
        binl = bin_up-bin_low
        return 2*box[0]*box[1]*binl
    
    @staticmethod
    def ydir(self,bin_low,bin_up):
        box = self.get_box(self.current_frame)
        binl = bin_up-bin_low
        return 2*box[0]*box[2]*binl
    
    @staticmethod
    def xdir(self,bin_low,bin_up):
        box = self.get_box(self.current_frame)
        binl = bin_up-bin_low
        return 2*box[1]*box[2]*binl

    @staticmethod
    def zcylindrical(self,bin_low,bin_up):
        box = self.get_box(self.current_frame)
        return np.pi*(bin_up**2-bin_low**2)*box[2]
    
    @staticmethod
    def spherical_particle(self,bin_low,bin_up):
        v = 4*np.pi*(bin_up**3-bin_low**3)/3
        return  v
    


    
class unit_vector_Functions():
    '''
    Depending on the confinemnt type one of these functions
    will be called.
    A unit vector for computing bond order is defined
    '''
    @staticmethod
    def zdir(self,r,c):
        uv = np.zeros((r.shape[0],3))
        uv[:,2] = 1
        return uv
    @staticmethod
    def ydir(self,r,c):
        uv = np.zeros((r.shape[0],3))
        uv[:,1] = 1
        return uv
    
    @staticmethod
    def xdir(self,r,c):
        uv = np.zeros((r.shape[0],3))
        uv[:,0] = 1
        return uv
    
    @staticmethod
    @jit(nopython=True,fastmath =True,parallel=True)
    def spherical_particle_inner(cperiodic,r,c,box_add):
        for i in prange(cperiodic.shape[0]):
            dist_min = 1e16    
            for j,L in enumerate(box_add):
                rr = r[i] - (c+L)
                dist = np.sum(rr*rr)
                if dist<dist_min:
                    jmin = j
                    dist_min = dist
            cperiodic[i] = c +box_add[jmin]
        return
    @staticmethod
    def spherical_particle(self,r,c):
        box = self.get_box(self.current_frame)
        box_add = Box_Additions.spherical_particle(box)
        cperiodic = np.empty(r.shape)
        unit_vector_Functions.spherical_particle_inner(cperiodic,r,c,box_add)
        uv = r-cperiodic
        return uv
    @staticmethod
    def zcylindrical(self,r,c):
        uv = np.ones((r.shape[0],3))
        uv[:,2] = 0
        return uv
    
    
class add_sudo_atoms:
    def __init__(self,obj1,num,sigma,frame=0,sep_dist=3.0,decrease_rate=0.9,positions=None,
            r='no',rd='z',rl=0,rh=0):
        self.obj1 = obj1
        self.nsudos = num
        self.sigma = sigma
        self.frame = frame
        self.r = r
        self.rd = rd
        self.rl = rl
        self.rh = rh
        self.sep_dist = sep_dist
        self.decrease_rate = decrease_rate
        self.grid = self.d3grid()
        
        if positions is None:
            self.find_positions()
        else:
            self.positions = positions
        self.update_topology()
        return
    def d3grid(self):
        # from long to short 
        boxsort = self.obj1.get_box(self.frame).argsort()[::-1]
        grid = np.array([0,0,0])
        i=0
        while grid.prod() < self.nsudos:
            j = i%3
            grid[j]+=1
            i+=1
        grid[(j+1)%3]+=1
        return grid[boxsort]
    @property
    def position_distances_nonperiodic(self):
        dists = []
        box = self.obj1.get_box(self.frame)
        for i,c1 in enumerate(self.positions):
            c2 = self.positions[i+1:]
            r = c2-c1
            d = np.sum(r*r,axis=1)*0.5
            dists.extend(d)
        return np.array(dists)
    @property
    def position_distances(self):
        dists = []
        box = self.obj1.get_box(self.frame)
        for i,c1 in enumerate(self.positions):
            c2 = self.positions[i+1:]
            d = React_two_systems.minimum_image_distance(c2,c1,box)
            dists.extend(d)
        return np.array(dists)
   
    def find_positions(self):
        self.separation_positions()
        return
    
    def random_positions(self):
        box = self.obj1.get_box(self.frame)
        self.positions = np.random.uniform([0,0,0],box,(self.nsudos,3))
        return
    
    def separation_positions(self):
        box = self.obj1.get_box(self.frame)
        sudo_coords = []
        sep_dist = self.sep_dist
        r = self.r
        rd= self.rd
        rl = self.rl
        rh = self.rh
        restricted=True
        
        if r == 'no':
            restricted=False
        elif r=='center':
            cr = box/2
        elif r in self.obj1.at_types or r in self.obj1.mol_names:
            if r in self.obj1.at_types:
                fc = self.obj1.at_types == r
            elif r in self.obj1.mol_names:
                fc = self.obj1.mol_names == r
            c = self.obj1.get_coords(self.frame) [fc]
            am = self.obj1.atom_mass [fc]
            cr = CM(c,am)
        else:
            raise Exception('r = {:s} is not a proper method to restrict the area; Give either center, group of atoms (atom types or molecule names)')

        for jadd in range(self.nsudos):
            accepted = False
            jfailed=0
            while not accepted:
                pos = np.random.uniform([0,0,0],box)
                if restricted:
                    rc = pos - cr
                    if rd == 'x':
                        if rl<= rc[0] <=rh:
                            continue
                    if rd == 'y':
                        if rl<= rc[1] <=rh:
                            continue
                    if rd == 'z':
                        if rl<= rc[2] <=rh:
                            continue
                    if rd == 'd':
                        d = np.sum(rc*rc)**0.5
                        if rl<= d <=rh:
                            continue
                if len(sudo_coords) ==0:
                    accepted = True
                else:
                    c_comp = np.array(sudo_coords)
                    dists = React_two_systems.minimum_image_distance(c_comp,pos,box)
                    if not (dists > sep_dist).all():
                        jfailed +=1
                    else:
                        accepted = True
                if jfailed>=30:
                    jfailed =0
                    print(' separation distance {:4.3f} too big for sudo atom {:d}. Reducing to {:4.3f}'.format(
                            sep_dist,jadd,sep_dist*self.decrease_rate))
                    sep_dist *=self.decrease_rate
                    
                    
                
            sudo_coords.append(pos)
        print(sudo_coords)
        self.positions = np.array(sudo_coords)
        return 
    
    def uniform_positions(self):
        box = self.obj1.get_box(self.frame)
        dx,dy,dz = box/self.grid
        gx,gy,gz = int(self.grid[0]), int(self.grid[1]), int(self.grid[2]) 
        positions = []
        num =0
        for i in range(gx):
            x = (i+1)*dx/2
            for j in range(gy):
                y = (j+1)*dy/2
                for k in range(gz):
                    if num>= self.nsudos:
                        break
                    z = (k+1)*dz/2
                    positions.append(np.array([x,y,z]))
                    num+=1

        self.positions = np.array(positions)
        
        return
    
    def update_topology(self):
        box = self.obj1.get_box(self.frame)
        timeframes = {0:{'coords':self.positions,'boxsize':box,'time':0,'step':0}}
        natoms = self.positions.shape[0]
        ty = 'SUDO'
        tyff = {ty:(ty,'10000.000','0.0','A','0.01','1E-06')}
        obj2 = Topology(natoms,at_types=ty,atom_code=ty,timeframes=timeframes,atomtypes=tyff)
        self.obj2 = obj2
        self.obj1.merge_system(self.obj2)
      
        return



class React_two_systems:
    
    def __init__(self,obj1,obj2,bb1,bb2,react1=0,react2=0,
                 rcut=3.5,method='breakBondsMerge',
                 frame=0,seed1=None,seed2=None,shape='flat',
                 morse_bond=(100,0.16,2),morse_overlaps=(0.2,5),use_bounds=True,bounds_z=(-1.5,1.5),
                 maptypes=dict(),updown_method=None,cite_method='random',cite_method_kwargs=dict(),
                 iapp=None):
        self.cite_method = cite_method
        for k,v in cite_method_kwargs.items():
            setattr(self,k,v)
        self.obj1 = obj1
        self.obj2 = obj2
        self.seed1 = seed1
        self.seed2 = seed2
        self.react1 = react1
        self.react2 = react2
        self.shape=shape
        self.updown_method = updown_method
        self.bb1 = bb1
        self.bb2 = bb2
        self.set_break_bond_id('1')
        self.set_break_bond_id('2')
        self.rcut = rcut
        self.frame = frame
        self.bond_to_create = (self.break_bondid1[react1],self.break_bondid2[react1])
        self.maptypes = maptypes
        self.morse_bond = morse_bond
        self.morse_overlaps = morse_overlaps
        self.bounds_z = bounds_z
        self.use_bounds = use_bounds
        if method == 'breakBondsMerge':
            self.break_bonds()
            
            self.react_id1 = self.obj1.old_to_new_ids[self.bond_to_create[0]]
            self.react_id2 = self.obj2.old_to_new_ids[self.bond_to_create[1]]
            
            
            
            self.place_obj2()
            
            natoms1 = self.obj1.natoms
            self.obj2.mol_names[:] = self.obj1.mol_names[0]
            if iapp is None:
                self.obj1.merge_system(self.obj2)
            else:
                self.obj1.merge_system(self.obj2,add=str(iapp))
                    
            #make bond
            self.reacted_id2 = natoms1 +self.react_id2 #renweing the id
            self.reacted_id1 = self.react_id1
            self.change_type(self.reacted_id1)
            self.change_type(self.reacted_id2)
            conn_id,c_type = self.obj1.sorted_id_and_type((self.reacted_id1,self.reacted_id2))
            self.obj1.connectivity[conn_id] = c_type
            
            new_angles = self.obj1.find_new_angdihs(conn_id)
            
            self.obj1.angles.update(new_angles)
            for newa in new_angles:
                new_dihs = self.obj1.find_new_angdihs(newa)
                self.obj1.dihedrals.update(new_dihs)
            # refine charge
            self.refine_charge()
            
           
        else:
            raise NotImplementedError('There is no such method as "{:s}"'.format(method))
        return
    
    def break_bonds(self):
        #remove product
        if self.react1 ==1:
            rbb1 = (self.break_bondid1[1],self.break_bondid1[0])
        else:
            rbb1 = self.break_bondid1
        if self.react2 ==1:
            
            rbb2 = (self.break_bondid2[1],self.break_bondid2[0]) 
        else:
            rbb2 = self.break_bondid2
        self.radicals1 = self.remove_trails(self.obj1,*rbb1)
        self.radicals2 = self.remove_trails(self.obj2,*rbb2)
        return
    def refine_charge(self):
        id1 = self.reacted_id1
        id2 = self.reacted_id2
        tch = 0.5*self.obj1.total_charge
        ty1 = self.obj1.at_types[id1]
        ty2 = self.obj1.at_types[id2]
        for t,i in zip([ty1,ty2],[id1,id2]):
            newc = self.obj1.atom_charge[i] - tch   
            self.obj1.atom_charge[i] = newc
            val = np.array(self.obj1.ff.atomtypes[t])
            val[2] = str(newc)
            self.obj1.ff.atomtypes[t] = tuple(val)
        totc = self.obj1.total_charge
        if abs(totc)>1e-10:
            raise Exception('Total charge is not newtral, total charge = {:.8e}'.format(totc))

        return 
    def change_type(self,at_id,iapp=''):
        #from copy import copy
        
        ty = self.obj1.at_types[at_id]
        if ty in self.maptypes:
            newty = self.maptypes[ty]
        else:
            newty = self.obj1.at_types[at_id] +'R' +iapp
        self.obj1.at_types[at_id] = newty
        self.obj1.atom_code[at_id] = newty
        
        for attr_name in ['connectivity','angles','dihedrals']:
            attr = getattr(self.obj1,attr_name)
            for c in list(attr.keys()):
                
                if at_id in c:
                    t = [i for i in attr[c]]      
                    i = np.where(np.array(t)==ty)[0][0]
                    t[i] = newty
                    logger.debug('changed {} to {}'.format(attr[c],tuple(t)))
                    attr[c] = tuple(t) 
                    
        
        for attr_name in ['bondtypes','angletypes','dihedraltypes']:
            attr = getattr(self.obj1.ff,attr_name)
            for k in list(attr.keys()):
                if ty in k:
                    arr = [i for i in k]
                    i = np.where(np.array(arr)==ty)[0][0]
                    arr[i] = newty
                    ty_new  = tuple(arr)
                    logger.debug('made type {} same as  {}'.format(k,newty))
                    val = list(attr[k])
                    
                    code = '  '.join(ty_new)
                    val[0] = code
                    attr[ty_new] = tuple(val)
                    #del attr[k]
            #setattr(self.obj1.ff,attr_name,attr)
        val = list(self.obj1.ff.atomtypes[ty])
        val[0] = newty
        self.obj1.ff.atomtypes[newty] = tuple(val)
        #self.obj1.filter_ff()
        return 
    
    def set_break_bond_id(self,prefix):
        obj = getattr(self,'obj'+prefix)
        bb = getattr(self,'bb'+prefix)
        seed = getattr(self,'seed'+prefix)
        react = getattr(self,'react'+prefix)
        
        if not ( react == 0 or react == 1):
            raise ValueError('react'+prefix +' must be zero or one')
            
        name = 'break_bondid' + prefix
        
        if type(bb[0]) is str and type(bb[1]) is str:
            if prefix =='1':
                pm = self.cite_method
            else:
                pm = 'random'
            self.prefix_break_id =prefix
            bond_id  = self.find_random_bond_of_type(obj,bb,seed,react,pm)
            assert bb == obj.connectivity[bond_id],'the bond id {} does not give the specified type {}'.format(bond_id,bb) 
            #if react == 1:
                #bond_id = (bond_id[1],bond_id[0])
                
        elif type(bb[0]) is int and type(bb[1]) is int:
            bond_id = bb
        else:
            raise NotImplementedError('value {} is not understood for {:s}'.format(bb,'bb'+prefix) )
        setattr(self,name,bond_id)
        return
    
    @staticmethod
    def numb_of_neibs(c,ctot):
        n = [np.sum(np.exp(-(1/0.265)*Distance_Functions.spherical_particle(None,ctot,c[i]))) for i in range(c.shape[0])]
        return np.array(n)
    
    
    def find_random_bond_of_type(self,obj,bb,seed,idr,propmethod=None):
        
        def height(zf):
            za = np.abs(zf)
            zrel = za - za.min()
            m = zrel.max()
            prop = 1-np.exp(-5*m*zrel)
            return prop
        def neibs(cf,c):
            prop = np.exp(-self.numb_of_neibs(cf,c))
            return prop
        def separation(nums):
            sids = self.obj1.at_ids [self.obj1.at_types ==self.separation_type]
            sep = False
            jmax = self.obj1.natoms
            j=0
            set_nums = set(nums)
            box = self.obj1.get_box(0)
            ncites =0
            while (sep==False):
                if len(set_nums) ==0:
                    self.separation_distance*=0.8
                    set_nums = set(nums)
                num = np.random.choice(list(set_nums))
                
                cre = c[ids[num]]
                c_comp = c[sids]
                
                dists = self.minimum_image_distance(c_comp,cre,box)

                if (dists< self.separation_distance).any():
                    #print('cite {:d} to close'.format(ids[num]))
                    ncites+=1
                    sep = False
                else:
                    print('# of Failed cites = {:d},  cite {:d} ok'.format(ncites,ids[num]))
                    sep = True
                set_nums.remove(num)
                j+=1
                if j>jmax:
                    raise Exception('infinite while loop')
            return num
        
        np.random.seed(seed)
        
        bbids = ass.numpy_keys(obj.connectivity)
        bbts = ass.numpy_values(obj.connectivity)
        
        f = bbts == np.array(bb)
        f = np.logical_and(f[:,0],f[:,1])
        bids = bbids[f]
        nums = np.arange(0,bids.shape[0],1,dtype=int)
        ids = bids[:,idr]
        c = obj.get_coords(0).copy()
        z = c[:,2]
        if self.shape=='flat' and self.prefix_break_id=='1':
            z = z[ids]
            zm = np.sum(z)/z.shape[0]
            z -= zm
            if self.updown_method =='random':
                    
                updown = np.random.choice([True,False]) 
            else:
                if not hasattr(self.obj1,'updown'):
                    self.obj1.updown = False
                self.obj1.updown = not self.obj1.updown
                updown = self.obj1.updown
            if updown:
                fz = z>0
            else:
                fz = z<0
            
            nums = nums[fz]
            zf = z[fz]
        else:
            num = np.random.choice(nums)
            return tuple(bids[num])
        print(propmethod)
         
        if propmethod=='random':
            num =  np.random.choice(nums)
        elif propmethod =='height':
            prop = height(zf)
        elif propmethod=='neibs':
            prop = neibs(c[ids][fz],z)
        elif propmethod =='height_neibs':
            prop = neibs(c[ids][fz],z)*height(zf)
        elif propmethod =='separation_distance':
            if f.any():
                num = separation(nums)
            else:
                num = np.random.choice(nums)
            return tuple(bids[num])
        elif propmethod =='uniform':
            self.initial_separation_distance = self.separation_distance
            grid_x,grid_y = self.grid
            box = self.obj1.get_box(0)
            Lx = box[0]
            Ly = box[1]
            dLx = Lx/grid_x
            dLy = Ly/grid_y
            areas = [(i,j,u) for i in range(grid_x) for j in range(grid_y) for u in [0,1]]
            if not hasattr(self.obj1,'filled_areas'):
                self.obj1.filled_areas = {k:0 for k in areas}
            if updown:
                kselect=(0,0,1)
                areas = {k:v for k,v in self.obj1.filled_areas.items() if k[2] == 1}
            else:
                kselect=(0,0,0)
                areas = {k:v for k,v in self.obj1.filled_areas.items() if k[2] == 0}
            for k,v in areas.items():
                if v < areas[kselect]:
                    kselect = k
            same = []
            for k,v in areas.items():
                if v == areas[kselect]:
                    same.append(k)
            kselect = same[ np.random.choice(np.arange(0,len(same),1,dtype=int)) ]
            #print(kselect)
            cids = c[ids,:][fz]
            cx = cids[:,0]
            cy = cids[:,1]
            kx = kselect[0]
            ky = kselect[1]
            fx = np.logical_and(kx*dLx < cx, cx <= (kx+1)*dLx)
            fy = np.logical_and(ky*dLy < cy, cy <= (ky+1)*dLy)
            f = np.logical_and(fx,fy)
            
            self.obj1.filled_areas[kselect]+=1
            #print(self.obj1.filled_areas)
            if f.any():
                num = separation(nums[f])
            else:
                num = np.random.choice(nums[f])
            #print(self.obj1.at_types[bids[num]])
            self.separation_distance = self.initial_separation_distance
            return tuple(bids[num])
                
        else:
           raise ValueError('There is no method name  as {:s}'.format(propmethod))
        
        prop/=prop.sum()
        num = np.random.choice(nums,p=prop)
        bond_id = tuple(bids[num])
        return bond_id
    @staticmethod
    @jit(nopython=True,fastmath=True,parallel=True)
    def minimum_image_distance(coords,cref,box):
            r = coords - cref
           
            for j in range(3):
                b = box[j]
                b2 = b/2
                fm = r[:,j] < - b2
                fp = r[:,j] >   b2
                r[:,j][fm] += b
                r[:,j][fp] -= b
            d = np.zeros((r.shape[0],),dtype=np.float64)
            for i in prange(r.shape[0]):
                for j in range(3):
                    x = r[i,j]
                    d[i] += x*x
                d[i] = np.sqrt(d[i])
            
            return d
    @staticmethod
    @jit(nopython=True,fastmath=True,parallel=True)
    def minimum_image_distance_coords(coords,cref,box):
            r = coords - cref
            imag_coords = coords.copy()
            for j in range(3):
                b = box[j]
                b2 = b/2
                fm = r[:,j] < - b2
                fp = r[:,j] >   b2
                            
                r[:,j][fm] += b
                imag_coords[:,j][fm] +=b
                
                r[:,j][fp] -= b
                imag_coords[:,j][fp] -= b
            d = np.zeros((r.shape[0],),dtype=np.float64)
            for i in prange(r.shape[0]):
                for j in range(3):
                    x = r[i,j]
                    d[i] += x*x
                d[i] = np.sqrt(d[i])
            
            return d,imag_coords
    
    @staticmethod
    def find_reaction_neibhour_coords(coords,cref,box,rcut):

       
        d,imag_coords =  React_two_systems.minimum_image_distance_coords(coords,cref,box)
        reaction_neibs = imag_coords[d<rcut]
        return reaction_neibs
    
    @staticmethod
    def identify_trail(obj,trail_from,trail_to):
        trailing_set_old = set()
        trailing_set = {trail_to}
        while len(trailing_set) != len(trailing_set_old):
            trailing_set_old = trailing_set.copy()
            for j in trailing_set_old:
                for neib in obj.neibs[j]:
                    if neib !=trail_from:
                        trailing_set.add(neib)
            
        return trailing_set
    
    @staticmethod
    def remove_trails(obj,trail_from,trail_to):
        trailing_ids = set()
        trailing_ids = React_two_systems.identify_trail(obj,trail_from,trail_to) 
        trailing_ids = np.array(list(trailing_ids))
        
        cf = obj.get_coords(0)[trailing_ids]
        
        free_radicals = pd.DataFrame({'at_ids':trailing_ids,
                                      'at_tys':obj.at_types[trailing_ids],
                                      'mol_ids':obj.mol_ids[trailing_ids],
                                      'mol_names':obj.mol_names[trailing_ids],
                                      'atom_charge':obj.atom_charge[trailing_ids],
                                      'atom_mass':obj.atom_mass[trailing_ids],
                                      'x':cf[:,0],
                                      'y':cf[:,1],
                                      'z':cf[:,2],
                                      })
        obj.remove_atoms_ids(trailing_ids)
        
        return free_radicals
    
    @staticmethod
    @jit(nopython=True,fastmath=True)
    def morse(r,De,re,alpha):
        return De*(np.exp(-2*alpha*(r-re))-2*np.exp(-alpha*(r-re)))
    
    @staticmethod
    @jit(nopython=True,fastmath=True)
    def morse_rep(r,re,alpha):
        return np.exp(-alpha*(r-re))
    @staticmethod
    def cost_overlaps(vector_n_angles,creact1,id2,coords_neib_obj1,coords_obj2,
                      morse_bond,morse_overlaps):
       
        ctr = RotTrans.trans_n_rot(vector_n_angles, coords_obj2)
        # distance of reaction bond
        refdist = RotTrans.distance(creact1,ctr[id2])
        
        f1 = React_two_systems.morse
        f2 = React_two_systems.morse_rep
        d = [Distance_Functions.spherical_particle(None, coords_neib_obj1, c)
             for i,c in enumerate(ctr) if i!=id2]
        d =np.array(d)# shape ctr.shape[0]
        return  f1(refdist,*morse_bond) + np.sum(f2(d,*morse_overlaps))

    def place_obj2(self):
        #t0 = perf_counter()
        frame = self.frame  
        coords_obj2 = self.obj2.get_coords(frame)
        coords_obj1 = self.obj1.get_coords(frame)
        
        creact1 = coords_obj1[self.react_id1]
        
        
        box = self.obj1.get_box(frame)
        bm = box/2
        cm2 = np.mean(coords_obj2,axis=0)
        coords_obj2 += bm-cm2
        
        coords_neibs_obj1 = self.find_reaction_neibhour_coords(coords_obj1,creact1,box,self.rcut)
        
        bounds= [(-m/2,m/2) for m in box] + [(-np.pi,np.pi)]*3
                
        arguments = (creact1,self.react_id2,
                     coords_neibs_obj1,coords_obj2,
                     self.morse_bond,self.morse_overlaps)
        
        if False:
            opt_res = dual_annealing(self.cost_overlaps, bounds,
                    args =  arguments,
                    maxiter = 1000,
                    restart_temp_ratio=1e-5,
                    minimizer_kwargs={'method':'SLSQP',#
                                     'bounds':bounds,
                                     
                    'options':{'maxiter':300,
                        'disp':False,
                        'ftol':1e-3},
                                },
                              )
            
        else:
            opt_res = differential_evolution(self.cost_overlaps, bounds,
                    args = arguments ,
                    maxiter = 60,disp=False,polish=True)
        print('Pseudo Energy = {:4.6f}'.format(opt_res.fun))
        
        if not hasattr(self.obj1,'se'):
            self.obj1.se =[]
        self.obj1.se.append(opt_res.fun)
        
        self.opt_res = opt_res
        self.p = opt_res.x
        
        new_coords = RotTrans.trans_n_rot(self.p,coords_obj2)
        
        miz = new_coords[:,2].min()
        mxz = new_coords[:,2].max()
        bz = self.bounds_z
        between_bounds = bz[0] <= 0.5*(miz+mxz) <= bz[1]
        if self.use_bounds and  between_bounds:
            self.morse_overlaps = (self.morse_overlaps[0]*2,self.morse_overlaps[1]/2)
            print('Min z surf = {:4.3f} Max z surf = {:4.3f}\n Min placement {:4.3f} Max placement {:4.3f}'.format(*bz,miz,mxz))
            print('refining placement')
            self.place_obj2()
            return 
        else:
            self.obj2.timeframes[frame]['coords'] = new_coords
        #tf = perf_counter()-t0
        #print('time of placing object = {:.3e} sec'.format(tf))
        return 
  
class RotTrans:
    
    def __init__(self):
        return
    
    @staticmethod
    def Rx(theta):
        c = np.cos(theta)
        s = np.sin(theta)
        r = np.zeros((3,3))
        r[0,0] = 1
        r[1,1] = c
        r[2,2] = c
        r[1,2] = -s
        r[2,1] = s
        return r
    @staticmethod
    def Ry(theta):
        c = np.cos(theta)
        s = np.sin(theta)
        r = np.zeros((3,3))
        r[1,1] = 1
        r[0,0] = c
        r[2,2] = c
        r[0,2] = -s
        r[2,0] = s
        return r
    @staticmethod
    def Rz(theta):
        c = np.cos(theta)
        s = np.sin(theta)
        r = np.zeros((3,3))
        r[2,2] = 1
        r[0,0] = c
        r[1,1] = c
        r[0,1] = -s
        r[1,0] = s
        return r
    
    @staticmethod
    def rotate(c,yaw,pitch,roll):
        Rx = RotTrans.Rx(yaw)
        Ry = RotTrans.Ry(pitch)
        Rz = RotTrans.Rz(roll)
        cn = c.copy()
        for r in [Rx,Ry,Rz]:
            for i in range(cn.shape[0]):
                cn[i] = np.dot(r,cn[i])
        return cn
    
    @staticmethod
    def distance(r1,r2):
        r = r2-r1
        return np.sum(r*r)**0.5
    
    @staticmethod
    def rhat(r1,r2):
        d = RotTrans.distance(r1,r2)
        return (r2-r1)/d
    
    @staticmethod
    def translate(r1,rx):
        return r1+rx
    @staticmethod
    def trans_n_rot(vector_n_angles,coords):
        vector = vector_n_angles[:3]
        angles = vector_n_angles[3:]
        translated_coords = coords + vector
        rotref = np.mean(translated_coords,axis=0)
        relc = translated_coords - rotref
        cr = RotTrans.rotate(relc, *angles) 
        return cr+rotref



class Topology:
    def __init__(self,natoms,at_types='ATOP',atom_code='ATOP',mol_ids=1,mol_names='MTOP',
                 atom_charge=0.0,atom_mass=10.0,timeframes= dict(),neibs=dict(),
                 connectivity=dict(),angles=dict(),dihedrals=dict(),pairs=dict(),exclusions=dict(),
                 atomtypes=dict(),bondtypes=dict(),angletypes=dict(),dihedraltypes=dict(),
                 ):
        self.at_ids = np.arange(0,natoms,1,dtype='i')
        self.ff = self.FFparams()
        defaults_str = ['at_types','mol_names','atom_code']
        defaults_int = ['mol_ids']
        defaults_float = ['atom_charge','atom_mass']
        update_defaults = ['timeframes','neibs','connectivity','angles','dihedrals','pairs','exclusions']
        update_ff_defaults = ['atomtypes','bondtypes','angletypes','dihedraltypes']
        self.exclusions_map = dict()
        for a in defaults_str:
            setattr(self,a,np.empty(natoms,dtype=object))
        for a in defaults_int:
            setattr(self,a,np.empty(natoms,dtype=int))
        for a in defaults_float:
            setattr(self,a,np.empty(natoms,dtype=float))
        
        if mol_ids is not None:
            self.mol_ids[:] = mol_ids
        if atom_code is not None:
            self.atom_code[:] = atom_code
        if mol_names is not None:
            self.mol_names[:] = mol_names
        if at_types is not None:
            self.at_types[:] = at_types
        
        if atom_charge is not None:
            self.atom_charge[:] = atom_charge
        if atom_mass is not None:
            self.atom_mass[:] = atom_mass

   
        for a in update_defaults:
            d = locals()[a]
            setattr(self,a,d)
        for a in update_ff_defaults:
            d = locals()[a]
            setattr(self.ff,a,d)
        return
    
    class FFparams():
        def __init__(self):
            self.atomtypes = dict()
            self.bondtypes = dict()
            self.angletypes = dict()
            self.dihedraltypes = dict()
            return
        def add_posres(self,data):
            if type(data) is list:
                if type(data[0]) is not dict:
                    raise Excepion('give prober informations see function documentation')
            if type(data) is dict:
                data = [data]
            self.posres = data
            return 
    @property
    def total_charge(self):
        return self.atom_charge.sum()
    @property
    def total_mass(self):
        return self.atom_mass.sum()
    @property
    def inspect_system(self):
        names = ['at_ids','at_types','mol_ids',
                 'mol_names','atom_charge','atom_mass','atom_code']

        return pd.DataFrame({ a:getattr(self,a) for a in names} )
            
    
    @property
    def nframes(self):
        return len(self.timeframes)
    
    def get_key(self):
        key_func = getattr(self,self.key_method)
        key = key_func()
        return key
    
    def get_timekey(self):
        t0 = self.get_time(self.first_frame)
        tf = self.get_time(self.current_frame)
        return round(tf-t0,self.round_dec)
    
    def get_exkey(self):
        e0 = self.get_box(self.first_frame)[0]
        et = self.get_box(self.current_frame)[0]
        return round((et-e0)/e0,self.round_dec)
    
    def dict_to_sorted_numpy(self,attr_name):
        #t0 = perf_counter()
        
        attr = getattr(self,attr_name)
        if type(attr) is not dict:
            raise TypeError('This function is for working with dictionaries')
            
        keys = attr.keys()
        x = np.empty((len(keys),2),dtype=int)
        for i,k in enumerate(keys):
            x[i,0]=k[0] ; x[i,1]=k[-1]
            
        x = x[x[:,0].argsort()]
        setattr(self,'sorted_'+attr_name+'_keys',x)
        
        #tf = perf_counter() - t0 
        #ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        return

    def keysTotype(self,attr_name):
        dictionary = getattr(self,attr_name)
        if type(dictionary) is not dict:
            raise TypeError('This function is for working with dictionaries')
        types = self.unique_values(dictionary.values())
        temp_ids = {i:[] for i in types}
        for k,v in dictionary.items():
            temp_ids[v].append(np.array(k))
        ids = {v : np.array(temp_ids[v]) for v in types}
        
        return ids 
    @property
    def connectivity_pertype(self):
        return self.keysTotype('connectivity')
    @property
    def angles_pertype(self):
        return self.keysTotype('angles')
    @property
    def dihedrals_pertype(self):
        return self.keysTotype('dihedrals')
    
    def define_connectivity(self,bond_dists):
        bond_dists = {tuple(np.sort(k)):v for k,v in bond_dists.items()}
        self.read_file(self.topol_file)
        c = self.get_coords(0)
        at_ids = self.at_ids
        at_types = self.at_types
        connectivity = dict()
        con_ty = dict()
        for i1,at1 in zip(at_ids,at_types):
            for i2,at2 in zip(at_ids,at_types):
                if i1==i2:
                    continue
                tyc = tuple(np.sort([at1,at2]))
                if tyc in bond_dists:
                    c1 = c[i1]
                    c2 = c[i2]
                    rd = c2 -c1
                    d = np.dot(rd,rd)**0.5
                    r = bond_dists[tyc]
                    if d<r[1]:
                        ids = (i1,i2)
                        conn_id,c_type = self.sorted_id_and_type(ids)
                        connectivity[conn_id] = c_type
                        if d>r[0]:
                            con_ty[conn_id] = 1
                        else:
                            con_ty[conn_id] = 2
                    
        self.con_ty = con_ty    
        self.connectivity = connectivity
        return
    
    def map_the_topology(self,mapper):
        self.mass_map = mapper['mass']
        self.charge_map = mapper['charge']
        self.define_connectivity(mapper['bonds'])
        return
    
    def read_topfile(self,file):
        with open(file,'r') as f:
            lines = f.readlines()
            f.closed
        lookfor = ['atomtypes','bondtypes','angletypes','dihedraltypes',
                   'atoms','bonds','angles','pairs','exclusions']
        nline = dict()
        un_at_types = np.unique(self.at_types)
        nt = un_at_types.shape[0]
        for i,line in enumerate(lines):
            l = line.strip().split('[')[-1].split(']')[0].strip()
            if l in lookfor:
                nline[l] = i
        
        #finding masses and charges
        
        mass_map = dict()
        charge_map = dict()
        n = nline['atomtypes']+1
        jt = 0
        for line in lines[n:]:
            if ';' in line:
                continue
            l = line.strip().split()
            k = l[0]
            mass_map[k] = float(l[1])
            charge_map[k] = float(l[2])
            jt+=1
            if jt==nt:
                break

        self.read_itp_file(file,find_mass=mass_map,find_charge=charge_map)
        self.local_to_global_topology()
        
        
        self.atom_charge = self.find_with_typemap(charge_map)
        self.atom_mass = self.find_with_typemap(mass_map)
        #connectiviy
        
        self.ff = self.FFparams()
        for name,jd in zip(['atomtypes','bondtypes','angletypes','dihedraltypes'],[1,2,3,4]):
            try:
                n = nline[name]+1
            except KeyError:
                setattr(self.ff,name,dict())
            else:
                attr = dict()
                for line in lines[n:]:
                    if '[' in line or ']' in line:
                        break
                    
                    if ';' in line or 'include' in line:
                        continue
                    l = line.strip().split()
                    if len(l)==0:
                        continue
                    if jd>1:
                        ty = self.sorted_type(l[:jd])
                        value = (' '.join(ty),*l[jd:])
                    else:
                        if len(l) ==7:
                            ty = l[1]
                            value = (l[0],*l[jd:])
                        else:
                            ty=l[0]
                            value = (ty,*l[jd:])
                    attr[ty] = value
                setattr(self.ff,name,attr)

        
        return
    def read_topfile_ff(self,file):
        
        with open(file,'r') as f:
            lines = f.readlines()
            f.closed
        lookfor = ['atomtypes','bondtypes','angletypes','dihedraltypes']
        nline = dict()

        for i,line in enumerate(lines):
            l = line.strip().split('[')[-1].split(']')[0].strip()
            if l in lookfor:
                nline[l] = i
        
        self.ff = self.FFparams()
        for name,jd in zip(lookfor,[1,2,3,4]):
            try:
                n = nline[name]+1
            except KeyError:
                setattr(self.ff,name,dict())
            else:
                attr = dict()
                for line in lines[n:]:
                    if '[' in line or ']' in line:
                        break
                    if ';' in line or 'include' in line or '#' in line:
                        continue
                    l = line.strip().split()
                    if len(l)==0:
                        continue
                    if jd>1:
                        ty = self.sorted_type(l[:jd])
                        value = (' '.join(ty),*l[jd:])
                    else:
                        if len(l) ==7:
                            ty = l[1]
                            value = (l[0],*l[jd:])
                        else:
                            ty=l[0]
                            value = (ty,*l[jd:])
                    attr[ty] = value
                
                setattr(self.ff,name,attr)
        return
 
                
    def find_with_typemap(self,usedmap):
        first_element_type =  type(list(usedmap.values())[0])
        arr = np.empty(self.natoms,dtype = first_element_type)
        for i,ty in enumerate(self.at_types):
            arr[i] = usedmap[ty]
        return arr
    def read_total_conangdih(self,lines,c,sub=1):
        cad = dict()
        if c not in ['connectivity','angles','dihedrals']:
            raise ValueError('c should be one of these {"connectivity","angles","dihedrals"}')
        if c =='connectivity':
            ld = 2
        elif c =='angles':
            ld = 3
        elif c=='dihedrals':
            ld = 4
        for line in lines:
            if ';' in line:
                continue
            l = line.strip().split()[:ld]  
            if len(l)==0 or '[' in line or ']' in line:
                break
            ids = tuple(np.array(l,dtype=int)-sub)
            conn_id,c_type = self.sorted_id_and_type(ids)
            cad[conn_id] = c_type
        setattr(self,c,cad)
        return
    
    def read_gromacs_topology(self):
        #t0 = perf_counter()

        if ass.iterable(self.connectivity_file):
            for cf in self.connectivity_file:
                if '.itp' in cf:
                    self.read_itp_file(cf)          
                else:   
                    raise NotImplementedError('Non itp files are not implemented for lists. You can give a top file')
            self.local_to_global_topology()
            self.make_ff_from_itp(cf)
        elif '.top' == self.connectivity_file[-4:]:
            self.read_topfile(self.connectivity_file)
        else:
            if '.itp' == self.connectivity_file[-4:]:
                cf = self.connectivity_file
                self.read_itp_file(cf)
                self.local_to_global_topology()
                self.make_ff_from_itp(cf)
            else:
                raise NotImplementedError('Non itp files are not yet implemented') 

           
        return
    
    def local_to_global_topology(self,):
        self.connectivity = dict()
        self.angles = dict()
        self.dihedrals = dict()
        self.pairs = dict()
        self.exclusions = dict()
        self.refine_angles = False
        self.refine_dihedrals = False
        if not hasattr(self,'atom_charge'):
            self.atom_charge = np.empty(self.natoms,dtype=float)
        if not hasattr(self,'atom_mass'):
            self.atom_mass = np.empty(self.natoms,dtype=float)
        if not hasattr(self,'atom_code'):
            self.atom_code = np.empty(self.natoms,dtype=object)
        
        for j in np.unique(self.mol_ids):
            #global_mol_at_ids = self.at_ids[self.mol_ids==j]
            res_nm = np.unique(self.mol_names[self.mol_ids==j])
            
            assert res_nm.shape ==(1,),'many names for a residue. Check code or topology file'
            
            res_nm = res_nm[0]
            mol = self.molecule_map[res_nm]
           
            for i,idm in enumerate(mol['at_ids']):
                id0 = self.loc_id_to_glob[j][i]
                self.atom_charge[id0] = mol['charge'][i]
                self.atom_mass[id0] = mol['mass'][i]
                self.atom_code[id0] = mol['code'][i]
            local_connectivity = self.connectivity_per_resname[res_nm]
            local_angles = self.angles_per_resname[res_nm]
            local_dihedrals = self.dihedrals_per_resname[res_nm]
            local_pairs =  self.pairs_per_resname[res_nm]
            local_exclusions = self.exclusions_per_resname[res_nm]
            for b in local_connectivity:       
                id0 = self.loc_id_to_glob[j][b[0]]
                id1 = self.loc_id_to_glob[j][b[1]]
                conn_id,c_type = self.sorted_id_and_type((id0,id1))
                self.connectivity[conn_id] = c_type
            for a in  local_angles:
                id0 = self.loc_id_to_glob[j][a[0]]
                id1 = self.loc_id_to_glob[j][a[1]]
                id2 = self.loc_id_to_glob[j][a[2]]
                a_id,a_t = self.sorted_id_and_type((id0,id1,id2))
                self.angles[a_id] = a_t
            for d in local_dihedrals:
                id0 = self.loc_id_to_glob[j][d[0]]
                id1 = self.loc_id_to_glob[j][d[1]]
                id2 = self.loc_id_to_glob[j][d[2]]
                id3 = self.loc_id_to_glob[j][d[3]]
                d_id,d_t = self.sorted_id_and_type((id0,id1,id2,id3))
                self.dihedrals[d_id] = d_t
            for p in local_pairs:       
                id0 = self.loc_id_to_glob[j][p[0]]
                id1 = self.loc_id_to_glob[j][p[1]]
                conn_id,c_type = self.sorted_id_and_type((id0,id1))
                self.pairs[conn_id] = c_type
            for e in local_exclusions:       
                id0 = self.loc_id_to_glob[j][e[0]]
                id1 = self.loc_id_to_glob[j][e[1]]
                conn_id,c_type = self.sorted_id_and_type((id0,id1))
                self.exclusions[conn_id] = c_type
        return 

            
    def find_neibs(self):
        '''
        Computes first (bonded) neihbours of a system in dictionary format
        key: atom_id
        value: set of neihbours
        '''
        neibs = dict()
        for k in self.connectivity.keys(): 
            for i in k: neibs[i] = set() # initializing set of neibs
        for j in self.connectivity.keys():
            neibs[j[0]].add(j[1])
            neibs[j[1]].add(j[0])
        self.neibs = neibs
        return
    
    def correct_types_from_itp(self,itp_atoms):
        for i,t in enumerate(self.at_types.copy()):
            try:
                self.at_types[i] = itp_atoms[t]
            except KeyError:
                pass
        return
    def identify_element(self,ty):
        if ty[1:2].islower():
            return ty[:2]
        else:
            return ty[:1]
    @property
    def attribute_names(self):
        return list(self.__dict__.keys())
    
    def read_itp_file(self,file,find_mass=dict(),find_charge=dict()):
        t0 = perf_counter()
        
        with open(file,'r') as f:
            lines = f.readlines()
            f.closed
            
        # Reading atoms
        jlines = {'atoms':set(),'bonds':set(),'angles':set(),
                  'dihedrals':set(),'moleculetype':set(),'pairs':set(),
                  'exclusions':set(),'atomtypes':set()}
        for j,line in enumerate(lines):
            for k in jlines:
                if k in line and '[' in line and ']' in line:
                    jlines[k].add(j)
        
        ma = jlines['atomtypes']
        if len(ma) ==1:
            charge_defaults = dict()
            for line in lines[list(ma)[0]+1:]:

                if ';' in line:continue
                if len(line.split()) ==0: continue
                if '[' in line or ']' in line:
                    break
                
                l = line.strip().split()
                charge_defaults[l[-6]] = float(l[-4])
      
        mt = jlines['moleculetype']
        
        exclusions_map = dict()
        if len(mt) ==1:
            for line in lines[list(mt)[0]+1:]:
                if ';' in line:continue
                if len(line.split()) ==0: continue
                if '[' in line or ']' in line:
                    break
                
                l = line.strip().split()
                #print(l[0],l[1])
                exclusions_map[l[0]] = int(l[1])
        at_ids = [] ; res_num=[];at_types = [] ;res_name = [] ; cngr =[] 
        code = [] ; charge = [] ; mass = []

        atomscode_map = dict()
        jli = list(jlines['atoms'])[0]
        i=0

        for line in lines[jli+1:]:
            l = line.split()
            try:
                int(l[0])
            except:
                pass
            else:
                nl  = len(l)
                #print(l)
                at_ids.append(i) #already int
                res_name.append(l[3])
                t = l[4]
                cngr.append(l[5])
                atomscode_map[l[1]] = i
                code.append(l[1])
                at_types.append(t)
                if nl >6:    
                    charge.append( float(l[6]))
                elif len(find_charge)==0:
                    try:
                        chd = charge_defaults
                    except:
                        charge.append( 0.0 )
                    else:
                        charge.append(chd[t])
                else:
                    charge.append(find_charge[l[1]])
                if nl>7:
                    ms= float(l[7])
                else:
                    if len(find_mass)==0:
                        try:
                            ms = maps.elements_mass[self.identify_element(t)]
                        except KeyError:
                            ms = maps.elements_mass[self.identify_element(t[0])]

                    else:
                        ms = find_mass[l[1]]
                mass.append(ms)
                res_num.append(int(l[2]))
                i+=1
            if '[' in line or ']' in line:
                break
        
        at_ids = np.array(at_ids)
        at_types = np.array(at_types)
        res_num = np.array(res_num) 
        res_name = np.array(res_name)
        code = np.array(code)
        charge = np.array(charge)
        mass = np.array(mass)
        
        cngr = np.array(cngr)
        resnames = np.unique(res_name) 
        f = {rn: res_name == rn for rn in res_name}
        molecule_map = {rn : {'at_ids':at_ids[f[rn]],
                              'at_types':at_types[f[rn]],
                              'res_num':res_num[f[rn]],
                              'res_name':res_name[f[rn]],
                              'code':code[f[rn]],
                              'charge':charge[f[rn]],
                              'cngr':cngr[f[rn]],
                              'mass':mass[f[rn]]} 
                            for rn in resnames
                            }
        
        if not hasattr(self,'molecule_map'):
            self.molecule_map = molecule_map
        else:    
            self.molecule_map.update(molecule_map)
        if not hasattr(self,'exclusions_map'):
            self.exclusions_map = exclusions_map
        else:    
            self.exclusions_map.update(exclusions_map)
        if not hasattr(self,'atomscode_map'):
            self.atomscode_map = atomscode_map
        else:    
            self.atomscode_map.update( atomscode_map)


        jd = {'bonds':2,'angles':3,'dihedrals':4,'pairs':2,'exclusions':2}
        topol =  {'bonds':[],'angles':[],'dihedrals':[],'pairs':[],'exclusions':[]}

            
        for key in ['bonds','angles','dihedrals','pairs','exclusions']:
            for jli in jlines[key]:
                ffd = dict()
                for line in lines[jli+1:]:
                    l = line.split()
                    if len(l)<2:
                        continue
                    try:
                        b = np.array(l[:jd[key]],dtype=int)
                    except:
                        pass
                    else:
                        topol[key].append(b)
                    if '[' in line or ']' in line:
                        break

        
        bonds = np.array(topol['bonds'])
        
        try:
            bonds[0]
        except IndexError as e:
            logger.warning('Warning: File {:s} probably contains no bonds\n Excepted ValueError : {:}'.format(file,e))
          
        try:
            sub = bonds.min()
        except:
            sub =1
        self.sub =sub
        
        bonds -= sub
        angles = np.array(topol['angles']) - sub
        dihedrals = np.array(topol['dihedrals']) - sub
        pairs = np.array(topol['pairs']) - sub
        exclusions = np.array(topol['exclusions']) - sub
        
        
        
        
        
        pairs_per_resname = {t:[] for t in resnames }
        exclusions_per_resname = {t:[] for t in resnames }
        connectivity_per_resname = {t:[] for t in resnames }
        angles_per_resname = {t:[] for t in resnames }
        dihedrals_per_resname = {t:[] for t in resnames }
        
        for b in bonds:
            i0 = np.where(at_ids == b[0])[0][0]
            i1 = np.where(at_ids == b[1])[0][0]
            assert res_name[i0] == res_name[i1], 'Bond {:d} - {:d} is between two different residues'.format(i0,i1)
            res_nm = res_name[i0]
            connectivity_per_resname[res_nm].append(b)
        
        for a in angles:
            i0 = np.where(at_ids == a[0])[0][0]
            i1 = np.where(at_ids == a[1])[0][0]
            i2 = np.where(at_ids == a[2])[0][0]
            assert res_name[i0] == res_name[i1], 'Angle ids {:d} - {:d} is between two different residues'.format(i0,i1)
            assert res_name[i0] == res_name[i2], 'Angle ids {:d} - {:d} is between two different residues'.format(i0,i2)
            res_nm = res_name[i0]
            angles_per_resname[res_nm].append(a)
        
        for d in dihedrals:
            i0 = np.where(at_ids == d[0])[0][0]
            i1 = np.where(at_ids == d[1])[0][0]
            i2 = np.where(at_ids == d[2])[0][0]
            i3 = np.where(at_ids == d[2])[0][0]
            assert res_name[i0] == res_name[i1], 'Dihedral ids {:d} - {:d} is between two different residues'.format(i0,i1)
            assert res_name[i0] == res_name[i2], 'Dihedral ids {:d} - {:d} is between two different residues'.format(i0,i2)
            assert res_name[i0] == res_name[i3], 'Dihedral ids {:d} - {:d} is between two different residues'.format(i0,i3)
            res_nm = res_name[i0]
            dihedrals_per_resname[res_nm].append(d)
        
        for p in pairs:
            i0 = np.where(at_ids == p[0])[0][0]
            i1 = np.where(at_ids == p[1])[0][0]
            assert res_name[i0] == res_name[i1], 'Pair {:d} - {:d} is between two different residues'.format(i0,i1)   
            res_nm = res_name[i0]
            pairs_per_resname[res_nm].append(p)
        for e in exclusions:
            i0 = np.where(at_ids == e[0])[0][0]
            i1 = np.where(at_ids == e[1])[0][0]
            assert res_name[i0] == res_name[i1], 'Exclusion {:d} - {:d} is between two different residues'.format(i0,i1)   
            res_nm = res_name[i0]
            exclusions_per_resname[res_nm].append(e)
            
        for c in ['connectivity','angles','dihedrals','pairs','exclusions']:
            name = c+'_per_resname'
            var  = locals()[name]
            if not hasattr(self,name):
                setattr(self,name,var)
            else:
                attr = getattr(self,name)
                #updating
                for t,bad in var.items():
                    
                    attr[t] = bad
                setattr(self,name,attr)
        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name)
        return 
    
    def make_ff_from_itp(self,file):
        if not hasattr(self,'ff'):
            self.ff = self.FFparams()
        with open(file,'r') as f:
            lines = f.readlines()
            f.closed
        nline = dict()            
        lookfor = ['atomtypes','atoms','bonds','angles','dihedrals','bondtypes','angletypes','dihedraltypes']
        nline = {k:set() for k in lookfor}
        for j,line in enumerate(lines):
            for k in lookfor:
                if k in line and '[' in line and ']' in line:
                    nline[k].add(j)
        residues = []
        for line in lines[list(nline['atoms'])[0]+1:]:
            if ';' in line :
                continue
            l = line.strip().split()
            if len(l) == 0 :
                continue
            
            if '[' in line or ']' in line :
                break
            residues.append(l[3])
        ures = np.unique(residues)
        if len(ures) != 1:
            raise ValueError('Residues in this itp file are not unique')
        
        jd = {'atomtypes':1,'bondtypes':2,'angletypes':3,'dihedraltypes':4,'bonds':2,'angles':3,'dihedrals':4}
        names = {'atomtypes':'atomtypes','bonds':'bondtypes',
                 'angles':'angletypes','dihedrals':'dihedraltypes',
                 'bondtypes':'bondtypes','angletypes':'angletypes','dihedraltypes':'dihedraltypes'}
        mol_ids = self.mol_ids[self.mol_names==ures[0]]
        
        for k in ['atomtypes','bondtypes','angletypes','dihedraltypes']:            
            attr = dict()
            attr_name = names[k]
            for nlin in list(nline[k]):
                for j,line in enumerate(lines[nlin+1:]):
                    
                    if ';' in line: 
                        continue
                    if '[' in line or ']' in line : break
                    l = line.strip().split()
                    
                    if len(l) ==0: continue
                
                    if k =='atomtypes':
                        if not hasattr(self.ff,'atomtypes'):
                            self.ff.atomtypes = dict()
                        aid = self.atomscode_map[l[0]]
                        for res_id in np.unique(mol_ids): 
                            a = self.loc_id_to_glob[res_id][aid]
                            ty = self.at_types[a]
                            
                            mass1 = self.atom_mass[a]
                            mass2 = float(l[2]) if len(l) == 7 else float(l[1])
                            assert round(mass1,3) == round(mass2,3),'mass1 = {:4.3f}  while mass2 = {:4.3f}'.format(mass1,mass2)  
                            ch = self.atom_charge[a]
                            code = self.atom_code[a]
                            atyc = ty if self.bytype else code
                            val = (atyc,str(mass1),str(ch),*l[-3:])
                            attr[ty] = val
                    elif k in ['bondtypes','angletypes','dihedraltypes']:
                        tys = getattr(self,k.split('types')[0]+'_'+'types')
                        for ty in tys:
                            
                            atyc = ty #if self.bytype else code
                            val = ('  '.join(atyc),*l[jd[k]:])
                            attr[ty] = val
                       
                    else:
                        raise Exception('something wrong with naming. Check your code')
                    ass.update_dict_in_object(self.ff, attr_name, attr)
        for k in ['bonds','angles','dihedrals']:            
            attr = dict()
            attr_name = names[k]
            for nlin in list(nline[k]):
                for j,line in enumerate(lines[nlin+1:]):
                    
                    if ';' in line: 
                        continue
                    if '[' in line or ']' in line : break
                    l = line.strip().split()
                    
                    if len(l) ==0: continue                            
                    aid =  tuple(int(i)-self.sub for i in l[:jd[k]])
                    for res_id in np.unique(mol_ids):
                        a = tuple(self.loc_id_to_glob[res_id][i] for i in aid)
                        cid,ty = self.sorted_id_and_type(a)
                        code = tuple(self.atom_code[i] for i in cid)
                        atyc = ty if self.bytype else code
                        try:
                            default_val = getattr(self.ff,attr_name)[atyc]
                        except KeyError:
                            default_val = ( 'no default value check your code',)
                        except AttributeError:
                            default_val = ( 'not in {:s} check your code'.format(attr_name),)
                        
                        proposed_val = ('  '.join(atyc),*l[jd[k]:])
                        if len(default_val)>len(proposed_val):
                            val = default_val
                        else:
                            val = proposed_val
                        attr[ty] = val
                    ass.update_dict_in_object(self.ff, attr_name, attr)
        return 
    
    def read_topology(self):
        if '.gro' == self.topol_file[-4:]:
            self.read_gro_topol()  # reads from gro file
            try:
                self.connectivity_file
            except:
                raise NotImplementedError('connectivity_info = {} is not implemented yet'.format(self.connectivity_file))
            else:
                if type(self.connectivity_info) is dict: 
                    self.map_the_topology(self.connectivity_info)
                else:
                    self.read_gromacs_topology() # reads your itp files to get the connectivity
            if hasattr(self,'fftop'):
                self.read_topfile_ff(self.fftop)
        elif '.ltop' == self.topol_file[-5:]:
            self.read_lammps_topol()
        else:
            raise Exception('file {:s} not implemented'.format(self.topol_file.split('.')[-1]))
        #elif '.mol2' == self.topol_file[-5:]:
            #self.read_mol2_topol()
        return
    
    def read_lammps_topol(self):
        t0 = perf_counter()
        with open(self.topol_file,'r') as f:
            lines = f.readlines()
            f.closed
        
        def get_value(lines,valuename,dtype=int):
            for line in lines:
                if valuename in line:
                    return dtype(line.split()[0])
        
        def get_line_of_header(lines,header):
            for i,line in enumerate(lines):
                if header in line:
                    return i
        values = ['atoms','bonds','angles','dihedrals','impropers',
                  'atom types', 'bond types', 'angle types', 
                  'dihedral types','improper types']
        headers = ['Masses','Atoms','Bonds','Angles','Dihedrals']
        
        numbers = {v:get_value(lines,v) for v in values}
        
        header_lines = {hl:get_line_of_header(lines,hl) for hl in headers}
        
        natoms = numbers['atoms']
        mol_ids = np.empty(natoms,dtype=int)
        mol_nms = np.empty(natoms,dtype=object)
        at_tys  = np.empty(natoms,dtype=object)
        at_ids  = np.empty(natoms,dtype=int)
        hla = header_lines['Atoms']
        atom_lines = lines[hla+2 : hla+2+natoms]
        ncols = len(atom_lines[0].split())
        self.charge_map = dict()
        for i,line in enumerate(atom_lines):
            l = line.strip().split()
            mol_ids[i] = int(l[1])
            mol_nms[i] = l[1]
            at_tys[i] = l[2]
            at_ids[i] = int(l[0])
            if ncols==4 or ncols ==7 or ncols==10:
                self.charge_map[l[2]] = float(l[3])
            
        sort_ids = at_ids.argsort()
        mol_ids = mol_ids[sort_ids]
        mol_nms = mol_nms[sort_ids]
        at_tys = at_tys[sort_ids]
        at_ids = at_ids[sort_ids]
        starts_from = at_ids.min()
        for i in range(1,natoms):
            if at_ids[i-1]+1 != at_ids[i]:
                raise Exception('There are missing atoms')
        at_ids -= starts_from
        
        self.mol_ids = mol_ids
        self.mol_names = mol_nms
        self.at_types = at_tys
        self.at_ids = at_ids
        
        self.find_locGlob()
        
        self.connectivity= dict()

        hlb = header_lines['Bonds'] 
        nbonds = numbers['bonds']
        bond_lines = lines[hlb+2:hlb + nbonds+2]
        for  i,line in enumerate(bond_lines):  
            b = line.strip().split() 
            id0 = int(b[-2]) - starts_from
            id1 = int(b[-1]) - starts_from
            conn_id,c_type =  self.sorted_id_and_type((id0,id1))
            if conn_id in self.connectivity:
                logger.warning('{} is already in connectivity '.format(conn_id))
            self.connectivity[conn_id] = c_type
        
        self.mass_map = dict()
        hlm = header_lines['Masses']
        
        for line in lines[hlm+2:+hlm+2+numbers['atom types']]:
            l = line.strip().split()
            self.mass_map[l[0]] = float(l[1])
        
        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name)
        return 
    
    def read_gro_topol(self):
        with open(self.topol_file,'r') as f:
            f.readline()
            natoms = int(f.readline().strip())
            #allocate memory
            mol_ids = np.empty(natoms,dtype=int)
            mol_nms = np.empty(natoms,dtype=object)
            at_tys  = np.empty(natoms,dtype=object)
            at_ids  = np.empty(natoms,dtype=int)
            for i in range(natoms):
                line = f.readline()
                mol_ids[i] = int(line[0:5].strip())
                mol_nms[i] = line[5:10].strip()
                at_tys[i] = line[10:15].strip()
                at_ids[i] = i
            f.close()
        

        self.mol_ids = mol_ids
        self.mol_names = mol_nms
        self.at_types = at_tys
        self.at_ids = at_ids
        
        self.find_locGlob()
 
        return 
    
    def find_locGlob(self):
        loc_id_to_glob = dict() ; glob_id_to_loc = dict()
        for j in np.unique(self.mol_ids):
            loc_id_to_glob[j] = dict()
            glob_id_to_loc[j] = dict()
            filt = self.mol_ids== j
            res_nm = np.unique(self.mol_names[filt])
            if res_nm.shape !=(1,):
                raise ValueError('many names for a residue, res_id = {:d}'.format(j))
            else:
                res_nm = res_nm[0]
            g_at_id = self.at_ids[filt]
            
            for i,g in enumerate(g_at_id):
                loc_id = i
                loc_id_to_glob[j][loc_id] = g
                glob_id_to_loc[j][g] = loc_id
        
        self.loc_id_to_glob = loc_id_to_glob
        self.glob_id_to_loc = glob_id_to_loc 
        return 

    @property
    def natoms(self):
        return self.at_ids.shape[0]
    @property
    def ndihedrals(self):
        return len(self.dihedrals)
    @property
    def nbonds(self):
        return len(self.connectivity)
    @property
    def nangles(self):
        return len(self.angles)
    @staticmethod
    def unique_values(iterable):
        try:
            iter(iterable)
        except:
            raise Exception('Give an ass.iterable variable')
        else:
            un = []
            for x in iterable:
                if x not in un:
                    un.append(x)
            return un
    @property
    def atom_types(self):
        return self.unique_values(self.at_types)
    @property
    def bond_types(self):
        return self.unique_values(self.connectivity.values())
    @property
    def angle_types(self):
        return self.unique_values(self.angles.values())
    @property
    def dihedral_types(self):
        return self.unique_values(self.dihedrals.values())
    

    def sorted_type(self,t):
        if t[0]<=t[-1]:
            t = tuple(t)
        else:
            t = tuple(t[::-1])
        return t
    def sorted_id_and_type(self,a_id):
        t = [self.at_types[i] for i in a_id]
        if t[0]<=t[-1]:
            t = tuple(t)
            a_id = tuple(a_id)
        else:
            t = tuple(t[::-1])
            a_id = tuple(a_id[::-1])
        #if a_id[0]<=a_id[-1]:
       #     a_id = tuple(a_id)
        #else:
        #    a_id = tuple(a_id[::-1])
        return a_id,t
    
    @property
    def refining_angles_condition(self):
        try:
            a  = self.refine_angles
        except:
            return True
        else:
            return  a
    @property
    def refining_dihedrals_condition(self):
        try:
            a  = self.refine_dihedrals
        except:
            return True
        else:
            return  a
    def find_new_angdihs(self,new):
        angdihs = dict()
        for neib in self.neibs[new[0]]:
            if neib in new:
                continue
            idn = (neib,*new)
            idns,t = self.sorted_id_and_type(idn)
            angdihs[idns] = t
        for neib in self.neibs[new[-1]]:
            if neib in new:
                continue
            idn = (*new,neib)
            idns,t = self.sorted_id_and_type(idn)
            angdihs[idns] = t
        return angdihs
    
    def find_angles(self):
        '''
        Computes the angles of a system in dictionary format
        key: (atom_id1,atom_id2,atom_id3)
        value: object Angle
        Method:
            We search the neihbours of bonded atoms.
            If another atom is bonded to one of them an angle is formed
        We add in the angle the atoms that participate
        '''
        if not self.refining_angles_condition:
            return

        #t0 = perf_counter()
        self.angles = dict()
        for k in self.connectivity.keys():
            #"left" side angles k[0]
            for neib in self.neibs[k[0]]:
                if neib in k: continue
                ang_id ,ang_type = self.sorted_id_and_type((neib,k[0],k[1]))
                if ang_id[::-1] not in self.angles.keys():
                    self.angles[ang_id] = ang_type
            #"right" side angles k[1]
            for neib in self.neibs[k[1]]:
                if neib in k: continue
                ang_id ,ang_type = self.sorted_id_and_type((k[0],k[1],neib))
                if ang_id[::-1] not in self.angles.keys():
                    self.angles[ang_id] = ang_type  
        #tf = perf_counter()-t0
        #ass.print_time(tf,inspect.currentframe().f_code.co_name)
        return

    def find_dihedrals(self):
        '''
        Computes dihedrals of a system based on angles in dictionary
        key: (atom_id1,atom_id2,atom_id3,atom_id4)
        value: object Dihedral
        Method:
            We search the neihbours of atoms at the edjes of Angles.
            If another atom is bonded to one of them a Dihedral is formed is formed
        We add in the angle the atoms that participate
        '''
        if not self.refining_dihedrals_condition:
            return
        #t0 = perf_counter()
        self.dihedrals=dict()
        for k in self.angles.keys():
            #"left" side dihedrals k[0]
            for neib in self.neibs[k[0]]:
                if neib in k: continue
                dih_id,dih_type = self.sorted_id_and_type((neib,k[0],k[1],k[2]))
                if dih_id[::-1] not in self.dihedrals:
                    self.dihedrals[dih_id] = dih_type
            #"right" side dihedrals k[2]
            for neib in self.neibs[k[2]]:
                if neib in k: continue
                dih_id,dih_type = self.sorted_id_and_type((k[0],k[1],k[2],neib))
                if dih_id[::-1] not in self.dihedrals:
                    self.dihedrals[dih_id] = dih_type
        #tf = perf_counter()
        #ass.print_time(tf,inspect.currentframe().f_code.co_name)
        return
    
    def find_masses(self):
        
        mass = np.empty(self.natoms,dtype=float)
        for i in range(self.natoms):
            mass[i] = self.mass_map[self.at_types[i]]
        self.atom_mass = mass
       
        return
    
    @staticmethod
    def get_equal_from_string(s,v,make=float):
        try:
            x = s.split(v+'=')[-1].split()[0]
        except:
            x = s.split(v+' =')[-1].split()[0]
        return make(x) 
    
    def read_gro_by_frame(self,ofile,frame):
        line = ofile.readline()
        l = line.strip().split()
        if len(l)==0:
            return False
        #first line
        try:
            time = self.get_equal_from_string(line.strip(),'t')
        except:
            logger.warning('Warning: in gro file. There is no time info')
            time = 0
        try:
            step = self.get_equal_from_string(line.strip(),'step',int)
        except:
            step = 0
            logger.warning('Warning: in gro file. There is no step info')
        self.timeframes[frame] = {'time':time,'step':step}
        # second line
        natoms = int(ofile.readline().strip())
        if natoms != self.natoms:
            raise ValueError('This frame has {} atoms instead of {}'.format(natoms,self.natoms))
        
        #file 
        coords = np.empty((natoms,3),dtype=float)
        for i in range(natoms):
            line = ofile.readline()
            l=line[20:44].split() 
            coords[i,0] = float(l[0]) 
            coords[i,1] = float(l[1])  
            coords[i,2] = float(l[2])
        
        box = np.array(ofile.readline().strip().split(),dtype=float)
        
        self.timeframes[frame]['coords'] = coords
        self.timeframes[frame]['boxsize'] = box
        return True
    
    def read_trr_by_frame(self,ofile,frame):
        try:
            header,data = ofile.read_frame()
        except EOFError:
            raise EOFError
        except Exception:
            return True
        self.timeframes[frame] = header
        self.timeframes[frame]['boxsize'] = np.diag(data['box']).copy()
        self.timeframes[frame]['coords'] = data['x']
        try:
            self.timeframes[frame]['velocities'] = data['v']
        except:
            pass
        try:
            self.timeframes[frame]['forces'] = data['f']
        except:
            pass
        return True
   
    def read_lammpstrj_by_frame(self,reader,frame):
        conf = reader.readNextStep()
        if conf is None: 
            return False
        if not reader.isSorted(): reader.sort()                
        uxs = conf['xu']
        uys = conf['yu']
        uzs = conf['zu']
        # allocate 
        natoms = uxs.shape[0]
        coords = np.empty((natoms,3))
        coords[:,0] = uxs ; coords[:,1] = uys ; coords[:,2] = uzs
        cbox = conf['box_bounds']
        tricl = cbox[:,2] != 0.0
        if tricl.any():
            raise NotImplementedError('Triclinic boxes are not implemented')
        offset = cbox[:,0]
        boxsize = cbox[:,1]-cbox[:,0]
        coords -= offset
        #make nm
        boxsize/=10
        coords/=10
        try:
            dt = self.kwargs['dt']
        except KeyError:
            dt =1e-6
        self.timeframes[frame] = {'conf':conf,'time':conf['step_no']*dt,'step':conf['step_no'],
                                  'boxsize':boxsize,'coords':coords}
        return True

    def read_from_disk_or_mem(self,ofile,frame):
        def exceptEOF():
            try:
                ret = self.read_by_frame(ofile, frame)
            except EOFError:
                return False
            else:
                return ret
            
        if self.memory_demanding:
            return exceptEOF()
        elif frame in self.timeframes.keys():
            self.is_the_frame_read =True
            return True
        else:
            try:
                if self.is_the_frame_read:
                    return False
            except AttributeError:
               return exceptEOF()

    def read_lammpstrj_file(self,num_end=int(1e16)):
        t0 = perf_counter()
        with lammpsreader.LammpsTrajReader(self.trajectory_file) as ofile:
            nframes = 0
            while( self.read_lammpstrj_by_frame(ofile, nframes) and nframes<=num_end):
                if self.memory_demanding:
                    del self.timeframes[nframes]
                nframes += 1
        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        return
    
    def read_trr_file(self,num_end=int(1e16)):
        t0 = perf_counter()
        with GroTrrReader(self.trajectory_file) as ofile:
            end = False
            nframes = 0
            while( end == False and nframes<=num_end):
                try:
                    self.read_trr_by_frame(ofile, nframes)
                except EOFError:
                    end = True
                else:
                    if self.memory_demanding:
                        del self.timeframes[nframes]
                    nframes += 1
        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        return 

    def read_gro_file(self,num_end=int(1e16)):
        t0 = perf_counter()
        with open(self.trajectory_file,'r') as ofile:
            nframes =0
            while(self.read_gro_by_frame(ofile,nframes) and nframes<=num_end):
                
                if self.memory_demanding:
                    del self.timeframes[nframes]
                nframes+=1
            ofile.close()
            tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        return 
    
    def read_file(self,trajectory_file=None,num_end=int(1e16)):    
       
        if trajectory_file is None:
            try:
                # just checking if these attributes excist 
                self.trajectory_file
                self.trajectory_file_type
                self.read_by_frame
                self.traj_opener
                self.traj_opener_args
            except AttributeError:
                raise Exception('You need to provide a trajectory file to read. The reading is not set')
        else:
            self.setup_reading(trajectory_file)
        
        if   self.traj_file_type == 'gro':
            self.read_gro_file(num_end)
        elif self.traj_file_type == 'trr': 
            self.read_trr_file(num_end)
        elif self.traj_file_type =='lammpstrj':
            self.read_lammpstrj_file(num_end)
        return 
    
    def setup_reading(self,trajectory_file):
        self.trajectory_file = trajectory_file
        if '.gro' == self.trajectory_file[-4:]:
            self.traj_file_type = 'gro'
            self.read_by_frame = self.read_gro_by_frame # function
            self.traj_opener = open
            self.traj_opener_args = (self.trajectory_file,)
        elif '.trr' == self.trajectory_file[-4:]:
            self.traj_file_type ='trr'
            self.read_by_frame =  self.read_trr_by_frame # function
            self.traj_opener = GroTrrReader
            self.traj_opener_args = (self.trajectory_file,)
        elif '.lammpstrj' == self.trajectory_file[-10:]:
            self.traj_file_type ='lammpstrj'
            self.read_by_frame = self.read_lammpstrj_by_frame
            self.traj_opener = lammpsreader.LammpsTrajReader
            self.traj_opener_args = (self.trajectory_file,)
        else:
            raise NotImplementedError('Trajectory file format ".{:s}" is not yet Implemented'.format(trajectory_file.split('.')[-1]))
        
    def write_gro_file(self,fname=None,whole=False,
                       option='',frames=None,step=None,**kwargs):
        t0 = perf_counter()
        options = ['','transmiddle','translate']
        if option not in options:
            raise ValueError('Available options are : {:s}'.format(', '.join(options)))
        
        if fname is None:
            fname = 'Analyisis_written.gro'
        with open(fname,'w') as ofile:
            for frame,d in self.timeframes.items():
                if frames is not None:
                    if  frame <frames[0] or frame>frames[1]:
                        continue
                if step is not None:
                    if frame%step  !=0: 
                        continue

                if option =='transmiddle':
                    coords = self.translate_particle_in_box_middle(self.get_coords(frame),
                                                          self.get_box(frame))
                elif option=='':
                    coords = self.get_coords(frame)
                    
                elif option=='translate':
                    v = np.array(kwargs['trans'])
                    coords = self.get_coords(frame) + v
                    coords = implement_pbc(coords,d['boxsize'])
                else:
                    raise NotImplementedError('option "{:}" Not implemented'.format(option))
                if whole:
                        coords = self.unwrap_coords(coords, d['boxsize'])                             
                
                self.write_gro_by_frame(ofile,
                                        coords, d['boxsize'],
                                        time = d['time'], 
                                        step =d['step'])
            ofile.close()
        tf = perf_counter() -t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name)
        return

    def write_gro_by_frame(self,ofile,coords,box,name='gro_by_frame',time=0,step=0):
        ofile.write('{:s},  t=   {:4.3f}  step=   {:8.0f} \n'.format(name,time,step))
        ofile.write('{:6d}\n'.format(coords.shape[0]))
        for i in range(coords.shape[0]):
            c = coords[i]
            ofile.write('%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n'\
            % (self.mol_ids[i],self.mol_names[i], self.at_types[i],self.at_ids[i%99999]+1 ,c[0],c[1] ,c[2] ))
        ofile.write('%f  %f  %f\n' % (box[0],box[1],box[2]))
        return  
    
    def renumber_residues(self,start_from=1):
        mol_ids = np.empty(self.natoms,dtype=int)        
        counter =start_from
        mol_ids[0] = start_from
        
        for i in range(1,self.natoms):
            mid0 = self.mol_ids[i-1]
            mid1 = self.mol_ids[i]
            if mid1!=mid0:
                counter+=1
            mol_ids[i] = counter
        self.mol_ids = mol_ids
     
        return
        
    def merge_ff(self,obj,add=''):
        a = self.add_tuple
        names = ['atomtypes','bondtypes','angletypes','dihedraltypes']
        def mod0(t):
            t = list(t)
            t[0] = ' '.join([b +add for b in t[0].split(' ')  ])
            return tuple(t)
        def mod01(t):
            t = list(t)
            t[0] = t[0] + add
            t[1] = t[1] + add
            try:
                float(t[1])
            except:
                t[1] =t[1][:-1]
            return tuple(t)
        for name in names:
            ffdata = getattr(self.ff,name)
            medata = getattr(obj.ff,name)
            if name =='atomtypes':
                ffdata.update({k+add:mod01(v) for k,v in medata.items() })
            else:
                ffdata.update({a(k,add):mod0(v) for k,v in medata.items() })
    @staticmethod
    def add_tuple(t,add=''):
        return tuple(y+add for y in list(t))
    def update_topology(self,n,obj,add=''):
        a = self.add_tuple
        self.connectivity.update( {(n+c[0],n+c[1]) : a(t,add) for c,t in obj.connectivity.items()} )
        self.pairs.update( {(n+c[0],n+c[1]):a(t,add) for c,t in obj.pairs.items()} )
        self.exclusions.update( {(n+c[0],n+c[1]):a(t,add) for c,t in obj.exclusions.items()} )
        #self.find_neibs()
        self.neibs.update({n+aid: {n+a for a in neibs} for aid,neibs in obj.neibs.items()} )
        self.angles.update( {(n+c[0],n+c[1],n+c[2]):a(t,add) for c,t in obj.angles.items()} )
        self.dihedrals.update( {(n+c[0],n+c[1],n+c[2],n+c[3]):a(t,add) for c,t in obj.dihedrals.items()} )
        return 
    def merge_system(self,obj,add=''):
        n = self.natoms

        self.update_topology(n,obj,add)
       
        self.at_ids = np.concatenate((self.at_ids,obj.at_ids))
        self.at_types = np.concatenate((self.at_types,obj.at_types +add))
        self.mol_ids = np.concatenate((self.mol_ids,obj.mol_ids))
        self.mol_names = np.concatenate((self.mol_names,obj.mol_names))
        self.atom_mass = np.concatenate((self.atom_mass,obj.atom_mass))
        self.atom_charge = np.concatenate((self.atom_charge,obj.atom_charge))
        self.atom_code = np.concatenate((self.atom_code,obj.atom_code+add))
        for frame in self.timeframes:
            c1 = self.get_coords(frame)
            c2 = obj.get_coords(frame)
            self.timeframes[frame]['coords'] = np.concatenate((c1,c2))
        
        self.renumber_ids()
        
        self.renumber_residues()
        self.find_locGlob()
        
        self.merge_ff(obj,add)
        
        self.exclusions_map.update(obj.exclusions_map)

        return
    
    
        
    def renumber_ids(self,start_from=0):
        at_ids = np.arange(0,self.natoms,1,dtype=int)
        at_ids+=start_from
        self.at_ids = at_ids
        return
    
            
    def remove_atoms_ids(self,ids,reinit=False,filter_topology=True):
        filt = np.logical_not(np.isin(self.at_ids,ids))
        self.filter_system(filt,reinit,filter_topology)
        return
    
    def remove_atoms(self,crit,frame=0,reinit=True):
        coords = self.get_coords(frame)
        filt = crit(coords)
        filt = np.logical_not(filt)
        self.filter_system(filt,reinit)
        return
    
    def remove_residues(self,crit,frame=0,reinit=True):
        coords = self.get_coords(frame)
        filt = crit(coords)
        res_cut = self.mol_ids[filt]
        filt = ~np.isin(self.mol_ids,res_cut)
        self.filter_system(filt,reinit)
        return
    
    def filter_topology(self,removed_ids):
        
        m = self.old_to_new_ids
        
        for i in removed_ids:
            try:
                del self.neibs[i]
            except KeyError:
                pass
            
        for c in ass.numpy_keys(self.connectivity):
            if c[0] in removed_ids or c[1] in removed_ids:
                del self.connectivity[tuple(c)]
        for c in ass.numpy_keys(self.pairs):
            if c[0] in removed_ids or c[1] in removed_ids:
                del self.pairs[tuple(c)]
        for c in ass.numpy_keys(self.exclusions):
            if c[0] in removed_ids or c[1] in removed_ids:
                del self.exclusions[tuple(c)]
        for a in ass.numpy_keys(self.angles):
            if a[0] in removed_ids or a[1] in removed_ids or a[2] in removed_ids:
                del self.angles[tuple(a)]
        for d in ass.numpy_keys(self.dihedrals):
            if d[0] in removed_ids or d[1] in removed_ids or d[2] in removed_ids or d[3] in removed_ids:
                del self.dihedrals[tuple(d)]
                    
        
        self.connectivity = {(m[i[0]],m[i[1]]):t for i,t in self.connectivity.items() }
        self.pairs = {(m[i[0]],m[i[1]]):t for i,t in self.pairs.items() }
        self.exclusions = {(m[i[0]],m[i[1]]):t for i,t in self.exclusions.items() }
        self.find_neibs()
        self.angles = {(m[i[0]],m[i[1]],m[i[2]]):t for i,t in self.angles.items() }
        self.dihedrals = {(m[i[0]],m[i[1]],m[i[2]],m[i[3]]):t for i,t in self.dihedrals.items()}
                
        return
    
    def filter_ff(self):
        at = self.atom_types
        bt  = self.bond_types
        angt = self.angle_types
        diht = self.dihedral_types
        tys = [at,bt,angt,diht]
        names = ['atomtypes','bondtypes','angletypes','dihedraltypes']
        for ty,name in zip(tys,names):
            ffdata = getattr(self.ff,name)
            for t in list(ffdata.keys()):
                if t not in ty:
                    del ffdata[t]
        return 
    def filter_system(self,filt,reinit=False,filter_topology=True):
        
        otn = dict()
        sub = 0 
        
        for j,f in enumerate(filt):
            if not f:
                sub+=1
                otn[j] = None
            else:
                otn[j] = j-sub
        
        self.old_to_new_ids = otn
        
        
        if filter_topology:
            removed_ids = self.at_ids[~filt]
            self.filter_topology(removed_ids)


        
        self.at_ids = self.at_ids[filt]
        self.at_types = self.at_types[filt]
        self.mol_ids = self.mol_ids[filt]
        self.mol_names = self.mol_names[filt]
        self.atom_mass = self.atom_mass[filt]
        self.atom_charge = self.atom_charge[filt]
        self.atom_code = self.atom_code[filt]
        for frame in self.timeframes:
            self.timeframes[frame]['coords'] = self.get_coords(frame)[filt]
        
        self.renumber_ids()
        self.renumber_residues()
        self.find_locGlob()
        self.filter_ff()
        if reinit:
            self.topology_initialization(reinit)
        return
    
    def write_residues(self,res,fname='selected_residues.gro',
                       frames=(0,0),box=None,boxoff=0.4):
        with open(fname,'w') as ofile:
            fres = np.isin(self.mol_ids, res)
            
            for frame in self.timeframes:
                if frames[0] <= frame <= frames[1]:

                    coords = self.get_coords(frame) [fres]
                    coords -= coords.min(axis=0)
                    at_ids = self.at_ids[fres] 
                    ofile.write('Made by write_residues\n')
                    ofile.write('{:6d}\n'.format(coords.shape[0]))
                    for j in range(coords.shape[0]):
                        i = at_ids[j]
                        c = coords[j]
                        ofile.write('%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n'\
                        % (self.mol_ids[i],self.mol_names[i], self.at_types[i]
                           ,self.at_ids[i%100000] ,c[0],c[1] ,c[2] ))
                    if box is None: 
                        box = coords.max(axis=0) - coords.min(axis=0) + boxoff
                    ofile.write('%f  %f  %f\n' % (box[0],box[1],box[2]))  
            
            ofile.closed
        return
    
    def apply_pbc(self):
        for frame in self.timeframes:
            box = self.get_box(frame)
            coords = self.get_coords(frame)
            self.timeframes[frame]['coords'] = implement_pbc(coords,box)
        return
    
    @property
    def nmolecules(self):
        return np.unique(self.mol_ids).shape[0]
    
    def find_args_per_residue(self,filt,attr_name):
        args = dict()
        for j in np.unique(self.mol_ids[filt]):
            x = np.array(np.where(self.mol_ids==j)[0],dtype=int)
            args[j] = x
        setattr(self,attr_name,args)
        setattr(self,'N'+attr_name, len(args))
        return 
    
    
    
    def multiply_periodic(self,multiplicity,one_molecule=False):
        if len(multiplicity) !=3:
            raise Exception('give the multyplicity in the form (times x, times y, times z)')
        for d,mult in enumerate(multiplicity):
            if mult ==0: continue
        
            natoms = self.natoms
            nmols = self.nmolecules
            totm = mult+1
        
            if totm<2:
                raise Exception('multiplicity {} is not valid'.format(multiplicity))
            #new topology
            shape = natoms*totm
            # allocate array
            at_ids = np.empty(shape,dtype=int)
            mol_ids = np.empty(shape,dtype=int)
            at_types = np.empty(shape,dtype=object)
            mol_names = np.empty(shape,dtype=object)
            atom_code = np.empty(shape,dtype=object)
            atom_mass = np.empty(shape,dtype=float)
            atom_charge = np.empty(shape,dtype=float)
            for m in range(0,mult+1):
                na = m*natoms
                self.update_topology(na,self)
                mm = m*nmols if not one_molecule else 0
                for i in range(natoms):
                    idx = i+na
                    at_ids[idx] = idx
                    at_types[idx] =self.at_types[idx%natoms]
                    mol_ids[idx] = self.mol_ids[idx%natoms]+mm
                    mol_names[idx] = self.mol_names[idx%natoms]
                    atom_code[idx] = self.atom_code[idx%natoms]
                    atom_mass[idx] = self.atom_mass[idx%natoms]
                    atom_charge[idx] = self.atom_charge[idx%natoms]
            self.at_ids = at_ids
            self.at_types = at_types
            self.mol_ids = mol_ids
            self.mol_names = mol_names
            self.atom_code = atom_code
            self.atom_mass = atom_mass
            self.atom_charge = atom_charge
            
            
            #allocate coords
            
            for frame in self.timeframes:
                coords = np.empty((shape,3))
                c = self.get_coords(frame)
                box = self.get_box(frame)
            
                idx = 0
                coords[idx:idx+natoms] = c.copy()
                for j in range(1,mult+1):
                    L = box[d]*j
                    idx=j*natoms
                    coords[idx:idx+natoms] = c.copy()
                    coords[idx:idx+natoms,d]+=L
            
                self.timeframes[frame]['coords'] = coords
                self.timeframes[frame]['boxsize'][d]*=mult+1

        self.resort_by_molname()
        self.find_locGlob()
        return
    
    def resort_by_molname(self):
        #first mol first
        names = []
        for name in self.mol_names:
            if name not in names:
                names.append(name)
        
        #map_ids old to new
        mapids = dict()
        n = 0
        for name in names:
            argsn = np.where(self.mol_names==name)[0]
            for i,j in enumerate(argsn):
                mapids[j] = i  + n
            n = argsn.shape[0]
        for attrname in ['at_types','mol_names','atom_code','atom_mass','atom_charge','mol_ids']:
            attrold = getattr(self,attrname)
            attrnew = np.empty_like(attrold)
            for j,i in mapids.items():
                attrnew[i] = attrold[j]
            setattr(self,attrname,attrnew)
        
        self.resorting_mapids = mapids
        self.renumber_residues()
        
        self.connectivity = {(mapids[c[0]],mapids[c[1]]): val for c,val in self.connectivity.items()}
        self.pairs = {(mapids[c[0]],mapids[c[1]]): val for c,val in self.pairs.items()}
        self.exclusions = {(mapids[c[0]],mapids[c[1]]): val for c,val in self.exclusions.items()}
        self.angles = {(mapids[c[0]],mapids[c[1]],mapids[c[2]]): val for c,val in self.angles.items()}
        self.dihedrals = {(mapids[c[0]],mapids[c[1]],mapids[c[2]],mapids[c[3]]): val for c,val in self.dihedrals.items()}
        self.neibs = {mapids[j]:{mapids[neib] for neib in s } for j,s in self.neibs.items()}
        
        for frame in self.timeframes:
            
            cold = self.get_coords(frame)
            cnew = np.empty_like(cold)
            
            for j,i in mapids.items():
                cnew[i] = cold[j]
            self.timeframes[frame]['coords'] = cnew
            
        return
    
    def get_coords(self,frame):
        return self.timeframes[frame]['coords']
    
    def get_velocities(self,frame):
        return self.timeframes[frame]['velocities']
    
    def get_forces(self,frame):
        return self.timeframes[frame]['forces']
    
    def get_box(self,frame):
        return self.timeframes[frame]['boxsize']
    
    def get_time(self,frame):
        return self.timeframes[frame]['time']
    
   
    def get_whole_coords(self,frame):
        coords = self.get_coords(frame)
        box = self.get_box(frame)
        coords = self.unwrap_coords(coords, box)
        return coords
    @property
    def dihedrals_per_type(self):
        
        tys = self.dihedral_types
        d = {t:[] for t in tys}
        for k,t in self.dihedrals.items():
            d[t].append(np.array(k))
        for t in tys:
            d[t] = np.array(d[t])
        return d
    @property
    def connectivity_per_type(self):
        
        tys = self.bond_types
        d = {t:[] for t in tys}
        for k,t in self.connectivity.items():
            d[t].append(np.array(k))
        for t in tys:
            d[t] = np.array(d[t])
        return d
    @property
    def angles_per_type(self):
        
        tys = self.angle_types
        d = {t:[] for t in tys}
        for k,t in self.angles.items():
            d[t].append(np.array(k))
        for t in tys:
            d[t] = np.array(d[t])
        return d
    
    def ids_from_topology(self,topol_vector):
        inter = len(topol_vector)
        if inter == 2: 
            ids = self.connectivity_per_type
        elif inter == 3: 
            ids = self.angles_per_type
        elif inter == 4: 
            ids = self.dihedrals_per_type
        else:
            raise Exception('Large topology vectors with size >= {} are not Implemented'.format(inter))
        topol_vector = tuple(topol_vector)
      
        if topol_vector in ids: tp = topol_vector
        elif topol_vector[::-1] in ids: tp = topol_vector[::-1]
        else:
            raise Exception('{} not in {}'.format(topol_vector,list(ids)))
        
        arr0 = ids[tp][:,0] ; arr1 = ids[tp][:,-1]
        
        return arr0,arr1
   
    def ids_from_keyword(self,keyword,exclude=[]):
        if keyword in ['4',4,'1-4']:
            ids = self.dihedrals_per_type
        if keyword in ['3',3,'1-3']:
            ids = self.angles_per_type
        if keyword in ['2',2,'1-2']:
            ids = self.connectivity_per_type
        ids1 = np.empty(0,dtype=int)
        ids2 = np.empty(0,dtype=int)
        for k,i in ids.items():
            if k in exclude:
                continue
            ids1 = np.concatenate( (ids1,i[:,0]) )
            ids2 = np.concatenate( (ids2,i[:,-1]) )
        return ids1,ids2
    

    def ids_from_backbone(self,bonddist):
        ids = self.polymer_ids
        if hasattr(self,'backbone_dist_matrix'):
            bd = self.backbone_dist_matrix
        else:
            bd = self.find_bond_distance_matrix(ids) 
            self.backbone_dist_matrix = bd
        
        b1,b2 = np.nonzero(bd == bonddist)
        ids1 = ids[b1]
        ids2 = ids[b2]
        
        return ids1,ids2
    
    def find_vector_ids(self,topol_vector,exclude=[]):
        '''
        

        Parameters
        ----------
        topol_vector : list of atom types, or int for e.g. 1-2,1-3,1-4 vectors
            Used to find e.g. the segmental vector ids
        exclude : TYPE, optional
            DESCRIPTION. The default is [].

        Returns
        -------
        ids1 : int array of atom ids
        ids2 : int array of atom ids

        '''
        t0 = perf_counter()
        ty = type(topol_vector)
        if ty is list or ty is tuple:
            ids1,ids2 = self.ids_from_topology(topol_vector)
        if  ty is int:
            if topol_vector<=4:
                ids1,ids2 = self.ids_from_keyword(topol_vector,exclude)
            else:
                ids1,ids2 = self.ids_from_backbone(int(topol_vector))
        if ty is str:
            dump = True
            for k in ['-','_',' ']:
                if k in topol_vector:
                    t1,t2 = tuple(topol_vector.split(k))
                    ids1 = self.at_ids[self.at_types == t1]
                    ids2 = self.at_ids[self.at_types == t2]
                    dump = False
            if dump:
                raise Exception('could not find topology ids')
        
        logger.info('time to find vector list --> {:.3e}'.format(perf_counter()-t0))
        return ids1,ids2
    
    
    def find_bond_distance_matrix(self,ids):
        '''
        takes an array of atom ids and finds how many bonds 
        are between each atom id with the rest on the array

        Parameters
        ----------
        ids : numpy array of int

        Returns
        -------
        distmatrix : numpy array shape = (ids.shape[0],ids.shape[0])
        
        '''
        
        t0 = perf_counter()
        size = ids.shape[0]
        distmatrix = np.zeros((size,size),dtype=int)
        for j1,i1 in enumerate(ids):
            nbonds = self.bond_distance_id_to_ids(i1,ids)
            distmatrix[j1,:] = nbonds
        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name)
        return distmatrix
    
    def bond_distance_id_to_ids(self,i,ids):
        '''
        takes an atom id and find the number of bonds between it 
        and the rest of ids. 
        If it is not connected the returns a very
        large number
        
        Parameters
        ----------
        i : int
        ids : array of int

        Returns
        -------
        nbonds : int array of same shape as ids
            number of bonds of atom i and atoms ids

        '''
        chunk = {i}
        n = ids.shape[0]
        nbonds = np.ones(n)*(-1)
        incr_bonds = 0
        new_neibs = np.array(list(chunk))
        while new_neibs.shape[0]!=0:
            f = np.zeros(ids.shape[0],dtype=bool)
            numba_isin(ids,new_neibs,f)
            nbonds[f] = incr_bonds
            new_set = set()
            for ii in new_neibs:
                for neib in self.neibs[ii]:
                    if neib not in chunk:
                        new_set.add(neib)
                        chunk.add(neib)
            new_neibs = np.array(list(new_set))
            incr_bonds+=1
        return nbonds

                
    def ids_nbondsFrom_args(self,ids,args):
        '''
        This function finds the minimum number of bonds
        that each atom id in ids has from args.
        If an id is not connected in any way with any of the args then
        a very large number is return within the nbonds array

        Parameters
        ----------
        ids : int array
        args : int array

        Returns
        -------
        nbonds : int array of shape as ids

        '''
        n = ids.shape[0]
        nbonds = np.ones(n)*10**10
        
        #old_chunk = set() ;
        new_neibs = args.copy()
        chunk = set(args)
        incr_bonds=0
        #same_set = False
        fnotin_args = np.logical_not(np.isin(ids,args))

        while  new_neibs.shape[0] !=0:
            
            #f = np.logical_and(np.isin(ids,new_neibs),fnotin_args)
            f = np.zeros(ids.shape[0],dtype=bool)
            numba_isin(ids,new_neibs,f)
            f = np.logical_and(f,fnotin_args)
            
            
            nbonds [ f ] = incr_bonds
            new_set = set()
            for ii in new_neibs:
                for neib in self.neibs[ii]:                    
                    if neib in chunk: continue        
                    new_set.add(neib)
                    chunk.add(neib)
            
            new_neibs = np.array(list(new_set)) 
            incr_bonds += 1
            
        return nbonds
    
    def append_timeframes(self,object2):
        '''
        Used to append one trajectory to another
        !It assumes that time of the second starts at the end of the first
        Parameters
        ----------
        object2 : A second object of the class Analysis

        Returns
        -------
        None.

        '''
        tlast = self.get_time(self.nframes-1)
        nfr = self.nframes
        for i,frame in enumerate(object2.timeframes):
            self.timeframes[nfr+i] = object2.timeframes[frame]
            self.timeframes[nfr+i]['time']+=tlast
        return    
    def unwrap_coords(self,coords,box):   
        '''
        Do not trust this function. Works only for linear polymers
        Parameters
        ----------
        coords : 
        box : 

        Returns
        -------
        unc : Unrwap coordinates.

        '''
        unc = coords.copy()
        b2 =  box/2
        k0 = self.sorted_connectivity_keys[:,0]
        k1 = self.sorted_connectivity_keys[:,1]
        n = np.arange(0,k0.shape[0],1,dtype=int)
        dim = np.array([0,1,2],dtype=int)

        unc = unwrap_coords_kernel(unc, k0, k1, b2, n, dim, box)
        
        return unc

    def unwrap_all(self):
        for k in self.timeframes:
            coords = self.timeframes[k]['coords']
            box =self.timeframes[k]['boxsize']
            self.timeframes[k]['coords'] = self.unwrap_coords(coords, box)
        return
    
    def get_reference_CM(self,coords):
        try:
            self.particle_filt
        except:
            cm = CM( coords, self.atom_mass )
        else:
            cm = CM( coords[self.particle_filt], self.particle_mass)
        return cm
    
    def box_mean(self):
        t0 = perf_counter()
        box = np.zeros(3)
        args = (box,)
        nframes = self.loop_trajectory('box_mean', args)
        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name)
        return box/nframes
    
    def find_EndGroup_args(self,ty = None,serial=True):
        eargs = []
        if serial:
            for j,args in self.chain_args.items():
                 eargs.append(args[0]) ; eargs.append(args[-1])
            eargs = np.array(eargs)
        elif ty is not None:
            eargs = self.at_ids[self.at_types == ty]
        else:
            raise NotImplementedError('give either the type or serial=True if your atom chains are seriarly stored')
        self.EndGroup_args = eargs
        return
    
    def get_EndGroup_args(self):
        try:
            args = self.EndGroup_args
        except AttributeError as err:
            raise AttributeError('{}\nCall function "find_EndGroup_args" to set this attribute'.format(err))
        else:
            return args   
    @staticmethod
    def element_from_type(ty):
        elements = list(maps.elements_mass.keys())
        n = len(ty)
        while ty[:n] not in elements:
            n-=1
            if n==0:
                raise ValueError('cannot find a corresponding element for type {:s}'.format(ty))
        return ty[:n]
    
    def match_types(self,types1,types2):
        
        for ctype1,ctype2 in zip(types1,types2):
            
            if len(ctype1) == 2:
                dictionary = self.ff.bondtypes
            elif len(ctype1) ==3:
                dictionary = self.ff.angletypes
            elif len(ctype1) == 4:
                dictionary = self.ff.dihedraltypes

            val = list(dictionary[ctype2])
            val[0] = '  '.join(ctype1)
            dictionary[ctype1] = tuple(val) 
        return
    
    
    def element_based_matching(self,types):
        
        for ctype1 in types:
            eletype1 = tuple(self.element_from_type(c) for c in ctype1) 
            if len(ctype1) == 2:
                dictionary = self.ff.bondtypes
            elif len(ctype1) ==3:
                dictionary = self.ff.angletypes
            elif len(ctype1) == 4:
                dictionary = self.ff.dihedraltypes
                
            for ctype2 in list(dictionary.keys()):
                eletype2 = tuple(self.element_from_type(c) for c in ctype2)
                if eletype1 == eletype2:
                    val = list(dictionary[ctype2])
                    val[0] = '  '.join(ctype1)
                    print('matching type {} to {}'.format(ctype1,ctype2))
                    dictionary[ctype1] = tuple(val) 
                    break
        return
                    

        
    def get_chem_per_molecule(self,mol_id):
        cad = ['connectivity','angles','dihedrals','pairs','exclusions']
        b = {c:[] for c in cad}
        at_ids = self.at_ids[self.mol_ids==mol_id]
        for c in cad:
            data = getattr(self,c)
            if len(data) ==0:
                b[c] = []
                continue
            c_ids = np.array(list(data.keys()))
            
            f2 = np.isin(c_ids, at_ids)
            f = np.ones(f2.shape[0],dtype=bool)
            for j in range(f2.shape[1]):
                f = np.logical_and(f,f2[:,j])
            cfi = c_ids [ f ]
            b[c] =  [tuple(j for j in cfi[i])  for i in range(cfi.shape[0])]
        return b
    @property
    def molecules(self):
        mols = dict()
        for nm in np.unique(self.mol_names):
            mols[nm] = np.unique(self.mol_ids[nm == self.mol_names]).shape[0]
        return mols
    
    def write_itp(self,path,Defaults=True,
                  include_pairs=[],include_exclusions=[]):
        
        mols = self.molecules 
        for k in mols:
            try:
                nexcl = self.exclusions_map[k]
            except KeyError:
                print('Exclusions not found for {:s}'.format(k))
                nexcl = 3
            #num = mols[k]
            if path !='':
                fname = '{:s}/{:s}.itp'.format(path,k)
            else:
                fname ='{:s}.itp'.format(k)
                
            lines = ['; generated by md_pipline library', '','']
            lines.extend(['','[ moleculetype ]', '; molname      nrexcl'])
            lines.append('{:5s}    {:1d}'.format(k,nexcl))
            lines.extend(['','[ atoms ]',';id atype resnr  resname atname cgnr'])
            s = 1
            mol_id = self.mol_ids[k==self.mol_names][0]
            chem = self.get_chem_per_molecule(mol_id)
            globloc = self.glob_id_to_loc[mol_id]
            #ids_mol = self.at_ids[mol_id==self.mol_ids]
            for j,a in enumerate(self.at_ids[mol_id == self.mol_ids]):
                code = self.atom_code[a]
                ty = self.at_types[a]
                mid = self.mol_ids[a]
                mn = self.mol_names[a]
                ch = self.atom_charge[a]
                ms = self.atom_mass[a]
                line = '{:5d}  {:5s}  {:5d}  {:5s}  {:5s}  {:5d}  {:8.6f}  {:8.6f}'.format(j+s,code,mid,mn,ty,j+s,ch,ms)
                lines.append(line)
            
            
            lines.extend(['','[ bonds ]',';ai   aj  func'])
            for c in chem['connectivity']:
                bc =self.connectivity[c]
                try:
                    func = self.ff.bondtypes[bc][1]
                except IndexError:
                    if Defaults:
                        func = ''
                    else:
                        raise Exception('could not find func for bondtype {:}'.format( bc))
                except KeyError:
                    if Defaults:
                        func = ''
                    else:
                        raise Exception('could not find bondtype {:}'.format( bc))
                i1,i2 = globloc[c[0]], globloc[c[1]]
                lines.append('{:5d}  {:5d}   {:}'.format(i1+s,i2+s,func))
            
            
            lines.extend(['','[ angles ]',';ai   aj   ak func'])
            for c in chem['angles']:
                ac = self.angles[c]
                try:
                    func = self.ff.angletypes[ac][1]
                except IndexError:
                    if Defaults:
                        func = ''
                    else:
                        raise Exception('could not find func for angletype {:}'.format( ac))
                except KeyError:
                    if Defaults:
                        func = ''
                    else:
                        raise Exception('could not find angletype {:}'.format( ac))
                i1,i2,i3 = globloc[c[0]], globloc[c[1]], globloc[c[2]]
                lines.append('{:5d}  {:5d}  {:5d}   {:}'.format(i1+s,i2+s,i3+s,func))
            
            
            lines.extend(['','[ dihedrals ]','; improper ai   aj   ak   al  func'])
            lines.extend(['','[ dihedrals ]','; proper ai   aj   ak   al  func'])
            for c in chem['dihedrals']:
                dc = self.dihedrals[c]
                try:
                    func = self.ff.dihedraltypes[dc][1]
                except IndexError:
                    if Defaults:
                        func = '3'
                    else:
                        raise Exception('could not find func for dihedraltype {:}'.format( dc))
                except KeyError:
                    if Defaults:
                        func = '3'
                    else:
                        raise Exception('could not find dihedraltype {:}'.format( dc))
    
                i1,i2,i3,i4 = globloc[c[0]], globloc[c[1]], globloc[c[2]], globloc[c[3]]
                lines.append('{:5d}  {:5d}  {:5d}  {:5d}    {:}'.format(i1+s,i2+s,i3+s,i4+s,func))
            lines.append('')
            
            lines.extend(['','[ pairs ]',';ai   aj  func'])
            for c in chem['pairs']:
                func = ''
                i1,i2 = globloc[c[0]], globloc[c[1]]
                lines.append('{:5d}  {:5d}   {:}'.format(i1+s,i2+s,func))
                        
            
            
            lines.extend(['','[ exclusions ]',';ai   aj  '])
            for c in chem['exclusions']:
                func = ''
                i1,i2 = globloc[c[0]], globloc[c[1]]
                lines.append('{:5d}  {:5d}   {:}'.format(i1+s,i2+s,func))
                
            lines.append('')
            for p in include_pairs:
                pairs1,pairs2 = self.find_vector_ids(p)
                lines.extend(['','[ pairs ]','; i  j '])
                for p1,p2 in zip(pairs1,pairs2):
                    lines.append('{:5d}   {:5d}    1'.format(p1+s,p2+s))
            for e in include_exclusions:
                pairs1,pairs2 = self.find_vector_ids(e)
                lines.extend(['','[ exclusions ]','; i  j '])
                for p1,p2 in zip(pairs1,pairs2):
                    lines.append('{:5d}   {:5d}  '.format(p1+s,p2+s))
                    
            if hasattr(self.ff,'posres'):
                data = getattr(self.ff,'posres')
                for d in data:
                    filt = getattr(self,d['by']) == d['val']
                    fmol = self.mol_names == k
                    f = np.logical_and(filt,fmol)
                    if f.any():
                        lines.extend(['','[ position_restraints ]','; i  j '])
                        for atid in self.at_ids[f]:
                            i = self.glob_id_to_loc[mol_id][atid]
                            #r = d['r']
                            k = d['k']
                            lines.append('{:d}  {:d}   {:8.5f}  {:8.5f}  {:8.5f}'.format(i+s,1,k,k,k))
            
            with open(fname,'w') as f:
                for line in lines:
                    f.write('{:s}\n'.format(line))
                f.close()
        return
    
    @property
    def system_name(self):
        name = []
        for m in self.sorted_mol_names:
            j = self.molecules[m]
            name.append(m+'_'+str(j))
            
        return '-'.join(name)
    @property
    def sorted_mol_names(self):
        sorted_names = []
        for m in self.mol_names:
            if m not in sorted_names:
                sorted_names.append(m)
        return sorted_names
    def write_topfile(self,fname,nbfunc='1',combrule='2',
                      genpairs='no',defaults=True,includes = [],
                      opls_convection=True,include_pairs=[],include_exclusions=[]):
        if opls_convection:
            fudgeLJ='0.5'
            fudgeQQ='0.5'
            combrule='3'
            genpairs='yes'
        else:
            fudgeLJ='1.0'
            fudgeQQ='1.0'
        if fname[-4:]!='.top': fname += '.top'
        lines = ['; generated by md_pipline library','','']
        
        
        lines.extend(['','','[ defaults ]',
                      '; nbfunc       comb-rule      gen-pairs      fudgeLJ   fudgeQQ'])
        lines.append('            '.join([str(nbfunc),str(combrule),str(genpairs),str(fudgeLJ),str(fudgeQQ)]))
        

        lines.extend(['','[ atomtypes ]'])
        lines.append('; type   name    mass    charge  ptype   sig     eps')
        def jt(k):
            return '  '.join([str(i) for i in k])
        
        for k,v in self.ff.atomtypes.items():
            vv = v[-5:]
            lines.append('{:10s} {:5s}  {:9.6f}  {:9.6f}  {:5s}  {:9.6f} {:9.6f}'.format(
                v[0],k,float(vv[0]),float(vv[1]),vv[2],float(vv[3]),float(vv[4])))
        
        lines.extend(['','[ bondtypes ]',';  i     j   func    b0    kb'])
        for k,v in self.ff.bondtypes.items():
            try:
                v[1]
                v[2]
                v[3]
            except IndexError:
                if not defaults:
                    raise Exception('I dont know the force field parameters of bondtype {}'.format(k))
            else:
                
                lines.append('{:5s}  {:5s}  {:1d}  {:9.6f}  {:9.6f}'.format(*k,int(v[1]),float(v[2]),float(v[3])))
        lines.extend(['','[ angletypes ]','; i    j    k func       th0         cth'])
        for k,v in self.ff.angletypes.items():
            try:
                v[1]
                v[2]
                v[3]
            except IndexError:
                if not defaults:
                    raise Exception('I dont know the force field parameters of angletype {}'.format(k))
            else:
                lines.append('{:5s}  {:5s} {:5s} {:1d}  {:9.6f}  {:9.6f}'.format(*k,int(v[1]),float(v[2]),float(v[3])))
        
        lines.extend(['','[ dihedraltypes ]','; i   j  k   l'])
        for k,v in self.ff.dihedraltypes.items():
            try:
                v[2]
            except IndexError:
                if not defaults:
                    raise Exception('I dont know the force field parameters of dihedraltype {}'.format(k))
            else:
                l = '   '.join(v[1:])
                lines.append('{:5s}  {:5s}  {:5s}  {:5s}  {:s}'.format(*k,l))
        
        lines.extend(['','',''])
        
        for k in includes:
            k1 = k + '.itp' if k[-4:] !='.itp' else k
            lines.append('#include "{:s}"'.format(k1))
        
        for k in self.molecules:
            lines.append('#include "{:s}.itp"'.format(k))
        if '/' in fname:
            path = '/'.join(fname.split('/')[:-1])
        else:
            path = ''
            
        self.write_itp(path,Defaults=defaults,
                       include_pairs=include_pairs,
                       include_exclusions=include_exclusions)
        
        lines.extend(['','',''])

        
        lines.extend(['','[ system ]',self.system_name])
        
        lines.extend(['','[ molecules ]',';molecule name number'])
        for k in self.sorted_mol_names:
            v = self.molecules[k]
            lines.append('   '.join([k,str(v)]) )
  
        with open(fname,'w') as f:
            for line in lines:
                f.write('{:s}\n'.format(line))
            f.close()
        return 
    def clean_dihedrals_from_topol_based_on_ff(self):
        nonex = self.nonexisting_types('dihedral')['inff']
        new_dihs = dict()
        for k,v in self.dihedrals.items():
            if v not in nonex:
                new_dihs[k] = v
        self.dihedrals=new_dihs
        return
    
    def nonexisting_types(self,which):
        listcheck = ['atom','bond','angle' ,'dihedral']
        if which not in listcheck:
            raise ValueError('give one of the following {}'.format(listcheck))
            
        attr = getattr(self.ff,which+'types')
        data = getattr(self, which+'_types')
        nonex_inff = [k for k in data if k not in attr]
        nonex = [k for k in attr if k not in data]
        re = {'inff': nonex_inff,'intopol':nonex}
        return re

class Analysis(Topology):
    '''
    The mother class. It's for simple polymer systems not confined ones
    '''
    def __init__(self,
                 topol_file, # gro/trr/ltop for now
                 connectivity_info, #itp #ltop
                 memory_demanding=False,
                 **kwargs):
        '''
        'connectivity_file': itp or list of itps. From the bonds it finds angles and dihedrals
        'gro_file': one frame gro file to read the topology (can be extended to other formats). It reads molecule name,molecule id, atom type
        memory_demanding: If True each time we loop over frames, these are readen from the disk and are not stored in memory
        types_from_itp: if different types excist in itp and gro then the itp ones are kept
        '''
        #t0 = perf_counter()
        self.topol_file = topol_file
        self.connectivity_info = connectivity_info
        self.connectivity_file = connectivity_info
        self.kwargs = kwargs
        self.memory_demanding = memory_demanding
        self.timeframes = dict()
        if 'types_from_itp' in kwargs:
            self.types_from_itp = kwargs['types_from_itp']
        else:
            self.types_from_itp = True

        if 'fftop' in kwargs:
            self.fftop = kwargs['fftop'] 

        if 'key_method' not in kwargs:
            self.key_method = 'get_timekey'
        else:
            self.key_method = 'get_' + kwargs['key_method'] + 'key'
        
        if 'round_dec' not in kwargs:
            self.round_dec = 7
        else:
            self.round_dec = kwargs['round_dec']
        False_defaults = ['refine_dihedrals','refine_angles','bytype']
        for nm in False_defaults:
            if nm not in kwargs:
                setattr(self,nm,False)
            else:
                if type(kwargs[nm]) is not bool:
                    raise ValueError('{:s} must be either True or False (Default)'.format(nm))
                setattr(self,nm,kwargs[nm])
        
        self.read_topology()
        self.topology_initialization()
 
        self.timeframes = dict() # we will store the coordinates,box,step and time here
        
        #tf = perf_counter()-t0
        #ass.print_time(tf,inspect.currentframe().f_code.co_name)
        
        return 
    

    
    def topology_initialization(self,reinit=False):
        '''
        Finds some essential information for the system and stores it in self
        like angles,dihedrals
        Also makes some data manipulation that might be used later
        '''
        t0 = perf_counter()
        ## We want masses into numpy array
        if reinit:
            self.connectivity = dict()
            self.neibs = dict()
            self.angles = dict()
            self.dihedrals = dict()
            self.find_locGlob()
           

        #Now we want to find the connectivity,angles and dihedrals
        
        self.find_neibs()
        self.find_angles()
        self.find_dihedrals()
        #Find the ids (numpy compatible) of each type and store them
        
        

        self.dict_to_sorted_numpy('connectivity') #necessary to unwrap the coords efficiently
        
        self.find_args_per_residue(np.ones(self.mol_ids.shape[0],dtype=bool),'molecule_args')
        self.find_args_per_residue(np.ones(self.mol_ids.shape[0],dtype=bool),'chain_args')
        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name)
            
        return
 
    
    def box_variance(self):
        t0 = perf_counter()
        box_var = np.zeros(3)
        box_mean = self.box_mean()
        args = (box_var,box_mean**2)
        nframes = self.loop_trajectory('box_var', args)
        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name)
        return box_var/nframes
    
    def frame_closer_tobox(self,target_box):
        mind = 10**9
        with self.traj_opener(*self.traj_opener_args) as ofile:
            t0 = perf_counter()
            nframes=0
            while(self.read_from_disk_or_mem(ofile, nframes)):
                box=self.timeframes[nframes]['boxsize']
                d = np.dot(target_box-box,target_box -box)**0.5
                if d < mind:
                    mind = d
                    frame_min = nframes
                if self.memory_demanding:
                    del self.timeframes[nframes]
                nframes+=1
        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name)
        return frame_min
    def loop_trajectory(self,fun,args):
        '''
        It's a Wrapper for looping over trajectories
        It takes a "core" function and it's arguments
        Then it loops over all trajectory calling the same
        function with it's arguments

        Parameters
        ----------
        fun : Function
        args : tuple of function arguments

        Returns
        -------
        nframes : the number of frames it had looped
        '''
        
        funtocall = getattr(coreFunctions,fun)
        
        if len(self.timeframes) == 0 or self.memory_demanding:
            with self.traj_opener(*self.traj_opener_args) as ofile:
                nframes=0
                while(self.read_from_disk_or_mem(ofile,nframes)):
                    self.current_frame = nframes
                    funtocall(self,*args)      
                    if self.memory_demanding:
                        del self.timeframes[nframes]
                    nframes+=1
        else:
            nframes = self.loop_timeframes(funtocall,args)
        del self.current_frame
        return nframes
    
    def loop_timeframes(self,funtocall,args):
        
        for frame in self.timeframes:
            self.current_frame = frame
            funtocall(self,*args)
        nframes = len(self.timeframes)
        return nframes
    
    @property
    def first_frame(self):
        if not self.memory_demanding:
            return list(self.timeframes.keys())[0]
        else:
            return 0
    
    def cut_timeframes(self,num_start=None,num_end=None):
        '''
        Used to cut the trajectory

        Parameters
        ----------
        num_start : int, frame the trajectory starts
        num_end :  int, frame the trajectory ends

        Returns
        -------
        None.

        '''
        if num_start is None and num_end is None:
            raise Exception('Give either a number to cut from the start or from the end for the timeframes dictionary')
        if num_start is not None:
            i1 = num_start
        else:
            i1 =0
        if num_end is not None:
            i2 = num_end
        else:
            i2 = len(self.timeframes)
        new_dict = ass.dict_slice(self.timeframes,i1,i2)
        if len(new_dict) ==0:
            raise Exception('Oh dear you have cut all your timeframes from memory')
            
        self.timeframes = new_dict
        return 
    
    def calc_atomic_coordination(self,maxdist,type1,type2):
        
        t0 = perf_counter()
        def find_args_of_type(types):
            if not ass.iterable(types):
                types = [types]
            ids = np.array([],dtype=int)
            for ty in types:
                a1 = np.where([self.at_types==ty])[1]
                ids = np.concatenate((ids,a1))
            ids = np.unique(ids)
            return ids
        
        args1 = find_args_of_type(type1)
        args2 = find_args_of_type(type2)
        
        
        coordination = np.zeros(args1.shape[0])
        args = (maxdist,args1,args2,coordination)
        
        nframes = self.loop_trajectory('atomic_coordination',args)
        
        coordination/=nframes
        def tkey(ty):
            if ass.iterable(ty):
                k = '_'.join(ty)
            else:
                k = ty
            return k
        
        key ='-'.join([tkey(type1),tkey(type2)])
        
        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        return {key:coordination}
    
    def find_q(self,dmin,dq,dmax,direction):
        qmag = np.arange(dmin,dmax+dq,dq)
       
        if direction is None:
            q = np.array([np.ones(3)*qm**(1/3) for qm in qmag] )
        else:
            if len(direction)!=3:
                raise Exception('Wrong direction vector')
            d = np.array([di for di in direction],dtype=float)
            dm = np.sum(d*d)**0.5
            d/=dm
            q = np.array([d*qm for qm in qmag])
                    
        return q
    
    def calc_Sq(self,qmin,dq,qmax,direction=None,ids=None):
        t0 = perf_counter()
        if isinstance(self,Analysis_Confined):
            ids = self.polymer_ids
        q = self.find_q(qmin,dq,qmax,direction)
        Sq = np.zeros(q.shape[0],dtype=float)
        args =(q,Sq,ids)
        nframes = self.loop_trajectory('Sq',args)
        Sq/=nframes
        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        return {'q':q,'Sq':1+Sq}
    
    def calc_Sq_byInverseGr(self,dr,dmax,dq,qmax,
                            qmin=0,
                            ids=None,direction=None,intra=False,inter=False):
        def Fourier3D(q,r,gr,rho):
            
            Sq = np.zeros(q.shape[0])
            for i in range(q.shape[0]):
                Sq[i] = simpson((gr-1)*r*np.sin(q[i]*r)/q[i],r) 
            return 1+4*np.pi*rho*Sq
        def Fourier2D(q,r,gr,rho):
            
            Sq = np.zeros(q.shape[0])
            for i in range(q.shape[0]):
                Sq[i] = simpson((gr-1)*np.sin(q[i]*r)/q[i],r) 
            return 1+np.pi*rho*Sq
        def Fourier1D(q,r,gr,rho):
            
            Sq = np.zeros(q.shape[0])
            for i in range(q.shape[0]):
                Sq[i] = simpson((gr-1)*np.sin(q[i]*r)/(q[i]*r),r) 
            return 1+rho*Sq
        
        res = self.calc_pair_distribution(binl,dmax,None,None,intra,inter) 
        
        q =np.arange(qmin+dq,qmax+dq,dq)
        d = res['d']
        g = res['gr']
        rho = res['rho']
        box = self.box_mean()
        if direction is None or direction =='':
            Sq = Fourier3D(q,d,g,rho)
        elif direction =='xy':
            Sq = Fourier2D(q,d,g,rho*box[2])
        elif direction =='z':
            Sq = Fourier1D(q,d,g,rho*box[1]*box[0])
        
        res['Sq'] = Sq
        res['q'] = q
             
        return res
    
    def calc_internal_distance(self,n,filters=dict()):
        ids1,ids2 = self.find_vector_ids(n)
        vect,filt_t = self.vects_per_t(ids1,ids2,filters=filters)
        dists_t =  {t:np.sum(v*v,axis=1)**0.5 for t,v in vect.items()}
        return dists_t,filt_t

    def list_molecule_ids(self,mol):
        mf = self.mol_names == mol
        ids = self.at_ids[mf]
        this_mol_ids = self.mol_ids[mf]
        molecule_ids = []

        for mid in np.unique(this_mol_ids):
            molecule_ids.append( ids[ this_mol_ids == mid ] )
        return molecule_ids

    def calc_cluster_size_t(self,mol,dcut,method='min',mol2=None):
        t0 = perf_counter()
        available_methods = ['com','min']
        if method not in available_methods:
            raise ValueError('Uknown method {:s} --> choose from {:}'.format(method,available_methods))
        ty_trick = self.at_types[self.mol_names == mol][0]
        topol_vector = ' '.join([ty_trick,ty_trick])
        ids1,ids2 = self.find_vector_ids(topol_vector)
        
        segmental_ids = self.find_segmental_ids(ids1, ids2,  ('A_TRICK_WHATEVER','BACD_TRICK_WHATEVER'))
        
        maptomols = dict()
        for i,seg in enumerate(segmental_ids):
            un = np.unique(self.mol_ids[seg])
            if un.shape[0] != 1:
                raise Exception('Segmental ids do not give a unique molecule value --> {}'.format(un))
            maptomols[i] = un[0]
        
        distribution = dict()
        clusters = dict()
        args = (segmental_ids,dcut,distribution,clusters)

        nframes = self.loop_trajectory('cluster_size'+'_'+method,args)
        
        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        
        data = dict()
        
        data['debugging'] = {'maptomols':maptomols}
        
        data['clusters'] = clusters
        data['time'] = np.array(list(distribution.keys()))/1000 
        data['cluster-mean'] = np.array([np.mean(v) for v in distribution.values()])
        data['cluster-std'] = np.array([np.std(v) for v in distribution.values()])
        data['cluster-maxsize'] = np.array([np.max(v) for v in distribution.values()])



        sizes = []
        for v in distribution.values():
            sizes.extend(v)
        sizes = np.array(sizes)
        data['sizes'] = sizes
        
        maxsize = sizes.max()
        
        data['nc'] = np.arange(1,maxsize+1,1,dtype=int)
        
        counts = np.array([np.count_nonzero(sizes == j) for j in range(1,maxsize+1) ])
        
        data['counts'] = counts
        data['counts/n'] = counts/len(segmental_ids)
        data['clustersize-distribution'] = counts/np.sum(counts)
        data['molecule-distribution'] = data['nc']*data['clustersize-distribution']/np.sum(data['nc']*data['clustersize-distribution'])
        
        data['probclust-time'] = dict()
        data['probmol-time'] = dict()
        nc = data['nc'] 
        for k,v in distribution.items():
            v = np.array([np.count_nonzero(np.array(v) == j) for j in nc ])
            data['probclust-time'][k] = v
            data['probmol-time'][k] = nc*v/np.sum(v*nc)
        
        data.update( {'probmol-{:d}-time'.format(j) : ass.numpy_values(data['probmol-time'])[:,i] for i,j in enumerate(nc)} )
        
        return data
    
    def calc_segmental_pair_distribution(self,binl,dmax,topol_vector,segbond,far_region=0.8):
        t0 = perf_counter()
        
        ids1,ids2 = self.find_vector_ids(topol_vector)
        
        segmental_ids = self.find_segmental_ids(ids1, ids2, segbond)
        
        bins = np.arange(0,dmax+binl,binl)
        gofr = np.zeros(bins.shape[0]-1,dtype=float)
        args = (bins,segmental_ids,gofr)
        
        nframes = self.loop_trajectory('gofr_segments', args)
        
        gofr/=nframes
        
        
        
        n1 = len(segmental_ids)/2 # because the segments are found with their selfs
        n = int((len(segmental_ids)-1)*n1)
        
        pair_distribution = self.normalize_gofr(bins,gofr,n,n1,far_region,dmax) 
        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        return pair_distribution 
    
    def get_ids_based_on_atomtype(self,ty):
        if ty is None:
            ids = self.at_ids.copy()
        elif ass.iterable(ty):
            f =  self.at_types == ty[0]
            for i in range(1,len(ty)):
                f = np.logical_or(f, self.at_types == ty[i])
                ids = self.at_ids[f].copy()
        elif type(ty) is str:
            ids = self.at_ids[self.at_types==ty].copy()
        else:
            raise ValueError('uknown way to find ids')
        return ids
    
    def find_gr_pairs(self,ty1,ty2,intra=False,inter=False):
        if inter and intra:
            raise ValueError('Cannot demand both intra and inter pairs at the same time')
        
        ids1 = self.get_ids_based_on_atomtype(ty1)
        ids2 = self.get_ids_based_on_atomtype(ty2)
        
        pairs = [[i,j] for i in ids1 for j in ids2 if i!=j ]
        npairs = len(pairs)
        pairs = np.array(pairs)
        
        if intra:
            filt = np.ones(npairs,dtype=bool)
            for i,j in enumerate(pairs):
                
                if self.mol_ids[j[0]] == self.mol_ids[j[1]]:
                    filt[i] = False
            
            pairs = pairs[filt]
            
        if inter:
            filt = np.ones(npairs,dtype=bool)
            for i,j in enumerate(pairs):
                
                if self.mol_ids[j[0]] != self.mol_ids[j[1]]:
                    filt[i] = False
            
            pairs = pairs[filt]
            
        return pairs[:,0], pairs[:,1], len(ids1), pairs.shape[0]
    
    def calc_pair_distribution(self,binl,dmax,type1=None,type2=None,intra=False,inter=False,
                               far_region=0.8):
        '''
        Used to calculate pair distribution functions between two atom types
        Could be the same type. If one type is None then the distribution is 
        between the the other type and all atom types. If both is none then
        the distribution is between all atoms.

        Parameters
        ----------
        binl : FLOAT
            Binning legth.
        dmax : float
            maximum distance to calculate.
        type1 : string 
            . The default is None.
        type2 : string
            . The default is None.
        density : string
            DESCRIPTION. The default is ''.
            'number' is to return number density [ 1/nm^3 ]
            'probability' returns probability density
        normalize : The default is True. 
            For density=="number" it normalizes with the mean of far values (by default with the mean between distances (0.75dmax,dmax))
            for density=="probability" it normalizes with gofr.sum()
            for density=="coordination" it does nothing
            for density=="" (default) it normalizes by gmax
        Raises
        ------
        NotImplementedError
            DESCRIPTION.

        Returns
        -------
        pair_distribution : dictionary 
            having the distance and the distribution
    

        '''
        t0 = perf_counter()   
            
        bins = np.arange(0,dmax+binl,binl)
        gofr = np.zeros(bins.shape[0]-1,dtype=float)
        
        ids1,ids2, n1 ,n = self.find_gr_pairs(type1,type2,intra,inter)
        
        args = (ids1,ids2,bins,gofr)
        
        nframes = self.loop_trajectory('gofr_pairs', args)
        

        gofr/=nframes

        pair_distribution = self.normalize_gofr(bins,gofr,n,n1,far_region,dmax)
        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        return pair_distribution
    
    @staticmethod
    def normalize_gofr(bins,gofr,n,n1,far_region,dmax):
        pair_distribution = dict()
        cb = center_of_bins(bins)
        pair_distribution['d'] = cb
        npairs = gofr.copy()
        pair_distribution['npairs'] = npairs
        
        vshell = (4*np.pi/3)*(bins[1:]**3-bins[:-1]**3)
        
        pairdens = npairs/vshell
        pair_distribution['v'] = (4*np.pi/3)*cb**3
        pair_distribution['pair density'] = pairdens
        
        numdens = pairdens/n1
        
        pair_distribution['number density'] = numdens
        pair_distribution['probablilty number density'] = numdens/numdens.sum()
        
        pair_distribution['coordination'] = npairs/n

        b =(far_region*dmax,dmax)
        f = np.logical_and(b[0]<cb,cb<=b[1])
        far_rho = numdens[f].mean()
        far_rho_std = numdens[f].std()
        
        pair_distribution['far_rho'] = far_rho
        pair_distribution['far_rho_std'] = far_rho_std
        
        norm_numdens= numdens/far_rho
        
        pair_distribution['gr'] = norm_numdens
        return pair_distribution
    
    def calc_size(self):
        size = np.zeros(3)
        args = (size,)
        nframes = self.loop_trajectory('minmax_size',args)
        return size/nframes
    
    
    
    
    def init_xt(self,xt,dtype=float):
        '''
        takes a dictionary of keys times and values some array
        and converts it to numby

        Parameters
        ----------
        xt : dictionary
            keys: times
            values: array
        dtype : type
            type of the array.

        Returns
        -------
        x_nump : numpy array of type dtype

        '''
        x0 = xt[list(xt.keys())[0]]
        nfr = len(xt)
        shape = (nfr,*x0.shape)
        x_nump = np.empty(shape,dtype=dtype)
        
        for i,t in enumerate(xt.keys()):
           x_nump[i] = xt[t]
        
        return  x_nump
    
    def init_prop(self,xt):
        '''
        Allocates the arrays to fill the dynamical or kinetic property
        that will be calculated

        Parameters
        ----------
        xt : dictionary
            keys: times
            values: array
            It is the input.

        Returns
        -------
        Prop_nump : float array
        nv : float array, is used as a counter
        '''
        nfr = len(xt)
        Prop_nump = np.zeros(nfr,dtype=float)
        nv = np.zeros(nfr,dtype=float)
        return Prop_nump,nv
    
    def vects_per_t(self,ids1,ids2,
                         filters={}):
        '''
        Takes two int arrays  of atom ids and finds 
        the vectors per time between them

        Parameters
        ----------
        ids1 : int array of atom ids 1
        ids2 : int array of atom ids 2
        filters : dictionary of filters. See filter documentation

        Returns
        -------
        vec_t : dictionary of keys the time and values the vectors in numpy array of shape (n,3)
        filt_per_t : dictionary of keys the filt name and values
            dictionary of keys the times and boolian arrays

        '''
        t0 = perf_counter()
        vec_t = dict()
        filt_per_t = dict()
        
        args = (ids1,ids2,filters,vec_t,filt_per_t)
        
        nframes = self.loop_trajectory('vects_t', args)
                
        filt_per_t = ass.rearrange_dict_keys(filt_per_t)
        tf = perf_counter()-t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        return vec_t,filt_per_t
    
    def calc_segmental_vectors_t(self,topol_vector,filters={}):
        '''
        Calculates 1-2,1-3 or 1-4 segmental vectors as a function of time
        Parameters
        ----------
        topol_vector : list of atom types, or int for e.g. 1-2,1-3,1-4 vectors
            Used to find e.g. the segmental vector ids
        filters : dictionary.
            see filters documentation

        Returns
        -------
        segvec_t : dictionary
            keysf times
            values: float array (nsegments,3)
        filt_per_t : dictionary
            see filter documentation
        '''
        ids1,ids2 = self.find_vector_ids(topol_vector)
        
        t0 = perf_counter()
        
        segvec_t, filt_per_t = self.vects_per_t(ids1, ids2, filters)
        
        tf = perf_counter()-t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name)
        return segvec_t,filt_per_t
    
    
    def get_Dynamics_inner_kernel_functions(self,prop,filt_option,weights_t,**kwargs):
        '''
        gets the options to set up the prober functions for
        the Dynamics kernel

        Parameters
        ----------
        prop : string
            P1: p1 dynamics
            P2: p2 dynamics
            MSD: mean square displacement
            
        filt_option : string
            simple: consider the population based on time origin
            strict: consider the population based on time origin and  time t
            change: consider the population based on time origin and changed on time t
        weights_t : dictionary
            keys: time
            values: float array (total population,)
        Returns
        -------
        funcs : the main kernel function, 
                the function which sets the arguments
                the inner function

        '''
        mapper = {'p1':'costh_kernel',
                  'p2':'cos2th_kernel',
                  'msd':'norm_square_kernel',
                  'scalar':'mult_kernel',
                  'fs':'Fs_kernel',
                  }
        prop = prop.lower()
        if prop in mapper:
            inner_func_name = mapper[prop.lower()] 
        else:
            inner_func_name = prop
            
        name = 'dynprop'
        af_name = 'get'
        
        if filt_option is not None:
            ps = '_{:s}'.format(filt_option)
            name += ps 
            af_name+= ps
        if weights_t is not None:
            ps = '_weighted'
            name += ps
            af_name += ps
        
        func_name = '{:s}__kernel'.format(name)
        args_func_name = '{:s}__args'.format(af_name)
        
        logger.info(' func name : "{:s}" \n argsFunc name : "{:s}" \n innerFunc name : "{:s}" '.format(func_name,args_func_name,inner_func_name))
        
        funcs = (globals()[func_name],
                 globals()[args_func_name],
                 globals()[inner_func_name])
        
        return funcs
    
    def set_partial_charge(self):
        if not hasattr(self,'partial_charge'):
            charge = np.empty((self.at_types.shape[0],1),dtype=float)
            for i,ty in enumerate(self.at_types):
                charge[i] = maps.charge_map[ty]
            self.partial_charge = charge
        return
    
    def segment_ids_per_chain(self,segmental_ids):
        seg0 = segmental_ids[:,0]
        segch = {j : np.isin(seg0,chargs)
                 for j,chargs in self.chain_args.items()}           
        return segch

    @staticmethod
    def id_neibs(id0,neibs):
        '''
        neibs should be a dictionary with integer keys and set values and id0 integer
        '''
        setids = neibs[id0]
        setids_old = set()
        while( len(setids) != len(setids_old) ):
            setids_old = setids
            for i in setids.copy():
                setids = setids | neibs[i]
        return np.array(list(setids))

    def find_general_ids(self, topol_vector, segbond=None):
        
        if segbond is not None:
            ids1,ids2 = self.find_vector_ids(topol_vector)
        
            segmental_ids = self.find_segmental_ids(ids1, ids2, segbond)
        else:
            if topol_vector in self.mol_names:
                segmental_ids = self.list_molecule_ids(topol_vector)
            else:
                raise ValueError(" topol_vector = {} is not eligible value ".format(topol_vector))

        return segmental_ids
    
    def find_segmental_ids(self,ids1,ids2,segbond):
        '''
        Used to find the segmental ids
        Works for linear polymers at the moment
        
        Parameters
        ----------
        ids1 : int array
            atom ids 1
        ids2 : int array
            atom ids 2
        segbond : tuple of two stings 
            the type of bond that connects the segments 
        Returns
        -------
        seg_ids_numpy : int array (nsegments,nids_per_segment)
            an array with the ids of each segment

        '''
        
        t0 = perf_counter()
        
        if ass.iterable(segbond[0]):
            conn_excluded = {k:v for k,v in self.connectivity.items()
                        if v in segbond
                            }
        else:
            conn_excluded = {k:v for k,v in self.connectivity.items()
                        if v != segbond
                            }
            
        neibs_excluded = dict()
        for k,v in self.neibs.items():
            s = v.copy()
            for i in s.copy():
                if not ( (k,i) in conn_excluded or (i,k) in conn_excluded):
                    s.remove(i)
            neibs_excluded[k] = s
        
        seg_ids = []
        for j in ids1:
            seg_ids.append( self.id_neibs(j,neibs_excluded) )
        
        useg_ids = []
        for j, seg in enumerate(seg_ids):
            inseg = False
            for i, preseg in enumerate(seg_ids[:j]):
                if (seg == preseg).all():
                    inseg=True
            if not inseg:
                useg_ids.append(seg)
        seg_ids = useg_ids
        self.segmental_args = seg_ids
        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name)
        return  seg_ids
    
    def calc_chain_dipole_moment_t(self,filters=dict(),**kwargs,):
        '''
        Chain dipole moment as a function of t

        Parameters
        ----------
        filters : dictionary
            see filter documentation.

        Returns
        -------
        dipoles_t : dictionary
            keys: times
            values: float array (nchains,3)
        filters_t : dictionary
            see filter documentation

        '''
        t0 = perf_counter()
        
        filters = {'chain_'+k : v for k,v in filters.items()}
        
        
        
        if 'option' in kwargs:
            option = kwargs['option']
        else:
            option =''
        dipoles_t = dict()
        filters_t = dict()
        if option =='contour':
            ids1,ids2 = self.find_vector_ids(kwargs['monomer'])
            segmental_ids = self.find_segmental_ids(ids1, ids2, kwargs['segbond'])
            segch = self.segment_ids_per_chain(segmental_ids)
            args = (filters,ids1,ids2,segmental_ids,
                    segch,dipoles_t,filters_t)
            ext ='__contour'
        elif option=='':
            ext =''
            args = (filters,dipoles_t,filters_t)
        elif option=='endproj':
            ext ='__endproj'
            args = (filters,dipoles_t,filters_t)
        elif option=='proj':
            ext ='__proj'
            projvec = np.array([x for x in kwargs['projvec']])
            args = (filters,projvec,dipoles_t,filters_t)
        nframes = self.loop_trajectory('chain_dipole_moment'+ext,args)
        
        filters_t = ass.rearrange_dict_keys(filters_t)
        
        tf = perf_counter() - t0
        
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        return dipoles_t,filters_t
    

    
    def calc_total_dipole_moment_t(self,q=None):
        '''
        Calculates total dipole moment vector
        Parameters
        -------
        dipoles_t : dictionary
            keys: times
            values: float array (nvector,3)

        '''
        t0 = perf_counter()

        
        self.set_partial_charge()
        
        dipoles_t = dict()
        if q is not None:
            q = np.array(q) ; q = q/np.sum(q*q)**0.5
            q = q.reshape(1,3)
        args = (dipoles_t,q)
        
        nframes = self.loop_trajectory('total_dipole_moment',args)
        
        tf = perf_counter() - t0
        
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        return dipoles_t 
    
    def calc_segmental_dipole_moment_t(self,topol_vector,
                                       segbond,filters=dict()):
        '''
        Calculates dipole moment vectors as a function of time


        Parameters
        ----------
        topol_vector : list of 4 strings
            finds the segment edge ids
        segbond : tuple of 2 strings
            the bond type connecting the segments
        filters : dictionary
            see filter documentation
        Returns
        -------
        dipoles_t : dictionary
            keys: times
            values: float array (nvector,3)
        filters_t : dictionary
            see filter documentatin

        '''
        t0 = perf_counter()
        
        #self.set_partial_charge()
            
        ids1,ids2 = self.find_vector_ids(topol_vector)
        
        segmental_ids = self.find_segmental_ids(ids1, ids2, segbond)
        
        dipoles_t = dict()
        filters_t = dict()
        
        args = (filters,ids1,ids2,segmental_ids,dipoles_t,filters_t)
        
        nframes = self.loop_trajectory('segmental_dipole_moment',args)
        
        filters_t = ass.rearrange_dict_keys(filters_t)
        
        tf = perf_counter() - t0
        
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        return dipoles_t,filters_t
    
    def calc_segmental_dipole_moment_correlation(self,topol_vector,
                                       segbond,filters=dict()):
        '''
        Computes static correlation between segments
        
         Parameters
        ----------
        topol_vector : list of 4 strings
            finds the segment edge ids
        segbond : tuple of 2 strings
            the bond type connecting the segments
        filters : dictionary
            see filter documentation

        Returns
        -------
        correlations : dictionary
            keys   : depent on filter 
            values : dictionary
                    keys: number of bonds connecting the segments
                    values: static correlation

        '''
        t0 = perf_counter()
        filters.update({'system':None})
        dipoles_t,filters_t = self.calc_segmental_dipole_moment_t(topol_vector,
                                       segbond,filters)
                   
        ids1,ids2 = self.find_vector_ids(topol_vector)
        bond_distmatrix = self.find_bond_distance_matrix(ids1)
        
        unb = np.unique(bond_distmatrix)
        b0 = dict()
        b1 = dict()
        for k in unb:
            if k>0:
                b =  np.nonzero(bond_distmatrix ==k)
                b0[k] = b[0]
                b1[k] = b[1]

        correlations ={filt:{k:[] for k in unb if k>0} for filt in filters_t}
        
        args = (dipoles_t,filters_t,b0,b1,correlations)
        
        nframes = self.loop_trajectory('vector_correlations', args)
       
        corrs = {kf:{'nc':[],'corr':[],'corr(std)':[]} for kf in correlations}
        for kf in filters_t:
            for k in correlations[kf]:
                c = correlations[kf][k]
                corrs[kf]['nc'].append(k)
                corrs[kf]['corr'].append(np.mean(c))
                corrs[kf]['corr(std)'].append(np.std(c))
            corrs[kf] = {key:np.array(corrs[kf][key]) for key in corrs[kf]}
                
        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
    
        return corrs 
    
    def stress_per_t(self,filters=dict()):
        '''
        Takes two int arrays  of atom ids and finds 
        the vectors per time between them

        Parameters
        ----------
        ids1 : int array of atom ids 1
        ids2 : int array of atom ids 2
        filters : dictionary of filters. See filter documentation

        Returns
        -------
        vec_t : dictionary of keys the time and values the vectors in numpy array of shape (n,3)
        filt_per_t : dictionary of keys the filt name and values
            dictionary of keys the times and boolian arrays

        '''
        t0 = perf_counter()
        atomstress_t = dict()
        filt_per_t = dict()
        
        args = (filters,atomstress_t,filt_per_t)
        
        nframes = self.loop_trajectory('stress_per_atom_t', args)
        
        filt_per_t = ass.rearrange_dict_keys(filt_per_t)
        
        tf = perf_counter()-t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        return atomstress_t,filt_per_t

    def chains_CM(self,coords):
        '''
        Given the coordinates finds the chains center of mass

        Parameters
        ----------
        coords : system coordinates

        Returns
        -------
        chain_cm : float array (nchains,3)

        '''
        chain_arg_keys = self.chain_args.keys()
        
        chain_cm = np.empty((len(chain_arg_keys),3),dtype=float)
        
        for i,args in enumerate(self.chain_args.values()):
            chain_cm[i] = CM(coords[args],self.atom_mass[args])
        
        return chain_cm

    def segs_CM(self,coords,segids):
        '''
        
        Given the coordinates and the segmental ids finds the segmental center of mass
        
        Parameters
        ----------
        coords : system coordinates 
        segids : int array (nsegments,nids_per_segment)

        Returns
        -------
        segcm : float array (nsegments,3)

        '''
        n = len(segids)
        segcm = np.empty((n,3),dtype=float)
        mass = self.atom_mass
        box = self.get_box(self.current_frame)
        for i,si in enumerate(segids):
            csi = coords[si]
            c0ref = csi[0]
            cunwrap = minimum_image_relative_coords(csi-c0ref,box) + c0ref
            segcm[i] = CM(cunwrap,mass[si])
        #numba_CM(coords,segids,self.atom_mass,segcm)
        return segcm
            

    def calc_dihedrals_t(self,dih_type,filters={'all':None}):
        '''
        Dihedrals per time

        Parameters
        ----------
        dih_type : list of 4 strings
            dihedral type
        filters : dictionary
            see filters documentation.

        Returns
        -------
        dihedrals_t : dictionary 
        keys: times
        values: float array (ndihedrals,)
        filt_per_t : see filters documentation

        '''
        t0 = perf_counter()
        
        dihedrals_t = dict()
        filt_per_t = dict()
       
        dih_ids = self.dihedrals_per_type[dih_type] #array (ndihs,4)
        ids1 = dih_ids[:,0] ; ids2 = dih_ids[:,3] 

        args = (dih_ids,ids1,ids2,filters,dihedrals_t,filt_per_t)
        
        nframes = self.loop_trajectory('dihedrals_t', args)
        
        filt_per_t = ass.rearrange_dict_keys(filt_per_t)
        
        tf = perf_counter()-t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        
        return dihedrals_t, filt_per_t
    


    def calc_Ree_t(self,filters={}):
        '''
        End to End vectors as a function of time

        Parameters
        ----------
        filters : dictionary
            see filter documentation

        Returns
        -------
        Ree_t : dictionary
            keys times
            values: float arrays (nchains,3)
        filt_per_t : dictionary
            see filter documentation

        '''
        
        t0 = perf_counter()
        #filters = {'chain_'+k: v for k,v in filters.items()}
        chain_is = []
        chain_ie = []
        for j,ch_args in self.chain_args.items():
            chain_is.append(ch_args[0])
            chain_ie.append(ch_args[-1])
        ids1 = np.array(chain_is)
        ids2 = np.array(chain_ie)
      
        Ree_t,filt_per_t = self.vects_per_t(ids1,ids2,filters)
        
        tf = perf_counter()-t0
        
        ass.print_time(tf,inspect.currentframe().f_code.co_name)
        #return {t:v[tot_filt] for t,v in dihedrals_t.items()}
        return Ree_t, filt_per_t
    
    def calc_Rg(self,option='__permol'):
        t0 = perf_counter()
        
        if option=='':
            Rg = {'Rg':0.0,'Rgstd':0.0}
        elif 'permol' in option:
            option = '__permol'
            Rg = dict()
            for m in self.molecules:
                Rg.update({m+'_Rg':0.0,m+'_Rgstd':0.0})
        else:
            raise ValueError('option "{:s}" is not specified'.format(option))
        
        args = (Rg,)
        
        nframes = self.loop_trajectory('Rg'+option,args)
    
        Rg = {k1:v/nframes for k1,v in Rg.items()}
        
        tf= perf_counter() -t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        return Rg

    def calc_segmolCM_t(self,topol_vector,segbond=None,filters=dict(),option=''):
        t0 = perf_counter()
        
        segcm_t = dict()
        filt_per_t = dict()
        
        segmol_ids = self.find_general_ids(topol_vector,segbond)

        args = (segmol_ids, filters, segcm_t, filt_per_t)

        nframes = self.loop_trajectory('segmolCM_t'+option, args)
      
        filt_per_t = ass.rearrange_dict_keys(filt_per_t)
        
        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        
        return segcm_t,filt_per_t
    
    def calc_chainCM_t(self,filters=dict(),option=''):
        '''
        Computes chains center of mass as a function of time
        Parameters
        ----------
        filters : dictionary
            see filter documentation
        option : string
            for feature use
        coord_type : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        vec_t : dictionary
            keys: times
            values: float array (nchains,3)
        filt_per_t : dictionary
            see filter documentation

        '''
        t0 = perf_counter()
        filters = {'chain_'+k: v for k,v in filters.items()} #Need to modify when considering chains
        
        vec_t = dict()
        filt_per_t = dict()
        
        args = (filters,vec_t,filt_per_t)
        
        nframes = self.loop_trajectory('chainCM_t'+option, args)
      
        filt_per_t = ass.rearrange_dict_keys(filt_per_t)
        
        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        
        return vec_t,filt_per_t

    def calc_coords_t(self,ids,filters=dict()):
        t0 = perf_counter()
        
        c_t = dict()
        filt_t = dict()
        args = (filters,ids,c_t,filt_t)
        nframes = self.loop_trajectory('coords_t', args)
        filt_t = ass.rearrange_dict_keys(filt_t)
        
        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        return c_t,filt_t
    
    def calc_segCM_t(self,topol_vector,segbond,
                     filters={'all':None}, option=''):
        '''
        Returns the segment center of mass as a function of time

        Parameters
        ----------
        topol_vector : list of 4 strings
            finds the segment edge ids
        segbond : tuple of 2 strings
            the bond type connecting the segments
        filters : dictionary
            see filter documentation
        option: string
            for feature use

        Returns
        -------
        vec_t : dictionary
            keys: times
            values: float array (nsegments,3)
        filt_per_t : dictionary
            see filter documentation

        '''
        t0 = perf_counter()
     
        ids1,ids2 = self.find_vector_ids(topol_vector)
        
        segmental_ids = self.find_segmental_ids(ids1, ids2, segbond)
        
        vec_t = dict()
        filt_per_t = dict()
        
        args = (filters,ids1,ids2,segmental_ids,vec_t,filt_per_t)
        
        nframes = self.loop_trajectory('segCM_t'+option, args)
      
        filt_per_t = ass.rearrange_dict_keys(filt_per_t)
        
        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        
        return vec_t,filt_per_t
    
    def Dynamics(self,prop,xt,filt_t=None,weights_t=None,
                 filt_option='simple', block_average=False,
                 multy_origin=True,every=1,q=None):
        '''
        

        Parameters
        ----------
        prop : string
            P1: p1 dynamics
            P2: p2 dynamics
            MSD: mean square displacement
        xt : dictionary
            keys: times
            values: vector array (total population,nd) where nd =1,2,3
            It is the input.

        filt_t : dictionary
            keys: times
            values: boolean array
        weights_t : dictionary
            keys: time
            values: float array (total population,)
        filt_option : string
            simple: consider the population based on time origin
            strict: consider the population based on time origin and  time t
            change: consider the population based on time origin and changed on time t

        block_average : bool
            Ways of averaginf
            True  <  < >|population  >|time origins
            False < >|population,time origins
        multy_origin: bool
            if True use multiple origins in averaging (assumes that all origins are equivalent)
        Returns
        -------
        dynamical_property : dictionary
            keys: time
            values: float

        '''
        tinit = perf_counter()
        
        Prop_nump,nv = self.init_prop(xt)
        x_nump = self.init_xt(xt)
        
        

        
        if filt_t is not None:
            f_nump = self.init_xt(filt_t,dtype=bool)
            if filt_option is None:
                filt_option = 'simple'
            elif filt_option =='const':
                filt_t = ass.stay_True(filt_t)
                filt_option='strict'
        else:
            f_nump = None
            filt_option = None
        
        if weights_t is not None:
            w_nump = self.init_xt(weights_t)
        else:
            w_nump = None
        
        
        func,func_args,func_inner = \
        self.get_Dynamics_inner_kernel_functions(prop,filt_option,weights_t)
        
        if prop.lower() == 'fs':
            if q is None:
                raise Exception('For {} calculation you must give a "q" value '.format(prop))
            kernel_args =(q,)
        else:
            kernel_args = tuple()
            
        args = (func,func_args,func_inner,
                Prop_nump,nv,
                x_nump,f_nump,w_nump,
                block_average,
                multy_origin,
                every,kernel_args)
        
        
        try:
            #prop_kernel(prop_nump, nv, x_nump, f_nump, nfr)
            DynamicProperty_kernel(*args)
        except ZeroDivisionError as err:
            logger.error('Dynamics Run {:s} --> There is a {} --> Check your filters or weights'.format(prop,err))
            return None
        
        
        #tf2 = perf_counter()
        if prop.lower() =='p2':
            Prop_nump = 0.5*(3*Prop_nump-1.0)
        t = ass.numpy_keys(xt)
        dynamical_property = {'time':t-t.min(), prop : Prop_nump }
        #tf3 = perf_counter() - tf2
        
        tf = perf_counter()-tinit
        #logger.info('Overhead: {:s} dynamics computing time --> {:.3e} sec'.format(prop,overheads+tf3))
        ass.print_time(tf,inspect.currentframe().f_code.co_name +'" ---> Property: "{}'.format(prop))
        return dynamical_property
    
    def multy_tau_average(self,xt,every=1):
        '''
    
        Parameters
        ----------
        xt : dictionary
            keys: times
            values: vector array (total population,nd) where nd =1,2,3
            It is the input.

        Returns
        -------
        dynamical_property : dictionary
            keys: time
            values: float

        '''
        tinit = perf_counter()
        
        Prop_nump,nv = self.init_prop(xt)
        x_nump = self.init_xt(xt)
        
        try:
            scalar_time_origin_average(Prop_nump,nv,x_nump,every)
        except ZeroDivisionError as err:
            logger.error('multy tau run  --> There is a {} --> Check your filters or weights'.format(err))
            return None
        
        
        t = ass.numpy_keys(xt)
        dynamical_property = {'time':t-t.min(), 'corr' : Prop_nump }
        
        tf = perf_counter()-tinit
        ass.print_time(tf,inspect.currentframe().f_code.co_name)
        return dynamical_property
    
    def get_TACF_inner_kernel_functions(self,prop,filt_option,weights_t):
        '''
        

        Parameters
        ----------
        prop : string
            cos: cos function
            sin: sin function

        filt_t : dictionary
            keys: times
            values: boolean array
        weights_t: dictionary
            keys: time
            values: float array (total population,)
        Returns
        -------
        funcs : 7 functions for the TACF kernel

        '''
        inner_mapper = {'cos':'cosCorrelation_kernel',
                        'sin':'sinCorrelation_kernel'}
        inner_zero_mapper = {'cos':'fcos_kernel',
                             'sin':'fsin_kernel'}
        
        inner_func_name = inner_mapper[prop.lower()] 
        inner_zero_func_name = inner_zero_mapper[prop.lower()]
        
        name = 'dynprop'
        af_name = 'get'
        args_z_name = 'get_zero'
        mean_func_name = 'mean'
        secmom_func_name = 'secmoment'
        
        if filt_option is not None:
            ps = '_{:s}'.format(filt_option)
            pz = '_filt'
            name += ps 
            af_name+= ps
            args_z_name +=pz 
            mean_func_name+=pz
            secmom_func_name+=pz
        if weights_t is not None:
            ps = '_weighted'
            args_z_name += ps
            name += ps
            af_name += ps
            mean_func_name+=ps
            secmom_func_name+=ps
        
        func_name = '{:s}__kernel'.format(name)
        args_func_name = '{:s}__args'.format(af_name)
        args_zero_func_name = '{:s}__args'.format(args_z_name)
        mean_func_name = '{:s}__kernel'.format(mean_func_name)
        secmom_func_name = '{:s}__kernel'.format(secmom_func_name)
        
        func_names = [func_name,args_func_name, inner_func_name,
                      mean_func_name,secmom_func_name,
                      args_zero_func_name,
                      inner_zero_func_name]
        
        s = ''.join( ['f{:d}={:s} \n'.format(i,f) for i,f in enumerate(func_names)] )
        
        logger.info(s)
        
        funcs = tuple([ globals()[f] for f in func_names])
        return funcs
  
    def TACF(self,prop,xt,filt_t=None,
             wt=None,filt_option=None,block_average=False):
        '''
        

        Parameters
        ----------
        prop : string
            cos: cos function
            sin: sin function
        xt : TYPE
            DESCRIPTION.
        filt_t : dictionary
            keys: times
            values: boolean array
        wt: dictionary
            keys: time
            values: float array (total population,)
        filt_option : string
            simple: consider the population based on time origin
            strict: consider the population based on time origin and  time t
            change: consider the population based on time origin and changed on time t
        block_average : bool
            Ways of averaginf
            True  <  < >|population  >|time origins
            False < >|population,time origins

        Returns
        -------
        TACF_property : dictionary
            keys: time
            value: float 

        '''
        tinit = perf_counter()
        
        Prop_nump,nv = self.init_prop(xt)
        mu_val,mu_num = self.init_prop(xt)
        secmom_val,secmom_num = self.init_prop(xt)
        
        x_nump = self.init_xt(xt,dtype=float)
        
        if filt_t is not None:
            f_nump = self.init_xt(filt_t,dtype=bool)
            if filt_option is None:
                filt_option= 'simple'
            if filt_option =='const':
                filt_option = 'strict'
                filt_t = ass.stay_True(filt_t)
        else:
            f_nump = None
            filt_option = None
        
        if wt is not None:
            w_nump = self.init_xt(wt)
        else:
            w_nump = None
        
        func_name, func_args, inner_func,\
        mean_func, secmoment_func, func_args_zero, inner_func_zero \
        = self.get_TACF_inner_kernel_functions(prop,filt_option,wt)
        

        args = (func_name, func_args, inner_func,
              mean_func, secmoment_func, func_args_zero, inner_func_zero,
              Prop_nump,nv,
              mu_val,mu_num,secmom_val,secmom_num,
              x_nump, f_nump, w_nump,
              block_average)
    
        TACF_kernel(*args)
        #print(Prop_nump)
        t = ass.numpy_keys(xt)
        TACF_property = {'time':t -t.min(),'tacf':Prop_nump}
        tf = perf_counter()-tinit
        
        ass.print_time(tf,inspect.currentframe().f_code.co_name)
        
        return TACF_property
    
    def get_Kinetics_inner_kernel_functions(self,wt):
        '''
        

        Parameters
        ----------
        wt :  dictionary
            keys: time
            values: float array (total population,)

        Returns
        -------
        returns function to call in kinetics kernel,
                function to set the arguments
        

        '''
        if wt is None:
            func_args = 'get__args'
            func_name = 'Kinetics_inner__kernel'
        else:
            func_args = 'get_weighted__args'
            func_name = 'Kinetics_inner_weighted__kernel'
        
        logger.info('func name : {:s} , argsFunc name : {:s}'.format(func_name,func_args))
        
        return globals()[func_name],globals()[func_args]
    
    def Kinetics(self,xt,wt=None,block_average=False,
                 multy_origin=True):
        '''

        Parameters
        ----------
        xt : dictionary
            keys:times
            values: boolean array (total population,)
        wt : dictionary
            keys: time
            values: float array (total population,)
        
        block_average : bool
            Ways of averaging
            True  <  < >|population  >|time origins
            False < >|population,time origins

        Returns
        -------
        kinetic_property :dictionary
            keys: time
            values: float (it represents the percentage of state change)
        '''
        tinit = perf_counter()
        
        Prop_nump,nv = self.init_prop(xt)
        x_nump = self.init_xt(xt,dtype=bool)
        
        if wt is not None:
            w_nump = self.init_xt(wt)
        else:
            w_nump = None
        
        func_name,func_args = self.get_Kinetics_inner_kernel_functions(wt)
        
        args = (func_name,func_args,
                Prop_nump,nv,
                x_nump,w_nump,
                block_average,
                multy_origin)
        Kinetics_kernel(*args)
        t = ass.numpy_keys(xt)
        kinetic_property = {'time':t-t.min(),'K': Prop_nump}
        tf = perf_counter()-tinit
        #logger.info('Overhead: {:s} dynamics computing time --> {:.3e} sec'.format(prop,overheads+tf3))
        ass.print_time(tf,inspect.currentframe().f_code.co_name +'" ---> Property: "Kinetics')
        return kinetic_property    
    
    
    
    
class Analysis_Confined(Analysis):
    '''
    Class for performing the analysis of confined systems
    Includes functions for Structure and Dynamics
    '''
    
    def __init__(self,    topol_file,
                 connectivity_info,       
                 memory_demanding=False,
                 **kwargs):
        super().__init__(topol_file,
                         connectivity_info,
                         memory_demanding,**kwargs)
        known_kwargs = ['conftype','adsorption_interval','particle_method','polymer_method',
                        'polymer','particle','cylinder_length']
        defaults = {'particle_method':'molname',
                    'polymer_method':'molname'
                    }
        #Obligatory keywords
        if 'conftype' in kwargs:
            self.conftype = kwargs['conftype']
        else:
            raise Exception('"conftype" has no default value. Available options are:\nzdir\nydir\nxdir\nzcylindrical\nsherical_particle')
        
        for k in ['particle','polymer']:
            if k not in kwargs:
                raise Exception('You need to pass the keyword "{:s}" since it does not have a default value'.format(k))
        
        #Unkown keywords
        for k in kwargs:
            if k not in known_kwargs:
                raise Exception('You have  passed the variable {:s} but I dont know what to do with it.\n Check also for typos'.format(k))
        
        #Special keywords/might have default
        if 'adsorption_interval' in kwargs:
           self.setup_adsorption(kwargs['adsorption_interval'])
        else:
            self.adsorption_interval = ((0,0),)
          
        #Keywords with default values
        for k,d in defaults.items():
            if k not in kwargs:
                setattr(self,k,d)
            else:
                setattr(self,k,kwargs[k])
            
        if 'cylinder_length' in kwargs:
            self.cylinder_length = kwargs['cylinder_lenght']
            try:
                self.ztrain = self.kwargs['ztrain']
            except KeyError as e:
                raise e('when you give a finite length of a cylinder you need to provide also the "ztrain", which corresponds to the adsorbed distance of the polymer above and below the cylinder')
            self.train_specific_method = self.train_in_finite_cylinder
        self.kwargs = kwargs
        

        self.confined_system_initialization()
        
        self.polymer_ids = np.where(self.polymer_filt)[0]
        self.particle_ids = np.where(self.particle_filt)[0]
        return
    
    def setup_adsorption(self,adsorption_interval):
        if not ass.iterable(adsorption_interval[0]):
                self.adsorption_interval = (tuple(adsorption_interval),)
        else:
            self.adsorption_interval = tuple(tuple(ai) for ai in adsorption_interval)
        for ai in self.adsorption_interval:
            if len(ai) != 2:
                raise ValueError('Wrong value of Adsorption interval = {} --> must be of up and low value'.format(ai))
    ############## General Supportive functions Section #####################
    
    def find_connectivity_per_chain(self):
        cargs =dict()
        x = self.sorted_connectivity_keys
        for j,args in self.chain_args.items():
            f1 = np.isin(x[:,0],args)
            f2 = np.isin(x[:,1],args)
            f = np.logical_or(f1,f2)
            cargs[j] = x[f]
        self.connectivity_per_chain = cargs
        return
    
    def find_systemic_filter(self,name):
        method = getattr(self,name+'_method')
        
        if method == 'molname':
            compare_array = self.mol_names
        elif method == 'atomtypes':
            compare_array = self.at_types
        elif method == 'molids':
            compare_array = self.mol_ids
        elif method =='atomids':
            compare_array = self.at_ids
        else:
            raise NotImplementedError('method "{}" is not Implemented'.format(method))
        
        try:
            look_name_s = self.kwargs[name]
        except KeyError:
            raise Exception('You need to provide the key word "{:s}"'.format(name))
        else:
            t1 = type(look_name_s) is str
            t2 = type(look_name_s) is list
            t3 = type(look_name_s) is np.ndarray
            if t1 or t2 or t3:
                if t2 or t3:
                    for i in range(1,len(look_name_s)):
                        if type(look_name_s[i-1]) != type(look_name_s[i]):
                                raise Exception('elements in variable {:s} must be of the same type'.format(name))
            else:
                raise NotImplementedError('{:s} variable is allowed to be either string, list or array'.format(name))
                
        filt = np.isin(compare_array,look_name_s)
        setattr(self,name+'_filt',filt)
        
        return
    
    def find_particle_filt(self):
        self.find_systemic_filter('particle')
        return 
    
    def find_polymer_filt(self):
        self.find_systemic_filter('polymer')
        return
       
    def translate_particle_in_box_middle(self,coords,box):
        particle_cm = self.get_particle_cm(coords)
        cds = coords.copy()
        cds += box/2 - particle_cm
        cds = implement_pbc(cds,box)
        return cds
    
    def translated_coords(self,frame):
        coords = self.get_coords(frame)
        box = self.get_box(frame)
        coords = self.translate_particle_in_box_middle(coords, box)
        return coords
   
    def get_whole_coords(self,frame):
        coords = self.get_coords(frame)
        box = self.get_box(frame)
        coords = self.translate_particle_in_box_middle(coords, box)
        coords = self.unwrap_coords(coords, box)
        return coords
   
    def confined_system_initialization(self):
        '''
        IMPORTANT Function
        This function gets the prober functions depending
        on the confinment type from a number of classes. 
        This is done in initialization to increase speed and modularity
        at the same time        
        
        '''
        t0 = perf_counter()
        self.find_particle_filt()
        self.nparticle = self.mol_ids[self.particle_filt].shape[0]
    
        self.find_polymer_filt()
        self.npol = self.mol_ids[self.polymer_filt].shape[0]
    
        self.particle_mass = self.atom_mass[self.particle_filt]
        self.polymer_mass = self.atom_mass[self.polymer_filt]
        
        #self.find_masses()
        self.unique_atom_types = np.unique(self.at_types)
        
        #Getting the prober functions
        self.dfun = self.get_class_function(Distance_Functions,self.conftype)
        self.box_add = self.get_class_function(Box_Additions,self.conftype)
        self.volfun = self.get_class_function(bin_Volume_Functions,self.conftype)
        
        self.unit_vectorFun = self.get_class_function(unit_vector_Functions,self.conftype)
        ##########
        
        self.find_args_per_residue(self.polymer_filt,'chain_args')
        self.find_connectivity_per_chain()
        self.find_args_per_residue(self.particle_filt,'particle_args')
        self.nparticles = len(self.particle_args.keys())
        
        self.all_args = np.arange(0,self.natoms,1,dtype=int)
        
        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name)
        return
       
    def get_class_function(self,_class,fun,inplace=False):
        fun = getattr(_class,fun)
        if inplace:
            attr_name = _class+'_function'
            setattr(self,attr_name, fun)
        return fun
     
    def get_frame_basics(self,frame):
        #coords = self.translated_coords(frame)
        coords = self.get_coords(frame)
        box  = self.get_box(frame)          
        
        coords = implement_pbc(coords,box)
        d = self.get_distance_from_particle(coords)
        return coords,box,d
    
    def get_whole_frame_basics(self,frame):
        coords = self.get_whole_coords(frame)
        box  = self.get_box(frame)          
        cm = self.get_particle_cm(coords)
        d = self.dfun(self,coords,cm)
        return coords,box,d,cm



    def get_particle_cm(self,coords):
        '''
        gets particles center of mass

        Parameters
        ----------
        coords : system coordinates

        Returns
        -------
        cm : center of mass (the number of coordinates depends on the confinment type)

        '''
        cm =CM ( coords[self.particle_filt], self.particle_mass)
        return cm
    

    
    ###############End of General Supportive functions Section#########

    ############### Conformation Calculation Supportive Functions #####
   


   
    def is_bridge(self,coords,istart,iend,periodic_image_args,box):
        # used to distinguish loops from bridges
        
        # check if one belong to the periodic_images and other not --> this means bridge
        e = iend in periodic_image_args 
        s = istart in periodic_image_args
        if (e and not s) or (s and not e):
            return True
        # If both in periodic image args check
        if e and s:
            res = coords[[istart,iend]]
            imin = [1e16,1e16]
            idxmin = [1e16,1e16]
            for idx,L in enumerate(self.box_add(box)):
                cm = self.get_particle_cm(coords+L)
                d = self.dfun(self,res,cm)
                for i in range(2):
                    if d[i]<imin[i]:
                        idxmin[i] = idx
                        imin[i]=d[i]
            return idxmin[0] != idxmin[1] #bridge if they are unequal
        
        # check if belong to different particle
        if self.nparticles !=1:
            logger.warning('WARNING Function {:s}: This Function was never examined in test cases of more than one nanoparticles'.format(inspect.currentframe().f_code.co_name))
            r0 = coords[istart]
            re = coords[iend]
            CMps = np.empty((self.nparticles,3),dtype=float)
            for i,(k,args) in enumerate(self.particle_args.items()):
                CMps[i] = CM(coords[args],self.atom_mass[args])
            
            if not e and not s: # if in main box no need to check periodic images
                rer = CMps-re
                r0r = CMps -r0
                de = np.sum(rer*rer,axis=1)
                d0 = np.sum(r0r*r0r,axis=1)
            else:
                de = 1e16 ; d0=1e16
                for L in self.box_add(self.get_box(self.current_frame)):
                    r0r = CMps+L -r0
                    rer = CMps+L -re
                    de = np.minimum(de,np.sum(rer*rer,axis=1))
                    d0 = np.minimum(d0,np.sum(r0r*r0r,axis=1))
            
            if de.argmin()!=d0.argmin():
                return True
                
        return False
    
    def train_in_finite_cylinder(self,coords,cmp,ftrain):
        
        Lc = self.cylinder_length
        zd_train = self.ztrain
        
        zd = Distance_Functions.zdir(None,coords,cmp)

        fz = np.logical_and(zd >=Lc/2,zd<=Lc/2+zd_train)                                 #
        fud_surf = np.logical_and(fz,ftrain)
        
        return np.logical_or(ftrain,fud_surf)       
        
    
    def get_filt_train(self):
        '''
        
        Returns
        -------
        ftrain : bool array (shape = (n,) )
        image_trains : bool_array (shape = (n,))
        where n is the number of TOTAL NUMBER OF ATOMS
        
        '''
        
        coords = self.get_coords(self.current_frame)
        ftrain = False
        nonp_ftrain = False
              
        d  = self.get_distance_from_particle() # minimum image distance
        
        cmp = self.get_particle_cm(coords)
        d_nonp = self.dfun(self,coords, cmp) #absolut distance not minimum image
        
        for interval in self.adsorption_interval:
            dlow = interval[0]
            dup = interval[1]
            fin = filt_uplow(d, dlow, dup)
            ftrain = np.logical_or(ftrain, fin)
            
            fin_nonp = filt_uplow(d_nonp,dlow,dup)
            nonp_ftrain = np.logical_or(nonp_ftrain,fin_nonp)
        try:
            ftrain = self.train_specific_method(coords,cmp,ftrain)
        except AttributeError:
            pass
        
            
            
        ftrain = np.logical_and(ftrain, self.polymer_filt)
        nonp_ftrain = np.logical_and(nonp_ftrain, self.polymer_filt)
        
        image_trains = np.logical_and(ftrain,np.logical_not(nonp_ftrain))
        
        return ftrain,image_trains
    
    def get_distance_from_particle(self,r=None,ids=None):
        '''
        Give either r or ids or nothing
        If you give r then the 
        minimum image distance of r from particle will be found
        If you give ids the minimum image distance of coords[ids] will be found
        else the minimum image distance of coords
        Parameters
        ----------
        r : coordinates (float array shape=(n,3) ).
        ids : int array

        Returns
        -------
        d : minimum image distance (float array shape = (n,) )
            Depends on the confinment type
            e.g. for flat surfaces parallel on the xy plane 
            only the z direction is considered

        '''
        frame = self.current_frame
        
        box = self.get_box(frame)
        coords = self.get_coords(frame)
        if r is None:
            if ids is not None:
                r = coords[ids]
            else:
                r = coords
        cm = self.get_particle_cm(coords)
        
        d = 1e16
        for L in self.box_add(box):
            d = np.minimum(d,self.dfun(self,r,cm+L))
        return d
    
    def conformations(self):
        '''
        Returns the ids of trains,loops,tails 
        and bridges given the coords of the current frame

        Returns
        -------
        ads_chains : int array
            ids of adsorbeds chains.
        args_train : int array
            ids of trains
        args_tail : int array
            ids of tails
        args_loop : int array
            ids of loops
        args_bridge : int arry
            ids of bridges

        '''
        box = self.get_box(self.current_frame)
        
        ftrain, image_trains = self.get_filt_train()
        args_train = np.nonzero(ftrain)[0]
        periodic_image_args = set(np.nonzero(image_trains)[0])
        #logger.debug('Number of periodic image trains ={:d}\n Number of trains = {:d}'.format(len(periodic_image_args),args_train.shape[0]))
        #ads_chains
        ads_chains = np.unique(self.mol_ids[ftrain])
        #check_occurances(ads_chains)
        fads_chains = np.isin(self.mol_ids,ads_chains)
        args_ads_chain_atoms = np.nonzero(fads_chains)[0]
        
        #tail_loop_bridge
        #f_looptailbridge = np.logical_and(fads_chains, np.logical_not(ftrain))
        #args_looptailbridge = np.nonzero(f_looptailbridge)[0]
        
        args_tail = np.empty(0,dtype=int) ; 
        args_bridge = np.empty(0,dtype=int) ; 
        args_loop  = np.empty(0,dtype=int)
        
        coords = self.get_coords(self.current_frame) # need to identify bridges over loops
        
        for j in ads_chains:
            #args_chain = self.chain_args[j]
            connectivity_args = self.connectivity_per_chain[j]
            #nch = args_chain.shape[0]
            
            #1) Identiyfy connectivity nodes
            nodes = []
            for c in connectivity_args:
                ft = ftrain[c]
                if ft[0] and not ft[1]: nodes.append( (c[1],c[0]) )
                if not ft[0] and ft[1]: nodes.append( (c[0],c[1]) )
            
            #2) Loop over nodes and identify chunks
            for node in nodes:

                chunk = {node[0]}
                istart = node[1]
                
                old_chunk = set() ; loopBridge = False ; found_iend = False
                new_neibs = chunk.copy()
                
                while old_chunk != chunk:
                    old_chunk = chunk.copy()
                    new_set = set()
                    for ii in new_neibs:
                        for neib in self.neibs[ii]:
                            
                            if neib == istart: continue
    
                            if ftrain[neib]: 
                                loopBridge = True
                                
                                if not found_iend:
                                    iend = neib 
                                    found_iend = True 
                                continue
                             
                            if neib not in chunk:        
                                new_set.add(neib)
                                chunk.add(neib)
                    new_neibs = new_set
    
                if not found_iend:
                    try: del iend
                    except: pass
  
                
                chunk = np.array(list(chunk),dtype=int)
                
                assert not ftrain[chunk].any(), \
                    'chunk in train chain {:d}, node {}, chunk size = {:d}\
                    ,istart = {:d}, iend ={:d} \n\n chunk =\n {} \n\n'.format(j,
                    node,chunk.shape[0],istart,iend,chunk)
                assert fads_chains[chunk].all(),\
                    'chunk out of adsorbed chains, j = {:d} ,\
                        node = {} n\n chunk \n {} \n\n'.format(j,
                        node,chunk)
                
                if loopBridge:
                    if not self.is_bridge(coords,istart,iend,periodic_image_args,box):    
                        args_loop = np.concatenate( (args_loop, chunk) )             
                    else:
                        #logger.debug('chain = {:d}, chunk | (istart,iend) = ({:d}-{:d}) is bridge'.format(j,istart,iend))
                        args_bridge = np.concatenate( (args_bridge, chunk) )
                else:
                    args_tail = np.concatenate( (args_tail, chunk) )

        #args_tail = np.unique(args_tail)
        args_loop = np.unique(args_loop)
        args_bridge = np.unique(args_bridge)
        assert not np.isin(args_train,args_tail).any(),\
        'tails in trains, there are {:d}'.format(np.count_nonzero(np.isin(args_train,args_tail)))
        
        assert not np.isin(args_train,args_loop).any(),\
        'loops in trains, there are {:d}'.format(np.count_nonzero(np.isin(args_train,args_loop)))
        
        assert not np.isin(args_tail,args_loop).any(),\
        'loops in tails, there are {:d}'.format(np.count_nonzero(np.isin(args_tail,args_loop)))
        
        
        assert not np.isin(args_train,args_bridge).any(),\
        'bridge in trains, there are {:d}'.format(np.count_nonzero(np.isin(args_train,args_bridge)))
        
        assert not np.isin(args_tail,args_bridge).any(),\
        'bridge in tails, there are {:d}'.format(np.count_nonzero(np.isin(args_tail,args_bridge)))
        
        assert args_train.shape[0] + args_tail.shape[0] +\
               args_loop.shape[0] + args_bridge.shape[0] \
               == np.count_nonzero(fads_chains) , 'Confomormations do not sum up correnctly,\
                   correct sum = {:d}, sum = {:d}'.format( args_ads_chain_atoms.shape[0],
                   args_train.shape[0] + args_tail.shape[0] +\
                   args_loop.shape[0] + args_bridge.shape[0] 
                                                      )
    
        
        return ads_chains,args_train,args_tail,args_loop,args_bridge
    
    def connected_chunks(self,args):
        '''
        takes a set of ids and finds the connected chunks from it
        It is very usefull to find conformation size distributions
        Parameters
        ----------
        args : int array of atom ids
        
        Returns
        -------
        chunks : list of sets. Each set has the ids of the connected chunks

        '''
        #t0 = perf_counter()
        set_args = set(args)
        chunks = []
        #aold = -1
        
        while(len(set_args)>0):

            a = set_args.pop()
            set_args.add(a)
            #assert a != aold,'while loop stuck on argument = {}'.format(a)
            old_set_a = set()
            new_set_a = {a}
                    
            new_neibs = new_set_a.copy()
            
            while new_set_a != old_set_a:
                old_set_a = new_set_a.copy()
                
                for j in new_neibs.copy():
                    new_neibs = set()
                    for neib in self.neibs[j]:
                        if neib in set_args:
                            new_set_a.add(neib)
                            if neib not in old_set_a:
                                new_neibs.add(neib)

            chunks.append(new_set_a)
            set_args.difference_update(new_set_a)
        
            #aold # (for debugging-combined with assertion above)
        #ass.print_time(perf_counter()-t0,'connected_chunks')
        return chunks
    
    def length_of_connected_chunks(self,args,coords,exclude_bonds=None):
        # this currently is suitable for linear chains in corse-grained or united atom represtation
        # if there is an all atom representation exclude_bonds must be used
        bonds = self.sorted_connectivity_keys
        f1 = np.isin(bonds[:,0],args)
        f2 = np.isin(bonds[:,1],args)
        f = np.logical_and(f1,f2)
        cb = bonds[f]
        coords1 = coords[cb[:,0]]
        coords2 = coords[cb[:,1]]
        r = coords2-coords1
        dists = np.sqrt(np.sum(r*r,axis=1))
        return dists.sum()
    ############### End of Conformation Calculation Supportive Functions #####
    
    ######## Main calculation Functions for structural properties ############      
    
    def get_bins(self,binl,dmax,offset=0):
        bins =np.arange(offset,offset+dmax+binl, binl)
        return bins
    
    def calc_density_profile(self,binl,dmax,offset=0,
                             option='',mode='mass',flux=False,types=None):
        '''
        Master function to compute the density profile

        Parameters
        ----------
        binl : float
            binningl legth
        dmax : float
            maximum distance to calculate
        mode : string
            'mass' for mass density
            'number' for number density
            For conformations option is set automatically to massnumber
        option : string
            'pertype' decomposes the density to each atomic type contribution
            '2side' computes the profile to both sides.
                Only valid for one directional confinement (i.e. zdir,ydir,xdir) 
        flux : bool
            if True it calculates density fluxuations
            by first calling this function to compute the density.
            density fluctuations are computed by the standard deviation

        Returns
        -------
        density_profile : dictionary containing 
            discription :  key  |  value
            the distance: 'd'  | float array 
            the density: 'rho' | float array
            density of  'type1' | float array1
            density of 'type2' | float array2
                          .
                          .
                          .
            density of 'typen' | float arrayn
            density fluctuations: 'rho_flux' | float array

        '''
        t0 = perf_counter()
        
        #initialize
        ##############
        scale = 1.660539e-3 if mode == 'mass' else 1.0
        density_profile = dict()
        
        if dmax is None:
            NotImplemented
        bins  =   self.get_bins(binl,dmax,offset)
        nbins = len(bins)-1
        rho = np.zeros(nbins,dtype=float)
        mass = self.atom_mass 
        
        
        if option =='':
            args = (nbins,bins,rho)
            contr=''
        elif option == 'pertype':
            if types is None:
                types = self.unique_atom_types
            elif type(types) is list or type(types) is tuple:
                types = types
            elif type(types) is str:
                types = [ types ]
            else:
                raise Exception('Wrong types')
            for ty in types:
                if ty not in self.unique_atom_types:
                    raise ValueError('{:s} is not regognized as an atom type'.format(ty))

            rho_per_atom_type = {t:np.zeros(nbins,dtype=float) for t in types }
            ftt = {t: t == self.at_types for t in rho_per_atom_type.keys() }
            
            args = (nbins,bins,rho,rho_per_atom_type,ftt)
            contr = '__pertype'
        
        elif option == 'bymol':
            rho_per_mol = {m:np.zeros(nbins,dtype=float) for m in np.unique(self.mol_names) }
            ftt = {m:  m == self.mol_names for m in rho_per_mol.keys() }
            args = (nbins,bins,rho,rho_per_mol,ftt)
            contr = '__bymol'

        elif option =='2side':
            rho_down = np.zeros(nbins,dtype=float)
            args =(nbins,bins,rho,rho_down)
            contr ='__2side'
        elif option =='conformations':              
            confdens = {nm+k:np.zeros(nbins,dtype=float) for nm in ['m','n']
                        for k in ['rho','train','tail','loop','bridge','free'] 
                        }         
            stats = { k : 0 for k in ['adschains','train','looptailbridge',
                                  'tail','loop','bridge']}
            dlayers = [(bins[i],bins[i+1]) for i in range(nbins)]
            args = (dlayers, confdens, stats)
            contr='__conformations'
            mode='massnumber'
            
        if mode =='mass': 
            args =(*args,mass)
        
        
        if flux:
            if mode =='number':
                raise NotImplementedError('number density fluxations are not implemented.Please use mass density and then rescale to number')
            density_profile.update(self.calc_density_profile(binl,dmax,
                                 offset=offset,mode=mode,option=option))
            rho_mean = density_profile['rho'].copy()
            rho_mean/=scale
            args = (nbins,bins,rho,mass,rho_mean**2)
            func =  'mass'+'_density_profile'+'_flux'
            if mode !='mass' or option!='':
                logger.warning('mass mode and total density fluxuations are calculated')
        else:
            func =  mode+'_density_profile'+contr    
        
        
        #############
        
        #calculate
        #############
        nframes = self.loop_trajectory(func, args)
        #############
        
        #post_process
        ##############
        
        if flux is not None and flux !=False:
            density_profile['rho_flux'] = rho*scale**2/nframes
        else:
            rho*=scale/nframes 
            d_center = center_of_bins(bins)
            density_profile.update({'d':d_center-offset})
            
            density_profile.update({'rho':rho})
            if option =='pertype':
                for t,rhot in rho_per_atom_type.items(): 
                    density_profile[t] = rhot*scale/nframes 
            elif option=='bymol':
                for m,rhom in rho_per_mol.items():
                    density_profile[m] = rhom*scale/nframes
            elif option =='2side':
                rho_down *=scale/nframes
                density_profile.update({'rho_up':rho,'rho_down':rho_down,'rho':0.5*(rho+rho_down)})
            elif option =='conformations':
                stats = {k+'_perc':v/self.npol/nframes if 'adschains'!=k else v/len(self.chain_args)/nframes
                         for k,v in stats.items()}
                dens= {k:v*scale/nframes for k,v in confdens.items()}

                for k in ['train','tail','loop','bridge','free']: 
                    dens['mrho'] += dens['m'+k] 
                    dens['nrho'] += dens['n'+k]
                    
                dens['rho'] = dens['mrho']
                
                density_profile.update(dens)
                density_profile.update(stats)
                
        #############
        
        tf = perf_counter() -t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        
        return density_profile
    

    
    def calc_P2(self,topol_vector,binl,dmax,offset=0,option=''):
        '''
        Calculates P2 bond parameter
        The unit vector is predifined by the type of confinement
        
        Parameters
        ----------
        binl : float
            binning legth
        dmax : float
            maximum distance.
        topol_vector : int or list of atom types
            int is 2,3,4 for 1-2, 1-3, 1-4 bond vectors respectively
            with a list of atom types it will look 
            the connectivity for len(topol_vector) ==2
            the angles for len(topol_vector) ==3
            the dihedrals for len(topol_vector) ==4
            and it will get the given vectors

        Returns
        -------
        orientation : dictionary containing
        discription :  key  |  value
        the distance: 'd'  | float array
        the function: 'P2' | float array
        '''

        t0 = perf_counter()

        bins  =   self.get_bins(binl,dmax,offset)
        
        dlayers=[]
        for i in range(0,len(bins)-1):
            dlayers.append((bins[i],bins[i+1]))
        d_center = np.array([0.5*(b[0]+b[1]) for b in dlayers])
        
        ids1, ids2 = self.find_vector_ids(topol_vector)
        nvectors = ids1.shape[0]
        logger.info('topol {}: {:d} vectors  '.format(topol_vector,nvectors))
        
        
        
        if option in ['conformation','conformations']:
            confs = ['train','loop','bridge','tail','free']
            costh_unv = {k:[[] for i in range(len(dlayers))] for k in confs }
            args = (ids1,ids2,dlayers,nvectors,costh_unv)
            
            s = '_conformation'
        elif option=='':
            s =''
            costh_unv = [[] for i in range(len(dlayers))]
            costh = np.empty(nvectors,dtype=float)
            args = (ids1,ids2,dlayers,costh,costh_unv)
        else:
            raise NotImplementedError('option "{}" not Implemented.\n Check your spelling when you give strings'.format(option))
       
        
        nframes = self.loop_trajectory('P2'+s, args)

        if option =='':
            
            costh2_mean = np.array([ np.array(c).mean() for c in costh_unv ])
            costh2_std  = np.array([ np.array(c).std()  for c in costh_unv ])
    
            s='P2'
            orientation = {'d':  d_center} 
            orientation.update({s: 1.5*costh2_mean-0.5, s+'(std)' : 1.5*costh2_std-0.5 })
        
        elif option in ['conformation','conformations']:
            
            orientation = {'d':  d_center} 
            
            for j in confs:
                costh2_mean = np.array([ np.array(c).mean() for c in costh_unv[j] ])
                costh2_std  = np.array([ np.array(c).std()  for c in costh_unv[j] ])
                s='P2'+j
                orientation.update({s: 1.5*costh2_mean-0.5, s+'(std)' : 1.5*costh2_std-0.5 })
            
            orientation.update(self.calc_P2(topol_vector,binl,dmax,offset))
        
        tf = perf_counter() - t0
        
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        
        return orientation   
        
    def calc_particle_size(self):
        t0 = perf_counter()
        part_size = np.zeros(3)
        args = (part_size,)
        nframes = self.loop_trajectory('particle_size',args)
        part_size /= nframes
        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        return part_size
        
    def calc_dihedral_distribution(self,phi,filters=dict()):
        '''
        Computes dihedral distributions as a function of given populations
        ----------
        dmax : float
            maximum distance
        binl : float
            binning length

        Returns
        -------
        dihedral_distr : dictionary

        '''
        t0 = perf_counter()
        diht,ft = self.calc_dihedrals_t(phi,filters = filters)
        distrib = {k: [] for filt in filters.values() for k in filt }
        
        for k in distrib:
            for t,dih in diht.items():
                distrib[k].extend(dih[ft[k][t]])
        
        distrib['system'] = []
        for t,dih in diht.items():
            distrib['system'].extend(dih)
            
        for k in distrib:
            distrib[k] = np.array(distrib[k])*180/np.pi
        
        tf = perf_counter()-t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name)
        return distrib

    def calc_chain_characteristics(self,binl,dmax,offset=0):
        '''
        Chain characteristics as a function of distance of chain center of mass
        Parameters
        ----------
        dmin : float
            minimum distance
        dmax : float
            maximumum distance
        binl : float
            binning length

        Returns
        -------
        chain_chars : dictionary with keys the characteristics in strings
            and values float arrays. Also the distance is included

        '''
        t0 = perf_counter()
        
        bins  =   self.get_bins(binl,dmax,offset)
        dlayers = [(bins[i],bins[i+1]) for i in range(len(bins)-1)]
        d_center = [0.5*(b[0]+b[1]) for b in dlayers]
        nl = len(dlayers)
        
        chars_strlist = ['k2','Rg2','Rg','Ree2','asph','acyl', 'Rgxx_plus_yy', 'Rgxx_plus_yy', 'Rgyy_plus_zz']
        chars = {k:[[] for d in dlayers] for k in chars_strlist }

        chain_args = self.chain_args
        
        # calculate
        args  = chain_args,dlayers,chars
        nframes = self.loop_trajectory('chain_characteristics', args)
                
        #post_process
        chain_chars = {'d':np.array(d_center)-offset}
        
        for k,v in chars.items():
            chain_chars[k] = np.array([ np.mean(chars[k][i]) for i in range(nl) ])
            chain_chars[k+'(std)'] = np.array([ np.std(chars[k][i]) for i in range(nl) ])
        
        tf= perf_counter() -t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
        
        return chain_chars
    
    ###### End of Main calculation Functions for structural properties ########
    

    
    def calc_conformations_t(self,option=''):
        '''
        computes conformations as a function of time

        Parameters
        ----------
        option : string
            for feature use

        Returns
        -------
        confs_t : dictionary
            keys: strings (conformations)
            values: dictionary
                keys: times
                values: number of atoms on the specific conformation

        '''
        t0 = perf_counter()
        
        confs_t = dict()
        args = (confs_t,)
        nframes = self.loop_trajectory('confs_t'+option, args)
        confs_t = ass.rearrange_dict_keys(confs_t)
        
        conforms_t = {'time' : ass.numpy_keys( confs_t[ list(confs_t.keys())[0] ] ) }
        conforms_t.update({k: ass.numpy_values(c) for k,c in confs_t.items()})
        tf = perf_counter() - t0
        ass.print_time(tf,inspect.currentframe().f_code.co_name,nframes)
    
        return conforms_t
    


    
 

    

class Filters():
    '''
    The Filters class contains the functions that produce
    boolean arrays ("filters") that are used to distinguish the 
    population, e.g. segments to trains, loops ..., chains at distance etc ...
    '''
    def __init__(self):
        pass
    
    @staticmethod
    def calc_filters(self,filters,*args):
        bool_data = dict()
        
        for k,filt in filters.items():
            bool_data.update(  getattr(Filters,k)(self, filt, *args)  )
        
        return bool_data
    
    @staticmethod
    def system(self,filt,ids1,*args):
        return {'system':np.ones(ids1.shape[0],dtype=bool)}
    
    @staticmethod
    def chain_system(self,filt,*args):
        return {'system':np.ones(len(self.chain_args),dtype=bool)}
   
    @staticmethod
    def x(self,layers,ids1,ids2,coords,*args):
        fd = Distance_Functions.xdir
        cmp = self.get_particle_cm()
        d1 =  fd(None,coords[ids1],cmp)
        d2 =  fd(None,coords[ids2],cmp)
        f1 = Filters.filtLayers(layers,0.5*(d1+d2))
        return f1
    
    @staticmethod
    def y(self,layers,ids1,ids2,coords,*args):
        fd = Distance_Functions.ydir
        cmp = self.get_particle_cm()
        d1 =  fd(None,coords[ids1],cmp)
        d2 =  fd(None,coords[ids2],cmp)
        f1 = Filters.filtLayers(layers,0.5*(d1+d2))
        return f1
    
    @staticmethod
    def z(self,layers,ids1,ids2,coords,*args):
        fd = Distance_Functions.zdir
        cmp = self.get_particle_cm()
        d1 =  fd(None,coords[ids1],cmp)
        d2 =  fd(None,coords[ids2],cmp)
        f1 = Filters.filtLayers(layers,0.5*(d1+d2))
        return f1
    
    @staticmethod
    def space(self,layers,ids1,ids2,coords,*args):
        #coords = self.get_whole_coords(self.current_frame)

        d1 = self.get_distance_from_particle(ids = ids1)
        d2 = self.get_distance_from_particle(ids = ids2)
        
        f1 = Filters.filtLayers(layers,0.5*(d1+d2))
        return f1
    
    @staticmethod
    def chain_z(self,layers,chain_cm,*args):
        cmp = self.get_particle_cm()
        d = Distance_Functions.zdir(None,chain_cm,cmp)
        return Filters.filtLayers(layers, d)
    @staticmethod
    def chain_y(self,layers,chain_cm,*args):
        cmp = self.get_particle_cm()
        d = Distance_Functions.ydir(None,chain_cm,cmp)
        return Filters.filtLayers(layers, d)
    @staticmethod
    def chain_x(self,layers,chain_cm,*args):
        cmp = self.get_particle_cm()
        d = Distance_Functions.xdir(None,chain_cm,cmp)
        return Filters.filtLayers(layers, d)
    
    @staticmethod
    def filtLayers(layers,d):
        if ass.iterable(layers):
            if ass.iterable(layers[0]):
                return {dl : filt_uplow(d , dl[0], dl[1]) for dl in layers}
            else:
                return {layers: filt_uplow(d , layers[0], layers[1])}
        return dict()
    
    @staticmethod
    def filtLayers_inclucive(layers,d):
        if ass.iterable(layers):
            if ass.iterable(layers[0]):
                return {dl : filt_uplow_inclucive(d , dl[0], dl[1]) for dl in layers}
            else:
                return {layers: filt_uplow_inclucive(d , layers[0], layers[1])}
        return dict()
    
    @staticmethod
    def BondsTrainFrom(self,bondlayers,ids1,ids2,coords,*args):
        #t0 = perf_counter()
        
        ds_chains, args_train, args_tail,\
        args_loop, args_bridge = self.conformations()
        
        args_rest_train = np.concatenate( (args_tail,args_loop,args_bridge ) )
        nbonds1 = self.ids_nbondsFrom_args(ids1,args_rest_train)
        nbonds2 = self.ids_nbondsFrom_args(ids2,args_rest_train)
        nbonds = np.minimum(nbonds1,nbonds2)
        
        return Filters.filtLayers_inclucive(bondlayers,nbonds)
    
    def BondsFromEndGroups(self,bondlayers,ids1,ids2,coords,*args):
        #t0 = perf_counter()
        args_endGroups = self.get_EndGroup_args()
        
        nbonds1 = self.ids_nbondsFrom_args(ids1,args_endGroups)
        nbonds2 = self.ids_nbondsFrom_args(ids2,args_endGroups)
        nbonds = np.minimum(nbonds1,nbonds2)
        
        return Filters.filtLayers_inclucive(bondlayers,nbonds)
    
    @staticmethod
    def BondsFromTrain(self,bondlayers,ids1,ids2,coords,*args):
        #t0 = perf_counter()
        
        ds_chains, args_train, args_tail,\
        args_loop, args_bridge = self.conformations()
        
        nbonds1 = self.ids_nbondsFrom_args(ids1,args_train)
        nbonds2 = self.ids_nbondsFrom_args(ids2,args_train)
        nbonds = np.minimum(nbonds1,nbonds2)
        
        return Filters.filtLayers_inclucive(bondlayers,nbonds)

    @staticmethod
    def conformationDistribution(self,fconfs,ids1,ids2,coords,*args):
        
        
        ads_chains, args_train, args_tail,\
        args_loop, args_bridge = self.conformations()
       
        filt = dict()
        for conf,intervals in fconfs.items():
            
            args = locals()['args_'+conf]
            
            connected_chunks = self.connected_chunks(args)
            
            sizes = np.array([chunk.__len__() for chunk in connected_chunks])
            
            
            filt['{}:distr'.format(conf)] = sizes
            
            for inter in intervals:
                
                chunk_int =set()
                for chunk, size in zip(connected_chunks,sizes):
                    if inter[0]<=size<inter[1]:
                        chunk_int = chunk_int | chunk
                        
                args_chunk = np.array(list(chunk_int),dtype=int)
                f = Filters.filt_bothEndsIn(ids1, ids2, args_chunk)
                
                filt['{}:{}'.format(conf,inter)] = f
                
        return filt
    
    @staticmethod
    def conformations(self,fconfs,ids1,ids2,coords,*args):
        
        ads_chains, args_train, args_tail,\
        args_loop, args_bridge = self.conformations()
        
        all_not_free = np.concatenate((args_train,args_tail,args_loop,args_bridge))
        
        all_args = self.all_args
        args_free = all_args [ np.logical_not( np.isin(all_args, all_not_free) ) ]

        filt = dict()

        for conf in fconfs:
            conf_args = locals()['args_'+conf]
            filt[conf] = Filters.filt_bothEndsIn(ids1, ids2, conf_args)
        #ass.print_time(perf_counter()-t0,inspect.currentframe().f_code.co_name)
        return filt
    
    @staticmethod
    def filt_bothEndsIn(ids1,ids2,args):
        f1 = np.isin(ids1, args)
        f2 = np.isin(ids2, args)
        return np.logical_and(f1,f2)
    

    @staticmethod
    def combine_filts(filts,filtc):
        filt_sc = dict()
        for s,fs in filts.items():
            for c,fc in filtc.items():
                filt_sc[(s,c)] = {k:np.logical_and(fs[k],fc[k]) for k in fs.keys()}
        return filt_sc
    
    @staticmethod
    def chain_space(self,layers,chain_cm,*args):
        d = self.get_distance_from_particle(chain_cm)
        return Filters.filtLayers(layers, d)
    
    @staticmethod
    def seg_space(self,layers,seg_cm,*args):
        d = self.get_distance_from_particle(seg_cm)
        return Filters.filtLayers(layers, d)

    @staticmethod
    def chain_characteristics(self,layers,chain_cm,part_cm,coords,*args):
        raise NotImplementedError('Filter on chain characteristics is not yet implemented')
    
    @staticmethod
    def get_chain_characteristic(self,char,chain_cm,coords):
        characteristic = np.empty(len(self.chain_args),dtype=float)
        
        for i,(j,ch_cm) in enumerate(zip(self.chain_args,chain_cm)):
            #find chain  center of mass
            c_ch = coords[self.chain_args[j]]
            at_mass_ch = self.atom_mass[self.chain_args[j]]
            #ch_cm = CM(c_ch,at_mass_ch)
            
            #compute gyration tensor,rg,asphericity,acylindricity,end-to-end-distance
            Sm = np.zeros((3,3),dtype=float)
            Ree2, Rg2, k2, asph, acyl, Rgxx_plus_yy, Rgxx_plus_zz, Rgyy_plus_zz \
            = chain_characteristics_kernel(c_ch, at_mass_ch,ch_cm,Sm)
            characteristic[i] = locals()[char]
            
        return characteristic
            
    @staticmethod
    def get_ads_degree(self):
        chain_arg_keys = self.chain_args.keys()
        cads = np.empty(len(chain_arg_keys),dtype=bool)
        degree = np.empty(cads.shape[0],dtype=float)
        fd,imtr = self.get_filt_train()
       
        for i,args in enumerate(self.chain_args.values()):
            f = fd[args]
            cads[i] = f.any()
            degree[i] = np.count_nonzero(f)/f.shape[0]
        
        return degree,cads
    
    @staticmethod
    def chain_adsorption(self,ads_degree, chain_cm, *args):
        
        
        degree,cads = Filters.get_ads_degree(self)
            
        filt_ads = dict()
        
        filt_ads.update( Filters.filtLayers(ads_degree,degree) )
        filt_ads.update({'ads':cads,'free':np.logical_not(cads),'degree':degree})
        
        return filt_ads
    
class coreFunctions():
    '''
    This class contains all functions that are called during 
    looping the trajectory frames. In this way the main functions are
    more compact and we do not have to rewrite the same looping
    method for each property. This functions fill dictionaries or arrays 
    allocated from the function that they
    were called (Usuallay starting with the word 'calc_')
    '''
    def __init__():
        pass
    
    @staticmethod
    def stress_per_atom_t(self,filters,atomstress,filt_per_t):
       
        frame = self.current_frame
       
        coords = self.get_coords(frame)
        ids = np.arange(0,coords.shape[0],1,dtype=int)
        
        v = self.get_velocities(frame)
        f = self.get_forces(frame)
        m = self.atom_mass
       
        vel_contr = np.array( [v[:,i]*v[:,j]/m for i in range(3) for j in range(3)] ) 
        virial = np.array([coords[:,i]*f[:,j] for i in range(3) for j in range(3)])
        stress = vel_contr + virial
        stress = stress.reshape(stress.shape[-1::-1])
        
        key = self.get_key()
        atomstress[key] = stress
        filt_per_t[key] = Filters.calc_filters(self,filters,
                                    ids,ids,coords)
        return
    @staticmethod
    def minmax_size(self,size):
        frame = self.current_frame
        coords = self.get_coords(frame)
        size+=coords.max(axis=0) - coords.min(axis=0)
        return
    @staticmethod
    def Sq(self,q,Sq,ids=None):
        frame = self.current_frame
        coords =self.get_coords(frame)
        
        box = self.get_box(frame)
            
        if ids is not None:
            coords = coords[ids]
        n = coords.shape[0]
        npairs = int(n*(n-1)/2)
        v = np.empty((npairs,3),dtype=float)
        pair_vects(coords,box,v)
        self.v = v
        numba_Sq2(n,v,q,Sq)
        return
    
    @staticmethod
    def atomic_coordination(self,maxdist,args1,args2,coordination):
        
        frame = self.current_frame
        coords = self.get_coords(frame)
        box = self.get_box(frame)
        coords1 = coords[args1]
        coords2 = coords[args2]
        numba_coordination(coords1,coords2,box,maxdist,coordination)
        return
                
    @staticmethod
    def particle_size(self,part_size):
        frame = self.current_frame
        coords = self.get_coords(frame)
        box = self.get_box(frame)
        coords = self.translate_particle_in_box_middle(coords,box)
        part_coords = coords[self.particle_filt]
        part_s = part_coords.max(axis = 0 ) - part_coords.min(axis = 0 )
        #logger.debug('frame = {:d} --> part size = {} '.format(frame,part_s))
        part_size += part_s
        return 
    
    @staticmethod
    def theFilt(self,filters,ids1,ids2,filt_per_t):
        frame = self.current_frame
        coords = self.get_coords(frame)
        
        key = self.get_key()
        filt_per_t[key] = Filters.calc_filters(self,filters,
                                    ids1,ids2,coords)
        return
    
    @staticmethod
    def theChainFilt(self,filters,filt_per_t):
        
        frame = self.current_frame
        coords = self.get_coords(frame)
        #box = self.get_box(frame)
        
        chain_cm = self.chains_CM(coords)
        

        key = self.get_key()
        
        filt_per_t[key] = Filters.calc_filters(self,filters,
                            chain_cm, coords)
        
        return 
    @staticmethod
    def box_mean(self,box):
        frame = self.current_frame
        box+=self.get_box(frame)
        return
    
    @staticmethod
    def box_var(self,box_var,box_mean_squared):
        frame = self.current_frame
        box_var += self.get_box(frame)**2 - box_mean_squared
        return
    
    @staticmethod
    def vector_correlations(self,vec_t,filt_t,bk0,bk1,correlation):
            
        
        timekey = self.get_key()
        vec = vec_t[timekey]
        for kf in correlation:
            f = filt_t[kf][timekey]
            for k in correlation[kf]:
              #  t0 = perf_counter()
                b0 = bk0[k]
                b1 = bk1[k]
                f01 = np.logical_and(f[b0],f[b1])
                v0 = vec[b0][f01]
                v1 = vec[b1][f01]
              
                try:
                    costh = costh__parallelkernel(v0,v1)
                    correlation[kf][k].append( costh )
                except ZeroDivisionError:
                    pass
        return
    @staticmethod
    def total_dipole_moment(self,dipoles_t,q=None):
        
        frame = self.current_frame
        coords = self.get_coords(frame)
        
        key = self.get_key()
    
        pc = self.partial_charge
        
        dipoles = np.sum(pc*coords,axis=0).reshape((1,3))
        if q is not None:
            dipoles = np.sum(q*dipoles)*q
        dipoles_t[key] = dipoles  
        
        return                
    
    @staticmethod
    def segmental_dipole_moment(self,filters,ids1,ids2,
                                segmental_args,dipoles_t,filt_per_t):
        
        frame = self.current_frame
        coords = self.get_coords(frame)
        
        key = self.get_key()
        
        n = len(segmental_args)
        
        dipoles = np.empty((n,3),dtype=float)
        pc = self.atom_charge.reshape(self.natoms,1)
        
        for i,sa in enumerate(segmental_args):
            dipoles[i] = np.sum(pc[sa]*coords[sa],axis=0)
        
        dipoles_t[key] = dipoles  
        
        filt_per_t[key] = Filters.calc_filters(self,filters,
                                    ids1,ids2,coords)
        
        return
    
    @staticmethod
    def chain_dipole_moment(self,filters,
                           dipoles_t,filt_per_t):
        
        frame = self.current_frame
        coords = self.get_coords(frame)

        chain_cm = self.chains_CM(coords)
        
        key = self.get_key()
        
        n = chain_cm.shape[0]
        
        dipoles = np.empty((n,3),dtype=float)
        pc = self.atom_charge.reshape(self.natoms,1)
        for i,(j, cargs) in enumerate(self.chain_args.items()):
            dipoles[i] = np.sum(pc[cargs]*coords[cargs],axis=0)
        
        dipoles_t[key] = dipoles  
        
        filt_per_t[key] = Filters.calc_filters(self,filters,
                            chain_cm, coords)
        
        return
    
    
    @staticmethod
    def chain_dipole_moment__proj(self,filters,projvec,
                           dipoles_t,filt_per_t):
        
        frame = self.current_frame
        coords = self.get_coords(frame)

        chain_cm = self.chains_CM(coords)
        
        key = self.get_key()
        
        n = chain_cm.shape[0]
        
        dipoles = np.empty((n,3),dtype=float)
        
        pc = self.atom_charge.reshape(self.natoms,1)
        for i,(j, cargs) in enumerate(self.chain_args.items()):
            c = coords[cargs]
            dipole = np.sum(pc[cargs]*c,axis=0)
            proj = np.sum(dipole*projvec)*projvec/np.sum(projvec*projvec)**0.5
            
            dipoles[i] = proj
        
        dipoles_t[key] = dipoles  
        
        filt_per_t[key] = Filters.calc_filters(self,filters,
                            chain_cm, coords)
        
        return
    
    
    @staticmethod
    def chain_dipole_moment__endproj(self,filters,
                           dipoles_t,filt_per_t):
        
        frame = self.current_frame
        coords = self.get_coords(frame)

        chain_cm = self.chains_CM(coords)
        
        key = self.get_key()
        
        n = chain_cm.shape[0]
        
        dipoles = np.empty((n,3),dtype=float)
        
        pc = self.atom_charge.reshape(self.natoms,1)
        for i,(j, cargs) in enumerate(self.chain_args.items()):
            c = coords[cargs]
            dipole = np.sum(pc[cargs]*c,axis=0)
            ree = c[199] - c[0]
            proj = np.sum(dipole*ree)*ree/np.sum(ree*ree)**0.5
            dipoles[i] = proj
        dipoles_t[key] = dipoles  
        
        filt_per_t[key] = Filters.calc_filters(self,filters,
                            chain_cm, coords)
        
        return
    
    
    @staticmethod
    def chain_dipole_moment__contour(self,filters,ids1,ids2,
                            segmental_ids,segs_per_chain,dipoles_t,filt_per_t):
        
        frame = self.current_frame
        coords = self.get_coords(frame)
        

        chain_cm = self.chains_CM(coords)
        
        key = self.get_key()
        
        n = segmental_ids.shape[0]
        seg_dipoles = np.empty((n,3),dtype=float)
        pc = self.atom_charge.reshape(self.natoms,1)
        numba_dipoles(pc,coords,segmental_ids,seg_dipoles)
        
        v = coords[segmental_ids[:,2]] - coords[segmental_ids[:,1]]
        #self.segmental_ids = segmental_ids
        vd = np.sum(v*v,axis=1)**0.5
        vd = vd.reshape((vd.shape[0],1))
        uv = v/vd
        #self.segs_per_chain = segs_per_chain
        chain_dipoles = np.empty_like(chain_cm)
        
        for i,j in enumerate(self.chain_args):
            f = segs_per_chain[j]
            segdp = seg_dipoles[f]
            v12 = v[f]
            uv12 = uv[f]
            vd12 = vd[f]
            #print(vd14.shape,uv14.shape)
            proj = 0
            for k in range(uv12.shape[0]):
                proj+= np.abs(np.sum(v12[k]*segdp[k])/(vd12[k]**2))*uv12[k]
            chain_dipoles[i] = proj

        
        dipoles_t[key] = chain_dipoles
        
        filt_per_t[key] = Filters.calc_filters(self,filters,
                            chain_cm, coords)
        
        return
    
    
    @staticmethod
    def mass_density_profile(self,nbins,bins,
                                  rho,mass):
        frame = self.current_frame
        
        coords,box,d = self.get_frame_basics(frame)
        
        #2) Caclulate profile
        for i in range(nbins):    
            vol_bin = self.volfun(self,bins[i],bins[i+1])
            fin_bin = filt_uplow(d,bins[i],bins[i+1])
            rho[i] += np.sum(mass[fin_bin])/vol_bin
        return

    @staticmethod
    def mass_density_profile__bymol(self,nbins,bins,
                                  rho,rho_per_mol,ftt,mass):
        
        frame = self.current_frame
        coords,box,d = self.get_frame_basics(frame)
        
        #2) Caclulate profile
        for i in range(nbins):    
            vol_bin = self.volfun(self,bins[i],bins[i+1])
            fin_bin = filt_uplow(d, bins[i], bins[i+1])
            rho[i] += numba_sum(mass[fin_bin])/vol_bin
  
            for t in rho_per_mol.keys():
                ft = np.logical_and( fin_bin,ftt[t])
                rho_per_mol[t][i] += numba_sum(mass[ft])/vol_bin
        return
    
    @staticmethod
    def mass_density_profile__pertype(self,nbins,bins,
                                  rho,rho_per_atom_type,ftt,mass):
        
        frame = self.current_frame
        coords,box,d = self.get_frame_basics(frame)
        
        #2) Caclulate profile
        for i in range(nbins):    
            vol_bin = self.volfun(self,bins[i],bins[i+1])
            fin_bin = filt_uplow(d, bins[i], bins[i+1])
            mass_bin = mass[fin_bin]
            rho[i] += numba_sum(mass_bin)/vol_bin
  
            for t in rho_per_atom_type.keys():
                #ft = np.logical_and( fin_bin,ftt[t])
                ft = ftt[t][fin_bin]
                rho_per_atom_type[t][i] += numba_sum(mass_bin[ft])/vol_bin
        return
    

    
    @staticmethod
    def mass_density_profile_flux(self,nbins,bins,
                                  rho,mass,rho_mean_sq):
        frame = self.current_frame
        coords,box,d = self.get_frame_basics(frame)
      
        #2) Caclulate profile
        
        for i in range(nbins):     
            vol_bin = self.volfun(self,bins[i],bins[i+1])
            fin_bin = filt_uplow(d,bins[i],bins[i+1])
            rho[i] += (np.sum(mass[fin_bin])/vol_bin)**2-rho_mean_sq[i]
        return
    
    @staticmethod
    def number_density_profile__2side(self,nbins,bins,
                                  rho_up,rho_down):
        frame = self.current_frame
        coords = self.translated_coords(frame)
        
        cs = self.get_particle_cm(coords)
        
        dfun = getattr(Distance_Functions,self.conftype +'__2side')
        d = dfun(self,coords,cs) 
         # needed because in volfun the volume of each bin is multiplied by 2
        #2) Caclulate profile
        
        for i in range(nbins):
            vol_bin = self.volfun(self,bins[i],bins[i+1])*0.5
            fin_bin_up =   filt_uplow(d,bins[i],bins[i+1])
            fin_bin_down = filt_uplow(d,-bins[i+1],-bins[i])
            rho_up[i] += np.count_nonzero(fin_bin_up)/vol_bin
            rho_down[i] += np.count_nonzero(fin_bin_down)/vol_bin
            
        return
    
    
    @staticmethod
    def mass_density_profile__2side(self,nbins,bins,
                                  rho_up,rho_down,mass):
        frame = self.current_frame
        coords = self.translated_coords(frame)

        cs = self.get_particle_cm(coords)
        
        dfun = getattr(Distance_Functions,self.conftype +'__2side')
        d = dfun(self,coords,cs) 
         # needed because in volfun the volume of each bin is multiplied by 2
        #2) Caclulate profile
        
        for i in range(nbins):
            vol_bin = self.volfun(self,bins[i],bins[i+1])*0.5
            fin_bin_up =   filt_uplow(d,bins[i],bins[i+1])
            fin_bin_down = filt_uplow(d,-bins[i+1],-bins[i])
            rho_up[i] += np.sum(mass[fin_bin_up])/vol_bin
            rho_down[i] += np.sum(mass[fin_bin_down])/vol_bin
            
        return
    
    @staticmethod
    def number_density_profile__pertype(self,nbins,bins,
                                  rho,rho_per_atom_type,ftt):
        frame = self.current_frame
        coords,box,d = self.get_frame_basics(frame)
       
        #2) Caclulate profile
        for i in range(nbins):    
            vol_bin = self.volfun(self,bins[i],bins[i+1])
            fin_bin = filt_uplow(d, bins[i], bins[i+1])
            rho[i] += np.count_nonzero(fin_bin)/vol_bin
  
            for t in rho_per_atom_type.keys():
                ft = np.logical_and( fin_bin,ftt[t])
                rho_per_atom_type[t][i] += np.count_nonzero(ft)/vol_bin
    
    @staticmethod
    def number_density_profile(self,nbins,bins,rho):
        frame = self.current_frame
        coords,box,d = self.get_frame_basics(frame)
        
        #2) Caclulate profile
        for i in range(nbins):    
            vol_bin = self.volfun(self,bins[i],bins[i+1])
            fin_bin = filt_uplow(d,bins[i],bins[i+1])
            rho[i] += np.count_nonzero(fin_bin)/vol_bin
        return
  
    @staticmethod
    def massnumber_density_profile__conformations(self,dlayers,dens,stats):                

        #1) ads_chains, trains,tails,loops,bridges
        ads_chains, args_train, args_tail, args_loop, args_bridge = self.conformations()
        
        #check_occurances(np.concatenate((args_train,args_tail,args_bridge,args_loop)))

        coreFunctions.conformation_dens(self, dlayers, dens,ads_chains,
                                           args_train, args_tail,
                                           args_loop, args_bridge)
        
        coreFunctions.conformation_stats(stats,ads_chains, args_train, args_tail, 
                             args_loop, args_bridge)
        return
    
    @staticmethod
    def get_args_free(self,ads_chains):
        fp = self.polymer_filt
        fnotin = np.logical_not(np.isin(self.mol_ids,ads_chains))
        f = np.logical_and(fp,fnotin)
        args_free = np.where(f)[0]
        return args_free
    
    @staticmethod
    def conformation_dens(self, dlayers,dens,ads_chains,
                             args_train, args_tail, 
                             args_loop, args_bridge):
        
        coords,box,d = self.get_frame_basics(self.current_frame)
        
        args_free = coreFunctions.get_args_free(self,ads_chains)
        
        d_tail = d[args_tail]
        d_loop = d[args_loop]          
        d_bridge = d[args_bridge]
        d_free = d[args_free]
        d_train = d[args_train]
        for l,dl in enumerate(dlayers):
            args_tl = args_tail[filt_uplow(d_tail, dl[0], dl[1])]
            args_lp = args_loop[filt_uplow(d_loop, dl[0], dl[1])]
            args_br =  args_bridge[filt_uplow(d_bridge, dl[0], dl[1])]
            args_fr = args_free[filt_uplow(d_free,dl[0],dl[1])]
            args_tr = args_train[filt_uplow(d_train,dl[0],dl[1])]
            
            vol_bin = self.volfun(self,dl[0],dl[1])
            
            dens['ntrain'][l] += args_tr.shape[0]/vol_bin
            dens['ntail'][l] += args_tl.shape[0]/vol_bin
            dens['nloop'][l] += args_lp.shape[0]/vol_bin
            dens['nbridge'][l] += args_br.shape[0]/vol_bin
            dens['nfree'][l] += args_fr.shape[0]/vol_bin
            
            dens['mtrain'][l] += np.sum(self.atom_mass[args_tr])/vol_bin
            dens['mtail'][l] += np.sum(self.atom_mass[args_tl])/vol_bin
            dens['mloop'][l] += np.sum(self.atom_mass[args_lp])/vol_bin
            dens['mbridge'][l] += np.sum(self.atom_mass[args_br])/vol_bin
            dens['mfree'][l] += np.sum(self.atom_mass[args_fr])/vol_bin
        return    

    @staticmethod
    def conformation_stats(stats,ads_chains, args_train, args_tail, 
                             args_loop, args_bridge):
        stats['train'] += args_train.shape[0]
        stats['adschains'] += ads_chains.shape[0] 
        stats['looptailbridge'] += (args_loop.shape[0]+args_tail.shape[0]+args_bridge.shape[0])
        stats['tail'] += args_tail.shape[0]
        stats['loop'] += args_loop.shape[0]
        stats['bridge'] += args_bridge.shape[0]
        return 
    

    
    @staticmethod
    def P2(self,ids1,ids2,dlayers,costh,costh_unv):
        frame = self.current_frame
        #1) coords
        coords = self.get_coords(frame)
        
        #2) calc_particle_cm
        cs = self.get_particle_cm(coords)
        
        r1 = coords[ids1]; r2 = coords[ids2]
        
        rm = 0.5*(r1+r2)
        
        d = self.get_distance_from_particle(rm)
        uv = self.unit_vectorFun(self,rm,cs)
        
        costhsquare__kernel(costh,r2-r1,uv)
        
        for i,dl in enumerate(dlayers):
            filt = filt_uplow(d, dl[0], dl[1])
            costh_unv[i].extend(costh[filt])
        
        return   
    @staticmethod
    def P2_conformation(self,ids1,ids2,dlayers,
                        nvectors,costh_unv):
        frame = self.current_frame
        #1) coords
        coords = self.get_coords(frame)
        box =self.get_box(frame)

        #2) calc_particle_cm
        cs = self.get_particle_cm(coords)
        ads_chains, args_train, args_tail,\
        args_loop, args_bridge = self.conformations()
        
        args_free = coreFunctions.get_args_free(self,ads_chains)
        
        for j in costh_unv:
            args = locals()['args_'+j]
            filt_ids = Filters.filt_bothEndsIn(ids1,ids2,args)
            r1 = coords[ids1[filt_ids]]; r2 = coords[ids2[filt_ids]]
            
            
            rm = 0.5*(r1+r2)
        
            d = self.get_distance_from_particle(rm)
            uv = self.unit_vectorFun(self,rm,cs)
            
            costh = np.empty(d.shape[0],dtype=float)
            costhsquare__kernel(costh,r2-r1,uv)
        
            for i,dl in enumerate(dlayers):
                filt = filt_uplow(d, dl[0], dl[1])
                costh_unv[j][i].extend(costh[filt])
        return  
    @staticmethod
    def Rg__permol(self,rgdict):
        #Rg per molecule name
        coords = self.get_coords(self.current_frame)
        for m in self.molecules:
            rgframe = []
            for a in self.chain_args.values():
                if m != self.mol_names[a[0]]:
                    if (m == self.mol_names[a]).any():
                        raise Exception('Something is wrong with the arguments. The names do not correspond to the right ids.')
                    continue
                c = coords[a]
                mass = self.atom_mass[a]
                cm = CM(c,mass)
                r = c - cm
                dsq = np.sum(r*r,axis=1)
                rgframe.append( np.average(dsq,weights=mass))
        
            rgdict[m+'_Rg'] += np.mean(rgframe)
            rgdict[m+'_Rgstd'] += np.std(rgframe)
        return
    @staticmethod
    def Rg(self,rgdict):
        coords = self.get_coords(self.current_frame)
        rgframe = []
        for a in self.chain_args.values():
            c = coords[a]
            mass = self.atom_mass[a]
            cm = CM(c,mass)
            r = c - cm
            dsq = np.sum(r*r,axis=1)
            rgframe.append( np.average(dsq,weights=mass))
        
        rgdict['Rg'] += np.mean(rgframe)
        rgdict['Rgstd'] += np.std(rgframe)
        return
    
    @staticmethod
    def chain_characteristics(self,chain_args,dlayers,chars):
        #1) translate the system
        frame = self.current_frame
        coords = self.get_coords(frame)
        box = self.get_box(frame)
        cs = self.get_particle_cm(coords)
        
        
        for j in chain_args:
            #find chain  center of mass
            c_ch = coords[chain_args[j]]
            at_mass_ch = self.atom_mass[chain_args[j]]
            ch_cm = CM(c_ch,at_mass_ch)
            
            #compute gyration tensor,rg,asphericity,acylindricity,end-to-end-distance
            Sm = np.zeros((3,3),dtype=float)
            Ree2, Rg2, k2, asph, acyl, Rgxx_plus_yy, Rgxx_plus_zz, Rgyy_plus_zz \
            = chain_characteristics_kernel(c_ch, at_mass_ch,ch_cm,Sm)
            
            Rg = Rg2**0.5
            local_dict = locals()
            #Assign values
            d =1e16 
            for L in self.box_add(box):
                d = np.minimum(d,self.dfun(self,ch_cm.reshape(1,3),cs+L))
            for i,dl in enumerate(dlayers):
                if dl[0]< d[0] <=dl[1]:
                    for char in chars:
                        chars[char][i].append(local_dict[char])
                    break
        return
    
    @staticmethod
    def dihedrals_t(self,
                dih_ids, ids1, ids2, filters,  dihedrals_t, filt_per_t):
        
        t0 = perf_counter()
        frame = self.current_frame
        coords = self.get_coords(frame)

        dih_val = np.empty(dih_ids.shape[0],dtype=float) # alloc 
        dihedral_values_kernel(dih_ids,coords,dih_val)

        key = self.get_key()
        dihedrals_t[key] = dih_val
        
        del dih_val #deallocating for safety
        tm = perf_counter()
        
        filt_per_t[key] = Filters.calc_filters(self,filters,
                                    ids1,ids2,coords)
        tf = perf_counter()
        if frame ==1:
            logger.info('Dihedrals_as_t: Estimate time consuption --> Main: {:2.1f} %, Filters: {:2.1f} %'.format((tm-t0)*100/(tf-t0),(tf-tm)*100/(tf-t0)))
        return

    @staticmethod
    def coords_t(self,filters,ids,c_t,filt_t):
        frame = self.current_frame
        coords = self.get_coords(frame)


        key = self.get_key()
        c_t[key] = coords[ids]
        
        filt_t[key] = Filters.calc_filters(self,filters,
                                    ids,ids,coords)
        return
    

    @staticmethod
    def vects_t(self,ids1,ids2,filters,vec_t,filt_per_t):
        frame = self.current_frame
        coords = self.get_coords(frame)

        vec = coords[ids2,:] - coords[ids1,:]

        key = self.get_key()
        vec_t[key] = vec
        
        filt_per_t[key] = Filters.calc_filters(self,filters,
                                    ids1,ids2,coords)
        return     
    
    @staticmethod
    def confs_t(self,confs_t):
        frame = self.current_frame
        coords = self.get_coords(frame)
        box = self.get_box(frame)

        ads_chains, args_train, args_tail, args_loop, args_bridge =\
                                self.conformations()
        x = dict()
        ntot = args_train.shape[0] + args_tail.shape[0] +\
               args_loop.shape[0] + args_bridge.shape[0]
        for k in ['train','tail','loop','bridge']:
            args = locals()['args_'+k]
            x['x_'+k] = args.shape[0]/ntot
            x['n_'+k] = args.shape[0]
        x['x_ads_chains'] = ads_chains.shape[0]/len(self.chain_args)
        x['n_ads_chains'] = ads_chains.shape[0]
        

        key = self.get_key()
        confs_t[key] = x
        
        return

    @staticmethod
    def confs_t__length(self,confs_t):
        frame = self.current_frame
        coords = self.get_coords(frame)
       
        ads_chains, args_train, args_tail, args_loop, args_bridge =\
                                self.conformations()
        x = dict()
        for args,lab in zip([args_train, args_tail, args_loop, args_bridge],
                        ['l_train','l_tail','l_loop','l_bridge']):
            connected_chunks = self.connected_chunks(args)
            lengths = [self.length_of_connected_chunks(list(ch),coords)
                       for ch in connected_chunks ]
            
            m = np.mean(lengths)
            std = np.std(lengths)
            x[lab] = m
            x[lab+'(std)'] = std
           #x[lab+'_lenghts'] = lengths 

        key = self.get_key()
        
        confs_t[key] = x
        
        return  
    
    @staticmethod
    def confs_t__size(self,confs_t):

        ads_chains, args_train, args_tail, args_loop, args_bridge =\
                                self.conformations()
        x = dict()
        for args,lab in zip([args_train, args_tail, args_loop, args_bridge],
                        ['s_train','s_tail','s_loop','s_bridge']):
            connected_chunks = self.connected_chunks(args)
            sizes = [s.__len__() for s in connected_chunks]
            size_m = np.mean(sizes)
            size_std = np.std(sizes)
            x[lab] = size_m 
            x[lab+'(std)'] = size_std
            #x[lab+'_sizes'] = sizes

        key = self.get_key()
        confs_t[key] = x
        
        return    
    
    
    @staticmethod
    def confs_t__perchain(self,confs_t):
        frame = self.current_frame
        coords,box,d,cs = self.get_whole_frame_basics(frame)
        
        ads_chains, args_train, args_tail, args_loop, args_bridge =\
                                self.conformations()
        x = dict()
        for k in ['train','tail','loop','bridge']:
            args = locals()['args_'+k]
            x[k] =  [ np.count_nonzero( np.isin(a, args ) )/a.shape[0] 
                                      for a in self.chain_args.values() ] 
                                    

        key = self.get_key()
        confs_t[key] = x
        
        return

    @staticmethod
    def segmolCM_t(self,segmol_ids, filters, segcm_t, filt_per_t):
        
        frame = self.current_frame

        coords = self.get_coords(frame)
        
        mol_cm = self.segs_CM(coords,segmol_ids)

        key = self.get_key()
        
        cm = self.get_reference_CM(coords)

        segcm_t[key] =  mol_cm - cm
        
        filt_per_t[key] = Filters.calc_filters(self,filters,
                            mol_cm, coords)
       
        return 
    
    @staticmethod
    def chainCM_t(self,filters,vec_t,filt_per_t):
        
        frame = self.current_frame
        coords = self.get_coords(frame)
        
        chain_cm = self.chains_CM(coords)

        key = self.get_key()
        
        cm = self.get_reference_CM(coords)
        vec_t[key] =  chain_cm - cm
        
        filt_per_t[key] = Filters.calc_filters(self,filters,
                            chain_cm, coords)
       
        return 
    
    @staticmethod
    def segCM_t(self,filters,
                  ids1,ids2,segment_ids,
                  vec_t,filt_per_t):
        
        frame = self.current_frame
        coords = self.get_coords(frame)
       
        seg_cm = self.segs_CM(coords,segment_ids)
    
        key = self.get_key()
        
        cm = self.get_reference_CM(coords)
        vec_t[key] =  seg_cm -cm
        
        filt_per_t[key] = Filters.calc_filters(self,filters,
                            ids1,ids2,coords)
        
        return 
    @staticmethod
    def find_clusters_given_pair_distances(n,pd,dcut):
        k=0
        neibs = {i:set() for i in range(n)}
        for i in range(n):
            for j in range(i+1,n):
                if pd[k]<=dcut:
                    neibs[i].add(j)
                    neibs[j].add(i)
                k+=1
        
        args = set()
        
        all_args = set([i for i in range(n)])
        
        sci = {0,}
        sizes = []

        sets = []
        while len(args) != n:
            
            old_set = set()
            while (len(sci) != len(old_set)):
                old_set = sci.copy()
                for i in old_set:
                    sci = sci | neibs[i]

            sizes.append(len(sci))
            sets.append(sci)
            
            args = args | sci
            jfind = list(all_args - args)
            if len(jfind) ==0:
                continue
            
            sci = {jfind[0],}
            
        return sizes, sets
    
    @staticmethod
    def cluster_size_min(self,segmental_ids,dcut,distribution,clusters):
        key = self.get_key()
        distribution[key] = dict()     
        
        frame = self.current_frame
        coords = self.get_coords(frame)
        box = self.get_box(frame)
        
        n = len(segmental_ids)
        
        pd = np.empty(int(n*(n-1)/2),dtype=float)
        
        k=0
        for i in range(n):
            ci = coords[segmental_ids[i]]
            for j in range(i+1,n):
                cj = coords[segmental_ids[j]]
                mic = minimum_image_relative_coords(ci-cj,box)
                d = np.sum(mic*mic,axis=1)**0.5
                pd[k] = d.min()
                k+=1

        sizes,sets = coreFunctions.find_clusters_given_pair_distances(n,pd,dcut)
        
        distribution[key] = sizes
        clusters[key] = sets
        return 
    
    @staticmethod
    def cluster_size_com(self,segmental_ids,dcut,distribution):
        key = self.get_key()
        distribution[key] = dict()     
        
        frame = self.current_frame
        coords = self.get_coords(frame)
        box = self.get_box(frame)
        seg_cm = self.segs_CM(coords,segmental_ids)
        n = len(segmental_ids)
        
        pd = np.empty(int(n*(n-1)/2),dtype=float)

        pair_dists(seg_cm,box,pd)
        
        sizes,sets = coreFunctions.find_clusters_given_pair_distances(n,pd,dcut)
            
        distribution[key] = sizes
        return 
    
    @staticmethod
    def gofr_segments(self,bins,segment_ids,gofr):
        frame = self.current_frame
        coords = self.get_coords(frame)
        box = self.get_box(frame)
        seg_cm = self.segs_CM(coords,segment_ids)
        n = len(seg_cm)
        
        pd = np.empty(int(n*(n-1)/2),dtype=float)

        pair_dists(seg_cm,box,pd)
        
        pd = pd[pd<=bins.max()]
        
        numba_bin_count(pd,bins,gofr)
        
        return

    @staticmethod
    def gofr_pairs(self,ids1,ids2,bins,gofr):
        frame = self.current_frame
        coords = self.get_coords(frame)
       
        c1 = coords[ids1]
        c2 = coords[ids2]
        box = self.get_box(frame)
        
        
        relc = minimum_image_relative_coords(c2-c1,box)
        pd =np.sum(relc*relc,axis=1)**0.5
        
        pd = pd[pd<=bins.max()]
        
        numba_bin_count(pd,bins,gofr)

        return


@jit(nopython=True,fastmath=True)
def fill_property(prop,nv,i,j,value,mi,block_average):
    
    idx = j-i
    if block_average:
        try:
            prop[idx] +=  value/mi
            nv[idx] += 1.0
        except:
            pass
    else:
        prop[idx] +=  value
        nv[idx] += mi
    
    return


@jit(nopython=True,fastmath=True,parallel=True)
def Kinetics_kernel(func,func_args,
                    Prop,nv,xt,wt=None,
                    block_average=False,
                    multy_origin=True):
    
    n = xt.shape[0]
    
    if multy_origin: mo = n
    else: mo = 1
    
    for i in range(mo):
        for j in prange(i,n):
            args = func_args(i,j,xt,None,wt)
            
            value,mi = func(*args)
            fill_property(Prop,nv,i,j,value,mi,block_average)
        
    for i in prange(n):  
        Prop[i] /= nv[i]
    return 

@jit(nopython=True,fastmath=True)
def Kinetics_inner__kernel(x0,xt):
    value = 0.0 ; m = 0.0
    for i in range(x0.shape[0]):
        if x0[i]:
            if xt[i]:
                value += 1.0
            m += 1.0
    return value,m

@jit(nopython=True,fastmath=True)
def Kinetics_inner_weighted__kernel(x0,xt,w0):
    value = 0.0 ; m = 0.0
    for i in range(x0.shape[0]):
        if x0[i]:
            wi = w0[i]
            if xt[i]:
                value += wi
            m += wi
    return value,m

@jit(nopython=True,fastmath=True)
def get_zero__args(i,xt,ft,wt):
    return (xt[i],)

@jit(nopython=True,fastmath=True)
def get_zero_filt__args(i,xt,ft,wt):
    return (xt[i],ft[i])

@jit(nopython=True,fastmath=True)
def get_zero_weighted__args(i,xt,ft,wt):
    return (xt[i],  wt[i])

@jit(nopython=True,fastmath=True)
def get_zero_filt_weighted__args(i,xt,ft,wt):
    return (xt[i],ft[i],wt[i])

@jit(nopython = True,fastmath=True)
def mean__kernel(ifunc,x):
    mean =0.0 ; mi=0.0
    for i in range(x.shape[0]):
        mean += ifunc(x[i])
    mi = float(x.shape[0])
    return mean, mi

@jit(nopython = True,fastmath=True)
def mean_weighted__kernel(ifunc,x,w):
    mean =0.0 ; mi=0.0
    for i in range(x.shape[0]):
        wi = w[i]
        mean += wi*ifunc(x[i])
        mi += wi
    return mean,mi

@jit(nopython = True,fastmath=True)
def mean_filt__kernel(ifunc,x,f):
    mean =0.0 ; mi=0.0
    for i in range(x.shape[0]):
        if f[i]:
            mean+=ifunc(x[i])
            mi+=1.0
    return mean, mi

@jit(nopython = True,fastmath=True)
def mean_filt_weighted__kernel(ifunc,x,f,w):
    mean =0 ; mi=0
    for i in range(x.shape[0]):
        if f[i]:
            wi = w[i]
            mean+=wi*ifunc(x[i])
            mi+=wi
    return mean, mi


@jit(nopython = True,fastmath=True)
def secmoment__kernel(ifunc,x):
    sec =0.0 ; mi=0.0
    for i in range(x.shape[0]):
        xi = ifunc(x[i])
        sec += xi*xi
    mi = float(x.shape[0])
    return sec, mi

@jit(nopython = True,fastmath=True)
def secmoment_weighted__kernel(ifunc,x,w):
    sec =0.0 ; mi=0.0
    for i in range(x.shape[0]):
        wi = w[i]
        xi = ifunc(x[i])
        sec += wi*xi*xi
        mi += wi
    return sec,mi

@jit(nopython = True,fastmath=True)
def secmoment_filt__kernel(ifunc,x,f):
    sec =0.0 ; mi=0.0
    for i in range(x.shape[0]):
        if f[i]:
            xi = ifunc(x[i])
            sec+=xi*xi
            mi+=1.0
    return sec, mi

@jit(nopython = True,fastmath=True)
def secmoment_filt_weighted__kernel(ifunc,x,f,w):
    sec =0.0 ; mi=0.0
    for i in range(x.shape[0]):
        if f[i]:
            wi = w[i]
            xi = ifunc(x[i])
            sec+=wi*xi*xi
            mi+=w[i]
    return sec, mi


@jit(nopython=True,fastmath=True,parallel=True)
def TACF_kernel(func, func_args, inner_func,
              mean_func, secmoment_func, func_args_zero, inner_func_zero,
              Prop,nv,
              mu_val,mu_num,secmom_val,secmom_num,
              xt, ft=None, wt=None,
              block_average=False):
    
    n= xt.shape[0]
    
    for i in range(n):

        args_zero = func_args_zero(i,xt,ft,wt)
        
        mu_val[i],mu_num[i] = mean_func(inner_func_zero,*args_zero)
        
        secmom_val[i],secmom_num[i] = secmoment_func(inner_func_zero,*args_zero)
        
        for j in prange(i,n):
                
            args = func_args(i,j,xt,ft,wt)

            value,mi = func(inner_func,*args)
            
            fill_property(Prop,nv,i,j,value,mi,block_average)
            
    if block_average:
        for i in range(n):
            
            mui = mu_val[i]/mu_num[i]
            seci = secmom_val[i]/secmom_num[i] 
            
            mui_square = mui*mui  
            vari = seci - mui_square
            
            Prop[i] = (Prop[i]/nv[i] - mui_square)/vari

        return
    else:
        mu =0 ;nmu =0
        sec = 0;nsec =0
        for i in prange(n):
            mu+=mu_val[i]  ; nmu+=mu_num[i]
            sec+= secmom_val[i] ; nsec += secmom_num[i]
        mu/=nmu 
        sec/=nsec
        
        mu_sq = mu*mu
        var = sec-mu_sq
        
        for i in prange(n):
            Prop[i] = (Prop[i]/nv[i] - mu_sq)/var
    
        return 
@jit(nopython=True,fastmath=True,parallel=True )
def scalar_time_origin_average(Prop,nv,xt,every=1):
    n = xt.shape[0]
    for i in range(0,n,every):
        s1 = xt[i]
        for j in prange(i,n):
            
            s2 = xt[j]
            
            value = s1*s2
            
            fill_property(Prop,nv,i,j,value,1.0,False)
    for i in prange(n):   
        Prop[i] /= nv[i]
    return 
       
@jit(nopython=True,fastmath=True,parallel=True)
def DynamicProperty_kernel(func,func_args,inner_func,
              Prop,nv,xt,ft,wt,
              block_average,multy_origin,
              every,kernel_args):
        
    n = xt.shape[0]
    
    if multy_origin: mo = n
    else: mo = 1
    
    for i in range(0,mo,every):
        for j in prange(i,n):
            
            args = func_args(i,j,xt,ft,wt)
            args=(*args,kernel_args)
            value,mi = func(inner_func,*args)
            
            fill_property(Prop,nv,i,j,value,mi,block_average)
        
    for i in prange(n):   
        Prop[i] /= nv[i]
    return 

@jit(nopython=True,fastmath=True)
def get__args(i,j,xt,ft,wt):
    return (xt[i],xt[j])

@jit(nopython=True,fastmath=True)
def get_weighted__args(i,j,xt,ft,wt):
    return (xt[i], xt[j],  wt[i])

@jit(nopython=True,fastmath=True)
def get_const__args(i,j,xt,ft,wt):
    const = np.empty_like(ft[i])
    for k in range(const.shape[0]):
        for t in range(i,j+1):
            const[k] = const[k] and ft[t][k]
    return (xt[i],xt[j],const)

@jit(nopython=True,fastmath=True)
def get_simple__args(i,j,xt,ft,wt):
    return (xt[i],xt[j],ft[i])

@jit(nopython=True,fastmath=True)
def get_simple_weighted__args(i,j,xt,ft,wt):
    return (xt[i],xt[j],ft[i],wt[i])


@jit(nopython=True,fastmath=True)
def get_strict__args(i,j,xt,ft,wt):
    return (xt[i],xt[j],ft[i],ft[j])

@jit(nopython=True,fastmath=True)
def get_strict_weighted__args(i,j,xt,ft,wt):
    return (xt[i],xt[j],ft[i],ft[j],wt[i])


@jit(nopython=True,fastmath=True)
def get_change__args(i,j,xt,ft,wt):
    return (xt[i],xt[j],ft[i],ft[j])

@jit(nopython=True,fastmath=True)
def get_change_weighted__args(i,j,xt,ft,wt):
    return (xt[i],xt[j],ft[i],ft[j],wt[i])

@jit(nopython=True,fastmath=True)
def dynprop__kernel(inner_kernel,r1,r2,kernel_args):
    tot = 0
    mi = 0
    N = r1.shape[0]
    for i in prange(N):
        inner = inner_kernel(r1[i],r2[i],*kernel_args)
        tot+=inner
        mi+=1
    return tot,mi

@jit(nopython=True,fastmath=True)
def dynprop_simple__kernel(inner_kernel,r1,r2,ft0,kernel_args):
    tot = 0
    mi = 0
    N = r1.shape[0]
    for i in prange(N):
        if ft0[i]:
            inner = inner_kernel(r1[i],r2[i],*kernel_args)
            tot+=inner
            mi+=1
    return tot,mi

@jit(nopython=True,fastmath=True)
def dynprop_strict__kernel(inner_kernel,r1,r2,ft0,fte,kernel_args):
    tot = 0
    mi = 0
    N = r1.shape[0]
    for i in prange(N):
        if ft0[i] and fte[i]:
            inner = inner_kernel(r1[i],r2[i],*kernel_args)
            tot+=inner
            mi+=1
    return tot,mi

@jit(nopython=True,fastmath=True)
def dynprop_change__kernel(inner_kernel,r1,r2,ft0,fte,kernel_args):
    tot = 0
    mi = 0
    N = r1.shape[0]
    for i in prange(N):
        if ft0[i] and not fte[i]:
            inner = inner_kernel(r1[i],r2[i],*kernel_args)
            tot+=inner
            mi+=1
    return tot,mi

@jit(nopython=True,fastmath=True)
def dynprop_const__kernel(inner_kernel,r1,r2,const,kernel_args):
    tot = 0
    mi = 0
    N = r1.shape[0]

    for i in prange(N):
        if const[i]:
            inner = inner_kernel(r1[i],r2[i],*kernel_args)
            tot+=inner
            mi+=1
    return tot,mi

@jit(nopython=True,fastmath=True)
def dynprop_weighted__kernel(inner_kernel,r1,r2,w,kernel_args):
    tot = 0
    mi = 0
    N = r1.shape[0]
    for i in prange(N):
        inner = inner_kernel(r1[i],r2[i],*kernel_args)
        tot+=w[i]*inner
        mi+=w[i]
    return tot,mi

@jit(nopython=True,fastmath=True)
def dynprop_simple_weighted__kernel(inner_kernel,r1,r2,ft0,w,kernel_args):
    tot = 0
    mi = 0
    N = r1.shape[0]
    for i in prange(N):
        if ft0[i]:
            inner = inner_kernel(r1[i],r2[i],*kernel_args)
            tot+=w[i]*inner
            mi+=w[i]
    return tot,mi

@jit(nopython=True,fastmath=True)
def dynprop_strict_weighted__kernel(inner_kernel,r1,r2,ft0,fte,w,kernel_args):
    tot = 0
    mi = 0
    N = r1.shape[0]
    for i in prange(N):
        if ft0[i] and fte[i]:
            inner = inner_kernel(r1[i],r2[i],*kernel_args)
            tot+=w[i]*inner
            mi+=w[i]
    return tot,mi

@jit(nopython=True,fastmath=True)
def dynprop_change_weighted__kernel(inner_kernel,r1,r2,ft0,fte,w,kernel_args):
    tot = 0
    mi = 0
    N = r1.shape[0]
    for i in prange(N):
        if ft0[i] and not fte[i]:
            inner = inner_kernel(r1[i],r2[i],*kernel_args)
            tot+=w[i]*inner
            mi+=w[i]
    return tot,mi




@jit(nopython=True,parallel=True)
def numba_bin_count(d,bins,counter):
    for j in prange(bins.shape[0]-1):
        for i in range(d.shape[0]):
            if bins[j]<d[i] and d[i] <=bins[j+1]:
                counter[j] +=1
    return

@jit(nopython=True,fastmath=True)
def fcos_kernel(x):
    return np.cos(x)

@jit(nopython=True,fastmath=True)
def fsin_kernel(x):
    return np.sin(x)

@jit(nopython=True,fastmath=True)
def cosCorrelation_kernel(x1,x2):
    
    c1= np.cos(x1)
    c2 = np.cos(x2)
    return c1*c2

@jit(nopython=True,fastmath=True)
def sinCorrelation_kernel(x1,x2):
    
    s1= np.sin(x1)
    s2 = np.sin(x2)
    return s1*s2

@jit(nopython=True,fastmath=True)
def mult_kernel(r1,r2,*args):
    return r1*r2

@jit(nopython=True,fastmath=True)
def cos2th_kernel(r1,r2,*args):
    costh = costh_kernel(r1,r2)
    return costh*costh

@jit(nopython=True,fastmath=True)
def costh_kernel(r1,r2,*args):
    costh=0 ; rn1 =0 ;rn2=0
    n = r1.shape[0]
    for j in range(n):
        costh += r1[j]*r2[j]
        rn1 += r1[j]*r1[j]
        rn2 += r2[j]*r2[j]
    rn1 = rn1**0.5
    rn2 = rn2**0.5
    costh/=rn1*rn2
    return costh

@jit(nopython=True,fastmath=True)
def Fs_kernel(r1,r2,q):
    ri =  q*(r2[0] +r2[1] + r2[2] - r1[0] - r1[1] -r1[2])
    return np.cos(ri)

@jit(nopython=True,fastmath=True)
def norm_square_kernel(r1,r2,*args):
    
    nm = 0
    for i in range(r1.shape[0]):
        ri = r2[i] - r1[i]
        nm+= ri*ri
    return nm

@jit(nopython=True,fastmath=True,parallel=True)
def costh__parallelkernel(r1,r2,*args):
    tot = 0
    for i in prange(r1.shape[0]):
        tot += costh_kernel(r1[i],r2[i])
    ave = tot/float(r1.shape[0])
    return ave

@jit(nopython=True,fastmath=True)
def costhsquare__kernel(costh,r1,r2,*args):
    for i in range(r1.shape[0]):
        costh[i] = cos2th_kernel(r1[i],r2[i])

@jit(nopython=True,fastmath=True)
def costhmean__kernel(r1,r2,*args):
    mean =0
    for i in range(r1.shape[0]):
        costh = costh_kernel(r1[i],r2[i])
        mean+=costh    
    ave = mean/r1.shape[0]
    return ave



@jit(nopython=True,fastmath=True)
def unwrap_coords_kernel(unc,k0,k1,b2,n,dim,box):
    for j in dim:
        for i in n:
            if unc[k0[i],j] - unc[k1[i],j] > b2[j]:
                unc[k1[i],j] += box[j]
            elif unc[k1[i],j] - unc[k0[i],j] > b2[j]:
                unc[k1[i],j] -= box[j]
    return unc

@jit(nopython=True,fastmath=True)
def end_to_end_distance(coords):
    rel = coords[0]-coords[coords.shape[0]-1]
    Ree2 = np.dot(rel,rel)
    return Ree2
@jit(nopython=True,fastmath=True)
def chain_characteristics_kernel(coords,mass,ch_cm,Gyt):
    rel = coords[0]-coords[coords.shape[0]-1]
    Ree2 = np.dot(rel,rel)
    rccm = coords-ch_cm   
    for i in range(rccm.shape[0]):
        Gyt+=mass[i]*np.outer(rccm[i],rccm[i])
    Gyt/=np.sum(mass)
    S =  np.linalg.eigvals(Gyt)
    S =-np.sort(-S) # sorting in desceanding order
    Rg2 =np.sum(S)
    #Shat = Gyt-Rg2*np.identity(3)/3
    asph = S[0] -0.5*(S[1]+S[2])
    acyl = S[1]-S[2]
    k2 = (asph**2 + 0.75*acyl**2)/Rg2**2
    
    Rgxx_plus_yy = Gyt[0][0] + Gyt[1][1]
    Rgxx_plus_zz = Gyt[0][0] + Gyt[2][2]
    Rgyy_plus_zz = Gyt[1][1] + Gyt[2][2]
    
    return Ree2, Rg2,k2,asph,acyl, Rgxx_plus_yy, Rgxx_plus_zz, Rgyy_plus_zz
   
@jit(nopython=True, fastmath=True,parallel=True)
def dihedral_values_kernel(dih_ids,coords,dih_val):
    for i in prange(dih_ids.shape[0]):
        r0 = coords[dih_ids[i,0]]
        r1 = coords[dih_ids[i,1]]  
        r2 = coords[dih_ids[i,2]]
        r3 = coords[dih_ids[i,3]]
        dih_val[i] = calc_dihedral(r0,r1,r2,r3)
    return 

@jit(nopython=True,fastmath=True)
def numba_sum_wfilt(x,filt):
    s = 0
    for i in range(x.shape[0]):
        if filt[i]: 
            s += x[i]
    return s

@jit(nopython=True,fastmath=True,parallel=True)
def numba_parallel_sum_wfilt(x,filt):
    s = 0
    for i in prange(x.shape[0]):
        if filt[i]: 
            s += x[i]
    return s

@jit(nopython=True,fastmath=True)
def numba_sum(x):
    s =0 
    for i in range(x.shape[0]):
        s+=x[i]
    return s

@jit(nopython=True,fastmath=True,parallel=True)
def numba_parallel_sum(x):
    s =0 
    for i in prange(x.shape[0]):
        s+=x[i]
    return s


class Analysis_Crystals(Analysis):
    def __init__(self,molecular_system,memory_demanding=True,topol=True):
        super().__init__(molecular_system,memory_demanding)
        self.topol =True
        return
    

def center_of_bins(bins):
    nbins = bins.shape[0] - 1
    r = [0.5*(bins[i]+bins[i+1]) for i in range(nbins)]
    return np.array(r)

def filt_uplow(x,yl,yup):
    return np.logical_and(np.greater(x,yl),np.less_equal(x,yup))

def filt_uplow_inclucive(x,yl,yup):
    return np.logical_and(np.greater_equal(x,yl),np.less_equal(x,yup))

def binning(x,bins):
    nbins = bins.shape[0]-1
    n_in_bins = np.empty(nbins,dtype=int) 
    for i in np.arange(0,nbins,1,dtype=int):
        filt = filt_uplow(x,bins[i],bins[i+1])
        n_in_bins[i] = np.count_nonzero(filt)
    return n_in_bins


@jit(nopython=True,fastmath=True)
def minimum_image_relative_coords(relative_coords,box):
    imaged_rel_coords = relative_coords.copy()
    for i in range(relative_coords.shape[0]):
        for j in range(3):
            if relative_coords[i][j] > 0.5*box[j]:
                imaged_rel_coords[i][j] -= box[j]
            elif relative_coords[i][j] < -0.5*box[j]:
                imaged_rel_coords[i][j] += box[j]  
    return imaged_rel_coords


@jit(nopython=True,fastmath=True,parallel=True)
def minimum_image_distance(coords,cref,box):
        r = coords - cref
       
        for j in range(3):
            b = box[j]
            b2 = b/2
            fm = r[:,j] < - b2
            fp = r[:,j] >   b2
            r[:,j][fm] += b
            r[:,j][fp] -= b
        d = np.zeros(r.shape[0],dtype=float)
        for i in prange(r.shape[0]):
            for j in range(3):
                x = r[i,j]
                d[i] += x*x
            d[i] = np.sqrt(d[i])
        
        return d

@jit(nopython=True,fastmath=True,parallel=True)
def minimum_image_distance_coords(coords,cref,box):
        r = coords - cref
        imag_coords = coords.copy()
        for j in range(3):
            b = box[j]
            b2 = b/2
            fm = r[:,j] < - b2
            fp = r[:,j] >   b2
                        
            r[:,j][fm] += b
            imag_coords[:,j][fm] +=b
            
            r[:,j][fp] -= b
            imag_coords[:,j][fp] -= b
        d = np.zeros(r.shape[0],dtype=float)
        for i in prange(r.shape[0]):
            for j in range(3):
                x = r[i,j]
                d[i] += x*x
            d[i] = np.sqrt(d[i])
        
        return d,imag_coords

@jit(nopython=True,fastmath=True,parallel=True)
def numba_Sq2(nc,v,q,Sq):
    nq = q.shape[0]
    npairs = v.shape[0]
    nc = float(nc)
    s = np.empty_like(Sq)
    qm = -q
    for i in range(nq):
        s[i] = 0.0
    for j in prange(npairs):
        s += np.cos(np.dot(qm,v[j]))
    Sq += 2*s/nc

    return


@jit(nopython=True,fastmath=True,parallel=True)
def pair_vects(coords,box,v):
    n = coords.shape[0]
    coords = implement_pbc(coords,box)
    for i in prange(n):
        rel_coords = coords[i] - coords[i+1:]
        rc = minimum_image_relative_coords(rel_coords,box)
        idx_i = i*n
        for k in range(0,i+1):
            idx_i-=k
        for j in range(rc.shape[0]):
            v[idx_i+j] = rc[j]
    return



@jit(nopython=True,fastmath=True,parallel=True)
def pair_dists(coords,box,dists):
    n = coords.shape[0]
    for i in prange(n):
        rel_coords = coords[i] - coords[i+1:]
        rc = minimum_image_relative_coords(rel_coords,box)
        dist = np.sum(rc*rc,axis=1)**0.5
        idx_i = i*n
        for k in range(0,i+1):
            idx_i-=k
        for j in range(rc.shape[0]):
            dists[idx_i+j] = dist[j]
    return

@jit(nopython=True,fastmath=True,parallel=True)
def numba_coordination(coords1,coords2,box,maxdist,coordination):
    n1 = coords1.shape[0]
    n2 = coords2.shape[0] 
    for i in prange(n1):
        rel_coords = coords1[i] - coords2
        rc = minimum_image_relative_coords(rel_coords,box)
        dist = np.sqrt(np.sum(rc*rc,axis=1))
        for j in range(n2):
            if dist[j]<maxdist:
                coordination[i]+=1
                
    return

@jit(nopython=True,fastmath=True,parallel=True)
def pair_dists_general(coords1,coords2,box,dists):
    n1 = coords1.shape[0]
    n2 = coords2.shape[0] 
    for i in prange(n1):
        rel_coords = coords1[i] - coords2
        #rc=rel_coords
        rc = minimum_image_relative_coords(rel_coords,box)
        dists[i*n2:(i+1)*n2] = np.sum(rc*rc,axis=1)**0.5
        #for j in range(n2):
         #   dists[i*n2+j] = dist[j]
    return

@jit(nopython=True,fastmath=True)
def CM(coords,mass):
    cm = np.sum(mass*coords.T,axis=1)/mass.sum()
    return cm

@jit(nopython=True,fastmath=True)
def implement_pbc(coords,boxsize):
    cn = coords%boxsize
    return cn

@jit(nopython=True,fastmath=True)
def square_diff(x,c):
    return ((x-c)/c)**2
@jit(nopython=True,fastmath=True)
def norm2(r):
    x =0
    for i in range(r.shape[0]): 
        x += r[i]*r[i]
    return x**0.5
@jit(nopython=True,fastmath=True)
def norm2_axis1(r):
    return (r*r).sum(axis=1)**0.5


@jit(nopython=True,fastmath=True)
def calc_dist(r1,r2):
    r = r2 - r1
    d = np.sqrt(np.dot(r,r))
    return d

@jit(nopython=True,fastmath=True)
def calc_angle(r1,r2,r3):
    d1 = r1 -r2 ; d2 = r3-r2
    nd1 = np.sqrt(np.dot(d1,d1))
    nd2 = np.sqrt(np.dot(d2,d2))
    cos_th = np.dot(d1,d2)/(nd1*nd2)
    return np.arccos(cos_th)

@jit(nopython=True,fastmath=True)
def calc_dihedral(r1,r2,r3,r4):
    d1 = r2-r1
    d2 = r3-r2
    d3 = r4-r3
    c1 = np.cross(d1,d2)
    c2 = np.cross(d2,d3)
    n1 = c1/np.sqrt(np.dot(c1,c1))
    n2 = c2/np.sqrt(np.dot(c2,c2))
    m1= np.cross(n1,d2/np.sqrt(np.dot(d2,d2)))
    x= np.dot(n1,n2)
    y= np.dot(m1,n2)
    dihedral = np.arctan2(y, x)
    return dihedral





class maps:
    charge_map = {'CD':-0.266,'C':0.154,'CE':0.164,
                  'hC':-0.01,'hCD':0.132,'hCE':-0.01
                      }
        
    elements_mass = {'H' : 1.008,'He' : 4.003, 'Li' : 6.941, 'Be' : 9.012,\
                 'B' : 10.811, 'C' : 12.011, 'N' : 14.007, 'O' : 15.999,\
                 'F' : 18.998, 'Ne' : 20.180, 'Na' : 22.990, 'Mg' : 24.305,\
                 'Al' : 26.982, 'Si' : 28.086, 'P' : 30.974, 'S' : 32.066,\
                 'Cl' : 35.453, 'Ar' : 39.948, 'K' : 39.098, 'Ca' : 40.078,\
                 'Sc' : 44.956, 'Ti' : 47.867, 'V' : 50.942, 'Cr' : 51.996,\
                 'Mn' : 54.938, 'Fe' : 55.845, 'Co' : 58.933, 'Ni' : 58.693,\
                 'Cu' : 63.546, 'Zn' : 65.38, 'Ga' : 69.723, 'Ge' : 72.631,\
                 'As' : 74.922, 'Se' : 78.971, 'Br' : 79.904, 'Kr' : 84.798,\
                 'Rb' : 84.468, 'Sr' : 87.62, 'Y' : 88.906, 'Zr' : 91.224,\
                 'Nb' : 92.906, 'Mo' : 95.95, 'Tc' : 98.907, 'Ru' : 101.07,\
                 'Rh' : 102.906, 'Pd' : 106.42, 'Ag' : 107.868, 'Cd' : 112.414,\
                 'In' : 114.818, 'Sn' : 118.711, 'Sb' : 121.760, 'Te' : 126.7,\
                 'I' : 126.904, 'Xe' : 131.294, 'Cs' : 132.905, 'Ba' : 137.328,\
                 'La' : 138.905, 'Ce' : 140.116, 'Pr' : 140.908, 'Nd' : 144.243,\
                 'Pm' : 144.913, 'Sm' : 150.36, 'Eu' : 151.964, 'Gd' : 157.25,\
                 'Tb' : 158.925, 'Dy': 162.500, 'Ho' : 164.930, 'Er' : 167.259,\
                 'Tm' : 168.934, 'Yb' : 173.055, 'Lu' : 174.967, 'Hf' : 178.49,\
                 'Ta' : 180.948, 'W' : 183.84, 'Re' : 186.207, 'Os' : 190.23,\
                 'Ir' : 192.217, 'Pt' : 195.085, 'Au' : 196.967, 'Hg' : 200.592,\
                 'Tl' : 204.383, 'Pb' : 207.2, 'Bi' : 208.980, 'Po' : 208.982,\
                 'At' : 209.987, 'Rn' : 222.081, 'Fr' : 223.020, 'Ra' : 226.025,\
                 'Ac' : 227.028, 'Th' : 232.038, 'Pa' : 231.036, 'U' : 238.029,\
                 'Np' : 237, 'Pu' : 244, 'Am' : 243, 'Cm' : 247, 'Bk' : 247,\
                 'Ct' : 251, 'Es' : 252, 'Fm' : 257, 'Md' : 258, 'No' : 259,\
                 'Lr' : 262, 'Rf' : 261, 'Db' : 262, 'Sg' : 266, 'Bh' : 264,\
                 'Hs' : 269, 'Mt' : 268, 'Ds' : 271, 'Rg' : 272, 'Cn' : 285,\
                 'Nh' : 284, 'Fl' : 289, 'Mc' : 288, 'Lv' : 292, 'Ts' : 294,\
                 'Og' : 294}


@jit(nopython=True,fastmath=True)
def numba_dipoles(pc,coords,segargs,dipoles):
    n = segargs.shape[0]
    for i in prange(n):
        cargs = segargs[i]
        dipoles[i] = np.sum(pc[cargs]*coords[cargs],axis=0)
    return
@jit(nopython=True,fastmath=True,parallel=True)
def numba_isin(x1,x2,f):
    for i in prange(x1.shape[0]):
        for x in x2:
            if x1[i] == x: 
                f[i] = True
    return
@jit(nopython=True,fastmath=True,parallel=True)
def numba_CM(coords,ids,mass,cm):
    for i in prange(ids.shape[0]):
        ji = ids[i]
        cm[i] = CM(coords[ji],mass[ji])
    return

@jit(nopython=True,fastmath=True,parallel=True)
def numba_elementwise_minimum(x1,x2):
    xmin = np.empty_like(x1)
    for i in prange(x1.shape[0]):
        if x1[i]<x2[i]:
            xmin[i] = x1[i]
        else:
            xmin[i] = x2[i]
    return xmin

