#!/bin/bash
#SBATCH --job-name=densPr
#SBATCH --nodes=1
#SBATCH --partition=a100
#SBATCH --ntasks-per-node=12
#SBATCH --time=23:59:59
#SBATCH --error=error
#SBATCH --output=output
#SBATCH --gres=gpu:1

module purge
#module load GROMACS/2021.5-foss-2021b-CUDA-11.4.1
#module load GROMACS/2023.3-foss-2022a-CUDA-11.7.0
module load matplotlib/3.4.3-foss-2021b
module load numba/0.54.1-foss-2021b
nproc=$((SLURM_NNODES*SLURM_NTASKS_PER_NODE))

density_profile() {
    local acc="$1"
    local conc="$2"
    local coupla="$3"
    local num="$4"
    
    if [ "$num" == "bare" ]; then
	name=$acc-$conc-$num
        sil="SIL"
    else
        name=$acc-$conc-$coupla-$num
    	sil="S${coupla:0:1}${num}"
    fi
    path="../merged/$name/production/"

    type_coords="raj"
    trajf="$path/t$type_coords.trr"
    itp_files="['${path}/cPI30.itp', '${path}/$acc.itp','$path/$sil.itp']"
    gro_file="${path}/confout.gro"
    property="density_profile"
    property_args="(0.02,5.5)"
    option=bymol
    property_kwargs="dict(offset=1.0,option='$option',types=None)"

    results_file="prodpickles/$name.pickle"

    python make_analysis_tool.py -f $trajf -c "$itp_files" -t "$gro_file" -p "$property" -pargs "$property_args" -pkwargs "$property_kwargs" \
	    			  -rf "$results_file" -ct zdir -particle $sil -polymer cPI30  -kf 10 9000 
}

#density_profile cPI30 "" "" bare
#for acc in CBS DBTH DPG
#for acc in DPGi
#for acc in TESPD NXT MPTES
#do
#    	density_profile $acc 3 whatever bare
#done

#for acc in CBS DBTH DPG DPGi
#do
#   for coupla in MPTES NXT TESPD 
#   do 
#  	for num in 84  166	
#	do	
#        density_profile $acc 3 $coupla $num
#	done
#   done
#done

for coupla in MPTES NXT TESPD 
do
     density_profile cPI30  "" $coupla 84 
done
