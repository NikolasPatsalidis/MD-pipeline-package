#!/bin/bash

density_profile() {
    local acc="$1"
    local conc="$2"
    local coupla="$3"
    local num="$4"

    path="../$acc-$conc-$coupla-$num"

    type_coords="raj"
    sil="S${coupla:0:1}${num}"
    trajf="$path/t$type_coords.trr"
    itp_files="['${path}/cPI30.itp', '${path}/$acc.itp','$path/$sil.itp']"
    gro_file="${path}/confout.gro"

    property="density_profile"
    property_args="(0.02,6)"
    property_kwargs="dict(offset=1.0,option='bymol')"

    results_file="Rpickles/${path:3}.pickle"

    python make_analysis_tool.py -f $trajf -c "$itp_files" -t "$gro_file" -p "$property" -pargs "$property_args" -pkwargs "$property_kwargs" \
	    			  -rf "$results_file" -ct zdir -particle $sil -polymer cPI30  -mem 1
}



for acc in CBS DBTH DPG 
do
   for coupla in MPTES NXT TESPD 
   do 
  	for num in 84  166	
	do	
    	    density_profile $acc 3 $coupla $num
	done
   done
done
