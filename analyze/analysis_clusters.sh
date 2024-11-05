#!/bin/bash
cluster_size_analysis() {
    local acc="$1"
    local conc="$2"

    path="PI30mer-$acc-$conc/prodcorr"
    type_coords="wh"

    trajf="$path/t$type_coords.trr"
    itp_files="['${path}/cPI30.itp', '${path}/$acc.itp']"
    gro_file="${path}/c0.gro"

    property="cluster_dynamics"
    property_args="('$acc', 0.45)"
    property_kwargs="dict()"

    results_file="Rpickles/${acc}-${conc}-${type_coords}.pickle"

    python3 make_analysis_tool.py -f "$trajf" -c "$itp_files" -t "$gro_file" -p "$property" -pargs "$property_args" -pkwargs "$property_kwargs" -rf "$results_file"
}

#combinations="CBS-2,CBS-3,CBS-1.1,DPG-2,DPG-3,DPG-1.3,DPGi-2,DPGi-3,DPGi-1.3,DBTH-2,DBTH-3,DBTH-4"
#combinations="DPG-3,DPGi-3"
IFS=',' read -ra combs <<< "$combinations"

for comb in "${combs[@]}"; do
    IFS='-' read -ra sub_combs <<< "$comb"

    # Now you can access the sub-combs
    acc="${sub_combs[0]}"
    conc="${sub_combs[1]}"
    cluster_size_analysis $acc $conc
done
