#!/bin/bash
nproc=12

rm tp.tpr
gmx grompp -c wacc.gro  -f steep.mdp -p topol.top -o tp.tpr 
gmx mdrun -s tp.tpr -ntmpi ${nproc} -ntomp 1  
rm tp.tpr
gmx grompp -c confout.gro -f gromp.mdp -p topol.top -o tp.tpr 
gmx mdrun -s tp.tpr -ntmpi 1  -ntomp ${nproc}

