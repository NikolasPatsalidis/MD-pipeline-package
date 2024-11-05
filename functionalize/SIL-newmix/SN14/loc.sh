#!/bin/bash
rm tp.tpr
gmx grompp -c confout.gro  -f gromp.mdp -p *.top -o tp.tpr  -maxwarn 3
gmx mdrun -s tp.tpr  -ntmpi 12 -ntomp 1
