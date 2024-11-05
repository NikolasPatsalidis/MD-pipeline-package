#!/bin/bash
rm tp.tpr
gmx grompp -c SM15.gro  -f steep.mdp -p SM15.top -o tp.tpr -maxwarn 3
gmx mdrun -s tp.tpr -ntmpi 8 -ntomp 1
cp confout.gro minimization.gro
#rm tp.tpr
#gmx grompp -c confout.gro  -f gromp.mdp -p SM15.top -o tp.tpr
#gmx mdrun -s tp.tpr
