#!/bin/bash
rm tp.tpr
gmx grompp -c SN3.gro  -f steep.mdp -p SN3.top -o tp.tpr -maxwarn 3
gmx mdrun -s tp.tpr -ntmpi 12 -ntomp 1
cp confout.gro minimization.gro
#rm tp.tpr
#gmx grompp -c confout.gro  -f gromp.mdp -p SN3.top -o tp.tpr
#gmx mdrun -s tp.tpr
