#!/bin/bash
rm tp.tpr
gmx grompp -c ST2.gro  -f steep.mdp -p ST2.top -o tp.tpr -maxwarn 3
gmx mdrun -s tp.tpr
cp confout.gro minimization.gro
#rm tp.tpr
#gmx grompp -c confout.gro  -f gromp.mdp -p ST2.top -o tp.tpr
#gmx mdrun -s tp.tpr
