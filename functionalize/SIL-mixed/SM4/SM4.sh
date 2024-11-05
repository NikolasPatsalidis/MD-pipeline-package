#!/bin/bash
rm tp.tpr
gmx grompp -c SM4.gro  -f steep.mdp -p SM4.top -o tp.tpr -maxwarn 3
gmx mdrun -s tp.tpr
cp confout.gro minimization.gro
#rm tp.tpr
#gmx grompp -c confout.gro  -f gromp.mdp -p SM4.top -o tp.tpr
#gmx mdrun -s tp.tpr
