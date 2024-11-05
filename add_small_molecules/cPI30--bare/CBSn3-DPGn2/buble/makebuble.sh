#!/bin/bash

filename=topol.top
search_variable=SUDO
nproc=32

# Check if the file exists
if [ ! -f "$filename" ]; then
    echo "Error: File $filename not found."
    exit 1
fi
rm tp.tpr


#cp piS.gro confout.gro
#for sigma in 0.1 0.2 0.4 0.7 1.2 1.9  2.9   4.6  7 12 
for sigma in 12 16 20 24 28
do


# Use awk to find and replace the specified line
awk -v var="$search_variable" -v new_val="$sigma" '$1 == var {$6 = new_val}1' "$filename" > temp_file && mv temp_file "$filename"

echo "Replacement complete. Check the modified file: $filename"

gmx grompp -c confout.gro -r piS.gro -f steep.mdp -p $filename -o tp.tpr 
gmx mdrun -s tp.tpr -ntmpi ${nproc} -ntomp 1  
rm tp.tpr
cp confout.gro sigma${sigma}steep.gro

gmx grompp -c confout.gro -r piS.gro -f buble.mdp -p $filename -o tp.tpr  -maxwarn 2
gmx mdrun -s tp.tpr -ntmpi ${nproc} -ntomp 1  
cp confout.gro sigma${sigma}.gro
rm \#*
rm step*
done
printf "0" | gmx trjconv -s tp.tpr -f state.cpt -o confoutwh.gro
