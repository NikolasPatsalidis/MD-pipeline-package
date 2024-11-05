#!bin/bash

#find SIL-MPTESonly -type d  | grep 'S' | while read -r dir; do (cd "$dir" && bash loc.sh); done
#find SIL-NXTonly -type d  | grep 'S' | while read -r dir; do (cd "$dir" && bash loc.sh); done
find SIL-newmix -type d  | grep 'S' | while read -r dir; do (cd "$dir" && bash loc.sh); done

