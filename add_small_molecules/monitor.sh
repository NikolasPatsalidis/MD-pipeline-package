#!/bin/bash

# Find all md.log files and iterate over them
find . -maxdepth 4 -type f -name "md.log"  -print0 | while IFS= read -r -d $'\0' file; do
    # Print the base directory

    base_dir=$(dirname "$file")
    if [[ $base_dir == *"production"* ]]; then
    	echo "Base directory: $base_dir"
    
   	 # Print the 12th line from the end using tail
   	 tail -n 12 "$file" | head -n 1
    
    	# Add a separator for better readability
    	echo "------------------------------------"
   fi
done
