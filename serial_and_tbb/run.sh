#!/bin/bash

#******************************************************************************
#	
# 	File: run.sh
# 	Purpose: make program then run all four variations of the program to 
#			 generate speedup and histogram diagrams
#	
#******************************************************************************

make

# ./main NUM_THREADS USE_OPEN_CV_BLUR MOTION_COMP

# Generate speedup graphs for all variations
for (( ocvblur = 1; ocvblur>=0; ocvblur-=1 )); do
	for (( motion = 0; motion<=1; motion+=1 )); do
		echo "ocvblur = $ocvblur   motion = $motion"
		echo "num_threads speedup"
		IFS=$' ' read num_threads serial_exec <<< $(./main 0 $ocvblur $motion)
		for (( threads = 1; threads<=512; threads*=2 )); do
			IFS=$' ' read num_threads par_t_exec <<< $(./main $threads $ocvblur $motion)
			speedup=$(bc <<< "scale = 10; $serial_exec / $par_t_exec")
			echo "$threads $speedup"
		done
	done
done

# Generate histograms for all variations
for (( ocvblur = 0; ocvblur<=1; ocvblur+=1 )); do
	for (( motion = 0; motion<=1; motion+=1 )); do
		echo "ocvblur = $ocvblur   motion = $motion"
		for (( threads = 0; threads<=512; threads*=2 )); do
			./main $threads $ocvblur $motion
		done
	done
done

echo 'Done!'

exit 0