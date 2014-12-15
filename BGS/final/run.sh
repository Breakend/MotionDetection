#!/bin/bash

make

# ./main NUM_THREADS USE_OPEN_CV_BLUR MOTION_COMP


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

exit 0

for (( ocvblur = 0; ocvblur<=1; ocvblur+=1 )); do
	for (( motion = 0; motion<=1; motion+=1 )); do
		echo "ocvblur = $ocvblur   motion = $motion"
		for (( threads = 0; threads<=512; threads*=2 )); do
			./main $threads $ocvblur $motion
		done
	done
done


echo "OpenCV blurring, no motion compensation"
echo "num_threads t_exec t_blur t_mtnc t_dsgm t_serl"
./main 0 1 0 # Serial version
./main 1 1 0
./main 2 1 0
./main 4 1 0
./main 8 1 0
./main 16 1 0
./main 32 1 0
./main 64 1 0
./main 128 1 0
./main 256 1 0
./main 512 1 0

echo "OpenCV blurring, motion compensation"
echo "num_threads t_exec t_blur t_mtnc t_dsgm t_serl"
./main 0 1 1 # Serial version
./main 1 1 1
./main 2 1 1
./main 4 1 1
./main 8 1 1
./main 16 1 1
./main 32 1 1
./main 64 1 1
./main 128 1 1
./main 256 1 1
./main 512 1 1

echo "Original blurring, no motion compensation"
echo "num_threads t_exec t_blur t_mtnc t_dsgm t_serl"
./main 0 0 0 # Serial version
./main 1 0 0
./main 2 0 0
./main 4 0 0
./main 8 0 0
./main 16 0 0
./main 32 0 0
./main 64 0 0
./main 128 0 0
./main 256 0 0
./main 512 0 0

echo "Original blurring, motion compensation"
echo "num_threads t_exec t_blur t_mtnc t_dsgm t_serl"
./main 0 0 1 # Serial version
./main 1 0 1
./main 2 0 1
./main 4 0 1
./main 8 0 1
./main 16 0 1
./main 32 0 1
./main 64 0 1
./main 128 0 1
./main 256 0 1
./main 512 0 1

echo 'Done!'

exit 0