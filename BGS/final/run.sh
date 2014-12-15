#!/bin/bash

make

# ./main NUM_THREADS USE_OPEN_CV_BLUR MOTION_COMP
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

echo 'Done!'

exit 0