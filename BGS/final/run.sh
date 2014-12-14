#!/bin/bash

make

# ./serial_main NUM_THREADS
./main 1
./main 2
./main 4
./main 8
./main 16
./main 32
./main 64
./main 128
./main 256

echo 'Done!'

exit 0