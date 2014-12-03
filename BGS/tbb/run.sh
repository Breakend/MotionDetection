#!/bin/bash

#time ./run.sh

make

# ./tbb_main NUM_THREADS
./tbb_main 1
./tbb_main 2
./tbb_main 4
./tbb_main 8
#./tbb_main 16
#./tbb_main 32
#./tbb_main 64
#./tbb_main 128
#./tbb_main 256

# cd ./p2p-cons && make && cd ..
# cd ./p2p-cycl && make && cd ..
# cd ./bcast-cons && make && cd ..
# cd ./bcast-cycl && make && cd ..

# # Size of matrix N (1024, 2048, 4096)

# cd ../serial
# make
# serial_exec="1"
# serial1024=$(./program 1024)
# echo "serial1024 = "$serial1024
# serial2048=$(./program 2048)
# echo "serial2048 = "$serial2048
# serial4096=$(./program 4096)
# echo "serial4096 = "$serial4096
# cd ../mpi

# #exit

# # Number of processes P (2, 4, 8, 16)
# # Assume folder hist created
# echo 'Creating Gaussian Elimination histograms and charts'
# for (( N = 1024; N<=4096; N*=2 )); do
# 	speedupfile="./speedup/speedup$N.dat"
# 	rm -f $speedupfile
# 	if [ "$N" = "1024" ]; then
# 		serial_exec=$serial1024 #"3.7552061081"
# 	elif [ "$N" = "2048" ]; then
# 		serial_exec=$serial2048 #"30.538867950"
# 	else  
# 		serial_exec=$serial4096 #"241.31324387"
# 	fi

# 	echo "P p2p-cons p2p-cycl bcast-cons bcast-cycl" >> $speedupfile
# 	for (( P = 2; P<=16; P*=2 )); do
# 		histfile="./hist/$N-$P.dat"
# 		rm -f $histfile
		
# 		IFS=$' ' read p2p_cons_exec p2p_cons_proc p2p_cons_comm <<< $(mpiexec -np $P ./p2p-cons/program $N)
# 		echo "p2p-cons $N $P | $p2p_cons_exec $p2p_cons_proc $p2p_cons_comm"
# 		IFS=$' ' read p2p_cycl_exec p2p_cycl_proc p2p_cycl_comm <<< $(mpiexec -np $P ./p2p-cycl/program $N)
# 		echo "p2p-cycl $N $P | $p2p_cycl_exec $p2p_cycl_proc $p2p_cycl_comm"
# 		IFS=$' ' read bcast_cons_exec bcast_cons_proc bcast_cons_comm <<< $(mpiexec -np $P ./bcast-cons/program $N)
# 		echo "bcast-cons $N $P | $bcast_cons_exec $bcast_cons_proc $bcast_cons_comm"
# 		IFS=$' ' read bcast_cycl_exec bcast_cycl_proc bcast_cycl_comm <<< $(mpiexec -np $P ./bcast-cycl/program $N)
# 		echo "bcast-cycl $N $P | $bcast_cycl_exec $bcast_cycl_proc $bcast_cycl_comm"

# 		echo "version t_exec t_proc t_comm" >> $histfile
# 		echo "p2p-cons $p2p_cons_exec $p2p_cons_proc $p2p_cons_comm" >> $histfile
# 		echo "p2p-cycl $p2p_cycl_exec $p2p_cycl_proc $p2p_cycl_comm" >> $histfile
# 		echo "bcast-cons $bcast_cons_exec $bcast_cons_proc $bcast_cons_comm" >> $histfile
# 		echo "bcast-cycl $bcast_cycl_exec $bcast_cycl_proc $bcast_cycl_comm" >> $histfile

# 		p2p_cons_su=$(bc <<< "scale = 10; $serial_exec / $p2p_cons_exec")
# 		p2p_cycl_su=$(bc <<< "scale = 10; $serial_exec / $p2p_cycl_exec")
# 		bcast_cons_su=$(bc <<< "scale = 10; $serial_exec/ $bcast_cons_exec")
# 		bcast_cycl_su=$(bc <<< "scale = 10; $serial_exec / $bcast_cycl_exec")
# 		echo "speedup | $P $p2p_cons_su $p2p_cycl_su $bcast_cons_su $bcast_cycl_su"
# 		echo "$P $p2p_cons_su $p2p_cycl_su $bcast_cons_su $bcast_cycl_su" >> $speedupfile
# 	done
# done

echo 'Done!'

exit 0