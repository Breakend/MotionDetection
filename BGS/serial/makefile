
serial_main: DualSGM.o serial_main.o
	g++ -std=c++11 DualSGM.o serial_main.o -o serial_main `pkg-config --cflags --libs opencv`
	rm -rf *o *gch

serial_main.o: serial_main.cpp 
	g++ -std=c++11 -c serial_main.cpp 

DualSGM.o: DualSGM.hpp DualSGM.cpp 
	g++ -std=c++11 -c DualSGM.hpp DualSGM.cpp 


clean:
	rm -rf *o *gch tbb_main
