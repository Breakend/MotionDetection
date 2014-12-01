

tbb_main: DualSGM.o tbb_main.o
	g++ -std=c++11 DualSGM.o tbb_main.o -o tbb_main `pkg-config --cflags --libs opencv`
	rm -rf *o *gch

tbb_main.o: tbb_main.cpp 
	g++ -std=c++11 -c tbb_main.cpp 

DualSGM.o: DualSGM.hpp DualSGM.cpp 
	g++ -std=c++11 -c DualSGM.hpp DualSGM.cpp 


clean:
	rm -rf *o *gch tbb_main
