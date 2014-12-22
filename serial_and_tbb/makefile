
main: Blur.o DualSGM.o main.o
	g++ -std=c++11 Blur.o DualSGM.o main.o -o main `pkg-config --cflags --libs opencv`
	rm -rf *o *gch

main.o: main.cpp 
	g++ -std=c++11 -c main.cpp 

DualSGM.o: Blur.hpp DualSGM.hpp DualSGM.cpp 
	g++ -std=c++11 -c Blur.hpp DualSGM.hpp DualSGM.cpp 

Blur.o: Blur.hpp Blur.cpp 
	g++ -std=c++11 -c Blur.hpp Blur.cpp 

clean:
	rm -rf *o *gch main
