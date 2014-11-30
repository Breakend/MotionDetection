main: main.o GaussianModel.o DualGaussianModel.o
	g++ -std=c++11 main.o GaussianModel.o DualGaussianModel.o -o main `pkg-config --cflags --libs opencv`

main.o: main.cpp
	g++ -std=c++11 -c main.cpp

GaussianModel.o: GaussianModel.h GaussianModel.cpp
	g++ -std=c++11 -c GaussianModel.h GaussianModel.cpp

DualGaussianModel.o: DualGaussianModel.h DualGaussianModel.cpp
	g++ -std=c++11 -c DualGaussianModel.h DualGaussianModel.cpp

clean:
	rm -rf *o *gch main