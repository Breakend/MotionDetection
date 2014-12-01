
omp_main: omp_main.o GaussianModel.o DualGaussianModel.o
	g++-4.9 -fopenmp -std=c++11 omp_main.o GaussianModel.o DualGaussianModel.o -o omp_main `pkg-config --cflags --libs opencv`
	rm -rf *o *gch

omp_main.o:
	g++-4.9 -fopenmp -std=c++11 -c omp_main.cpp 

GaussianModel.o: GaussianModel.h GaussianModel.cpp
	g++-4.9 -fopenmp -std=c++11 -c GaussianModel.h GaussianModel.cpp 

DualGaussianModel.o: DualGaussianModel.h DualGaussianModel.cpp
	g++-4.9 -fopenmp -std=c++11 -c DualGaussianModel.h DualGaussianModel.cpp 

clean:
	rm -rf *o *gch main
