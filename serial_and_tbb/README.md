
DSGM with TBB Parallelization
=============================

To build the program, pkg-config and OpenCV must be installed.
On Mac OSX, this can be done using Homebrew.
Once built, the executable 'main' is produced.

Program Structure 
=================

Blur.hpp and Blur.cpp contains both the serial and parallel versions of the implemented Gaussian and median blur. 
DuelSGM.hpp and DuelSGM.cpp contains both the serial and parallel versions of the Duel Mode Single Gaussian Model update functions. It also contains the motion compensation algorithm implemented using OpenCV. The preprocessing is also called here.
Main.cpp parses the test images and iterates through the set by passing the frames into the DuelSGM. A few helper functions to extract timing information for figure generation are implemented here as well.


