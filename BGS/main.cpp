//
//  main.cpp
//  BGS
//
//  Created by Peter Henderson on 2014-11-02.
//  Copyright (c) 2014 Peter Henderson. All rights reserved.
//

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <stdlib.h>
#include "DualGaussianModel.h"
#include <sys/time.h>

double timer(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + (((double) tv.tv_usec)/1e6);
}

int main(int argc, const char * argv[]) {
    
    double startt = timer();
    
    //
//    std::string testFile = "../Videos/cheetah_test.mp4";
    
    //Trying a test video from http://wordpress-jodoin.dmi.usherb.ca/dataset2014/
    //TODO: less hardcoding more configurables.
    int start = 1;
    int end = 500;
//    IplImage *frame = cvLoadImage("/Users/Breakend/Documents/code/BGS/Videos/sofa/input/in000050.jpg");

    // /Users/Breakend/Documents/code/BGS/Videos/

    // /Users/Breakend/Documents/code/BGS/

    Mat frame = imread("/Users/Breakend/Documents/code/BGS/Videos/badminton/input/in000001.jpg",  CV_LOAD_IMAGE_GRAYSCALE);

    DualGaussianModel gm(&frame, 10);
    char buff[100];

    
    for(int i = start + 1; i < end; i++){

        sprintf(buff, "/Users/Breakend/Documents/code/BGS/Videos/badminton/input/in%06d.jpg", i);
        std::string buffAsStdStr = buff;
        const char * c = buffAsStdStr.c_str();
        frame = imread(c,  CV_LOAD_IMAGE_GRAYSCALE);
        Mat destination;
        GaussianBlur( frame, destination, Size(9,9), 0, 0 );
        Mat dst;
        medianBlur ( destination, dst, 3 );
        cvWaitKey(1);
        gm.updateModel(&dst);
    }
    
//    getchar();
    
    double endd=timer();
    
    printf("\nexecution time: %f\n", endd-startt);
    
    return 0;

}
