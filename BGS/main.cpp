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
#include <string>
#include <stdlib.h>
#include "DualGaussianModel.h"

int main(int argc, const char * argv[]) {
    
    //
//    std::string testFile = "../Videos/cheetah_test.mp4";
    
    //Trying a test video from http://wordpress-jodoin.dmi.usherb.ca/dataset2014/
    //TODO: less hardcoding more configurables.
    int start = 01;
    int end = 500;
//    IplImage *frame = cvLoadImage("/Users/Breakend/Documents/code/BGS/Videos/sofa/input/in000050.jpg");
    Mat frame = imread("/Users/Breakend/Documents/code/BGS/Videos/sofa/input/in000001.jpg",  CV_LOAD_IMAGE_GRAYSCALE);
//    CvCapture* capture = cvCaptureFromCAM( CV_CAP_ANY );
//    
//    if ( !capture ) {
//        fprintf( stderr, "ERROR: capture is NULL \n" );
//        getchar();
//        return -1;
//    }
    
//    IplImage* frame = cvQueryFrame(capture);
//    int i = 0;
//    while(!frame && i++<500){
//        fprintf( stderr, "ERROR: frame is NULL \n" );
////        getchar();
////        return -1;
//    }

    DualGaussianModel gm(&frame, 10);
    char buff[100];

    for(int i = start+1; i<end; i++){
//        frame = cvQueryFrame(capture);

        sprintf(buff, "/Users/Breakend/Documents/code/BGS/Videos/sofa/input/in%06d.jpg", i);
        std::string buffAsStdStr = buff;
        const char * c = buffAsStdStr.c_str();
        frame = imread(c,  CV_LOAD_IMAGE_GRAYSCALE);

//        IplImage* tempframe = cvLoadImage(c);
//        cvShowImage("origin", tempframe);
        cvWaitKey(1);
//        cvWaitKey(0);
        gm.updateModel(&frame);
//        getchar();
    }
    
//    getchar();
    
    std::cout << "Running Background subtraction test";
    return 0;
}
