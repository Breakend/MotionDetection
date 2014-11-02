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
#include "GaussianModel.h"

int main(int argc, const char * argv[]) {
    
    //
//    std::string testFile = "../Videos/cheetah_test.mp4";
    
    //Trying a test video from http://wordpress-jodoin.dmi.usherb.ca/dataset2014/
    
    IplImage frame = cv::imread("../Videos/sofa/in0000001.jpg");
    
    GaussianModel gm(&frame, 10);
    
    for(int i = 50; i<250; i++){
        
        //TODO: format this properly
        frame = cv::imread("../Videos/sofa/in0000001.jpg");
        gm.updateModel(frame);
    }
    
    
    std::cout << "Running Background subtraction test";
    return 0;
}
