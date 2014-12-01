

#ifndef __DualSGM__
#define __DualSGM__

#include <stdio.h>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
//#include <stdlib>
#include <ctime>

using namespace cv;

class DualSGM {
public:
    ~DualSGM();
    DualSGM(Mat* first_image, int N);

    void serialUpdateModel(Mat* next_frame);
    
    float meanThreshold = 9.0; //This constant was taken from the paper, but could probably be varied

    // initial std?
    double std_init = 20.0;
    //initialized var
    double var_init = std_init * std_init;

    cv::Mat *bin_mat;
    cv::Mat *app_u_mat;
    cv::Mat *app_var_mat;
    cv::Mat *can_u_mat;
    cv::Mat *can_var_mat;
    int **app_ages;
    int **can_ages;

private:


};
#endif /* defined(__DualSGM__) */