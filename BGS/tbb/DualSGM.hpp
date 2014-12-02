

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

    const static int NUM_THREADS = 4;
    static int num_rows;
    static int num_cols;

    void serialUpdateModel(Mat* next_frame);
    void tbbUpdateModel(Mat *next_frame);
    
    const static int MEAN_THRESH = 9.0; //This constant was taken from the paper, but could probably be varied
    const static int VAR_INIT = 400;

    // initial std?
    double std_init = 20.0;
    //initialized var
    double var_init = std_init * std_init;

private:
    cv::Mat *bin_mat;
    cv::Mat *app_u_mat;
    cv::Mat *app_var_mat;
    cv::Mat *can_u_mat;
    cv::Mat *can_var_mat;
    int **app_ages;
    int **can_ages;

};
#endif /* defined(__DualSGM__) */