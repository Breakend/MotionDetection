
#ifndef __DualSGM__
#define __DualSGM__

#include <stdio.h>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
//#include <stdlib>
#include <ctime>
#include <sys/time.h>

using namespace cv;

class DualSGM {
public:
    ~DualSGM();
    DualSGM(Mat* first_image, int N);

    static int num_rows;
    static int num_cols;

    void serialUpdateModel(Mat* next_frame);

private:
    double timer(void);

    cv::Mat prev_frame;
    cv::Mat *bin_mat;
    cv::Mat *app_u_mat;
    cv::Mat *app_var_mat;
    cv::Mat *can_u_mat;
    cv::Mat *can_var_mat;
    int **app_ages;
    int **can_ages;

};
#endif /* defined(__DualSGM__) */