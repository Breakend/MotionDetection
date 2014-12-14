
#ifndef __DualSGM__
#define __DualSGM__

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <ctime>
#include <sys/time.h>

#include "Blur.hpp"

using namespace cv;

class DualSGM {
    public:
        ~DualSGM();
        DualSGM(Mat* first_image, int N);

        static int num_rows;
        static int num_cols;

        void serialUpdateModel(Mat* next_frame);
        double tbbUpdateModel(Mat *next_frame, int num_threads);

    private:
        void motionCompensation(Mat* next_frame);
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

inline void core_dsgm_update(cv::Mat *next_frame, cv::Mat *bin_mat, 
    cv::Mat *app_u_mat, cv::Mat *app_var_mat, 
    cv::Mat *can_u_mat, cv::Mat *can_var_mat, 
    int **app_ages, int **can_ages, 
    int offset, int row_lim); 

#endif /* defined(__DualSGM__) */