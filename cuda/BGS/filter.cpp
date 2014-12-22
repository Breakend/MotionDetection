#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <sys/time.h>

#define STD_DEV 2.f

/**
* generate an NxN gaussian filter based on the specified N value
* standard deviation is set
*/
float * create2DGaussianFilter(int N) 
{  

	/*
	* generate a contiguous char array in memory
	*/
	float * filter_vals =  (float *) std::calloc(N*N, sizeof(float));

    const int half = N/2; 
	float summed = 0.f;
  
  	/*
  	* Note: some of this was taken from: https://code.msdn.microsoft.com/windowsdesktop/Gaussian-blur-with-CUDA-5-df5db506
  	*/

    // Calculate filter sum first 
    for (int r = -half; r <= half; ++r) 
    { 
        for (int c = -half; c <= half; ++c) 
        { 
            float weight = expf(-static_cast<float>(c * c + r * r) / (2.f * STD_DEV * STD_DEV)); 
            int idx = (r + half) * N + c + half; 
 
            filter_vals[idx] = weight; 
            summed += weight; 
        } 
    } 
 
    // Normalize weight: sum of weights must equal 1 
    float normal = 1.f / summed; 
 
    for (int r = -half; r <= half; ++r) 
    { 
        for (int c = -half; c <= half; ++c) 
        { 
            int idx = (r + half) * N + c + half; 
 
            filter_vals[idx] *= normal; 
        } 
    } 

    return filter_vals;
} 