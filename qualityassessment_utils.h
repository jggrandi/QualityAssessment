#ifndef QUALITY_ASSESSMENT_UTILS
#define QUALITY_ASSESSMENT_UTILS

#include <cstdlib>

#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/gpu/gpu.hpp> 


#include "itkImage.h"
#include "itkImportImageFilter.h"
#include "itkJoinImageFilter.h"
#include "itkImageToHistogramFilter.h"


#define ijn(a,b,n) ((a)*(n))+b

#define GB_R 10.5 //used in tests
#define GB_S 31  //used in tests

// #define GB_R 1.5 //original
// #define GB_S 11  //original

// #define GB_R 0.5 //used in tests
// #define GB_S 3  //used in tests


using namespace cv;



struct BufferPSNR                                     // Optimized GPU versions
{   // Data allocations are very expensive on GPU. Use a buffer to solve: allocate once reuse later.
    gpu::GpuMat gI1, gI2, gs, t1,t2;
    gpu::GpuMat buf;
};


struct BufferMSSIM                                     // Optimized GPU versions
{   // Data allocations are very expensive on GPU. Use a buffer to solve: allocate once reuse later.
    gpu::GpuMat gI1, gI2, gs, t1,t2;
    gpu::GpuMat I1_2, I2_2, I1_I2;
    vector<gpu::GpuMat> vI1, vI2;
    gpu::GpuMat mu1, mu2;
    gpu::GpuMat mu1_2, mu2_2, mu1_mu2;
    gpu::GpuMat sigma1_2, sigma2_2, sigma12;
    gpu::GpuMat t3;
    gpu::GpuMat ssim_map;
    gpu::GpuMat buf;
};


#endif // QUALITY_ASSESSMENT_UTILS

