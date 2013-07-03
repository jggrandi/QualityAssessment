/* Class to handle quality assessment algorithms
	by: Jeronimo G. Grandi
	Jul,2013
*/  

#ifndef QUALITY_ASSESSMENT
#define QUALITY_ASSESSMENT

#include <cstdlib>

#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/gpu/gpu.hpp> 

#include "qualityassessment_utils.h"

class QualityAssessment
{
public:
	QualityAssessment();
	~QualityAssessment();
	Scalar getPSNR(  const Mat& i1, const Mat& i2);
	Scalar getMSSIM( const Mat& i1, const Mat& i2);
	Scalar getSURF(  const Mat& i1, const Mat& i2);
	Scalar getPSNR_GPU_optimized(  const gpu::GpuMat& I1, const gpu::GpuMat& I2, BufferPSNR& b);	
	Scalar getMSSIM_GPU_optimized( const gpu::GpuMat& i1, const gpu::GpuMat& i2, BufferMSSIM& b);	
private:
    BufferPSNR bufferPSNR;
    BufferMSSIM bufferMSSIM;

};


#endif // QUALITY_ASSESSMENT