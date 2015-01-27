/* Class to handle quality assessment algorithms
	by: Jeronimo G. Grandi
	Jul,2013
*/  

#ifndef QUALITY_ASSESSMENT
#define QUALITY_ASSESSMENT

#include <cstdlib>
#include <cmath>
// #include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
// #include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
// #include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
// #include <opencv2/features2d/features2d.hpp>
// #include <opencv2/nonfree/nonfree.hpp>
// #include <opencv2/gpu/gpu.hpp> 

//#include "qualityassessment_utils.h"

#define ijn(a,b,n) ((a)*(n))+b



//using namespace cv;


class QualityAssessment
{
public:
	QualityAssessment(){}
	~QualityAssessment(){}


	 /* MSE(a,b) = 1/N * SUM((a-b)^2) */
	template <class D>
	float getMSE(const D *ref, const D *cmp, const int w, const int h, const int stride)
	{
		int error, offset;
		unsigned long long sum=0;
		int ww,hh;
		for (hh=0; hh<h; hh++) 
		{
			offset = hh*stride;
			for (ww=0; ww<w; ww++, offset++) 
			{	
				error = ref[offset] - cmp[offset];
				sum += error * error;
			}
		}
		if(sum==0)
			return 0;

		return (float)( (double)sum / (double)(w*h) );
	}

	/* PSNR(a,b) = 10*log10(L^2 / MSE(a,b)), where L=2^b - 1 (8bit = 255) */
	template <class D>
	float getPSNR(  const D *img1_data, const D *img2_data, const int w, const int h, const int stride)
	{
		const int L_sqd = 255 * 255;
		float psnr;
		float mse_result = getMSE(img1_data,img2_data,w,h,stride);
		if (mse_result == 0)
			return 0;
		psnr = (float)( 10.0 * log10( L_sqd / mse_result) );		
		return psnr;
	}
};

#endif // QUALITY_ASSESSMENT