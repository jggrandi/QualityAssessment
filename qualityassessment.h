/* Class to handle quality assessment algorithms
	by: Jeronimo G. Grandi
	Jul,2013
*/  

#ifndef QUALITY_ASSESSMENT
#define QUALITY_ASSESSMENT


#include "qualityassessment_utils.h"


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
	Scalar getPSNR(  const D *img1_data, const D *img2_data, const int w, const int h, const int stride)
	{
		const int L_sqd = 255 * 255;
		Scalar psnr;
		float mse_result = getMSE(img1_data,img2_data,w,h,stride);
		if (mse_result == 0)
			return 0;
		psnr.val[0] = (float)( 10.0 * log10( L_sqd / mse_result) );		
		return psnr;
	}

	
	Scalar getPSNR(  const Mat& i1, const Mat& i2)
	{
	    Mat s1;
	    absdiff(i1, i2, s1);       // |i1 - i2|
	    s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
	    s1 = s1.mul(s1);           // |i1 - i2|^2

	    Scalar s = sum(s1);         // sum elements per channel

	    double sse = s.val[0];// + s.val[1] + s.val[2]; // sum channels

	    if( sse <= 1e-10) // for small values return zero
	        return 0;
	    else
	    {
	    	Scalar mse;
	    	 mse.val[0] = sse /(double)(i1.channels());// * i1.total());
	    	 return mse;
	  //       double  mse =sse /(double)(i1.channels() * i1.total());
	  //       Scalar psnr;
			// psnr.val[0]= 10.0*log10((255*255)/mse);
	  //       return psnr;
	    }		
	}
	template <class D>
	Scalar getMSSIM( const D *ref, const D *cmp, int resW, int resH)
//	Scalar getMSSIM( const Mat& i1, const Mat& i2)
	{

	    Mat i1(resW,resH,CV_16UC1, ref);
	    Mat i2(resW,resH,CV_16UC1, cmp);

	    const double C1 = 6.5025, C2 = 58.5225;
	    /***************************** INITS **********************************/
	    int d     = CV_32F;

	    Mat I1, I2;
	    i1.convertTo(I1, d);           // cannot calculate on one byte large values
	    i2.convertTo(I2, d);

	    Mat I2_2   = I2.mul(I2);        // I2^2
	    Mat I1_2   = I1.mul(I1);        // I1^2
	    Mat I1_I2  = I1.mul(I2);        // I1 * I2

	    /*************************** END INITS **********************************/

	    Mat mu1, mu2;   // PRELIMINARY COMPUTING
	    GaussianBlur(I1, mu1, Size(GB_S, GB_S), GB_R);
	    GaussianBlur(I2, mu2, Size(GB_S, GB_S), GB_R);

	    Mat mu1_2   =   mu1.mul(mu1);
	    Mat mu2_2   =   mu2.mul(mu2);
	    Mat mu1_mu2 =   mu1.mul(mu2);

	    Mat sigma1_2, sigma2_2, sigma12;

	    GaussianBlur(I1_2, sigma1_2, Size(GB_S, GB_S), GB_R);
	    sigma1_2 -= mu1_2;

	    GaussianBlur(I2_2, sigma2_2, Size(GB_S, GB_S), GB_R);
	    sigma2_2 -= mu2_2;

	    GaussianBlur(I1_I2, sigma12, Size(GB_S, GB_S), GB_R);
	    sigma12 -= mu1_mu2;

	    ///////////////////////////////// FORMULA ////////////////////////////////
	    Mat t1, t2, t3;

	    t1 = 2 * mu1_mu2 + C1;
	    t2 = 2 * sigma12 + C2;
	    t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

	    t1 = mu1_2 + mu2_2 + C1;
	    t2 = sigma1_2 + sigma2_2 + C2;
	    t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

	    Mat ssim_map;
	    divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;

	    Scalar mssim = mean( ssim_map ); // mssim = average of ssim map
		return mssim;

	}

	Scalar getSURF(  const Mat& i1, const Mat& i2)
	{

		//-- Step 1: Detect the keypoints using SURF Detector
		int minHessian = 5000;

		SurfFeatureDetector detector( minHessian );

		std::vector<KeyPoint> keypoints_1, keypoints_2;

		detector.detect( i1, keypoints_1 );
		detector.detect( i2, keypoints_2 );

		SurfDescriptorExtractor extractor; //Calculate descriptors (feature vectors)

		Mat descriptors_1, descriptors_2;

		extractor.compute( i1, keypoints_1, descriptors_1 );
		extractor.compute( i2, keypoints_2, descriptors_2 );


		FlannBasedMatcher matcher; //Matching descriptor vectors using FLANN matcher
		std::vector< DMatch > matches;
		matcher.match( descriptors_1, descriptors_2, matches );

		double max_dist = 0; double min_dist = 100;

		//Calculation of max and min distances between keypoints
		for( int i = 0; i < descriptors_1.rows; i++ )
		{ double dist = matches[i].distance;
		if( dist < min_dist ) min_dist = dist;
		if( dist > max_dist ) max_dist = dist;
		}

		//-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist )
		//-- PS.- radiusMatch can also be used here.
		std::vector< DMatch > good_matches;

		Scalar v;
		v.val[0]=min_dist;
		v.val[1]=max_dist;
		return v;
	}

	Scalar getPSNR_GPU_optimized(  const gpu::GpuMat i1, const gpu::GpuMat i2)
	{
	    bufferPSNR.gI1=i1;
	    bufferPSNR.gI2=i2;

	    bufferPSNR.gI1.convertTo(bufferPSNR.t1, CV_32F);
	    bufferPSNR.gI2.convertTo(bufferPSNR.t2, CV_32F);

	    gpu::absdiff(bufferPSNR.t1.reshape(1), bufferPSNR.t2.reshape(1), bufferPSNR.gs);
	    gpu::multiply(bufferPSNR.gs, bufferPSNR.gs, bufferPSNR.gs);

	    double sse = gpu::sum(bufferPSNR.gs, bufferPSNR.buf)[0];

	    if( sse <= 1e-10) // for small values return zero
	        return 0;
	    else
	    {
	    	Scalar mse;
	    	mse.val[0] = sse /(double)(i1.channels());// * I1.total());
			return mse;
	        //double mse = sse /(double)(i1.channels());// * I1.total());
	       

	  //       Scalar psnr;
			// psnr.val[0]= 10.0*log10((255*255)/mse);
	  //       return psnr;
	    }
	}


	Scalar getMSSIM_GPU_optimized( const gpu::GpuMat i1, const gpu::GpuMat i2)
	{
	    const float C1 = 6.5025f, C2 = 58.5225f;
	    /***************************** INITS **********************************/

	    bufferMSSIM.gI1=i1;
	    bufferMSSIM.gI2=i2;

	    gpu::Stream stream;

	    stream.enqueueConvert(bufferMSSIM.gI1, bufferMSSIM.t1, CV_32F);
	    stream.enqueueConvert(bufferMSSIM.gI2, bufferMSSIM.t2, CV_32F);

	    gpu::split(bufferMSSIM.t1, bufferMSSIM.vI1, stream);
	    gpu::split(bufferMSSIM.t2, bufferMSSIM.vI2, stream);
	    Scalar mssim;
	    gpu::GpuMat buf;

	    for( int i = 0; i < bufferMSSIM.gI1.channels(); ++i )
	    {
	        gpu::multiply(bufferMSSIM.vI2[i], bufferMSSIM.vI2[i], bufferMSSIM.I2_2, stream);        // I2^2
	        gpu::multiply(bufferMSSIM.vI1[i], bufferMSSIM.vI1[i], bufferMSSIM.I1_2, stream);        // I1^2
	        gpu::multiply(bufferMSSIM.vI1[i], bufferMSSIM.vI2[i], bufferMSSIM.I1_I2, stream);       // I1 * I2

	        gpu::GaussianBlur(bufferMSSIM.vI1[i], bufferMSSIM.mu1, Size(31, 31), buf, 10.5, 0, BORDER_DEFAULT, -1, stream);
	        gpu::GaussianBlur(bufferMSSIM.vI2[i], bufferMSSIM.mu2, Size(31, 31), buf, 10.5, 0, BORDER_DEFAULT, -1, stream);

	        gpu::multiply(bufferMSSIM.mu1, bufferMSSIM.mu1, bufferMSSIM.mu1_2, stream);
	        gpu::multiply(bufferMSSIM.mu2, bufferMSSIM.mu2, bufferMSSIM.mu2_2, stream);
	        gpu::multiply(bufferMSSIM.mu1, bufferMSSIM.mu2, bufferMSSIM.mu1_mu2, stream);

	        gpu::GaussianBlur(bufferMSSIM.I1_2, bufferMSSIM.sigma1_2, Size(31, 31), buf, 10.5, 0, BORDER_DEFAULT, -1, stream);
	        gpu::subtract(bufferMSSIM.sigma1_2, bufferMSSIM.mu1_2, bufferMSSIM.sigma1_2, gpu::GpuMat(), -1, stream);
	        //bufferMSSIM.sigma1_2 -= bufferMSSIM.mu1_2;  - This would result in an extra data transfer operation

	        gpu::GaussianBlur(bufferMSSIM.I2_2, bufferMSSIM.sigma2_2, Size(31, 31), buf, 10.5, 0, BORDER_DEFAULT, -1, stream);
	        gpu::subtract(bufferMSSIM.sigma2_2, bufferMSSIM.mu2_2, bufferMSSIM.sigma2_2, gpu::GpuMat(), -1, stream);
	        //bufferMSSIM.sigma2_2 -= bufferMSSIM.mu2_2;

	        gpu::GaussianBlur(bufferMSSIM.I1_I2, bufferMSSIM.sigma12, Size(31, 31), buf, 10.5, 0, BORDER_DEFAULT, -1, stream);
	        gpu::subtract(bufferMSSIM.sigma12, bufferMSSIM.mu1_mu2, bufferMSSIM.sigma12, gpu::GpuMat(), -1, stream);
	        //bufferMSSIM.sigma12 -= bufferMSSIM.mu1_mu2;

	        //here too it would be an extra data transfer due to call of operator*(Scalar, Mat)
	        gpu::multiply(bufferMSSIM.mu1_mu2, 2, bufferMSSIM.t1, 1, -1, stream); //bufferMSSIM.t1 = 2 * bufferMSSIM.mu1_mu2 + C1;
	        gpu::add(bufferMSSIM.t1, C1, bufferMSSIM.t1, gpu::GpuMat(), -1, stream);
	        gpu::multiply(bufferMSSIM.sigma12, 2, bufferMSSIM.t2, 1, -1, stream); //bufferMSSIM.t2 = 2 * bufferMSSIM.sigma12 + C2;
	        gpu::add(bufferMSSIM.t2, C2, bufferMSSIM.t2, gpu::GpuMat(), -12, stream);

	        gpu::multiply(bufferMSSIM.t1, bufferMSSIM.t2, bufferMSSIM.t3, 1, -1, stream);     // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

	        gpu::add(bufferMSSIM.mu1_2, bufferMSSIM.mu2_2, bufferMSSIM.t1, gpu::GpuMat(), -1, stream);
	        gpu::add(bufferMSSIM.t1, C1, bufferMSSIM.t1, gpu::GpuMat(), -1, stream);

	        gpu::add(bufferMSSIM.sigma1_2, bufferMSSIM.sigma2_2, bufferMSSIM.t2, gpu::GpuMat(), -1, stream);
	        gpu::add(bufferMSSIM.t2, C2, bufferMSSIM.t2, gpu::GpuMat(), -1, stream);


	        gpu::multiply(bufferMSSIM.t1, bufferMSSIM.t2, bufferMSSIM.t1, 1, -1, stream);     // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
	        gpu::divide(bufferMSSIM.t3, bufferMSSIM.t1, bufferMSSIM.ssim_map, 1, -1, stream);      // ssim_map =  t3./t1;

	        stream.waitForCompletion();

	        Scalar s = gpu::sum(bufferMSSIM.ssim_map, bufferMSSIM.buf);
	        mssim.val[i] = s.val[0] / (bufferMSSIM.ssim_map.rows * bufferMSSIM.ssim_map.cols);
	    }
	    return mssim;		
	}

	
	template <class D>
	Scalar getMutualInformation(const D *ref, const D *cmp, int resW, int resH)
	{

		const unsigned int Dimension = 2;
		typedef unsigned short imgT;
		typedef itk::Image< imgT, Dimension > ImageType;
		typedef itk::ImportImageFilter< imgT, Dimension >   ImportFilterType;
		ImportFilterType::Pointer imgRef = ImportFilterType::New();
		ImportFilterType::Pointer imgComp= ImportFilterType::New();
		ImportFilterType::SizeType  sizeImg;
		sizeImg[0]  = resW;  // size along X
		sizeImg[1]  = resH;  // size along Y
		const itk::SpacePrecisionType  spacing[ Dimension ] =  { 1.0, 1.0};
		const itk::SpacePrecisionType origin[ Dimension ] = { 0.0, 0.0};
		ImportFilterType::IndexType start;
		ImportFilterType::RegionType region;
		start.Fill( 0 );
		region.SetIndex( start );
		region.SetSize(  sizeImg  );
		imgRef->SetRegion( region );
		imgComp->SetRegion( region );
		imgRef->SetOrigin( origin );
		imgComp->SetOrigin( origin );
		imgRef->SetSpacing( spacing );
		imgComp->SetSpacing( spacing );

		const bool importImageFilterWillOwnTheBuffer = false;
		const unsigned int numberOfPixels =  resW*resH;    
		imgRef->SetImportPointer( ref, numberOfPixels ,importImageFilterWillOwnTheBuffer );
		imgRef->Update();
		imgComp->SetImportPointer( cmp, numberOfPixels ,importImageFilterWillOwnTheBuffer );
		imgComp->Update();


		typedef itk::JoinImageFilter< ImageType, ImageType >  JoinFilterType;
		JoinFilterType::Pointer joinFilter = JoinFilterType::New();
		joinFilter->SetInput1( imgRef->GetOutput() );
		joinFilter->SetInput2( imgComp->GetOutput() );

		try
		{
		joinFilter->Update();
		}
		catch( itk::ExceptionObject & excp )
		{
		std::cerr << excp << std::endl;
		return -1;
		}

		typedef JoinFilterType::OutputImageType VectorImageType;
		typedef itk::Statistics::ImageToHistogramFilter<VectorImageType >  HistogramFilterType;
		HistogramFilterType::Pointer histogramFilter = HistogramFilterType::New();
		histogramFilter->SetInput(  joinFilter->GetOutput()  );
		histogramFilter->SetMarginalScale( 10.0 );
		typedef HistogramFilterType::HistogramSizeType   HistogramSizeType;
		HistogramSizeType size( 2 );
		size[0] = 128;  // number of bins for the first  channel
		size[1] = 128;  // number of bins for the second channel

		histogramFilter->SetHistogramSize( size );
		typedef HistogramFilterType::HistogramMeasurementVectorType
		HistogramMeasurementVectorType;
		HistogramMeasurementVectorType binMinimum( 3 );
		HistogramMeasurementVectorType binMaximum( 3 );
		binMinimum[0] = -0.5;
		binMinimum[1] = -0.5;
		binMinimum[2] = -0.5;
		binMaximum[0] = 128.5;
		binMaximum[1] = 128.5;
		binMaximum[2] = 128.5;
		histogramFilter->SetHistogramBinMinimum( binMinimum );
		histogramFilter->SetHistogramBinMaximum( binMaximum );
		histogramFilter->Update();

		typedef HistogramFilterType::HistogramType  HistogramType;
		const HistogramType * histogram = histogramFilter->GetOutput();
		HistogramType::ConstIterator itr = histogram->Begin();
		HistogramType::ConstIterator end = histogram->End();
		const double Sum = histogram->GetTotalFrequency();

		double JointEntropy = 0.0;
		while( itr != end )
		{
		const double count = itr.GetFrequency();
		if( count > 0.0 )
		  {
		  const double probability = count / Sum;
		  JointEntropy +=
		    - probability * std::log( probability ) / std::log( 2.0 );
		  }
		++itr;
		}

		size[0] = 128;  // number of bins for the first  channel
		size[1] =   1;  // number of bins for the second channel
		histogramFilter->SetHistogramSize( size );
		histogramFilter->Update();

		itr = histogram->Begin();
		end = histogram->End();
		double Entropy1 = 0.0;
		while( itr != end )
		{
		const double count = itr.GetFrequency();
		if( count > 0.0 )
		  {
		  const double probability = count / Sum;
		  Entropy1 += - probability * std::log( probability ) / std::log( 2.0 );
		  }
		++itr;
		}

		size[0] =   1;  // number of bins for the first channel
		size[1] = 128;  // number of bins for the second channel
		histogramFilter->SetHistogramSize( size );
		histogramFilter->Update();

		itr = histogram->Begin();
		end = histogram->End();
		double Entropy2 = 0.0;
		while( itr != end )
		{
		const double count = itr.GetFrequency();
		if( count > 0.0 )
		  {
		  const double probability = count / Sum;
		  Entropy2 += - probability * std::log( probability ) / std::log( 2.0 );
		  }
		++itr;
		}

		//double MutualInformation = Entropy1 + Entropy2 - JointEntropy;

		//double NormalizedMutualInformation1 = 2.0 * MutualInformation / ( Entropy1 + Entropy2 );

		double NormalizedMutualInformation2 = ( Entropy1 + Entropy2 ) / JointEntropy;


		//std::cout << NormalizedMutualInformation2 <<  std::endl;

		Scalar MI;
		MI.val[0] = NormalizedMutualInformation2;
		return MI;
	}


private:
    BufferPSNR bufferPSNR;
    BufferMSSIM bufferMSSIM;

};


#endif // QUALITY_ASSESSMENT