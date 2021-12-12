#include <stdio.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include "functions/convolution.h"
#include "functions/anisotropic.h"
#include "functions/FourierTransform.h"
#include "functions/NoiseRemoval.h"

int main()
{

    /* convolution Filter (Exercise 2) */
    //printConvolution();

    /* Anisotropic Filter (Exercise 3) */
    //printAnisotropic();

    /* Fourier Transform (Exercise 4) */
    //PrintFourier("/home/bartu/Documents/dzo_vsc/images/lena64.png");
    
    /* Frecuency Filter (Filter 5) */
    PrintFilter("/home/bartu/Documents/dzo_vsc/images/lena64_noise.png");
    
    cv::waitKey( 0 ); // wait until keypressed
    
    return 0;
}