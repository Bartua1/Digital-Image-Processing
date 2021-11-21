#ifndef FourierTransform_H
#define FourierTransform_H

#include <opencv2/opencv.hpp>
double * fourierTransform(cv::Mat image, int m, int n, int k, int l, double * arr){
    double r = cos(2*2*acos(0.0)*((m*k/image.rows)+(n*l/image.cols)))/sqrt(image.cols*image.rows);
    double i = sin(-2*2*acos(0.0)*((m*k/image.rows)+(n*l/image.cols)))/sqrt(image.cols*image.rows);
    arr[0]=pow(sqrt(pow(r,2)+pow(i,2)),2);
    arr[1]=atan(i/r);
    return arr;
}


double * fourierAt(cv::Mat image, int k, int l){
    static double arr[2];
    double I=0.0;
    double R=0.0;
    for (int m = 0; m < image.rows-1; m++) {
        for (int n = 0; n < image.cols-1; n++) {
            I+=image.at<double>(n,m)*sin(-2*M_PI*((m*k/image.rows)+(n*l/image.cols)))/sqrt(image.cols*image.rows);
            R+=image.at<double>(n,m)*cos(2*M_PI*((m*k/image.rows)+(n*l/image.cols)))/sqrt(image.cols*image.rows);
        }
    }
    arr[0]=R;
    arr[1]=I;
    return arr;
}

/*
double fourierAt(cv::Mat image, int k, int l,int type){
    double sum = 0.0;
    for (int m = 0; m < image.rows-1; m++) {
        for (int n = 0; n < image.cols-1; n++) {
            sum+=fourierTransform(image,m,n,k,l,type)*image.at<double>(n,m);
        }
    }
    return sum;
}

*/

cv::Mat swapQ(cv::Mat image){
    int cx = image.cols/2;
    int cy = image.rows/2;
    cv::Mat q0(image, cv::Rect(0, 0, cx, cy));   // Top-Left
    cv::Mat q1(image, cv::Rect(cx, 0, cx, cy));  // Top-Right
    cv::Mat q2(image, cv::Rect(0, cy, cx, cy));  // Bottom-Left
    cv::Mat q3(image, cv::Rect(cx, cy, cx, cy)); // Bottom-Right
    cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);                        // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
    return image;
}

/*

void DiscreteFourierTransform(){
    cv::Mat I = cv::imread("/home/bartu/Downloads/dzo_vsc/images/lena64.png", cv::IMREAD_COLOR);

    cv::Mat gray_8uc1_img; // declare variable to hold grayscale version of img variable, gray levels wil be represented using 8 bits (uchar)
    cv::Mat image; // declare variable to hold grayscale version of img variable, gray levels wil be represented using 32 bits (float)

    cv::cvtColor(I, gray_8uc1_img, cv::COLOR_BGR2GRAY);
    gray_8uc1_img.convertTo(image, CV_32FC1, 1.0/255.0);

    if( image.empty()){
        printf("empty");
    }
    cv::Mat real = image.clone();
    cv::Mat imaginary = image.clone();
    for (int k = 0; k < image.rows-1; k++) {
        for (int l = 0; l < image.cols-1; l++) {
            double *arr = fourierAt(image,k,l);
            real.at<double>(l,k) = arr[0];
            imaginary.at<double>(l,k) = arr[1];
        }
    }

    //Logarithming
    cv::Mat processed_image;
    cv::Mat phase;
    
    cv::cartToPolar(real,imaginary,processed_image,phase);

    //processed_image += cv::Scalar::all(1);
    log(processed_image, processed_image);

    //Swapping quadrants
    
    cv::Mat power = swapQ(processed_image);

    normalize(power, power, 0, 1, cv::NORM_MINMAX);
    cv::namedWindow("Power",cv::WINDOW_NORMAL);
    cv::imshow("Power", power);
    //cv::namedWindow("Phase",cv::WINDOW_NORMAL);
    //cv::imshow("Phase", phase);
}

*/

void DiscreteFourierTransform(){
    cv::Mat I = cv::imread("/home/bartu/Documents/dzo_vsc/images/lena64.png", cv::IMREAD_GRAYSCALE);
    cv::Mat padded;                            //expand input image to optimal size
    int m = cv::getOptimalDFTSize( I.rows );
    int n = cv::getOptimalDFTSize( I.cols ); // on the border add zero values
    copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complexI;
    cv::merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

    cv::dft(complexI, complexI);            // this way the result may fit in the source matrix

    // compute the magnitude and switch to logarithmic scale
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    cv::Mat magI;
    cv::Mat phase;
    cv::cartToPolar(planes[0], planes[1], magI, phase);// planes[0] = magnitude

    magI += cv::Scalar::all(1);                    // switch to logarithmic scale
    log(magI, magI);

    // crop the spectrum, if it has an odd number of rows or columns
    magI = magI(cv::Rect(0, 0, magI.cols & -2, magI.rows & -2));

    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = magI.cols/2;
    int cy = magI.rows/2;

    cv::Mat q0(magI, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    cv::Mat q1(magI, cv::Rect(cx, 0, cx, cy));  // Top-Right
    cv::Mat q2(magI, cv::Rect(0, cy, cx, cy));  // Bottom-Left
    cv::Mat q3(magI, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

    swapQ(magI);

    normalize(magI, magI, 0, 1, cv::NORM_MINMAX); // Transform the matrix with float values into a
                                            // viewable image form (float between values 0 and 1).

    normalize(phase, phase, 0, 1, cv::NORM_MINMAX);
    cv::namedWindow("Power",cv::WINDOW_NORMAL);
    cv::imshow("Power", magI);
    cv::namedWindow("Phase",cv::WINDOW_NORMAL);
    cv::imshow("Phase", phase);
}

void PrintFourier(){
    cv::Mat src_8uc3_img = cv::imread("/home/bartu/Documents/dzo_vsc/images/lena64.png", cv::IMREAD_COLOR); // load color image from file system to Mat variable, this will be loaded using 8 bits (uchar)

    if (src_8uc3_img.empty()) {
        printf("Unable to read input file (%s, %d).", __FILE__, __LINE__);
    }

    cv::Mat gray_8uc1_img; // declare variable to hold grayscale version of img variable, gray levels wil be represented using 8 bits (uchar)
    cv::Mat gray_32fc1_img; // declare variable to hold grayscale version of img variable, gray levels wil be represented using 32 bits (float)

    cv::cvtColor(src_8uc3_img, gray_8uc1_img, cv::COLOR_BGR2GRAY); // convert input color image to grayscale one, CV_BGR2GRAY specifies direction of conversion
    gray_8uc1_img.convertTo(gray_32fc1_img, CV_32FC1, 1.0/255.0); // convert grayscale image from 8 bits to 32 bits, resulting values will be in the interval 0.0 - 1.0

    /* Discrete Fourier Transform (Exercise 4) */
    cv::namedWindow("Photo",cv::WINDOW_NORMAL);
    cv::imshow("Photo", gray_32fc1_img);
    DiscreteFourierTransform();
    /*
    cv::Mat phase = DiscreteFourierTransform(gray_32fc1_img);
    cv::namedWindow("Phase",cv::WINDOW_NORMAL);
    cv::imshow("Phase", phase);
    */
}

#endif