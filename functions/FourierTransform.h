#ifndef FourierTransform_H
#define FourierTransform_H

#include <opencv2/opencv.hpp>
#include "swapQ.h"
#include <stdio.h>

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

cv::Mat InverseFourierTransform(cv::Mat complexI){
    cv::Mat inverseTransform;
    cv::dft(complexI, inverseTransform, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);
    normalize(inverseTransform, inverseTransform, 0, 1, cv::NORM_MINMAX);
    return inverseTransform;
}

cv::Mat * DiscreteFourierTransform(std::string file){
    cv::Mat I = cv::imread(file, cv::IMREAD_GRAYSCALE);
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
    cv::Mat Re = planes[0];
    cv::Mat Im = planes[1];
    cv::Mat magI;
    cv::Mat phase;
    cv::cartToPolar(planes[0], planes[1], magI, phase);// planes[0] = magnitude

    magI += cv::Scalar::all(1);                    // switch to logarithmic scale
    log(magI, magI);

    // crop the spectrum, if it has an odd number of rows or columns
    magI = magI(cv::Rect(0, 0, magI.cols & -2, magI.rows & -2));
    swapQ(magI);

    normalize(magI, magI, 0, 1, cv::NORM_MINMAX); // Transform the matrix with float values into a
                                            // viewable image form (float between values 0 and 1).

    normalize(phase, phase, 0, 1, cv::NORM_MINMAX);
    // DiscreteFourierTransform = [0=originalPhoto,1=power,2=phase]
    static cv::Mat DiscreteFourierTransform[6] = {I, magI, phase, complexI, Re, Im};
    return DiscreteFourierTransform;
}

void PrintFourier(std::string file){

    // Discrete Fourier Transform
    cv::Mat *fourier = DiscreteFourierTransform(file);
    // fourier = [0=originalPhoto,1=power,2=phase]
    cv::namedWindow("Photo",cv::WINDOW_NORMAL);
    cv::imshow("Photo", fourier[0]);
    cv::namedWindow("Power",cv::WINDOW_NORMAL);
    cv::imshow("Power", fourier[1]);
    cv::namedWindow("Phase",cv::WINDOW_NORMAL);
    cv::imshow("Phase", fourier[2]);
    // Inverse Discrete Fourier Transform
    cv::Mat ifourier = InverseFourierTransform(fourier[3]);
    cv::namedWindow("Photo2",cv::WINDOW_NORMAL);
    cv::imshow("Photo2", ifourier);
}

#endif