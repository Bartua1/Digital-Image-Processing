#ifndef NoiseRemoval_H
#define NoiseRemoval_H

#include <opencv2/opencv.hpp>
#include "swapQ.h"
#include "FourierTransform.h"
#include <stdio.h>

cv::Mat *frecuency_filter(std::string file){

    cv::Mat image = cv::imread(file, cv::IMREAD_GRAYSCALE);
    // Creating the Low Pass Filter
    cv::Mat LPF(image.cols, image.rows, cv::COLOR_BGR2GRAY, cv::Scalar(0));
    cv::circle(LPF, cv::Point(image.cols/2, image.rows/2), 23, cv::Scalar(1), -1);

    // Creating the High Pass Filter
    cv::Mat HPF(image.cols, image.rows, cv::COLOR_BGR2GRAY, cv::Scalar(0));
    cv::circle(HPF, cv::Point(image.cols/2, image.rows/2), 18, cv::Scalar(1), -1);

    cv::Mat *f = DiscreteFourierTransform(file);
    cv::Mat fourier = f[3]; 

    cv::Mat LPI = fourier.clone();
    cv::Mat HPI = fourier.clone();

    for(int i=0;i<image.rows;i++){
        for(int j=0;j<image.cols;j++){
            if(LPF.at<double>(j,i)==1){
                LPI.at<double>(j,i)=f[4].at<double>(j,i);
            }
            if(HPF.at<double>(j,i)==0){
                HPI.at<double>(j,i)=f[5].at<double>(j,i);
            }
        }
    }

    cv::Mat f1 = InverseFourierTransform(LPI);
    cv::Mat f2 = InverseFourierTransform(HPI);
    static cv::Mat res[3] = {f1,f2,image};  // [0=Low Pass, 1=High pass, 2=original]
    return res;
}

void PrintFilter(std::string file){
    cv::Mat *f = frecuency_filter(file);
    cv::namedWindow("Original",cv::WINDOW_NORMAL);
    cv::imshow("Original", f[2]);
    cv::namedWindow("Low Pass Filter",cv::WINDOW_NORMAL);
    cv::imshow("Low Pass Filter", f[0]);
    cv::namedWindow("High Pass Filter",cv::WINDOW_NORMAL);
    cv::imshow("High Pass Filter", f[1]);
}

#endif