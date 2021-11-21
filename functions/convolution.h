#ifndef convolution_H
#define convolution_H

#include <opencv2/opencv.hpp>
void convolution_pixel(cv::Mat image,int b, int a){
    int matrix[3][3] = {{1,2,1},{2,4,2},{1,2,1}};
    int sum = 0;
    uchar paco = 0;
    for (int y = b-1; y <= b+1; y++) {
        for (int x = a-1; x <= a+1; x++) {
            paco = image.at<uchar>(y, x);
            sum+=matrix[y-b+1][x-a+1]*paco;
        }
    }
    sum=sum/16;
    image.at<uchar>(b,a) = sum;
}

void convolution(cv::Mat image){
    for (int y = 1; y < image.rows-1; y++) {
        for (int x = 1; x < image.cols-1; x++) {
            convolution_pixel(image,y,x);
        }
    }
}

void printConvolution(){
    cv::Mat src_8uc3_img = cv::imread("/home/bartu/Documents/dzo_vsc/images/lena.png", cv::IMREAD_COLOR); // load color image from file system to Mat variable, this will be loaded using 8 bits (uchar)

    if (src_8uc3_img.empty()) {
        printf("Unable to read input file (%s, %d).", __FILE__, __LINE__);
    }

    cv::Mat gray_8uc1_img; // declare variable to hold grayscale version of img variable, gray levels wil be represented using 8 bits (uchar)
    cv::Mat gray_32fc1_img; // declare variable to hold grayscale version of img variable, gray levels wil be represented using 32 bits (float)

    cv::cvtColor(src_8uc3_img, gray_8uc1_img, cv::COLOR_BGR2GRAY); // convert input color image to grayscale one, CV_BGR2GRAY specifies direction of conversion
    gray_8uc1_img.convertTo(gray_32fc1_img, CV_32FC1, 1.0/255.0); // convert grayscale image from 8 bits to 32 bits, resulting values will be in the interval 0.0 - 1.0

    /* Convolution algorithm (exercise 2) */
    convolution(gray_8uc1_img);

    // diplay images
    cv::imshow( "Gradient", gray_8uc1_img);
}

#endif