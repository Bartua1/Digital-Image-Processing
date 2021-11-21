#ifndef anisotropic_H
#define anisotropic_H

#include <opencv2/opencv.hpp>
float g_function(cv::Mat image, int b, int a, int dir){
    float res;
    switch (dir)
    {
    case 0:
        /* North */
        res = exp(-pow((image.at<float>(b-1,a)-image.at<float>(b,a)),2)/pow(0.015,2));
        break;
    case 1:
        /* South */
        res = exp(-pow((image.at<float>(b+1,a)-image.at<float>(b,a)),2)/pow(0.015,2));
        break;
    case 2:
        /* East */
        res = exp(-pow((image.at<float>(b,a+1)-image.at<float>(b,a)),2)/pow(0.015,2));
        break;
    default:
        /* West */
        res = exp(-pow((image.at<float>(b,a-1)-image.at<float>(b,a)),2)/pow(0.015,2));
        break;
    }
    return res;
}

void anisotropic_at(cv::Mat image, cv::Mat res, int b, int a){
    float dirs [4] = {g_function(image,b,a,0),g_function(image,b,a,1),g_function(image,b,a,2),g_function(image,b,a,3)};
    float left = image.at<float>(b,a)*(1-0.1*(dirs[0]+dirs[1]+dirs[2]+dirs[3]));
    float right = 0.1*(dirs[0]*image.at<float>(b-1,a)+dirs[1]*image.at<float>(b+1,a)+dirs[2]*image.at<float>(b,a+1)+dirs[3]*image.at<float>(b,a-1));
    res.at<float>(b,a) = left+right;
}

cv::Mat anisotropic(cv::Mat image, int iterations){
    cv::Mat res = image.clone();
    while(iterations>0){
        for (int y = 1; y < image.rows-1; y++) {
            for (int x = 1; x < image.cols-1; x++) {
                anisotropic_at(image,res,y,x);
            }
        }
        image = res.clone();
        iterations--;
    }
    return res;
}

void printAnisotropic(){
    cv::Mat src_8uc3_img = cv::imread("/home/bartu/Documents/dzo_vsc/images/lena.png", cv::IMREAD_COLOR); // load color image from file system to Mat variable, this will be loaded using 8 bits (uchar)

    if (src_8uc3_img.empty()) {
        printf("Unable to read input file (%s, %d).", __FILE__, __LINE__);
    }

    cv::Mat gray_8uc1_img; // declare variable to hold grayscale version of img variable, gray levels wil be represented using 8 bits (uchar)
    cv::Mat gray_32fc1_img; // declare variable to hold grayscale version of img variable, gray levels wil be represented using 32 bits (float)

    cv::cvtColor(src_8uc3_img, gray_8uc1_img, cv::COLOR_BGR2GRAY); // convert input color image to grayscale one, CV_BGR2GRAY specifies direction of conversion
    gray_8uc1_img.convertTo(gray_32fc1_img, CV_32FC1, 1.0/255.0); // convert grayscale image from 8 bits to 32 bits, resulting values will be in the interval 0.0 - 1.0

    /* anisotropic algorith (exercise 3) */
    cv::Mat test = gray_32fc1_img.clone();
    cv::Mat image2 = anisotropic(gray_32fc1_img,1000);

    cv::imshow("after", image2);
    cv::imshow("before", test);
}

#endif
