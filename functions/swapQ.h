#ifndef swapQ_H
#define swapQ_H

#include <opencv2/opencv.hpp>
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

#endif