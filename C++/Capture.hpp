#pragma once

#include <opencv2/opencv.hpp>

class Capture
{
private:
    cv::VideoCapture cap;
    int cameraIndex;

public:
    explicit Capture(int index = 0);

    bool start();
};