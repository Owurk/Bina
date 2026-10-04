#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

class Capture
{
public:
    explicit Capture(int index = 0);

    bool open();
    bool read(cv::Mat& frame);
    void release();
    bool isOpened() const;

private:
    cv::VideoCapture cap_;
    int cameraIndex_;
};