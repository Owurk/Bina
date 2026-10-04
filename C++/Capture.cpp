#include "Capture.hpp"

#include <iostream>

Capture::Capture(int index)
    : cameraIndex_(index)
{
}

bool Capture::open()
{
    if (isOpened())
        return true;

    cap_.open(cameraIndex_);
    if (!cap_.isOpened())
    {
        std::cerr << "Could not open webcam!" << '\n';
        return false;
    }
    return true;
}

bool Capture::read(cv::Mat& frame)
{
    if (!isOpened())
    {
        std::cerr << "Camera is not opened!" << '\n';
        return false;
    }
    return cap_.read(frame);
}

void Capture::release()
{
    if (isOpened())
        cap_.release();
}

bool Capture::isOpened() const
{
    return cap_.isOpened();
}