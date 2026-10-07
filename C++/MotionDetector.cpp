#include "MotionDetector.hpp"

#include <iostream>
#include <opencv2/imgproc.hpp>

MotionDetector::MotionDetector(const MotionConfig& cfg)
    :cfg_(cfg)
{

}

double MotionDetector::getLastRatio() const
{
    return lastRatio_;
}

bool MotionDetector::detect(const cv::Mat& frame)
{
    if (frame.empty())
    {
        motionStreak_=0;
        return false;
    }

    cv::Mat currentGray;

    if (frame.channels() == 1)
    {
        currentGray = frame.clone();
    }
    else if (frame.channels() == 3)
    {
        cv::cvtColor(frame, currentGray, cv::COLOR_BGR2GRAY);
    }
    else
    {
        motionStreak_ = 0;
        return false;
    }

    cv::GaussianBlur(currentGray , currentGray , cv::Size(5,5) , 0);
    
    if (prevGray_.empty())
    {
        prevGray_ = currentGray.clone();
        return false;
    }

    
    if (currentGray.size() != prevGray_.size())
    {
        motionStreak_ = 0;
        prevGray_ = currentGray.clone();
        return false;
    }

    

    cv::Mat diff;
    cv::absdiff(currentGray, prevGray_, diff);

    cv::Mat thresholded;
    cv::threshold(
        diff,
        thresholded,
        cfg_.pixelThreshold,
        255,
        cv::THRESH_BINARY
    );

    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE ,
        cv::Size(3,3)
    );

    cv::morphologyEx(
    thresholded,
    thresholded,
    cv::MORPH_OPEN,
    kernel
    );

    int nonZero = cv::countNonZero(thresholded);

    int pixels = currentGray.rows * currentGray.cols;

    double ratio =
        static_cast<double>(nonZero) / pixels;

    lastRatio_ = ratio;

    prevGray_ = currentGray.clone();

    if(ratio >= cfg_.minChangedRatio)
        ++motionStreak_;
    else
        motionStreak_=0;

    return motionStreak_ >= cfg_.requiredConsecutiveFrames;
}

const MotionConfig& MotionDetector::getConfig() const
{
    return cfg_;
}

void MotionDetector::setConfig(const MotionConfig& cfg){
    cfg_ = cfg;
}