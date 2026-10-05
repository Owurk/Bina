#pragma once

#include <opencv2/core.hpp>

struct MotionConfig 
{
    int pixelThreshold = 25;
    double minChangedRatio = 0.002;
    
};

class MotionDetector 
{
public:
    explicit MotionDetector(MotionConfig cfg);
    bool detect(const cv::Mat& frame);
    double getLastRatio() const;
    

private:
    MotionConfig cfg_;
    cv::Mat prevGray_;
    double lastRatio_ = 0;

};