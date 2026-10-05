#pragma once

#include <opencv2/core.hpp>

struct MotionConfig 
{
    int pixelThreshold = 25;
    double minChangedRatio = 0.002;
    int requiredConsecutiveFrames = 2;
    
};

class MotionDetector 
{
public:
    explicit MotionDetector(const MotionConfig& cfg);
    bool detect(const cv::Mat& frame);
    double getLastRatio() const;
    
    const MotionConfig& getConfig() const;
    void setConfig(const MotionConfig& cfg);

private:
    MotionConfig cfg_;
    cv::Mat prevGray_;
    double lastRatio_ = 0;
    int motionStreak_ = 0;

};