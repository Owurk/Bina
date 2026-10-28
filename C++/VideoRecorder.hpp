#pragma once

#include <opencv2/core.hpp>
#include<deque>
#include<chrono>
#include <vector>
#include<string>

class VideoRecorder{
private:
    struct BufferedFrame
    {
        cv::Mat frame;
        std::chrono::steady_clock::time_point timestamp;
    };
    
    std::deque<BufferedFrame> frameBuffer;
    std::chrono::seconds limitedTime_{2};
    std::chrono::seconds limitedRecordingTime_{1};
    std::chrono::seconds recordingCooldown_{2};
    std::chrono::steady_clock::time_point lastRecordingTime_{};
    bool isRecording_ = false;
    std::chrono::steady_clock::time_point detectionTime_;
    std::vector<cv::Mat> recordingFrames_;
    std::string detectedType_;
public:
    void addFrame(const cv::Mat&);
    void startRecording(const std::string& );
    void saveRecording();
};