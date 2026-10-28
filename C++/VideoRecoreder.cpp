#include "VideoRecorder.hpp"

#include<iostream>
#include<opencv2/videoio.hpp>
#include<string>
#include<filesystem>
#include<chrono>
#include<ctime>
#include<iomanip>
#include<sstream>


std::string makeVideoTimestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t timeNow = std::chrono::system_clock::to_time_t(now);

    std::tm localTime{};
    localtime_s(&localTime, &timeNow);

    std::ostringstream oss;
    oss << std::put_time(&localTime, "%Y%m%d_%H%M%S");

    return oss.str();
}

void VideoRecorder::addFrame(const cv::Mat& frame){
    auto now = std::chrono::steady_clock::now();
    BufferedFrame currentFrame;
    currentFrame.timestamp = now;
    currentFrame.frame = frame.clone();

    frameBuffer.push_back(currentFrame);
    

    if(isRecording_){
        if(now - detectionTime_ <limitedRecordingTime_){
            recordingFrames_.push_back(currentFrame.frame.clone());
        }else if(now - detectionTime_ >= limitedRecordingTime_){
            isRecording_ = false;
            saveRecording();
        }
    }

    while(!frameBuffer.empty() && now - frameBuffer.front().timestamp > limitedTime_){
        frameBuffer.pop_front();
    }
}

void VideoRecorder::startRecording(const std::string& detectedType)
{
    const auto now = std::chrono::steady_clock::now();

    if (isRecording_)
        return;

    if (now - lastRecordingTime_ < recordingCooldown_)
        return;

    isRecording_ = true;
    detectionTime_ = now;
    lastRecordingTime_ = now;

    recordingFrames_.clear();
    detectedType_ = detectedType;

    for (const auto& bufferedFrame : frameBuffer)
    {
        recordingFrames_.push_back(bufferedFrame.frame.clone());
    }
}

void VideoRecorder::saveRecording(){
    std::filesystem::create_directories("videoshot");
    const std::string timestamp = makeVideoTimestamp();
    std::string filename = detectedType_ + "_" + timestamp + ".avi";
    std::string outputPath = "videoshot/" + filename;

    std::error_code ec;
    for (int i = 1; std::filesystem::exists(outputPath, ec); ++i)
    {
    filename = detectedType_ + "_" + timestamp + "_"
        + std::to_string(i) + ".avi";
    outputPath = "videoshot/" + filename;
    }
    int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    double fps = 20.0;
    if(!recordingFrames_.empty()){
        cv::Size framesize(
            recordingFrames_[0].cols,
            recordingFrames_[0].rows
        );

        cv::VideoWriter writer(outputPath,fourcc,fps,framesize);
        if (!writer.isOpened()) {
    std::cerr << "Failed to open video file: " << outputPath << '\n';
    return;
    }
    
    for (const auto& frame : recordingFrames_) {
    writer.write(frame);
    }
    std::cout << "Pre-detection frames: "
          << recordingFrames_.size() << '\n';
    writer.release();
    }

}