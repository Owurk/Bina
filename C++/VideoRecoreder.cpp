#include "VideoRecorder.hpp"

#include<iostream>
#include<opencv2/videoio.hpp>
#include<string>
#include<filesystem>
#include<chrono>
#include<ctime>
#include<iomanip>
#include<sstream>
#include <cmath>

VideoRecorder::VideoRecorder(double fps)
    : fps_(std::isfinite(fps) && fps > 0.0 ? fps : 20.0)
{
}

std::string makeVideoTimestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t timeNow = std::chrono::system_clock::to_time_t(now);

    std::tm localTime{};

#ifdef _WIN32
    localtime_s(&localTime, &timeNow);
#else
    localtime_r(&timeNow, &localTime);
#endif

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

void VideoRecorder::saveRecording()
{
    if (recordingFrames_.empty())
        return;

    std::error_code ec;
    std::filesystem::create_directories("videoshot", ec);

    if (ec)
    {
        std::cerr << "Failed to create videoshot directory: "
                  << ec.message() << '\n';

        recordingFrames_.clear();
        return;
    }

    const std::string timestamp = makeVideoTimestamp();

    std::string filename = detectedType_ + "_" + timestamp + ".avi";
    std::string outputPath = "videoshot/" + filename;

    for (int i = 1; ; ++i)
    {
    const bool exists = std::filesystem::exists(outputPath, ec);

    if (ec)
        break;

    if (!exists)
        break;

    filename = detectedType_ + "_" + timestamp + "_"
             + std::to_string(i) + ".avi";

    outputPath = "videoshot/" + filename;
    }

    if (ec)
    {
        std::cerr << "Failed to check output path: "
                  << ec.message() << '\n';

        recordingFrames_.clear();
        return;
    }

    const int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');

    const cv::Size frameSize(
        recordingFrames_[0].cols,
        recordingFrames_[0].rows
    );

    for (const auto& frame : recordingFrames_)
    {
    if (frame.empty() || frame.size() != frameSize)
    {
        std::cerr << "Invalid frame in recording\n";
        recordingFrames_.clear();
        return;
    }
    }

    cv::VideoWriter writer(outputPath, fourcc, fps_, frameSize);

    if (!writer.isOpened())
    {
        std::cerr << "Failed to open video file: "
                  << outputPath << '\n';

        recordingFrames_.clear();
        return;
    }

    for (const auto& frame : recordingFrames_)
        writer.write(frame);

    writer.release();

    std::cout << "Video saved: " << outputPath << '\n';
    std::cout << "Total frames: "
              << recordingFrames_.size() << '\n';

    recordingFrames_.clear();
}

void VideoRecorder::stopRecording()
{
    if(!isRecording_)
        return;

    isRecording_ = false;
    saveRecording();
}