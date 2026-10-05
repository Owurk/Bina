#include "Capture.hpp"
#include "MotionDetector.hpp"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <iostream>
#include <string>
#include <thread>

int main(int argc , char* argv[])
{
    const std::string windowName = "Webcam";
    constexpr int kMaxConsecutiveFailures = 5;
    constexpr auto kRetryDelay = std::chrono::milliseconds(30);

    Capture camera(0);
    if (!camera.open())
        return 1;

    int consecutiveFailures = 0;
    bool ok = true;

    MotionConfig config;

    if(argc > 1 )
        config.pixelThreshold = std::stoi(argv[1]);
    if(argc > 2)
        config.minChangedRatio = std::stod(argv[2]);

    MotionDetector detector(config);

    while (true)
    {
        cv::Mat frame;

        if (!camera.read(frame) || frame.empty())
        {
            
            if (++consecutiveFailures >= kMaxConsecutiveFailures)
            {
                std::cerr << "Stream lost after " << consecutiveFailures
                          << " consecutive failed reads." << '\n';
                ok = false;
                break;
            }
            std::this_thread::sleep_for(kRetryDelay);
            continue;
        }

        consecutiveFailures = 0;

        bool motion = detector.detect(frame);

        cv::Scalar textColor = motion
            ? cv::Scalar(0, 0, 255)
            : cv::Scalar(0, 255, 0);  
            
        std::string ratioText =
        "Ratio: " + std::to_string(detector.getLastRatio());

        std::string text = motion ? "Motion: TRUE" : "Motion: FALSE";

        cv::putText(frame , text , cv::Point(20,40) , cv::FONT_HERSHEY_SIMPLEX , 1.0 , textColor ,2);
        cv::putText(frame , ratioText , cv::Point(20,75) , cv::FONT_HERSHEY_SIMPLEX , 1.0 , textColor ,2); 
        
        std::string thresholdText = "Threshold: " + std::to_string(config.minChangedRatio);
         cv::putText(frame , thresholdText , cv::Point(20,110) , cv::FONT_HERSHEY_SIMPLEX , 1.0 , textColor ,2); 

        cv::imshow(windowName, frame);

        if (cv::waitKey(1) == 'q')
            break;
    }

    camera.release();
    cv::destroyWindow(windowName);

    return ok ? 0 : 1;
}