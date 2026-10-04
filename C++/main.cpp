#include "Capture.hpp"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <chrono>
#include <iostream>
#include <string>
#include <thread>

int main()
{
    const std::string windowName = "Webcam";
    constexpr int kMaxConsecutiveFailures = 5;
    constexpr auto kRetryDelay = std::chrono::milliseconds(30);

    Capture camera(0);
    if (!camera.open())
        return 1;

    int consecutiveFailures = 0;
    bool ok = true;

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

        cv::imshow(windowName, frame);

        if (cv::waitKey(1) == 'q')
            break;
    }

    camera.release();
    cv::destroyWindow(windowName);

    return ok ? 0 : 1;
}