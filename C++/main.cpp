#include "Capture.hpp"
#include "MotionDetector.hpp"
#include "YoloDetector.hpp"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <future>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace
{
    bool parseDouble(const char* str, double& out, double min, double max)
    {
        if (str == nullptr || *str == '\0')
            return false;

        char* end = nullptr;
        errno = 0;
        double value = std::strtod(str, &end);

        if (errno != 0 || end == str || *end != '\0')
            return false;

        if (!std::isfinite(value))
            return false;

        if (value < min || value > max)
            return false;

        out = value;
        return true;
    }

    bool parseFloat(const char* str, float& out, float min, float max)
    {
        if (str == nullptr || *str == '\0')
            return false;

        char* end = nullptr;
        errno = 0;
        float value = std::strtof(str, &end);

        if (errno != 0 || end == str || *end != '\0')
            return false;

        if (!std::isfinite(value))
            return false;

        if (value < min || value > max)
            return false;

        out = value;
        return true;
    }

    bool parseInt(const char* str, int& out, int min, int max)
    {
        if (str == nullptr || *str == '\0')
            return false;

        char* end = nullptr;
        errno = 0;
        long value = std::strtol(str, &end, 10);

        if (errno != 0 || end == str || *end != '\0')
            return false;

        if (value < min || value > max)
            return false;

        out = static_cast<int>(value);
        return true;
    }
}

int main(int argc, char* argv[])
{
    const std::string windowName = "Webcam";
    cv::namedWindow(windowName);
    constexpr int kMaxConsecutiveFailures = 5;
    constexpr auto kRetryDelay = std::chrono::milliseconds(30);

    Capture camera(0);
    if (!camera.open())
        return 1;

    int consecutiveFailures = 0;
    bool ok = true;

    MotionConfig motionConfig;
    YoloConfig yoloConfig;
    int yoloInterval = 10;

    if (argc > 1)
    {
        if (!parseInt(argv[1], motionConfig.pixelThreshold, 1, 255))
        {
            std::cerr << "Invalid pixelThreshold (expected 1..255): "
                << argv[1] << '\n';
            return 1;
        }
    }

    if (argc > 2)
    {
        if (!parseDouble(argv[2], motionConfig.minChangedRatio, 0.0, 1.0))
        {
            std::cerr << "Invalid minChangedRatio (expected 0..1): "
                << argv[2] << '\n';
            return 1;
        }
    }

    if (argc > 3)
    {
        if (!parseFloat(argv[3], yoloConfig.confidenceThreshold, 0.0f, 1.0f))
        {
            std::cerr << "Invalid confidenceThreshold (expected 0..1): "
                << argv[3] << '\n';
            return 1;
        }
    }

    if (argc > 4)
    {
        if (!parseInt(argv[4], yoloInterval, 1, 1000))
        {
            std::cerr << "Invalid yoloInterval (expected 1..1000): "
                << argv[4] << '\n';
            return 1;
        }
    }

    MotionDetector motionDetector(motionConfig);

    YoloDetector yoloDetector(yoloConfig);
    if (!yoloDetector.load())
        return 1;

    std::future<std::vector<Detection>> yoloFuture;
    std::vector<Detection> latestDetections;
    int frameCounter = 0;
    int framesSinceDetection = 0;
    constexpr int kDetectionDisplayFrames = 15;

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

        if (yoloFuture.valid() &&
            yoloFuture.wait_for(std::chrono::milliseconds(0)) ==
            std::future_status::ready)
        {
            try
            {
                latestDetections = yoloFuture.get();
                framesSinceDetection = 0;
            }
            catch (const cv::Exception& e)
            {
                std::cerr << "YOLO inference failed: "
                    << e.what() << '\n';
                latestDetections.clear();
            }
            catch (const std::exception& e)
            {
                std::cerr << "YOLO inference failed: "
                    << e.what() << '\n';
                latestDetections.clear();
            }
        }
        else
        {
            ++framesSinceDetection;
            if (framesSinceDetection > kDetectionDisplayFrames)
                latestDetections.clear();
        }

        bool motion = motionDetector.detect(frame);

        if (motion)
        {
            ++frameCounter;

            if (frameCounter >= yoloInterval && !yoloFuture.valid())
            {
                frameCounter = 0;

                cv::Mat yoloFrame = frame.clone();

                try
                {
                    yoloFuture = std::async(
                        std::launch::async,
                        [&yoloDetector, yoloFrame]()
                        {
                            return yoloDetector.detect(yoloFrame);
                        }
                    );
                }
                catch (const std::exception& e)
                {
                    std::cerr << "Failed to launch async YOLO: "
                        << e.what() << '\n';
                }
            }
        }

        for (const Detection& d : latestDetections)
        {
            cv::rectangle(frame, d.box, cv::Scalar(0, 255, 0), 2);

            std::string label = cv::format(
                "%s %.2f",
                d.className.c_str(),
                d.confidence
            );

            cv::putText(
                frame,
                label,
                cv::Point(d.box.x, d.box.y - 10),
                cv::FONT_HERSHEY_SIMPLEX,
                0.6,
                cv::Scalar(0, 255, 0),
                2
            );
        }

        cv::Scalar textColor = motion
            ? cv::Scalar(0, 0, 255)
            : cv::Scalar(0, 255, 0);

        std::string ratioText =
            cv::format("Ratio: %.4f", motionDetector.getLastRatio());

        std::string text = motion ? "Motion: TRUE" : "Motion: FALSE";

        cv::putText(frame, text, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0, textColor, 2);
        cv::putText(frame, ratioText, cv::Point(20, 75), cv::FONT_HERSHEY_SIMPLEX, 1.0, textColor, 2);

        std::string thresholdText = cv::format("Threshold: %.4f", motionDetector.getConfig().minChangedRatio);
        cv::putText(frame, thresholdText, cv::Point(20, 110), cv::FONT_HERSHEY_SIMPLEX, 1.0, textColor, 2);

        std::string confText = cv::format("Conf: %.2f", yoloDetector.getConfidenceThreshold());
        cv::putText(frame, confText, cv::Point(20, 145), cv::FONT_HERSHEY_SIMPLEX, 1.0, textColor, 2);

        cv::imshow(windowName, frame);

        if ((cv::waitKey(1) & 0xFF) == 'q')
            break;
    }

    if (yoloFuture.valid())
    {
        try
        {
            yoloFuture.get();
        }
        catch (const std::exception& e)
        {
            std::cerr << "YOLO inference failed at shutdown: "
                << e.what() << '\n';
        }
    }

    camera.release();
    cv::destroyWindow(windowName);

    return ok ? 0 : 1;
}