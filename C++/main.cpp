#include "Capture.hpp"
#include "MotionDetector.hpp"
#include "YoloDetector.hpp"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <future>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

int main(int argc, char* argv[])
{
    const std::string windowName = "Webcam";
    cv::namedWindow(windowName);
    constexpr int kMaxConsecutiveFailures = 5;
    constexpr auto kRetryDelay = std::chrono::milliseconds(30);
    constexpr int kYoloInterval = 1;

    Capture camera(0);
    if (!camera.open())
        return 1;

    int consecutiveFailures = 0;
    bool ok = true;

    MotionConfig motionConfig;

    if (argc > 1)
        motionConfig.pixelThreshold = std::stoi(argv[1]);
    if (argc > 2)
        motionConfig.minChangedRatio = std::stod(argv[2]);

    MotionDetector motionDetector(motionConfig);

    YoloConfig yoloConfig;
    YoloDetector yoloDetector(yoloConfig);
    if (!yoloDetector.load())
        return 1;

    std::future<std::vector<Detection>> yoloFuture;
    std::vector<Detection> latestDetections;
    int frameCounter = 0;

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

        bool motion = motionDetector.detect(frame);

        if (motion && ++frameCounter >= kYoloInterval)
        {
            frameCounter = 0;

            bool yoloIdle = !yoloFuture.valid() ||
                yoloFuture.wait_for(std::chrono::milliseconds(0)) ==
                std::future_status::ready;

            if (yoloIdle)
            {
                if (yoloFuture.valid())
                    latestDetections = yoloFuture.get();

                cv::Mat yoloFrame = frame.clone();

                yoloFuture = std::async(
                    std::launch::async,
                    [&yoloDetector, yoloFrame]()
                    {
                        return yoloDetector.detect(yoloFrame);
                    }
                );
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

        cv::imshow(windowName, frame);

        if ((cv::waitKey(1) & 0xFF) == 'q')
            break;
    }

    if (yoloFuture.valid())
        yoloFuture.wait();

    camera.release();
    cv::destroyWindow(windowName);

    return ok ? 0 : 1;
}