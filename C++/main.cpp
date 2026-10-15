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
#include <ctime>
#include <filesystem>
#include <future>
#include <iomanip>
#include <iostream>
#include <sstream>
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

    bool isPersonOrAnimal(const std::string& className)
    {
        static const char* kTargetClasses[] = {
            "person",
            "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe"
        };

        for (const char* target : kTargetClasses)
        {
            if (className == target)
                return true;
        }
        return false;
    }

    
    std::string makeSnapshotTimestamp()
    {
        const std::time_t now = std::chrono::system_clock::to_time_t(
            std::chrono::system_clock::now());

        std::tm localTime{};
#ifdef _WIN32
        localtime_s(&localTime, &now);
#else
        localtime_r(&now, &localTime);
#endif

        std::ostringstream oss;
        oss << std::put_time(&localTime, "%Y%m%d_%H%M%S");
        return oss.str();
    }

    
    std::string buildSnapshotPath(const std::string& dir)
    {
        const std::string timestamp = makeSnapshotTimestamp();
        std::string path = dir + "/detect_" + timestamp + ".jpg";

        
        std::error_code ec;
        for (int i = 1; std::filesystem::exists(path, ec); ++i)
        {
            path = dir + "/detect_" + timestamp + "_"
                + std::to_string(i) + ".jpg";
        }
        return path;
    }

    
    bool saveSnapshot(const std::string& dir,
                      const cv::Mat& frame,
                      const std::vector<Detection>& detections)
    {
        cv::Mat annotated = frame.clone();

        for (const Detection& d : detections)
        {
            cv::rectangle(annotated, d.box, cv::Scalar(0, 255, 0), 2);

            std::string label = cv::format(
                "%s %.2f",
                d.className.c_str(),
                d.confidence
            );

            int baseline = 0;
            const cv::Size textSize = cv::getTextSize(
                label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);

            cv::Point labelPos(d.box.x, d.box.y - 10);
            if (labelPos.y < textSize.height) 
                labelPos.y = d.box.y + d.box.height + textSize.height + 5;

            cv::putText(annotated, label, labelPos,
                cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 255, 0), 2);
        }

        const std::string path = buildSnapshotPath(dir);

        if (!cv::imwrite(path, annotated))
        {
            std::cerr << "Failed to write snapshot: " << path << '\n';
            return false;
        }

        std::cout << "Snapshot saved: " << path << '\n';
        return true;
    }
}

int main(int argc, char* argv[])
{
    MotionConfig motionConfig;
    YoloConfig yoloConfig;
    int yoloInterval = 10;
    int cameraIndex = 0;

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

    if (argc > 5)
    {
        if (!parseInt(argv[5], cameraIndex, 0, 10))
        {
            std::cerr << "Invalid cameraIndex (expected 0..10): "
                << argv[5] << '\n';
            return 1;
        }
    }

    if (argc == 1)
    {
        std::cout << "Usage: bina [pixelThreshold] [minChangedRatio] "
            << "[confidenceThreshold] [yoloInterval] [cameraIndex]\n"
            << "  pixelThreshold     : 1..255  (default 25)\n"
            << "  minChangedRatio    : 0..1    (default 0.002)\n"
            << "  confidenceThreshold: 0..1    (default 0.5)\n"
            << "  yoloInterval       : 1..1000 (default 10)\n"
            << "  cameraIndex        : 0..10   (default 0)\n";
    }

    
    const std::string snapshotDir = "snapshots";
    {
        std::error_code fsError;
        std::filesystem::create_directories(snapshotDir, fsError);
        if (fsError)
        {
            std::cerr << "Failed to create snapshots directory: "
                << fsError.message() << '\n';
            return 1;
        }
    }

    MotionDetector motionDetector(motionConfig);

    YoloDetector yoloDetector(yoloConfig);
    if (!yoloDetector.load())
        return 1;

    const std::string windowName = "Webcam";
    cv::namedWindow(windowName);
    constexpr int kMaxConsecutiveFailures = 5;
    constexpr auto kRetryDelay = std::chrono::milliseconds(30);

    Capture camera(cameraIndex);
    if (!camera.open())
        return 1;

    int consecutiveFailures = 0;
    bool ok = true;

    std::future<std::vector<Detection>> yoloFuture;
    std::vector<Detection> latestDetections;
    int frameCounter = 0;
    int framesSinceDetection = 0;
    constexpr int kDetectionDisplayFrames = 90;

    
    constexpr auto kSnapshotCooldown = std::chrono::seconds(5);
    std::chrono::steady_clock::time_point lastSnapshotTime =
        std::chrono::steady_clock::now() - kSnapshotCooldown;

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

        
        bool personOrAnimalPresent = false;
        for (const Detection& d : latestDetections)
        {
            if (isPersonOrAnimal(d.className))
            {
                personOrAnimalPresent = true;
                break;
            }
        }

        const auto currentTime = std::chrono::steady_clock::now();
        if (motion && personOrAnimalPresent &&
            currentTime - lastSnapshotTime >= kSnapshotCooldown)
        {
            lastSnapshotTime = currentTime;
            saveSnapshot(snapshotDir, frame, latestDetections);
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

        std::string camText = cv::format("Cam: %d", cameraIndex);
        cv::putText(frame, camText, cv::Point(20, 180), cv::FONT_HERSHEY_SIMPLEX, 1.0, textColor, 2);

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