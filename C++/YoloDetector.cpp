#include "YoloDetector.hpp"

#include <opencv2/imgproc.hpp>

#include <fstream>
#include <iostream>
#include <set>

YoloDetector::YoloDetector(const YoloConfig& cfg)
    : cfg_(cfg)
{
}

bool YoloDetector::load()
{
    try
    {
        net_ = cv::dnn::readNetFromONNX(cfg_.modelPath);
    }
    catch (const cv::Exception& e)
    {
        std::cerr << "Failed to load ONNX model: " << e.what() << '\n';
        loaded_ = false;
        return false;
    }

    if (net_.empty())
    {
        std::cerr << "ONNX model is empty!" << '\n';
        loaded_ = false;
        return false;
    }

    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    std::ifstream file(cfg_.classesPath);
    if (!file.is_open())
    {
        std::cerr << "Could not open classes file: "
                  << cfg_.classesPath << '\n';
        loaded_ = false;
        return false;
    }

    classNames_.clear();
    std::string line;
    while (std::getline(file, line))
    {
        if (!line.empty())
            classNames_.push_back(line);
    }

    if (classNames_.empty())
    {
        std::cerr << "Classes file is empty!" << '\n';
        loaded_ = false;
        return false;
    }

    loaded_ = true;
    return true;
}

std::vector<Detection> YoloDetector::detect(const cv::Mat& frame)
{
    if (!loaded_ || frame.empty())
        return {};

    cv::Mat blob = preprocess(frame);

    net_.setInput(blob);
    cv::Mat output = net_.forward();

    return postprocess(output, frame.size());
}

bool YoloDetector::hasPersonOrAnimal(const cv::Mat& frame)
{
    return !detect(frame).empty();
}

float YoloDetector::getConfidenceThreshold() const
{
    return cfg_.confidenceThreshold;
}

void YoloDetector::setConfidenceThreshold(float value)
{
    cfg_.confidenceThreshold = value;
}

const YoloConfig& YoloDetector::getConfig() const
{
    return cfg_;
}

void YoloDetector::setConfig(const YoloConfig& cfg)
{
    cfg_ = cfg;
}

bool YoloDetector::isAllowedClass(int classId)
{
    static const std::set<int> allowed = {
        0,
        14, 15, 16, 17, 18, 19,
        20, 21, 22, 23
    };
    return allowed.count(classId) > 0;
}

cv::Mat YoloDetector::preprocess(const cv::Mat& frame) const
{
    cv::Mat blob;
    cv::dnn::blobFromImage(
        frame,
        blob,
        1.0 / 255.0,
        cv::Size(cfg_.inputWidth, cfg_.inputHeight),
        cv::Scalar(),
        true,
        false,
        CV_32F
    );
    return blob;
}

std::vector<Detection> YoloDetector::postprocess(
    const cv::Mat& output,
    const cv::Size& originalSize
) const
{
    cv::Mat out = output.reshape(1, output.size[1]);

    cv::Mat outT;
    cv::transpose(out, outT);

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> classIds;

    const int rows = outT.rows;
    const int cols = outT.cols;
    const int numClasses = cols - 4;

    const float sx = static_cast<float>(originalSize.width) / cfg_.inputWidth;
    const float sy = static_cast<float>(originalSize.height) / cfg_.inputHeight;

    for (int i = 0; i < rows; ++i)
    {
        const float* row = outT.ptr<float>(i);

        int bestId = 0;
        float bestScore = 0.f;

        for (int c = 0; c < numClasses; ++c)
        {
            if (row[4 + c] > bestScore)
            {
                bestScore = row[4 + c];
                bestId = c;
            }
        }

        if (bestScore < cfg_.confidenceThreshold)
            continue;

        if (!isAllowedClass(bestId))
            continue;

        float cx = row[0] * sx;
        float cy = row[1] * sy;
        float w = row[2] * sx;
        float h = row[3] * sy;

        int left = static_cast<int>(cx - w / 2);
        int top = static_cast<int>(cy - h / 2);

        boxes.emplace_back(left, top, static_cast<int>(w), static_cast<int>(h));
        confidences.push_back(bestScore);
        classIds.push_back(bestId);
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxes(
        boxes,
        confidences,
        cfg_.confidenceThreshold,
        cfg_.nmsThreshold,
        keep
    );

    const cv::Rect frameRect(0, 0, originalSize.width, originalSize.height);

    std::vector<Detection> result;
    result.reserve(keep.size());

    for (int idx : keep)
    {
        if (classIds[idx] < 0 ||
            classIds[idx] >= static_cast<int>(classNames_.size()))
            continue;

        cv::Rect box = boxes[idx] & frameRect;

        if (box.width <= 0 || box.height <= 0)
            continue;

        Detection d;
        d.classId = classIds[idx];
        d.className = classNames_[classIds[idx]];
        d.confidence = confidences[idx];
        d.box = box;
        result.push_back(d);
    }

    return result;
}