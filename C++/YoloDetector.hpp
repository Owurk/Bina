#pragma once

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include <string>
#include <vector>

struct YoloConfig
{
    std::string modelPath = "models/yolov8n.onnx";
    std::string classesPath = "models/coco.names";
    float confidenceThreshold = 0.5f;
    float nmsThreshold = 0.45f;
    int inputWidth = 640;
    int inputHeight = 640;
};

struct Detection
{
    int classId = -1;
    std::string className;
    float confidence = 0.f;
    cv::Rect box;
};

class YoloDetector
{
public:
    explicit YoloDetector(const YoloConfig& cfg = YoloConfig());

    bool load();

    std::vector<Detection> detect(const cv::Mat& frame);

    bool hasPersonOrAnimal(const cv::Mat& frame);

    float getConfidenceThreshold() const;
    void setConfidenceThreshold(float value);

    const YoloConfig& getConfig() const;
    void setConfig(const YoloConfig& cfg);

private:
    YoloConfig cfg_;
    cv::dnn::Net net_;
    std::vector<std::string> classNames_;
    bool loaded_ = false;

    static bool isAllowedClass(int classId);

    cv::Mat preprocess(const cv::Mat& frame) const;

    std::vector<Detection> postprocess(
        const cv::Mat& output,
        const cv::Size& originalSize
    ) const;
};