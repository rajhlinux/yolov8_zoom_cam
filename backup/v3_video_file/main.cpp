#include <iostream>
#include <iomanip>
#include "inference.h"
#include <filesystem>
#include <fstream>
#include <random>
#include <opencv2/opencv.hpp>

void Detector(YOLO_V8*& p, cv::VideoCapture& cap) {
    cv::Mat frame;
    while (cap.read(frame)) {
        if (frame.empty()) {
            break;
        }

        std::vector<DL_RESULT> res;
        p->RunSession(frame, res);

        for (auto& re : res) {
            cv::RNG rng(cv::getTickCount());
            cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

            cv::rectangle(frame, re.box, color, 3);

            float confidence = floor(100 * re.confidence) / 100;
            std::cout << std::fixed << std::setprecision(2);
            std::string label = p->classes[re.classId] + " " +
                std::to_string(confidence).substr(0, std::to_string(confidence).size() - 4);

            cv::rectangle(
                frame,
                cv::Point(re.box.x, re.box.y - 25),
                cv::Point(re.box.x + label.length() * 15, re.box.y),
                color,
                cv::FILLED
            );

            cv::putText(
                frame,
                label,
                cv::Point(re.box.x, re.box.y - 5),
                cv::FONT_HERSHEY_SIMPLEX,
                0.75,
                cv::Scalar(0, 0, 0),
                2
            );
        }

        cv::imshow("Result of Detection", frame);
        if (cv::waitKey(1) >= 0) {
            break;
        }
    }
    cv::destroyAllWindows();
}

void Classifier(YOLO_V8*& p, cv::VideoCapture& cap) {
    cv::Mat frame;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 255);

    while (cap.read(frame)) {
        if (frame.empty()) {
            break;
        }

        std::vector<DL_RESULT> res;
        char* ret = p->RunSession(frame, res);

        float positionY = 50;
        for (int i = 0; i < res.size(); i++) {
            int r = dis(gen);
            int g = dis(gen);
            int b = dis(gen);
            cv::putText(frame, std::to_string(i) + ":", cv::Point(10, positionY), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(b, g, r), 2);
            cv::putText(frame, std::to_string(res.at(i).confidence), cv::Point(70, positionY), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(b, g, r), 2);
            positionY += 50;
        }

        cv::imshow("TEST_CLS", frame);
        if (cv::waitKey(1) >= 0) {
            break;
        }
    }
    cv::destroyAllWindows();
}

int ReadCocoYaml(YOLO_V8*& p) {
    std::ifstream file("F:/AI_Componets/AI_Projects/ultralytics-main/examples/YOLOv8-ONNXRuntime-CPP/build/coco.yaml");
    if (!file.is_open()) {
        std::cerr << "Failed to open 'coco.yaml' file." << std::endl;
        return 1;
    }

    std::string line;
    std::vector<std::string> lines;
    while (std::getline(file, line)) {
        lines.push_back(line);
    }

    std::size_t start = 0;
    std::size_t end = 0;
    for (std::size_t i = 0; i < lines.size(); i++) {
        if (lines[i].find("names:") != std::string::npos) {
            start = i + 1;
        } else if (start > 0 && lines[i].find(':') == std::string::npos) {
            end = i;
            break;
        }
    }

    std::vector<std::string> names;
    for (std::size_t i = start; i < end; i++) {
        std::stringstream ss(lines[i]);
        std::string name;
        std::getline(ss, name, ':');
        std::getline(ss, name);
        names.push_back(name);
    }

    p->classes = names;
    return 0;
}

void DetectTest() {
    YOLO_V8* yoloDetector = new YOLO_V8;
    ReadCocoYaml(yoloDetector);
    DL_INIT_PARAM params;
    params.rectConfidenceThreshold = 0.8;
    params.iouThreshold = 0.8;
    params.modelPath = "F:/AI_Componets/AI_Projects/ultralytics-main/examples/YOLOv8-ONNXRuntime-CPP/yolov8n.onnx";
    params.imgSize = { 640, 640 };

#ifdef USE_CUDA
    params.cudaEnable = true;
    params.modelType = YOLO_DETECT_V8_HALF;

#else
    params.modelType = YOLO_DETECT_V8_HALF;
    params.cudaEnable = false;

#endif

    yoloDetector->CreateSession(params);

    std::string videoFilePath = "F:/AI_Componets/AI_Projects/Datasets/SIG_experience_center.mp4"; // Change this to the correct video file path
    cv::VideoCapture cap(videoFilePath);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file" << std::endl;
        return;
    }

    Detector(yoloDetector, cap);
}

void ClsTest() {
    YOLO_V8* yoloDetector = new YOLO_V8;
    std::string model_path = "F:/AI_Componets/AI_Projects/ultralytics-main/examples/YOLOv8-ONNXRuntime-CPP/yolov8n.onnx";
    ReadCocoYaml(yoloDetector);
    DL_INIT_PARAM params{ model_path, YOLO_CLS_HALF, {640, 640} };
    yoloDetector->CreateSession(params);

    std::string videoFilePath = "path/to/your/video.mp4"; // Change this to the correct video file path
    cv::VideoCapture cap(videoFilePath);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file" << std::endl;
        return;
    }

    Classifier(yoloDetector, cap);
}

int main() {
    DetectTest();
    //ClsTest();
    return 0;
}
