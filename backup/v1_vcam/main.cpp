/*

Webcam Mode , injects frames to akvirtualcam.

How to compile:

mkdir build
cd build
cmake ..
cmake --build . --config Release -- /m:4

Make sure to specify the WebCam Index.

*/


#include <iostream>
#include <iomanip>
#include "inference.h"
#include <filesystem>
#include <fstream>
#include <random>
#include <opencv2/opencv.hpp>

// For akvirtualcamera:
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

extern "C"  // FFMPEG Stuff, needed for akvirtualcamera:
{
    #include <libavformat/avformat.h>
    #include <libavcodec/avcodec.h>
    #include <libswscale/swscale.h>
    #include <libavutil/opt.h>
    #include <libavutil/imgutils.h>
    #include <libavdevice/avdevice.h>
}

// Virtual Cam Output parameters:
#define VIDEO_OUTPUT "AkVCamVideoDevice0"  // <-----------------
#define FPS 30
#define WIDTH 640
#define HEIGHT 640

// For akvirtualcamera:
struct StreamProcess 
{
    HANDLE stdinReadPipe;
    HANDLE stdinWritePipe;
    SECURITY_ATTRIBUTES pipeAttributes;
    STARTUPINFOA startupInfo;
    PROCESS_INFORMATION procInfo;
};

void Detector(YOLO_V8*& p, cv::VideoCapture& cap, HANDLE stdinWritePipe) 
{
    cv::Mat frame;
    while (cap.read(frame)) 
    {
        if (frame.empty()) 
        {
            break;
        }

        std::vector<DL_RESULT> res;
        p->RunSession(frame, res);

        for (auto& re : res) 
        {
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
        if (cv::waitKey(1) >= 0) 
        {
            break;
        }

        // Convert the frame to RGB24 format
        cv::Mat rgbFrame;
        cv::cvtColor(frame, rgbFrame, cv::COLOR_BGR2RGB);

        // Send the frame to the virtual camera
        DWORD bytesWritten = 0;
        if (!WriteFile(stdinWritePipe,
                       rgbFrame.data,
                       DWORD(rgbFrame.total() * rgbFrame.elemSize()),
                       &bytesWritten,
                       NULL)) {
            fprintf(stderr, "Failed to write frame to pipe: %lu\n", GetLastError());
            break;
        }
    }
    cv::destroyAllWindows();
}

void Classifier(YOLO_V8*& p, cv::VideoCapture& cap, HANDLE stdinWritePipe) {
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

        // Convert the frame to RGB24 format
        cv::Mat rgbFrame;
        cv::cvtColor(frame, rgbFrame, cv::COLOR_BGR2RGB);

        // Send the frame to the virtual camera
        DWORD bytesWritten = 0;
        if (!WriteFile(stdinWritePipe,
                       rgbFrame.data,
                       DWORD(rgbFrame.total() * rgbFrame.elemSize()),
                       &bytesWritten,
                       NULL)) {
            fprintf(stderr, "Failed to write frame to pipe: %lu\n", GetLastError());
            break;
        }
    }
    cv::destroyAllWindows();
}

int ReadCocoYaml(YOLO_V8*& p) 
{
    std::ifstream file("F:/AI_Componets/AI_Projects/ultralytics-main/examples/YOLOv8-ONNXRuntime-CPP/build/coco.yaml");
    if (!file.is_open()) 
    {
        std::cerr << "Failed to open 'coco.yaml' file." << std::endl;
        return 1;
    }

    std::string line;
    std::vector<std::string> lines;
    while (std::getline(file, line)) 
    {
        lines.push_back(line);
    }

    std::size_t start = 0;
    std::size_t end = 0;
    for (std::size_t i = 0; i < lines.size(); i++) 
    {
        if (lines[i].find("names:") != std::string::npos) 
        {
            start = i + 1;
        } 
        else if (start > 0 && lines[i].find(':') == std::string::npos) 
        {
            end = i;
            break;
        }
    }

    std::vector<std::string> names;
    for (std::size_t i = start; i < end; i++) 
    {
        std::stringstream ss(lines[i]);
        std::string name;
        std::getline(ss, name, ':');
        std::getline(ss, name);
        names.push_back(name);
    }

    p->classes = names;
    return 0;
}

void DetectTest(HANDLE stdinWritePipe) 
{
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

    int deviceIndex = 0; // Change this to the correct device index if needed
    cv::VideoCapture cap(deviceIndex, cv::CAP_MSMF);
    if (!cap.isOpened()) 
    {
        std::cerr << "Error opening video stream or file" << std::endl;
        return;
    }

    Detector(yoloDetector, cap, stdinWritePipe);
}

void ClsTest(HANDLE stdinWritePipe) {
    YOLO_V8* yoloDetector = new YOLO_V8;
    std::string model_path = "F:/AI_Componets/AI_Projects/ultralytics-main/examples/YOLOv8-ONNXRuntime-CPP/yolov8n.onnx";
    ReadCocoYaml(yoloDetector);
    DL_INIT_PARAM params{ model_path, YOLO_CLS_HALF, {640, 640} };
    yoloDetector->CreateSession(params);

    int deviceIndex = 0; // Change this to the correct device index if needed
    cv::VideoCapture cap(deviceIndex, cv::CAP_MSMF);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream or file" << std::endl;
        return;
    }

    Classifier(yoloDetector, cap, stdinWritePipe);
}

int main() 
{
    // Set the parameters of the stream.
    char cmd[1024];
    const char format[] = "RGB24";
    memset(cmd, 0, 1024);
    snprintf(cmd,
             1024,
             "F:\\AI_Componets\\VirtualCam\\akvirtualcamera\\build\\build\\x64\\Release\\AkVCamManager.exe stream %s %s %d %d",
             VIDEO_OUTPUT,
             format,
             WIDTH,
             HEIGHT);

    // Get the handles to the standard input and standard output.
    struct StreamProcess streamProc;
    memset(&streamProc, 0, sizeof(struct StreamProcess));
    streamProc.stdinReadPipe = NULL;
    streamProc.stdinWritePipe = NULL;
    memset(&streamProc.pipeAttributes, 0, sizeof(SECURITY_ATTRIBUTES));
    streamProc.pipeAttributes.nLength = sizeof(SECURITY_ATTRIBUTES);
    streamProc.pipeAttributes.bInheritHandle = TRUE;
    streamProc.pipeAttributes.lpSecurityDescriptor = NULL;
    CreatePipe(&streamProc.stdinReadPipe,
               &streamProc.stdinWritePipe,
               &streamProc.pipeAttributes,
               0);
    SetHandleInformation(streamProc.stdinWritePipe,
                         HANDLE_FLAG_INHERIT,
                         0);

    memset(&streamProc.startupInfo, 0, sizeof(STARTUPINFOA));
    streamProc.startupInfo.cb = sizeof(STARTUPINFOA);
    streamProc.startupInfo.hStdInput = streamProc.stdinReadPipe;
    streamProc.startupInfo.dwFlags = STARTF_USESHOWWINDOW | STARTF_USESTDHANDLES;
    streamProc.startupInfo.wShowWindow = SW_HIDE;

    memset(&streamProc.procInfo, 0, sizeof(PROCESS_INFORMATION));

    // Start the stream.
    if (!CreateProcessA(NULL,
                        cmd,
                        NULL,
                        NULL,
                        TRUE,
                        0,
                        NULL,
                        NULL,
                        &streamProc.startupInfo,
                        &streamProc.procInfo)) {
        fprintf(stderr, "Failed to create process: %lu\n", GetLastError());
        return -1;
    }

    // Yolov8 inference call:
    DetectTest(streamProc.stdinWritePipe);
    //ClsTest(streamProc.stdinWritePipe);

    // Close the standard input and standard output handles
    CloseHandle(streamProc.stdinWritePipe);
    CloseHandle(streamProc.stdinReadPipe);

    // Stop the stream
    WaitForSingleObject(streamProc.procInfo.hProcess, INFINITE);
    CloseHandle(streamProc.procInfo.hProcess);
    CloseHandle(streamProc.procInfo.hThread);

    return 0;
}