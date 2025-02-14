/*

Update:
2/13/2025:
Code does as intended, source input and display real time in high resolution 1080P while performing inference using optimized model models @ 640x640.

Features to add:
Possibly add a background image.

Note:
This code is not intended for any practical applications other than being used in WebCam meetings to show AI used in realtime webcam.
A presentation of factual AI coding advanced development, showing that anything is possible with computer vision.
Such as in Zoom discussions. 
I'll be the only person in the entire meeting where my webcam display box has AI inference being done in real time and I'll be the only person able to do this.

More over, doing this in Windows OS is mission impossible... made possible.

How the code works:
It uses a `akvirtualcam` virtual cam instance, then uses `akvirtualcam` C-code to inject yolov8 opencv processed inference decoded image frames into the virtual cam.
Thus the virtual cam now displays the yolov8 inference in real time.

The yolov8 code was obtain from ultralytics's ONNX CPP example and then modified to contain `akvirtualcam` virtual cam code. The Cmakelist is also modified.

------------------------------------------------
How to compile:

mkdir build
cd build
cmake ..
cmake --build . --config Release -- /m:4
------------------------------------------------

Make sure to specify the WebCam device Index. <--------------------------------------

---------------------------------------------

Make sure to first initialize `akvirtualcam` virtual cam instance before proceeding.
Here is how to create a `akvirtualcam` virtual cam instance:

Use command prompt for that:
- Go to the location where the program is built (akvirtualcamera):
- Then perform the following:
- Make sure to enter the commands as Admin !!!!!!!!!

F:
cd F:\AI_Componets\VirtualCam\akvirtualcamera\build\build\x64\Release

AkVCamManager add-device "AkVCamVideoDevice0"
AkVCamManager add-format AkVCamVideoDevice0 RGB24 1920 1080 30
AkVCamManager add-format AkVCamVideoDevice0 RGB24 640 360 30
AkVCamManager set-picture F:\AI_Componets\AI_Projects\Datasets\originals\russian_model.jpg

- To remove the picture:
AkVCamManager set-picture /

- Must needed to take affect system wide. !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
AkVCamManager update

- Other commands:
    - AkVCamManager remove-formats AkVCamVideoDevice0
    - AkVCamManager remove-device AkVCamVideoDevice0
    - AkVCamManager remove-devices
    - AkVCamManager devices
---------------------------------------------

Tools used to obtain proper aspect ratios:

https://calculateaspectratio.com/16-9-calculator

Benchmarking are disabled and to enable, simply copy the `inference_benchmark.cpp` code into inference.cpp file.


To obtain the specific frame size which Yolov8 OpenCV program is outputting, uncomment here:
std::cout << "Image size:

--------------------------------------------------

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
#define VIDEO_OUTPUT "AkVCamVideoDevice0"
#define FPS 30
#define WIDTH 1920
#define HEIGHT 1080

struct StreamProcess 
{
    HANDLE stdinReadPipe;
    HANDLE stdinWritePipe;
    SECURITY_ATTRIBUTES pipeAttributes;
    STARTUPINFOA startupInfo;
    PROCESS_INFORMATION procInfo;
};

void Detector(YOLO_V8*& p, cv::VideoCapture& cap, StreamProcess& streamProc) 
{
    cv::Mat frame;
    cv::Mat frame_1080P;

    while (cap.read(frame)) 
    {
        if (frame.empty()) 
        {
            break;
        }

        // Copy the `frame` webcam data to `frame_1080P`.
        frame.copyTo(frame_1080P);

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

        // Print the dimensions of the frame
        // std::cout << "Image size: " << frame.cols << "x" << frame.rows << std::endl;

        // Convert frame to RGB format and write to virtual camera
        cv::Mat rgbFrame;
        cv::cvtColor(frame, rgbFrame, cv::COLOR_BGR2RGB);
        cv::resize(rgbFrame, rgbFrame, cv::Size(WIDTH, HEIGHT));  // Ensure frame size matches the virtual camera

        DWORD bytesWritten = 0;
        if (!WriteFile(streamProc.stdinWritePipe, rgbFrame.data, DWORD(rgbFrame.total() * rgbFrame.elemSize()), &bytesWritten, NULL)) {
            fprintf(stderr, "Failed to write frame to pipe: %lu\n", GetLastError());
            break;
        }

        // To confirm if output is working correctly, comment it to reduce overhead. Create a window with the original frame size:-------------------------------------
        // cv::namedWindow("foo", cv::WINDOW_NORMAL);          // `foo` is the name of the window GUI.
        // cv::resizeWindow("foo", 1920, 1080);

        // cv::imshow("foo", frame);
        
        // if (cv::waitKey(1) >= 0) 
        // {
        //     break;
        // }
    }
    cv::destroyAllWindows();
}



void Classifier(YOLO_V8*& p, cv::VideoCapture& cap, StreamProcess& streamProc) {
    cv::Mat frame;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 255);

    while (cap.read(frame)) {
        if (frame.empty()) {
            break;
        }

        std::vector<DL_RESULT> res;
        p->RunSession(frame, res);

        float positionY = 50;
        for (int i = 0; i < res.size(); i++) 
        {
            int r = dis(gen);
            int g = dis(gen);
            int b = dis(gen);
            cv::putText(frame, std::to_string(i) + ":", cv::Point(10, positionY), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(b, g, r), 2);
            cv::putText(frame, std::to_string(res.at(i).confidence), cv::Point(70, positionY), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(b, g, r), 2);
            positionY += 50;
        }

        // Convert frame to RGB format and write to virtual camera
        cv::Mat rgbFrame;
        cv::cvtColor(frame, rgbFrame, cv::COLOR_BGR2RGB);
        cv::resize(rgbFrame, rgbFrame, cv::Size(WIDTH, HEIGHT));  // Ensure frame size matches the virtual camera

        DWORD bytesWritten = 0;
        if (!WriteFile(streamProc.stdinWritePipe, rgbFrame.data, DWORD(rgbFrame.total() * rgbFrame.elemSize()), &bytesWritten, NULL)) {
            fprintf(stderr, "Failed to write frame to pipe: %lu\n", GetLastError());
            break;
        }

        cv::imshow("TEST_CLS", frame);
        if (cv::waitKey(1) >= 0) {
            break;
        }
    }
    cv::destroyAllWindows();
}


int ReadCocoYaml(YOLO_V8*& p) 
{
    std::ifstream file("F:/AI_Componets/AI_Projects/ultralytics-main/examples/YOLOv8-ONNXRuntime-CPP/build/coco.yaml");
    if (!file.is_open()) {
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



void DetectTest(StreamProcess& streamProc) 
{
    YOLO_V8* yoloDetector = new YOLO_V8;
    ReadCocoYaml(yoloDetector);
    DL_INIT_PARAM params;
    params.rectConfidenceThreshold = 0.8;
    params.iouThreshold = 0.8;
    params.modelPath = "F:/AI_Componets/AI_Projects/ultralytics-main/examples/YOLOv8-ONNXRuntime-CPP/yolov8n.onnx";
    params.imgSize = { 640, 640 };  // <---- Model's expected input image inference size.

#ifdef USE_CUDA
    params.cudaEnable = true;
    params.modelType = YOLO_DETECT_V8_HALF;
// #else
//     params.modelType = YOLO_DETECT_V8_HALF;
//     params.cudaEnable = false;
#endif

    yoloDetector->CreateSession(params);

    int deviceIndex = 0; // Change this to the correct device index if needed

    cv::VideoCapture cap(deviceIndex, cv::CAP_MSMF);
    
    // Set the desired resolution !!!!!!!!!!!!!!!!!!!!!!! This is needed. OpenCV does retarded shit in the backend which reduces the source input as default. AI too retarded to catch it as a whole code.
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);

    // Check if the camera opened successfully
    if (!cap.isOpened()) 
    {
        std::cerr << "Error: Could not open camera." << std::endl;
        return;
    }

    Detector(yoloDetector, cap, streamProc);
}



void ClsTest(StreamProcess& streamProc) {
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

    Classifier(yoloDetector, cap, streamProc);
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
    StreamProcess streamProc;
    memset(&streamProc, 0, sizeof(StreamProcess));
    streamProc.stdinReadPipe = NULL;
    streamProc.stdinWritePipe = NULL;
    memset(&streamProc.pipeAttributes, 0, sizeof(SECURITY_ATTRIBUTES));
    streamProc.pipeAttributes.nLength = sizeof(SECURITY_ATTRIBUTES);
    streamProc.pipeAttributes.bInheritHandle = TRUE;
    streamProc.pipeAttributes.lpSecurityDescriptor = NULL;
    CreatePipe(&streamProc.stdinReadPipe, &streamProc.stdinWritePipe, &streamProc.pipeAttributes, 0);
    SetHandleInformation(streamProc.stdinWritePipe, HANDLE_FLAG_INHERIT, 0);

    memset(&streamProc.startupInfo, 0, sizeof(STARTUPINFOA));
    streamProc.startupInfo.cb = sizeof(STARTUPINFOA);
    streamProc.startupInfo.hStdInput = streamProc.stdinReadPipe;
    streamProc.startupInfo.dwFlags = STARTF_USESHOWWINDOW | STARTF_USESTDHANDLES;
    streamProc.startupInfo.wShowWindow = SW_HIDE;

    memset(&streamProc.procInfo, 0, sizeof(PROCESS_INFORMATION));

    // Start the stream.
    if (!CreateProcessA(NULL, cmd, NULL, NULL, TRUE, 0, NULL, NULL, &streamProc.startupInfo, &streamProc.procInfo)) {
        fprintf(stderr, "Failed to create process: %lu\n", GetLastError());
        return -1;
    }

    // Yolov8 inference call:
    DetectTest(streamProc);
    //ClsTest(streamProc);

    // Close the standard input and standard output handles
    CloseHandle(streamProc.stdinWritePipe);
    CloseHandle(streamProc.stdinReadPipe);

    // Stop the stream
    WaitForSingleObject(streamProc.procInfo.hProcess, INFINITE);
    CloseHandle(streamProc.procInfo.hProcess);
    CloseHandle(streamProc.procInfo.hThread);

    return 0;
}