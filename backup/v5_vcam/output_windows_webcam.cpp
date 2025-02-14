/*
https://chat.mistral.ai/chat/b71da884-3399-48fb-87d1-d681a53c817c

Real time Rendering

Uses FFMPEG libraries to open and decode a live stream webcam_a.
Then simply copies those video frames and injects into virtual webcam_b.

From Webcam_A to Virtual_Webcam_B.

Injects a Video to virtual cam.
Note: Example uses CPU for decoding.

Make sure to obtain FFMPEG Header files and also built bin files:
Obtain from:
https://www.gyan.dev/ffmpeg/builds/

FFMPEG is used for image processing.

Make sure to build akvirtualcamera to create virtual camera instances.
Obtain from github:
https://github.com/webcamoid/akvirtualcamera

#Make sure to create virtual cam instance.
Use command prompt for that:
- Go to the location where the program is built (akvirtualcamera):
- Then perform the following:
F:
cd F:\AI_Componets\VirtualCam\akvirtualcamera\build\build\x64\Release
    - AkVCamManager add-device "AkVCamVideoDevice0"
    - AkVCamManager add-format AkVCamVideoDevice0 RGB24 1920 1080 30
    - AkVCamManager set-picture F:\AI_Componets\AI_Projects\Datasets\originals\russian_model.jpg
        - To remove the picture:
            - AkVCamManager set-picture /
    - AkVCamManager update
        - Must needed to take affect system wide.
- Other commands:
    - AkVCamManager remove-formats AkVCamVideoDevice0
    - AkVCamManager remove-device AkVCamVideoDevice0
    - AkVCamManager remove-devices
    - AkVCamManager devices


Now inject video frames into the virtual cam using this C code, in C++ file and C++ compilation mode:

Tested: 1/17/2025 @ 11:31PM
Working Code.


How to compile:

#set path variables since windows path can not store more paths:
set PATH=%PATH%;F:\AI_Componets\FFMPEG\ffmpeg-7.0.2-full_build-shared\bin

#Set MVSC Compiler:
"F:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Auxiliary\\Build\\vcvarsall.bat" x64

# Compile code:
cl /EHsc /I "F:\AI_Componets\FFMPEG\ffmpeg-7.0.2-full_build-shared\include" output_windows_webcam.cpp /link /LIBPATH:"F:\AI_Componets\FFMPEG\ffmpeg-7.0.2-full_build-shared\lib" /OUT:output_windows_webcam.exe avformat.lib avcodec.lib avutil.lib swscale.lib avdevice.lib
*/



#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

extern "C"
{
    #include <libavformat/avformat.h>
    #include <libavcodec/avcodec.h>
    #include <libswscale/swscale.h>
    #include <libavutil/opt.h>
    #include <libavutil/imgutils.h>
    #include <libavdevice/avdevice.h>
}

// Virtual Cam Output parameters:

#define VIDEO_OUTPUT "AkVCamVideoDevice0"  //<-----------------
#define FPS 60
#define WIDTH 640
#define HEIGHT 640

struct StreamProcess {
    HANDLE stdinReadPipe;
    HANDLE stdinWritePipe;
    SECURITY_ATTRIBUTES pipeAttributes;
    STARTUPINFOA startupInfo;
    PROCESS_INFORMATION procInfo;
};

int main() {
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

    // Initialize FFmpeg
    avdevice_register_all();
    avformat_network_init();

    // Open the webcam
    AVFormatContext *pFormatCtx = NULL;
    AVDictionary *options = NULL;
    av_dict_set(&options, "framerate", "60", 0);
    av_dict_set(&options, "video_size", "1920x1080", 0);
    //av_dict_set(&options, "pixel_format", "yuvj422p", 0); // Use the supported pixel format
    av_dict_set(&options, "rtbufsize", "0.1M", 0); // Increase buffer size to cause latency.
    //av_dict_set(&options, "fflags", "nobuffer", 0); // Reduce latency   

    const char *device_name = "video=@device_pnp_\\\\?\\usb#vid_eba4&pid_7588&mi_02#6&63de0d9&0&0002#{65e8773d-8f56-11d0-a3b9-00a0c9223196}\\global";
    const AVInputFormat *input_format = av_find_input_format("dshow");
    if (avformat_open_input(&pFormatCtx, device_name, input_format, &options) != 0) {
        fprintf(stderr, "Could not open webcam\n");
        return -1;
    }

    // Retrieve stream information
    if (avformat_find_stream_info(pFormatCtx, NULL) < 0) {
        fprintf(stderr, "Could not retrieve stream information\n");
        return -1;
    }

    // Find the first video stream
    int videoStream = -1;
    for (int i = 0; i < pFormatCtx->nb_streams; i++) {
        if (pFormatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            videoStream = i;
            break;
        }
    }

    if (videoStream == -1) {
        fprintf(stderr, "Could not find a video stream\n");
        return -1;
    }

    // Get a pointer to the codec context for the video stream
    AVCodecParameters *pCodecParams = pFormatCtx->streams[videoStream]->codecpar;
    const AVCodec *pCodec = avcodec_find_decoder(pCodecParams->codec_id);
    if (pCodec == NULL) {
        fprintf(stderr, "Unsupported codec!\n");
        return -1;
    }

    // Open the codec
    AVCodecContext *pCodecCtx = avcodec_alloc_context3(pCodec);
    if (avcodec_parameters_to_context(pCodecCtx, pCodecParams) < 0) {
        fprintf(stderr, "Could not copy codec context\n");
        return -1;
    }
    if (avcodec_open2(pCodecCtx, pCodec, NULL) < 0) {
        fprintf(stderr, "Could not open codec\n");
        return -1;
    }

    // Allocate video frame
    AVFrame *pFrame = av_frame_alloc();
    AVFrame *pFrameRGB = av_frame_alloc();
    if (pFrameRGB == NULL) {
        fprintf(stderr, "Could not allocate video frame\n");
        return -1;
    }

    // Allocate an AVFrame structure for the RGB image
    pFrameRGB->format = AV_PIX_FMT_RGB24;
    pFrameRGB->width = WIDTH;
    pFrameRGB->height = HEIGHT;
    av_frame_get_buffer(pFrameRGB, 32);

    // Initialize SWS context for software scaling with the correct color range
    struct SwsContext *sws_ctx = sws_getContext(pCodecCtx->width,
                                                pCodecCtx->height,
                                                pCodecCtx->pix_fmt,
                                                WIDTH,
                                                HEIGHT,
                                                AV_PIX_FMT_RGB24,
                                                SWS_BILINEAR,
                                                NULL,
                                                NULL,
                                                NULL);

// Read frames and send to virtual camera
AVPacket packet;
while (av_read_frame(pFormatCtx, &packet) >= 0) 
{
    if (packet.stream_index == videoStream) {
        // Decode video frame
        avcodec_send_packet(pCodecCtx, &packet);
        while (avcodec_receive_frame(pCodecCtx, pFrame) == 0) 
        {
            // Convert the image from its native format to RGB
            sws_scale(sws_ctx, (uint8_t const * const *)pFrame->data,
                      pFrame->linesize, 0, pCodecCtx->height,
                      pFrameRGB->data, pFrameRGB->linesize);

            // Debug output: Print frame dimensions and format
            printf("Decoded frame: width=%d, height=%d, format=%d\n", pFrameRGB->width, pFrameRGB->height, pFrameRGB->format);
            printf("Frame data pointers: %p, %p, %p\n", pFrameRGB->data[0], pFrameRGB->data[1], pFrameRGB->data[2]);
            printf("Frame linesize: %d, %d, %d\n", pFrameRGB->linesize[0], pFrameRGB->linesize[1], pFrameRGB->linesize[2]);

            // Verify the format is RGB24
            if (pFrameRGB->format == AV_PIX_FMT_RGB24) {
                printf("Frame is in RGB24 format\n");
            } else {
                fprintf(stderr, "Frame is not in RGB24 format\n");
            }

            // Send the frame to the virtual camera
            DWORD bytesWritten = 0;
            if (!WriteFile(streamProc.stdinWritePipe,
                           pFrameRGB->data[0],
                           DWORD(pFrameRGB->linesize[0] * HEIGHT),
                           &bytesWritten,
                           NULL)) {
                fprintf(stderr, "Failed to write frame to pipe: %lu\n", GetLastError());
                break;
            }
        }
    }
    av_packet_unref(&packet);
}

    // Release the frame buffer
    av_frame_free(&pFrameRGB);
    av_frame_free(&pFrame);

    // Close the codec
    avcodec_close(pCodecCtx);
    avcodec_free_context(&pCodecCtx);

    // Close the video file
    avformat_close_input(&pFormatCtx);

    // Close the standard input and standard output handles
    CloseHandle(streamProc.stdinWritePipe);
    CloseHandle(streamProc.stdinReadPipe);

    // Stop the stream
    WaitForSingleObject(streamProc.procInfo.hProcess, INFINITE);
    CloseHandle(streamProc.procInfo.hProcess);
    CloseHandle(streamProc.procInfo.hThread);

    return 0;
}
