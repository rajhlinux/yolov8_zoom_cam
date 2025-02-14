/*


How to compile:
"F:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Auxiliary\\Build\\vcvarsall.bat" x64
cl /EHsc /I"F:\AI_Componets\OpenCV\build\install\include" background_subtraction.cpp /link /LIBPATH:"F:\AI_Componets\OpenCV\build\install\x64\vc17\lib" opencv_core500.lib opencv_imgproc500.lib opencv_highgui500.lib opencv_videoio500.lib opencv_imgcodecs500.lib


Video cropping:
CPU:
ffmpeg -i output_reduced.mp4 -vf "crop=in_w-354:in_h-196:275:164" output_cropped.mp4

GPU:
ffmpeg -hwaccel cuvid -i output_reduced.mp4 -vf "hwdownload,format=nv12,crop=in_w-354:in_h-196:275:164,hwupload" -c:v h264_nvenc output_cropped.mp4

Snapshot:
removes 354px from the right, 275 from the left, 196 from the top and 164 from the bottom:
ffmpeg -i output_reduced.mp4 -vf "crop=in_w-354:in_h-196:275:164" -frames:v 1 output_frame_cropped.png

-------------

    // video.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    // video.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);

    // video.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    // video.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    VideoCapture oceanVideo("E:/miracle_plus_2024/output_reduced_first_4_mins.mp4");


-------------

*/

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() 
{
    VideoCapture cap(0);                          // Video source input.
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);

    VideoCapture bg_v("E:/miracle_plus_2024/output_reduced_first_4_mins.mp4");    // Background image replacer media.

    if (!cap.isOpened() || !bg_v.isOpened()) 
    {
        cerr << "Error: Unable to open video source." << endl;
        return -1;
    }

    //----------------------

    Mat ref_img, img, bg;

    cap.read(ref_img);
    int flag = 0;

    while (true) 
    {
        cap.read(img);        // read the capture source video input.

        bg_v.read(bg);        // read the background overlay video input.

        if (bg.empty()) 
        {
            bg_v.set(CAP_PROP_POS_FRAMES, 0);
            bg_v.read(bg);
        }

        Mat resized;
        resize(bg, resized, ref_img.size(), 0, 0, INTER_AREA);
        bg = resized;

        if (flag == 0) 
        {
            ref_img = img.clone();
        }

        // Create a mask
        Mat diff1, diff2, diff, gray, fgmask, fgmask_inv, fgimg, bgimg, dst;
        absdiff(img, ref_img, diff1);
        absdiff(ref_img, img, diff2);
        diff = diff1 + diff2;

        // Apply Gaussian blur to reduce noise
        GaussianBlur(diff, diff, Size(9, 9), 0);

        // Adjust thresholds to reduce noise
        diff.setTo(0, abs(diff) < 33.0);  // Increase threshold <-------------- 30 seems to be the ideal threshold.
        cvtColor(diff, gray, COLOR_BGR2GRAY);
        gray.setTo(0, abs(gray) < 4);    // Increase threshold

        // Apply morphological operations
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(9, 9));
        threshold(gray, fgmask, 1, 255, THRESH_BINARY);
        erode(fgmask, fgmask, kernel);  // Erode to remove small noise
        dilate(fgmask, fgmask, kernel); // Dilate to fill gaps

        // Invert the mask
        bitwise_not(fgmask, fgmask_inv);

        // Use the masks to extract the relevant parts from FG and BG
        bitwise_and(img, img, fgimg, fgmask);
        bitwise_and(bg, bg, bgimg, fgmask_inv);

        // Combine both the BG and the FG images
        add(bgimg, fgimg, dst);

        imshow("Background Removal", dst);


        char key = (char)waitKey(5);
        if (key == 'q') 
        {
            break;
        } 
        else if (key == 'd') 
        {
            flag = 1;
            cout << "Background Captured" << endl;
        } 
        else if (key == 'r') 
        {
            flag = 0;
            cout << "Ready to Capture new Background" << endl;
        }
    }

    destroyAllWindows();
    cap.release();
    bg_v.release();

    return 0;
}

