cmake_minimum_required(VERSION 3.5)

set(PROJECT_NAME Yolov8OnnxRuntimeCPPInference)
project(${PROJECT_NAME} VERSION 0.0.1 LANGUAGES CXX)

# -------------- Support C++17 for using filesystem ------------------#
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(OpenCV_DNN_CUDA ON)

# -------------- OpenCV ------------------#
# Set the path to the OpenCV installation directory
set(OpenCV_DIR "F:/AI_Componets/OpenCV/build/install")

# Find the OpenCV package
find_package(OpenCV REQUIRED)

# Include the OpenCV include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# -------------- Compile CUDA for FP16 inference if needed ------------------#
# Option to enable CUDA support
option(USE_CUDA "Enable CUDA support" ON)

# Set the path to the CUDA installation directory explicitly
set(CUDA_TOOLKIT_ROOT_DIR "F:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6" CACHE PATH "CUDA Toolkit Directory")

if (USE_CUDA)
    find_package(CUDA 12.6 REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    add_definitions(-DUSE_CUDA)
endif ()

# -------------- ONNXRUNTIME ------------------#
set(ONNXRUNTIME_VERSION 1.20.1)
set(ONNXRUNTIME_ROOT "F:/AI_Componets/ONNX/windows_bin/onnxruntime-win-x64-gpu-1.20.1_(MS)")

include_directories(${ONNXRUNTIME_ROOT}/include)

# -------------- FFmpeg for akvirtualcamera ------------------#
# Set the path to the FFmpeg installation directory
set(FFMPEG_DIR "F:/AI_Componets/FFMPEG/ffmpeg-7.0.2-full_build-shared")

# Include FFmpeg headers
include_directories(${FFMPEG_DIR}/include)

# Link FFmpeg libraries
set(FFMPEG_LIBS
    ${FFMPEG_DIR}/lib/avformat.lib
    ${FFMPEG_DIR}/lib/avcodec.lib
    ${FFMPEG_DIR}/lib/avutil.lib
    ${FFMPEG_DIR}/lib/swscale.lib
    ${FFMPEG_DIR}/lib/avdevice.lib
)

# -------------- Project Sources ------------------#
set(PROJECT_SOURCES
    main.cpp
    inference.h
    inference.cpp
)

add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib ${FFMPEG_LIBS})
if (USE_CUDA)
    target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
endif ()

# Ensure the destination directory exists
set(DEST_DIR "${CMAKE_CURRENT_BINARY_DIR}/Release")
file(MAKE_DIRECTORY ${DEST_DIR})

# Copy onnxruntime.dll to the executable folder
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
    ${DEST_DIR}
)

# Copy additional required files
configure_file(coco.yaml ${CMAKE_CURRENT_BINARY_DIR}/coco.yaml COPYONLY)
configure_file(yolov8n.onnx ${CMAKE_CURRENT_BINARY_DIR}/yolov8n.onnx COPYONLY)

# Create an images folder
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/images
)