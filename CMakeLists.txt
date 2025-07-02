cmake_minimum_required(VERSION 3.10)
project(SAM2-TRT)

# Let users specify -DCMAKE_BUILD_TYPE
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Debug or Release)" FORCE)
endif()
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set compiler flags based on build type
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -Wall -Wextra -Wno-write-strings")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -O0")
    message(STATUS "Building in Debug mode")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wno-write-strings -Wall")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
    message(STATUS "Building in Release mode")
endif()

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath=\$ORIGIN")

# Add cmake modules path
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Include header directories
include_directories(/usr/local/cuda/include include)

# Link directories
link_directories(/usr/local/cuda/lib64)

# Dependencies
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV 4.5 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(nlohmann_json REQUIRED)

# 手动设置 TensorRT 路径
set(TensorRT_ROOT "/usr/local/TensorRT" CACHE PATH "TensorRT installation path")

# 添加 TensorRT 的头文件路径
include_directories(${TensorRT_ROOT}/include)

# 添加 TensorRT 的库文件路径
link_directories(${TensorRT_ROOT}/lib)

# 查找 TensorRT 库文件
find_library(TensorRT_nvinfer nvinfer HINTS ${TensorRT_ROOT}/lib)
find_library(TensorRT_nvinfer_plugin nvinfer_plugin HINTS ${TensorRT_ROOT}/lib)
find_library(TensorRT_nvonnxparser nvonnxparser HINTS ${TensorRT_ROOT}/lib)

# 检查是否找到了必要的库
if(NOT TensorRT_nvinfer)
    message(FATAL_ERROR "Could not find nvinfer library in ${TensorRT_ROOT}/lib")
endif()

if(NOT TensorRT_nvonnxparser)
    message(FATAL_ERROR "Could not find nvonnxparser library in ${TensorRT_ROOT}/lib")
endif()
# 可选：打印找到的库路径用于调试
message(STATUS "TensorRT nvinfer: ${TensorRT_nvinfer}")
message(STATUS "TensorRT nvinfer_plugin: ${TensorRT_nvinfer_plugin}")
message(STATUS "TensorRT nvonnxparser: ${TensorRT_nvonnxparser}")

# Define source files
set(SOURCES
  src/main.cpp
  src/sam2_image_inference.cpp
  src/sam2_encoder.cpp
  src/sam2_decoder.cpp
  src/sam2_ctypes.cpp
  src/utils.cpp
  src/colormap.cpp
  src/tensorrt_common/profiler.cpp
  src/tensorrt_common/tensorrt_common.cpp
)

# Create shared library for Python bindings
add_library(trt_sam2_infer SHARED ${SOURCES})
target_link_libraries(trt_sam2_infer PRIVATE 
  ${OpenCV_LIBS}
  CUDA::cudart
  OpenMP::OpenMP_CXX
  ${TensorRT_nvinfer}
  ${TensorRT_nvinfer_plugin}
  ${TensorRT_nvonnxparser}
  boost_system
  boost_filesystem
  dl
  "stdc++fs"
  "stdc++"
  nlohmann_json::nlohmann_json
)
target_include_directories(trt_sam2_infer PRIVATE include)

# Define executable
add_executable(trtsam2 ${SOURCES})

# Link libraries
target_link_libraries(trtsam2 PRIVATE 
  ${OpenCV_LIBS}
  CUDA::cudart
  OpenMP::OpenMP_CXX
  ${TensorRT_nvinfer}
  ${TensorRT_nvinfer_plugin}
  ${TensorRT_nvonnxparser}
  boost_system
  boost_filesystem
  dl
  "stdc++fs"
  "stdc++"
)
target_include_directories(trtsam2 PRIVATE include)

# Install targets
install(TARGETS trt_sam2_infer
  LIBRARY DESTINATION lib
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)

