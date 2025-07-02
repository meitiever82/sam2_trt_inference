// Copyright 2025 Tier IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cuda_utils/cuda_unique_ptr.hpp>
#include <cuda_utils/stream_unique_ptr.hpp>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <tensorrt_common/tensorrt_common.hpp>
#include <unordered_map>
#include <vector>

using cuda_utils::CudaUniquePtr;
using cuda_utils::CudaUniquePtrHost;
using cuda_utils::makeCudaStream;
using cuda_utils::StreamUniquePtr;

class SAM2ImageEncoder
{
   public:
    // Constructor
    SAM2ImageEncoder(const std::string& onnx_path,
                     const std::string& engine_precision = "fp16",
                     const size_t max_workspace_size = (1ULL << 30U),
                     const cv::Size& input_size = cv::Size(1024, 1024),
                     const std::vector<std::string>& plugin_paths = {},
                     const std::string& engine_path = "",
                     const int32_t dla_core_id = -1,
                     const bool profile_per_layer = false);

    ~SAM2ImageEncoder();

    // Encode images
    bool EncodeImage(const std::vector<cv::Mat>& images);

    // Setup with custom profile dimensions and network IO
    bool Setup(tensorrt_common::ProfileDimsPtr profile_dims = nullptr,
               tensorrt_common::NetworkIOPtr network_io = nullptr);

    // Get encoded features
    CudaUniquePtrHost<float[]>& GetFeats0Data()
    {
        return feats_0_data_;
    }
    CudaUniquePtrHost<float[]>& GetFeats1Data()
    {
        return feats_1_data_;
    }
    CudaUniquePtrHost<float[]>& GetEmbedData()
    {
        return embed_data_;
    }

    // Get feature sizes
    size_t GetFeats0Size() const
    {
        return feats_0_size_;
    }
    size_t GetFeats1Size() const
    {
        return feats_1_size_;
    }
    size_t GetEmbedSize() const
    {
        return embed_size_;
    }

    // Get input dimensions
    int GetInputHeight() const
    {
        return input_height_;
    }
    int GetInputWidth() const
    {
        return input_width_;
    }
    int GetBatchSize() const
    {
        return batch_size_;
    }

    // Get precision info
    std::string GetPrecision() const;

    // Print profiling information
    void PrintProfiling() const;

    // Get TensorRT information
    int32_t GetNbIOTensors() const;
    const char* GetIOTensorName(const int32_t index) const;
    nvinfer1::Dims GetTensorShape(const int32_t index) const;
    nvinfer1::Dims GetTensorShape(const char* tensor_name) const;

    // Get TensorRT components for advanced usage
    std::shared_ptr<tensorrt_common::Profiler> GetModelProfiler() const;
    std::shared_ptr<tensorrt_common::Profiler> GetHostProfiler() const;
    std::shared_ptr<tensorrt_common::TrtCommonConfig> GetTrtCommonConfig() const;

   private:
    // Initialize TensorRT common instance
    bool InitializeTensorRT();

    // Setup input/output tensors
    bool SetupTensors();

    // Allocate GPU memory
    bool AllocateGPUMemory();

    // Calculate memory sizes based on tensor shapes
    bool CalculateMemorySizes();

    // Get input details from ONNX model
    bool GetInputDetails();

    // Prepare input tensor
    cv::Mat Preprocess(const std::vector<cv::Mat>& images);

    // Set input shapes for dynamic tensors
    bool SetInputShapes(const int batch_size);

    // Execute inference
    bool Infer(const cv::Mat& input_tensor);

    // Copy results from device to host
    bool CopyResultsToHost();

    // Validate tensor shapes and names
    bool ValidateTensorConfiguration();

    // Create optimization profile for dynamic shapes
    tensorrt_common::ProfileDimsPtr CreateOptimizationProfile();

    // Create network IO configuration
    tensorrt_common::NetworkIOPtr CreateNetworkIO();

    // Configuration and TensorRT
    std::unique_ptr<tensorrt_common::TrtCommon> trt_encoder_;
    tensorrt_common::TrtCommonConfig trt_config_;
    std::shared_ptr<tensorrt_common::Profiler> profiler_;

    // Model configuration
    cv::Size input_size_;
    std::string encoder_precision_;
    int input_height_;
    int input_width_;
    int batch_size_;

    // Tensor names (assuming standard SAM2 encoder naming)
    struct TensorNames
    {
        const char* input = "input";
        const char* feats_0 = "feats_0";
        const char* feats_1 = "feats_1";
        const char* image_embed = "image_embed";
    } tensor_names_;

    // Tensor information storage
    std::unordered_map<std::string, tensorrt_common::NetworkIO> network_io_map_;
    std::unordered_map<std::string, tensorrt_common::ProfileDims> profile_dims_map_;
    std::vector<std::string> plugin_paths_;

    // Feature sizes
    size_t feats_0_size_;
    size_t feats_1_size_;
    size_t embed_size_;
    size_t input_size_bytes_;

    // Memory sizes
    struct MemorySizes
    {
        size_t input = 0;
        size_t feats_0 = 0;
        size_t feats_1 = 0;
        size_t embed = 0;
    } memory_sizes_;

    // CPU data buffers
    CudaUniquePtrHost<float[]> input_host_;
    CudaUniquePtrHost<float[]> feats_0_data_;
    CudaUniquePtrHost<float[]> feats_1_data_;
    CudaUniquePtrHost<float[]> embed_data_;

    // GPU data buffers
    CudaUniquePtr<float[]> input_device_;
    CudaUniquePtr<float[]> feats_0_data_device_;
    CudaUniquePtr<float[]> feats_1_data_device_;
    CudaUniquePtr<float[]> embed_data_device_;

    // CUDA stream
    StreamUniquePtr stream_;

    // Current batch size
    int current_batch_size_;

    // Utility functions
    size_t GetTensorSize(const nvinfer1::Dims& dims) const;
    bool CopyHostToDevice(void* host_ptr, void* device_ptr, size_t size);
    bool CopyDeviceToHost(void* device_ptr, void* host_ptr, size_t size);

    // Tensor shape utilities
    nvinfer1::Dims CreateDims(const std::vector<int32_t>& shape) const;
    std::vector<int32_t> DimsToVector(const nvinfer1::Dims& dims) const;

    // Tensor operations using TrtCommon interface
    bool SetTensorAddresses();
    bool SetTensorShapes();

    // Dynamic shape support
    bool SupportsDynamicShapes() const;
    bool UpdateDynamicShapes(const int batch_size);

    // Image preprocessing utilities
    cv::Mat NormalizeImage(const cv::Mat& image) const;
    cv::Mat ResizeImage(const cv::Mat& image) const;
    void ImageToTensor(const cv::Mat& image, float* tensor_data) const;
};