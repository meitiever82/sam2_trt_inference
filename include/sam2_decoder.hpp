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
#include <numeric>
#include <opencv2/opencv.hpp>
#include <string>
#include <tensorrt_common/tensorrt_common.hpp>
#include <unordered_map>
#include <vector>

using cuda_utils::CudaUniquePtr;
using cuda_utils::CudaUniquePtrHost;
using cuda_utils::makeCudaStream;
using cuda_utils::StreamUniquePtr;

class SAM2ImageDecoder
{
   public:
    // Constructor
    SAM2ImageDecoder(const std::string& onnx_path,
                     const std::string& engine_precision = "fp16",
                     const size_t max_workspace_size = (1ULL << 30U),
                     const cv::Size& encoder_input_size = cv::Size(1024, 1024),
                     const std::vector<int>& encoder_output_sizes = {256, 64, 64},
                     float mask_threshold = 0.0,
                     const std::vector<std::string>& plugin_paths = {},
                     const std::string& engine_path = "",
                     const int32_t dla_core_id = -1,
                     const bool profile_per_layer = false);

    ~SAM2ImageDecoder();

    // Prediction method
    bool Predict(CudaUniquePtrHost<float[]>& image_embed,
                 CudaUniquePtrHost<float[]>& high_res_feats_0,
                 CudaUniquePtrHost<float[]>& high_res_feats_1,
                 const std::vector<std::vector<cv::Point2f>>& point_coords,
                 const std::vector<std::vector<float>>& point_labels,
                 const cv::Size& orig_im_size,
                 const int batch_idx,
                 const int current_batch_size);

    // Setup with custom profile dimensions and network IO
    bool Setup(tensorrt_common::ProfileDimsPtr profile_dims = nullptr,
               tensorrt_common::NetworkIOPtr network_io = nullptr);

    // Get results
    std::vector<cv::Mat> GetResultMasks() const
    {
        return result_masks_;
    }
    std::vector<cv::Mat> GetMatEntropies() const
    {
        return mat_entropies_;
    }
    std::vector<float> GetEntropies() const
    {
        return entropies_;
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

    // Prepare input data for inference
    bool PrepareInputData(const std::vector<std::vector<cv::Point2f>>& point_coords,
                          const std::vector<std::vector<float>>& point_labels,
                          const cv::Size& orig_im_size,
                          CudaUniquePtrHost<float[]>& image_embed,
                          CudaUniquePtrHost<float[]>& high_res_feats_0,
                          CudaUniquePtrHost<float[]>& high_res_feats_1,
                          const int batch_idx);

    // Set input shapes for dynamic tensors
    bool SetInputShapes(const int current_batch_size, const cv::Size& orig_im_size);

    // Run inference
    bool RunInference();

    // Process inference results
    bool PostProcess(const cv::Size& orig_im_size, const int current_batch_size);

    // Calculate entropy from masks
    void CalculateEntropy(const cv::Size& orig_im_size,
                          const int current_batch_size,
                          const std::vector<std::vector<cv::Point2f>>& point_coords);

    // Reset internal state
    void ResetState();

    // Validate tensor shapes and names
    bool ValidateTensorConfiguration();

    // Create optimization profile for dynamic shapes
    tensorrt_common::ProfileDimsPtr CreateOptimizationProfile();

    // Create network IO configuration
    tensorrt_common::NetworkIOPtr CreateNetworkIO();

    // Configuration and TensorRT
    std::unique_ptr<tensorrt_common::TrtCommon> trt_common_;
    tensorrt_common::TrtCommonConfig trt_config_;
    std::shared_ptr<tensorrt_common::Profiler> profiler_;

    // Model configuration
    cv::Size encoder_input_size_;
    std::vector<int> encoder_output_sizes_;
    float mask_threshold_;
    int scale_factor_;

    // Tensor names (assuming standard SAM2 decoder naming)
    struct TensorNames
    {
        const char* image_embed = "image_embed";
        const char* high_res_feats_0 = "high_res_feats_0";
        const char* high_res_feats_1 = "high_res_feats_1";
        const char* point_coords = "point_coords";
        const char* point_labels = "point_labels";
        const char* mask_input = "mask_input";
        const char* has_mask_input = "has_mask_input";
        const char* output_masks = "output_masks";
        const char* output_confidence = "output_confidence";
    } tensor_names_;

    // Tensor information storage
    std::unordered_map<std::string, tensorrt_common::NetworkIO> network_io_map_;
    std::unordered_map<std::string, tensorrt_common::ProfileDims> profile_dims_map_;
    std::vector<std::string> plugin_paths_;

    // Memory sizes
    struct MemorySizes
    {
        size_t image_embed = 0;
        size_t high_res_feats_0 = 0;
        size_t high_res_feats_1 = 0;
        size_t point_coords = 0;
        size_t point_labels = 0;
        size_t mask_input = 0;
        size_t has_mask_input = 0;
        size_t output_masks = 0;
        size_t output_confidence = 0;
    } memory_sizes_;

    // CPU data buffers
    CudaUniquePtrHost<float[]> image_embed_host_;
    CudaUniquePtrHost<float[]> high_res_feats_0_host_;
    CudaUniquePtrHost<float[]> high_res_feats_1_host_;
    CudaUniquePtrHost<float[]> point_coords_host_;
    CudaUniquePtrHost<float[]> point_labels_host_;
    CudaUniquePtrHost<float[]> mask_input_host_;
    CudaUniquePtrHost<float[]> has_mask_input_host_;
    CudaUniquePtrHost<float[]> output_masks_host_;
    CudaUniquePtrHost<float[]> output_confidence_host_;

    // GPU data buffers
    CudaUniquePtr<float[]> image_embed_device_;
    CudaUniquePtr<float[]> high_res_feats_0_device_;
    CudaUniquePtr<float[]> high_res_feats_1_device_;
    CudaUniquePtr<float[]> point_coords_device_;
    CudaUniquePtr<float[]> point_labels_device_;
    CudaUniquePtr<float[]> mask_input_device_;
    CudaUniquePtr<float[]> has_mask_input_device_;
    CudaUniquePtr<float[]> output_masks_device_;
    CudaUniquePtr<float[]> output_confidence_device_;

    // Results
    std::vector<cv::Mat> result_masks_;
    std::vector<cv::Mat> mat_entropies_;
    std::vector<float> entropies_;

    // CUDA stream
    StreamUniquePtr stream_;

    // Current batch size and tensor shapes
    int current_batch_size_;
    std::unordered_map<std::string, nvinfer1::Dims> current_input_shapes_;

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
    bool UpdateDynamicShapes(const int batch_size, const cv::Size& image_size);
};