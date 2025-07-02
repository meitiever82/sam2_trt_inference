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

#include "sam2_decoder.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "omp.h"

SAM2ImageDecoder::SAM2ImageDecoder(const std::string& onnx_path,
                                   const std::string& engine_precision,
                                   const size_t max_workspace_size,
                                   const cv::Size& encoder_input_size,
                                   const std::vector<int>& encoder_output_sizes,
                                   float mask_threshold,
                                   const std::vector<std::string>& plugin_paths,
                                   const std::string& engine_path,
                                   const int32_t dla_core_id,
                                   const bool profile_per_layer)
    : trt_config_(onnx_path,
                  engine_precision,
                  engine_path,
                  max_workspace_size,
                  dla_core_id,
                  profile_per_layer),
      encoder_input_size_(encoder_input_size),
      encoder_output_sizes_(encoder_output_sizes),
      mask_threshold_(mask_threshold),
      plugin_paths_(plugin_paths),
      scale_factor_(4),
      stream_(makeCudaStream()),
      current_batch_size_(0)
{
    // Create profiler
    profiler_ = std::make_shared<tensorrt_common::Profiler>();

    // Initialize TensorRT
    if (!InitializeTensorRT())
    {
        throw std::runtime_error("Failed to initialize TensorRT");
    }

    // Setup tensors and allocate memory
    if (!SetupTensors() || !AllocateGPUMemory())
    {
        throw std::runtime_error("Failed to setup tensors or allocate memory");
    }
}

SAM2ImageDecoder::~SAM2ImageDecoder() = default;

bool SAM2ImageDecoder::InitializeTensorRT()
{
    try
    {
        // Create TrtCommon instance
        trt_common_ =
            std::make_unique<tensorrt_common::TrtCommon>(trt_config_, profiler_, plugin_paths_);

        // Setup the TensorRT engine
        if (!trt_common_->setup())
        {
            std::cerr << "Failed to setup TensorRT engine" << std::endl;
            return false;
        }

        return ValidateTensorConfiguration();
    }
    catch (const std::exception& e)
    {
        std::cerr << "Exception in InitializeTensorRT: " << e.what() << std::endl;
        return false;
    }
}

bool SAM2ImageDecoder::ValidateTensorConfiguration()
{
    // Get number of IO tensors
    const int32_t num_tensors = trt_common_->getNbIOTensors();

    std::cout << "Total IO tensors: " << num_tensors << std::endl;

    // Print tensor information for debugging
    for (int32_t i = 0; i < num_tensors; ++i)
    {
        const char* tensor_name = trt_common_->getIOTensorName(i);
        nvinfer1::Dims dims = trt_common_->getTensorShape(i);

        std::cout << "Tensor " << i << ": " << tensor_name << " Shape: [";
        for (int j = 0; j < dims.nbDims; ++j)
        {
            std::cout << dims.d[j];
            if (j < dims.nbDims - 1)
                std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        // Store tensor information
        std::string name_str(tensor_name);
        network_io_map_.emplace(name_str, tensorrt_common::NetworkIO(name_str, dims));
    }

    return true;
}

bool SAM2ImageDecoder::SetupTensors()
{
    // Calculate memory sizes based on tensor shapes
    return CalculateMemorySizes();
}

bool SAM2ImageDecoder::CalculateMemorySizes()
{
    // Calculate sizes for each tensor based on network IO map
    for (const auto& [name, network_io] : network_io_map_)
    {
        size_t tensor_size = GetTensorSize(network_io.dims);

        if (name == tensor_names_.image_embed)
        {
            memory_sizes_.image_embed = tensor_size;
        }
        else if (name == tensor_names_.high_res_feats_0)
        {
            memory_sizes_.high_res_feats_0 = tensor_size;
        }
        else if (name == tensor_names_.high_res_feats_1)
        {
            memory_sizes_.high_res_feats_1 = tensor_size;
        }
        else if (name == tensor_names_.point_coords)
        {
            memory_sizes_.point_coords = tensor_size;
        }
        else if (name == tensor_names_.point_labels)
        {
            memory_sizes_.point_labels = tensor_size;
        }
        else if (name == tensor_names_.mask_input)
        {
            memory_sizes_.mask_input = tensor_size;
        }
        else if (name == tensor_names_.has_mask_input)
        {
            memory_sizes_.has_mask_input = tensor_size;
        }
        else if (name == tensor_names_.output_masks)
        {
            memory_sizes_.output_masks = tensor_size;
        }
        else if (name == tensor_names_.output_confidence)
        {
            memory_sizes_.output_confidence = tensor_size;
        }
    }

    // Fallback calculation if tensors not found in network IO map
    if (memory_sizes_.image_embed == 0)
    {
        memory_sizes_.image_embed = encoder_output_sizes_[0] * sizeof(float);
    }
    if (memory_sizes_.high_res_feats_0 == 0)
    {
        memory_sizes_.high_res_feats_0 = encoder_output_sizes_[1] * sizeof(float);
    }
    if (memory_sizes_.high_res_feats_1 == 0)
    {
        memory_sizes_.high_res_feats_1 = encoder_output_sizes_[2] * sizeof(float);
    }

    // Calculate dynamic tensor sizes (will be updated based on batch size)
    const int max_batch_size = 32;  // Reasonable default
    memory_sizes_.point_coords = max_batch_size * 2 * 2 * sizeof(float);
    memory_sizes_.point_labels = max_batch_size * 2 * sizeof(float);

    int scaled_height = encoder_input_size_.height / scale_factor_;
    int scaled_width = encoder_input_size_.width / scale_factor_;
    memory_sizes_.mask_input = max_batch_size * 1 * scaled_height * scaled_width * sizeof(float);
    memory_sizes_.has_mask_input = 1 * sizeof(float);
    memory_sizes_.output_masks = max_batch_size * 1 * scaled_height * scaled_width * sizeof(float);
    memory_sizes_.output_confidence = max_batch_size * sizeof(float);

    return true;
}

bool SAM2ImageDecoder::AllocateGPUMemory()
{
    try
    {
        // Allocate host memory
        image_embed_host_ = cuda_utils::make_unique_host<float[]>(
            memory_sizes_.image_embed / sizeof(float), cudaHostAllocPortable);
        high_res_feats_0_host_ = cuda_utils::make_unique_host<float[]>(
            memory_sizes_.high_res_feats_0 / sizeof(float), cudaHostAllocPortable);
        high_res_feats_1_host_ = cuda_utils::make_unique_host<float[]>(
            memory_sizes_.high_res_feats_1 / sizeof(float), cudaHostAllocPortable);
        point_coords_host_ = cuda_utils::make_unique_host<float[]>(
            memory_sizes_.point_coords / sizeof(float), cudaHostAllocPortable);
        point_labels_host_ = cuda_utils::make_unique_host<float[]>(
            memory_sizes_.point_labels / sizeof(float), cudaHostAllocPortable);
        mask_input_host_ = cuda_utils::make_unique_host<float[]>(
            memory_sizes_.mask_input / sizeof(float), cudaHostAllocPortable);
        has_mask_input_host_ = cuda_utils::make_unique_host<float[]>(
            memory_sizes_.has_mask_input / sizeof(float), cudaHostAllocPortable);
        output_masks_host_ = cuda_utils::make_unique_host<float[]>(
            memory_sizes_.output_masks / sizeof(float), cudaHostAllocPortable);
        output_confidence_host_ = cuda_utils::make_unique_host<float[]>(
            memory_sizes_.output_confidence / sizeof(float), cudaHostAllocPortable);

        // Allocate device memory
        image_embed_device_ =
            cuda_utils::make_unique<float[]>(memory_sizes_.image_embed / sizeof(float));
        high_res_feats_0_device_ =
            cuda_utils::make_unique<float[]>(memory_sizes_.high_res_feats_0 / sizeof(float));
        high_res_feats_1_device_ =
            cuda_utils::make_unique<float[]>(memory_sizes_.high_res_feats_1 / sizeof(float));
        point_coords_device_ =
            cuda_utils::make_unique<float[]>(memory_sizes_.point_coords / sizeof(float));
        point_labels_device_ =
            cuda_utils::make_unique<float[]>(memory_sizes_.point_labels / sizeof(float));
        mask_input_device_ =
            cuda_utils::make_unique<float[]>(memory_sizes_.mask_input / sizeof(float));
        has_mask_input_device_ =
            cuda_utils::make_unique<float[]>(memory_sizes_.has_mask_input / sizeof(float));
        output_masks_device_ =
            cuda_utils::make_unique<float[]>(memory_sizes_.output_masks / sizeof(float));
        output_confidence_device_ =
            cuda_utils::make_unique<float[]>(memory_sizes_.output_confidence / sizeof(float));

        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to allocate memory: " << e.what() << std::endl;
        return false;
    }
}

bool SAM2ImageDecoder::Setup(tensorrt_common::ProfileDimsPtr profile_dims,
                             tensorrt_common::NetworkIOPtr network_io)
{
    return trt_common_->setup(std::move(profile_dims), std::move(network_io));
}

bool SAM2ImageDecoder::Predict(CudaUniquePtrHost<float[]>& image_embed,
                               CudaUniquePtrHost<float[]>& high_res_feats_0,
                               CudaUniquePtrHost<float[]>& high_res_feats_1,
                               const std::vector<std::vector<cv::Point2f>>& point_coords,
                               const std::vector<std::vector<float>>& point_labels,
                               const cv::Size& orig_im_size,
                               const int batch_idx,
                               const int current_batch_size)
{
    current_batch_size_ = current_batch_size;

    // Reset previous results
    ResetState();

    // Prepare input data
    if (!PrepareInputData(point_coords,
                          point_labels,
                          orig_im_size,
                          image_embed,
                          high_res_feats_0,
                          high_res_feats_1,
                          batch_idx))
    {
        return false;
    }

    // Set input shapes for dynamic tensors
    if (!SetInputShapes(current_batch_size, orig_im_size))
    {
        return false;
    }

    // Run inference
    if (!RunInference())
    {
        return false;
    }

    // Process results
    if (!PostProcess(orig_im_size, current_batch_size))
    {
        return false;
    }

    // Calculate entropy
    CalculateEntropy(orig_im_size, current_batch_size, point_coords);

    return true;
}

bool SAM2ImageDecoder::PrepareInputData(const std::vector<std::vector<cv::Point2f>>& point_coords,
                                        const std::vector<std::vector<float>>& point_labels,
                                        const cv::Size& orig_im_size,
                                        CudaUniquePtrHost<float[]>& image_embed,
                                        CudaUniquePtrHost<float[]>& high_res_feats_0,
                                        CudaUniquePtrHost<float[]>& high_res_feats_1,
                                        const int batch_idx)
{
    // Copy input embeddings
    size_t image_embed_bytes = memory_sizes_.image_embed;
    size_t high_res_feats_0_bytes = memory_sizes_.high_res_feats_0;
    size_t high_res_feats_1_bytes = memory_sizes_.high_res_feats_1;

    std::memcpy(image_embed_host_.get(),
                image_embed.get() + batch_idx * (image_embed_bytes / sizeof(float)),
                image_embed_bytes);
    std::memcpy(high_res_feats_0_host_.get(),
                high_res_feats_0.get() + batch_idx * (high_res_feats_0_bytes / sizeof(float)),
                high_res_feats_0_bytes);
    std::memcpy(high_res_feats_1_host_.get(),
                high_res_feats_1.get() + batch_idx * (high_res_feats_1_bytes / sizeof(float)),
                high_res_feats_1_bytes);

    // Normalize point coordinates
    int coords_idx = 0;
    for (int i = 0; i < static_cast<int>(point_coords.size()); i++)
    {
        for (int j = 0; j < static_cast<int>(point_coords[i].size()); j++)
        {
            point_coords_host_[coords_idx++] =
                point_coords[i][j].x / orig_im_size.width * encoder_input_size_.width;
            point_coords_host_[coords_idx++] =
                point_coords[i][j].y / orig_im_size.height * encoder_input_size_.height;
        }
    }

    // Copy point labels
    int labels_idx = 0;
    for (int i = 0; i < static_cast<int>(point_labels.size()); i++)
    {
        for (int j = 0; j < static_cast<int>(point_labels[i].size()); j++)
        {
            point_labels_host_[labels_idx++] = point_labels[i][j];
        }
    }

    // Initialize mask input
    size_t mask_input_elements = memory_sizes_.mask_input / sizeof(float);
    for (size_t i = 0; i < mask_input_elements; i++)
    {
        mask_input_host_[i] = 0.0f;
    }

    // Initialize has mask input
    has_mask_input_host_[0] = 0.0f;

    // Copy to device
    if (!CopyHostToDevice(image_embed_host_.get(), image_embed_device_.get(), image_embed_bytes) ||
        !CopyHostToDevice(
            high_res_feats_0_host_.get(), high_res_feats_0_device_.get(), high_res_feats_0_bytes) ||
        !CopyHostToDevice(
            high_res_feats_1_host_.get(), high_res_feats_1_device_.get(), high_res_feats_1_bytes) ||
        !CopyHostToDevice(point_coords_host_.get(),
                          point_coords_device_.get(),
                          current_batch_size_ * 2 * 2 * sizeof(float)) ||
        !CopyHostToDevice(point_labels_host_.get(),
                          point_labels_device_.get(),
                          current_batch_size_ * 2 * sizeof(float)) ||
        !CopyHostToDevice(mask_input_host_.get(),
                          mask_input_device_.get(),
                          current_batch_size_ * 1 * (encoder_input_size_.height / scale_factor_) *
                              (encoder_input_size_.width / scale_factor_) * sizeof(float)) ||
        !CopyHostToDevice(has_mask_input_host_.get(), has_mask_input_device_.get(), sizeof(float)))
    {
        return false;
    }

    return true;
}

bool SAM2ImageDecoder::SetInputShapes(const int current_batch_size, const cv::Size& orig_im_size)
{
    // Set dynamic input dimensions for point coordinates
    nvinfer1::Dims point_coords_dims = CreateDims({current_batch_size, 2, 2});
    if (!trt_common_->setInputShape(tensor_names_.point_coords, point_coords_dims))
    {
        std::cerr << "Failed to set point_coords shape" << std::endl;
        return false;
    }

    // Set dynamic input dimensions for point labels
    nvinfer1::Dims point_labels_dims = CreateDims({current_batch_size, 2});
    if (!trt_common_->setInputShape(tensor_names_.point_labels, point_labels_dims))
    {
        std::cerr << "Failed to set point_labels shape" << std::endl;
        return false;
    }

    // Set dynamic input dimensions for mask input
    int scaled_height = encoder_input_size_.height / scale_factor_;
    int scaled_width = encoder_input_size_.width / scale_factor_;
    nvinfer1::Dims mask_input_dims =
        CreateDims({current_batch_size, 1, scaled_height, scaled_width});
    if (!trt_common_->setInputShape(tensor_names_.mask_input, mask_input_dims))
    {
        std::cerr << "Failed to set mask_input shape" << std::endl;
        return false;
    }

    return true;
}

bool SAM2ImageDecoder::RunInference()
{
    // Set tensor addresses using TrtCommon interface
    if (!trt_common_->setTensorAddress(tensor_names_.image_embed, image_embed_device_.get()) ||
        !trt_common_->setTensorAddress(tensor_names_.high_res_feats_0,
                                       high_res_feats_0_device_.get()) ||
        !trt_common_->setTensorAddress(tensor_names_.high_res_feats_1,
                                       high_res_feats_1_device_.get()) ||
        !trt_common_->setTensorAddress(tensor_names_.point_coords, point_coords_device_.get()) ||
        !trt_common_->setTensorAddress(tensor_names_.point_labels, point_labels_device_.get()) ||
        !trt_common_->setTensorAddress(tensor_names_.mask_input, mask_input_device_.get()) ||
        !trt_common_->setTensorAddress(tensor_names_.has_mask_input,
                                       has_mask_input_device_.get()) ||
        !trt_common_->setTensorAddress(tensor_names_.output_masks, output_masks_device_.get()) ||
        !trt_common_->setTensorAddress(tensor_names_.output_confidence,
                                       output_confidence_device_.get()))
    {
        std::cerr << "Failed to set tensor addresses" << std::endl;
        return false;
    }

    // Execute inference
    if (!trt_common_->enqueueV3(*stream_))
    {
        std::cerr << "Failed to execute inference" << std::endl;
        return false;
    }

    // Synchronize stream
    cudaStreamSynchronize(*stream_);

    return true;
}

bool SAM2ImageDecoder::PostProcess(const cv::Size& orig_im_size, const int current_batch_size)
{
    // Copy results from device to host
    int scaled_height = encoder_input_size_.height / scale_factor_;
    int scaled_width = encoder_input_size_.width / scale_factor_;
    size_t mask_output_bytes =
        current_batch_size * 1 * scaled_height * scaled_width * sizeof(float);
    size_t confidence_output_bytes = current_batch_size * sizeof(float);

    if (!CopyDeviceToHost(
            output_masks_device_.get(), output_masks_host_.get(), mask_output_bytes) ||
        !CopyDeviceToHost(output_confidence_device_.get(),
                          output_confidence_host_.get(),
                          confidence_output_bytes))
    {
        return false;
    }

    // Process masks and convert to OpenCV format
    const float* mask_data = output_masks_host_.get();
    const int64_t h = scaled_height, w = scaled_width;

    result_masks_.clear();
    result_masks_.resize(current_batch_size);

#pragma omp parallel for
    for (int i = 0; i < current_batch_size; i++)
    {
        // Create Mat directly from mask_data to avoid unnecessary data copy
        cv::Mat mask_i(
            h, w, CV_32FC1, const_cast<void*>(static_cast<const void*>(mask_data + i * h * w)));

        // Perform resize and threshold operations
        cv::Mat resized_mask;
        cv::resize(mask_i, resized_mask, orig_im_size, 0, 0, cv::INTER_LINEAR);

        // Convert to 8-bit and binarize
        cv::Mat binary_mask;
        resized_mask = resized_mask > mask_threshold_;
        resized_mask.convertTo(binary_mask, CV_8U, 255);

        result_masks_[i] = binary_mask;
    }

    return true;
}

void SAM2ImageDecoder::CalculateEntropy(const cv::Size& orig_im_size,
                                        const int current_batch_size,
                                        const std::vector<std::vector<cv::Point2f>>& point_coords)
{
    const float* mask_data = output_masks_host_.get();
    int scaled_height = encoder_input_size_.height / scale_factor_;
    int scaled_width = encoder_input_size_.width / scale_factor_;
    const int64_t h = scaled_height, w = scaled_width;

    int batch_size = static_cast<int>(point_coords.size());
    entropies_.resize(batch_size);
    mat_entropies_.resize(batch_size);

#pragma omp parallel for schedule(static, 4) num_threads(4)
    for (int i = 0; i < batch_size; i++)
    {
        cv::Mat mask(h, w, CV_32FC1, (void*)(mask_data + i * h * w));
        cv::Mat ent = cv::Mat::zeros(h, w, CV_8UC1);

        float sum_ent = 0.0;
        int y0 = (h / static_cast<float>(orig_im_size.height)) * (point_coords[i][0]).y;
        int y1 = (h / static_cast<float>(orig_im_size.height)) * (point_coords[i][1]).y;
        int x0 = (w / static_cast<float>(orig_im_size.width)) * (point_coords[i][0]).x;
        int x1 = (w / static_cast<float>(orig_im_size.width)) * (point_coords[i][1]).x;
        int width = x1 - x0;
        int height = y1 - y0;

        for (int y = y0; y < y1; y++)
        {
            for (int x = x0; x < x1; x++)
            {
                float p = mask.at<float>(y, x);
                p = 1.f / (1.f + expf(-p));
                float value = -p * log(p + 1e-10);
                ent.at<unsigned char>(y, x) += static_cast<unsigned char>(255 * value);
                sum_ent += value;
            }
        }

        if ((width * height) < 1)
        {
            sum_ent = 0.0;
        }
        else
        {
            sum_ent /= (width * height);
        }

        entropies_[i] = sum_ent;
        mat_entropies_[i] = ent;
    }
}

void SAM2ImageDecoder::ResetState()
{
    result_masks_.clear();
    mat_entropies_.clear();
    entropies_.clear();
}

// Public interface methods
std::string SAM2ImageDecoder::GetPrecision() const
{
    return trt_common_ ? trt_common_->getPrecision() : "Unknown";
}

void SAM2ImageDecoder::PrintProfiling() const
{
    if (trt_common_)
    {
        trt_common_->printProfiling();
    }
}

int32_t SAM2ImageDecoder::GetNbIOTensors() const
{
    return trt_common_ ? trt_common_->getNbIOTensors() : 0;
}

const char* SAM2ImageDecoder::GetIOTensorName(const int32_t index) const
{
    return trt_common_ ? trt_common_->getIOTensorName(index) : nullptr;
}

nvinfer1::Dims SAM2ImageDecoder::GetTensorShape(const int32_t index) const
{
    return trt_common_ ? trt_common_->getTensorShape(index) : nvinfer1::Dims {};
}

nvinfer1::Dims SAM2ImageDecoder::GetTensorShape(const char* tensor_name) const
{
    return trt_common_ ? trt_common_->getTensorShape(tensor_name) : nvinfer1::Dims {};
}

std::shared_ptr<tensorrt_common::Profiler> SAM2ImageDecoder::GetModelProfiler() const
{
    return trt_common_ ? trt_common_->getModelProfiler() : nullptr;
}

std::shared_ptr<tensorrt_common::Profiler> SAM2ImageDecoder::GetHostProfiler() const
{
    return trt_common_ ? trt_common_->getHostProfiler() : nullptr;
}

std::shared_ptr<tensorrt_common::TrtCommonConfig> SAM2ImageDecoder::GetTrtCommonConfig() const
{
    return trt_common_ ? trt_common_->getTrtCommonConfig() : nullptr;
}

// Utility functions
size_t SAM2ImageDecoder::GetTensorSize(const nvinfer1::Dims& dims) const
{
    size_t size = sizeof(float);
    for (int i = 0; i < dims.nbDims; ++i)
    {
        size *= dims.d[i];
    }
    return size;
}

bool SAM2ImageDecoder::CopyHostToDevice(void* host_ptr, void* device_ptr, size_t size)
{
    cudaError_t err = cudaMemcpyAsync(device_ptr, host_ptr, size, cudaMemcpyHostToDevice, *stream_);
    return err == cudaSuccess;
}

bool SAM2ImageDecoder::CopyDeviceToHost(void* device_ptr, void* host_ptr, size_t size)
{
    cudaError_t err = cudaMemcpyAsync(host_ptr, device_ptr, size, cudaMemcpyDeviceToHost, *stream_);
    return err == cudaSuccess;
}

nvinfer1::Dims SAM2ImageDecoder::CreateDims(const std::vector<int32_t>& shape) const
{
    nvinfer1::Dims dims;
    dims.nbDims = static_cast<int>(shape.size());
    for (size_t i = 0; i < shape.size(); ++i)
    {
        dims.d[i] = shape[i];
    }
    return dims;
}

std::vector<int32_t> SAM2ImageDecoder::DimsToVector(const nvinfer1::Dims& dims) const
{
    std::vector<int32_t> shape;
    for (int i = 0; i < dims.nbDims; ++i)
    {
        shape.push_back(dims.d[i]);
    }
    return shape;
}

bool SAM2ImageDecoder::SetTensorAddresses()
{
    return trt_common_->setTensorAddress(tensor_names_.image_embed, image_embed_device_.get()) &&
           trt_common_->setTensorAddress(tensor_names_.high_res_feats_0,
                                         high_res_feats_0_device_.get()) &&
           trt_common_->setTensorAddress(tensor_names_.high_res_feats_1,
                                         high_res_feats_1_device_.get()) &&
           trt_common_->setTensorAddress(tensor_names_.point_coords, point_coords_device_.get()) &&
           trt_common_->setTensorAddress(tensor_names_.point_labels, point_labels_device_.get()) &&
           trt_common_->setTensorAddress(tensor_names_.mask_input, mask_input_device_.get()) &&
           trt_common_->setTensorAddress(tensor_names_.has_mask_input,
                                         has_mask_input_device_.get()) &&
           trt_common_->setTensorAddress(tensor_names_.output_masks, output_masks_device_.get()) &&
           trt_common_->setTensorAddress(tensor_names_.output_confidence,
                                         output_confidence_device_.get());
}

bool SAM2ImageDecoder::SetTensorShapes()
{
    return SetInputShapes(current_batch_size_,
                          cv::Size(1024, 1024));  // Default size, should be updated
}

bool SAM2ImageDecoder::SupportsDynamicShapes() const
{
    // Check if any tensor has dynamic dimensions (-1)
    for (const auto& [name, network_io] : network_io_map_)
    {
        for (int i = 0; i < network_io.dims.nbDims; ++i)
        {
            if (network_io.dims.d[i] == -1)
            {
                return true;
            }
        }
    }
    return false;
}

bool SAM2ImageDecoder::UpdateDynamicShapes(const int batch_size, const cv::Size& image_size)
{
    current_batch_size_ = batch_size;
    return SetInputShapes(batch_size, image_size);
}

tensorrt_common::ProfileDimsPtr SAM2ImageDecoder::CreateOptimizationProfile()
{
    auto profile_dims = std::make_unique<std::vector<tensorrt_common::ProfileDims>>();

    // Add optimization profiles for dynamic tensors
    // Point coordinates: min [1,2,2], opt [4,2,2], max [32,2,2]
    nvinfer1::Dims min_coords = CreateDims({1, 2, 2});
    nvinfer1::Dims opt_coords = CreateDims({4, 2, 2});
    nvinfer1::Dims max_coords = CreateDims({32, 2, 2});
    profile_dims->emplace_back(tensor_names_.point_coords, min_coords, opt_coords, max_coords);

    // Point labels: min [1,2], opt [4,2], max [32,2]
    nvinfer1::Dims min_labels = CreateDims({1, 2});
    nvinfer1::Dims opt_labels = CreateDims({4, 2});
    nvinfer1::Dims max_labels = CreateDims({32, 2});
    profile_dims->emplace_back(tensor_names_.point_labels, min_labels, opt_labels, max_labels);

    // Mask input: min [1,1,H,W], opt [4,1,H,W], max [32,1,H,W]
    int scaled_h = encoder_input_size_.height / scale_factor_;
    int scaled_w = encoder_input_size_.width / scale_factor_;
    nvinfer1::Dims min_mask = CreateDims({1, 1, scaled_h, scaled_w});
    nvinfer1::Dims opt_mask = CreateDims({4, 1, scaled_h, scaled_w});
    nvinfer1::Dims max_mask = CreateDims({32, 1, scaled_h, scaled_w});
    profile_dims->emplace_back(tensor_names_.mask_input, min_mask, opt_mask, max_mask);

    return profile_dims;
}

tensorrt_common::NetworkIOPtr SAM2ImageDecoder::CreateNetworkIO()
{
    auto network_io = std::make_unique<std::vector<tensorrt_common::NetworkIO>>();

    // Add network IO specifications
    for (const auto& [name, io] : network_io_map_)
    {
        network_io->push_back(io);
    }

    return network_io;
}