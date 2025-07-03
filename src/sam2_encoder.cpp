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

#include "sam2_encoder.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

SAM2ImageEncoder::SAM2ImageEncoder(const std::string& onnx_path,
                                   const std::string& engine_precision,
                                   const size_t max_workspace_size,
                                   const cv::Size& input_size,
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
      input_size_(input_size),
      encoder_precision_(engine_precision),
      input_height_(input_size.height),
      input_width_(input_size.width),
      plugin_paths_(plugin_paths),
      batch_size_(1),
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

SAM2ImageEncoder::~SAM2ImageEncoder() = default;

bool SAM2ImageEncoder::InitializeTensorRT()
{
    try
    {
        // Create TrtCommon instance first (without setup)
        trt_encoder_ =
            std::make_unique<tensorrt_common::TrtCommon>(trt_config_, profiler_, plugin_paths_);

        // Create optimization profile for dynamic shapes
        // Use default tensor name for now
        auto profile_dims = std::make_unique<std::vector<tensorrt_common::ProfileDims>>();
        nvinfer1::Dims min_input = CreateDims({1, 3, input_height_, input_width_});
        nvinfer1::Dims opt_input = CreateDims({4, 3, input_height_, input_width_});
        nvinfer1::Dims max_input = CreateDims({32, 3, input_height_, input_width_});

        // Use common tensor names for SAM2 encoder input
        const char* input_names[] = {"input", "images", "image", "x"};

        // Try the most common name first
        profile_dims->emplace_back(input_names[0], min_input, opt_input, max_input);

        std::cout << "Creating optimization profile for input tensor: " << input_names[0]
                  << std::endl;
        std::cout << "Min: [" << min_input.d[0] << "," << min_input.d[1] << "," << min_input.d[2]
                  << "," << min_input.d[3] << "]" << std::endl;
        std::cout << "Opt: [" << opt_input.d[0] << "," << opt_input.d[1] << "," << opt_input.d[2]
                  << "," << opt_input.d[3] << "]" << std::endl;
        std::cout << "Max: [" << max_input.d[0] << "," << max_input.d[1] << "," << max_input.d[2]
                  << "," << max_input.d[3] << "]" << std::endl;

        // Setup the TensorRT engine with optimization profile
        if (!trt_encoder_->setup(std::move(profile_dims), nullptr))
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

bool SAM2ImageEncoder::ValidateTensorConfiguration()
{
    // Get number of IO tensors
    const int32_t num_tensors = trt_encoder_->getNbIOTensors();

    std::cout << "Total IO tensors: " << num_tensors << std::endl;

    // Print tensor information for debugging
    for (int32_t i = 0; i < num_tensors; ++i)
    {
        const char* tensor_name = trt_encoder_->getIOTensorName(i);
        nvinfer1::Dims dims = trt_encoder_->getTensorShape(i);

        std::cout << "Tensor " << i << ": " << tensor_name << " Shape: [";
        for (int j = 0; j < dims.nbDims; ++j)
        {
            std::cout << dims.d[j];
            if (j < dims.nbDims - 1)
                std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        // Check if this is the input tensor and has dynamic shapes
        if (i == 0)
        {  // Assuming first tensor is input
            std::cout << "Using tensor '" << tensor_name << "' as input tensor" << std::endl;
            // Don't update tensor_names_.input here since it's const char*
        }

        // Store tensor information
        std::string name_str(tensor_name);
        network_io_map_.emplace(name_str, tensorrt_common::NetworkIO(name_str, dims));
    }

    return GetInputDetails();
}

bool SAM2ImageEncoder::GetInputDetails()
{
    // Try to get input tensor dimensions (use index 0 for first tensor)
    nvinfer1::Dims input_dims = trt_encoder_->getTensorShape(0);

    if (input_dims.nbDims >= 4)
    {
        batch_size_ = input_dims.d[0];
        input_height_ = input_dims.d[2];
        input_width_ = input_dims.d[3];

        // Update input_size_ based on actual tensor dimensions
        input_size_ = cv::Size(input_width_, input_height_);

        std::cout << "Input details - Batch: " << batch_size_ << ", Height: " << input_height_
                  << ", Width: " << input_width_ << std::endl;
        return true;
    }
    else
    {
        std::cerr << "Invalid input tensor dimensions" << std::endl;
        return false;
    }
}

bool SAM2ImageEncoder::SetupTensors()
{
    // Calculate memory sizes based on tensor shapes
    return CalculateMemorySizes();
}

bool SAM2ImageEncoder::CalculateMemorySizes()
{
    // Calculate sizes for each tensor based on network IO map
    for (const auto& [name, network_io] : network_io_map_)
    {
        size_t tensor_size = GetTensorSize(network_io.dims);

        if (name == tensor_names_.input)
        {
            memory_sizes_.input = tensor_size;
            input_size_bytes_ = tensor_size;
        }
        else if (name == tensor_names_.feats_0)
        {
            memory_sizes_.feats_0 = tensor_size;
            feats_0_size_ = tensor_size / sizeof(float);
        }
        else if (name == tensor_names_.feats_1)
        {
            memory_sizes_.feats_1 = tensor_size;
            feats_1_size_ = tensor_size / sizeof(float);
        }
        else if (name == tensor_names_.image_embed)
        {
            memory_sizes_.embed = tensor_size;
            embed_size_ = tensor_size / sizeof(float);
        }
    }

    // Fallback calculation if tensors not found in network IO map
    if (memory_sizes_.input == 0)
    {
        memory_sizes_.input = batch_size_ * 3 * input_height_ * input_width_ * sizeof(float);
        input_size_bytes_ = memory_sizes_.input;
    }

    return true;
}

bool SAM2ImageEncoder::AllocateGPUMemory()
{
    try
    {
        // Allocate host memory
        input_host_ = cuda_utils::make_unique_host<float[]>(memory_sizes_.input / sizeof(float),
                                                            cudaHostAllocPortable);
        feats_0_data_ = cuda_utils::make_unique_host<float[]>(feats_0_size_, cudaHostAllocPortable);
        feats_1_data_ = cuda_utils::make_unique_host<float[]>(feats_1_size_, cudaHostAllocPortable);
        embed_data_ = cuda_utils::make_unique_host<float[]>(embed_size_, cudaHostAllocPortable);

        // Allocate device memory
        input_device_ = cuda_utils::make_unique<float[]>(memory_sizes_.input / sizeof(float));
        feats_0_data_device_ = cuda_utils::make_unique<float[]>(feats_0_size_);
        feats_1_data_device_ = cuda_utils::make_unique<float[]>(feats_1_size_);
        embed_data_device_ = cuda_utils::make_unique<float[]>(embed_size_);

        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to allocate memory: " << e.what() << std::endl;
        return false;
    }
}

bool SAM2ImageEncoder::Setup(tensorrt_common::ProfileDimsPtr profile_dims,
                             tensorrt_common::NetworkIOPtr network_io)
{
    return trt_encoder_->setup(std::move(profile_dims), std::move(network_io));
}

bool SAM2ImageEncoder::EncodeImage(const std::vector<cv::Mat>& images)
{
    if (images.empty())
    {
        std::cerr << "No images provided for encoding" << std::endl;
        return false;
    }

    current_batch_size_ = static_cast<int>(images.size());

    // Set input shapes for dynamic tensors
    if (!SetInputShapes(current_batch_size_))
    {
        return false;
    }

    cv::Mat input_tensor = Preprocess(images);
    if (input_tensor.empty())
    {
        std::cerr << "Failed to preprocess images" << std::endl;
        return false;
    }

    bool success = Infer(input_tensor);
    if (!success)
    {
        std::cerr << "Failed to encode image" << std::endl;
        return false;
    }

    return CopyResultsToHost();
}

cv::Mat SAM2ImageEncoder::Preprocess(const std::vector<cv::Mat>& images)
{
    // RGB mean values
    cv::Scalar mean(123.675, 116.28, 103.53);
    // RGB standard deviation values
    std::vector<float> std_vals {0.229f, 0.224f, 0.225f};

    int num_images = static_cast<int>(images.size());
    assert(num_images <= batch_size_);

    // Resize images to input size if needed
    std::vector<cv::Mat> resized_images;
    for (const auto& image : images)
    {
        cv::Mat resized = ResizeImage(image);
        resized_images.push_back(resized);
    }

    // Normalize images: subtract mean, scale to 0~1, convert to NCHW format
    cv::Mat normalized_images = cv::dnn::blobFromImages(resized_images,
                                                        1.0 / 255.0,
                                                        cv::Size(input_width_, input_height_),
                                                        mean,
                                                        true,
                                                        false,
                                                        CV_32F);

    // Normalize by standard deviation
    auto ptr = normalized_images.ptr<float>();
    for (int n = 0; n < num_images; ++n)
    {
        auto bias_batch = n * 3 * input_height_ * input_width_;
        for (int i = 0; i < 3; i++)
        {
            auto bias_channel = i * input_height_ * input_width_;
            for (int j = 0; j < input_height_ * input_width_; ++j)
            {
                ptr[bias_batch + bias_channel + j] /= std_vals[i];
            }
        }
    }

    return normalized_images;
}

cv::Mat SAM2ImageEncoder::ResizeImage(const cv::Mat& image) const
{
    cv::Mat resized;
    if (image.size() != input_size_)
    {
        cv::resize(image, resized, input_size_, 0, 0, cv::INTER_LINEAR);
    }
    else
    {
        resized = image.clone();
    }
    return resized;
}

cv::Mat SAM2ImageEncoder::NormalizeImage(const cv::Mat& image) const
{
    cv::Mat normalized;
    image.convertTo(normalized, CV_32F, 1.0 / 255.0);
    return normalized;
}

void SAM2ImageEncoder::ImageToTensor(const cv::Mat& image, float* tensor_data) const
{
    // Convert HWC to CHW format
    std::vector<cv::Mat> channels;
    cv::split(image, channels);

    int channel_size = input_height_ * input_width_;
    for (int c = 0; c < 3; ++c)
    {
        std::memcpy(
            tensor_data + c * channel_size, channels[c].ptr<float>(), channel_size * sizeof(float));
    }
}

bool SAM2ImageEncoder::SetInputShapes(const int batch_size)
{
    // Set dynamic input dimensions if needed
    nvinfer1::Dims input_dims = CreateDims({batch_size, 3, input_height_, input_width_});

    if (SupportsDynamicShapes())
    {
        if (!trt_encoder_->setInputShape(tensor_names_.input, input_dims))
        {
            std::cerr << "Failed to set input shape" << std::endl;
            return false;
        }
    }

    return true;
}

bool SAM2ImageEncoder::Infer(const cv::Mat& input_tensor)
{
    // Ensure contiguous memory for input tensor
    cv::Mat input_tensor_continuous = input_tensor.isContinuous()
                                          ? input_tensor.reshape(1, input_tensor.total())
                                          : input_tensor.reshape(1, input_tensor.total()).clone();

    // Copy input to device
    if (!CopyHostToDevice(input_tensor_continuous.ptr<float>(),
                          input_device_.get(),
                          input_tensor_continuous.total() * sizeof(float)))
    {
        std::cerr << "Failed to copy input to device" << std::endl;
        return false;
    }

    // Set tensor addresses using TrtCommon interface
    if (!trt_encoder_->setTensorAddress(tensor_names_.input, input_device_.get()) ||
        !trt_encoder_->setTensorAddress(tensor_names_.image_embed, embed_data_device_.get()) ||
        !trt_encoder_->setTensorAddress(tensor_names_.feats_1, feats_1_data_device_.get()) ||
        !trt_encoder_->setTensorAddress(tensor_names_.feats_0, feats_0_data_device_.get()))
    {
        std::cerr << "Failed to set tensor addresses" << std::endl;
        return false;
    }

    // Execute inference
    if (!trt_encoder_->enqueueV3(*stream_))
    {
        std::cerr << "Failed to execute inference" << std::endl;
        return false;
    }

    // Synchronize CUDA stream
    cudaStreamSynchronize(*stream_);

    return true;
}

bool SAM2ImageEncoder::CopyResultsToHost()
{
    // Copy output to CPU
    if (!CopyDeviceToHost(
            feats_0_data_device_.get(), feats_0_data_.get(), feats_0_size_ * sizeof(float)) ||
        !CopyDeviceToHost(
            feats_1_data_device_.get(), feats_1_data_.get(), feats_1_size_ * sizeof(float)) ||
        !CopyDeviceToHost(embed_data_device_.get(), embed_data_.get(), embed_size_ * sizeof(float)))
    {
        std::cerr << "Failed to copy results to host" << std::endl;
        return false;
    }

    return true;
}

// Public interface methods
std::string SAM2ImageEncoder::GetPrecision() const
{
    return trt_encoder_ ? trt_encoder_->getPrecision() : "Unknown";
}

void SAM2ImageEncoder::PrintProfiling() const
{
    if (trt_encoder_)
    {
        trt_encoder_->printProfiling();
    }
}

int32_t SAM2ImageEncoder::GetNbIOTensors() const
{
    return trt_encoder_ ? trt_encoder_->getNbIOTensors() : 0;
}

const char* SAM2ImageEncoder::GetIOTensorName(const int32_t index) const
{
    return trt_encoder_ ? trt_encoder_->getIOTensorName(index) : nullptr;
}

nvinfer1::Dims SAM2ImageEncoder::GetTensorShape(const int32_t index) const
{
    return trt_encoder_ ? trt_encoder_->getTensorShape(index) : nvinfer1::Dims {};
}

nvinfer1::Dims SAM2ImageEncoder::GetTensorShape(const char* tensor_name) const
{
    return trt_encoder_ ? trt_encoder_->getTensorShape(tensor_name) : nvinfer1::Dims {};
}

std::shared_ptr<tensorrt_common::Profiler> SAM2ImageEncoder::GetModelProfiler() const
{
    return trt_encoder_ ? trt_encoder_->getModelProfiler() : nullptr;
}

std::shared_ptr<tensorrt_common::Profiler> SAM2ImageEncoder::GetHostProfiler() const
{
    return trt_encoder_ ? trt_encoder_->getHostProfiler() : nullptr;
}

std::shared_ptr<tensorrt_common::TrtCommonConfig> SAM2ImageEncoder::GetTrtCommonConfig() const
{
    return trt_encoder_ ? trt_encoder_->getTrtCommonConfig() : nullptr;
}

// Utility functions
size_t SAM2ImageEncoder::GetTensorSize(const nvinfer1::Dims& dims) const
{
    size_t size = sizeof(float);
    for (int i = 0; i < dims.nbDims; ++i)
    {
        size *= dims.d[i];
    }
    return size;
}

bool SAM2ImageEncoder::CopyHostToDevice(void* host_ptr, void* device_ptr, size_t size)
{
    cudaError_t err = cudaMemcpyAsync(device_ptr, host_ptr, size, cudaMemcpyHostToDevice, *stream_);
    return err == cudaSuccess;
}

bool SAM2ImageEncoder::CopyDeviceToHost(void* device_ptr, void* host_ptr, size_t size)
{
    cudaError_t err = cudaMemcpyAsync(host_ptr, device_ptr, size, cudaMemcpyDeviceToHost, *stream_);
    return err == cudaSuccess;
}

nvinfer1::Dims SAM2ImageEncoder::CreateDims(const std::vector<int32_t>& shape) const
{
    nvinfer1::Dims dims;
    dims.nbDims = static_cast<int>(shape.size());
    for (size_t i = 0; i < shape.size(); ++i)
    {
        dims.d[i] = shape[i];
    }
    return dims;
}

std::vector<int32_t> SAM2ImageEncoder::DimsToVector(const nvinfer1::Dims& dims) const
{
    std::vector<int32_t> shape;
    for (int i = 0; i < dims.nbDims; ++i)
    {
        shape.push_back(dims.d[i]);
    }
    return shape;
}

bool SAM2ImageEncoder::SetTensorAddresses()
{
    return trt_encoder_->setTensorAddress(tensor_names_.input, input_device_.get()) &&
           trt_encoder_->setTensorAddress(tensor_names_.image_embed, embed_data_device_.get()) &&
           trt_encoder_->setTensorAddress(tensor_names_.feats_1, feats_1_data_device_.get()) &&
           trt_encoder_->setTensorAddress(tensor_names_.feats_0, feats_0_data_device_.get());
}

bool SAM2ImageEncoder::SetTensorShapes()
{
    return SetInputShapes(current_batch_size_);
}

bool SAM2ImageEncoder::SupportsDynamicShapes() const
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

bool SAM2ImageEncoder::UpdateDynamicShapes(const int batch_size)
{
    current_batch_size_ = batch_size;
    return SetInputShapes(batch_size);
}

tensorrt_common::ProfileDimsPtr SAM2ImageEncoder::CreateOptimizationProfile()
{
    auto profile_dims = std::make_unique<std::vector<tensorrt_common::ProfileDims>>();

    // Get the actual input tensor name from the model
    const char* actual_input_name = trt_encoder_->getIOTensorName(0);

    // Add optimization profiles for dynamic tensors
    // Input: min [1,3,H,W], opt [4,3,H,W], max [32,3,H,W]
    nvinfer1::Dims min_input = CreateDims({1, 3, input_height_, input_width_});
    nvinfer1::Dims opt_input = CreateDims({4, 3, input_height_, input_width_});
    nvinfer1::Dims max_input = CreateDims({32, 3, input_height_, input_width_});
    profile_dims->emplace_back(actual_input_name, min_input, opt_input, max_input);

    // Print debug information
    std::cout << "Creating optimization profile for input tensor: " << actual_input_name
              << std::endl;
    std::cout << "Min: [" << min_input.d[0] << "," << min_input.d[1] << "," << min_input.d[2] << ","
              << min_input.d[3] << "]" << std::endl;
    std::cout << "Opt: [" << opt_input.d[0] << "," << opt_input.d[1] << "," << opt_input.d[2] << ","
              << opt_input.d[3] << "]" << std::endl;
    std::cout << "Max: [" << max_input.d[0] << "," << max_input.d[1] << "," << max_input.d[2] << ","
              << max_input.d[3] << "]" << std::endl;

    return profile_dims;
}

tensorrt_common::NetworkIOPtr SAM2ImageEncoder::CreateNetworkIO()
{
    auto network_io = std::make_unique<std::vector<tensorrt_common::NetworkIO>>();

    // Add network IO specifications
    for (const auto& [name, io] : network_io_map_)
    {
        network_io->push_back(io);
    }

    return network_io;
}