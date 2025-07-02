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
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <tensorrt_common/tensorrt_common.hpp>
#include <vector>

#include "sam2_decoder.hpp"
#include "sam2_encoder.hpp"

class SAM2Image
{
   public:
    // Constructor
    SAM2Image(const std::string& encoder_path,
              const std::string& decoder_path,
              const cv::Size& encoder_input_size = cv::Size(1024, 1024),
              const std::string& model_precision = "fp16",
              const int decoder_batch_limit = 32,
              const size_t max_workspace_size = (1ULL << 30U),
              const std::vector<std::string>& plugin_paths = {},
              const int32_t dla_core_id = -1,
              const bool profile_per_layer = false);

    ~SAM2Image();

    // Set input images and run encoder
    bool RunEncoder(const std::vector<cv::Mat>& images);

    // Set bounding boxes and generate masks
    bool RunDecoder(const std::vector<std::vector<cv::Rect>>& boxes);

    // Alternative decoder input with point coordinates
    bool RunDecoder(const std::vector<std::vector<cv::Point2f>>& point_coords,
                    const std::vector<std::vector<float>>& point_labels);

    // Decode masks for specific image
    bool DecodeMask(const cv::Size& orig_im_size,
                    const int img_batch_idx,
                    std::vector<cv::Mat>& masks_per_image,
                    const int current_batch_size);

    // Get all generated masks
    const std::vector<std::vector<cv::Mat>>& GetMasks() const;

    // Get maximum entropy map and score
    cv::Mat GetMaxEntropy(float& peak_entropy_score) const;

    // Get entropy scores
    const std::vector<float>& GetEntropies() const;

    // Get entropy matrices
    const std::vector<cv::Mat>& GetMatEntropies() const;

    // Get encoder output features
    const cuda_utils::CudaUniquePtrHost<float[]>& GetImageEmbed() const;
    const cuda_utils::CudaUniquePtrHost<float[]>& GetHighResFeats0() const;
    const cuda_utils::CudaUniquePtrHost<float[]>& GetHighResFeats1() const;

    // Get configuration information
    cv::Size GetEncoderInputSize() const;
    std::string GetModelPrecision() const;
    int GetDecoderBatchLimit() const;

    // Setup methods for advanced configuration
    bool SetupEncoder(tensorrt_common::ProfileDimsPtr profile_dims = nullptr,
                      tensorrt_common::NetworkIOPtr network_io = nullptr);
    bool SetupDecoder(tensorrt_common::ProfileDimsPtr profile_dims = nullptr,
                      tensorrt_common::NetworkIOPtr network_io = nullptr);

    // Print profiling information
    void PrintEncoderProfiling() const;
    void PrintDecoderProfiling() const;

    // Get precision info
    std::string GetEncoderPrecision() const;
    std::string GetDecoderPrecision() const;

   private:
    // Initialize encoder and decoder
    bool InitializeEncoder();
    bool InitializeDecoder();

    // Clear box coordinates and labels
    void ClearBoxes();

    // Convert bounding boxes to point coordinates and labels
    void BoxesToPointCoords(const std::vector<std::vector<cv::Rect>>& boxes,
                            std::vector<std::vector<cv::Point2f>>& point_coords,
                            std::vector<std::vector<float>>& point_labels) const;

    // Validate input parameters
    bool ValidateInputs(const std::vector<cv::Mat>& images) const;
    bool ValidateBoxes(const std::vector<std::vector<cv::Rect>>& boxes) const;

    // Copy encoder features to local storage
    bool CopyEncoderFeatures();

    // Process decoder results
    bool ProcessDecoderResults();

    // Encoder object
    std::unique_ptr<SAM2ImageEncoder> encoder_;

    // Decoder object
    std::unique_ptr<SAM2ImageDecoder> decoder_;

    // Configuration
    std::string encoder_path_;
    std::string decoder_path_;
    cv::Size encoder_input_size_;
    std::string model_precision_;
    int decoder_batch_limit_;
    size_t max_workspace_size_;
    std::vector<std::string> plugin_paths_;
    int32_t dla_core_id_;
    bool profile_per_layer_;

    // Encoder intermediate features (local copies)
    cuda_utils::CudaUniquePtrHost<float[]> high_res_feats_0_;
    cuda_utils::CudaUniquePtrHost<float[]> high_res_feats_1_;
    cuda_utils::CudaUniquePtrHost<float[]> image_embed_;

    // Feature sizes
    size_t feats_0_size_;
    size_t feats_1_size_;
    size_t embed_size_;

    // Boxes and masks
    std::vector<std::vector<cv::Mat>> masks_;
    std::vector<std::vector<cv::Point2f>> box_coords_;
    std::vector<std::vector<float>> box_labels_;

    // Entropy data
    std::vector<cv::Mat> mat_entropies_;
    std::vector<float> entropies_;

    // Original input image dimensions
    std::vector<cv::Size> orig_im_sizes_;

    // State tracking
    bool encoder_initialized_;
    bool decoder_initialized_;
    bool features_ready_;
    int current_batch_size_;
};