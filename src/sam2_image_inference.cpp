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

#include "sam2_image_inference.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "colormap.hpp"
#include "omp.h"
#include "utils.hpp"

SAM2Image::SAM2Image(const std::string& encoder_path,
                     const std::string& decoder_path,
                     const cv::Size& encoder_input_size,
                     const std::string& model_precision,
                     const int decoder_batch_limit,
                     const size_t max_workspace_size,
                     const std::vector<std::string>& plugin_paths,
                     const int32_t dla_core_id,
                     const bool profile_per_layer)
    : encoder_path_(encoder_path),
      decoder_path_(decoder_path),
      encoder_input_size_(encoder_input_size),
      model_precision_(model_precision),
      decoder_batch_limit_(decoder_batch_limit),
      max_workspace_size_(max_workspace_size),
      plugin_paths_(plugin_paths),
      dla_core_id_(dla_core_id),
      profile_per_layer_(profile_per_layer),
      encoder_initialized_(false),
      decoder_initialized_(false),
      features_ready_(false),
      current_batch_size_(0),
      feats_0_size_(0),
      feats_1_size_(0),
      embed_size_(0)
{
    cv::setNumThreads(1);

    // Initialize encoder and decoder
    if (!InitializeEncoder() || !InitializeDecoder())
    {
        throw std::runtime_error("Failed to initialize SAM2Image components");
    }
}

SAM2Image::~SAM2Image() = default;

bool SAM2Image::InitializeEncoder()
{
    try
    {
        encoder_ = std::make_unique<SAM2ImageEncoder>(encoder_path_,
                                                      model_precision_,
                                                      max_workspace_size_,
                                                      encoder_input_size_,
                                                      plugin_paths_,
                                                      "",  // engine_path - let it auto-generate
                                                      dla_core_id_,
                                                      profile_per_layer_);

        encoder_initialized_ = true;
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to initialize encoder: " << e.what() << std::endl;
        return false;
    }
}

bool SAM2Image::InitializeDecoder()
{
    try
    {
        // Get encoder output sizes
        std::vector<int> encoder_output_sizes;
        if (encoder_initialized_)
        {
            encoder_output_sizes = {static_cast<int>(encoder_->GetEmbedSize()),
                                    static_cast<int>(encoder_->GetFeats0Size()),
                                    static_cast<int>(encoder_->GetFeats1Size())};
        }
        else
        {
            // Default sizes if encoder not initialized yet
            encoder_output_sizes = {256, 64, 64};
        }

        decoder_ = std::make_unique<SAM2ImageDecoder>(decoder_path_,
                                                      model_precision_,
                                                      max_workspace_size_,
                                                      encoder_input_size_,
                                                      encoder_output_sizes,
                                                      0.0f,  // mask_threshold
                                                      plugin_paths_,
                                                      "",  // engine_path - let it auto-generate
                                                      dla_core_id_,
                                                      profile_per_layer_);

        decoder_initialized_ = true;
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to initialize decoder: " << e.what() << std::endl;
        return false;
    }
}

bool SAM2Image::RunEncoder(const std::vector<cv::Mat>& images)
{
    if (!encoder_initialized_)
    {
        std::cerr << "Encoder not initialized" << std::endl;
        return false;
    }

    if (!ValidateInputs(images))
    {
        return false;
    }

    // Clear all previous results
    masks_.clear();
    orig_im_sizes_.clear();
    mat_entropies_.clear();
    entropies_.clear();
    features_ready_ = false;

    // Run encoder to get results
    if (!encoder_->EncodeImage(images))
    {
        std::cerr << "Failed to encode images" << std::endl;
        return false;
    }

    // Store original image sizes
    for (const auto& image : images)
    {
        orig_im_sizes_.push_back(image.size());
    }

    // Copy encoder features to local storage
    if (!CopyEncoderFeatures())
    {
        std::cerr << "Failed to copy encoder features" << std::endl;
        return false;
    }

    features_ready_ = true;
    current_batch_size_ = static_cast<int>(images.size());

    return true;
}

bool SAM2Image::RunDecoder(const std::vector<std::vector<cv::Rect>>& boxes)
{
    if (!ValidateBoxes(boxes))
    {
        return false;
    }

    // Convert boxes to point coordinates and labels
    std::vector<std::vector<cv::Point2f>> point_coords;
    std::vector<std::vector<float>> point_labels;
    BoxesToPointCoords(boxes, point_coords, point_labels);

    return RunDecoder(point_coords, point_labels);
}

bool SAM2Image::RunDecoder(const std::vector<std::vector<cv::Point2f>>& point_coords,
                           const std::vector<std::vector<float>>& point_labels)
{
    if (!decoder_initialized_)
    {
        std::cerr << "Decoder not initialized" << std::endl;
        return false;
    }

    if (!features_ready_)
    {
        std::cerr << "Encoder features not ready. Run encoder first." << std::endl;
        return false;
    }

    assert(point_coords.size() == orig_im_sizes_.size());

    for (size_t i = 0; i < point_coords.size(); i++)
    {
        auto coords_per_image = point_coords[i];
        auto labels_per_image = point_labels[i];
        std::vector<cv::Mat> masks_per_image;

        for (int z = 0; z < static_cast<int>(coords_per_image.size()); z += decoder_batch_limit_)
        {
            int current_batch_size =
                std::min(decoder_batch_limit_, static_cast<int>(coords_per_image.size()) - z);

            ClearBoxes();

            // Pre-allocate local storage
            std::vector<std::vector<cv::Point2f>> local_coords(current_batch_size);
            std::vector<std::vector<float>> local_labels(current_batch_size);

            // Generate batch information in parallel
#pragma omp parallel for
            for (int j = 0; j < current_batch_size; j++)
            {
                if (z + j < static_cast<int>(coords_per_image.size()))
                {
                    local_coords[j] = {coords_per_image[z + j]};
                    local_labels[j] = {labels_per_image[z + j]};
                }
            }

            // Merge local results into member variables
            box_coords_ = std::move(local_coords);
            box_labels_ = std::move(local_labels);

            if (!DecodeMask(
                    orig_im_sizes_[i], static_cast<int>(i), masks_per_image, current_batch_size))
            {
                std::cerr << "Failed to decode mask for image " << i << std::endl;
                return false;
            }
        }

        masks_.push_back(masks_per_image);
    }

    return true;
}

bool SAM2Image::DecodeMask(const cv::Size& orig_im_size,
                           const int img_batch_idx,
                           std::vector<cv::Mat>& masks_per_image,
                           const int current_batch_size)
{
    if (!decoder_->Predict(image_embed_,
                           high_res_feats_0_,
                           high_res_feats_1_,
                           box_coords_,
                           box_labels_,
                           orig_im_size,
                           img_batch_idx,
                           current_batch_size))
    {
        return false;
    }

    // Get results from decoder
    auto masks_per_image_per_decoder_batch = decoder_->GetResultMasks();
    auto mat_entropies_batch = decoder_->GetMatEntropies();
    auto entropies_batch = decoder_->GetEntropies();

    // Append results
    masks_per_image.insert(masks_per_image.end(),
                           masks_per_image_per_decoder_batch.begin(),
                           masks_per_image_per_decoder_batch.end());
    mat_entropies_.insert(
        mat_entropies_.end(), mat_entropies_batch.begin(), mat_entropies_batch.end());
    entropies_.insert(entropies_.end(), entropies_batch.begin(), entropies_batch.end());

    return true;
}

bool SAM2Image::CopyEncoderFeatures()
{
    try
    {
        // Get feature sizes
        embed_size_ = encoder_->GetEmbedSize();
        feats_0_size_ = encoder_->GetFeats0Size();
        feats_1_size_ = encoder_->GetFeats1Size();

        // Allocate local storage
        image_embed_ = cuda_utils::make_unique_host<float[]>(embed_size_, cudaHostAllocPortable);
        high_res_feats_0_ =
            cuda_utils::make_unique_host<float[]>(feats_0_size_, cudaHostAllocPortable);
        high_res_feats_1_ =
            cuda_utils::make_unique_host<float[]>(feats_1_size_, cudaHostAllocPortable);

        // Copy data from encoder
        std::memcpy(
            image_embed_.get(), encoder_->GetEmbedData().get(), embed_size_ * sizeof(float));
        std::memcpy(high_res_feats_0_.get(),
                    encoder_->GetFeats0Data().get(),
                    feats_0_size_ * sizeof(float));
        std::memcpy(high_res_feats_1_.get(),
                    encoder_->GetFeats1Data().get(),
                    feats_1_size_ * sizeof(float));

        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to copy encoder features: " << e.what() << std::endl;
        return false;
    }
}

void SAM2Image::BoxesToPointCoords(const std::vector<std::vector<cv::Rect>>& boxes,
                                   std::vector<std::vector<cv::Point2f>>& point_coords,
                                   std::vector<std::vector<float>>& point_labels) const
{
    point_coords.clear();
    point_labels.clear();

    for (const auto& boxes_per_image : boxes)
    {
        std::vector<cv::Point2f> coords_per_image;
        std::vector<float> labels_per_image;

        for (const auto& box : boxes_per_image)
        {
            // Calculate two corner points of the box
            coords_per_image.push_back(cv::Point2f(box.x, box.y));
            coords_per_image.push_back(cv::Point2f(box.x + box.width, box.y + box.height));

            // Label data for top-left and bottom-right corners of bbox
            labels_per_image.push_back(2.0f);
            labels_per_image.push_back(3.0f);
        }

        point_coords.push_back(coords_per_image);
        point_labels.push_back(labels_per_image);
    }
}

bool SAM2Image::ValidateInputs(const std::vector<cv::Mat>& images) const
{
    if (images.empty())
    {
        std::cerr << "No input images provided" << std::endl;
        return false;
    }

    for (const auto& image : images)
    {
        if (image.empty())
        {
            std::cerr << "Empty image found in input" << std::endl;
            return false;
        }
    }

    return true;
}

bool SAM2Image::ValidateBoxes(const std::vector<std::vector<cv::Rect>>& boxes) const
{
    if (boxes.size() != orig_im_sizes_.size())
    {
        std::cerr << "Number of box sets does not match number of images" << std::endl;
        return false;
    }

    return true;
}

void SAM2Image::ClearBoxes()
{
    box_coords_.clear();
    box_labels_.clear();
}

const std::vector<std::vector<cv::Mat>>& SAM2Image::GetMasks() const
{
    return masks_;
}

cv::Mat SAM2Image::GetMaxEntropy(float& peak_entropy_score) const
{
    cv::Mat max_ent, max_ent_jet;

    if (mat_entropies_.empty())
    {
        peak_entropy_score = 0.0f;
        return max_ent;
    }

    int height = mat_entropies_[0].rows;
    int width = mat_entropies_[0].cols;
    max_ent = cv::Mat::zeros(height, width, CV_8UC1);
    max_ent_jet = cv::Mat::zeros(height, width, CV_8UC3);

    // Find maximum entropy at each pixel
    for (const auto& entropy_mat : mat_entropies_)
    {
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                max_ent.at<unsigned char>(y, x) =
                    std::max(entropy_mat.at<unsigned char>(y, x), max_ent.at<unsigned char>(y, x));
            }
        }
    }

    // Apply colormap
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            const auto& color = jet_colormap[max_ent.at<unsigned char>(y, x)];
            max_ent_jet.at<cv::Vec3b>(y, x)[0] = color[0];
            max_ent_jet.at<cv::Vec3b>(y, x)[1] = color[1];
            max_ent_jet.at<cv::Vec3b>(y, x)[2] = color[2];
        }
    }

    // Calculate average entropy
    float sum_ent = 0.0f;
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            sum_ent += max_ent.at<unsigned char>(y, x);
        }
    }
    sum_ent /= (width * height);

    // Add text annotation
    cv::putText(max_ent_jet,
                std::to_string(sum_ent),
                cv::Point(32, 32),
                cv::FONT_HERSHEY_SIMPLEX,
                1.0,
                cv::Scalar(255, 255, 255),
                1);

    peak_entropy_score = sum_ent;
    return max_ent_jet;
}

const std::vector<float>& SAM2Image::GetEntropies() const
{
    return entropies_;
}

const std::vector<cv::Mat>& SAM2Image::GetMatEntropies() const
{
    return mat_entropies_;
}

// Getter methods for encoder features
const cuda_utils::CudaUniquePtrHost<float[]>& SAM2Image::GetImageEmbed() const
{
    return image_embed_;
}

const cuda_utils::CudaUniquePtrHost<float[]>& SAM2Image::GetHighResFeats0() const
{
    return high_res_feats_0_;
}

const cuda_utils::CudaUniquePtrHost<float[]>& SAM2Image::GetHighResFeats1() const
{
    return high_res_feats_1_;
}

// Configuration getters
cv::Size SAM2Image::GetEncoderInputSize() const
{
    return encoder_input_size_;
}

std::string SAM2Image::GetModelPrecision() const
{
    return model_precision_;
}

int SAM2Image::GetDecoderBatchLimit() const
{
    return decoder_batch_limit_;
}

// Setup methods
bool SAM2Image::SetupEncoder(tensorrt_common::ProfileDimsPtr profile_dims,
                             tensorrt_common::NetworkIOPtr network_io)
{
    if (!encoder_initialized_)
    {
        std::cerr << "Encoder not initialized" << std::endl;
        return false;
    }

    return encoder_->Setup(std::move(profile_dims), std::move(network_io));
}

bool SAM2Image::SetupDecoder(tensorrt_common::ProfileDimsPtr profile_dims,
                             tensorrt_common::NetworkIOPtr network_io)
{
    if (!decoder_initialized_)
    {
        std::cerr << "Decoder not initialized" << std::endl;
        return false;
    }

    return decoder_->Setup(std::move(profile_dims), std::move(network_io));
}

// Profiling methods
void SAM2Image::PrintEncoderProfiling() const
{
    if (encoder_initialized_)
    {
        encoder_->PrintProfiling();
    }
}

void SAM2Image::PrintDecoderProfiling() const
{
    if (decoder_initialized_)
    {
        decoder_->PrintProfiling();
    }
}

// Precision getters
std::string SAM2Image::GetEncoderPrecision() const
{
    if (encoder_initialized_)
    {
        return encoder_->GetPrecision();
    }
    return "Unknown";
}

std::string SAM2Image::GetDecoderPrecision() const
{
    if (decoder_initialized_)
    {
        return decoder_->GetPrecision();
    }
    return "Unknown";
}