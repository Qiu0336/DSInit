/**
 * @file image_processor.h
 * @author junyin qiu (j.qiu.1@research.gla.ac.uk)
 * @brief load and undistort images
 * @version 1.0
 * @date 2026-04-19
 *
 * @copyright Copyright (c) 2026
 */

#pragma once

#include <string>
#include <cstring>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
#include "utils/camera.h"

class ImageProcessor
{
public:
    ImageProcessor(const std::string &dataset,
                   const std::string &dataset_path,
                   std::shared_ptr<CameraModel> &_camera0,
                   std::shared_ptr<CameraModel> &_camera1);
    bool GetStereoImage(int id, cv::Mat &img0, cv::Mat &img1);
    std::uint64_t GetTimestamp(int id);
    std::shared_ptr<CameraModel> camera0, camera1;
    std::vector<std::string> image0_file_names, image1_file_names;
    std::vector<std::uint64_t> image0_timestamps, image1_timestamps;
    int image0_data_size, image1_data_size;
};
