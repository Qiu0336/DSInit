/**
 * @file image_processor.cpp
 * @author junyin qiu (j.qiu.1@research.gla.ac.uk)
 * @brief load and undistort images
 * @version 1.0
 * @date 2026-04-19
 *
 * @copyright Copyright (c) 2026
 */

#include "image_processor.h"

ImageProcessor::ImageProcessor(const std::string &dataset,
                               const std::string &dataset_path,
                               std::shared_ptr<CameraModel> &_camera0,
                               std::shared_ptr<CameraModel> &_camera1)
{
    camera0 = _camera0;
    camera1 = _camera1;

    if(dataset == "euroc" || dataset == "tumvi")
    {
        std::string image0_data_path = dataset_path + "/mav0/cam0/data/";
        std::string image0_timestamp_path = dataset_path + "/mav0/cam0/data.csv";
        std::string image1_data_path = dataset_path + "/mav0/cam1/data/";
        std::string image1_timestamp_path = dataset_path + "/mav0/cam1/data.csv";

        std::ifstream f0;
        f0.open(image0_timestamp_path.c_str());
        image0_timestamps.reserve(5000);
        image0_file_names.reserve(5000);
        while(!f0.eof())
        {
            std::string line;
            char tmp;
            std::uint64_t t;
            std::string path;
            while(std::getline(f0, line))
            {
                if(line[0] == '#') continue;
                std::stringstream ss(line);
                ss >> t >> tmp >> path;
                image0_timestamps.push_back(t);
                image0_file_names.push_back(image0_data_path + path);
            }
        }
        f0.close();

        std::ifstream f1;
        f1.open(image1_timestamp_path.c_str());
        image1_timestamps.reserve(5000);
        image1_file_names.reserve(5000);
        while(!f1.eof())
        {
            std::string line;
            char tmp;
            std::uint64_t t;
            std::string path;
            while(std::getline(f1, line))
            {
                if(line[0] == '#') continue;
                std::stringstream ss(line);
                ss >> t >> tmp >> path;
                image1_timestamps.push_back(t);
                image1_file_names.push_back(image1_data_path + path);
            }
        }
        f1.close();
    }
    else
    {
        std::cerr << "Unknown dataset !!!" << std::endl;
    }


    image0_data_size = image0_file_names.size();
    image1_data_size = image1_file_names.size();
    std::cout << " read " << image0_data_size << " image0 !!!" << std::endl;
    std::cout << " read " << image1_data_size << " image1 !!!" << std::endl;

    // check if image0 and image1 are synchronized
    bool is_synchronized = true;
    if(image0_data_size == image1_data_size)
    {
        for(size_t i = 0; i < image0_timestamps.size(); ++i)
        {
            if(image0_timestamps[i] != image1_timestamps[i])
                is_synchronized = false;
        }
    }
    else
        is_synchronized = false;
    if(is_synchronized)
        std::cout << "is_synchronized" << std::endl;
    else
        std::cout << "image0 and image1 are not synchronized !!!" << std::endl;

    if(!is_synchronized)
    {
        std::cout << " image synchronizing ..." << std::endl;
        std::vector<std::string> image0_file_names_tmp, image1_file_names_tmp;
        std::vector<std::uint64_t> image0_timestamps_tmp, image1_timestamps_tmp;
        size_t j = 0;
        for(size_t i = 0; i < image0_timestamps.size() && j < image1_timestamps.size();)
        {
            if(image0_timestamps[i] > image1_timestamps[j])
            {
                ++j;
                continue;
            }
            else if(image0_timestamps[i] == image1_timestamps[j])
            {
                image0_timestamps_tmp.push_back(image0_timestamps[i]);
                image0_file_names_tmp.push_back(image0_file_names[i]);
                image1_timestamps_tmp.push_back(image1_timestamps[j]);
                image1_file_names_tmp.push_back(image1_file_names[j]);
            }
            ++i;
        }
        image0_file_names = image0_file_names_tmp;
        image0_timestamps = image0_timestamps_tmp;
        image1_file_names = image1_file_names_tmp;
        image1_timestamps = image1_timestamps_tmp;
        image0_data_size = image0_file_names.size();
        image1_data_size = image1_file_names.size();
        std::cout << " read " << image0_data_size << " image0 !!!" << std::endl;
        std::cout << " read " << image1_data_size << " image1 !!!" << std::endl;
    }


}

bool ImageProcessor::GetStereoImage(int id, cv::Mat &img0, cv::Mat &img1)
{
    if(id < 0 || id >= image0_data_size)
    {
        std::cerr << "GetImage error !!!" << std::endl;
        return false;
    }
    cv::Mat origin_img0 = cv::imread(image0_file_names[id], CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat origin_img1 = cv::imread(image1_file_names[id], CV_LOAD_IMAGE_GRAYSCALE);
    img0 = camera0->UndistortedImage(origin_img0);
    img1 = camera1->UndistortedImage(origin_img1);
    return true;
}

std::uint64_t ImageProcessor::GetTimestamp(int id)
{
    if(id >= 0 && id < image0_data_size)
        return image0_timestamps[id];
    std::cerr << "GetImageTimestamp error !!!" << std::endl;
    return -1;
}

