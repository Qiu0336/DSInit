/**
 * @file gt_loader.h
 * @author junyin qiu (j.qiu.1@research.gla.ac.uk)
 * @brief ground truth state loader
 * @version 1.0
 * @date 2026-04-19
 *
 * @copyright Copyright (c) 2026
 */

#pragma once

#include <string>
#include <cstring>
#include <vector>
#include <list>
#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include "utils/so3.h"

struct state_t
{
    std::uint64_t t;
    Eigen::Quaterniond q;
    Eigen::Vector3d p;
    Eigen::Vector3d v;
    Eigen::Vector3d bg;
    Eigen::Vector3d ba;

    Eigen::Isometry3d T() const
    {
        Eigen::Isometry3d T(q);
        T.pretranslate(p);
        return T;
    }

};


class GtLoader
{
public:
    GtLoader(const std::string &dataset, const std::string &dataset_path);
    bool GetGtData(int id, state_t  &state);
    std::uint64_t GetTimestamp(int id);
    bool GetGtDataAtTimestamp(std::uint64_t t, state_t &state);

    std::vector<state_t> gt_data;
    int gt_data_size;
    std::uint64_t timestamp_start;
    std::uint64_t timestamp_end;
};

