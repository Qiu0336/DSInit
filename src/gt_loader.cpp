/**
 * @file gt_loader.cpp
 * @author junyin qiu (j.qiu.1@research.gla.ac.uk)
 * @brief ground truth state loader
 * @version 1.0
 * @date 2026-04-19
 *
 * @copyright Copyright (c) 2026
 */

#include "gt_loader.h"

GtLoader::GtLoader(const std::string &dataset,
                   const std::string &dataset_path)
{
    if(dataset == "euroc")
    {
        std::string gt_data_path = dataset_path + "/mav0/state_groundtruth_estimate0/data.csv";
        std::ifstream f;
        f.open(gt_data_path.c_str());
        while(!f.eof())
        {
            std::string line;
            char tmp;
            while(std::getline(f, line))
            {
                if(line[0] == '#') continue;
                std::stringstream ss(line);
                state_t sta_t;
                ss >> sta_t.t >> tmp
                   >> sta_t.p.x() >> tmp >> sta_t.p.y() >> tmp >> sta_t.p.z() >> tmp
                   >> sta_t.q.w() >> tmp >> sta_t.q.x() >> tmp >> sta_t.q.y() >> tmp >> sta_t.q.z() >> tmp
                   >> sta_t.v.x() >> tmp >> sta_t.v.y() >> tmp >> sta_t.v.z() >> tmp
                   >> sta_t.bg.x() >> tmp >> sta_t.bg.y() >> tmp >> sta_t.bg.z() >> tmp
                   >> sta_t.ba.x() >> tmp >> sta_t.ba.y() >> tmp >> sta_t.ba.z();
                gt_data.push_back(sta_t);
            }
        }
        f.close();
    }
    else if(dataset == "tumvi")
    {
        std::string gt_data_path = dataset_path + "/mav0/mocap0/data.csv";
        std::ifstream f;
        f.open(gt_data_path.c_str());
        while(!f.eof())
        {
            std::string line;
            char tmp;
            while(std::getline(f, line))
            {
                if(line[0] == '#') continue;
                std::stringstream ss(line);
                state_t sta_t;
                ss >> sta_t.t >> tmp
                   >> sta_t.p.x() >> tmp >> sta_t.p.y() >> tmp >> sta_t.p.z() >> tmp
                   >> sta_t.q.w() >> tmp >> sta_t.q.x() >> tmp >> sta_t.q.y() >> tmp >> sta_t.q.z();
                gt_data.push_back(sta_t);
            }
        }
        f.close();
    }
    else
    {
        std::cerr << "Unknown dataset !!!" << std::endl;
    }

    gt_data_size = gt_data.size();
    timestamp_start = gt_data[0].t;
    timestamp_end = gt_data[gt_data_size - 1].t;

    std::cout << " read " << gt_data.size() << " gt data !!!" << std::endl;
}

bool GtLoader::GetGtData(int id, state_t &state)
{
    if(id < 0 || id >= gt_data_size)
    {
        std::cerr << "GetGtData error !!!" << std::endl;
        return false;
    }
    state = gt_data[id];
    return true;
}

std::uint64_t GtLoader::GetTimestamp(int id)
{
    if(id >= 0 && id < gt_data_size)
        return gt_data[id].t;
    std::cerr << "GetGtTimestamp error !!!" << std::endl;
    return -1;
}

bool GtLoader::GetGtDataAtTimestamp(std::uint64_t t, state_t &state)
{
    if(t < timestamp_start || t >= timestamp_end)
    {
        std::cerr << "GetGtData AtTimestamp error !!!" << std::endl;
        return false;
    }
    double tos = double(t - timestamp_start)/double(timestamp_end - timestamp_start);
    int idx = std::max(0, int(std::floor(gt_data_size*tos) - 1));
    std::uint64_t idx_t = GetTimestamp(idx);

    while(idx_t > t)
    {
        --idx;
        idx_t = GetTimestamp(idx);
        if(idx_t < 0) return false;
    }
    while(idx_t <= t)
    {
        ++idx;
        idx_t = GetTimestamp(idx);
        if(idx_t < 0) return false;
    }
    --idx;

    state_t state1, state2;
    if(GetGtData(idx, state1) && GetGtData(idx+1, state2))
    {
        std::uint64_t t1 = state1.t;
        std::uint64_t t2 = state2.t;
        double tos = double(t - t1)/double(t2 - t1);
        Eigen::Matrix<double, 12, 1> pvbgba1;
        Eigen::Matrix<double, 12, 1> pvbgba2;
        pvbgba1 << state1.p, state1.v, state1.bg, state1.ba;
        pvbgba2 << state2.p, state2.v, state2.bg, state2.ba;
        Eigen::Matrix<double, 12, 1> s = pvbgba1 + tos*(pvbgba2 - pvbgba1);
        Eigen::Matrix3d R1 = state1.q.toRotationMatrix();
        Eigen::Matrix3d R2 = state2.q.toRotationMatrix();
        Eigen::Matrix3d R = R1*ExpSO3(tos*LogSO3(R1.transpose()*R2));
        Eigen::Quaterniond q(R);
        q.normalize();
        state.q = q;
        state.p = s.segment<3>(0);
        state.v = s.segment<3>(3);
        state.bg = s.segment<3>(6);
        state.ba = s.segment<3>(9);
        state.t = t;
        return true;
    }
    return false;
}
