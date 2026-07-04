/**
 * @file imu_processor.h
 * @author junyin qiu (j.qiu.1@research.gla.ac.uk)
 * @brief load imu measurements, and preintegration
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

struct imu_data_t
{
    std::uint64_t t;
    Eigen::Vector3d w;
    Eigen::Vector3d a;
};


class ImuIntegration
{
    public:
    ImuIntegration(const Eigen::Matrix<double, 6, 1> bias_ = Eigen::Matrix<double, 6, 1>::Zero());
    ~ImuIntegration() {}
    void Clear();
    void Initialize();
    void SetBias(const Eigen::Matrix<double, 6, 1> bias);
    Eigen::Quaterniond Get_dQ() const;
    Eigen::Matrix3d Get_dR() const;
    Eigen::Vector3d Get_dV() const;
    Eigen::Vector3d Get_dP() const;
    Eigen::Matrix<double, 5, 5> Get_dr() const;
    double Get_dT() const;

    Eigen::Matrix3d Get_JRg() const;
    Eigen::Matrix3d Get_JVg() const;
    Eigen::Matrix3d Get_JVa() const;
    Eigen::Matrix3d Get_JPg() const;
    Eigen::Matrix3d Get_JPa() const;
    Eigen::Matrix<double, 9, 6> Get_Jrb() const;

    void IntegrateNewMeasurement(const Eigen::Vector3d& w1, const Eigen::Vector3d& w2,
                                 const Eigen::Vector3d& a1, const Eigen::Vector3d& a2,
                                 const double dt);

    bool IntegrationInterval(const std::vector<imu_data_t>& imu_data_, std::uint64_t a, std::uint64_t b,
                             bool record = false);
    bool Reintegrated();
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    public:

    double dT;
    Eigen::Matrix<double, 5, 5> r;
    Eigen::Matrix<double, 9, 6> Jrb;
    Eigen::Matrix<double, 6, 1> b;
    std::vector<imu_data_t> imu_data;
    std::uint64_t time_start, time_end;
};

class ImuProcessor
{
public:
    ImuProcessor(const std::string &dataset, const std::string &dataset_path);
    bool GetImuData(int id, imu_data_t &imu_data);
    bool GetImuData(std::uint64_t t, imu_data_t &imu_data);
    std::uint64_t GetTimestamp(int id);
    int GetIdxBeforeEqual(std::uint64_t t);
    int GetIdxAfterEqual(std::uint64_t t);
    std::shared_ptr<ImuIntegration> GetImuIntegration(std::uint64_t t1, std::uint64_t t2);

    std::vector<imu_data_t> imu_dataset;
    int imu_dataset_size;
    std::uint64_t timestamp_start;
    std::uint64_t timestamp_end;
    double imu_rate;
};

