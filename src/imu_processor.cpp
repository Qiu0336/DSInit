/**
 * @file imu_processor.cpp
 * @author junyin qiu (j.qiu.1@research.gla.ac.uk)
 * @brief load imu measurements, and preintegration
 * @version 1.0
 * @date 2026-04-19
 *
 * @copyright Copyright (c) 2026
 */

#include "imu_processor.h"

ImuIntegration::ImuIntegration(const Eigen::Matrix<double, 6, 1> bias_)
{
    Clear();
    b = bias_;
}

void ImuIntegration::Clear()
{
    Initialize();
    b.setZero();
    imu_data.clear();
    time_start = 0;
    time_end = 0;
}

void ImuIntegration::Initialize()
{
    dT = 0.0;
    r.setIdentity();
    Jrb.setZero();
}

void ImuIntegration::SetBias(const Eigen::Matrix<double, 6, 1> bias) {
    b = bias;
}

Eigen::Quaterniond ImuIntegration::Get_dQ() const{
    Eigen::Quaterniond dQ(r.block<3, 3>(0, 0));
    return dQ;
}
Eigen::Matrix3d ImuIntegration::Get_dR() const{
    return r.block<3, 3>(0, 0);
}
Eigen::Vector3d ImuIntegration::Get_dV() const{
    return r.block<3, 1>(0, 3);
}
Eigen::Vector3d ImuIntegration::Get_dP() const{
    return r.block<3, 1>(0, 4);
}
Eigen::Matrix<double, 5, 5> ImuIntegration::Get_dr() const{
    return r;
}
double ImuIntegration::Get_dT() const{
    return dT;
}

Eigen::Matrix3d ImuIntegration::Get_JRg() const{
    return Jrb.block<3, 3>(0, 0);
}
Eigen::Matrix3d ImuIntegration::Get_JVg() const{
    return Jrb.block<3, 3>(3, 0);
}
Eigen::Matrix3d ImuIntegration::Get_JVa() const{
    return Jrb.block<3, 3>(3, 3);
}
Eigen::Matrix3d ImuIntegration::Get_JPg() const{
    return Jrb.block<3, 3>(6, 0);
}
Eigen::Matrix3d ImuIntegration::Get_JPa() const{
    return Jrb.block<3, 3>(6, 3);
}
Eigen::Matrix<double, 9, 6> ImuIntegration::Get_Jrb() const{
    return Jrb;
}

void ImuIntegration::IntegrateNewMeasurement(const Eigen::Vector3d& w1, const Eigen::Vector3d& w2,
                                             const Eigen::Vector3d& a1, const Eigen::Vector3d& a2,
                                             const double dt)
{
    const Eigen::Vector3d w = 0.5*(w1 + w2) - b.head<3>();
    const Eigen::Vector3d a = 0.5*(a1 + a2) - b.tail<3>();
    const Eigen::Matrix3d Skew_a = Skew(a);
    const Eigen::Matrix3d Rij_1 = r.block<3, 3>(0, 0);
    const Eigen::Matrix3d Rjj_1 = ExpSO3(w*dt).transpose();

    Eigen::Matrix<double, 15, 15> A;
    A.setIdentity();
    Eigen::Matrix<double, 15, 12> B;
    B.setZero();

    A.block<3, 3>(0, 0) = Rjj_1;
    A.block<3, 3>(3, 0) = -Rij_1*Skew_a*dt;
    A.block<3, 3>(6, 0) = -0.5*Rij_1*Skew_a*dt*dt;
    A.block<3, 3>(6, 3) = Eigen::Matrix3d::Identity()*dt;

    B.block<3, 3>(0, 0) = RightJacobianSO3(w*dt)*dt;
    B.block<3, 3>(3, 3) = Rij_1*dt;
    B.block<3, 3>(6, 3) = 0.5*Rij_1*dt*dt;
    B.block<6, 6>(9, 6) = Eigen::Matrix<double, 6, 6>::Identity()*dt;

    A.block<9, 6>(0, 9) = - B.block<9, 6>(0, 0);

    Jrb = A.block<9, 9>(0, 0)*Jrb + A.block<9, 6>(0, 9);
    dT += dt;

    Eigen::Matrix<double, 5, 5> fai = r;
    fai.block<3, 1>(0, 4) = r.block<3, 1>(0, 4) + r.block<3, 1>(0, 3)*dt;
    Eigen::Matrix<double, 5, 5> dr;
    dr.setIdentity();
    dr.block<3, 3>(0, 0) = ExpSO3(w*dt);
    dr.block<3, 1>(0, 3) = a*dt;
    dr.block<3, 1>(0, 4) = 0.5*a*dt*dt;
    r = fai*dr;
}


bool ImuIntegration::IntegrationInterval(const std::vector<imu_data_t>& imu_data_, std::uint64_t a, std::uint64_t b, bool record)
{
    if(a >= b)
    {
        std::cerr << "Integration time error !!!";
        return false;
    }
    Eigen::Vector3d w1, w2, a1, a2;
    std::uint64_t t1, t2;
    double dt;
    auto it = imu_data_.cbegin();
    w1 = it->w;
    a1 = it->a;
    t1 = it->t;
    if(t1 > a)
    {
        dt = (t1 - a)*1e-9;
        IntegrateNewMeasurement(w1, w1, a1, a1, dt);
    }

    for(it ++; it != imu_data_.cend(); it ++)
    {
        w2 = it->w;
        a2 = it->a;
        t2 = it->t;
        dt = (t2 - t1)*1e-9;

        if(t1 >= a && t2 <= b)
        {
            IntegrateNewMeasurement(w1, w2, a1, a2, dt);
        }
        else if(t1 < a && t2 > a)
        {
            double dl = (a - t1)*1e-9;
            double dr = (t2 - a)*1e-9;
            IntegrateNewMeasurement(w1*dr/dt + w2*dl/dt, w2, a1*dr/dt + a2*dl/dt, a2, dr);
        }
        else if(t1 < b && t2 > b)
        {
            double dl = (b - t1)*1e-9;
            double dr = (t2 - b)*1e-9;
            IntegrateNewMeasurement(w1, w1*dr/dt + w2*dl/dt, a1, a1*dr/dt + a2*dl/dt, dl);
        }

        w1 = w2;
        a1 = a2;
        t1 = t2;
    }
    if(record)
    {
        imu_data = imu_data_;
        time_start = a;
        time_end = b;
    }
    return true;
}

bool ImuIntegration::Reintegrated()
{
    if(imu_data.empty())
        return false;
    Initialize();
    return IntegrationInterval(imu_data, time_start, time_end);
}

ImuProcessor::ImuProcessor(const std::string &dataset,
                           const std::string &dataset_path)
{
    if(dataset == "euroc" || dataset == "tumvi")
    {
        std::string imu_dataset_path = dataset_path + "/mav0/imu0/data.csv";
        std::ifstream f;
        f.open(imu_dataset_path.c_str());
        while(!f.eof())
        {
            std::string line;
            char tmp;
            while(std::getline(f, line))
            {
                if(line[0] == '#') continue;
                std::stringstream ss(line);
                imu_data_t imu_t;
                ss >> imu_t.t >> tmp
                   >> imu_t.w.x() >> tmp >> imu_t.w.y() >> tmp >> imu_t.w.z() >> tmp
                   >> imu_t.a.x() >> tmp >> imu_t.a.y() >> tmp >> imu_t.a.z();
                imu_dataset.push_back(imu_t);
            }
        }
        f.close();
    }
    else
    {
        std::cerr << "Unknown dataset !!!" << std::endl;
    }

    imu_dataset_size = imu_dataset.size();
    std::cout << " read " << imu_dataset.size() << " imu data !!!" << std::endl;

    timestamp_start = imu_dataset[0].t;
    timestamp_end = imu_dataset.back().t;
    imu_rate = static_cast<double>(imu_dataset_size - 1) / ((timestamp_end - timestamp_start)*1e-9);
    std::cout << " imu_rate: " << imu_rate << std::endl;
}

bool ImuProcessor::GetImuData(int id, imu_data_t &imu_data)
{
    if(id < 0 || id >= imu_dataset_size)
    {
        std::cerr << "GetImuData error !!!" << std::endl;
        return false;
    }
    imu_data = imu_dataset[id];
    return true;
}

bool ImuProcessor::GetImuData(std::uint64_t t, imu_data_t &imu_data)
{
    int idx = GetIdxBeforeEqual(t);
    return GetImuData(idx, imu_data);
}

std::uint64_t ImuProcessor::GetTimestamp(int id)
{
    if(id >= 0 && id < imu_dataset_size)
        return imu_dataset[id].t;
    std::cerr << "GetImuTimestamp error !!!" << std::endl;
    return -1;
}

int ImuProcessor::GetIdxBeforeEqual(std::uint64_t t)
{
    if(t < timestamp_start || t >= timestamp_end)
    {
        std::cerr << "GetIdxBeforeEqual error !!!" << std::endl;
        return -1;
    }
    int idx = std::max(0, (int)(std::floor((t - timestamp_start)*1e-9*imu_rate) - 1));

    std::uint64_t idx_t = GetTimestamp(idx);
    if(idx_t < 0) return -1;
    while(idx_t > t)
    {
        --idx;
        idx_t = GetTimestamp(idx);
        if(idx_t < 0) return -1;
    }
    while(idx_t <= t)
    {
        ++idx;
        idx_t = GetTimestamp(idx);
        if(idx_t < 0) return -1;
    }
    --idx;
    return idx;
}

int ImuProcessor::GetIdxAfterEqual(std::uint64_t t)
{
    if(t <= timestamp_start || t > timestamp_end)
    {
        std::cerr << "GetIdxBeforeEqual error !!!" << std::endl;
        return -1;
    }
    int idx = std::max(0, (int)(std::floor((t - timestamp_start)*1e-9*imu_rate) - 1));

    std::uint64_t idx_t = GetTimestamp(idx);
    if(idx_t < 0) return -1;
    while(idx_t > t)
    {
        --idx;
        idx_t = GetTimestamp(idx);
        if(idx_t < 0) return -1;
    }
    while(idx_t < t)
    {
        ++idx;
        idx_t = GetTimestamp(idx);
        if(idx_t < 0) return -1;
    }
    return idx;
}

std::shared_ptr<ImuIntegration> ImuProcessor::GetImuIntegration(std::uint64_t t1, std::uint64_t t2)
{
    int start_idx = GetIdxBeforeEqual(t1);
    int end_idx = GetIdxAfterEqual(t2);
    if(start_idx == -1 || end_idx == -1)
        return nullptr;
    std::vector<imu_data_t> imu_data_tg;
    imu_data_tg.insert(imu_data_tg.begin(), imu_dataset.begin() + start_idx, imu_dataset.begin() + end_idx + 1);
    std::shared_ptr<ImuIntegration> imu_integration = std::make_shared<ImuIntegration>();
    imu_integration->IntegrationInterval(imu_data_tg, t1, t2, true);
    return imu_integration;
}

