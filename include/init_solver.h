/**
 * @file init_solver.h
 * @author junyin qiu (j.qiu.1@research.gla.ac.uk)
 * @brief direct sparse initialization for stereo VIO
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
#include <numeric>
#include <random>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/utility.hpp>

#include "utils/camera.h"
#include "utils/so3.h"
#include "imu_processor.h"

extern Eigen::Vector3d gravity_vector;

const int staticPattern[9][2] =
{
    {0,0}, {-1,-1}, {1,-1}, {-1,1}, {0,-2}, {2,0}, {-2,0}, {0,2}, {1,1}
};

#define patternNum 8
#define patternP staticPattern
#define patternPadding 2
#define MAX_PYR_LEVELS 5

class LambdaBody : public cv::ParallelLoopBody
{
public:
    explicit LambdaBody(const std::function<void(const cv::Range &)> &body): body_(body) {}
    void operator()(const cv::Range &range) const override { body_(range); }
private:
    std::function<void(const cv::Range &)> body_;
};


class initRes
{
public:
    initRes(){}
    void eval(const Eigen::Vector3d& gt_bg, const Eigen::Vector3d& gt_v0, const Eigen::Vector3d& gt_g0,
              const std::vector<Eigen::Isometry3d>& gt_dT)
    {
        if(!success)
        {
            return;
        }
        statistics[0] = runtime;
        statistics[1] = std::abs(gt_bg.norm() - bg.norm())*100.0 / gt_bg.norm();
        statistics[2] = std::fabs(gt_v0.norm() - v0.norm());
        statistics[3] = 180. * std::acos(gt_g0.normalized().dot(g0.normalized())) / EIGEN_PI;

        double err_R = 0;
        double err_t = 0;
        for(size_t i = 0; i < gt_dT.size(); ++i)
        {
            double w = 0.5*((delta_R[i].transpose()*gt_dT[i].linear()).trace() - 1.0);
            double er = std::acos( std::max(-1.0, std::min(1.0, w)) ) * 180.0 / M_PI;
            err_R += er*er;
            err_t += (delta_t[i] - gt_dT[i].translation()).squaredNorm();
        }
        err_R /= gt_dT.size();
        err_t /= gt_dT.size();

        statistics[4] = std::sqrt(err_R);
        statistics[5] = std::sqrt(err_t);
    }

    void print()
    {
        if(!success)
        {
            std::cout << "init failed !!!" << std::endl;
            return;
        }

        std::cout << "=====init success=====" << std::endl;
        std::cout << "runtime: " << statistics[0] << " ms" << std::endl;
        std::cout << "bg error: " << statistics[1] << " %" << std::endl;
        std::cout << "v0 error: " << statistics[2] << " m/s" << std::endl;
        std::cout << "g0 error: " << statistics[3] << " deg" << std::endl;
        std::cout << "RRE: " << statistics[4] << " deg" << std::endl;
        std::cout << "RTE: " << statistics[5] << " m" << std::endl;
    }

    bool success = false;
    Eigen::Vector3d bg = Eigen::Vector3d::Zero();
    Eigen::Vector3d v0;
    Eigen::Vector3d g0;
    std::vector<Eigen::Vector3d> delta_t;
    std::vector<Eigen::Matrix3d> delta_R;
    double runtime;

    Eigen::Matrix<double, 6, 1> statistics;// runtime, bg, v0, g0, rot, trans
};


class InitSolver
{
public:
    InitSolver(std::shared_ptr<CameraModel> &_camera0,
               std::shared_ptr<CameraModel> &_camera1,
               int _pyr_levels);
    ~InitSolver(){}

    bool SolveDSInit(const std::vector<std::array<cv::Mat, 2>>& image_seq,
                     const std::vector<std::shared_ptr<ImuIntegration>> &preint_seq,
                     initRes& res);

    bool SolveTwoFrame(const std::array<cv::Mat, 2>& image_i,
                       const std::array<cv::Mat, 2>& image_j,
                       const std::shared_ptr<ImuIntegration> &preint_ij,
                       initRes& res);



private:

    void BuildPyramid(const cv::Mat& img, std::vector<cv::Mat>& img_pyr);
    void GradPyramidImg(const std::vector<cv::Mat>& img_pyr, std::vector<cv::Mat>& img_dx_pyr, std::vector<cv::Mat>& img_dy_pyr, int ddepth);
    std::vector<cv::Point2f> StereoMatch(const std::array<cv::Mat, 2>& image, const std::vector<cv::Point2f>& points,
                                         std::vector<float>& depths, std::vector<uchar>& status);

    int pyr_levels;
    int w[MAX_PYR_LEVELS], h[MAX_PYR_LEVELS];

    float fx, fy, cx, cy;
    float fxi, fyi, cxi, cyi;

    float fx2, fy2, cx2, cy2;
    float fxi2, fyi2, cxi2, cyi2;

    int winsize;

    std::vector<std::vector<Eigen::Vector3f>> pattern_add;
    float huber;
    Eigen::Matrix3f Rbcl, Rbcr, Rclcr;
    Eigen::Vector3f tbcl, tbcr, tclcr;

    std::shared_ptr<CameraModel> camera0, camera1;

};

