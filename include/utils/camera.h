/**
 * @file camera.h
 * @author junyin qiu (j.qiu.1@research.gla.ac.uk)
 * @brief image undistortion by creating virtual camera
 * @version 1.0
 * @date 2026-04-19
 *
 * @copyright Copyright (c) 2026
 */

#pragma once

#include <Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>

class VirtualCamera {
public:
    VirtualCamera(){}

    Eigen::Vector2f pixel2norm(const Eigen::Vector2f &pixel)
    {
        return Eigen::Vector2f(fxi*pixel(0) + cxi, fyi*pixel(1) + cyi);
    }
    Eigen::Vector3f pixel2hnorm(const Eigen::Vector2f &pixel)
    {
        return pixel2norm(pixel).homogeneous();
    }
    Eigen::Vector2f norm2pixel(const Eigen::Vector2f &norm)
    {
        return Eigen::Vector2f(fx*norm(0) + cx, fy*norm(1) + cy);
    }
    Eigen::Vector2f mappoint2pixel(const Eigen::Vector3f &mappoint)
    {
        return norm2pixel(mappoint.hnormalized());
    }

    int w, h;
    float fx, fy, cx, cy;
    float fxi, fyi, cxi, cyi;
};

class CameraModel {
public:

    CameraModel(){}
    ~CameraModel()
    {
        if(remapX != nullptr) {delete[] remapX; remapX = nullptr;}
        if(remapY != nullptr) {delete[] remapY; remapY = nullptr;}
        if(remap_w00 != nullptr) {delete[] remap_w00; remap_w00 = nullptr;}
        if(remap_w01 != nullptr) {delete[] remap_w01; remap_w01 = nullptr;}
        if(remap_w10 != nullptr) {delete[] remap_w10; remap_w10 = nullptr;}
        if(remap_w11 != nullptr) {delete[] remap_w11; remap_w11 = nullptr;}
    }
    bool readParameters(const std::string filename, const std::string cam);
    void CreateVirtualCamera();
    void ResizeVirtualCamera(const int new_w, const int new_h);
    void CreateVirtualRemap();

    cv::Mat UndistortedImage(const cv::Mat &origin_img);

    void distortpoint(const Eigen::Vector2d& p, Eigen::Vector2d& dp) const;
    void backprojectSymmetric(const Eigen::Vector2d& p_u, double& theta, double& phi) const;
    Eigen::Vector2d pixel2norm(const Eigen::Vector2d &pixel, bool ignore_dist = false);
    Eigen::Vector3d pixel2hnorm(const Eigen::Vector2d &pixel, bool ignore_dist = false);
    Eigen::Vector2d norm2pixel(const Eigen::Vector2d &norm, bool ignore_dist = false);
    Eigen::Vector2d mappoint2pixel(const Eigen::Vector3d &mappoint, bool ignore_dist = false);

    Eigen::Matrix3d Ric;
    Eigen::Vector3d tic;
    double focal_length = 400;
    double FOV_top, FOV_bottom, FOV_left, FOV_right;
    VirtualCamera virtual_camera;
    bool virtual_camera_is_built = false;

    int w, h;

    // projection: pinhole
    double fx, fy, cx, cy;
    double fxi, fyi, cxi, cyi;

    enum DistortionModel
    {
        Radtan = 0,
        Equidistant = 1, // also named Kannala_Brandt or fisheye
        None = 2 // distortion-free
    };
    DistortionModel distortion_model;
    // distortion
    // common
    double k1, k2;
    // radtan
    double p1, p2;
    // equidistant/Kannala_Brandt/fisheye
    double k3, k4;

    short* remapX = nullptr;
    short* remapY = nullptr;
    float* remap_w00 = nullptr;
    float* remap_w01 = nullptr;
    float* remap_w10 = nullptr;
    float* remap_w11 = nullptr;
};
