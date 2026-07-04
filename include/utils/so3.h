/**
 * @file so3.h
 * @author junyin qiu (j.qiu.1@research.gla.ac.uk)
 * @brief SO3 functions
 * @version 1.0
 * @date 2026-04-19
 *
 * @copyright Copyright (c) 2026
 */

#pragma once

#include <Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <vector>
#include <Eigen/SVD>

Eigen::Matrix3f SkewFloat(const Eigen::Vector3f &w);
Eigen::Matrix3d Skew(const Eigen::Vector3d &w);
Eigen::Matrix3f ExpSO3Float(const Eigen::Vector3f &w);
Eigen::Matrix3d ExpSO3(const Eigen::Vector3d &w);
Eigen::Vector3f LogSO3Float(const Eigen::Matrix3f &R);
Eigen::Vector3d LogSO3(const Eigen::Matrix3d &R);
Eigen::Matrix3f LeftJacobianSO3Float(const Eigen::Vector3f &w);
Eigen::Matrix3d LeftJacobianSO3(const Eigen::Vector3d &w);
inline Eigen::Matrix3f RightJacobianSO3Float(const Eigen::Vector3f &w) { return LeftJacobianSO3Float(-w); }
inline Eigen::Matrix3d RightJacobianSO3(const Eigen::Vector3d &w) { return LeftJacobianSO3(-w); }
