/**
 * @file so3.cpp
 * @author junyin qiu (j.qiu.1@research.gla.ac.uk)
 * @brief SO3 functions
 * @version 1.0
 * @date 2026-04-19
 *
 * @copyright Copyright (c) 2026
 */

#include "utils/so3.h"

Eigen::Matrix3f SkewFloat(const Eigen::Vector3f &w)
{
    Eigen::Matrix3f W;
    W << 0.0, -w.z(), w.y(), w.z(), 0.0, -w.x(), -w.y(),  w.x(), 0.0;
    return W;
}

Eigen::Matrix3d Skew(const Eigen::Vector3d &w)
{
    Eigen::Matrix3d W;
    W << 0.0, -w.z(), w.y(), w.z(), 0.0, -w.x(), -w.y(),  w.x(), 0.0;
    return W;
}

Eigen::Matrix3f ExpSO3Float(const Eigen::Vector3f &w)
{
    const float theta = w.norm();
    const float theta2 = theta*theta;
    Eigen::Matrix3f W = SkewFloat(w);
    if(theta < 1e-4f)
        return Eigen::Matrix3f::Identity() + W + 0.5f*W*W;
    else
        return Eigen::Matrix3f::Identity() + W*sinf(theta)/theta + W*W*(1.f-cosf(theta))/theta2;
}

Eigen::Matrix3d ExpSO3(const Eigen::Vector3d &w)
{
    const double theta = w.norm();
    const double theta2 = theta*theta;
    Eigen::Matrix3d W = Skew(w);
    if(theta < 1e-4)
        return Eigen::Matrix3d::Identity() + W + 0.5*W*W;
    else
        return Eigen::Matrix3d::Identity() + W*sin(theta)/theta + W*W*(1.0-cos(theta))/theta2;
}

Eigen::Vector3f LogSO3Float(const Eigen::Matrix3f &R)
{
    float costheta = 0.5f*(R.trace()-1.f);
    if(costheta > +1.f) costheta = +1.f;
    if(costheta < -1.f) costheta = -1.f;
    const float theta = acos(costheta);
    const Eigen::Vector3f w(R(2,1)-R(1,2), R(0,2)-R(2,0), R(1,0)-R(0,1));
    if(theta < 1e-4f)
        return 0.5f*w;
    else
        return 0.5f*theta*w/sin(theta);
}

Eigen::Vector3d LogSO3(const Eigen::Matrix3d &R)
{
    double costheta = 0.5*(R.trace()-1.0);
    if(costheta > +1.0) costheta = +1.0;
    if(costheta < -1.0) costheta = -1.0;
    const double theta = acos(costheta);
    const Eigen::Vector3d w(R(2,1)-R(1,2), R(0,2)-R(2,0), R(1,0)-R(0,1));
    if(theta < 1e-4)
        return 0.5*w;
    else
        return 0.5*theta*w/sin(theta);
}

Eigen::Matrix3f LeftJacobianSO3Float(const Eigen::Vector3f &w)
{
    const float theta = w.norm();
    const float theta2 = theta*theta;
    Eigen::Matrix3f W = SkewFloat(w);
    if(theta < 1e-4f)
        return Eigen::Matrix3f::Identity() + 0.5f*W + W*W/6.f;
    else
        return Eigen::Matrix3f::Identity() + W*(1.f-cos(theta))/theta2 + W*W*(theta-sin(theta))/(theta2*theta);
}

Eigen::Matrix3d LeftJacobianSO3(const Eigen::Vector3d &w)
{
    const double theta = w.norm();
    const double theta2 = theta*theta;
    Eigen::Matrix3d W = Skew(w);
    if(theta < 1e-4)
        return Eigen::Matrix3d::Identity() + 0.5*W + W*W/6.0;
    else
        return Eigen::Matrix3d::Identity() + W*(1.0-cos(theta))/theta2 + W*W*(theta-sin(theta))/(theta2*theta);
}

