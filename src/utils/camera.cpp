/**
 * @file camera.cpp
 * @author junyin qiu (j.qiu.1@research.gla.ac.uk)
 * @brief image undistortion by creating virtual camera
 * @version 1.0
 * @date 2026-04-19
 *
 * @copyright Copyright (c) 2026
 */

#include "utils/camera.h"

bool CameraModel::readParameters(const std::string filename, const std::string cam)
{
    cv::FileStorage fs(filename, cv::FileStorage::READ);

    if (!fs.isOpened())
    {
        return false;
    }

    cv::FileNode n = fs[cam];

    std::vector<double> distortion_coeffs;
    std::string distortion_model_str;
    std::vector<double> intrinsics;
    std::vector<int> resolution;
    n["distortion_coeffs"] >> distortion_coeffs;
    n["distortion_model"] >> distortion_model_str;
    n["resolution"] >> resolution;
    n["intrinsics"] >> intrinsics;
    w = resolution[0];
    h = resolution[1];
    fx = intrinsics[0];
    fy = intrinsics[1];
    cx = intrinsics[2];
    cy = intrinsics[3];

    fxi = 1.0/fx;
    fyi = 1.0/fy;
    cxi = -cx/fx;
    cyi = -cy/fy;

    if(distortion_model_str == "radtan")
    {
        distortion_model = Radtan;
        k1 = distortion_coeffs[0];
        k2 = distortion_coeffs[1];
        p1 = distortion_coeffs[2];
        p2 = distortion_coeffs[3];
    }
    else if(distortion_model_str == "equidistant")
    {
        distortion_model = Equidistant;
        k1 = distortion_coeffs[0];
        k2 = distortion_coeffs[1];
        k3 = distortion_coeffs[2];
        k4 = distortion_coeffs[3];
    }
    else if(distortion_model_str == "none")
    {
        distortion_model = None;
        k1 = 0;
        k2 = 0;
        k3 = 0;
        k4 = 0;
    }
    else
    {
        std::cerr << "No distortion_model !" << std::endl;
    }

    for(const auto &item : n)
    {
        if(item.name() == "T_cam_imu")
        {
            Eigen::Matrix3d Rci;
            Eigen::Vector3d tci;
            for(int r = 0; r < 3; ++r)
            {
                Rci(r, 0) = double(item[r][0]);
                Rci(r, 1) = double(item[r][1]);
                Rci(r, 2) = double(item[r][2]);
                tci(r) = double(item[r][3]);
            }
            Ric = Rci.transpose();
            tic = - Rci.transpose()*tci;
            break;
        }
        else if(item.name() == "T_imu_cam")
        {
            for(int r = 0; r < 3; ++r)
            {
                Ric(r, 0) = double(item[r][0]);
                Ric(r, 1) = double(item[r][1]);
                Ric(r, 2) = double(item[r][2]);
                tic(r) = double(item[r][3]);
            }
            break;
        }
    }

    Eigen::Quaterniond qic(Ric);
    qic.normalize();
    Ric = qic.toRotationMatrix();

    virtual_camera_is_built = false;

    if(1)
    {
        std::cout << "resolution: " << w << ", " << h << std::endl;
        std::cout << "intrinsics: " << fx << ", " << fy << ", " << cx << ", " << cy << std::endl;
        std::cout << "intrinsics_inv: " << fxi << ", " << fyi << ", " << cxi << ", " << cyi << std::endl;
        if(distortion_model == Radtan)
            std::cout << "distortion_coeffs Radtan: " << k1 << ", " << k2 << ", " << p1 << ", " << p2 << std::endl;
        else if(distortion_model == Equidistant)
            std::cout << "distortion_coeffs Equidistant: " << k1 << ", " << k2 << ", " << k3 << ", " << k4 << std::endl;
        else if(distortion_model == None)
            std::cout << "distortion free" << std::endl;
        std::cout << "Ric:\n" << Ric << std::endl;
        std::cout << "tic: " << tic.transpose() << std::endl;
    }

    return true;
}

void CameraModel::distortpoint(const Eigen::Vector2d& p, Eigen::Vector2d& dp) const
{
    if(distortion_model == Radtan)
    {
      const double x2 = p(0)*p(0);
      const double y2 = p(1)*p(1);
      const double _2xy = 2.0*p(0)*p(1);
      const double r2 = x2 + y2;
      const double u = 1 + k1*r2 + k2*r2*r2;
      dp << p(0)*u + p1*_2xy + p2*(r2 + 2.0*x2),
            p(1)*u + p2*_2xy + p1*(r2 + 2.0*y2);
    }
    else if(distortion_model == Equidistant)
    {
        const double r = p.norm();
        if(r < 1e-8)
            dp = p;
        else
        {
            const double theta = atanf(r);
            const double theta2 = theta*theta;
            const double theta4 = theta2*theta2;
            const double dtheta = theta*(1.0 + k1*theta2
                                      + k2*theta4
                                      + k3*theta4*theta2
                                      + k4*theta4*theta4);
            const double scaling = dtheta / r;
            dp = scaling*p;
        }
    }
    else if(distortion_model == None)
    {
        dp = p;
    }
}

Eigen::Vector2d CameraModel::pixel2norm(const Eigen::Vector2d &pixel, bool ignore_dist)
{
    Eigen::Vector2d p_d;
    p_d << fxi*pixel(0) + cxi, fyi*pixel(1) + cyi;
    Eigen::Vector2d p_u;
    if(ignore_dist || distortion_model == None)
    {
        p_u = p_d;
    }
    else
    {
        if(distortion_model == Radtan)
        {
            // Recursive distortion model
            Eigen::Vector2d d_p;
            distortpoint(p_d, d_p);
            p_u = p_d - (d_p - p_d);
            for (int i = 1; i < 8; ++i)// Recursive 8 times
            {
                distortpoint(p_u, d_p);
                p_u += (p_d - d_p);
            }
        }
        else if(distortion_model == Equidistant)
        {
            float theta_d = p_d.norm();
            if(theta_d < 1e-8)
                p_u = p_d;
            else
            {
                if(theta_d > M_PI_2) theta_d = M_PI_2;
                float theta = theta_d;
                for(int j = 0; j < 10; ++j)
                {
                    float theta2 = theta*theta, theta4 = theta2*theta2;
                    float k1_theta2 = k1*theta2, k2_theta4 = k2*theta4, k3_theta6 = k3*theta4*theta2, k4_theta8 = k4*theta4*theta4;
                    float theta_fix = (theta * (1 + k1_theta2 + k2_theta4 + k3_theta6 + k4_theta8) - theta_d) /
                                       (1 + 3*k1_theta2 + 5*k2_theta4 + 7*k3_theta6 + 9*k4_theta8);
                    theta -= theta_fix;
                    if(fabs(theta_fix) < 1e-8)
                        break;
                }
                float scale = tan(theta) / theta_d;
                p_u = scale*p_d;
            }
        }
        else if(distortion_model == None)
        {
            p_u = p_d;
        }
    }
    return p_u;
}
Eigen::Vector3d CameraModel::pixel2hnorm(const Eigen::Vector2d &pixel, bool ignore_dist)
{
    return pixel2norm(pixel, ignore_dist).homogeneous();
}

Eigen::Vector2d CameraModel::norm2pixel(const Eigen::Vector2d &norm, bool ignore_dist)
{
    Eigen::Vector2d pixel;
    Eigen::Vector2d p_d;
    if(ignore_dist || distortion_model == None)
    {
        p_d = norm;
    }
    else
    {
        distortpoint(norm, p_d);
    }

    pixel << fx*p_d(0) + cx, fy*p_d(1) + cy;
    return pixel;
}

Eigen::Vector2d CameraModel::mappoint2pixel(const Eigen::Vector3d &mappoint, bool ignore_dist)
{
    return norm2pixel(mappoint.hnormalized(), ignore_dist);
}


void CameraModel::CreateVirtualCamera()
{
    double top_inner = -10000;
    double bottom_inner = 10000;
    double left_inner = -10000;
    double right_inner = 10000;
    for(int x = w>>2, xend = w-(w>>2); x < xend; ++x)
    {
        Eigen::Vector2d p1 = pixel2norm(Eigen::Vector2d(x, 0));
        Eigen::Vector2d p2 = pixel2norm(Eigen::Vector2d(x, h-1));

        if(top_inner < p1.y()) top_inner = p1.y();
        if(bottom_inner > p2.y()) bottom_inner = p2.y();
    }
    for(int y = h>>2, yend = h-(h>>2); y < yend; ++y)
    {
        Eigen::Vector2d p1 = pixel2norm(Eigen::Vector2d(0, y));
        Eigen::Vector2d p2 = pixel2norm(Eigen::Vector2d(w-1, y));
        if(left_inner < p1.x()) left_inner = p1.x();
        if(right_inner > p2.x()) right_inner = p2.x();
    }

    FOV_top = atan(top_inner)*180.0/M_PI;
    FOV_bottom = atan(bottom_inner)*180.0/M_PI;
    FOV_left = atan(left_inner)*180.0/M_PI;
    FOV_right = atan(right_inner)*180.0/M_PI;

    if(FOV_top < -45)
    {
        FOV_top = -45;
        top_inner = tan(FOV_top*M_PI/180.0);
    }
    if(FOV_bottom > 45)
    {
        FOV_bottom = 45;
        bottom_inner = tan(FOV_bottom*M_PI/180.0);
    }
    if(FOV_left < -45)
    {
        FOV_left = -45;
        left_inner = tan(FOV_left*M_PI/180.0);
    }
    if(FOV_right > 45)
    {
        FOV_right = 45;
        right_inner = tan(FOV_right*M_PI/180.0);
    }


    Eigen::Vector2d lt_inner = focal_length*Eigen::Vector2d(left_inner, top_inner);
    Eigen::Vector2d rb_inner = focal_length*Eigen::Vector2d(right_inner, bottom_inner);

    double ww_f = rb_inner.x()-lt_inner.x();
    double hh_f = rb_inner.y()-lt_inner.y();
    int ww_i = (int(ww_f) >> 4) << 4;// max pyramid level = 5
    int hh_i = (int(hh_f) >> 4) << 4;
    double ww_r = ww_f - ww_i;
    double hh_r = hh_f - hh_i;

    virtual_camera.w = ww_i;
    virtual_camera.h = hh_i;
    virtual_camera.fx = float(focal_length);
    virtual_camera.fy = float(focal_length);
    virtual_camera.cx = float( - (lt_inner.x() + 0.5*ww_r));
    virtual_camera.cy = float( - (lt_inner.y() + 0.5*hh_r));

    virtual_camera.fxi = 1.f/virtual_camera.fx;
    virtual_camera.fyi = 1.f/virtual_camera.fy;
    virtual_camera.cxi = -virtual_camera.cx/virtual_camera.fx;
    virtual_camera.cyi = -virtual_camera.cy/virtual_camera.fy;
    virtual_camera_is_built = true;

    if(1)
    {
        std::cout << "Cam FOV:\n";
        std::cout << "top~bottom: " << FOV_top << " ~ " << FOV_bottom << std::endl;
        std::cout << "left~right: " << FOV_left << " ~ " << FOV_right << std::endl;
        std::cout << "virtual_camera.w: " << virtual_camera.w << std::endl;
        std::cout << "virtual_camera.h: " << virtual_camera.h << std::endl;
        std::cout << "virtual_camera.fx: " << virtual_camera.fx << std::endl;
        std::cout << "virtual_camera.fy: " << virtual_camera.fy << std::endl;
        std::cout << "virtual_camera.cx: " << virtual_camera.cx << std::endl;
        std::cout << "virtual_camera.cy: " << virtual_camera.cy << std::endl;
    }
}


void CameraModel::ResizeVirtualCamera(const int new_w, const int new_h)
{
    if(!virtual_camera_is_built)
    {
        std::cerr << "No Virtual Camera to Resize!" << std::endl;
        return;
    }
    int dw = new_w - virtual_camera.w;
    int dh = new_h - virtual_camera.h;
    virtual_camera.w = new_w;
    virtual_camera.h = new_h;
    virtual_camera.cx += 0.5f*dw;
    virtual_camera.cy += 0.5f*dh;
    virtual_camera.cxi = -virtual_camera.cx/virtual_camera.fx;
    virtual_camera.cyi = -virtual_camera.cy/virtual_camera.fy;
}

void CameraModel::CreateVirtualRemap()
{
    if(!virtual_camera_is_built)
    {
        std::cerr << "No Virtual Camera for Virtual Remap Creation!" << std::endl;
        return;
    }
    if(remapX != nullptr) {delete[] remapX; remapX = nullptr;}
    if(remapY != nullptr) {delete[] remapY; remapY = nullptr;}
    if(remap_w00 != nullptr) {delete[] remap_w00; remap_w00 = nullptr;}
    if(remap_w01 != nullptr) {delete[] remap_w01; remap_w01 = nullptr;}
    if(remap_w10 != nullptr) {delete[] remap_w10; remap_w10 = nullptr;}
    if(remap_w11 != nullptr) {delete[] remap_w11; remap_w11 = nullptr;}

    const int wh = virtual_camera.w*virtual_camera.h;
    remapX = new short[wh];
    remapY = new short[wh];
    remap_w00 = new float[wh];
    remap_w01 = new float[wh];
    remap_w10 = new float[wh];
    remap_w11 = new float[wh];
    int idx = 0;
    for(int y = 0; y < virtual_camera.h; ++y)
    {
        for(int x = 0; x < virtual_camera.w; ++x)
        {
            Eigen::Vector2f np = virtual_camera.pixel2norm(Eigen::Vector2f(x, y));
            Eigen::Vector2f uv = norm2pixel(np.cast<double>()).cast<float>();
            float u = uv.x();
            float v = uv.y();
            short iu = short(u);
            short iv = short(v);
            float a = u - iu;
            float b = v - iv;
            float w11 = a*b;
            float w01 = a - w11;
            float w10 = b - w11;
            float w00 = 1 - w11 - w01 - w10;
            remapX[idx] = iu;
            remapY[idx] = iv;
            remap_w00[idx] = w00;
            remap_w01[idx] = w01;
            remap_w10[idx] = w10;
            remap_w11[idx] = w11;
            ++idx;
        }
    }
}

cv::Mat CameraModel::UndistortedImage(const cv::Mat &origin_img)
{
    cv::Mat virtual_image = cv::Mat(virtual_camera.h, virtual_camera.w, CV_8UC1, cv::Scalar(0));
    const int step = int(origin_img.step/origin_img.elemSize1());
    uchar* dst = virtual_image.data;
    for(int i = 0, iend = virtual_camera.h*virtual_camera.w; i < iend; ++i)
    {
        short iu = remapX[i];
        short iv = remapY[i];
        float w11 = remap_w11[i];
        float w01 = remap_w01[i];
        float w10 = remap_w10[i];
        float w00 = remap_w00[i];
        const uchar* src = origin_img.ptr(iv) + iu;
        uchar val = uchar(src[0]*w00 + src[1]*w01 + src[step]*w10 + src[step+1]*w11);
        dst[i] = val;
    }
    return virtual_image;
}

