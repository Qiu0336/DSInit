/**
 * @file main.cpp
 * @author junyin qiu (j.qiu.1@research.gla.ac.uk)
 * @brief main function
 * @version 1.0
 * @date 2026-04-19
 *
 * @copyright Copyright (c) 2026
 */

#include <iostream>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include "image_processor.h"
#include "imu_processor.h"
#include "gt_loader.h"
#include "init_solver.h"

std::shared_ptr<CameraModel> g_camera[2];
Eigen::Vector3d gravity_vector(0, 0, -9.81);

int main(int argc, char **argv)
{
    if(argc != 2)
    {
        std::cerr << "invalid config path!!!" << std::endl;
        return -1;
    }
    
    std::string config_file = argv[1];
    cv::FileStorage fs(config_file, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cerr << "config_file is not exist!!!" << std::endl;
        return -1;
    }

    std::string dataset = std::string(fs["dataset"]);
    std::string dataset_path = std::string(fs["dataset_path"]);
    int init_frame_num = fs["init_frame_num"];
    int pyramid_num = fs["pyramid_num"];

    g_camera[0] = std::make_shared<CameraModel>();
    g_camera[1] = std::make_shared<CameraModel>();
    g_camera[0]->readParameters(config_file, "cam0");
    g_camera[1]->readParameters(config_file, "cam1");
    g_camera[0]->CreateVirtualCamera();
    g_camera[1]->CreateVirtualCamera();

    if(g_camera[0]->virtual_camera.w != g_camera[1]->virtual_camera.w
       || g_camera[0]->virtual_camera.h != g_camera[1]->virtual_camera.h)
    {
        int min_w = std::min(g_camera[0]->virtual_camera.w, g_camera[1]->virtual_camera.w);
        int min_h = std::min(g_camera[0]->virtual_camera.h, g_camera[1]->virtual_camera.h);
        if(min_w != g_camera[0]->virtual_camera.w || min_h != g_camera[0]->virtual_camera.h)
        {
            g_camera[0]->ResizeVirtualCamera(min_w, min_h);
        }
        if(min_w != g_camera[1]->virtual_camera.w || min_h != g_camera[1]->virtual_camera.h)
        {
            g_camera[1]->ResizeVirtualCamera(min_w, min_h);
        }
    }
    g_camera[0]->CreateVirtualRemap();
    g_camera[1]->CreateVirtualRemap();

    std::shared_ptr<ImageProcessor> image_processor = std::make_shared<ImageProcessor>(dataset, dataset_path, g_camera[0], g_camera[1]);
    std::shared_ptr<ImuProcessor> imu_processor = std::make_shared<ImuProcessor>(dataset, dataset_path);
    std::shared_ptr<GtLoader> gt_data = std::make_shared<GtLoader>(dataset, dataset_path);

    std::shared_ptr<InitSolver> init_solver = std::make_shared<InitSolver>(g_camera[0], g_camera[1], pyramid_num);
    bool first_img = true;

    std::vector<std::uint64_t> time_seq;
    std::vector<std::array<cv::Mat, 2>> images_seq;
    std::vector<std::shared_ptr<ImuIntegration>> imu_preints;

    std::vector<initRes> results;

    int total_cnt = 0;

    for(int img_id = 0; img_id < image_processor->image0_data_size; ++img_id)
    {
        std::uint64_t t_img = image_processor->GetTimestamp(img_id);
        if(t_img < 0)
        {
            std::cerr << "t_img < 0" << std::endl;
            continue;
        }
        cv::Mat img0, img1;
        if(!image_processor->GetStereoImage(img_id, img0, img1))
        {
            std::cerr << "!image_processor->GetImage(img_id, img)" << std::endl;
            continue;
        }

        if(first_img)
        {
            time_seq.push_back(t_img);
            images_seq.push_back(std::array<cv::Mat, 2>{img0, img1});
            first_img = false;
            continue;
        }

        std::shared_ptr<ImuIntegration> preint_scratch;
        preint_scratch = imu_processor->GetImuIntegration(time_seq[0], t_img);

        if(!preint_scratch)
        {
            time_seq.clear();
            images_seq.clear();
            imu_preints.clear();
            first_img = true;
            continue;
        }

        time_seq.push_back(t_img);
        images_seq.push_back(std::array<cv::Mat, 2>{img0, img1});
        imu_preints.push_back(preint_scratch);

        if(init_frame_num == static_cast<int>(time_seq.size()))
        {
            initRes res;
            init_solver->SolveDSInit(images_seq, imu_preints, res);
            state_t gt_state0;
            std::vector<Eigen::Isometry3d> gt_dT_seq(time_seq.size() - 1);
            bool get_gt = true;
            {
                state_t gt_state;
                std::vector<Eigen::Isometry3d> gt_T_seq(time_seq.size());
                for(size_t i = 0; i < time_seq.size(); ++i)
                {
                    if(!gt_data->GetGtDataAtTimestamp(time_seq[i], gt_state))
                    {
                        get_gt = false;
                        break;
                    }
                    gt_T_seq[i] = gt_state.T();
                    if(i == 0)
                        gt_state0 = gt_state;
                }
                if(get_gt)
                {
                    for(size_t i = 0; i < gt_dT_seq.size(); ++i)
                    {
                        gt_dT_seq[i] = gt_T_seq[i].inverse()*gt_T_seq[i+1];
                    }
                }
            }
            if(get_gt)
            {
                Eigen::Vector3d gt_bg = gt_state0.bg;
                Eigen::Vector3d gt_v0 = gt_state0.q.inverse()*gt_state0.v;
                Eigen::Vector3d gt_g0 = gt_state0.q.inverse()*gravity_vector;
                res.eval(gt_bg, gt_v0, gt_g0, gt_dT_seq);
                res.print();
                results.push_back(res);
                ++total_cnt;
            }
            else
            {
                std::cout << "get gt data failed" << std::endl;
            }

            time_seq.clear();
            images_seq.clear();
            imu_preints.clear();
            first_img = true;
        }
    }

    int success_cnt = 0;
    Eigen::Matrix<double, 6, 1> eva = Eigen::Matrix<double, 6, 1>::Zero();
    for(int i = 0; i < total_cnt; ++i)
    {
        if(results[i].success)
        {
            ++success_cnt;
            eva += results[i].statistics;
        }
    }
    eva /= static_cast<double>(success_cnt);
    std::cout << "============evaluation result===============" << std::endl;
    std::cout << "dataset: " << dataset << std::endl;
    std::cout << "dataset_path: " << dataset_path << std::endl;
    std::cout << "init_frame_num: " << init_frame_num << std::endl;
    std::cout << "pyramid_num: " << pyramid_num << std::endl;
    std::cout << "runtime: " << eva[0] << " ms" << std::endl;
    std::cout << "bg error: " << eva[1] << " %" << std::endl;
    std::cout << "v0 error: " << eva[2] << " m/s" << std::endl;
    std::cout << "g0 error: " << eva[3] << " deg" << std::endl;
    std::cout << "RRE: " << eva[4] << " deg" << std::endl;
    std::cout << "RTE: " << eva[5] << " m" << std::endl;
    std::cout << "success rate: " << static_cast<double>(success_cnt)*100 / total_cnt << " %" << std::endl;
    return 0;
}
