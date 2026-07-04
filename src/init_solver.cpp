/**
 * @file init_solver.cpp
 * @author junyin qiu (j.qiu.1@research.gla.ac.uk)
 * @brief direct sparse initialization for stereo VIO
 * @version 1.0
 * @date 2026-04-19
 *
 * @copyright Copyright (c) 2026
 */

#include "init_solver.h"

template<typename T1, typename T2>
void ReduceVector(std::vector<T1> &v, std::vector<T2> s)
{
    int j = 0;
    for(int i = 0, iend = int(v.size()); i < iend; ++i)
        if(s[i])
            v[j++] = v[i];
    v.resize(j);
}

InitSolver::InitSolver(std::shared_ptr<CameraModel> &_camera0,
                       std::shared_ptr<CameraModel> &_camera1,
                       int _pyr_levels)
{
    camera0 = _camera0;
    camera1 = _camera1;
    w[0] = camera0->virtual_camera.w;
    h[0] = camera0->virtual_camera.h;
    pyr_levels = _pyr_levels;
    if(pyr_levels > MAX_PYR_LEVELS)
    {
        pyr_levels = MAX_PYR_LEVELS;
    }

    for(int lvl = 1; lvl < pyr_levels; ++lvl)
    {
        if(w[lvl-1]%2 != 0 || h[lvl-1]%2 != 0)
        {
            pyr_levels = lvl;
            break;
        }
        w[lvl] = w[lvl-1]/2;
        h[lvl] = h[lvl-1]/2;
    }
    winsize = 2*patternPadding + 1;

    Rbcl = camera0->Ric.cast<float>();
    tbcl = camera0->tic.cast<float>();
    Rbcr = camera1->Ric.cast<float>();
    tbcr = camera1->tic.cast<float>();

    Rclcr = Rbcl.transpose()*Rbcr;
    tclcr = Rbcl.transpose()*(tbcr - tbcl);

    fx = camera0->virtual_camera.fx;
    fy = camera0->virtual_camera.fy;
    cx = camera0->virtual_camera.cx;
    cy = camera0->virtual_camera.cy;
    fxi = camera0->virtual_camera.fxi;
    fyi = camera0->virtual_camera.fyi;
    cxi = camera0->virtual_camera.cxi;
    cyi = camera0->virtual_camera.cyi;

    fx2 = camera1->virtual_camera.fx;
    fy2 = camera1->virtual_camera.fy;
    cx2 = camera1->virtual_camera.cx;
    cy2 = camera1->virtual_camera.cy;
    fxi2 = camera1->virtual_camera.fxi;
    fyi2 = camera1->virtual_camera.fyi;
    cxi2 = camera1->virtual_camera.cxi;
    cyi2 = camera1->virtual_camera.cyi;

    Eigen::Matrix3f K1i;
    K1i << fxi, 0, cxi,
            0, fyi, cyi,
            0, 0, 1;
    for(int lvl = 0; lvl < pyr_levels; ++lvl)
    {
        std::vector<Eigen::Vector3f> pattern_add_lvl;
        for(int k = 0; k < patternNum; ++k)
        {
            Eigen::Vector3f add = K1i*Eigen::Vector3f(patternP[k][0] << lvl, patternP[k][1] << lvl, 1);
            pattern_add_lvl.push_back(add);
        }
        pattern_add.push_back(pattern_add_lvl);
    }
    huber = 1;
}

void InitSolver::BuildPyramid(const cv::Mat& img, std::vector<cv::Mat>& img_pyr)
{
    img_pyr.clear();
    cv::Size sz = img.size();
    for(int level = 0; level < pyr_levels; ++level)
    {
        cv::Mat temp;
        temp.create(sz.height + winsize*2, sz.width + winsize*2, img.type());
        if(level == 0)
            cv::copyMakeBorder(img, temp, winsize, winsize, winsize, winsize, cv::BORDER_REFLECT_101);
        else
        {
            cv::Mat thisLevel = temp(cv::Rect(winsize, winsize, sz.width, sz.height));
            cv::pyrDown(img_pyr[level-1], thisLevel, sz);
            cv::copyMakeBorder(thisLevel, temp, winsize, winsize, winsize, winsize, cv::BORDER_REFLECT_101|cv::BORDER_ISOLATED);
        }
        temp.adjustROI(-winsize, -winsize, -winsize, -winsize);
        img_pyr.push_back(temp);
        sz = cv::Size((sz.width+1)/2, (sz.height+1)/2);
    }
}

void InitSolver::GradPyramidImg(const std::vector<cv::Mat>& img_pyr, std::vector<cv::Mat>& img_dx_pyr, std::vector<cv::Mat>& img_dy_pyr, int ddepth)
{
    img_dx_pyr.clear();
    img_dy_pyr.clear();
    for(int i = 0; i < pyr_levels; ++i)
    {
        cv::Mat derivX, derivY;
        cv::Scharr(img_pyr[i], derivX, ddepth, 1, 0);
        cv::Scharr(img_pyr[i], derivY, ddepth, 0, 1);
        cv::copyMakeBorder(derivX, derivX, winsize, winsize, winsize, winsize, cv::BORDER_CONSTANT);
        cv::copyMakeBorder(derivY, derivY, winsize, winsize, winsize, winsize, cv::BORDER_CONSTANT);

        derivX.adjustROI(-winsize, -winsize, -winsize, -winsize);
        derivY.adjustROI(-winsize, -winsize, -winsize, -winsize);

        img_dx_pyr.push_back(derivX);
        img_dy_pyr.push_back(derivY);
    }
}

std::vector<cv::Point2f> InitSolver::StereoMatch(const std::array<cv::Mat, 2>& image, const std::vector<cv::Point2f>& points,
                                                 std::vector<float>& depths, std::vector<uchar>& status)
{
    std::vector<cv::Point2f> points_right;
    status.clear();
    depths.clear();

    if(points.empty()) return points_right;

    std::vector<cv::Point2f> src_pts = points;
    std::vector<float> src_deps;
    std::vector<int> map2idx(points.size());
    std::iota(map2idx.begin(), map2idx.end(), 0);
    std::vector<cv::Point2f> dst_pts;
    std::vector<uchar> sta;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(image[0], image[1], src_pts, dst_pts, sta, err);
    ReduceVector(src_pts, sta);
    ReduceVector(dst_pts, sta);
    ReduceVector(map2idx, sta);
    src_deps.resize(src_pts.size());
    sta.resize(src_pts.size());

    for(size_t i = 0; i < src_pts.size(); ++i)
    {
        Eigen::Vector3f pi(fxi*src_pts[i].x + cxi, fyi*src_pts[i].y + cyi, 1);
        Eigen::Vector3f pj(fxi2*dst_pts[i].x + cxi2, fyi2*dst_pts[i].y + cyi2, 1);
        const Eigen::Vector3f line = SkewFloat(tclcr)*Rclcr*pj;
        const float num = fabs(line.dot(pi)) / line.head<2>().norm();
        if(num > 3.0/camera0->focal_length)
        {
            sta[i] = 0;
            continue;
        }
        Eigen::Vector3f a = SkewFloat(Rclcr*pj)*pi;
        Eigen::Vector3f b = SkewFloat(Rclcr*pj)*tclcr;
        src_deps[i] = (a.dot(b)) / (a.dot(a));
        if(src_deps[i] < 0.1 || src_deps[i] > 100)
            sta[i] = 0;
        else
            sta[i] = 1;
    }
    ReduceVector(src_pts, sta);
    ReduceVector(dst_pts, sta);
    ReduceVector(map2idx, sta);
    ReduceVector(src_deps, sta);

    status.resize(points.size(), 0);
    depths.resize(points.size(), 0);
    points_right.resize(points.size());

    for(size_t i = 0; i < src_deps.size(); ++i)
    {
        int idx = map2idx[i];
        depths[idx] = src_deps[i];
        status[idx] = 1;
        points_right[idx] = dst_pts[i];
    }

    return points_right;
}


bool InitSolver::SolveDSInit(const std::vector<std::array<cv::Mat, 2>>& image_seq,
                             const std::vector<std::shared_ptr<ImuIntegration>> &preint_seq,
                             initRes& res)
{
    const int preint_size = preint_seq.size();
    const int frame_size = image_seq.size();
    if(frame_size == 2)
        return SolveTwoFrame(image_seq[0], image_seq[1], preint_seq[0], res);

    auto time_start = std::chrono::high_resolution_clock::now();

    std::vector<Eigen::Matrix3f> dR_seq;
    std::vector<Eigen::Matrix3f> JRg_seq;
    std::vector<Eigen::Vector3f> dP_seq;
    std::vector<Eigen::Matrix3f> JPg_seq;
    std::vector<float> dt_seq;

    for(int i = 0; i < preint_size; ++i)
    {
        Eigen::Matrix3f dR = preint_seq[i]->Get_dR().cast<float>();
        Eigen::Matrix3f JRg = preint_seq[i]->Get_JRg().cast<float>();
        Eigen::Vector3f dP = preint_seq[i]->Get_dP().cast<float>();
        Eigen::Matrix3f JPg = preint_seq[i]->Get_JPg().cast<float>();
        float dt = preint_seq[i]->Get_dT();
        dR_seq.push_back(dR);
        JRg_seq.push_back(JRg);
        dP_seq.push_back(dP);
        JPg_seq.push_back(JPg);
        dt_seq.push_back(dt);
    }

    std::vector<std::vector<cv::Mat>> imgpyr_seq, imgpyr_dx_seq, imgpyr_dy_seq;
    imgpyr_seq.resize(pyr_levels);
    imgpyr_dx_seq.resize(pyr_levels);
    imgpyr_dy_seq.resize(pyr_levels);
    for(int i = 0; i < frame_size; ++i)
    {
        std::vector<cv::Mat> img_pyr, imgpyr_dx, imgpyr_dy;
        BuildPyramid(image_seq[i][0], img_pyr);
        GradPyramidImg(img_pyr, imgpyr_dx, imgpyr_dy, CV_16S);
        for(int lvl = 0; lvl < pyr_levels; ++lvl)
        {
            imgpyr_seq[lvl].push_back(img_pyr[lvl]);
            imgpyr_dx_seq[lvl].push_back(imgpyr_dx[lvl]);
            imgpyr_dy_seq[lvl].push_back(imgpyr_dy[lvl]);
        }
    }

    std::vector<cv::Point2f> src_pts;
    std::vector<float> src_pts_depths;
    {
        std::vector<uchar> status;
        cv::goodFeaturesToTrack(image_seq[0][0], src_pts, 0, 0.01, 30);
        StereoMatch(image_seq[0], src_pts, src_pts_depths, status);
        ReduceVector(src_pts, status);
        ReduceVector(src_pts_depths, status);
    }

    std::vector<std::vector<std::vector<uchar>>> query_values(pyr_levels);
    std::vector<std::vector<std::vector<Eigen::Vector3f>>> query_Pci(pyr_levels);
    for(int lvl = pyr_levels - 1; lvl >= 0; --lvl)
    {
        int wlvl = w[lvl];
        int hlvl = h[lvl];
        const cv::Mat& img_src = imgpyr_seq[lvl][0];
        const int step = (int)(img_src.step/img_src.elemSize1());
        float is = 1.f / (1 << lvl);
        std::vector<std::vector<uchar>> I_buffer;
        std::vector<std::vector<Eigen::Vector3f>> Pci_buffer;
        for(int pt_id = 0, pt_id_end = src_pts.size(); pt_id < pt_id_end; ++pt_id)
        {
            float x1 = src_pts[pt_id].x;
            float y1 = src_pts[pt_id].y;
            float x1s = x1*is;
            float y1s = y1*is;
            if(int(x1s) - patternPadding < 1 || int(x1s) + patternPadding > wlvl-3
               || int(y1s) - patternPadding < 1 || int(y1s) + patternPadding > hlvl-3)
                continue;

            std::vector<Eigen::Vector3f> Pi_buf;
            std::vector<uchar> I_buf;
            {
                int ix1s = int(x1s);
                int iy1s = int(y1s);
                float a = x1s - ix1s;
                float b = y1s - iy1s;
                float w11 = a*b;
                float w01 = a - w11;
                float w10 = b - w11;
                float w00 = 1 - w11 - w01 - w10;
                float dep = src_pts_depths[pt_id];
                Eigen::Vector3f pi_base = Eigen::Vector3f(fxi*x1, fyi*y1, 0);
                for(int k = 0; k < patternNum; ++k)
                {
                    int ix1sp = ix1s + patternP[k][0];
                    int iy1sp = iy1s + patternP[k][1];
                    const uchar* src = img_src.ptr(iy1sp) + ix1sp;
                    uchar val = uchar(src[0]*w00 + src[1]*w01 + src[step]*w10 + src[step+1]*w11 + 0.5f);
                    I_buf.push_back(val);
                    Eigen::Vector3f pi = pi_base + pattern_add[lvl][k];
                    Pi_buf.push_back(dep*pi);
                }
            }
            I_buffer.push_back(I_buf);
            Pci_buffer.push_back(Pi_buf);
        }
        query_values[lvl] = I_buffer;
        query_Pci[lvl] = Pci_buffer;
    }




    const int para_size = 9 + preint_size*2;
    const int upper_size = (1 + para_size)*para_size / 2;
    Eigen::Vector3f bg_copy = Eigen::Vector3f::Zero();
    Eigen::Vector3f v0_copy = Eigen::Vector3f::Zero();
    Eigen::Vector3f g0_copy = -(preint_seq[0]->Get_dV() / preint_seq[0]->Get_dT()).cast<float>();
    Eigen::VectorXf alpha_copy = Eigen::VectorXf::Constant(preint_size, 1.f);
    Eigen::VectorXf beta_copy = Eigen::VectorXf::Constant(preint_size, 0.f);

    for(int lvl = pyr_levels - 1; lvl >= 0; --lvl)
    {
        int wlvl = w[lvl];
        int hlvl = h[lvl];

        float is = 1.f / (1 << lvl);
        float s_J = 1.f / (1 << (5+lvl));

        const std::vector<cv::Mat>& imglvl_seq = imgpyr_seq[lvl];
        const std::vector<cv::Mat>& imglvl_dx_seq = imgpyr_dx_seq[lvl];
        const std::vector<cv::Mat>& imglvl_dy_seq = imgpyr_dy_seq[lvl];

        const int step = (int)(imglvl_seq[0].step/imglvl_seq[0].elemSize1());
        const int dstep = (int)(imglvl_dx_seq[0].step/imglvl_dx_seq[0].elemSize1());

        float last_ave_cost = 0;

        Eigen::Vector3f bg = bg_copy;
        Eigen::Vector3f v0 = v0_copy;
        Eigen::Vector3f g0 = g0_copy;
        Eigen::VectorXf alpha = alpha_copy;
        Eigen::VectorXf beta = beta_copy;
        std::vector<std::vector<uchar>> valid_queries = query_values[lvl];
        std::vector<std::vector<Eigen::Vector3f>> valid_Pi = query_Pci[lvl];

        for(int iter = 0; iter < 10; ++iter)
        {
            Eigen::MatrixXf H(para_size, para_size);
            Eigen::VectorXf B(para_size);
            H.setZero();
            B.setZero();
            float ave_cost = 0;
            int cct = 0;
            std::vector<Eigen::Matrix3f> Rcjci_seq;
            std::vector<Eigen::Vector3f> tcjci_seq;
            std::vector<Eigen::Matrix3f> JRug_seq;
            std::vector<Eigen::Matrix<float, 3, 9>> JPj_init_seq;
            for(int i = 0; i < preint_size; ++i)
            {
                Eigen::Matrix3f dRu = dR_seq[i]*ExpSO3Float(JRg_seq[i]*bg);
                Eigen::Vector3f term = dRu.transpose()*(tbcl - dP_seq[i] - JPg_seq[i]*bg - v0*dt_seq[i] - 0.5*g0*dt_seq[i]*dt_seq[i]);
                Eigen::Matrix3f Rcjci = Rbcl.transpose()*dRu.transpose()*Rbcl;
                Eigen::Vector3f tcjci = Rbcl.transpose()*(term - tbcl);
                Eigen::Matrix3f JRug = RightJacobianSO3Float(JRg_seq[i]*bg)*JRg_seq[i];
                Eigen::Matrix<float, 3, 9> JPj_init;
                JPj_init.block<3, 3>(0, 0) = Rbcl.transpose()*(SkewFloat(term)*JRug - dRu.transpose()*JPg_seq[i]);
                JPj_init.block<3, 3>(0, 3) = -Rbcl.transpose()*dRu.transpose()*dt_seq[i];
                JPj_init.block<3, 3>(0, 6) = 0.5*dt_seq[i]*JPj_init.block<3, 3>(0, 3);
                Rcjci_seq.push_back(Rcjci);
                tcjci_seq.push_back(tcjci);
                JRug_seq.push_back(JRug);
                JPj_init_seq.push_back(JPj_init);
            }

            {
                std::vector<Eigen::VectorXf> Hf_upper_vector(valid_queries.size());
                std::vector<Eigen::VectorXf> Bf_vector(valid_queries.size());
                std::vector<float> cost_vector(valid_queries.size());
                std::vector<int> cct_vector(valid_queries.size());

                auto optical_flow_alignment = [&](const cv::Range &range)
                {
                    for(int pt_id = range.start; pt_id < range.end; ++pt_id)
                    {
                        Eigen::VectorXf Hf_upper_block(upper_size);
                        Eigen::VectorXf Bf_block(para_size);
                        Hf_upper_block.setZero();
                        Bf_block.setZero();

                        const std::vector<uchar>& I_buf = valid_queries[pt_id];
                        const std::vector<Eigen::Vector3f>& Pi_buf = valid_Pi[pt_id];

                        float patch_cost = 0;
                        int patch_cct = 0;

                        for(int i = 0; i < preint_size; ++i)
                        {
                            for(int k = 0; k < patternNum; ++k)
                            {
                                Eigen::Vector3f Pj = Rcjci_seq[i]*Pi_buf[k] + tcjci_seq[i];
                                const float inv_dep_j = 1.f/Pj.z();
                                float xj_norm = Pj.x()*inv_dep_j;
                                float yj_norm = Pj.y()*inv_dep_j;
                                float xx2s = is*(fx*xj_norm + cx);
                                float yy2s = is*(fy*yj_norm + cy);
                                if(int(xx2s) > 0 && int(xx2s) < (wlvl-2) &&
                                   int(yy2s) > 0 && int(yy2s) < (hlvl-2))
                                {
                                    int ixx2s = int(xx2s);
                                    int iyy2s = int(yy2s);
                                    float a = xx2s - ixx2s;
                                    float b = yy2s - iyy2s;
                                    float w11 = a*b;
                                    float w01 = a - w11;
                                    float w10 = b - w11;
                                    float w00 = 1 - w11 - w01 - w10;

                                    const uchar* dst = imglvl_seq[i+1].ptr(iyy2s) + ixx2s;
                                    float val = dst[0]*w00 + dst[1]*w01 + dst[step]*w10 + dst[step+1]*w11;

                                    const short* ddstX = imglvl_dx_seq[i+1].ptr<short>(iyy2s) + ixx2s;
                                    const short* ddstY = imglvl_dy_seq[i+1].ptr<short>(iyy2s) + ixx2s;
                                    float val_dx = ddstX[0]*w00 + ddstX[1]*w01 + ddstX[dstep]*w10 + ddstX[dstep+1]*w11;
                                    float val_dy = ddstY[0]*w00 + ddstY[1]*w01 + ddstY[dstep]*w10 + ddstY[dstep+1]*w11;

                                    float cost = val - alpha[i]*I_buf[k] + beta[i];
                                    float abs_cost = fabs(cost);
                                    float w_cost = abs_cost > huber? std::sqrt(2*abs_cost*huber-huber*huber) : cost;
                                    float J_wc = abs_cost > huber? (cost > 0? huber/w_cost : -huber/w_cost) : 1.f;
                                    Eigen::Vector3f Jr_Pj;
                                    Jr_Pj << val_dx*fx, val_dy*fy, -val_dx*fx*xj_norm - val_dy*fy*yj_norm;
                                    Jr_Pj = (s_J*inv_dep_j)*Jr_Pj;

                                    Eigen::Matrix<float, 3, 9> JPj_init = JPj_init_seq[i];
                                    JPj_init.block<3, 3>(0, 0) += SkewFloat(Rcjci_seq[i]*Pi_buf[k])*Rbcl.transpose()*JRug_seq[i];
                                    Eigen::VectorXf Jx(para_size);
                                    Jx.setZero();
                                    Jx.head<9>() = Jr_Pj.transpose()*JPj_init;
                                    Jx[9+i] = -I_buf[k];
                                    Jx[9+preint_size+i] = 1.f;
                                    Jx *= J_wc;

                                    if(Jx.array().isInf().any() || Jx.array().isNaN().any())
                                        continue;

                                    int c = 0;
                                    for(int r = 0; r < para_size; ++r)
                                        for(int l = r; l < para_size; ++l)
                                            Hf_upper_block[c++] += Jx(r)*Jx(l);
                                    Bf_block -= w_cost*Jx;
                                    patch_cost += w_cost*w_cost;
                                    ++patch_cct;
                                }
                            }
                        }

                        Hf_upper_vector[pt_id] = Hf_upper_block;
                        Bf_vector[pt_id] = Bf_block;
                        cost_vector[pt_id] = patch_cost;
                        cct_vector[pt_id] = patch_cct;
                    }
                };

                LambdaBody body(optical_flow_alignment);
                cv::parallel_for_(cv::Range(0, valid_queries.size()), body);

                Eigen::VectorXf Hf_upper(upper_size);
                Hf_upper.setZero();
                for(int i = 0, iend = valid_queries.size(); i < iend; ++i)
                {
                    Hf_upper += Hf_upper_vector[i];
                    B += Bf_vector[i];
                    ave_cost += cost_vector[i];
                    cct += cct_vector[i];
                }
                int c = 0;
                for(int r = 0; r < para_size; ++r)
                {
                    for(int l = r; l < para_size; ++l)
                    {
                        H(r, l) = Hf_upper[c++];
                        if(r != l)
                            H(l, r) = H(r, l);
                    }
                }
            }
            ave_cost = ave_cost/cct;

            if(iter > 0 && ave_cost >= last_ave_cost)
                break;

            Eigen::VectorXf dx = H.ldlt().solve(B);

            if(dx.array().isInf().any() || dx.array().isNaN().any())
            {
                std::cout << "dx is Inf or NaN !!!" << std::endl;
                break;
            }

            bg += dx.head<3>();
            v0 += dx.segment<3>(3);
            g0 += dx.segment<3>(6);
            alpha += dx.segment(9, preint_size);
            beta += dx.tail(preint_size);

            float ratio = 1.f;
            if(last_ave_cost > ave_cost)
                ratio = (last_ave_cost - ave_cost) / ave_cost;

            last_ave_cost = ave_cost;

            float dxx = dx.head<9>().norm();

            if(dxx < 1e-3 || ratio < 0.01)
            {
                if(lvl == 0 && ave_cost < 30)
                      res.success = true;
                break;
            }
        }
        bg_copy = bg;
        v0_copy = v0;
        g0_copy = g0;
        alpha_copy = alpha;
        beta_copy = beta;
    }

    res.bg = bg_copy.cast<double>();
    res.v0 = v0_copy.cast<double>();
    res.g0 = g0_copy.cast<double>();

    if(fabs(res.g0.norm() - gravity_vector.norm()) > 3)
        res.success = false;

    {
        Eigen::Matrix<double, 6, 1> bias;
        bias << res.bg, 0, 0, 0;
        for(size_t i = 0; i < preint_seq.size(); ++i)
        {
            preint_seq[i]->SetBias(bias);
            preint_seq[i]->Reintegrated();
        }
        std::vector<Eigen::Matrix3d> Rb0bj_seq;
        std::vector<Eigen::Vector3d> tb0bj_seq;

        for(size_t i = 0; i < preint_seq.size(); ++i)
        {
            double dt = preint_seq[i]->Get_dT();
            Eigen::Matrix3d Rb0bj = preint_seq[i]->Get_dR();
            Eigen::Vector3d tb0bj = res.v0*dt + 0.5*res.g0*dt*dt
                    + preint_seq[i]->Get_dP();
            Rb0bj_seq.push_back(Rb0bj);
            tb0bj_seq.push_back(tb0bj);
        }
        for(size_t i = 0; i < preint_seq.size(); ++i)
        {
            if(i == 0)
            {
                res.delta_R.push_back(Rb0bj_seq[i]);
                res.delta_t.push_back(tb0bj_seq[i]);
                continue;
            }
            res.delta_R.push_back(Rb0bj_seq[i-1].transpose()*Rb0bj_seq[i]);
            res.delta_t.push_back(Rb0bj_seq[i-1].transpose()*
                    (tb0bj_seq[i]-tb0bj_seq[i-1]));
        }
    }

    auto time_end = std::chrono::high_resolution_clock::now();
    res.runtime = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count()*1e-3;

    return res.success;
}

bool InitSolver::SolveTwoFrame(const std::array<cv::Mat, 2>& image_i,
                               const std::array<cv::Mat, 2>& image_j,
                               const std::shared_ptr<ImuIntegration> &preint_ij,
                               initRes& res)
{
    auto time_start = std::chrono::high_resolution_clock::now();

    Eigen::Matrix3f dR_ij = preint_ij->Get_dR().cast<float>();
    Eigen::Matrix3f JRg_ij = preint_ij->Get_JRg().cast<float>();

    std::vector<cv::Mat> img_i_pyr;
    std::vector<cv::Mat> img_j_pyr;
    std::vector<cv::Mat> img_j_dx_pyr;
    std::vector<cv::Mat> img_j_dy_pyr;
    {
        std::vector<cv::Mat> img_pyr, img_pyr_dx, img_pyr_dy;
        BuildPyramid(image_i[0], img_pyr);
        img_i_pyr = img_pyr;
        BuildPyramid(image_j[0], img_pyr);
        GradPyramidImg(img_pyr, img_pyr_dx, img_pyr_dy, CV_16S);
        img_j_pyr = img_pyr;
        img_j_dx_pyr = img_pyr_dx;
        img_j_dy_pyr = img_pyr_dy;
    }

    std::vector<cv::Point2f> src_pts;
    std::vector<float> src_pts_depths;

    {
        std::vector<uchar> status;
        cv::goodFeaturesToTrack(image_i[0], src_pts, 0, 0.01, 30);
        StereoMatch(image_i, src_pts, src_pts_depths, status);
        ReduceVector(src_pts, status);
        ReduceVector(src_pts_depths, status);
    }

    std::vector<std::vector<std::vector<uchar>>> query_values(pyr_levels);
    std::vector<std::vector<std::vector<Eigen::Vector3f>>> query_Pci(pyr_levels);
    for(int lvl = pyr_levels - 1; lvl >= 0; --lvl)
    {
        int wlvl = w[lvl];
        int hlvl = h[lvl];
        const cv::Mat& img_src = img_i_pyr[lvl];
        const int step = (int)(img_src.step/img_src.elemSize1());
        float is = 1.f / (1 << lvl);
        std::vector<std::vector<uchar>> I_buffer;
        std::vector<std::vector<Eigen::Vector3f>> Pci_buffer;
        for(int pt_id = 0, pt_id_end = src_pts.size(); pt_id < pt_id_end; ++pt_id)
        {
            float x1 = src_pts[pt_id].x;
            float y1 = src_pts[pt_id].y;
            float x1s = x1*is;
            float y1s = y1*is;
            if(int(x1s) - patternPadding < 1 || int(x1s) + patternPadding > wlvl-3
               || int(y1s) - patternPadding < 1 || int(y1s) + patternPadding > hlvl-3)
                continue;

            std::vector<Eigen::Vector3f> Pi_buf;
            std::vector<uchar> I_buf;
            {
                int ix1s = int(x1s);
                int iy1s = int(y1s);
                float a = x1s - ix1s;
                float b = y1s - iy1s;
                float w11 = a*b;
                float w01 = a - w11;
                float w10 = b - w11;
                float w00 = 1 - w11 - w01 - w10;
                float dep = src_pts_depths[pt_id];
                Eigen::Vector3f pi_base = Eigen::Vector3f(fxi*x1, fyi*y1, 0);
                for(int k = 0; k < patternNum; ++k)
                {
                    int ix1sp = ix1s + patternP[k][0];
                    int iy1sp = iy1s + patternP[k][1];
                    const uchar* src = img_src.ptr(iy1sp) + ix1sp;
                    uchar val = uchar(src[0]*w00 + src[1]*w01 + src[step]*w10 + src[step+1]*w11 + 0.5f);
                    I_buf.push_back(val);
                    Eigen::Vector3f pi = pi_base + pattern_add[lvl][k];
                    Pi_buf.push_back(dep*pi);
                }
            }
            I_buffer.push_back(I_buf);
            Pci_buffer.push_back(Pi_buf);
        }
        query_values[lvl] = I_buffer;
        query_Pci[lvl] = Pci_buffer;
    }


    Eigen::Vector3f bg_copy = Eigen::Vector3f::Zero();
    Eigen::Vector3f tji_copy = Eigen::Vector3f::Zero();
    float alpha_copy = 1.f;
    float beta_copy = 0.f;
    for(int lvl = pyr_levels - 1; lvl >= 0; --lvl)
    {
        int wlvl = w[lvl];
        int hlvl = h[lvl];
        const cv::Mat& imgj_dst = img_j_pyr[lvl];
        const cv::Mat& imgj_dst_dx = img_j_dx_pyr[lvl];
        const cv::Mat& imgj_dst_dy = img_j_dy_pyr[lvl];
        const int step = (int)(imgj_dst.step/imgj_dst.elemSize1());
        const int dstep = (int)(imgj_dst_dx.step/imgj_dst_dx.elemSize1());

        float is = 1.f / (1 << lvl);
        float s_J = 1.f / (1 << (5+lvl));
        float last_ave_cost = 0;

        Eigen::Vector3f bg = bg_copy;
        Eigen::Vector3f tji = tji_copy;
        float alpha = alpha_copy;
        float beta = beta_copy;

        std::vector<std::vector<uchar>> valid_queries = query_values[lvl];
        std::vector<std::vector<Eigen::Vector3f>> valid_Pi = query_Pci[lvl];

        for(int iter = 0; iter < 10; ++iter)
        {
            Eigen::Matrix<float, 8, 8> H;
            Eigen::Matrix<float, 8, 1> B;
            H.setZero();
            B.setZero();
            float ave_cost = 0;
            int cct = 0;

            Eigen::Matrix3f Rji = Rbcl.transpose()*(dR_ij*ExpSO3Float(JRg_ij*bg)).transpose()*Rbcl;
            Eigen::Matrix3f JRug_ij = RightJacobianSO3Float(JRg_ij*bg)*JRg_ij;

            {
                std::vector<Eigen::Matrix<float, 36, 1>> Hf_upper_vector(valid_queries.size());
                std::vector<Eigen::Matrix<float, 8, 1>> Bf_vector(valid_queries.size());
                std::vector<float> cost_vector(valid_queries.size());
                std::vector<int> cct_vector(valid_queries.size());

                auto optical_flow_alignment = [&](const cv::Range &range)
                {
                    for(int pt_id = range.start; pt_id < range.end; ++pt_id)
                    {
                        Eigen::Matrix<float, 36, 1> Hf_upper_block;
                        Eigen::Matrix<float, 8, 1> Bf_block;
                        Hf_upper_block.setZero();
                        Bf_block.setZero();

                        const std::vector<uchar>& I_buf = valid_queries[pt_id];
                        const std::vector<Eigen::Vector3f>& Pi_buf = valid_Pi[pt_id];

                        float patch_cost = 0;
                        int patch_cct = 0;

                        for(uint k = 0; k < patternNum; ++k)
                        {
                            Eigen::Vector3f Pj = Rji*Pi_buf[k] + tji;
                            const float inv_dep_j = 1.f/Pj.z();
                            float xj_norm = Pj.x()*inv_dep_j;
                            float yj_norm = Pj.y()*inv_dep_j;
                            float xx2s = is*(fx*xj_norm + cx);
                            float yy2s = is*(fy*yj_norm + cy);
                            if(int(xx2s) > 0 && int(xx2s) < (wlvl-2) &&
                               int(yy2s) > 0 && int(yy2s) < (hlvl-2))
                            {
                                int ixx2s = int(xx2s);
                                int iyy2s = int(yy2s);
                                float a = xx2s - ixx2s;
                                float b = yy2s - iyy2s;
                                float w11 = a*b;
                                float w01 = a - w11;
                                float w10 = b - w11;
                                float w00 = 1 - w11 - w01 - w10;
                                const uchar* dst = imgj_dst.ptr(iyy2s) + ixx2s;
                                float val = dst[0]*w00 + dst[1]*w01 + dst[step]*w10 + dst[step+1]*w11;

                                const short* ddstX = imgj_dst_dx.ptr<short>(iyy2s) + ixx2s;
                                const short* ddstY = imgj_dst_dy.ptr<short>(iyy2s) + ixx2s;
                                float val_dx = ddstX[0]*w00 + ddstX[1]*w01 + ddstX[dstep]*w10 + ddstX[dstep+1]*w11;
                                float val_dy = ddstY[0]*w00 + ddstY[1]*w01 + ddstY[dstep]*w10 + ddstY[dstep+1]*w11;

                                float cost = val - alpha*I_buf[k] + beta;
                                float abs_cost = fabs(cost);
                                float w_cost = abs_cost > huber? std::sqrt(2*abs_cost*huber-huber*huber) : cost;
                                float J_wc = abs_cost > huber? (cost > 0? huber/w_cost : -huber/w_cost) : 1.f;

                                Eigen::Vector3f Jr_Pj;
                                Jr_Pj << val_dx*fx, val_dy*fy, -val_dx*fx*xj_norm - val_dy*fy*yj_norm;
                                Jr_Pj = (s_J*inv_dep_j)*Jr_Pj;

                                Eigen::Matrix<float, 6, 1> Jr_pose;
                                Jr_pose.head<3>() = Jr_Pj.transpose()*SkewFloat(Rji*Pi_buf[k])*Rbcl.transpose()*JRug_ij;
                                Jr_pose.tail<3>() = Jr_Pj;
                                Eigen::Matrix<float, 8, 1> Jx;
                                Jx << Jr_pose, -I_buf[k], 1.f;
                                Jx *= J_wc;
                                if(Jx.array().isInf().any() || Jx.array().isNaN().any())
                                    continue;

                                int c = 0;
                                for(int r = 0; r < 8; ++r)
                                    for(int l = r; l < 8; ++l)
                                        Hf_upper_block[c++] += Jx(r)*Jx(l);
                                Bf_block -= w_cost*Jx;

                                patch_cost += w_cost*w_cost;
                                ++patch_cct;
                            }
                        }
                        Hf_upper_vector[pt_id] = Hf_upper_block;
                        Bf_vector[pt_id] = Bf_block;
                        cost_vector[pt_id] = patch_cost;
                        cct_vector[pt_id] = patch_cct;
                    }
                };

                LambdaBody body(optical_flow_alignment);
                cv::parallel_for_(cv::Range(0, valid_queries.size()), body);

                Eigen::Matrix<float, 36, 1> Hf_upper;
                Hf_upper.setZero();
                for(int i = 0, iend = valid_queries.size(); i < iend; ++i)
                {
                    Hf_upper += Hf_upper_vector[i];
                    B += Bf_vector[i];
                    ave_cost += cost_vector[i];
                    cct += cct_vector[i];
                }
                int l = 0;
                for(int i = 0; i < 8; ++i)
                {
                    for(int j = i; j < 8; ++j)
                    {
                        H(i, j) = Hf_upper[l++];
                        if(i != j)
                            H(j, i) = H(i, j);
                    }
                }
            }
            ave_cost = ave_cost/cct;

            if(iter > 0 && ave_cost >= last_ave_cost)
                break;

            Eigen::VectorXf dx = H.ldlt().solve(B);

            if(dx.array().isInf().any() || dx.array().isNaN().any())
            {
                std::cout << "dx is Inf or NaN !!!" << std::endl;
                break;
            }

            bg += dx.head<3>();
            tji += dx.segment<3>(3);
            alpha += dx.tail<2>()[0];
            beta += dx.tail<2>()[1];

            float ratio = 1.f;
            if(last_ave_cost > ave_cost)
                ratio = (last_ave_cost - ave_cost) / ave_cost;

            last_ave_cost = ave_cost;

            float dxx = dx.head<6>().norm();
            if(dxx < 1e-3 || ratio < 0.01)
            {
                if(lvl == 0 && ave_cost < 30)
                      res.success = true;
                break;
            }
        }
        bg_copy = bg;
        tji_copy = tji;
        alpha_copy = alpha;
        beta_copy = beta;
    }

    Eigen::Matrix3f Rbibj = dR_ij*ExpSO3Float(JRg_ij*bg_copy);
    Eigen::Matrix3f Rcicj = Rbcl.transpose()*Rbibj*Rbcl;
    Eigen::Vector3f tcicj = -Rcicj*tji_copy;
    Eigen::Vector3f tbibj = Rbcl*tcicj + tbcl - Rbibj*tbcl;

    res.bg = bg_copy.cast<double>();
    Eigen::Vector3d dV = preint_ij->Get_dV();
    Eigen::Matrix3d JVg = preint_ij->Get_JVg();
    double dt1 = preint_ij->Get_dT();
    res.v0 = tbibj.cast<double>()/dt1;
    res.g0 = - (dV + JVg*res.bg)/dt1;

    if(fabs(res.g0.norm() - gravity_vector.norm()) > 3)
        res.success = false;

    {
        Eigen::Matrix<double, 6, 1> bias;
        bias << res.bg, 0, 0, 0;
        preint_ij->SetBias(bias);
        preint_ij->Reintegrated();
        std::vector<Eigen::Matrix3d> Rc0cj_seq;
        std::vector<Eigen::Vector3d> tc0cj_seq;
        double dt = preint_ij->Get_dT();
        res.delta_R.push_back(preint_ij->Get_dR());
        res.delta_t.push_back(res.v0*dt + 0.5*res.g0*dt*dt
                              + preint_ij->Get_dP());
    }
    auto time_end = std::chrono::high_resolution_clock::now();
    res.runtime = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count()*1e-3;

    return res.success;
}


