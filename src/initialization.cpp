// This file is part of HSO: Hybrid Sparse Monocular Visual Odometry 
// With Online Photometric Calibration
//
// Copyright(c) 2021, Dongting Luo, Dalian University of Technology, Dalian
// Copyright(c) 2021, Robotics Group, Dalian University of Technology
//
// This program is highly based on the previous implementation 
// of SVO: https://github.com/uzh-rpg/rpg_svo
// and PL-SVO: https://github.com/rubengooj/pl-svo
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.



#include <hso/config.h>
#include <hso/frame.h>
#include <hso/point.h>
#include <hso/feature.h>
#include <hso/initialization.h>
#include <hso/feature_detection.h>

#include "hso/vikit/homography.h"
#include "hso/vikit/math_utils.h"

namespace hso {
namespace initialization {

// 处理第一帧
// 如果提取到的点太少，则返回failure，某则返回success
InitResult KltHomographyInit::addFirstFrame(FramePtr frame_ref) {

/*  初始化第一帧提取到的特征
  && frame_ref_*/
  reset();

  detectFeatures(frame_ref, px_ref_, f_ref_, ftr_type_);

  if (px_ref_.size() < 200) {
    HSO_WARN_STREAM_THROTTLE(2.0, "First image has less than 100(200?) features. Retry in more textured environment.");
    return FAILURE;
  }

  // HSO_INFO_STREAM("Init: First Frame_px_ref_: " << px_ref_.size() << " features");
  HSO_INFO_STREAM("Init: First Frame_f_ref_: " << f_ref_.size() << " features");
  frame_ref_ = frame_ref;
  px_cur_.insert(px_cur_.begin(), px_ref_.begin(), px_ref_.end());

  // HSO_INFO_STREAM("px_cur_size: "<<px_cur_.size() << "|| px_ref_size: "<<px_ref_.size());
  img_prev_ = frame_ref_->img_pyr_[0].clone();
  px_prev_ = px_ref_;

  return SUCCESS;
}

InitResult KltHomographyInit::addSecondFrame(FramePtr frame_cur) {
  // detectCandidate(frame_cur, px_ref_, f_ref_, ftr_type_, false);

  /*
  Adaptive selection of image alignment methods(采用两种klt跟踪方法，能够适用更多的场景，该方法仅仅用于初始化部分)

  frame_ref_:参考帧->初始化的第一帧
  frame_cur：当前帧->初始化的第二帧
  初始化的时候，px_prev_和px_cur_应该是一样的
  px_ref_:第一帧提取到的特征
  px_cur_:第一帧提取到的特征
  f_ref_:第一帧提取特征的归一化3D坐标
  f_cur_：第一帧提取特征的归一化3D坐标

  disparities_:初始化第一帧和第二帧之间的视差
  img_prev_：第一帧第0层图像金字塔（原始图）图像
  ftr_type_：第一帧提取到的特征的类型
  */

  // HSO_INFO_STREAM("px_cur_size: "<<px_cur_.size() << "|| px_ref_size: "<<px_ref_.size());
  trackKlt(frame_ref_, frame_cur, px_ref_, px_cur_, f_ref_, f_cur_,
           disparities_, img_prev_, px_prev_, ftr_type_);
  // trackCandidate(frame_ref_, frame_cur, px_ref_, px_cur_, f_ref_, f_cur_, disparities_, img_prev_, px_prev_, ftr_type_);

  HSO_INFO_STREAM("Init: KLT tracked " << disparities_.size() << " features");

  // 初始化跟踪到的特征点数量太小则宣布跟踪失败（这里设置的阈值为50）
  if (disparities_.size() < Config::initMinTracked())
    return FAILURE;

  double disparity = hso::getMedian(disparities_);
  // 初始化跟踪到的特征点的视差太小则返回（这里设置的视差阈值是40，即要保证初始化两帧具有足够的基线长度）
  HSO_INFO_STREAM("Init: KLT " << disparity << "px average disparity.");
  if (disparity < Config::initMinDisparity())
    return NO_KEYFRAME;

  // 比较单应矩阵 和 本质矩阵 的内点数（得分系数）来选择初始化方式
  computeInitializeMatrix(
      f_ref_, f_cur_, frame_ref_->cam_->errorMultiplier2(),
      Config::poseOptimThresh(), inliers_, xyz_in_cur_, T_cur_from_ref_);
  // ReconstructEH(
  //     frame_ref_, frame_cur, f_ref_, f_cur_, frame_ref_->cam_->errorMultiplier2(),
  //     Config::poseOptimThresh(), inliers_, xyz_in_cur_, T_cur_from_ref_);


  // 初始化化后的内点数小于40，则初始化失败，返回 failure
  if (inliers_.size() < Config::initMinInliers()) {
    HSO_WARN_STREAM("Init WARNING: " << Config::initMinInliers() << " inliers minimum required.");
    return FAILURE;
  }

  // Rescale the map such that the mean scene depth is equal to the specified scale
  // 重新缩放地图，使平均场景深度等于指定比例
  vector<double> depth_vec;
  for (size_t i = 0; i < xyz_in_cur_.size(); ++i)
    depth_vec.push_back((xyz_in_cur_[i]).z());
  double scene_depth_median = hso::getMedian(depth_vec);

  // Config::mapScale()设置的为1，这样得到的scale就是场景中值的逆深度
  double scale = Config::mapScale() / scene_depth_median;

  // 根据本质矩阵或单应矩阵的结果，得到当前帧在世界坐标系下的刚体变换
  // 这里作者采用对初始两帧的平移向量t归一化作为后续的单位，从而解决单目slam\单目vo的尺度问题
  frame_cur->T_f_w_ = T_cur_from_ref_ * frame_ref_->T_f_w_;
  frame_cur->T_f_w_.translation() =
      -frame_cur->T_f_w_.rotation_matrix() * (frame_ref_->pos() + scale * (frame_cur->pos() - frame_ref_->pos()));

  // For each inlier create 3D point and add feature in both frames
  SE3 T_world_cur = frame_cur->T_f_w_.inverse();

  // inliers_存储的是内点特征点的id
  for (vector<int>::iterator it = inliers_.begin(); it != inliers_.end(); ++it) {
    Vector2d px_cur(px_cur_[*it].x, px_cur_[*it].y);
    Vector2d px_ref(px_ref_[*it].x, px_ref_[*it].y);
    Vector3d fts_type(ftr_type_[*it][0], ftr_type_[*it][1], ftr_type_[*it][2]);

    // int id = *it;
    // Vector2d px_cur(vFinalPxCur_[id].x, vFinalPxCur_[id].y);
    // Vector2d px_ref(vFinalPxRef_[id].x, vFinalPxRef_[id].y);
    // int type = vFinalType_[id];

    // 选择初始化两帧内点特征点在图像有效范围内 且 三角化正确的点
    if (frame_ref_->cam_->isInFrame(px_cur.cast<int>(), 10) && frame_ref_->cam_->isInFrame(px_ref.cast<int>(), 10)
        && xyz_in_cur_[*it].z() > 0) {
      // 将当前帧坐标系下的地图点转换为世界坐标系下，并对采用逆场景深度中值初始化地图点。
      Vector3d pos = T_world_cur * (xyz_in_cur_[*it] * scale);
      Point *new_point = new Point(pos);

      // idist_为地图点模长的倒数
      new_point->idist_ = 1.0 / pos.norm();

      /*
      ftr_type_[i][2]==0 -> 角点特征(CORNER)
      ftr_type_[i][2]==1 -> 边缘特征(EDGELET)
      ftr_type_[i][2]==2 -> GRADIENT
      */
      if (fts_type[2] == 0) {
        new_point->ftr_type_ = Point::FEATURE_CORNER;
        Feature *ftr_cur(new Feature(frame_cur.get(), new_point, px_cur, f_cur_[*it], 0));
        frame_cur->addFeature(ftr_cur);

        Feature *ftr_ref(new Feature(frame_ref_.get(), new_point, px_ref, f_ref_[*it], 0));
        frame_ref_->addFeature(ftr_ref);

        // 对该地图点添加2D观测特征（首先观测到该地图点的帧下的特征）
        new_point->addFrameRef(ftr_ref);
        // new_point->addFrameRef(ftr_cur);
        // 对该地图点添加宿主特征
        new_point->hostFeature_ = ftr_ref;
      } else if (fts_type[2] == 1) {
        // TODO:初始化的时候没有提取边特征，所以fts_type[2] == 1的部分并没有执行
        new_point->ftr_type_ = Point::FEATURE_EDGELET;

        Vector2d grad(fts_type[0], fts_type[1]);
        Feature *ftr_cur(new Feature(frame_cur.get(), new_point, px_cur, f_cur_[*it], grad, 0));
        frame_cur->addFeature(ftr_cur);

        Feature *ftr_ref(new Feature(frame_ref_.get(), new_point, px_ref, f_ref_[*it], grad, 0));
        frame_ref_->addFeature(ftr_ref);

        new_point->addFrameRef(ftr_ref);
        // new_point->addFrameRef(ftr_cur);
        new_point->hostFeature_ = ftr_ref;
      } else {
        new_point->ftr_type_ = Point::FEATURE_GRADIENT;

        Feature *ftr_cur(new Feature(frame_cur.get(), new_point, px_cur, 0, Feature::GRADIENT));
        frame_cur->addFeature(ftr_cur);

        Feature *ftr_ref(new Feature(frame_ref_.get(), new_point, px_ref, 0, Feature::GRADIENT));
        frame_ref_->addFeature(ftr_ref);

        new_point->addFrameRef(ftr_ref);
        // new_point->addFrameRef(ftr_cur);
        new_point->hostFeature_ = ftr_ref;
      }
    }
  }

  return SUCCESS;
}

void KltHomographyInit::reset() {
  //当前帧跟踪到的特征点清空，参考关键帧被释放
  px_cur_.clear();
  frame_ref_.reset(); //当智能指针调用了reset函数的时候，就不会再指向这个对象了，即该指针已经被释放了
}

void detectFeatures(FramePtr frame, vector<cv::Point2f> &px_vec, vector<Vector3d> &f_vec, vector<Vector3d> &ftr_type) {

  Features new_features;

  feature_detection::FeatureExtractor *featureExt(
      new feature_detection::FeatureExtractor(frame->img().cols, frame->img().rows, 20, 1, true));
  featureExt->detect(frame.get(), 20, frame->gradMean_ + 0.5f, new_features);

  // now for all maximum corners, initialize a new seed
  px_vec.clear();
  px_vec.reserve(new_features.size());
  f_vec.clear();
  f_vec.reserve(new_features.size());
  Vector3d fts_type_temp;

  std::for_each(new_features.begin(), new_features.end(), [&](Feature *ftr) {
    if (ftr->type == Feature::EDGELET) {
      fts_type_temp[0] = ftr->grad[0];
      fts_type_temp[1] = ftr->grad[1];
      fts_type_temp[2] = 1;
    } else if (ftr->type == Feature::CORNER) {
      fts_type_temp[0] = ftr->grad[0];
      fts_type_temp[1] = ftr->grad[1];
      fts_type_temp[2] = 0;
    } else {
      fts_type_temp[0] = ftr->grad[0];
      fts_type_temp[1] = ftr->grad[1];
      fts_type_temp[2] = 2;
    }

    ftr_type.push_back(fts_type_temp);
    px_vec.push_back(cv::Point2f(ftr->px[0], ftr->px[1]));
    f_vec.push_back(ftr->f);

    delete ftr;
  });

  delete featureExt;
}

/*

frame_ref_:参考帧->初始化的第一帧
frame_cur：当前帧->初始化的第二帧
初始化的时候，px_prev_和px_cur_应该是一样的
px_ref_:第一帧提取到的特征
px_cur_:第一帧提取到的特征
f_ref_:第一帧提取特征的归一化3D坐标
f_cur_：当前帧（第二帧）下的特征的归一化3D坐标

disparities_:初始化第一帧和第二帧之间的视差
img_prev_：第一帧第0层图像金字塔（原始图）图像
ftr_type_：第一帧提取到的特征的类型
*/
void trackKlt(FramePtr frame_ref,
              FramePtr frame_cur,
              vector<cv::Point2f> &px_ref,
              vector<cv::Point2f> &px_cur,
              vector<Vector3d> &f_ref,
              vector<Vector3d> &f_cur,
              vector<double> &disparities,
              cv::Mat &img_prev,
              vector<cv::Point2f> &px_prev,
              vector<Vector3d> &fts_type) {
  const double klt_win_size = 30.0; // 每层金字塔的搜索窗口
  const int klt_max_iter = 30; // 最大迭代次数
  const double klt_eps = 0.0001; // 期望精度
  vector<unsigned char> status;
  vector<float> error;

  // 指定迭代搜索算法的终止准则，通过cv::TermCriteria设置。此处要同时满足最大迭代次数和期望精度的要求
  cv::TermCriteria termcrit(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, klt_max_iter, klt_eps);

  // cv::OPTFLOW_USE_INITIAL_FLOW：使用上帧速度作为光流速度。因为：通常在序列视频中，相邻帧运动方向具有相似性，故可以使用上帧速度场作为初始猜测。
  cv::calcOpticalFlowPyrLK(
      img_prev, frame_cur->img_pyr_[0], px_prev, px_cur, status, error,
      cv::Size2i(klt_win_size, klt_win_size), 4, termcrit, cv::OPTFLOW_USE_INITIAL_FLOW);

/*  std::vector<int> status; status.resize(px_prev.size(), 1);
  if(img_prev.cols > 1000 && img_prev.rows > 1000)
  {
      // TUM 在线光度标定里面的
      GainRobustTracker tracker(4,5);
      tracker.trackImagePyramids(img_prev, frame_cur->img_pyr_[0], px_prev, px_cur, status);
  }
  else
  {
      GainRobustTracker tracker(4,4);
      tracker.trackImagePyramids(img_prev, frame_cur->img_pyr_[0], px_prev, px_cur, status);
  }*/


  vector<cv::Point2f>::iterator px_ref_it = px_ref.begin();
  vector<cv::Point2f>::iterator px_cur_it = px_cur.begin();
  vector<cv::Point2f>::iterator px_pre_it = px_prev.begin();

  vector<Vector3d>::iterator f_ref_it = f_ref.begin();
  vector<Vector3d>::iterator fts_type_it = fts_type.begin();

  // 清空当前帧提取到的特征的归一化3D坐标
  f_cur.clear();
  f_cur.reserve(px_cur.size());

  // 清空视差容器
  disparities.clear();
  disparities.reserve(px_cur.size());

  // 遍历参考帧(上一帧(初始化第一帧))下提取到的特征
  for (size_t i = 0; px_ref_it != px_ref.end(); ++i) {
    // 如果光流跟踪失败 或 该特征点在最底层图像中太靠近边缘->无法创建patch/residual pattern，那么就将该特征点剔除
    if (!status[i] || !patchCheck(img_prev, frame_cur->img_pyr_[0], *px_pre_it, *px_cur_it)) {
      px_ref_it = px_ref.erase(px_ref_it);
      px_cur_it = px_cur.erase(px_cur_it);
      f_ref_it = f_ref.erase(f_ref_it);
      fts_type_it = fts_type.erase(fts_type_it);

      px_pre_it = px_prev.erase(px_pre_it);

      continue;
    }

    // 当前帧（第二帧）下的特征的归一化3D坐标
    f_cur.push_back(frame_cur->c2f(px_cur_it->x, px_cur_it->y));
    disparities.push_back(Vector2d(px_ref_it->x - px_cur_it->x, px_ref_it->y - px_cur_it->y).norm());

    ++px_ref_it;
    ++px_cur_it;
    ++f_ref_it;
    ++fts_type_it;

    ++px_pre_it;
  }

  // 保存当前帧的信息，即当前帧特征 变成 上一帧特征
  img_prev = frame_cur->img_pyr_[0].clone();
  px_prev = px_cur;

}

void computeInitializeMatrix(
    const vector<Vector3d> &f_ref,
    const vector<Vector3d> &f_cur,
    double focal_length,
    double reprojection_threshold,
    vector<int> &inliers,
    vector<Vector3d> &xyz_in_cur,
    SE3 &T_cur_from_ref) {
  vector<cv::Point2f> x1(f_ref.size()), x2(f_cur.size());
  for (size_t i = 0, i_max = f_ref.size(); i < i_max; ++i) {
    x1[i] = cv::Point2f(f_ref[i][0] / f_ref[i][2], f_ref[i][1] / f_ref[i][2]);
    x2[i] = cv::Point2f(f_cur[i][0] / f_cur[i][2], f_cur[i][1] / f_cur[i][2]);
  }
  const cv::Point2d pp(0, 0);
  const double focal = 1.0;
  cv::Mat E = findEssentialMat(x1, x2, focal, pp, cv::RANSAC, 0.99, 2.0 / focal_length, cv::noArray());

  cv::Mat R_cf, t_cf;
  cv::recoverPose(E, x1, x2, R_cf, t_cf, focal, pp);
  Vector3d t;
  Matrix3d R;
  R << R_cf.at<double>(0, 0), R_cf.at<double>(0, 1), R_cf.at<double>(0, 2),
      R_cf.at<double>(1, 0), R_cf.at<double>(1, 1), R_cf.at<double>(1, 2),
      R_cf.at<double>(2, 0), R_cf.at<double>(2, 1), R_cf.at<double>(2, 2);
  t << t_cf.at<double>(0), t_cf.at<double>(1), t_cf.at<double>(2);

  vector<int> outliers_E, inliers_E;
  vector<Vector3d> xyz_E;
  // double E_error = hso::computeInliers(
  //     f_cur, f_ref, R, t, reprojection_threshold, focal_length, xyz_E, inliers_E, outliers_E);
  double E_error = computeP3D(
      f_cur, f_ref, R, t, reprojection_threshold, focal_length, xyz_E, inliers_E);
  SE3 T_E = SE3(R, t);

  vector<Vector2d> uv_ref(f_ref.size());
  vector<Vector2d> uv_cur(f_cur.size());
  for (size_t i = 0, i_max = f_ref.size(); i < i_max; ++i) {
    uv_ref[i] = hso::project2d(f_ref[i]);
    uv_cur[i] = hso::project2d(f_cur[i]);
  }
  hso::Homography Homography(uv_ref, uv_cur, focal_length, reprojection_threshold);
  Homography.computeSE3fromMatches();

  vector<int> outliers_H, inliers_H;
  vector<Vector3d> xyz_H;
  // double H_error = hso::computeInliers(
  //     f_cur, f_ref, Homography.T_c2_from_c1.rotation_matrix(), Homography.T_c2_from_c1.translation(),
  //     reprojection_threshold, focal_length, xyz_H, inliers_H, outliers_H);
  double H_error = computeP3D(
      f_cur, f_ref, Homography.T_c2_from_c1.rotation_matrix(), Homography.T_c2_from_c1.translation(),
      reprojection_threshold, focal_length, xyz_H, inliers_H);

  HSO_INFO_STREAM("Init: H error =  " << H_error);
  HSO_INFO_STREAM("Init: E error =  " << E_error);

  // if(H_error / (H_error + E_error) < 0.6)
  if (H_error < E_error) {
    inliers = inliers_H;
    xyz_in_cur = xyz_H;
    T_cur_from_ref = Homography.T_c2_from_c1;

    HSO_INFO_STREAM("Init: Homography RANSAC " << inliers.size() << " inliers.");
  } else {
    inliers = inliers_E;
    xyz_in_cur = xyz_E;
    T_cur_from_ref = T_E;

    HSO_INFO_STREAM("Init: Essential RANSAC " << inliers.size() << " inliers.");
  }
}

double computeP3D(
    const vector<Vector3d> &vBearing1,   //cur
    const vector<Vector3d> &vBearing2,   //ref
    const Matrix3d &R,
    const Vector3d &t,
    const double reproj_thresh,
    double error_multiplier2,
    vector<Vector3d> &vP3D,
    vector<int> &inliers) {
  inliers.clear();
  inliers.reserve(vBearing1.size());
  vP3D.clear();
  vP3D.reserve(vBearing1.size());

  SE3 T_c_r = SE3(R, t);
  SE3 T_r_c = T_c_r.inverse();
  double totalEnergy = 0;
  for (size_t i = 0; i < vBearing1.size(); ++i) {
    //first: triangulate TODO
    Vector3d p3d_cur_old(hso::triangulateFeatureNonLin(R, t, vBearing1[i], vBearing2[i]));
    Vector3d p3d_ref_old(T_r_c * p3d_cur_old);
    Vector3d pWorld_new(distancePointOnce(p3d_ref_old, vBearing2[i], vBearing1[i], T_c_r));
    Vector3d pTarget_new(T_c_r * pWorld_new);

    double e1 = hso::reprojError(vBearing1[i], pTarget_new, error_multiplier2);
    // double e2 = hso::reprojError(vBearing2[i], pWorld_new, error_multiplier2);


    totalEnergy += e1;
    vP3D.push_back(pTarget_new);

    if (pWorld_new[2] < 0.01 || pTarget_new[2] < 0.01) continue;

    float ratio = p3d_ref_old.norm() / pWorld_new.norm();
    if (ratio < 0.9 || ratio > 1.1) continue;

    if (e1 < reproj_thresh)
      inliers.push_back(i);
  }

  return totalEnergy;
}

// #define POINT_OPTIMIZER_DEBUG
Vector3d distancePointOnce(
    const Vector3d pointW, Vector3d bearingRef, Vector3d bearingCur, SE3 T_c_r) {
  //create new point
  double idist_old = 1. / pointW.norm();
  double idist_new = idist_old;
  Vector3d pHost(bearingRef * (1.0 / idist_old));

  double oldEnergy = 0;
  double H = 0, b = 0;
  for (int iter = 0; iter < 3; ++iter) {
    double newEnergy = 0;
    H = 0;
    b = 0;

    Vector3d pTarget(T_c_r * pHost);
    Vector2d e(hso::project2d(bearingCur) - hso::project2d(pTarget));
    newEnergy += e.squaredNorm();

    Vector2d Juvdd;
    Point::jacobian_id2uv(pTarget, T_c_r, idist_new, bearingRef, Juvdd);
    H += Juvdd.transpose() * Juvdd;
    b -= Juvdd.transpose() * e;

    double step = (1.0 / H) * b;
    if ((iter > 0 && newEnergy > oldEnergy) || (bool) std::isnan(step)) {
#ifdef POINT_OPTIMIZER_DEBUG
      cout << "it " << iter << "\t FAILURE \t new_chi2 = " << newEnergy << endl;
#endif

      idist_new = idist_old; // roll-back
      break;
    }

    idist_old = idist_new;
    idist_new += step;
    oldEnergy = newEnergy;
    pHost = Vector3d(bearingRef * (1.0 / idist_new));

#ifdef POINT_OPTIMIZER_DEBUG
    cout << "it " << iter
    << "\t Success \t new_chi2 = " << newEnergy
    << "\t idist step = " << step
    << endl;
#endif

    if (step <= 0.000001 * idist_new) break;
  }

  return Vector3d(bearingRef * (1.0 / idist_new));
}

bool patchCheck(
    const cv::Mat &imgPre, const cv::Mat &imgCur, const cv::Point2f &pxPre, const cv::Point2f &pxCur) {
  const int patchArea = 64;

  float patch_pre[patchArea], patch_cur[patchArea];
  if (!createPatch(imgPre, pxPre, patch_pre) || !createPatch(imgCur, pxCur, patch_cur)) return false;

  return checkSSD(patch_pre, patch_cur);
}

// img:输入图像
// px：输入图像中特征点坐标
// patch：residual pattern？
bool createPatch(const cv::Mat &img, const cv::Point2f &px, float *patch) {
  const int halfPatchSize = 4;
  const int patchSize = halfPatchSize * 2;
  const int stride = img.cols;

  float u = px.x;
  float v = px.y;
  int ui = floorf(u); // 向下取整
  int vi = floorf(v); // 向下取整

  // 如果特征点坐标太靠近图像边界（4个像素以内），那么就无法围绕该特征点创建一个pattern，则返回false
  if (ui < halfPatchSize || ui >= img.cols - halfPatchSize || vi < halfPatchSize || vi >= img.rows - halfPatchSize)
    return false;

  // 双线性插值
  float subpix_u = u - ui;
  float subpix_v = v - vi;
  float w_ref_tl = (1.0 - subpix_u) * (1.0 - subpix_v);
  float w_ref_tr = subpix_u * (1.0 - subpix_v);
  float w_ref_bl = (1.0 - subpix_u) * subpix_v;
  float w_ref_br = subpix_u * subpix_v;

  float *patch_ptr = patch;

  // patch的行循环
  for (int y = 0; y < patchSize; ++y) {
    uint8_t *cur_patch_ptr = img.data + (vi - halfPatchSize + y) * stride + (ui - halfPatchSize); // patch的整数级像素指针
    for (int x = 0; x < patchSize; ++x, ++patch_ptr, ++cur_patch_ptr) {
      *patch_ptr = w_ref_tl * cur_patch_ptr[0] + w_ref_tr * cur_patch_ptr[1] + w_ref_bl * cur_patch_ptr[stride]
          + w_ref_br * cur_patch_ptr[stride + 1];  // 插值得到
    }
  }

  return true;
}

bool checkSSD(float *patch1, float *patch2) {
  const int patchArea = 64;

  // const float threshold = 2000*patchArea;
  // float sumA=0, sumB=0, sumAA=0, sumBB=0, sumAB=0;
  // for(int r = 0; r < patchArea; r++)
  // {
  //     float pixel1 = patch1[r], pixel2 = patch2[r];

  //     sumA += pixel1;
  //     sumB += pixel2;
  //     sumAA += pixel1*pixel1;
  //     sumBB += pixel2*pixel2;
  //     sumAB += pixel1*pixel2;
  // }
  // return (sumAA - 2*sumAB + sumBB - (sumA*sumA - 2*sumA*sumB + sumB*sumB)/patchArea) < threshold;

  const float threshold = 0.8f;

  float mean1 = 0, mean2 = 0;
  for (int i = 0; i < patchArea; ++i) {
    mean1 += patch1[i];
    mean2 += patch2[i];
  }

  mean1 /= patchArea;
  mean2 /= patchArea;

  float numerator = 0, demoniator1 = 0, demoniator2 = 0;
  for (int i = 0; i < patchArea; i++) {
    numerator += (patch1[i] - mean1) * (patch2[i] - mean2); // 协方差
    demoniator1 += (patch1[i] - mean1) * (patch1[i] - mean1); // 方差
    demoniator2 += (patch2[i] - mean2) * (patch2[i] - mean2); // 方差
  }

  // 相关系数
  // 检查被当前帧跟踪到的特征点 和 上一帧中的该点是否具有相关性
  return numerator / (sqrt(demoniator1 * demoniator2) + 1e-12) > threshold;
}

} // namespace initialization
} // namespace hso
