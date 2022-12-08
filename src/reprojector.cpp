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


#include <algorithm>
#include <stdexcept>
#include <hso/reprojector.h>
#include <hso/frame.h>
#include <hso/point.h>
#include <hso/feature.h>
#include <hso/map.h>
#include <hso/config.h>
#include <hso/depth_filter.h>
#include <boost/bind.hpp>
#include <boost/thread.hpp>

#include "hso/camera.h"

namespace hso {

Reprojector::Reprojector(hso::AbstractCamera *cam, Map &map) :
    map_(map), sum_seed_(0), sum_temp_(0), nFeatures_(0) {
  initializeGrid(cam);
}

Reprojector::~Reprojector() {
  std::for_each(grid_.cells.begin(), grid_.cells.end(), [&](Cell *c) { delete c; });
  std::for_each(grid_.seeds.begin(), grid_.seeds.end(), [&](Sell *s) { delete s; });
}

inline int Reprojector::caculateGridSize(const int wight, const int height, const int N) {
  return floorf(sqrt(float(wight * height) / N) * 0.6);
}

void Reprojector::initializeGrid(hso::AbstractCamera *cam) {
  // grid_.cell_size = 30;
  // 计算每个小单元格的大小
  grid_.cell_size = caculateGridSize(cam->width(), cam->height(), Config::maxFts());

  // grid_.cell_size_h = floorf(sqrt(cam->height()*cam->height()/Config::maxFts())*0.68);
  // grid_.cell_size_w = floorf(cam->width()/cam->height()*grid_.cell_size_h);

  // 计算图像的行列能够被分为多少个单元格
  // std::ceil是向上取整
  grid_.grid_n_cols = std::ceil(static_cast<double>(cam->width()) / grid_.cell_size);
  grid_.grid_n_rows = std::ceil(static_cast<double>(cam->height()) / grid_.cell_size);
  grid_.cells.resize(grid_.grid_n_cols * grid_.grid_n_rows);

  // 每个单元格只搜索一个种子，即一个地图点，因此种子集合的大小和方格的数量一致
  grid_.seeds.resize(grid_.grid_n_cols * grid_.grid_n_rows);
  std::for_each(grid_.cells.begin(), grid_.cells.end(), [&](Cell *&c) { c = new Cell; });
  std::for_each(grid_.seeds.begin(), grid_.seeds.end(), [&](Sell *&s) { s = new Sell; });

  grid_.cell_order.resize(grid_.cells.size());
  for (size_t i = 0; i < grid_.cells.size(); ++i)
    grid_.cell_order[i] = i;

  // 打乱单元格序号
  std::random_shuffle(grid_.cell_order.begin(), grid_.cell_order.end()); // maybe we should do it at every iteration!
}

void Reprojector::resetGrid() {
  n_matches_ = 0, n_trials_ = 0, n_seeds_ = 0, n_filters_ = 0;
  std::for_each(grid_.cells.begin(), grid_.cells.end(), [&](Cell *c) { c->clear(); });
  std::for_each(grid_.seeds.begin(), grid_.seeds.end(), [&](Sell *s) { s->clear(); });
  std::random_shuffle(grid_.cell_order.begin(), grid_.cell_order.end());
  nFeatures_ = 0;
}

//overlap_kfs:关联上的关键帧
void Reprojector::reprojectMap(
    FramePtr frame, std::vector<std::pair<FramePtr, std::size_t> > &overlap_kfs) {

  // 复位单元格
  resetGrid();

  // 待分配的特征点
  std::vector<pair<Vector2d, Point *> > allPixelToDistribute;


  HSO_START_TIMER("reproject_kfs");

  // 检查临时生成的地图点是否为空
  if (!map_.point_candidates_.temporaryPoints_.empty()) {
    // delete temporary point before.
    DepthFilter::lock_t lock(depth_filter_->seeds_mut_);

    size_t n = 0;
    auto ite = map_.point_candidates_.temporaryPoints_.begin();
    while (ite != map_.point_candidates_.temporaryPoints_.end()) {
      if (ite->first->seedStates_ == 0) {
        ite++;
        continue;
      }

      boost::unique_lock<boost::mutex> lock(map_.point_candidates_.mut_);
      map_.safeDeleteTempPoint(*ite);
      ite = map_.point_candidates_.temporaryPoints_.erase(ite);
      n++;
    }
    sum_seed_ -= n;
  }


  // 为关联上的关键帧预分配空间
  overlap_kfs.reserve(options_.max_n_kfs);

  FramePtr LastFrame = frame->m_last_frame;
  size_t nCovisibilityGraph = 0; // <= 5
  for (vector<Frame *>::iterator it = LastFrame->connectedKeyFrames.begin(); it != LastFrame->connectedKeyFrames.end();
       ++it) {
    Frame *repframe = *it;
    FramePtr repFrame = nullptr;
    if (!map_.getKeyframeById(repframe->id_, repFrame))
      continue;

    // This should not happen
    if (repFrame->lastReprojectFrameId_ == frame->id_)
      continue;
    repFrame->lastReprojectFrameId_ = frame->id_;

    overlap_kfs.push_back(pair<FramePtr, size_t>(repFrame, 0));

    for (auto ite = repFrame->fts_.begin(); ite != repFrame->fts_.end(); ++ite) {
      if ((*ite)->point == NULL)
        continue;

      if ((*ite)->point->type_ == Point::TYPE_TEMPORARY)
        continue;

      if ((*ite)->point->last_projected_kf_id_ == frame->id_)
        continue;

      (*ite)->point->last_projected_kf_id_ = frame->id_;

      if (reprojectPoint(frame, (*ite)->point, allPixelToDistribute)) {
        overlap_kfs.back().second++;
      }
    }

    nCovisibilityGraph++;
  }
  assert(nCovisibilityGraph == LastFrame->connectedKeyFrames.size());
  LastFrame->connectedKeyFrames.clear();

  // Identify those Keyframes which share a common field of view.
  // 找到有重叠的关键帧，返回共享指针和距离 (pair中的double是帧与帧之间的距离关系)
  list<pair<FramePtr, double> > close_kfs;
  map_.getCloseKeyframes(frame, close_kfs); // 那五个特殊点派上了用场

  // Sort KFs with overlap according to their closeness
  // 根据KF与frame的距离对共视的KFs进行排序（从小到大排序）
  close_kfs.sort(boost::bind(&std::pair<FramePtr, double>::second, _1) <
      boost::bind(&std::pair<FramePtr, double>::second, _2));

  // Reproject all mappoints of the closest N kfs with overlap. We only store in which grid cell the points fall.
  size_t n = nCovisibilityGraph;
  // 遍历距离近的所有KFs
  for (auto it_frame = close_kfs.begin(), ite_frame = close_kfs.end(); it_frame != ite_frame && n < options_.max_n_kfs;
       ++it_frame) {

    FramePtr ref_frame = it_frame->first;

    // 防止对同一帧进行重复操作
    if (ref_frame->lastReprojectFrameId_ == frame->id_)
      continue;

    // 对投影过的帧做标记
    ref_frame->lastReprojectFrameId_ = frame->id_;

    //pair（与当前frame有共视部分的KF，共视的特征点的数量）
    overlap_kfs.push_back(pair<FramePtr, size_t>(ref_frame, 0));

    // Try to reproject each mappoint that the other KF observes
    for (auto it_ftr = ref_frame->fts_.begin(), ite_ftr = ref_frame->fts_.end(); it_ftr != ite_ftr; ++it_ftr) {
      // check if the feature has a mappoint assigned

      //拒绝地图点为空
      if ((*it_ftr)->point == nullptr)
        continue;

      // 拒绝临时地图点
      if ((*it_ftr)->point->type_ == Point::TYPE_TEMPORARY)
        continue;

      // 拒绝将要被删除的地图点
      assert((*it_ftr)->point->type_ != Point::TYPE_DELETED);

      // make sure we project a point only once
      // 把投影帧id赋值给投影地图点，并进行投影到图像上，放入单元格中
      if ((*it_ftr)->point->last_projected_kf_id_ == frame->id_) // 防止对同一个地图点进行重复操作
        continue;
      (*it_ftr)->point->last_projected_kf_id_ = frame->id_;
      if (reprojectPoint(frame, (*it_ftr)->point, allPixelToDistribute)) {
        overlap_kfs.back().second++;
      }
    }
    ++n;
  }
  HSO_STOP_TIMER("reproject_kfs");

  //地图候选点和之前的 closekeyframe 不重复么?
  //答: 不重复，候选点是未分配的收敛点，关键帧上是已经收敛的地图点
  //怎么选的候选点?
  //答: 候选点是深度滤波得到的收敛的点
  // Now project all point candidates
  // 投影候选地图点
  HSO_START_TIMER("reproject_candidates");
  {
    boost::unique_lock<boost::mutex> lock(map_.point_candidates_.mut_); // //多线程上锁
    auto it = map_.point_candidates_.candidates_.begin();
    while (it != map_.point_candidates_.candidates_.end()) {
      // assert(it->first->last_projected_kf_id_ != frame->id_);
      // 投影候选点
      if (!reprojectPoint(frame, it->first, allPixelToDistribute)) {
        // 投影失败就增加权重 +3
        it->first->n_failed_reproj_ += 3;
        // 失败超过10次，就删除后面会失败的点
        if (it->first->n_failed_reproj_ > 30) {
          map_.point_candidates_.deleteCandidate(*it);
          it = map_.point_candidates_.candidates_.erase(it);
          continue;
        }
      }
      ++it;
    }
  } // unlock the mutex when out of scope
  HSO_STOP_TIMER("reproject_candidates");

  // Now project all point temp
  // 投影所有的临时地图点
  auto itk = map_.point_candidates_.temporaryPoints_.begin();
  while (itk != map_.point_candidates_.temporaryPoints_.end()) {

    // 拒绝向当前帧投影过临时地图点
    assert(itk->first->last_projected_kf_id_ != frame->id_);

    // 如果临时地图点是坏点则跳过
    if (itk->first->isBad_) {
      itk++;
      continue;
    }

    itk->first->last_projected_kf_id_ = frame->id_;

    //update pos
    Point *tempPoint = itk->first;
    Feature *tempFeature = itk->second;
    tempPoint->pos_ = tempFeature->frame->T_f_w_.inverse() * (tempFeature->f * (1.0 / tempPoint->idist_));

    assert((tempPoint->pos_ - tempFeature->point->pos_).norm() < 0.0001); // TODO: add by wzh

    if (!reprojectPoint(frame, itk->first, allPixelToDistribute)) {
      // 同上
      itk->first->n_failed_reproj_ += 3;
      if (itk->first->n_failed_reproj_ > 30) {
        itk->first->isBad_ = true;
      }
    }
    itk++;
  }


  // Now we go through（遍历） each grid cell and select one point to match. At the end, we should have at maximum one reprojected point per cell.
  // 现在我们遍历每个网格单元并选择一个点进行匹配。最后，每个单元最多应该有一个重投影点。
  HSO_START_TIMER("feature_align");

  // 小于200+25
  if (allPixelToDistribute.size() < Config::maxFts() + 50) {
    reprojectCellAll(allPixelToDistribute, frame);
  } else {
    // 1st
    // 随机选择网格进行对齐，网格中只要有一个特征点匹配成功即可，超过一点数量则匹配成功
    for (size_t i = 0; i < grid_.cells.size(); ++i) {
      // we prefer good quality points over unkown quality (more likely to match)
      // and unknown quality over candidates (position not optimized)
      if (reprojectCell(*grid_.cells.at(grid_.cell_order[i]), frame, false, false)) {
        ++n_matches_;
      }
      if (n_matches_ >= (size_t) Config::maxFts())
        break;
    }

    // 2nd
    if (n_matches_ < (size_t) Config::maxFts()) {
      for (size_t i = grid_.cells.size() - 1; i > 0; --i) {
        if (reprojectCell(*grid_.cells.at(grid_.cell_order[i]), frame, true, false)) {
          ++n_matches_;
          // grid_.cell_occupandy.at(grid_.cell_order[i]) = true;
        }
        if (n_matches_ >= (size_t) Config::maxFts())
          break;
      }
    }

    // 3rd
    if (n_matches_ < (size_t) Config::maxFts()) {
      for (size_t i = 0; i < grid_.cells.size(); ++i) {
        reprojectCell(*grid_.cells.at(grid_.cell_order[i]), frame, true, true);

        if (n_matches_ >= (size_t) Config::maxFts())
          break;
      }
    }
  }

  // reproject seed
  if (n_matches_ < 100 && options_.reproject_unconverged_seeds) {
    DepthFilter::lock_t lock(depth_filter_->seeds_mut_);
    for (auto it = depth_filter_->seeds_.begin(); it != depth_filter_->seeds_.end(); ++it) {
      if (sqrt(it->sigma2) < it->z_range / options_.reproject_seed_thresh && !it->haveReprojected)
        reprojectorSeed(frame, *it, it);
    }

    for (size_t i = 0; i < grid_.seeds.size(); ++i) {
      if (reprojectorSeeds(*grid_.seeds.at(grid_.cell_order[i]), frame)) {
        ++n_matches_;
        // grid_.cell_occupandy.at(grid_.cell_order[i]) = true;
      }
      if (n_matches_ >= (size_t) Config::maxFts())
        break;
    }
  }

  HSO_STOP_TIMER("feature_align");
}

bool Reprojector::pointQualityComparator(Candidate &lhs, Candidate &rhs) {

  // state_1:特征点的质量不一样时，高质量的点在前面
  // state_2:特征点的质量一样时，按照 角点>边缘点>梯度点 的顺序排列
  if (lhs.pt->type_ != rhs.pt->type_)
    return (lhs.pt->type_ > rhs.pt->type_);
  else {
    if (lhs.pt->ftr_type_ > rhs.pt->ftr_type_)
      return true;
    return false;
  }
}

bool Reprojector::seedComparator(SeedCandidate &lhs, SeedCandidate &rhs) {
  return (lhs.seed.sigma2 < rhs.seed.sigma2);
}

bool Reprojector::reprojectCell(Cell &cell, FramePtr frame, bool is_2nd, bool is_3rd) {

  if (cell.empty()) return false;

  // 在网格内，按照点的质量排序，优先使用优质点
  if (!is_2nd)
    cell.sort(boost::bind(&Reprojector::pointQualityComparator, _1, _2));

  Cell::iterator it = cell.begin();

  int succees = 0;

  while (it != cell.end()) {
    ++n_trials_;

    // 如果是删除的点，则从cell去掉
    if (it->pt->type_ == Point::TYPE_DELETED) {
      it = cell.erase(it);
      continue;
    }

    // 定义了直接找的方法
    if (!matcher_.findMatchDirect(*it->pt, *frame, it->px)) {
      it->pt->n_failed_reproj_++;
      if (it->pt->type_ == Point::TYPE_UNKNOWN && it->pt->n_failed_reproj_ > 15)
        map_.safeDeletePoint(it->pt);
      if (it->pt->type_ == Point::TYPE_CANDIDATE && it->pt->n_failed_reproj_ > 30)
        map_.point_candidates_.deleteCandidatePoint(it->pt);

      // DD added in 7.16
      if (it->pt->type_ == Point::TYPE_TEMPORARY && it->pt->n_failed_reproj_ > 30)
        it->pt->isBad_ = true;

      it = cell.erase(it);
      continue;
    }

    it->pt->n_succeeded_reproj_++;
    if (it->pt->type_ == Point::TYPE_UNKNOWN && it->pt->n_succeeded_reproj_ > 10)
      it->pt->type_ = Point::TYPE_GOOD;

    Feature *new_feature = new Feature(frame.get(), it->px, matcher_.search_level_);
    frame->addFeature(new_feature);

    // Here we add a reference in the feature to the 3D point, the other way
    // round is only done if this frame is selected as keyframe.
    new_feature->point = it->pt;

    if (matcher_.ref_ftr_->type == Feature::EDGELET) {
      new_feature->type = Feature::EDGELET;
      new_feature->grad = matcher_.A_cur_ref_ * matcher_.ref_ftr_->grad;
      // new_feature->grad = matcher_.A_cur_ref_* it->pt->hostFeature_->grad;
      new_feature->grad.normalize();
    } else if (matcher_.ref_ftr_->type == Feature::GRADIENT)
      new_feature->type = Feature::GRADIENT;
    else
      new_feature->type = Feature::CORNER;


    // If the keyframe is selected and we reproject the rest, we don't have to
    // check this point anymore.
    it = cell.erase(it);

    if (!is_3rd)
      return true;
    else {
      succees++;
      n_matches_++;
      if (succees >= 3 || n_matches_ >= Config::maxFts())
        return true;
    }
  }

  return false;
}

bool Reprojector::reprojectorSeeds(Sell &sell, FramePtr frame) {
  sell.sort(boost::bind(&Reprojector::seedComparator, _1, _2));
  Sell::iterator it = sell.begin();
  while (it != sell.end()) {
    if (matcher_.findMatchSeed(it->seed, *frame, it->px)) {
      assert(it->seed.ftr->point == NULL);

      ++n_seeds_;
      sum_seed_++;

      Vector3d pHost = it->seed.ftr->f * (1. / it->seed.mu);
      Vector3d xyz_world(it->seed.ftr->frame->T_f_w_.inverse() * pHost);
      Point *point = new Point(xyz_world, it->seed.ftr);

      point->idist_ = it->seed.mu;
      point->hostFeature_ = it->seed.ftr;

      // point->n_succeeded_reproj_++;
      point->type_ = Point::TYPE_TEMPORARY;

      if (it->seed.ftr->type == Feature::EDGELET)
        point->ftr_type_ = Point::FEATURE_EDGELET;
      else if (it->seed.ftr->type == Feature::CORNER)
        point->ftr_type_ = Point::FEATURE_CORNER;
      else
        point->ftr_type_ = Point::FEATURE_GRADIENT;

      // TODO?
      // it->seed.ftr->point = point;

      // depth_filter_->seed_converged_cb_(point, it->seed.sigma2);
      Feature *new_feature = new Feature(frame.get(), it->px, matcher_.search_level_);
      if (matcher_.ref_ftr_->type == Feature::EDGELET) {
        new_feature->type = Feature::EDGELET;
        new_feature->grad = matcher_.A_cur_ref_ * matcher_.ref_ftr_->grad;
        new_feature->grad.normalize();
      } else if (matcher_.ref_ftr_->type == Feature::GRADIENT)
        new_feature->type = Feature::GRADIENT;
      else
        new_feature->type = Feature::CORNER;

      new_feature->point = point;

      //old: point->addFrameRef(new_feature);
      // point->addFrameRef(it->seed.ftr);
      frame->addFeature(new_feature);

      // it->seed.ftr->frame->addFeature(it->seed.ftr);
      // it->seed.ftr->frame->setKeyPoints();

      //old: depth_filter_->seeds_.erase(it->index);
      it->seed.haveReprojected = true;
      it->seed.temp = point;
      // it->seed.ftr->haveAdded = true;

      point->seedStates_ = 0;
      map_.point_candidates_.addPauseSeedPoint(point);

      it = sell.erase(it);
      return true;
    } else
      ++it;
  }

  return false;
}

bool Reprojector::reprojectPoint(FramePtr frame, Point *point, vector<pair<Vector2d, Point *> > &cells) {

  // 得到最先观测到该地图点的2D特征
  Vector3d pHost = point->hostFeature_->f * (1.0 / point->idist_);

  assert((point->pos_ - (point->hostFeature_->frame->T_f_w_.inverse() * pHost)).norm() < 0.0001); // TODO:add by wzh
  // 将参考帧坐标系下的地图点 投影到 当前帧坐标系下
  Vector3d pTarget = (frame->T_f_w_ * point->hostFeature_->frame->T_f_w_.inverse()) * pHost;
  if (pTarget[2] < 0.00001) return false;

  // 将当前帧坐标系下的3D点 投影到 相机平面 得到 2D 特征
  Vector2d px(frame->cam_->world2cam(pTarget));

  // 判断共视关键帧的地图点 投影到 当前帧下的2D特征 是否在边界的8个像素内，因此一个patch_size是8个像素？
  if (frame->cam_->isInFrame(px.cast<int>(), 8)) // 8px is the patch size in the matcher
  {

    // 计算在图像的第几个单元格中
    const int k = static_cast<int>(px[1] / grid_.cell_size) * grid_.grid_n_cols
        + static_cast<int>(px[0] / grid_.cell_size);

    grid_.cells.at(k)->push_back(Candidate(point, px));

    cells.push_back(make_pair(px, point));

    nFeatures_++;

    return true;
  }
  return false;
}

bool Reprojector::reprojectorSeed(
    FramePtr frame, Seed &seed,
    list<Seed, aligned_allocator<Seed> >::iterator index) {
  // Vector3d pos_w(seed.ftr->frame->T_f_w_.inverse()*(1.0/seed.mu * seed.ftr->f));

  SE3 Tth = frame->T_f_w_ * seed.ftr->frame->T_f_w_.inverse();
  Vector3d pTarget = Tth * (1.0 / seed.mu * seed.ftr->f);
  if (pTarget[2] < 0.001) return false;

  Vector2d px(frame->cam_->world2cam(pTarget));

  // Vector2d px(frame->w2c(pos_w));

  if (frame->cam_->isInFrame(px.cast<int>(), 8)) {
    const int k = static_cast<int>(px[1] / grid_.cell_size) * grid_.grid_n_cols
        + static_cast<int>(px[0] / grid_.cell_size);
    grid_.seeds.at(k)->push_back(SeedCandidate(seed, px, index));
    return true;
  }
  return false;
}

void Reprojector::reprojectCellAll(vector<pair<Vector2d, Point *> > &cell, FramePtr frame) {
  if (cell.empty()) return;

  vector<pair<Vector2d, Point *> >::iterator it = cell.begin();
  while (it != cell.end()) {
    ++n_trials_;

    if (it->second->type_ == Point::TYPE_DELETED) {
      it = cell.erase(it);
      continue;
    }

    if (!matcher_.findMatchDirect(*(it->second), *frame, it->first)) {
      it->second->n_failed_reproj_++;

      if (it->second->type_ == Point::TYPE_UNKNOWN && it->second->n_failed_reproj_ > 15)
        map_.safeDeletePoint(it->second);
      if (it->second->type_ == Point::TYPE_CANDIDATE && it->second->n_failed_reproj_ > 30)
        map_.point_candidates_.deleteCandidatePoint(it->second);
      if (it->second->type_ == Point::TYPE_TEMPORARY && it->second->n_failed_reproj_ > 30)
        it->second->isBad_ = true;

      it = cell.erase(it);
      continue;
    }

    it->second->n_succeeded_reproj_++;
    if (it->second->type_ == Point::TYPE_UNKNOWN && it->second->n_succeeded_reproj_ > 10)
      it->second->type_ = Point::TYPE_GOOD;

    Feature *new_feature = new Feature(frame.get(), it->first, matcher_.search_level_);
    frame->addFeature(new_feature);

    // Here we add a reference in the feature to the 3D point, the other way
    // round is only done if this frame is selected as keyframe.
    new_feature->point = it->second;

    if (matcher_.ref_ftr_->type == Feature::EDGELET) {
      new_feature->type = Feature::EDGELET;
      new_feature->grad = matcher_.A_cur_ref_ * matcher_.ref_ftr_->grad;
      new_feature->grad.normalize();
    } else if (matcher_.ref_ftr_->type == Feature::GRADIENT)
      new_feature->type = Feature::GRADIENT;
    else
      new_feature->type = Feature::CORNER;

    // If the keyframe is selected and we reproject the rest, we don't have to
    // check this point anymore.
    it = cell.erase(it);

    n_matches_++;
    if (n_matches_ >= (size_t) Config::maxFts()) return;
  }
}

} // namespace hso
