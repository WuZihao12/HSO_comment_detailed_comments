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


#ifdef __SSE2__
#include <emmintrin.h>
#endif
#ifdef __ARM_NEON__
#include <arm_neon.h>
#endif
#include <hso/feature_alignment.h>
#include <hso/feature.h>
#include <hso/point.h>
#include <hso/matcher.h>

namespace hso {
namespace feature_alignment {

#define SUBPIX_VERBOSE 0

bool align1D(
    const cv::Mat &cur_img,
    const Vector2f &dir,                  // direction in which the patch is allowed to move
    uint8_t *ref_patch_with_border,
    uint8_t *ref_patch,
    const int n_iter,
    Vector2d &cur_px_estimate,
    double &h_inv) {
  const int halfpatch_size_ = 4;
  const int patch_size = 8;
  const int patch_area = 64;
  bool converged = false;

  // compute derivative of template and prepare inverse compositional
  float __attribute__((__aligned__(16))) ref_patch_dv[patch_area];
  Matrix2f H;
  H.setZero();

  // compute gradient and hessian
  const int ref_step = patch_size + 2;
  float *it_dv = ref_patch_dv;
  Vector2f J;
  for (int y = 0; y < patch_size; ++y) {
    uint8_t *it = ref_patch_with_border + (y + 1) * ref_step + 1;
    for (int x = 0; x < patch_size; ++x, ++it, ++it_dv) {
      J[0] = 0.5 * (dir[0] * (it[1] - it[-1]) + dir[1] * (it[ref_step] - it[-ref_step]));
      J[1] = 1;
      *it_dv = J[0];
      H += J * J.transpose();
    }
  }
  h_inv = 1.0 / H(0, 0) * patch_size * patch_size;
  Matrix2f Hinv = H.inverse();
  float mean_diff = 0;

  // Compute pixel location in new image:
  float u = cur_px_estimate.x();
  float v = cur_px_estimate.y();

  // termination condition
  const float min_update_squared = 0.01 * 0.01;
  const int cur_step = cur_img.step.p[0];
  float chi2 = 0;
  Vector2f update;
  update.setZero();
  Vector2f Jres;
  Jres.setZero();
  for (int iter = 0; iter < n_iter; ++iter) {
    int u_r = floor(u);
    int v_r = floor(v);
    if (u_r < halfpatch_size_ || v_r < halfpatch_size_ || u_r >= cur_img.cols - halfpatch_size_
        || v_r >= cur_img.rows - halfpatch_size_)
      break;

    if (isnan(u) || isnan(v)) // TODO very rarely this can happen, maybe H is singular? should not be at corner.. check
      return false;

    // compute interpolation weights
    float subpix_x = u - u_r;
    float subpix_y = v - v_r;
    float wTL = (1.0 - subpix_x) * (1.0 - subpix_y);
    float wTR = subpix_x * (1.0 - subpix_y);
    float wBL = (1.0 - subpix_x) * subpix_y;
    float wBR = subpix_x * subpix_y;

    // loop through search_patch, interpolate
    uint8_t *it_ref = ref_patch;
    float *it_ref_dv = ref_patch_dv;
    float new_chi2 = 0.0;
    Jres.setZero();
    for (int y = 0; y < patch_size; ++y) {
      uint8_t *it = (uint8_t *) cur_img.data + (v_r + y - halfpatch_size_) * cur_step + u_r - halfpatch_size_;
      for (int x = 0; x < patch_size; ++x, ++it, ++it_ref, ++it_ref_dv) {
        float search_pixel = wTL * it[0] + wTR * it[1] + wBL * it[cur_step] + wBR * it[cur_step + 1];
        float res = search_pixel - *it_ref + mean_diff;
        Jres[0] -= res * (*it_ref_dv);
        Jres[1] -= res;
        new_chi2 += res * res;
      }
    }

    if (iter > 0 && new_chi2 > chi2) {
#if SUBPIX_VERBOSE
      cout << "error increased." << endl;
#endif
      // u -= update[0];
      // v -= update[1];
      u -= update[0] * dir[0];
      v -= update[0] * dir[1];
      break;
    }

    chi2 = new_chi2;
    update = Hinv * Jres;
    u += update[0] * dir[0];
    v += update[0] * dir[1];
    mean_diff += update[1];

#if SUBPIX_VERBOSE
    cout << "Iter " << iter << ":"
         << "\t u=" << u << ", v=" << v
         << "\t update = " << update[0] << ", " << update[1]
         << "\t new chi2 = " << new_chi2 << endl;
#endif

    // if(update[0]*update[0]+update[1]*update[1] < min_update_squared)
    if (update[0] * update[0] < min_update_squared) {
#if SUBPIX_VERBOSE
      cout << "converged." << endl;
#endif
      converged = true;
      break;
    }
  }

  cur_px_estimate << u, v;
  return converged;
}

bool align1D(
    const cv::Mat &cur_img,
    const Vector2f &dir,                  // direction in which the patch is allowed to move
    float *ref_patch_with_border,
    float *ref_patch,
    const int n_iter,
    Vector2d &cur_px_estimate,
    double &h_inv,
    float *cur_patch) {
  const int halfpatch_size_ = 4;
  const int patch_size = 8;
  const int patch_area = 64;
  bool converged = false;

  // compute derivative of template and prepare inverse compositional
  float ref_patch_dv[patch_area];
  Matrix2f H;
  H.setZero();

  float grad_weight[patch_area];

  // compute gradient and hessian
  const int ref_step = patch_size + 2;
  float *it_dv = ref_patch_dv;
  float *it_weight = grad_weight;
  Vector2f J;
  for (int y = 0; y < patch_size; ++y) {
    float *it = ref_patch_with_border + (y + 1) * ref_step + 1;
    for (int x = 0; x < patch_size; ++x, ++it, ++it_dv, ++it_weight) {
      J[0] = 0.5 * (dir[0] * (it[1] - it[-1]) + dir[1] * (it[ref_step] - it[-ref_step]));
      J[1] = 1.;
      *it_dv = J[0];

      *it_weight = sqrtf(250.0 / (250.0 + J[0] * J[0]));

      H += J * J.transpose() * (*it_weight);
    }
  }

  for (int i = 0; i < 2; i++) H(i, i) *= (1 + 0.001);

  h_inv = 1.0 / H(0, 0) * patch_size * patch_size;
  Matrix2f Hinv = H.inverse();
  float mean_diff = 0;

  // Compute pixel location in new image:
  float u = cur_px_estimate.x();
  float v = cur_px_estimate.y();

  // termination condition
  const float min_update_squared = 0.01 * 0.01;
  const int cur_step = cur_img.step.p[0];
  float chi2 = 0;
  Vector2f update;
  update.setZero();
  Vector2f Jres;
  Jres.setZero();
  for (int iter = 0; iter < n_iter; ++iter) {
    float *cur_patch_ptr = cur_patch;

    int u_r = floor(u);
    int v_r = floor(v);
    if (u_r < halfpatch_size_ || v_r < halfpatch_size_ || u_r >= cur_img.cols - halfpatch_size_
        || v_r >= cur_img.rows - halfpatch_size_)
      break;

    if (isnan(u) || isnan(v)) // TODO very rarely this can happen, maybe H is singular? should not be at corner.. check
      return false;

    // compute interpolation weights
    float subpix_x = u - u_r;
    float subpix_y = v - v_r;
    float wTL = (1.0 - subpix_x) * (1.0 - subpix_y);
    float wTR = subpix_x * (1.0 - subpix_y);
    float wBL = (1.0 - subpix_x) * subpix_y;
    float wBR = subpix_x * subpix_y;

    // loop through search_patch, interpolate
    float *it_ref = ref_patch;
    float *it_ref_dv = ref_patch_dv;
    float *it_weight = grad_weight;
    float new_chi2 = 0.0;
    Jres.setZero();
    for (int y = 0; y < patch_size; ++y) {
      uint8_t *it = (uint8_t *) cur_img.data + (v_r + y - halfpatch_size_) * cur_step + u_r - halfpatch_size_;
      for (int x = 0; x < patch_size; ++x, ++it, ++it_ref, ++it_ref_dv, ++it_weight) {
        float search_pixel = wTL * it[0] + wTR * it[1] + wBL * it[cur_step] + wBR * it[cur_step + 1];
        float res = search_pixel - *it_ref + mean_diff;

        Jres[0] -= res * (*it_ref_dv) * (*it_weight);
        Jres[1] -= res * (*it_weight);

        new_chi2 += res * res * (*it_weight);

        if (cur_patch != NULL) {
          *cur_patch_ptr = search_pixel;
          ++cur_patch_ptr;
        }

      }
    }

    // if(iter > 0 && new_chi2 > chi2)
    // {
    //     #if SUBPIX_VERBOSE
    //         cout << "error increased." << endl;
    //     #endif
    //     // u -= update[0];
    //     // v -= update[1];
    //     u -= update[0]*dir[0];
    //     v -= update[0]*dir[1];
    //     break;
    // }

    chi2 = new_chi2;
    update = Hinv * Jres;
    u += update[0] * dir[0];
    v += update[0] * dir[1];
    mean_diff += update[1];

#if SUBPIX_VERBOSE
    cout << "Iter " << iter << ":"
    << "\t u=" << u << ", v=" << v
    << "\t update = " << update[0] << ", " << update[1]
    << "\t new chi2 = " << new_chi2 << endl;
#endif

    if (update[0] * update[0] < min_update_squared) {
#if SUBPIX_VERBOSE
      cout << "converged." << endl;
#endif

      converged = true;
      break;
    }
  }

  if (chi2 > 1000 * patch_area) converged = false;

  cur_px_estimate << u, v;
  return converged;
}

bool align2D(
    const cv::Mat &cur_img,
    uint8_t *ref_patch_with_border,
    uint8_t *ref_patch,
    const int n_iter,
    Vector2d &cur_px_estimate,
    bool no_simd,
    float *cur_patch) {
// #ifdef __ARM_NEON__
//   if(!no_simd)
//     return align2D_NEON(cur_img, ref_patch_with_border, ref_patch, n_iter, cur_px_estimate);
// #endif

  const int halfpatch_size_ = 4;
  const int patch_size_ = 8;
  const int patch_area_ = 64;
  bool converged = false;

  // compute derivative of template and prepare inverse compositional
  float __attribute__((__aligned__(16))) ref_patch_dx[patch_area_];
  float __attribute__((__aligned__(16))) ref_patch_dy[patch_area_];
  Matrix3f H;
  H.setZero();

  // compute gradient and hessian
  const int ref_step = patch_size_ + 2;
  float *it_dx = ref_patch_dx;
  float *it_dy = ref_patch_dy;
  Vector3f J;
  for (int y = 0; y < patch_size_; ++y) {
    uint8_t *it = ref_patch_with_border + (y + 1) * ref_step + 1;
    for (int x = 0; x < patch_size_; ++x, ++it, ++it_dx, ++it_dy) {
      J[0] = 0.5 * (it[1] - it[-1]);
      J[1] = 0.5 * (it[ref_step] - it[-ref_step]);
      J[2] = 1;
      *it_dx = J[0];
      *it_dy = J[1];
      H += J * J.transpose();
    }
  }
  Matrix3f Hinv = H.inverse();
  float mean_diff = 0;

  // Compute pixel location in new image:
  float u = cur_px_estimate.x();
  float v = cur_px_estimate.y();

  // termination condition
  const float min_update_squared = 0.03 * 0.03;
  const int cur_step = cur_img.step.p[0];
  // float chi2 = 0;
  Vector3f update;
  update.setZero();
  Vector3f Jres;
  Jres.setZero();
  // float aff_a = 1, aff_b = 0;
  for (int iter = 0; iter < n_iter; ++iter) {
    int u_r = floor(u);
    int v_r = floor(v);
    if (u_r < halfpatch_size_ || v_r < halfpatch_size_ || u_r >= cur_img.cols - halfpatch_size_
        || v_r >= cur_img.rows - halfpatch_size_)
      break;

    if (isnan(u) || isnan(v)) // TODO very rarely this can happen, maybe H is singular? should not be at corner.. check
      return false;

    // compute interpolation weights
    float subpix_x = u - u_r;
    float subpix_y = v - v_r;
    float wTL = (1.0 - subpix_x) * (1.0 - subpix_y);
    float wTR = subpix_x * (1.0 - subpix_y);
    float wBL = (1.0 - subpix_x) * subpix_y;
    float wBR = subpix_x * subpix_y;

    // loop through search_patch, interpolate
    uint8_t *it_ref = ref_patch;
    float *it_ref_dx = ref_patch_dx;
    float *it_ref_dy = ref_patch_dy;
    // float new_chi2 = 0.0;
    Jres.setZero();
    // float sxx=0, syy=0, sxy=0, sx=0, sy=0, sw=0;
    for (int y = 0; y < patch_size_; ++y) {
      uint8_t *it = (uint8_t *) cur_img.data + (v_r + y - halfpatch_size_) * cur_step + u_r - halfpatch_size_;
      for (int x = 0; x < patch_size_; ++x, ++it, ++it_ref, ++it_ref_dx, ++it_ref_dy) {
        float search_pixel = wTL * it[0] + wTR * it[1] + wBL * it[cur_step] + wBR * it[cur_step + 1];
        // float res = search_pixel - (*it_ref)*aff_a+aff_b + mean_diff;
        // Jres[0] -= res*(*it_ref_dx * aff_a);
        // Jres[1] -= res*(*it_ref_dy * aff_a);
        // Jres[2] -= res*aff_a;

        float res = search_pixel - (*it_ref) + mean_diff;
        Jres[0] -= res * (*it_ref_dx);
        Jres[1] -= res * (*it_ref_dy);
        Jres[2] -= res;
        // const float weight_aff = fabsf(res) < 10.0f? fabsf(res) < 5.0f? 1.0 : 5.0f / fabsf(res) : 0; 
        // sxx += (*it_ref)*(*it_ref)*weight_aff; 
        // syy += search_pixel*search_pixel*weight_aff;
        // sxy += (*it_ref)*search_pixel*weight_aff;
        // sx += (*it_ref)*weight_aff;
        // sy += search_pixel*weight_aff;
        // sw += weight_aff;

        // new_chi2 += res*res;
        if (iter == n_iter - 1 && cur_patch != NULL) {
          *cur_patch = search_pixel;
          ++cur_patch;
        }
      }
    }

//     if(iter > 0 && new_chi2 > chi2 && no_simd)
//     {
// #if SUBPIX_VERBOSE
//       cout << "error increased." << endl;
// #endif
//       u -= update[0];
//       v -= update[1];
//       break;
//     }

    // chi2 = new_chi2;
    update = Hinv * Jres;
    u += update[0];
    v += update[1];
    mean_diff += update[2];

    // aff_a = sqrtf((syy - sy*sy/sw) / (sxx - sx*sx/sw));
    // aff_b = (sy - aff_a*sx)/sw;


#if SUBPIX_VERBOSE
    cout << "Iter " << iter << ":"
         << "\t u=" << u << ", v=" << v
         << "\t update = " << update[0] << ", " << update[1]
         << "\t new chi2 = " << new_chi2 << endl;
#endif

    if (update[0] * update[0] + update[1] * update[1] < min_update_squared) {
#if SUBPIX_VERBOSE
      cout << "converged." << endl;
#endif
      converged = true;
      break;
    }
  }

  cur_px_estimate << u, v;
  return converged;
}
/********************************
 * @ function:    使用逆向组合法, 进行2D图像对齐
 *
 * @ param:       const cv::Mat& cur_img            cur_image在searchlevel的金字塔图像
 *                float* ref_patch_with_border    从ref变换到cur上的参考patch_border(不准确的, 带1个大的边界)!!
 *                float* ref_patch                从ref变换到cur上的参考patch(不准确的)
 *                const int n_iter                  最大迭代次数
 *                Vector2d& cur_px_estimate         当前估计的cur上特征像素位置
 *                bool no_simd
 *
 * @ note:        优化方程: p = min Sum_x[ T(x+△x,y+△y) + △i - I(x,y) + i]^2
 *                优化变量: p = [x, y, i]
 *                其他的和1D情况相同
 *******************************/
bool align2D(
    const cv::Mat &cur_img, float *ref_patch_with_border, float *ref_patch,
    const int n_iter, Vector2d &cur_px_estimate, bool no_simd, float *cur_patch) {

  const int halfpatch_size_ = 4;
  const int patch_size_ = 8;
  const int patch_area_ = 64;
  bool converged = false;

  // compute derivative of template and prepare inverse compositional
  float ref_patch_dx[patch_area_];
  float ref_patch_dy[patch_area_];
  Matrix3f H;
  H.setZero();

  float grad_weight[patch_area_];

  // compute gradient and hessian
  const int ref_step = patch_size_ + 2;
  float *it_dx = ref_patch_dx;
  float *it_dy = ref_patch_dy;
  float *it_weight = grad_weight;

  Vector3f J;
  // 行循环
  for (int y = 0; y < patch_size_; ++y) {
    float *it = ref_patch_with_border + (y + 1) * ref_step + 1;
    for (int x = 0; x < patch_size_; ++x, ++it, ++it_dx, ++it_dy, ++it_weight) {
      J[0] = 0.5 * (it[1] - it[-1]); // x方向导数
      J[1] = 0.5 * (it[ref_step] - it[-ref_step]); // y方向导数
      J[2] = 1.; // 亮度误差导数
      *it_dx = J[0];
      *it_dy = J[1];

      *it_weight = sqrtf(250.0 / (250.0 + (J[0] * J[0] + J[1] * J[1])));

      H += J * J.transpose() * (*it_weight);
    }
  }

  for (int i = 0; i < 3; i++) H(i, i) *= (1 + 0.001);

  Matrix3f Hinv = H.inverse();



  // Compute pixel location in new image:
  float u = cur_px_estimate.x();
  float v = cur_px_estimate.y();

  // termination condition
  const float min_update_squared = 0.03 * 0.03;
  const int cur_step = cur_img.step.p[0];

  float mean_diff = 0;
  float chi2 = 0;
  Vector3f update;
  update.setZero();
  Vector3f Jres;
  Jres.setZero();

  for (int iter = 0; iter < n_iter; ++iter) {
    float *cur_patch_ptr = cur_patch;

    int u_r = floor(u);
    int v_r = floor(v);
    if (u_r < halfpatch_size_ || v_r < halfpatch_size_ || u_r >= cur_img.cols - halfpatch_size_
        || v_r >= cur_img.rows - halfpatch_size_)
      break;

    if (isnan(u) || isnan(v)) // TODO very rarely this can happen, maybe H is singular? should not be at corner.. check
      return false;

    // compute interpolation weights
    // 双线性插值
    float subpix_x = u - u_r;
    float subpix_y = v - v_r;
    float wTL = (1.0 - subpix_x) * (1.0 - subpix_y);
    float wTR = subpix_x * (1.0 - subpix_y);
    float wBL = (1.0 - subpix_x) * subpix_y;
    float wBR = subpix_x * subpix_y;

    // loop through search_patch, interpolate
    float *it_ref = ref_patch;
    float *it_ref_dx = ref_patch_dx;
    float *it_ref_dy = ref_patch_dy;
    float *it_weight = grad_weight;
    float new_chi2 = 0.0;
    Jres.setZero();
    for (int y = 0; y < patch_size_; ++y) {
      uint8_t *it = (uint8_t *) cur_img.data + (v_r + y - halfpatch_size_) * cur_step + u_r - halfpatch_size_;
      for (int x = 0; x < patch_size_; ++x, ++it, ++it_ref, ++it_ref_dx, ++it_ref_dy, ++it_weight) {

        float search_pixel = wTL * it[0] + wTR * it[1] + wBL * it[cur_step] + wBR * it[cur_step + 1];
        // 残差e，需要不断更新
        float res = search_pixel - (*it_ref) + mean_diff;

        Jres[0] -= res * (*it_ref_dx) * (*it_weight);
        Jres[1] -= res * (*it_ref_dy) * (*it_weight);
        Jres[2] -= res * (*it_weight);

        new_chi2 += res * res * (*it_weight);

        if (cur_patch != NULL) {
          *cur_patch_ptr = search_pixel;
          ++cur_patch_ptr;
        }
      }
    }

    chi2 = new_chi2;
    update = Hinv * Jres;
    u += update[0];
    v += update[1];
    mean_diff += update[2];

#if SUBPIX_VERBOSE
    cout << "Iter " << iter << ":"
    << "\t u=" << u << ", v=" << v
    << "\t update = " << update[0] << ", " << update[1]
    << "\t new chi2 = " << new_chi2 << endl;
#endif

    if (update[0] * update[0] + update[1] * update[1] < min_update_squared) {
#if SUBPIX_VERBOSE
      cout << "converged." << endl;
#endif
      converged = true;
      break;
    }
  }

  if (chi2 > 1000 * patch_area_) converged = false;

  cur_px_estimate << u, v;
  return converged;
}

#define  DESCALE(x, n)     (((x) + (1 << ((n)-1))) >> (n)) // rounds to closest integer and descales

bool align2D_SSE2(
    const cv::Mat &cur_img,
    uint8_t *ref_patch_with_border,
    uint8_t *ref_patch,
    const int n_iter,
    Vector2d &cur_px_estimate) {
  // TODO: This function should not be used as the alignment is not robust to illumination changes!
  const int halfpatch_size = 4;
  const int patch_size = 8;
  const int patch_area = 64;
  bool converged = false;
  const int W_BITS = 14;

  // compute derivative of template and prepare inverse compositional
  int16_t __attribute__((__aligned__(16))) ref_patch_dx[patch_area];
  int16_t __attribute__((__aligned__(16))) ref_patch_dy[patch_area];

  // compute gradient and hessian
  const int ref_step = patch_size + 2;
  int16_t *it_dx = ref_patch_dx;
  int16_t *it_dy = ref_patch_dy;
  float A11 = 0, A12 = 0, A22 = 0;
  for (int y = 0; y < patch_size; ++y) {
    uint8_t *it = ref_patch_with_border + (y + 1) * ref_step + 1;
    for (int x = 0; x < patch_size; ++x, ++it, ++it_dx, ++it_dy) {
      int16_t dx = static_cast<int16_t>(it[1]) - it[-1];
      int16_t dy = static_cast<int16_t>(it[ref_step]) - it[-ref_step];
      *it_dx = dx;
      *it_dy = dy;  // we are missing a factor 1/2
      A11 += static_cast<float>(dx * dx); // we are missing a factor 1/4
      A12 += static_cast<float>(dx * dy);
      A22 += static_cast<float>(dy * dy);
    }
  }

  // Compute pixel location in new image:
  float u = cur_px_estimate.x();
  float v = cur_px_estimate.y();

  // termination condition
  const float min_update_squared = 0.03 * 0.03;
  const int cur_step = cur_img.step.p[0];
  const float Dinv = 1.0f / (A11 * A22 - A12 * A12); // we are missing an extra factor 16
  float chi2 = 0;
  float update_u = 0, update_v = 0;

  for (int iter = 0; iter < n_iter; ++iter) {
    int u_r = floor(u);
    int v_r = floor(v);
    if (u_r < halfpatch_size || v_r < halfpatch_size || u_r >= cur_img.cols - halfpatch_size
        || v_r >= cur_img.rows - halfpatch_size)
      break;

    if (isnan(u) || isnan(v)) // TODO very rarely this can happen, maybe H is singular? should not be at corner.. check
      return false;

    float subpix_x = u - u_r;
    float subpix_y = v - v_r;
    float b1 = 0, b2 = 0;
    float new_chi2 = 0.0;

#ifdef __SSE2__
    // compute bilinear interpolation weights
    int wTL = static_cast<int>((1.0f - subpix_x) * (1.0f - subpix_y) * (1 << W_BITS));
    int wTR = static_cast<int>(subpix_x * (1.0f - subpix_y) * (1 << W_BITS));
    int wBL = static_cast<int>((1.0f - subpix_x) * subpix_y * (1 << W_BITS));
    int wBR = (1 << W_BITS) - wTL - wTR - wBL;

    __m128i qw0 = _mm_set1_epi32(wTL + (wTR << 16)); // Sets the 4 signed 32-bit integer values to [wTL, wTR].
    __m128i qw1 = _mm_set1_epi32(wBL + (wBR << 16));
    __m128i z = _mm_setzero_si128();
    __m128 qb0 = _mm_setzero_ps(); // 4 floats
    __m128 qb1 = _mm_setzero_ps(); // 4 floats
    __m128i qdelta = _mm_set1_epi32(1 << (W_BITS - 1));
    for (int y = 0; y < patch_size; ++y) {
      const uint8_t *it = (const uint8_t *) cur_img.data + (v_r + y - halfpatch_size) * cur_step + u_r - halfpatch_size;

      // Iptr is aligned!
      //__m128i diff0 = _mm_load_si128((const __m128i*)(ref_patch + y*8));
      __m128i diff = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *) (ref_patch + y * 8)), z);

      // load the lower 64 bits and unpack [8u 0 8u 0..]
      __m128i v00 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *) (it)), z);
      __m128i v01 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *) (it + 1)), z);
      __m128i v10 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *) (it + cur_step)), z);
      __m128i v11 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *) (it + cur_step + 1)), z);

      // interpolate top row and bottom row
      // _mm_unpacklo_epi16: Interleaves the lower 4 signed or unsigned 16-bit integers in a with the lower 4 signed or unsigned 16-bit integers in b.
      // _mm_madd_epi16: Multiplies the 8 signed 16-bit integers from a by the 8 signed 16-bit integers from b.
      __m128i t0 = _mm_add_epi32(_mm_madd_epi16(_mm_unpacklo_epi16(v00, v01), qw0),
                                 _mm_madd_epi16(_mm_unpacklo_epi16(v10, v11), qw1));
      __m128i t1 = _mm_add_epi32(_mm_madd_epi16(_mm_unpackhi_epi16(v00, v01), qw0),
                                 _mm_madd_epi16(_mm_unpackhi_epi16(v10, v11), qw1));

      // _mm_srai_epi32: Shifts the 4 signed 32-bit integers in A right by count bits while shifting in the sign bit.
      t0 = _mm_srai_epi32(_mm_add_epi32(t0, qdelta), W_BITS); // adding qdelta is for rounding closest int
      t1 = _mm_srai_epi32(_mm_add_epi32(t1, qdelta), W_BITS);

      // compute 8xres:
      // _mm_packs_epi32: Packs the 8 signed 32-bit integers from a and b into signed 16-bit integers and saturates.
      diff = _mm_subs_epi16(_mm_packs_epi32(t0, t1), diff);

      // load gradient dX and dY, both are aligned!
      v00 = _mm_load_si128((const __m128i *) (ref_patch_dx + y * patch_size)); // [dx1, dx2, dx3, dx4 ...]
      v01 = _mm_load_si128((const __m128i *) (ref_patch_dy + y * patch_size)); // [dy1, dy2, dy3, dy4 ...]

      // _mm_mulhi_epi16: Multiplies the 8 signed 16-bit integers from a by the 8 signed 16-bit integers from b.
      v10 = _mm_mullo_epi16(v00, diff); // Packs the lower 16 bits of the 8 signed 32-bit results. [15:0]
      v11 = _mm_mulhi_epi16(v00, diff); // Packs the upper 16 bits of the 8 signed 32-bit results. [31:16]

      // _mm_unpacklo_epi16: Interleaves the lower 4 signed or unsigned 16-bit integers with the lower 4 signed or unsigned 16-bit integers in b.
      v00 = _mm_unpacklo_epi16(v10, v11);
      v10 = _mm_unpackhi_epi16(v10, v11);

      // convert to float and add to dx
      qb0 = _mm_add_ps(qb0, _mm_cvtepi32_ps(v00));
      qb0 = _mm_add_ps(qb0, _mm_cvtepi32_ps(v10));

      // same with dY
      v10 = _mm_mullo_epi16(v01, diff);
      v11 = _mm_mulhi_epi16(v01, diff);
      v00 = _mm_unpacklo_epi16(v10, v11);
      v10 = _mm_unpackhi_epi16(v10, v11);
      qb1 = _mm_add_ps(qb1, _mm_cvtepi32_ps(v00));
      qb1 = _mm_add_ps(qb1, _mm_cvtepi32_ps(v10));
    }

    float __attribute__((__aligned__(16))) buf[4];
    _mm_store_ps(buf, qb0);
    b1 += buf[0] + buf[1] + buf[2] + buf[3];
    _mm_store_ps(buf, qb1);
    b2 += buf[0] + buf[1] + buf[2] + buf[3];
#endif

    // compute -A^-1*b
    update_u = ((A12 * b2 - A22 * b1) * Dinv)
        * 2; // * 2 to compensate because above, we did not compute the derivative correctly
    update_v = ((A12 * b1 - A11 * b2) * Dinv) * 2;
    u += update_u;
    v += update_v;

#if SUBPIX_VERBOSE
    cout << "Iter " << iter << ":"
         << "\t u=" << u << ", v=" << v
         << "\t update = " << update_u << ", " << update_v
         << "\t new chi2 = " << new_chi2 << endl;
#endif

    if (update_u * update_u + update_v * update_v < min_update_squared) {
#if SUBPIX_VERBOSE
      cout << "converged." << endl;
#endif
      converged = true;
      break;
    }
    chi2 = new_chi2;
  }

  cur_px_estimate << u, v;
  return converged;
}

bool align2D_NEON(
    const cv::Mat &cur_img,
    uint8_t *ref_patch_with_border,
    uint8_t *ref_patch,
    const int n_iter,
    Vector2d &cur_px_estimate) {
  const int halfpatch_size = 4;
  const int patch_size = 8;
  const int patch_area = 64;
  bool converged = false;
  const int W_BITS = 14;

  // compute derivative of template and prepare inverse compositional
  int16_t __attribute__((__aligned__(16))) ref_patch_dx[patch_area];
  int16_t __attribute__((__aligned__(16))) ref_patch_dy[patch_area];

  // compute gradient and hessian
  const int ref_step = patch_size + 2;
  int16_t *it_dx = ref_patch_dx;
  int16_t *it_dy = ref_patch_dy;
  Matrix3f H;
  H.setZero();
  for (int y = 0; y < patch_size; ++y) {
    uint8_t *it = ref_patch_with_border + (y + 1) * ref_step + 1;
    for (int x = 0; x < patch_size; ++x, ++it, ++it_dx, ++it_dy) {
      *it_dx = static_cast<int16_t>(it[1] - it[-1]);
      *it_dy = static_cast<int16_t>(it[ref_step] - it[-ref_step]); // divide by 2 missing
      Vector3f J(*it_dx, *it_dy, 1.0f);
      H += J * J.transpose();
    }
  }
  Matrix3f Hinv = H.inverse();
  float mean_diff = 0.0;

  // Compute pixel location in new image:
  float u = cur_px_estimate.x();
  float v = cur_px_estimate.y();

  // termination condition
  const float min_update_squared = 0.03 * 0.03;
  const int cur_step = cur_img.step.p[0];
  Vector3f update;
  Vector3f Jres;
  for (int iter = 0; iter < n_iter; ++iter) {
    int u_r = floor(u);
    int v_r = floor(v);
    if (u_r < halfpatch_size || v_r < halfpatch_size || u_r >= cur_img.cols - halfpatch_size
        || v_r >= cur_img.rows - halfpatch_size)
      break;

    if (isnan(u) || isnan(v)) // TODO very rarely this can happen, maybe H is singular? should not be at corner.. check
      return false;

    float subpix_x = u - u_r;
    float subpix_y = v - v_r;
    float b1 = 0, b2 = 0;

#ifdef __ARM_NEON__
    const int SHIFT_BITS = 7;
    const uint16_t wTL = static_cast<uint16_t>((1.0f-subpix_x)*(1.0f-subpix_y)*(1<<SHIFT_BITS));
    const uint16_t wTR = static_cast<uint16_t>(subpix_x*(1.0f-subpix_y)*(1<<SHIFT_BITS));
    const uint16_t wBL = static_cast<uint16_t>((1.0f-subpix_x)*subpix_y*(1<<SHIFT_BITS));
    const uint16_t wBR = static_cast<uint16_t>((1 << SHIFT_BITS) - wTL - wTR - wBL);

    // initialize result to zero
    int32x4_t vb1 = vdupq_n_s32(0);
    int32x4_t vb2 = vdupq_n_s32(0);
    int16x8_t vmean_diff = vdupq_n_s16( (int16_t) (mean_diff+0.5) );
    int16x8_t vres_sum = vdupq_n_s16(0);
    for(int y=0; y<patch_size; ++y)
    {
      const uint8_t* it  = (const uint8_t*) cur_img.data + (v_r+y-halfpatch_size)*cur_step + u_r-halfpatch_size;

      // load and convert from uint8 to uint16
      uint16x8_t v00 = vmovl_u8( vld1_u8( it ) );
      uint16x8_t v01 = vmovl_u8( vld1_u8( it + 1 ) );
      uint16x8_t v10 = vmovl_u8( vld1_u8( it + cur_step ) );
      uint16x8_t v11 = vmovl_u8( vld1_u8( it + cur_step + 1 ) );

      // vector multiply by scalar
      v00 = vmulq_n_u16( v00, wTL );
      v01 = vmulq_n_u16( v01, wTR );
      v10 = vmulq_n_u16( v10, wBL );
      v11 = vmulq_n_u16( v11, wBR );

      // add all results together
      v00 = vaddq_u16( v00, vaddq_u16( v01, vaddq_u16( v10, v11 ) ) );

      // descale: shift right by constant
      v00 = vshrq_n_u16(v00, SHIFT_BITS);

      // compute difference between reference and interpolated patch,
      // use reinterpet-cast to make signed [-255,255]
      int16x8_t res = vsubq_s16(vreinterpretq_s16_u16(v00), vreinterpretq_s16_u16(vmovl_u8(vld1_u8( ref_patch + y*8 ))));

      // correct res with mean difference
      res = vaddq_s16(res, vmean_diff);

      // compute sum of the residual
      vres_sum = vaddq_s16(vres_sum, res); 

      // Vector multiply accumulate long: vmla -> Vr[i] := Va[i] + Vb[i] * Vc[i]
      // int32x4_t  vmlal_s16(int32x4_t a, int16x4_t b, int16x4_t c);    // VMLAL.S16 q0,d0,d0
      int16x8_t grad = vld1q_s16(ref_patch_dx + y*patch_size);
      vb1 = vmlal_s16(vb1, vget_low_s16(grad), vget_low_s16(res));
      vb1 = vmlal_s16(vb1, vget_high_s16(grad), vget_high_s16(res));

      grad = vld1q_s16(ref_patch_dy + y*patch_size);
      vb2 = vmlal_s16(vb2, vget_low_s16(grad), vget_low_s16(res));
      vb2 = vmlal_s16(vb2, vget_high_s16(grad), vget_high_s16(res));
    }

    // finally, sum results of vb1, vb2 and vres_sum
    int32x2_t tmp;
    tmp = vpadd_s32(vget_low_s32(vb1), vget_high_s32(vb1));
    Jres[0] = -vget_lane_s32(tmp, 0) - vget_lane_s32(tmp, 1);

    tmp = vpadd_s32(vget_low_s32(vb2), vget_high_s32(vb2));
    Jres[1] = -vget_lane_s32(tmp, 0) - vget_lane_s32(tmp, 1);

    int32x4_t vres_sum1 = vpaddlq_s16(vres_sum);
    tmp = vpadd_s32(vget_low_s32(vres_sum1), vget_high_s32(vres_sum1));
    Jres[2] = -vget_lane_s32(tmp, 0) - vget_lane_s32(tmp, 1);
#endif

    update = Hinv * Jres * 2; // * 2 to compensate because above, we did not compute the derivative correctly
    u += update[0];
    v += update[1];
    mean_diff += update[2];

#if SUBPIX_VERBOSE
    cout << "Iter " << iter << ":"
         << "\t u=" << u << ", v=" << v
         << "\t update = " << update[0] << ", " << update[1] << endl;
#endif

    if (update[0] * update[0] + update[1] * update[1] < min_update_squared) {
#if SUBPIX_VERBOSE
      cout << "converged." << endl;
#endif
      converged = true;
      break;
    }
  }

  cur_px_estimate << u, v;
  return converged;
}

} // namespace feature_alignment
} // namespace hso
