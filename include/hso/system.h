//
// Please Call me FeiZi. Thank You!
// Created by wzh on 2022/10/8.
//

#ifndef HSO_INCLUDE_HSO_SYSTEM_H_
#define HSO_INCLUDE_HSO_SYSTEM_H_

#include <boost/thread.hpp>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include <sophus/se3.h>

#include "viewer.h"
#include "camera.h"
#include "ImageReader.h"
#include "config.h"
#include "frame_handler_mono.h"
#include "frame_handler_base.h"
#include "map.h"
#include "frame.h"
#include "feature.h"
#include "point.h"
#include "viewer.h"
#include "depth_filter.h"

namespace hso
{
class System
{
  public:
    hso::AbstractCamera *cam_;
    FrameHandlerMono *vo_;
    hso::Viewer *viewer_;
    boost::thread *viewer_thread_;
    System(int argc, const char **argv);
    ~System();

    void paraIni();
    void parseArgument(const char *arg);
    void saveResult(bool stamp_valid, string ros_path);

  public:
    void runFromFolder();
    void runFromRos(const cv::Mat &im, int img_id, const double &timestamp);

  public:
    hso::ImageReader* image_reader_;
  public:
    std::string g_image_folder;
    std::string g_stamp_folder;
    std::string g_calib_path;
    std::string g_result_name;

    int g_start;
    int g_end;
};
}

#endif //HSO_INCLUDE_HSO_SYSTEM_H_
