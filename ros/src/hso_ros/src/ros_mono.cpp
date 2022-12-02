//
// Please Call me FeiZi. Thank You!
// Created by wzh on 2022/10/8.
//

#include <iostream>
#include <string>
#include <chrono>
#include <fstream>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <queue>
#include<thread>
#include<mutex>

#include <opencv2/core.hpp>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include "../../include/hso/system.h"

#include <rosbag/bag.h>
#include <rosbag/view.h>

#ifdef USE_BACKWARD
#include "hso/backward.hpp"
#define BACKWARD_HAS_DW 1
namespace backward
{
backward::SignalHandling sh;
}

#endif

#ifdef USE_BAG_LOAD

rosbag::Bag bag;
rosbag::View view_full;
rosbag::View view;
std::vector<std::string> topics;
#endif
std::string imu_topic, cam0_topic, cam1_topic;
std::string Save_Path;

long long id = 0;
bool is_stop = false;

void command()
{
    while (1)
    {
	char c = getchar();
	if (c == 'q')
	{
	    is_stop = true;
	}

	std::chrono::milliseconds dura(5);
	std::this_thread::sleep_for(dura);
    }
}

class ImuGrabber
{
  public:
    ImuGrabber()
    {
    };
    void GrabImu(const sensor_msgs::ImuConstPtr &imu_msg);
    std::queue<sensor_msgs::ImuConstPtr> imuBUf;
    std::mutex mBufMutex;
};

class ImageGrabber
{
  public:
    ImageGrabber(hso::System *Hso) : mHso(Hso)
    {
    }
    void GrabImage(const sensor_msgs::ImageConstPtr &msg);
    cv::Mat GetImage(const sensor_msgs::ImageConstPtr &msg);
    void saveResult();
    hso::System *mHso;

    std::queue<sensor_msgs::ImageConstPtr> img0Buf;
    std::mutex mBuf0Mutex;

    void run();
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "Mono");
    ros::NodeHandle node_handle;
    ros::start();
    std::cout << "argc: " << argc << std::endl;
    // if (argc != 7)
    // {
    // std::cerr << std::endl << "Usage: rosrun HSO Mono path_to_vocabulary path_to_settings" << std::endl;
    // ros::shutdown();
    // return 1;
    // }

    /*    /home/wzh/Music/HSO/ros/devel/lib/hso_ros/Mono  argv[0]
	calib=../test/cameras/euroc.txt 		 argv[1]
		times=../test/timestamp/MH05.txt	argv[2]
			start=0				argv[3]
				end=22000		argv[4]
					name=MH05_0	argv[5]
						src/hso_ros/config/bag_load.yaml argv[6]
						*/



//读取加载的包的参数
    cv::FileStorage fsSettings(argv[argc - 1], cv::FileStorage::READ);
    if (!fsSettings.isOpened())
    {
	std::cerr << "ERROR: Wrong path to bag settings" << std::endl;
	return -1;
    }

    fsSettings["Save_Path"] >> Save_Path;
    std::cout << "result save path: " << Save_Path << std::endl;

    std::string mbag_name, mcalib_file, mtimestamp_file, moutput_name;
    fsSettings["bag_name"] >> mbag_name;
    fsSettings["calib_file"] >> mcalib_file;
    fsSettings["timestamp_file"] >> mtimestamp_file;
    fsSettings["output_name"] >> moutput_name;

    mtimestamp_file = mtimestamp_file + mbag_name + ".txt";
    moutput_name = moutput_name + mbag_name + "_ros" + "_0";

#ifdef USE_BAG_LOAD
    std::string bag_path;
    double bag_start = 0.0;

    fsSettings["BAG_PATH"] >> bag_path;
    bag_path = bag_path + mbag_name + ".bag";
    bag_start = fsSettings["BAG_START"];
#endif

    if (fsSettings["cam0_topic"].empty())
    {
	std::cerr << " plese provide cam and imu topics' name!!!!" << std::endl;
	return -1;
    }
    else
    {
	fsSettings["imu_topic"] >> imu_topic;
	fsSettings["cam0_topic"] >> cam0_topic;
	std::cout << "imu_topic is : " << imu_topic << std::endl;
	std::cout << "cam0_topic is : " << cam0_topic << std::endl;
    }
    fsSettings.release();

    const char *input_string[] = {" ", mcalib_file.c_str(), mtimestamp_file.c_str(), moutput_name.c_str()};
    int length = sizeof(input_string) / sizeof(char *);
    hso::System HSO(length + 1, input_string);
    ImageGrabber igb(&HSO);
    ImuGrabber imugb;

    std::thread keyboard = std::thread(command);

#ifdef USE_BAG_LOAD

    bag.open(bag_path, rosbag::bagmode::Read);
    view_full.addQuery(bag);
    ros::Time time_start = view_full.getBeginTime();
    time_start += ros::Duration(bag_start);
    ros::Time time_end = view_full.getEndTime();

    std::cout << "time_start : " << time_start << std::endl;
    std::cout << "time_end : " << time_end << std::endl;

    topics.push_back(cam0_topic);
    view.addQuery(bag, rosbag::TopicQuery(topics), time_start, time_end);
    for (rosbag::View::iterator it = view.begin(); it != view.end(); ++it)
    {
	if (it->getTopic() == cam0_topic)
	{

	    igb.GrabImage(it->instantiate<sensor_msgs::Image>());

	}
	if (it->getTopic() == imu_topic)
	{
	    imugb.GrabImu(it->instantiate<sensor_msgs::Imu>());
	}
    }
    bag.close();
#else
    ros::Subscriber _cam1_sub = node_handle.subscribe(cam0_topic, 100, &ImageGrabber::GrabImage, &igb);
#endif

    // std::thread run_thread(&ImageGrabber::run, &igb);
    igb.run();
    ros::spin();
    // ros::shutdown();  //会终结所有开放的订阅，发布，服务，调用。
    return 0;
}

void ImageGrabber::GrabImage(const sensor_msgs::ImageConstPtr &msg)
{

    mBuf0Mutex.lock();
#ifndef USE_BAG_LOAD
    if (!img0Buf.empty())
    {
	img0Buf.pop();
    }
#endif
    img0Buf.push(msg);
    mBuf0Mutex.unlock();


    // mHso->runFromRos(cv_ptr->image, cv_ptr->header.stamp.toSec(), id++);

}

cv::Mat ImageGrabber::GetImage(const sensor_msgs::ImageConstPtr &msg)
{
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
	cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO8);
    }
    catch (cv_bridge::Exception &e)
    {
	ROS_ERROR("cv_bridge exception: %s", e.what());
	// return;
    }
    cv::Mat img = cv_ptr->image.clone();
    return img;
}

void ImuGrabber::GrabImu(const sensor_msgs::ImuConstPtr &imu_msg)
{
    mBufMutex.lock();
#ifndef USE_BAG_LOAD
    if (!imuBUf.empty())
    {
	imuBUf.pop();
    }
#endif
    imuBUf.push(imu_msg);
    mBufMutex.unlock();
}

void ImageGrabber::run()
{

    while (1)
    {
	cv::Mat img;
	double tIm = 0;
	if (!img0Buf.empty())
	{
	    this->mBuf0Mutex.lock();
	    tIm = img0Buf.front()->header.stamp.toSec();
	    img = GetImage(img0Buf.front());
	    img0Buf.pop();
	    this->mBuf0Mutex.unlock();
	}
	else
	{
	    break;
	}
	if (is_stop)
	{
	    std::cout << "force quit\n";
	    exit(-1); //exit非0表示异常退出
	}
	this->mHso->runFromRos(img, id++, tIm);

    }
    this->mHso->saveResult(true, Save_Path);

}

void ImageGrabber::saveResult()
{
    // mHso->saveResult()
    // this->mHso->saveResult();
}