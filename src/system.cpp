//
// Please Call me FeiZi. Thank You!
// Created by wzh on 2022/10/8.
//

#include <boost/thread.hpp>
#include <cstdio>

#include "hso/system.h"
const int G_MAX_RESOLUTION = 848 * 800;
hso::System::~System()
{
    delete vo_;
    delete cam_;
    delete viewer_;
    delete viewer_thread_;

    //TODO:这里是否需要释放 or image_reader_没有析构函数，使用的是默认析构函数
    delete image_reader_;
}
hso::System::System(int argc, const char **argv)
{
    // read calibration
    paraIni();
    for (int i = 1; i < argc; ++i)
    {
	parseArgument(argv[i]);
    }
    std::string calib_dir = g_calib_path;
    std::ifstream f_cam(calib_dir.c_str());
    if (!f_cam.good())
    {
	f_cam.close();
	printf("Camera calibration file not found, shutting down.\n");
	return;
    }

    std::string line_1;
    std::getline(f_cam, line_1);

    float ic[8];
    char camera_type[20];

    //读取相机内参 以及 畸变系数
    if (std::sscanf(line_1.c_str(), "%s %f %f %f %f %f %f %f %f",
		    camera_type, &ic[0], &ic[1], &ic[2], &ic[3], &ic[4], &ic[5], &ic[6], &ic[7]) == 9)
    {
	if (camera_type[0] == 'P' || camera_type[0] == 'p')
	{
	    std::string line_2;
	    std::getline(f_cam, line_2);
	    float wh[2] = {0};
	    assert(std::sscanf(line_2.c_str(), "%f %f", &wh[0], &wh[1]) == 2);

	    int width_i = wh[0], height_i = wh[1];
	    if (wh[0] * wh[1] > G_MAX_RESOLUTION + 0.00000001)
	    {
		double resize_rate = sqrt(wh[0] * wh[1] / G_MAX_RESOLUTION);
		width_i = int(wh[0] / resize_rate);
		height_i = int(wh[1] / resize_rate);
		resize_rate = sqrt(wh[0] * wh[1] / width_i * height_i);
		ic[0] /= resize_rate;
		ic[1] /= resize_rate;
		ic[2] /= resize_rate;
		ic[3] /= resize_rate;
	    }

	    cam_ = new hso::PinholeCamera(width_i, height_i, ic[0], ic[1], ic[2], ic[3], ic[4], ic[5], ic[6], ic[7]);

	    cout << "Camera: " << "Pinhole\t" << "Width=" << wh[0] << "\tHeight=" << wh[1] << endl;
	}
	else if (camera_type[0] == 'E' || camera_type[0] == 'e')
	{
	    std::string line_2;
	    std::getline(f_cam, line_2);
	    float wh[2] = {0};
	    assert(std::sscanf(line_2.c_str(), "%f %f", &wh[0], &wh[1]) == 2);

	    int width_i = wh[0], height_i = wh[1];
	    if (wh[0] * wh[1] > G_MAX_RESOLUTION + 0.00000001)
	    {
		double resize_rate = sqrt(wh[0] * wh[1] / G_MAX_RESOLUTION);
		width_i = int(wh[0] / resize_rate);
		height_i = int(wh[1] / resize_rate);
		resize_rate = sqrt(wh[0] * wh[1] / width_i * height_i);
		ic[0] /= resize_rate;
		ic[1] /= resize_rate;
		ic[2] /= resize_rate;
		ic[3] /= resize_rate;
	    }

	    cam_ = new hso::EquidistantCamera(width_i,
					      height_i,
					      ic[0],
					      ic[1],
					      ic[2],
					      ic[3],
					      ic[4],
					      ic[5],
					      ic[6],
					      ic[7]);

	    cout << "Camera: " << "Equidistant\t" << "Width=" << wh[0] << "\t" << "Height=" << wh[1] << endl;
	}
	else
	{
	    printf("Camera type error.\n");
	    f_cam.close();
	    return;
	}

    }
    else if (std::sscanf(line_1.c_str(), "%s %f %f %f %f %f", camera_type, &ic[0], &ic[1], &ic[2], &ic[3], &ic[4]) == 6)
    {
	assert(camera_type[0] == 'F' || camera_type[0] == 'f');

	std::string line_2;
	std::getline(f_cam, line_2);
	float wh[2] = {0};
	assert(std::sscanf(line_2.c_str(), "%f %f", &wh[0], &wh[1]) == 2);

	int width_i = wh[0], height_i = wh[1];
	if (wh[0] * wh[1] > G_MAX_RESOLUTION + 0.00000001)
	{
	    double resize_rate = sqrt(wh[0] * wh[1] / G_MAX_RESOLUTION);
	    width_i = int(wh[0] / resize_rate);
	    height_i = int(wh[1] / resize_rate);
	    resize_rate = sqrt(wh[0] * wh[1] / width_i * height_i);

	    if (ic[2] > 1 && ic[3] > 1)
	    {
		ic[0] /= resize_rate;
		ic[1] /= resize_rate;
		ic[2] /= resize_rate;
		ic[3] /= resize_rate;
	    }
	}

	std::string line_3;
	std::getline(f_cam, line_3);

	if (line_3 == "true")
	    cam_ = new hso::FOVCamera(width_i, height_i, ic[0], ic[1], ic[2], ic[3], ic[4], true);
	else
	    cam_ = new hso::FOVCamera(width_i, height_i, ic[0], ic[1], ic[2], ic[3], ic[4], false);

	cout << "Camera: " << "FOV\t" << "Width=" << wh[0] << "\t" << "Height=" << wh[1] << endl;
    }
    else
	printf("Camera file error.\n");

    f_cam.close();

    //读取图片和时间序列
    image_reader_ = new hso::ImageReader(g_image_folder, cv::Size(cam_->width(), cam_->height()), g_stamp_folder);

    // TODO:FrameHandlerMono的构造函数里会执行:初始化特征提取器 和 深度滤波器
    vo_ = new FrameHandlerMono(cam_, false);

    //implement set_start_ = true;
    vo_->start();

    //下面是可视化,可视化部分是一个分离的线程
    viewer_ = new hso::Viewer(vo_);
    viewer_thread_ = new boost::thread(&hso::Viewer::run, viewer_);
    viewer_thread_->detach();

    //选择测试数据集终点位置；如果用非ros运行该系统，那么可以设置g_start和g_end；如果使用的是ros系统，那么无需管这两个参数
    g_end = std::min(image_reader_->getNumImages(), g_end);

}
void hso::System::paraIni()
{
    g_image_folder = "";
    g_stamp_folder = "None";
    g_calib_path = "";
    g_result_name = "KeyFrameTrajectory";

    g_start = 0;
    g_end = 60000;
}
void hso::System::parseArgument(const char *arg)
{

    int option;
    char buf[1000];
    if (1 == sscanf(arg, "image=%s", buf))
    {
	g_image_folder = buf;
	printf("loading images from %s!\n", g_image_folder.c_str());
	return;
    }

    if (1 == sscanf(arg, "calib=%s", buf))
    {
	g_calib_path = buf;
	printf("loading calibration from %s!\n", g_calib_path.c_str());
	return;
    }

    if (1 == sscanf(arg, "times=%s", buf))
    {
	g_stamp_folder = buf;
	printf("loading timestamp from %s!\n", g_stamp_folder.c_str());
	return;
    }

    if (1 == sscanf(arg, "name=%s", buf))
    {
	g_result_name = buf;
	printf("set result file name =  %s!\n", g_result_name.c_str());
	return;
    }

    if (1 == sscanf(arg, "start=%d", &option))
    {
	g_start = option;
	printf("START AT %d!\n", g_start);
	return;
    }

    if (1 == sscanf(arg, "end=%d", &option))
    {
	g_end = option;
	printf("END AT %d!\n", g_end);
	return;
    }
}

//hso非ros运行
void hso::System::runFromFolder()
{

    for (int img_id = g_start; img_id < g_end; ++img_id)
    {
	//读取图片
	cv::Mat image = image_reader_->readImage(img_id);

	if (cam_->getUndistort()) cam_->undistortImage(image, image);

	//如果输入的参数有采样时间，那么image_reader_->stampValid()返回true，否则返回false
	if (image_reader_->stampValid())
	{
	    std::string time_stamp = image_reader_->readStamp(img_id);

	    // process frame
	    vo_->addImage(image, img_id, &time_stamp);
	}
	else
	    vo_->addImage(image, img_id);


	// display tracking quality
	cv::Mat tracking_img(image);
	cv::cvtColor(tracking_img, tracking_img, cv::COLOR_GRAY2RGB);
	if (vo_->lastFrame() != NULL)
	{
	    for (auto &ft: vo_->lastFrame()->fts_)
	    {
		if (ft->point == NULL) continue;

		if (ft->type == hso::Feature::EDGELET)
		    cv::rectangle(tracking_img,
				  cv::Point2f(ft->px.x() - 3, ft->px.y() - 3),
				  cv::Point2f(ft->px.x() + 3, ft->px.y() + 3),
				  cv::Scalar(0, 255, 255),
				  cv::FILLED);
		else
		    cv::rectangle(tracking_img,
				  cv::Point2f(ft->px.x() - 3, ft->px.y() - 3),
				  cv::Point2f(ft->px.x() + 3, ft->px.y() + 3),
				  cv::Scalar(0, 255, 0),
				  cv::FILLED);
	    }
	}
	cv::imshow("Tracking Image", tracking_img);
	cv::waitKey(1);
    }

    saveResult(image_reader_->stampValid(), "");
    // cv::waitKey (0);
}
void hso::System::runFromRos(const cv::Mat &im, int img_id, const double &timestamp)
{
    cv::Mat image = im.clone();
    if (cam_->getUndistort()) cam_->undistortImage(image, image);

    char buffer[100];
    sprintf(buffer, "%.10f", timestamp);
    std::string time_stamp = buffer;
    // process frame
    vo_->addImage(image, img_id, &time_stamp);


    // display tracking quality
    cv::Mat tracking_img(image);
    cv::cvtColor(tracking_img, tracking_img, cv::COLOR_GRAY2RGB);
    if (vo_->lastFrame() != NULL)
    {
	for (auto &ft: vo_->lastFrame()->fts_)
	{
	    if (ft->point == NULL) continue;

	    if (ft->type == hso::Feature::EDGELET)
		cv::rectangle(tracking_img,
			      cv::Point2f(ft->px.x() - 3, ft->px.y() - 3),
			      cv::Point2f(ft->px.x() + 3, ft->px.y() + 3),
			      cv::Scalar(0, 255, 255),
			      cv::FILLED);
	    else
		cv::rectangle(tracking_img,
			      cv::Point2f(ft->px.x() - 3, ft->px.y() - 3),
			      cv::Point2f(ft->px.x() + 3, ft->px.y() + 3),
			      cv::Scalar(0, 255, 0),
			      cv::FILLED);
	}
    }
    cv::imshow("Tracking Image", tracking_img);
    cv::waitKey(1);

    // std::cout<<"!!!"<<image_reader_->stampValid()<<"!!!\n";
    // saveResult(true);

}
void hso::System::saveResult(bool stamp_valid, string ros_path)
{
    // Trajectory
    if (ros_path.empty())
	ros_path = "./result";

    std::ofstream okt(ros_path + "/" + g_result_name + ".txt");

    for (auto it = vo_->map_.keyframes_.begin(); it != vo_->map_.keyframes_.end(); ++it)
    {
	SE3 Tinv = (*it)->T_f_w_.inverse();
	if (!stamp_valid)
	    okt << (*it)->id_ << " ";
	else
	    okt << (*it)->m_timestamp_s << " ";

	okt << Tinv.translation()[0] << " "
	    << Tinv.translation()[1] << " "
	    << Tinv.translation()[2] << " "
	    << Tinv.unit_quaternion().x() << " "
	    << Tinv.unit_quaternion().y() << " "
	    << Tinv.unit_quaternion().z() << " "
	    << Tinv.unit_quaternion().w() << endl;
    }
    okt.close();
    std::cout << "write over\n";
}

