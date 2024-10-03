#ifndef WINDMILL_H_
#define WINDMILL_H_
//引入 OpenCV 库用于图像处理，以及 <chrono> 和 <random> 用于时间和随机数的处理。
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <chrono>
#include <random>

namespace WINDMILL
{
    class WindMill
    {
    private:
        int cnt;
        bool direct;
        double A;
        double w;
        double A0;
        double fai;
        double now_angle;
        double start_time;
        cv::Point2i R_center;
        //这些函数用于绘制风车的不同部分。
        void drawR(cv::Mat &img, const cv::Point2i &center);
        void drawHitFan(cv::Mat &img, const cv::Point2i &center, double angle);
        void drawOtherFan(cv::Mat &img, const cv::Point2i &center, double angle);
        //计算给定角度和半径下的点的坐标，返回一个新的点。
        cv::Point calPoint(const cv::Point2f &center, double angle_deg, double r)
        {
            return center + cv::Point2f((float)cos(angle_deg / 180 * 3.1415926), (float)-sin(angle_deg / 180 * 3.1415926)) * r;
        }
        //计算角度的函数

        //根据时间和风车参数计算新的角度，确保角度保持在 0 到 360 度之间。
        double SumAngle(double angle_now, double t0, double dt)
        {
            double dangle = A0 * dt + (A / w) * (cos(w * t0 + 1.81) - cos(w * (t0 + dt) + 1.81));
            angle_now += dangle / 3.1415926 * 180;
            if (angle_now < 0)
            {
                angle_now = 360 + angle_now;
            }
            if (angle_now > 360)
            {
                angle_now -= 360;
            }
            return angle_now;
        }
    //构造函数：接收一个时间参数（默认为 0），用于初始化风车。
    //getMat 方法：生成并返回风车的图像矩阵，基于当前时间的计算。
    public:
        WindMill(double time = 0);
        cv::Mat getMat(double time);

    };
} // namespace WINDMILL

#endif