#include "windmill.hpp"
#include <stdlib.h>
#include <time.h>

namespace WINDMILL
{
    WindMill::WindMill(double time)
    {
    // 初始化成员变量
        cnt = 0;//计数
        direct = false;//风车移动的方向
        start_time = time;
        A = 0.785;// 振幅
        w = 1.884;// 角速度
        fai = 1.65;// 相位
        A0 = 1.305;// 另一个参数
        now_angle = 0.0;// 当前角度
        std::srand((unsigned)std::time(NULL));// 随机数种子
        int x = rand() % 200 + 400;// 随机x坐标
        int y = rand() % 100 + 420;// 随机y坐标
        R_center = cv::Point2i(x, y);// 风车中心点
    }

    //该函数在给定图像上绘制一个标记“R”，表示风车的中心点。
    void WindMill::drawR(cv::Mat &img, const cv::Point2i &center)
    {
        cv::putText(img, "R", cv::Point2i(center.x - 5, center.y + 5), cv::FONT_HERSHEY_COMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
    }

    //该函数根据角度绘制风车的扇叶，通过计算位置并使用cv::line函数绘制。
    void WindMill::drawHitFan(cv::Mat &img, const cv::Point2i &center, double angle)
    {
        // 计算中间点并绘制扇叶
        cv::Point2i mid1 = calPoint(center, angle, 40);
        cv::Point2i mid2 = calPoint(center, angle, 150);
        cv::Point2i mid3 = calPoint(center, angle, 190);
        // 绘制扇叶
        cv::line(img, mid1, mid2, cv::Scalar(0, 0, 255), 8);
        cv::line(img, calPoint(mid2, angle + 90, 30) , calPoint(mid2, angle - 90, 30), cv::Scalar(0, 0, 255), 8);
        cv::line(img, calPoint(mid3, angle + 90, 30), calPoint(mid3, angle - 90, 30), cv::Scalar(0, 0, 255), 8);
        cv::line(img, calPoint(mid2, angle + 90, 30), calPoint(mid3, angle + 90, 30), cv::Scalar(0, 0, 255), 8);
        cv::line(img, calPoint(mid2, angle - 90, 30), calPoint(mid3, angle - 90, 30), cv::Scalar(0, 0, 255), 8);
    }


    void WindMill::drawOtherFan(cv::Mat &img, const cv::Point2i &center, double angle)
    {
        cv::Point2i mid1 = calPoint(center, angle, 40);
        cv::Point2i mid2 = calPoint(center, angle, 150);
        cv::Point2i mid3 = calPoint(center, angle, 190);
        cv::Point2i mid4 = calPoint(center, angle, 200);
        cv::line(img, mid1, mid2, cv::Scalar(0, 0, 255), 8);
        cv::line(img, calPoint(mid1, angle + 90, 10), calPoint(mid1, angle - 90, 10), cv::Scalar(0, 0, 255), 3);
        cv::line(img, calPoint(mid2, angle + 90, 40), calPoint(mid1, angle + 90, 10), cv::Scalar(0, 0, 255), 3);
        cv::line(img, calPoint(mid2, angle + 90, 40), calPoint(mid4, angle + 90, 40), cv::Scalar(0, 0, 255), 3);
        cv::line(img, calPoint(mid4, angle + 90, 40), calPoint(mid4, angle - 90, 40), cv::Scalar(0, 0, 255), 3);
        cv::line(img, calPoint(mid4, angle - 90, 40), calPoint(mid2, angle - 90, 40), cv::Scalar(0, 0, 255), 3);
        cv::line(img, calPoint(mid2, angle - 90, 40), calPoint(mid1, angle - 90, 10), cv::Scalar(0, 0, 255), 3);
        cv::line(img, calPoint(mid2, angle + 90, 30), calPoint(mid2, angle - 90, 30), cv::Scalar(0, 0, 255), 3);
        cv::line(img, calPoint(mid3, angle + 90, 30), calPoint(mid3, angle - 90, 30), cv::Scalar(0, 0, 255), 3);
        cv::line(img, calPoint(mid2, angle + 90, 30), calPoint(mid3, angle + 90, 30), cv::Scalar(0, 0, 255), 3);
        cv::line(img, calPoint(mid2, angle - 90, 30), calPoint(mid3, angle - 90, 30), cv::Scalar(0, 0, 255), 3);
    }
    //获得图像
    cv::Mat WindMill::getMat(double now_time)
    {
        cv::Mat windmill = cv::Mat(720, 1080, CV_8UC3, cv::Scalar(0, 0, 0));
        cnt++;// 更新计数器
        // 根据y坐标更新风车移动方向
        if (R_center.y > 460)
            direct = false;
        if (R_center.y < 260)
            direct = true;
        // 更新风车位置
        if (direct && cnt % 50 < 5)
        {
            R_center.y += 1;
            R_center.x += 1;
        }
        if (!direct && cnt % 50 < 5)
        {
            R_center.y -= 1;
            R_center.x -= 1;
        }
        // 添加噪声
        /*
            v=Asin(wt+a)+b
        */
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine gen(seed);
        std::normal_distribution<double> noise(0, 0.2);
        // 计算当前角度
        now_angle = SumAngle(0.0, start_time, now_time - start_time) + noise(gen);
        drawR(windmill, R_center);// 绘制中心
        drawHitFan(windmill, R_center, now_angle);// 绘制扇叶
        // 绘制其他扇叶
        for (int i = 1; i < 5; i++)
        {
            drawOtherFan(windmill, R_center, now_angle + 72 * i);
        }

        return windmill;// 返回绘制的图像
    }
}