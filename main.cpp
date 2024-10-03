#include "windmill.hpp"
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>
#include <numeric>
#include <cmath>

using namespace std;
using namespace cv;

// 固定初始值
double A_est ;
double w_est;//不更新
double fai_est;
double A0_est;

double A_rea = 0.785;// 振幅
double w_rea = 1.884;// 角速度
double fai_rea =1.65;// 相位
double A0_rea = 1.305;// 另一个参数

// 用于存储时间、风车位置和旋转角速度的向量
vector<double> timeData;
vector<double> positionData;
vector<double> angularVelocityData;

// 移动平均函数
vector<double> movingAverage(const vector<double>& data, int windowSize) {
    vector<double> averagedData;
    for (size_t i = 0; i < data.size(); ++i) {
        double sum = 0.0;
        int count = 0;
        for (int j = -windowSize; j <= windowSize; ++j) {
            if (i + j >= 0 && i + j < data.size()) {
                sum += data[i + j];
                count++;
            }
        }
        averagedData.push_back(sum / count);
    }
    return averagedData;
}

// 收敛判断
bool isConverged(double estimated, double trueValue) {
    return std::abs(estimated - trueValue) <= 0.05 * std::abs(trueValue);
}



struct CosineFunctor {
    static const int InputsAtCompileTime = 4; // 输入参数数量
    static const int ValuesAtCompileTime = Eigen::Dynamic; // 输出参数数量

    using Scalar = double;
    using InputType = Eigen::VectorXd;
    using ValueType = Eigen::VectorXd;
    using JacobianType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    const vector<double>& t;
    const vector<double>& y;

    CosineFunctor(const vector<double>& t, const vector<double>& y) : t(t), y(y) {}

    int operator()(const InputType& params, ValueType& residuals) const {
        residuals.resize(t.size());
        for (size_t i = 0; i < t.size(); ++i) {
            double A = params[0];
            double w = params[1];
            double fai = params[2];
            double A0 = params[3];
            residuals[i] = y[i] - (A * cos(w * t[i] + fai) + A0);
        }
        return 0;
    }

    int df(const InputType& params, JacobianType& jacobian) const {
        jacobian.resize(t.size(), 4); // 4 个输入参数
        for (size_t i = 0; i < t.size(); ++i) {
            double w = params[1];
            double A = params[0];
            double fai = params[2];
            double t_val = t[i];

            jacobian(i, 0) = -cos(w * t_val + fai); // 对 A 的偏导数
            jacobian(i, 1) = A * t_val * sin(w * t_val + fai); // 对 w 的偏导数
            jacobian(i, 2) = A * sin(w * t_val + fai); // 对 fai 的偏导数
            jacobian(i, 3) = 1; // 对 A0 的偏导数
        }
        return 0;
    }

    int values() const {
        return t.size();
    }
};

void fitNonLinearLeastSquares(const vector<double>& timeData, const vector<double>& angularVelocityData,
                               double& A, double& w, double& fai, double& A0) {
    Eigen::VectorXd params(4);
    params << A, w, fai, A0; // 初始参数

    CosineFunctor functor(timeData, angularVelocityData);
    Eigen::NumericalDiff<CosineFunctor> numDiff(functor);
    Eigen::LevenbergMarquardt<Eigen::NumericalDiff<CosineFunctor>, double> lm(numDiff);

    // 调整优化参数
    lm.parameters.maxfev = 1000; // 增加最大函数评估次数
    lm.parameters.xtol = 1e-8;      // 增加参数容忍度

    // 执行优化
    lm.minimize(params);
    // 执行优化后打印参数
    // cout << "Current params: A=" << params[0] << ", w=" << params[1] << ", fai=" << params[2] << ", A0=" << params[3] << endl;

    // 更新参数
    A = params[0];
    w = params[1];
    fai = params[2];
    A0 = params[3];
}


void plotAngularVelocity(const vector<double>& timeData, const vector<double>& angularVelocityData) {
    int width = 800, height = 400;
    Mat plotImage = Mat::zeros(height, width, CV_8UC3);

    // 画坐标轴
    line(plotImage, Point(50, height / 2), Point(width - 50, height / 2), Scalar(255, 255, 255), 2); // X轴
    line(plotImage, Point(50, 50), Point(50, height - 50), Scalar(255, 255, 255), 2); // Y轴

    double maxY = *max_element(angularVelocityData.begin(), angularVelocityData.end());
    double minY = *min_element(angularVelocityData.begin(), angularVelocityData.end());
    double rangeY = maxY - minY;

    // 画数据曲线
    for (size_t i = 1; i < timeData.size(); ++i) {
        int x1 = static_cast<int>((timeData[i - 1] - timeData.front()) / (timeData.back() - timeData.front()) * (width - 100)) + 50;
        int y1 = static_cast<int>(height / 2 - (angularVelocityData[i - 1] - minY) / rangeY * (height - 100) / 2);
        
        int x2 = static_cast<int>((timeData[i] - timeData.front()) / (timeData.back() - timeData.front()) * (width - 100)) + 50;
        int y2 = static_cast<int>(height / 2 - (angularVelocityData[i] - minY) / rangeY * (height - 100) / 2);

        line(plotImage, Point(x1, y1), Point(x2, y2), Scalar(0, 255, 0), 2); // 绿色曲线
    }

    // 显示图像
    imshow("Angular Velocity Plot", plotImage);
}

// 异常值剔除函数
vector<double> removeOutliers(const vector<double>& data) {
    vector<double> filteredData;
    double mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    double sq_sum = std::inner_product(data.begin(), data.end(), data.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / data.size() - mean * mean);
    double threshold = 2.0 * stdev; // 设置阈值

    for (double value : data) {
        if (std::abs(value - mean) < threshold) {
            filteredData.push_back(value);
        }
    }
    return filteredData;
}


void init()
{
     A_est = A_rea-1;
     w_est = w_rea;
     fai_est = fai_rea;
     A0_est = A0_rea+1;
     return;
}

double l_A,l_w,l_fai,l_A0;
Point hammerCenter(0, 0); // 在while外部声明并初始化


// 找到峰值的函数
vector<double> findPeaks(const vector<double>& data) {
    vector<double> peaks;
    for (size_t i = 1; i < data.size() - 1; ++i) {
        if (data[i] > data[i - 1] && data[i] > data[i + 1]) {
            peaks.push_back(data[i]);
        }
    }
    return peaks;
}

// 计算周期的函数
double calculatePeriod(const vector<double>& timeData, const vector<double>& data) {
    vector<double> peaks = findPeaks(data);
    if (peaks.size() < 2) return 0;

    double minPeriod = numeric_limits<double>::max();
    for (size_t i = 1; i < peaks.size(); ++i) {
        double timeDiff = timeData[peaks[i] * 10] - timeData[peaks[i - 1] * 10];
        minPeriod = min(minPeriod, timeDiff);
    }
    return minPeriod;
}


int main() {
    std::chrono::milliseconds t = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
    
    WINDMILL::WindMill wm(t.count());
    cv::Mat src;

    Mat hammerTemplate = imread("../image/target.png"); // 锤子模板路径
    Mat rTemplate = imread("../image/R.png"); // R图案模板路径

    if (hammerTemplate.empty() || rTemplate.empty()) {
        cout << "Could not load hammer or R image!" << endl;
        return -1;
    }

      double previousAngle = 0.0; // 存储前一帧的角度
    Point rCenter;
    bool rFound = false, hammerFound = false;
    double hh_tim=static_cast<double>(t.count()) / 1000;;

     A_est = 1.8;
     w_est = 100.884;
     fai_est = 100.65;
     A0_est = 3.305;

     double sum_T=0;
     double sta_tim,end_tim;
     int now=1;
     while(now <= 10)
     {
            sta_tim=static_cast<double>(t.count()) / 1000;
                    while (true) {
                        t = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
                        src = wm.getMat(static_cast<double>(t.count()) / 1000);
                        double tTime = static_cast<double>(t.count()) / 1000;
                        double rea_tim=(tTime-hh_tim);
                        // cout<<"tTime"<<tTime-hh_tim<<"\n";
                        Mat gray;
                        cvtColor(src, gray, COLOR_BGR2GRAY);
                        Mat blurred;
                        GaussianBlur(gray, blurred, Size(3, 3), 2);
                        Mat denoised;
                        bilateralFilter(blurred, denoised, 9, 75, 75);

                        vector<vector<Point>> contours;
                        findContours(denoised, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

                        // 识别R图案
                        Mat rResult;
                        matchTemplate(src, rTemplate, rResult, TM_CCOEFF_NORMED);
                        double rMaxVal; bool rFound = false;
                        Point rLoc;
                        minMaxLoc(rResult, nullptr, &rMaxVal, nullptr, &rLoc);

                        Point rCenter;
                        if (rMaxVal >= 0.8) {
                            rCenter = Point(rLoc.x + rTemplate.cols / 2, rLoc.y + rTemplate.rows / 2);
                            rFound = true;
                            circle(src, rCenter, 5, Scalar(0, 0, 255), -1); // 标记R的中心点
                            // cout << "R center: " << rCenter << endl; // 输出R的中心坐标
                        }

                        // 识别特殊锤子
                        hammerFound = false; // 每帧重置锤子找到标志
                        for (const auto& contour : contours) {
                            vector<Point> approx;
                            double epsilon = 0.01 * arcLength(contour, true);
                            approxPolyDP(contour, approx, epsilon, true);

                            if (approx.size() >= 6) {
                                double area = contourArea(contour);
                                if (5000 <= area && area < 7000) {
                                    polylines(src, approx, true, Scalar(0, 255, 0), 2);

                                    // 计算锤子的中心点
                                    Moments m = moments(contour);
                                hammerCenter = Point(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00));
                                    circle(src, hammerCenter, 5, Scalar(255, 255, 0), -1); // 标记锤子的中心点
                                    // cout << "Hammer center: " << hammerCenter << endl; // 输出锤子的中心坐标
                                    hammerFound = true; // 找到锤子
                                }
                            }
                        }

                    // 计算角速度并存储数据
                        if (rFound && hammerFound) {
                            // 计算角度
                            // cout << "R center: " << rCenter << "Hammer center: " << hammerCenter << endl; // 输出R的中心坐标    
                            double angleDiff = atan2(hammerCenter.y - rCenter.y, hammerCenter.x - rCenter.x);
                            if (previousAngle != 0.0) {
                                // 转过的角度
                                double angularDisplacement = angleDiff - previousAngle;
                                // 确保角度在[-π, π]范围内
                                angularDisplacement = atan2(sin(angularDisplacement), cos(angularDisplacement));
                                // 计算时间间隔
                                double deltaTime;
                                if(!timeData.empty())
                                deltaTime = tTime - timeData.back(); // 当前时间 - 上一时间
                                else
                                deltaTime = tTime;
                                if (deltaTime > 0) {
                                    // 计算角速度
                                    double angularVelocity = -angularDisplacement / deltaTime;
                                    angularVelocityData.push_back(angularVelocity);
                                    // cout<<"deltaTime="<<deltaTime<<"\n";
                                    timeData.push_back(tTime);
                                }

                            }
                            previousAngle = angleDiff; // 更新前一帧的角度
                        } 

                //   cout << "R center: " << rCenter << "Hammer center: " << hammerCenter << endl; // 输出R的中心坐标        
                    line(src, hammerCenter,rCenter, Scalar(255, 0, 255), 3); // 绘制连线

                        imshow("Windmill", src);
                        if(A_est >= A_rea +10 ||A_est <= A_rea -10 ) init();
                        if(w_est >= w_rea +10 ||w_est <= w_rea -10 ) init();
                        if(fai_est >= fai_rea +10 ||fai_est <= fai_rea -10 ) init();
                        if(A0_est >= A0_rea +10 ||A0_est <= A0_rea -10 ) init();
                        if(A0_est == l_A0) init();
                        if(A_est == l_A) init();

                        // 数据处理与拟合
                        if (angularVelocityData.size() > 10) {
                            // 异常值剔除
                            auto filteredAngularVelocityData = removeOutliers(angularVelocityData);
                            // // // 数据平滑
                            auto smoothedAngularVelocityData = movingAverage(filteredAngularVelocityData, 5); // 3点移动平均


                            // // 计算周期
                            // double period = -calculatePeriod(timeData, smoothedAngularVelocityData);
                            // cout<<period<<"\n";
                            // if (period > 0) {
                            //      w_est = 2 * M_PI / period; // 计算角频率

                            //     // 计算相位
                            //     double firstDataPoint = smoothedAngularVelocityData.front();
                            //     double timeOfFirstDataPoint = timeData.front();
                            //      fai_est = firstDataPoint - w_est * timeOfFirstDataPoint;

                            // 绘制角速度与时间的曲线
                            // plotAngularVelocity(timeData,filteredAngularVelocityData);
                            // 执行拟合
                            fitNonLinearLeastSquares(timeData, filteredAngularVelocityData, A_est, w_est, fai_est, A0_est);

                            l_A=A_est;l_A0=A0_est;

                            if (isConverged(abs(A_est), A_rea) && isConverged(w_est, w_rea) && isConverged(fai_est, fai_rea) && isConverged(A0_est, A0_rea)) {
                                // cout << "Converged!" << endl;
                                // cout << "A=" << A_est << " w=" << w_est << " fai=" << fai_est << " A0=" << A0_est << "\n";
                                break;
                            } else {
                                // cout << "Not converged yet." << endl;
                                // cout << "A=" << A_est << " w=" << w_est << " fai=" << fai_est << " A0=" << A0_est << "\n";
                            }
                            
                        }

                        waitKey(1);
                    }
                end_tim=static_cast<double>(t.count()) / 1000;
                sum_T+=end_tim-sta_tim;
                now++;
     }
        cout<<sum_T/10<<"\n";
    return 0;
}
