#include "windmill.hpp"
#include <iostream>
#include <vector>
#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>
#include <numeric>
#include <cmath>

using namespace std;

using namespace cv;

// 固定初始值
double A_est = 1.8;
double w_est = 100.884; // 不更新
double fai_est = 100.65;
double A0_est = 3.305;

double A_rea = 0.785; // 振幅
double w_rea = 1.884; // 角速度
double fai_rea = 1.65; // 相位
double A0_rea = 1.305; // 另一个参数

// 用于存储时间、风车位置和旋转角速度的向量
vector<double> timeData;
vector<double> positionData;
vector<double> angularVelocityData;



struct SinusoidalCostFunctor {
    SinusoidalCostFunctor(double t, double v) : t(t), v(v) {}

    template <typename T>
    bool operator()(const T* const params, T* residual) const {
        residual[0] = T(v) - (params[0] * ceres::cos(params[1] * T(t) + params[2]) + params[3]);
        return true;
    }

    private:
    const double t;
    const double v;
};

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

// 收敛判断
bool isConverged(double estimated, double trueValue) {
    return std::abs(estimated - trueValue) <= 0.05 * std::abs(trueValue);
}

int main(int argc, char** argv)
{
    double sum_T = 0;
    int now=1;

    while(now <= 10) {
        vector<double> t_data;
        vector<double> v_data;

        std::chrono::milliseconds t = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
        double t_start = (double)t.count();
        WINDMILL::WindMill wm(t.count());
        cv::Mat src;
        Point previous_center(-1, -1);
        double previous_time = 0;
        Point current_center;
        double eve_t=0;
        while (t_data.size() < 100) {
            t = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
            src = wm.getMat((double)t.count()/1000);

            // 图像处理
            Mat gray, blur, edges;
            cvtColor(src, gray, COLOR_BGR2GRAY);
            GaussianBlur(gray, blur, Size(3,3), 0); 
            Canny(gray, edges, 50, 150);
            imshow(" ", edges);
            vector<vector<Point>> contours;
            findContours(edges, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

            for (const auto& contour : contours) {
                vector<Point> approx;
                double epsilon = 0.02 * arcLength(contour, true);
                approxPolyDP(contour, approx, epsilon, true);

                if (approx.size() == 7) {
                    Rect rect = boundingRect(approx);
                    current_center = Point(rect.x + rect.width / 2, rect.y + rect.height / 2);
                    circle(src, current_center, 20, Scalar(0, 255, 0), 2);

                    double current_time = static_cast<double>(getTickCount()) / getTickFrequency();
                    if (previous_center.x != -1) {
                        double delta_time = current_time - previous_time;
                        if (delta_time > 0) {
                            double distance = norm(current_center - previous_center);
                            double speed = distance / delta_time;
                            t_data.push_back(current_time);
                            v_data.push_back(speed);
                        }
                    }
                    previous_center = current_center;
                    previous_time = current_time;
                }
            }
            imshow("Windmill", src);
            // Ceres问题设置
            ceres::Problem problem;
            double params[4] = { A_est, w_est, fai_est, A0_est };
            for (size_t i = 0; i < t_data.size(); ++i) {
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<SinusoidalCostFunctor, 1, 4>(
                        new SinusoidalCostFunctor(t_data[i], v_data[i])),
                    nullptr, params);
            }

            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_QR;
            options.minimizer_progress_to_stdout = false;

            ceres::Solver::Summary summary;  // 声明 summary 变量
            double tolerance = 0.05; // 5% 收敛阈值
            bool converged = false;
            auto start_time = std::chrono::high_resolution_clock::now();
            int max_iterations = 130; // 最大迭代次数
            int iteration = 0;

            while (!converged && iteration < max_iterations) {
                ceres::Solve(options, &problem, &summary);  // 现在 summary 已声明
                iteration++;

           
                // 检查收敛条件
               
                            if (isConverged(abs(A_est), A_rea) && isConverged(abs(w_est), w_rea) && isConverged(abs(fai_est), fai_rea) && isConverged(abs(A0_est), A0_rea)) {
                    converged = true;
                }
            }

            // 记录拟合结束时间
            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end_time - start_time;
            sum_T += duration.count();
            eve_t+=duration.count();
        }
        now++;
        cout<<eve_t<<" ";
    }
    cout<<"\n";
    cout << sum_T / 10 ;
    return 0;
}
