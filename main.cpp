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
double amplitude_guess = 1.8;
double angular_freq_guess = 100.884; // 不更新
double phase_shift_guess = 100.65;
double offset_guess = 3.305;

double amplitude_real = 0.785; // 振幅
double angular_freq_real = 1.884; // 角速度
double phase_shift_real = 1.65; // 相位
double offset_real = 1.305; // 另一个参数

// 用于存储时间、风车位置和旋转角速度的向量
vector<double> timestamps;
vector<double> positions;
vector<double> velocities;

struct CosineModelFunctor {
    CosineModelFunctor(double timestamp, double velocity) : t(timestamp), v(velocity) {}

    template <typename T>
    bool operator()(const T* const params, T* residual) const {
        residual[0] = T(v) - (params[0] * ceres::cos(params[1] * T(t) + params[2]) + params[3]);
        return true;
    }

    private:
    const double t;
    const double v;
};

// 平滑数据的移动平均函数
vector<double> smoothData(const vector<double>& data, int window_size) {
    vector<double> smoothed_data;
    for (size_t i = 0; i < data.size(); ++i) {
        double sum = 0.0;
        int count = 0;
        for (int j = -window_size; j <= window_size; ++j) {
            if (i + j >= 0 && i + j < data.size()) {
                sum += data[i + j];
                count++;
            }
        }
        smoothed_data.push_back(sum / count);
    }
    return smoothed_data;
}

struct SinWaveFitFunctor {
    static const int InputsAtCompileTime = 4; 
    static const int ValuesAtCompileTime = Eigen::Dynamic; 

    using Scalar = double;
    using InputType = Eigen::VectorXd;
    using ValueType = Eigen::VectorXd;
    using JacobianType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    const vector<double>& times;
    const vector<double>& y_values;  // 重命名为 y_values

    SinWaveFitFunctor(const vector<double>& times, const vector<double>& values) : times(times), y_values(values) {}

    int operator()(const InputType& params, ValueType& residuals) const {
        residuals.resize(times.size());
        for (size_t i = 0; i < times.size(); ++i) {
            double A = params[0];
            double w = params[1];
            double phi = params[2];
            double A0 = params[3];
            residuals[i] = y_values[i] - (A * cos(w * times[i] + phi) + A0);
        }
        return 0;
    }

    int df(const InputType& params, JacobianType& jacobian) const {
        jacobian.resize(times.size(), 4); 
        for (size_t i = 0; i < times.size(); ++i) {
            double w = params[1];
            double A = params[0];
            double phi = params[2];
            double t = times[i];

            jacobian(i, 0) = -cos(w * t + phi); 
            jacobian(i, 1) = A * t * sin(w * t + phi); 
            jacobian(i, 2) = A * sin(w * t + phi); 
            jacobian(i, 3) = 1; 
        }
        return 0;
    }

    int values() const {
        return times.size();
    }
};


// 检查估计值与真实值是否收敛
bool checkConvergence(double estimated, double true_value) {
    return std::abs(estimated - true_value) <= 0.05 * std::abs(true_value);
}

int main(int argc, char** argv)
{
    double total_time = 0;
    int experiment_count = 1;

    while (experiment_count <= 10) {
        vector<double> time_series;
        vector<double> speed_series;

        std::chrono::milliseconds timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
        double start_time = (double)timestamp.count();
        WINDMILL::WindMill wm(timestamp.count());
        cv::Mat frame;
        Point last_center(-1, -1);
        double last_time = 0;
        Point current_center;
        double single_experiment_time = 0;
        
        while (time_series.size() < 100) {
            timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
            frame = wm.getMat((double)timestamp.count()/1000);

            // 图像处理步骤
            Mat gray_frame, blurred_frame, edge_frame;
            cvtColor(frame, gray_frame, COLOR_BGR2GRAY);
            GaussianBlur(gray_frame, blurred_frame, Size(3,3), 0); 
            Canny(blurred_frame, edge_frame, 50, 150);
            imshow("Edge Detection", edge_frame);
            vector<vector<Point>> contours;
            findContours(edge_frame, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

            for (const auto& contour : contours) {
                vector<Point> approx;
                double epsilon = 0.02 * arcLength(contour, true);
                approxPolyDP(contour, approx, epsilon, true);

                if (approx.size() == 7) {
                    Rect bounding_rect = boundingRect(approx);
                    current_center = Point(bounding_rect.x + bounding_rect.width / 2, bounding_rect.y + bounding_rect.height / 2);
                    circle(frame, current_center, 20, Scalar(0, 255, 0), 2);

                    double current_time = static_cast<double>(getTickCount()) / getTickFrequency();
                    if (last_center.x != -1) {
                        double time_diff = current_time - last_time;
                        if (time_diff > 0) {
                            double displacement = norm(current_center - last_center);
                            double velocity = displacement / time_diff;
                            time_series.push_back(current_time);
                            speed_series.push_back(velocity);
                        }
                    }
                    last_center = current_center;
                    last_time = current_time;
                }
            }
            imshow("Windmill Detection", frame);

            // 使用 Ceres 进行问题求解
            ceres::Problem optimization_problem;
            double param_estimates[4] = { amplitude_guess, angular_freq_guess, phase_shift_guess, offset_guess };
            for (size_t i = 0; i < time_series.size(); ++i) {
                optimization_problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<CosineModelFunctor, 1, 4>(
                        new CosineModelFunctor(time_series[i], speed_series[i])),
                    nullptr, param_estimates);
            }

            ceres::Solver::Options solver_options;
            solver_options.linear_solver_type = ceres::DENSE_QR;
            solver_options.minimizer_progress_to_stdout = false;

            ceres::Solver::Summary solver_summary;
            double convergence_tolerance = 0.05; 
            bool has_converged = false;
            auto fitting_start_time = std::chrono::high_resolution_clock::now();
            int max_iterations = 130; 
            int current_iteration = 0;

            while (!has_converged && current_iteration < max_iterations) {
                ceres::Solve(solver_options, &optimization_problem, &solver_summary);
                current_iteration++;

                // 检查收敛条件
                if (checkConvergence(abs(amplitude_guess), amplitude_real) && checkConvergence(abs(angular_freq_guess), angular_freq_real) && checkConvergence(abs(phase_shift_guess), phase_shift_real) && checkConvergence(abs(offset_guess), offset_real)) {
                    has_converged = true;
                }
            }

            // 记录拟合结束时间
            auto fitting_end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> fitting_duration = fitting_end_time - fitting_start_time;
            total_time += fitting_duration.count();
            single_experiment_time += fitting_duration.count();
        }
        experiment_count++;
        cout << single_experiment_time << " ";
    }
    cout << "\n";
    cout << total_time / 10;
    return 0;
}
