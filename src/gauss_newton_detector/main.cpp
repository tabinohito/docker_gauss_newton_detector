#include <iostream>
#include <opencv2/opencv.hpp>
#include <complex>
#include <string>
#include <Eigen/Core>
#include <Eigen/LU>
#include <string>
#include <charconv>

int main(int argc, char* argv[]) {
    std::string image_path = "../../image/";

    auto rad2deg = [](double radians) {
        return radians * 180.0 / M_PI;
    };

    auto deg2rad = [](double degrees) {
      return degrees * M_PI / 180.0;
    };

    // 入力変換パラメータ
    if(argc != 5 && argc != 3) {
        std::cerr << "Wrong number of input parameters" << std::endl;
        return -1;
    }
    std::string inputImage_path = std::string(argv[1]);
    std::string inputSimilarityImage_path = std::string(argv[2]);

    //初期値を適当に与える
    double estimate_theta = deg2rad(0);
    double estimate_scale = 1;

    if(argc == 3) {
        // 入力変換パラメータがない場合はデフォルト値を設定する
        double theta_ = 0.0;
        double scale_ = 1.0;
    } else {
        // 入力変換パラメータがある場合はそれを設定する
        double theta_;
        if(std::from_chars(argv[3], argv[3] + std::strlen(argv[3]), theta_).ec != std::errc()) {
            std::cerr << "Wrong theta param" << std::endl;
            return -1;
        }

        double scale_;
        if(std::from_chars(argv[4], argv[4] + std::strlen(argv[4]), scale_).ec != std::errc()) {
            std::cerr << "Wrong scale param" << std::endl;
            return -1;
        }

        estimate_theta = deg2rad(theta_);
        estimate_scale = scale_;
    }

    // 入力画像を読み込む
    cv::Mat colorImage = cv::imread(image_path + inputImage_path);
    // 相似変換済みの入力画像を読み込む
    cv::Mat colorSimilarityImage = cv::imread(image_path + inputSimilarityImage_path);
    if (colorImage.empty() || colorSimilarityImage.empty()) {
        std::cerr << "入力画像を読み込めませんでした。" << std::endl;
        return -1;
    }

    // グレースケール画像に変換する
    cv::Mat inputImage;
    cv::cvtColor(colorImage, inputImage, cv::COLOR_BGR2GRAY);
    cv::Mat inputSimilarityImage;
    cv::cvtColor(colorSimilarityImage, inputSimilarityImage, cv::COLOR_BGR2GRAY);

    // ガウシアンフィルタ
    const int filtersize = 3;
    cv::Mat gaussianInputImage;// = inputImage;
    cv::GaussianBlur(inputImage, gaussianInputImage, cv::Size(filtersize, filtersize), cv::BORDER_REFLECT);
    cv::Mat gaussianInputSimilarityImage; // = inputSimilarityImage;
    cv::GaussianBlur(inputSimilarityImage, gaussianInputSimilarityImage, cv::Size(filtersize, filtersize), cv::BORDER_REFLECT);

    //マスク
    cv::Mat differential_filter_x = (cv::Mat_<double>(3, 3) << 
        0, 0, 0,
        0, -1, 1,
        0, 0, 0);
    cv::Mat differential_filter_y = (cv::Mat_<double>(3, 3) << 
        0, 0, 0,
        0, -1, 0,
        0, 1, 0);
    
    while(1){
        double J_theta = 0; // thetaでの1回微分
        double J_theta_theta = 0.0; // thetaでの2回微分
        double J_scale = 0; // scaleでの1回微分
        double J_scale_scale = 0.0; // scaleでの2回微分
        double J_theta_scale = 0.0; // thetaとscaleの混合微分
        double J = 0; // 目的関数

        //画像I'に対するx方向の平滑微分画像I'x を計算する
        cv::Mat gradientX(gaussianInputSimilarityImage.size(), CV_64F, cv::Scalar(0)); //平滑微分画像I'x
        for (int row = 0; row < gaussianInputSimilarityImage.rows - 2; ++row) {
            for (int col = 0; col < gaussianInputSimilarityImage.cols - 2; ++col) {
                // 3x3のブロックを抽出
                cv::Mat block = gaussianInputSimilarityImage(cv::Range(col, col + 3),cv::Range(row, row + 3));
                for(int i = 0; i < block.rows; ++i) {
                    for(int j = 0; j < block.cols; ++j) {
                        gradientX.at<double>(col + 1,row + 1) += static_cast<double>(block.at<uint8_t>(j,i)) * differential_filter_x.at<double>(j, i);
                    }
                }
            }
        }

        //画像I'に対するy方向の平滑微分画像I'y を計算する
        cv::Mat gradientY(gaussianInputSimilarityImage.size(), CV_64F, cv::Scalar(0)); //平滑微分画像I'y
        for (int row = 0; row < gaussianInputSimilarityImage.rows - 2; ++row) {
            for (int col = 0; col < gaussianInputSimilarityImage.cols - 2; ++col) {
                // 3x3のブロックを抽出
                cv::Mat block = gaussianInputSimilarityImage(cv::Range(col, col + 3),cv::Range(row, row + 3));
                for(int i = 0; i < block.rows; ++i) {
                    for(int j = 0; j < block.cols; ++j) {
                        gradientY.at<double>(col + 1,row + 1) += static_cast<double>(block.at<uint8_t>(j, i)) * differential_filter_y.at<double>(j , i);
                    }
                }
            }
        }

        // 出力画像の中心座標を計算する
        cv::Point2d center(gaussianInputSimilarityImage.cols / 2.0, gaussianInputSimilarityImage.rows / 2.0);

        for (int row = 0; row < gaussianInputSimilarityImage.rows; ++row) {
            for (int col = 0; col < gaussianInputSimilarityImage.cols; ++col) {
                // 出力画像の座標を計算する
                double x = estimate_scale * ((static_cast<double>(row) - center.x) * std::cos(estimate_theta) - (static_cast<double>(col) - center.y) * std::sin(estimate_theta)) + center.x;
                double y = estimate_scale * ((static_cast<double>(row) - center.x) * std::sin(estimate_theta) + (static_cast<double>(col) - center.y) * std::cos(estimate_theta)) + center.y;

                // 出力画像の座標が入力画像の範囲内であるかをチェックする
                if (x < 0 || x >= gaussianInputSimilarityImage.rows || y < 0 || y >= gaussianInputSimilarityImage.cols) {
                    continue;
                }
                else{
                    double diff_I = static_cast<double>(gaussianInputSimilarityImage.at<uint8_t>(std::round(y),std::round(x)) - static_cast<double>(gaussianInputImage.at<uint8_t>(col,row)));
                    J += 0.5 * (diff_I * diff_I);
                    double tmp_theta = 
                    gradientX.at<double>(std::round(y),std::round(x)) * estimate_scale *(-1 * std::sin(estimate_theta) * static_cast<double>(row) - std::cos(estimate_theta) * static_cast<double>(col))
                    +
                    gradientY.at<double>(std::round(y),std::round(x)) * estimate_scale *(std::cos(estimate_theta) * static_cast<double>(row) - std::sin(estimate_theta) * static_cast<double>(col));

                    double tmp_scale =
                    gradientX.at<double>(std::round(y),std::round(x)) * (std::cos(estimate_theta) * static_cast<double>(row) - std::sin(estimate_theta) * static_cast<double>(col))
                    +
                    gradientY.at<double>(std::round(y),std::round(x)) * (std::sin(estimate_theta) * static_cast<double>(row) + std::cos(estimate_theta) * static_cast<double>(col));

                    J_theta += diff_I * tmp_theta;
                    J_theta_theta += tmp_theta * tmp_theta;
                    J_scale += diff_I * tmp_scale;
                    J_scale_scale += tmp_scale * tmp_scale;
                    J_theta_scale +=  tmp_theta * tmp_scale;
                    
                }
            }
        }

        std::cerr << "J: " << J << std::endl;

        std::cerr << "J_theta: " << J_theta << std::endl;
        std::cerr << "J_theta_theta: " << J_theta_theta << std::endl;
        std::cerr << "J_scale: " << J_scale << std::endl;
        std::cerr << "J_scale_scale: " << J_scale_scale << std::endl;
        std::cerr << "J_theta_scale: " << J_theta_scale << std::endl;

        // ヤコビアンの計算
        Eigen::Matrix2d A;
        A << J_theta_theta, J_theta_scale,
            J_theta_scale, J_scale_scale;
        Eigen::Vector2d B;
        B << J_theta, J_scale;

        Eigen::Vector2d X = -A.inverse() * B;
        std::cerr << "X: \n" << X << std::endl;

        // 収束判定
        if(std::abs(X(0)) < 1.0e-4 && std::abs(X(1)) < 1.0e-4){
            break;
        }
        else{
            estimate_theta += X(0);
            estimate_scale += X(1);
            std::cerr << "theta: " << X(0) << std::endl;
            std::cerr << "scale: " << X(1) << std::endl;
            std::cerr << "estimate_theta: " << estimate_theta  * 180 / M_PI << std::endl;
            std::cerr << "estimate_scale: " << estimate_scale << std::endl;
        }
    }

    std::cerr << "theta: " << estimate_theta * 180 / M_PI << std::endl;
    std::cerr << "scale: " << estimate_scale << std::endl;

    return 0;
}
