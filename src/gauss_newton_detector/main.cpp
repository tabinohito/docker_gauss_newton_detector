#include <iostream>
#include <opencv2/opencv.hpp>
#include <complex>
#include <string>
#include <Eigen/Core>
#include <Eigen/LU>

int main() {
    std::string image_path = "../../image/";
    // 入力画像を読み込む
    cv::Mat colorImage = cv::imread(image_path + "Lenna.png");
    // 入力画像を読み込む
    cv::Mat colorSimilarityImage = cv::imread(image_path + "Lenna_similarity.png");
    if (colorImage.empty() || colorSimilarityImage.empty()) {
        std::cout << "入力画像を読み込めませんでした。" << std::endl;
        return -1;
    }

    cv::Mat inputImage;
    cv::cvtColor(colorImage, inputImage, cv::COLOR_BGR2GRAY);

    cv::Mat inputSimilarityImage;
    cv::cvtColor(colorSimilarityImage, inputSimilarityImage, cv::COLOR_BGR2GRAY);

    // cv::imshow("Input Image", inputImage);
    // cv::imshow("Input Similarity Image", inputSimilarityImage);
    // cv::waitKey(0);

    //初期値を適当に与える
    double estimate_theta = 0;
    double estimate_scale = 1;

    while(1){
        //画像I'に対する平滑微分画像I'x を計算する
        cv::Mat differential_filter_x = (cv::Mat_<double>(3, 3) << 
            0, 0, 0,
            0, -1, 1,
            0, 0, 0);

        cv::Mat gradientX(inputSimilarityImage.size(), CV_16S, cv::Scalar(0));

        for (int row = 0; row < inputSimilarityImage.rows - 2; ++row) {
            for (int col = 0; col < inputSimilarityImage.cols - 2; ++col) {
                // 3x3のブロックを抽出
                cv::Mat block = inputSimilarityImage(cv::Range(row, row + 3), cv::Range(col, col + 3));
                for(int i = 0; i < block.rows; ++i) {
                    for(int j = 0; j < block.cols; ++j) {
                        gradientX.at<int16_t>(row, col) += block.at<int16_t>(i, j) * differential_filter_x.at<double>(i, j);
                    }
                }
            }
        }

        //画像I'に対する平滑微分画像I'y を計算する
        cv::Mat differential_filter_y = (cv::Mat_<double>(3, 3) << 
            0, 0, 0,
            0, -1, 0,
            0, 1, 0);

        cv::Mat gradientY(inputSimilarityImage.size(), CV_16S, cv::Scalar(0));

        for (int row = 0; row < inputSimilarityImage.rows - 2; ++row) {
            for (int col = 0; col < inputSimilarityImage.cols - 2; ++col) {
                // 3x3のブロックを抽出
                cv::Mat block = inputSimilarityImage(cv::Range(row, row + 3), cv::Range(col, col + 3));
                for(int i = 0; i < block.rows; ++i) {
                    for(int j = 0; j < block.cols; ++j) {
                        gradientY.at<int16_t>(row, col) += block.at<int16_t>(i, j) * differential_filter_y.at<double>(i, j);
                    }
                }
            }
        }

        double J_theta = 0; // thetaでの1回微分
        double J_theta_theta = 0.0; // thetaでの2回微分
        double J_scale = 0; // scaleでの1回微分
        double J_scale_scale = 0.0; // scaleでの2回微分
        double J_theta_scale = 0.0; // thetaとscaleの混合微分

        cv::Point2f center(inputSimilarityImage.cols / 2.0, inputSimilarityImage.rows / 2.0);
        for (int row = 0; row < inputSimilarityImage.rows; ++row) {
            for (int col = 0; col < inputSimilarityImage.cols; ++col) {
                // 出力画像の座標を計算する
                double x = (static_cast<double>(row) - center.x) * std::cos(estimate_theta) - (static_cast<double>(col) - center.y) * std::sin(estimate_theta) + center.x;
                double y = (static_cast<double>(row) - center.x) * std::sin(estimate_theta) + (static_cast<double>(col) - center.y) * std::cos(estimate_theta) + center.y;

                double diff_I = static_cast<double>(inputSimilarityImage.at<int16_t>(std::round(x),std::round(y))) - static_cast<double>(inputImage.at<int16_t>(row, col));

                double tmp_theta = 
                gradientX.at<int16_t>(std::round(x),std::round(y))*(-1 * std::sin   (estimate_theta) * static_cast<double>(row) - std::cos(estimate_theta) * static_cast<double>(col))
                +
                gradientY.at<int16_t>(std::round(x),std::round(y))*(std::cos(estimate_theta) * static_cast<double>(row) - std::sin(estimate_theta) * static_cast<double>(col));

                double tmp_scale =
                gradientX.at<int16_t>(std::round(x),std::round(y))*(std::cos(estimate_theta) * static_cast<double>(row) - std::sin(estimate_theta) * static_cast<double>(col))
                +
                gradientY.at<int16_t>(std::round(x),std::round(y))*(std::sin(estimate_theta) * static_cast<double>(row) + std::cos(estimate_theta) * static_cast<double>(col));

                J_theta += diff_I * tmp_theta;
                J_theta_theta += tmp_theta * tmp_theta;
                J_scale += diff_I * tmp_scale;
                J_scale_scale += diff_I * tmp_scale * tmp_scale;
            }
        }

        std::cout << "J_theta: " << J_theta << std::endl;
        std::cout << "J_theta_theta: " << J_theta_theta << std::endl;
        std::cout << "J_scale: " << J_scale << std::endl;
        std::cout << "J_scale_scale: " << J_scale_scale << std::endl;
        std::cout << "J_theta_scale: " << J_theta_scale << std::endl;

        Eigen::Matrix2d A;
        A << J_theta_theta, J_theta_scale,
            J_theta_scale, J_scale_scale;
        Eigen::Vector2d B;
        B << J_theta, J_scale;

        Eigen::Vector2d X = -A.inverse() * B;
        std::cout << "X: \n" << X << std::endl;

        if(std::abs(X(0)) < 1e-3 && std::abs(X(1)) < 1e-3 ){
            break;
        }
        else{
            estimate_theta += X(0);
            estimate_scale += X(1);
            std::cout << "estimate_theta: " << estimate_theta << std::endl;
            std::cout << "estimate_scale: " << estimate_scale << std::endl;
        }
    }

    std::cout << "theta: " << estimate_theta << std::endl;
    std::cout << "scale: " << estimate_scale << std::endl;


    // cv::imwrite(image_path + "gradientX.jpg", gradientX);
    // cv::imwrite(image_path + "gradientY.jpg", gradientY);
    // // cv::waitKey(0);
    // // cv::destroyAllWindows();

    return 0;
}