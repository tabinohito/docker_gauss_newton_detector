#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

int main() {
    std::string image_path = "../../image/";
    // 入力画像を読み込む
    cv::Mat colorImage = cv::imread(image_path + "Lenna.png");
    if (colorImage.empty()) {
        std::cout << "入力画像を読み込めませんでした。" << std::endl;
        return -1;
    }

    cv::Mat inputImage;
    cv::cvtColor(colorImage, inputImage, cv::COLOR_BGR2GRAY);


    cv::imshow("Input Image", inputImage);
    cv::waitKey(0);
    cv::destroyAllWindows();

    //初期値を適当に与える
    double initial_estimate_theta = 0.0;
    double initial_estimate_scale = 1.0;

    //画像I'に対する平滑微分画像I'x を計算する
    cv::Mat differential_filter_x = (cv::Mat_<double>(3, 3) << 
        0, 0, 0,
        0, -1, 1,
        0, 0, 0);

    // cv::Mat gradientX(inputImage.size(), inputImage.type(), cv::Scalar(0));
    cv::Mat gradientX(inputImage.size(), CV_16S, cv::Scalar(0));

    for (int row = 0; row < inputImage.rows - 2; ++row) {
        for (int col = 0; col < inputImage.cols - 2; ++col) {
            // 3x3のブロックを抽出
            cv::Mat block = inputImage(cv::Range(row, row + 3), cv::Range(col, col + 3));
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

    // cv::Mat gradientX(inputImage.size(), inputImage.type(), cv::Scalar(0));
    cv::Mat gradientY(inputImage.size(), CV_16S, cv::Scalar(0));

    for (int row = 0; row < inputImage.rows - 2; ++row) {
        for (int col = 0; col < inputImage.cols - 2; ++col) {
            // 3x3のブロックを抽出
            cv::Mat block = inputImage(cv::Range(row, row + 3), cv::Range(col, col + 3));
            for(int i = 0; i < block.rows; ++i) {
                for(int j = 0; j < block.cols; ++j) {
                    gradientY.at<int16_t>(row, col) += block.at<int16_t>(i, j) * differential_filter_y.at<double>(i, j);
                }
            }
        }
    }
    // cv::imshow("Input Image", gradientX);
    // cv::imshow("Input Image", gradientY);
    cv::imwrite(image_path + "gradientX.jpg", gradientX);
    cv::imwrite(image_path + "gradientY.jpg", gradientY);
    // cv::waitKey(0);
    // cv::destroyAllWindows();

    return 0;
}