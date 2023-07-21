#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <charconv>

int main(int argc, char* argv[])
{
    //入力画像を読み込む
    cv::Mat inputImage = cv::imread("../../image/Lenna.png");
    if (inputImage.empty()) {
        std::cerr << "入力画像を読み込めませんでした。" << std::endl;
        return -1;
    }

    std::cout << "入力画像のサイズ: " << inputImage.rows << " : " << inputImage.cols << std::endl;

    // 入力変換パラメータ
    if(argc != 3) {
        std::cerr << "Wrong number of input parameters" << std::endl;
        return -1;
    }

    double theta;
    if(std::from_chars(argv[1], argv[1] + std::strlen(argv[1]), theta).ec != std::errc()) {
        std::cerr << "Wrong theta param" << std::endl;
        return -1;
    }

    double scale;
    if(std::from_chars(argv[2], argv[2] + std::strlen(argv[2]), scale).ec != std::errc()) {
        std::cerr << "Wrong scale param" << std::endl;
        return -1;
    }

    std::cout << "Theta is " << theta << " deg" << std::endl;
    std::cout << "Scale is " << scale << std::endl;

    // 3. アフィン変換行列を計算する
    cv::Point2f center(inputImage.cols / 2.0, inputImage.rows / 2.0); // 回転中心を画像中心に指定
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, theta, scale); // 回転行列を結果

    // 4. 変換を行う
    cv::Mat outputImage;
    cv::warpAffine(inputImage, outputImage, rotationMatrix, inputImage.size());

    // imgの表示
    cv::imshow("img", outputImage);

    // キーが押されるまで待機
    cv::waitKey(0);

    return 0;
}