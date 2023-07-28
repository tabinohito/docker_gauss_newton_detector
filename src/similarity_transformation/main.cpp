#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <charconv>

// 画像の拡大縮小関数
void resizeImage(const cv::Mat& input, cv::Mat& output, double scale)
{
    int newWidth = static_cast<int>(input.cols * scale);
    int newHeight = static_cast<int>(input.rows * scale);

    output.create(newHeight, newWidth, input.type());

    for (int y = 0; y < newHeight; ++y)
    {
        for (int x = 0; x < newWidth; ++x)
        {
            int srcX = static_cast<int>(x / scale);
            int srcY = static_cast<int>(y / scale);

            output.at<cv::Vec3b>(y, x) = input.at<cv::Vec3b>(srcY, srcX);
        }
    }
}

// 画像の回転関数
void rotateImage(const cv::Mat& input, cv::Mat& output, double angle)
{
    double radianAngle = angle * CV_PI / 180.0;
    double centerX = input.cols / 2.0;
    double centerY = input.rows / 2.0;

    double cosTheta = std::cos(radianAngle);
    double sinTheta = std::sin(radianAngle);

    output.create(input.rows, input.cols, input.type());

    for (int y = 0; y < input.rows; ++y)
    {
        for (int x = 0; x < input.cols; ++x)
        {
            double newX = (x - centerX) * cosTheta - (y - centerY) * sinTheta + centerX;
            double newY = (x - centerX) * sinTheta + (y - centerY) * cosTheta + centerY;

            if (newX >= 0 && newX < input.cols && newY >= 0 && newY < input.rows)
            {
                output.at<cv::Vec3b>(y, x) = input.at<cv::Vec3b>(newY, newX);
            }
            else
            {
                output.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0); // 黒色で埋める
            }
        }
    }
}

int main(int argc, char* argv[])
{
    //入力画像を読み込む
    cv::Mat inputImage = cv::imread("../../image/Sample.png");
    if (inputImage.empty()) {
        std::cerr << "入力画像を読み込めませんでした。" << std::endl;
        return -1;
    }

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

    std::cout << "Theta is " << theta << std::endl;
    std::cout << "Scale is " << scale << std::endl;

    // 3. アフィン変換行列を計算する
    cv::Point2f center(inputImage.cols / 2.0, inputImage.rows / 2.0);
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, (-1) * theta, scale);

    // 4. 変換を行う
    cv::Mat outputImage;
    cv::warpAffine(inputImage, outputImage, rotationMatrix, inputImage.size(),cv::INTER_NEAREST);

    std::cout << "入力画像のサイズ: " << inputImage.rows << " : " << inputImage.cols << std::endl;
    std::cout << "出力画像のサイズ: " << outputImage.rows << " : " << outputImage.cols << std::endl;

    // imgの表示
    cv::imshow("img", outputImage);


    // cv::Mat rerotationMatrix = cv::getRotationMatrix2D(center, -theta, 1/scale);
    // cv::Mat resized_rotate_Image;
    // cv::warpAffine(outputImage, resized_rotate_Image, rerotationMatrix, outputImage.size());

    // cv::Mat resized_rotate_Image;
    // rotateImage(outputImage, resized_rotate_Image, theta);
    // resizeImage(resized_rotate_Image, resized_rotate_Image, 1 / scale);

    // 画像の表示
    // cv::imshow("Resized and rotated image", resized_rotate_Image);

    // 画像を保存する
    cv::imwrite("../../image/Sample_Similarity.png", outputImage);

    // キーが押されるまで待機
    cv::waitKey(0);

    return 0;
}