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

    // 入力変換パラメータ
    if(argc != 3) {
        std::cerr << "Wrong number of input parameters" << std::endl;
        return -1;
    }
    std::string inputImage_path = std::string(argv[1]);
    std::string inputSimilarityImage_path = std::string(argv[2]);

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

    // キーポイント検出と特徴量計算
    cv::Ptr<cv::Feature2D> detector = cv::ORB::create();
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    detector->detectAndCompute(inputImage, cv::Mat(), keypoints1, descriptors1);
    detector->detectAndCompute(inputSimilarityImage, cv::Mat(), keypoints2, descriptors2);

    // 特徴量マッチング
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
    std::vector<cv::DMatch> matches;
    matcher->match(descriptors1, descriptors2, matches);

    // マッチング結果を表示
    cv::Mat match_image;
    cv::drawMatches(inputImage, keypoints1, inputSimilarityImage, keypoints2, matches, match_image);
    cv::imshow("Matches", match_image);
    cv::waitKey(0);

    // マッチング結果からスケールパラメータと角度パラメータの推定を行う
    std::vector<cv::Point2f> points1, points2;
    for (const auto &match : matches)
    {
        points1.push_back(keypoints1[match.queryIdx].pt);
        points2.push_back(keypoints2[match.trainIdx].pt);
    }

    cv::Mat homography = cv::findHomography(points1, points2, cv::RANSAC);

    // スケールパラメータと角度パラメータを取得
    double scale = sqrt(homography.at<double>(0, 0) * homography.at<double>(0, 0) +
                        homography.at<double>(1, 0) * homography.at<double>(1, 0));
    double angle = atan2(homography.at<double>(1, 0), homography.at<double>(0, 0)) * 180 / CV_PI;

    std::cout << "推定されたスケールパラメータ: " << scale << std::endl;
    std::cout << "推定された角度パラメータ: " << angle << " 度" << std::endl;

    return 0;
}
