#include <iostream>
#include <opencv2/opencv.hpp>

// ガウスニュートン法で相似変換のパラメータを推定する関数
cv::Mat estimateSimilarityParameters(const std::vector<cv::Point2f>& srcPoints, const std::vector<cv::Point2f>& dstPoints) {
    cv::Mat similarityParams = cv::Mat::eye(2, 3, CV_64FC1); // 初期値は単位行列

    int maxIterations = 100; // 最大反復回数
    double epsilon = 1e-6; // 収束判定の閾値
    double lambda = 0.01; // 初期重み

    for (int iter = 0; iter < maxIterations; ++iter) {
        cv::Mat A = cv::Mat::zeros(4, 4, CV_64FC1);
        cv::Mat b = cv::Mat::zeros(4, 1, CV_64FC1);

        for (size_t i = 0; i < srcPoints.size(); ++i) {
            double x = srcPoints[i].x;
            double y = srcPoints[i].y;

            double a11 = similarityParams.at<double>(0, 0);
            double a12 = similarityParams.at<double>(0, 1);
            double a13 = similarityParams.at<double>(0, 2);
            double a21 = similarityParams.at<double>(1, 0);
            double a22 = similarityParams.at<double>(1, 1);
            double a23 = similarityParams.at<double>(1, 2);

            double u = a11 * x + a12 * y + a13;
            double v = a21 * x + a22 * y + a23;

            double dx = dstPoints[i].x - u;
            double dy = dstPoints[i].y - v;

            A.at<double>(0, 0) += x * x + y * y;
            A.at<double>(0, 1) += 0;
            A.at<double>(0, 2) += x;
            A.at<double>(0, 3) += y;

            A.at<double>(1, 0) += 0;
            A.at<double>(1, 1) += x * x + y * y;
            A.at<double>(1, 2) += -y;
            A.at<double>(1, 3) += x;

            A.at<double>(2, 0) += x;
            A.at<double>(2, 1) += -y;
            A.at<double>(2, 2) += 1;
            A.at<double>(2, 3) += 0;

            A.at<double>(3, 0) += y;
            A.at<double>(3, 1) += x;
            A.at<double>(3, 2) += 0;
            A.at<double>(3, 3) += 1;

            b.at<double>(0, 0) += (x * dx + y * dy);
            b.at<double>(1, 0) += (-y * dx + x * dy);
            b.at<double>(2, 0) += dx;
            b.at<double>(3, 0) += dy;
        }

        A.at<double>(0, 0) += lambda;
        A.at<double>(1, 1) += lambda;
        A.at<double>(2, 2) += lambda;
        A.at<double>(3, 3) += lambda;

        cv::Mat delta;
        cv::solve(A, b, delta, cv::DECOMP_CHOLESKY);

        similarityParams.at<double>(0, 0) += delta.at<double>(0, 0);
        similarityParams.at<double>(0, 1) += delta.at<double>(1, 0);
        similarityParams.at<double>(0, 2) += delta.at<double>(2, 0);
        similarityParams.at<double>(1, 0) += delta.at<double>(3, 0);
        similarityParams.at<double>(1, 1) = similarityParams.at<double>(0, 0); // スケールは回転と同じ
        similarityParams.at<double>(1, 2) = similarityParams.at<double>(0, 2); // スケールは回転と同じ

        double diff = cv::norm(delta);
        if (diff < epsilon) {
            break; // 収束判定
        }

        lambda *= 0.1; // 重みを減少させる
    }

    return similarityParams;
}

int main() {
    // 入力画像を読み込む
    cv::Mat inputImage = cv::imread("input_image.jpg");
    if (inputImage.empty()) {
        std::cout << "入力画像を読み込めませんでした。" << std::endl;
        return -1;
    }

    // 入力画像を表示して特徴点を指定する
    cv::Mat inputImageCopy = inputImage.clone();
    std::vector<cv::Point2f> srcPoints, dstPoints;
    std::cout << "画像ウィンドウが表示されました。特徴点を3箇所選択してください。" << std::endl;
    cv::imshow("Input Image", inputImageCopy);
    cv::setMouseCallback("Input Image", [](int event, int x, int y, int flags, void* userdata) {
        if (event == cv::EVENT_LBUTTONDOWN) {
            std::vector<cv::Point2f>& points = *(std::vector<cv::Point2f>*)userdata;
            points.emplace_back(x, y);
            cv::circle(inputImageCopy, cv::Point(x, y), 5, cv::Scalar(0, 0, 255), -1);
            cv::imshow("Input Image", inputImageCopy);
        }
    }, &srcPoints);

    cv::waitKey(0);
    cv::destroyAllWindows();

    // 対応する特徴点を手動で指定する必要があるため、dstPointsも手動で設定する
    // ここでは例として、srcPointsに対して画像中心を中心として45度回転と1.5倍の拡大を行った点を指定する
    double centerX = inputImage.cols / 2.0;
    double centerY = inputImage.rows / 2.0;
    double angle = 45.0; // 回転角度（度数法）
    double scale = 1.5; // スケール倍率

    for (const auto& point : srcPoints) {
        double x = centerX + (point.x - centerX) * scale * std::cos(angle * CV_PI / 180.0) - (point.y - centerY) * scale * std::sin(angle * CV_PI / 180.0);
        double y = centerY + (point.x - centerX) * scale * std::sin(angle * CV_PI / 180.0) + (point.y - centerY) * scale * std::cos(angle * CV_PI / 180.0);
        dstPoints.emplace_back(x, y);
    }

    // ガウスニュートン法で相似変換のパラメータを推定
    cv::Mat similarityParams = estimateSimilarityParameters(srcPoints, dstPoints);

    // スケールパラメータと角度のパラメータ(theta)を導出
    double scaleParameter = std::sqrt(similarityParams.at<double>(0, 0) * similarityParams.at<double>(0, 0) +
                                       similarityParams.at<double>(0, 1) * similarityParams.at<double>(0, 1));
    double theta = std::atan2(similarityParams.at<double>(1, 0), similarityParams.at<double>(0, 0)) * 180.0 / CV_PI;

    // 推定した相似変換行列を表示
    std::cout << "Estimated Similarity Parameters:" << std::endl;
    std::cout << similarityParams << std::endl;
    std::cout << "スケールパラメータ: " << scaleParameter << std::endl;
    std::cout << "角度のパラメータ (theta): " << theta << "度" << std::endl;

    return 0;
}