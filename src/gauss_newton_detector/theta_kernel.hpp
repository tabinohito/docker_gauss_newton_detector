#include <opencv2/opencv.hpp>
#include <cmath>

inline double rad2deg(double rad) {
    return rad * 180.0 / M_PI;
}

inline double deg2rad(double deg) {
    return deg * M_PI / 180.0;
}

// TODO: thetaの符号が逆になってるので後で直す
cv::Mat rotated_kernel_x(double theta) {
    cv::Mat k;
    if (!((0 < theta) and (theta < 2 * M_PI))) {
        theta = std::fmod(theta, 2 * M_PI);
        theta += (theta < 0) ? 2 * M_PI : 0;
    }

    bool f_inv = false;
    if (theta >= M_PI) {
        f_inv = true;
        theta -= M_PI;
    }

    if (theta < deg2rad(45)) {
        double r = (deg2rad(45) - theta) / deg2rad(45);
        k        = (cv::Mat_<double>(3, 3) << 0, 0, 1 - r, -r, 0, r, -(1 - r), 0, 0);
    } else if (theta < (deg2rad(90))) {
        double r = (deg2rad(90) - theta) / deg2rad(45);
        k        = (cv::Mat_<double>(3, 3) << 0, 1 - r, r, 0, 0, 0, -r, -(1 - r), 0);
    } else if (theta < (deg2rad(135))) {
        double r = (deg2rad(135) - theta) / deg2rad(45);
        k        = (cv::Mat_<double>(3, 3) << 1 - r, r, 0, 0, 0, 0, 0, -r, -(1 - r));
    } else if (theta < (deg2rad(180))) {
        double r = (deg2rad(180) - theta) / deg2rad(45);
        k        = (cv::Mat_<double>(3, 3) << r, 0, 0, 1 - r, 0, -(1 - r), 0, 0, -r);
    }

    if (f_inv) k *= -1;

    return k;
}

cv::Mat rotated_kernel_y(double theta) {
    return rotated_kernel_x(theta - M_PI_2);
}