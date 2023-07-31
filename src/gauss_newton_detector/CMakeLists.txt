cmake_minimum_required(VERSION 2.8.3)
project(OPEN_CV_TEST)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O2")

set(SOURCE_FILE main.cpp)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Python3 COMPONENTS NumPy)

# OpenCVのビルド環境の設定ファイルを読み込む
find_package(OpenCV REQUIRED)  

# Eigen3のビルド環境の設定ファイルを読み込む
find_package(Eigen3 REQUIRED)

# OpenCV関係のインクルードディレクトリのパスを設定
include_directories(
    ${OpenCV_INCLUDE_DIRS}
)
 
# 実行ファイル名を設定
add_executable(main
    ${SOURCE_FILE}
)

# リンクするライブラリを読み込む
target_link_libraries(main
    ${OpenCV_LIBRARIES}
    Eigen3::Eigen
    Python3::Python
    Python3::Module
    Python3::NumPy
)
