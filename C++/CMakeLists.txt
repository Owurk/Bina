cmake_minimum_required(VERSION 3.10)

project(OpenCVTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(OpenCVTest
    main.cpp
    Capture.cpp
)

target_include_directories(OpenCVTest PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(OpenCVTest PRIVATE
    ${OpenCV_LIBS}
)