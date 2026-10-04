cmake_minimum_required(VERSION 3.10)

project(Bina)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(bina
    main.cpp
    Capture.cpp
)

target_include_directories(bina PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(bina PRIVATE
    ${OpenCV_LIBS}
)