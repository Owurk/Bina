cmake_minimum_required(VERSION 3.10)

project(Bina)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)

add_executable(bina
    main.cpp
    Capture.cpp
    MotionDetector.cpp
)

target_include_directories(bina PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(bina PRIVATE
    ${OpenCV_LIBS}
)

if(MSVC)
    target_compile_options(bina PRIVATE /W4)
else()
    target_compile_options(bina PRIVATE -Wall -Wextra -Wpedantic)
endif()