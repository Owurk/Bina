cmake_minimum_required(VERSION 3.10)

project(Bina)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio dnn)
find_package(Threads REQUIRED)

add_executable(bina
    main.cpp
    Capture.cpp
    MotionDetector.cpp
    YoloDetector.cpp
)

target_include_directories(bina PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(bina PRIVATE
    ${OpenCV_LIBS}
    Threads::Threads
)

add_custom_command(TARGET bina POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/models
        $<TARGET_FILE_DIR:bina>/models
    COMMENT "Copying models next to executable"
)

if(MSVC)
    target_compile_options(bina PRIVATE /W4)
else()
    target_compile_options(bina PRIVATE -Wall -Wextra -Wpedantic)
endif()