#include "Capture.hpp"

#include <iostream>

Capture::Capture(int index)
    : cameraIndex(index)
{
}

bool Capture::start()
{
    cap.open(cameraIndex);

    
    if (!cap.isOpened())
    {
        std::cerr << "Could not open webcam!" << std::endl;
        return false;
    }

    while (true)
    {
        cv::Mat frame;

        
        if (!cap.read(frame))
        {
            std::cerr << "Failed to read frame!" << std::endl;
            break;
        }

        // Display the frame
        cv::imshow("Webcam", frame);

        // Exit when 'q' is pressed
        if (cv::waitKey(1) == 'q')
        {
            break;
        }
    }


    cap.release();
    cv::destroyAllWindows();

    return true;
}