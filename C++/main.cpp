#include "Capture.hpp"

int main()
{
    Capture camera(0);

    if (!camera.start())
    {
        return 1;
    }

    return 0;
}