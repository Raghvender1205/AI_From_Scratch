#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // Read Image
    Mat image = imread("D:\\AI_FROM_SCRATCH\\ComputerVision\\OpenCV_C++\\image.jpg");

    // Check for Faliure
    if (image.empty()) {
        cout << "Could not open or find the image" << endl;
        cin.get(); 
        
        return -1;
    }

    String windowName = "Load Image";
    namedWindow(WindowName);

    imshow(windowName, image); // Display Image
    waitKey(0);
    destroyAllWindows(windowName);

    return 0;
}