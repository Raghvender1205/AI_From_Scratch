#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char **argv) {
    // Read Image
    Mat img = imread("image.jpg", 0);
    if (img.empty()) {
        cout << "Could not open the image" << endl;
        return -1;
    }

    String win_name = "Load Image";
    namedWindow(win_name);

    // Show Image
    imshow(win_name, img);
    waitKey(0);
    // DestroyWindows
    destroyWindow(win_name);
    
    return 0;
}