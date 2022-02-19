// OpenCV_Matching.cpp : Defines the entry point for the application.
//

#include "OpenCV_Matching.h"
#include<opencv2/opencv.hpp>


using namespace cv;
using namespace std;

Mat readImage(string name)
{
	Mat im11 = imread("../../../img/"+name, IMREAD_COLOR);
	if (im11.empty())
	{
		cout << "Can't read image" << endl;
		char c;
		cin >> c;
		exit(0);
	}
	return im11;
}



vector<Point2f> cornerHarris_myShell(Mat src)
{
    Mat src_gray;
    cvtColor(src, src_gray, COLOR_BGR2GRAY);
    int blockSize = 3;
    int apertureSize = 3;
    double k = 0.04;
    int thresh = 200;

    Mat dst = Mat::zeros(src.size(), CV_32FC1);
    cornerHarris(src_gray, dst, blockSize, apertureSize, k);
    Mat dst_norm, dst_norm_scaled;
    normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat()); // привели к диапазону 0..255 float
    convertScaleAbs(dst_norm, dst_norm_scaled); // привели к CV_8U (saturate_cast)
    
    vector<Point2f> mass;

    for (int i = 0; i < dst_norm.rows; i++)
    {
        for (int j = 0; j < dst_norm.cols; j++)
        {
            if ((int)dst_norm.at<float>(i, j) > thresh)
            {
                circle(dst_norm_scaled, Point(j, i), 5, Scalar(0), 2, 8, 0);
                mass.push_back(Point2f(i, j));
            }
        }
    }
    imshow("Corners detected", dst_norm_scaled);
    return mass;
}

int main()
{
	Mat im1 = readImage("mon1.jpg");
	Mat im2 = readImage("mon2.jpg");
	
    
    vector<Point2f> mass1, mass2;
    mass1 = cornerHarris_myShell(im1);
    mass2 = cornerHarris_myShell(im2);
    cout << mass1 << endl;
    cout << mass2 << endl;
    Mat status, errors;
    calcOpticalFlowPyrLK(im1,im2,mass1,mass2,status,errors,Size(21,21),3);
    cout << status << endl;
    cout << errors << endl;
    cout << mass2 << endl;

    Mat H;
    

    imshow("origin", im1);
    waitKey();
	system("pause");
	return 0;
}
