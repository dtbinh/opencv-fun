#include "opencv2/opencv.hpp"
#include <opencv2/features2d/features2d.hpp>
// only needed in Fedora
//#include <opencv2/nonfree/features2d.hpp>

//#include <unistd.h>
#include <string>
#include <iostream>

#define MIN_TRACKED_POINTS 150
#define MIN_DETECTED_POINTS 50
#define MIN_HOMOGRAPHY_POINTS 5

using namespace cv;
using namespace std;


void showResult(Mat result)
{
    namedWindow("res", 1);
    imshow("res", result);
    waitKey(0);
}

/**
 * Converts a vector of KeyPoints to vector of Points
 * @params vector<KeyPoint>& in
 * @returns vector<Point2f> out
 */
vector<Point2f> keypoints2points(const vector<KeyPoint>& in)
{
  vector<Point2f> out;
  out.reserve(in.size());
  for (size_t i = 0; i < in.size(); ++i)
  {
    out.push_back(in[i].pt);
  }

  return out;
}


/**
 * Converts a vector of Points to vector of KeyPoints
 * @params vector<Point2f>& in
 * @returns vector<KeyPoint> out
 */
vector<KeyPoint> points2keypoints(const vector<Point2f>& in)
{
  vector<KeyPoint> out;
  out.reserve(in.size());
  for (size_t i = 0; i < in.size(); ++i)
  {
    out.push_back(KeyPoint(in[i], 1));
  }

  return out;
}


/**
 * Detects Feature Points in given image
 * @params Mat frame
 * @return vector<Keypoint> keypoints
 */
vector<KeyPoint> detectFP(Mat frame) {

    int minHessian = 350;
    SurfFeatureDetector detector(minHessian);

    vector<KeyPoint> keypoints;
    detector.detect(frame, keypoints);

    return keypoints;
}

/**
 * Main function of the application
 *
 */
int main(int argc, char* argv[])
{

    // Parsing parameters:

    bool from_file = false;
    string filename;
    int camera_num;

    if (argc == 2) {
        if (string(argv[1]).compare("--camera") == 0) {
            cerr << "Nebylo zadano cislo zarizeni." << endl;
            exit(1);
        }
        else if (string(argv[1]).compare("--file") == 0) {
            cerr << "Nebyl zadan nazev souboru." << endl;
            exit(1);
        }
    }
    else if (argc == 3) {
        if (string(argv[1]).compare("--camera") == 0) {
            camera_num = atoi(argv[2]);
        }
        else if (string(argv[1]).compare("--file") == 0) {
            from_file = true;
            filename = argv[2];
        }
    }
    else {
        cerr << "Musi byt zadan zdroj. Bud prepinac --camera [cislo zarizeni]" << endl;
        cerr << "nebo prepinac --file [nazev souboru]." << endl;
        exit(1);
    }

    VideoCapture cap;

    if (from_file) {
        cout << "Nacitam soubor " << filename << "." << endl;
        cap.open(filename);
    }
    else {
        cout << "Vybran zdroj camera " << camera_num << "." << endl;
        cap.open(camera_num);
    }

    if(!cap.isOpened())  // check if we succeeded
        return -1;

    namedWindow("img",1);


    vector<uchar> status;
    vector<float> err;

    int tracked_points, homography_points;

    Mat frame, prev_frame, warped_frame;
    vector<KeyPoint> keypoints, prev_keypoints;
    vector <Point2f> points, prev_points;

    // first frame capture:
    cap >> frame;
    if (frame.empty()) {
        cerr << "No data could be read!" << endl;
        return 0;
    }
    keypoints = detectFP(frame);

    // check if the number of detected key points is enough:
    while(keypoints.size() < MIN_DETECTED_POINTS) {
        cout << "Not enough keypoints detected. (" << keypoints.size() << ")." << endl;
        cap >> frame;
        if (frame.empty()) {
            cerr << "Not enough data read!" << endl;
            return 0;
        }
        keypoints = detectFP(frame);
        //drawKeypoints(frame, keypoints, frame, Scalar(255, 234, 0), DrawMatchesFlags::DRAW_OVER_OUTIMG);
    }


    std::vector<uchar> mask;
    Mat H;
    //Mat H_orig = Mat::ones(3, 3, DataType<double>::type);
    Mat H_orig = Mat::ones(3, 3, CV_64FC1);
    Mat result = Mat::zeros(Size(frame.cols*3, frame.rows*5), CV_8UC3);

    // for every other frame:
    for(;;)
    {
        tracked_points = 0;
        homography_points = 0;

        // copy the last frame and keypoints:
        prev_frame = frame.clone();
        prev_points = keypoints2points(keypoints);

        // get a new frame from camera
        cap >> frame;
        if (frame.empty()) {
            showResult(result);
            return 0;
        }

        // perform tracking of keypoints:
        calcOpticalFlowPyrLK(prev_frame, frame, prev_points, points, status, err);

        H = findHomography(prev_points, points, CV_RANSAC, 3.0, mask);

        // count the number of successfully tracked points
        for (int i = 0; i < status.size(); i++) {
            if (status.at(i)) {
                tracked_points++;
            }
        }

        for (int i = 0; i < mask.size(); i++) {
            if (mask.at(i)) {
                homography_points++;
            }
        }
        if (homography_points < MIN_HOMOGRAPHY_POINTS) {
            cerr << "ERROR: Only " << homography_points << "points detected but " << MIN_HOMOGRAPHY_POINTS << " needed." << endl;
        }
        //cout << "Tracked/After RANSAC: " << tracked_points << "/" << homography_points << endl;

        // if the number of tracked points is less than desired:
        if (tracked_points < MIN_TRACKED_POINTS) {
            cout << "Not enough tracking points. (" << tracked_points << " found but " << MIN_TRACKED_POINTS << " needed)." << endl;
            // detect new keypoints:
            keypoints = detectFP(frame);
            // if the number of detected keypoints is less than desired:
            while(keypoints.size() < MIN_DETECTED_POINTS) {
                cout << "Not enough keypoints detected. (" << keypoints.size() << ")." << endl;
                // skip the actual frame, grab another one and detect keypoints on it
                cap >> frame;
                if (frame.empty()) {
                    showResult(result);
                    return 0;
                }
                keypoints = detectFP(frame);
                //drawKeypoints(frame, keypoints, frame, Scalar(255, 234, 0), DrawMatchesFlags::DRAW_OVER_OUTIMG);
            }
            continue;
        }

        H_orig *= H;
        //cout << H << endl;

        keypoints = points2keypoints(points);

        // draw lines representing points movement:
        for (int i = 0; i < status.size(); i++) {
            if (status.at(i)) {
                if (mask.at(i))
                    line(frame, prev_points.at(i), points.at(i), Scalar(0,255,0));
                else
                    line(frame, prev_points.at(i), points.at(i), Scalar(0,0,255));
            }
        }

        imshow("img", frame);

        if(waitKey(30) >= 0) break;
    }

    cout << H << endl;
    showResult(result);
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}

//http://www.cs.ucsb.edu/~holl/CS290I/Assignments/Assignments-3/Assignment3Mosaicing.html
