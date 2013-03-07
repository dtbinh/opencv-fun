#include "opencv2/opencv.hpp"
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include <unistd.h>
#include <string>
#include <iostream>

#define MIN_POINTS 75

using namespace cv;
using namespace std;

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

vector<KeyPoint> detectFP(Mat frame) {

    int minHessian = 400;
    SurfFeatureDetector detector(minHessian);

    vector<KeyPoint> keypoints;
    detector.detect(frame, keypoints);

    return keypoints;
}


int main(int argc, char* argv[])
{

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

    //VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;

    namedWindow("img",1);

    // parametry optical flow:
    //vector <Point2f> points, prev_points;
    vector<uchar> status;
    vector<float> err;
    Size winSize = Size(21,21);
    int maxLevel = 3;
    TermCriteria criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);
    int flags = 0;
    double minEigThreshold = 1e-4;
    // =========================================================================

    int tracked_points;

    Mat frame, prev_frame;
    vector<KeyPoint> keypoints, prev_keypoints;
    vector <Point2f> points, prev_points;

    // first frame
    cap >> frame;
    keypoints = detectFP(frame);

    while(keypoints.size() < 10) {
        cout << "Not enough keypoints detected. (" << keypoints.size() << ")." << endl;
        cap >> frame;
        keypoints = detectFP(frame);
        //drawKeypoints(frame, keypoints, frame, Scalar(255, 234, 0), DrawMatchesFlags::DRAW_OVER_OUTIMG);
    }


    for(;;)
    {

        tracked_points = 0;

        // backup
        prev_frame = frame.clone();
        prev_points = keypoints2points(keypoints);

        //cout << "Prev_points.size() = " << prev_points.size() << endl;

        cap >> frame; // get a new frame from camera
        //points.clear();

        calcOpticalFlowPyrLK(prev_frame, frame, prev_points, points, status, err);

        for (int i = 0; i < status.size(); i++) {
            if (status.at(i)) {
                tracked_points++;
            }
        }

        if (tracked_points < MIN_POINTS) {
            cout << "Not enough tracking points. (" << tracked_points << " found but " << MIN_POINTS << " needed)." << endl;
            keypoints = detectFP(frame);
            while(keypoints.size() < 10) {
                cout << "Not enough keypoints detected. (" << keypoints.size() << ")." << endl;
                cap >> frame;
                keypoints = detectFP(frame);
                drawKeypoints(frame, keypoints, frame, Scalar(255, 234, 0), DrawMatchesFlags::DRAW_OVER_OUTIMG);
            }
            continue;
        }

        keypoints = points2keypoints(points);


        for (int i = 0; i < status.size(); i++) {
            if (status.at(i)) {
                line(frame, prev_points.at(i), points.at(i), Scalar(0,0,255));
            }
        }

        imshow("img", frame);

        if(waitKey(30) >= 0) break;
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}

    //http://www.cs.ucsb.edu/~holl/CS290I/Assignments/Assignments-3/Assignment3Mosaicing.html

    //while ( videoReader.read(colorFrame) )
    //{
        //// convert frame to black and white
        //cvtColor(colorFrame, currentFrame, CV_RGB2GRAY, 1);

        //// compute the optical flow
        //calcOpticalFlowPyrLK(previousFrame, currentFrame, *previousPoints, *currentPoints, *status, *err);

        //// display the results
        //for ( i = 0 ; i < status->size() ; ++i )
        //{
                //if ( status->at(i) )
                //{
                        //line(colorFrame, previousPoints->at(i), currentPoints->at(i), Scalar(0, 0, 255), 2);
                //}
        //}
        //imshow("Optical flow", colorFrame);

        //// switch to next frame
        //previousFrame = currentFrame;
