#include "opencv2/opencv.hpp"
#include <opencv2/features2d/features2d.hpp>
// only needed in Fedora
#include <opencv2/nonfree/features2d.hpp>

#include <string>
#include <iostream>

#include "ImageStitcher.hpp"

#define MIN_TRACKED_POINTS 120
#define MIN_HOMOGRAPHY_POINTS 5

using namespace cv;
using namespace std;


/**
 * Shows a window containing final image
 * @params Mat result
 */
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
 * @returns vector<KeyPoint> keypoints
 */
vector<KeyPoint> detectFP(Mat frame)
{
    int minHessian = 350;
    SurfFeatureDetector detector(minHessian);
    //detector = ORB()
    //SURF detector = SURF()

    vector<KeyPoint> keypoints;
    detector.detect(frame, keypoints);

    //Mat gray = cvtColor(frame, COLOR_BGR2GRAY);

    return keypoints;
}


/**
 * Returns matches of detected keypoints
 * @params vector<KeyPoint> kp01, kp02, Mat img01, img02
 * @returns vector<DMatch> matches
 */
vector<DMatch> findMatches(Mat img01, Mat img02, vector<KeyPoint> kp01, vector<KeyPoint> kp02)
{

    // Calculate descriptors (feature vectors):
    SurfDescriptorExtractor extractor;

    Mat descriptors01, descriptors02;

    extractor.compute(img01, kp01, descriptors01);
    extractor.compute(img02, kp02, descriptors02);

    // Matching descriptor vectors using FLANN matcher:
    FlannBasedMatcher matcher;

    vector<DMatch> matches;

    matcher.match(descriptors01, descriptors02, matches);

    double max_dist = 0;
    double min_dist = 100;

    // Calculation of max and min distances between keypoints
    for (int i = 0; i < descriptors01.rows; i++) {
        double dist = matches[i].distance;
        if (dist < min_dist)
            min_dist = dist;
        if (dist > max_dist)
            max_dist = dist;
    }

    // Return only "good" matches (whose distance is less than 3*min_dist)
    vector<DMatch> good_matches;

    for (int i = 0; i < descriptors01.rows; i++) {
        if (matches[i].distance < 3*min_dist) {
            good_matches.push_back(matches[i]);
        }
    }

    return good_matches;
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
    vector<Point2f> points, prev_points;
    std::vector<Point2f> gkp01, gkp02;

    vector<DMatch> matches;

    Point2f movement;

    vector<uchar> mask;
    Mat H, H_new;

    // first frame capture:
    cap >> frame;
    if (frame.empty()) {
        cerr << "No data could be read!" << endl;
        return 0;
    }

    Mat result = Mat::zeros(Size(frame.cols*2, frame.rows*2), CV_8UC3);

    // put the first frame in the center of the final picture:
    //Rect roi(result.cols/2-frame.cols/2, result.rows/2-frame.rows/2, frame.cols, frame.rows);
    //Mat resultROI = result(roi);
    //frame.copyTo(resultROI);

    // this only works in OpenCV 2.4:
    frame.copyTo(result(Rect(result.cols/2-frame.cols/2, result.rows/2-frame.rows/2, frame.cols, frame.rows)));

    keypoints = detectFP(result);
    // clone keypoints to res_keypoints:
    vector<KeyPoint> res_keypoints(keypoints);

    // for every other frame:
    for(;;)
    {
        double tmp_x, tmp_y;

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

        // count the number of successfully tracked points:
        for (int i = 0; i < status.size(); i++) {
            if (status.at(i)) {
                tracked_points++;
            }
        }

        // count the number of inliers:
        for (int i = 0; i < mask.size(); i++) {
            if (mask.at(i)) {
                homography_points++;
            }
        }

        // count average movement:
        for (int i = 0; i< mask.size(); i++) {
            if (mask.at(i)) {
                tmp_x += points[i].x - prev_points[i].x;
                tmp_y += points[i].y - prev_points[i].y;
            }
        }
        tmp_x = tmp_x/homography_points;
        tmp_y = tmp_y/homography_points;
        movement += Point2f(tmp_x, tmp_y);

        // if the movement is larger than 1/4 of the frame size:
        if (movement.x > frame.cols/8 || movement.y > frame.rows/8) {
            cout << "Summary of movement: " << movement << endl;
            movement = Point2f(0.0, 0.0);

            res_keypoints = detectFP(result);
            keypoints = detectFP(frame);

            /* zde bude matching bodu
             */
            matches.clear(); // just to be sure
            matches = findMatches(result, frame, res_keypoints, keypoints);
            cout << "Found " << matches.size() << " matches ";

             /* vypocet homografie
              */
            // Get keypoints from the good matches

            gkp01.clear();
            gkp02.clear();
            for (int i = 0; i < matches.size(); i++) {
                gkp01.push_back(res_keypoints[matches[i].queryIdx].pt);
                gkp02.push_back(keypoints[matches[i].trainIdx].pt);
            }

            vector<uchar> gkp_mask;
            gkp_mask.clear();
            H_new = findHomography(gkp01, gkp02, CV_RANSAC, 1.0, gkp_mask);

            // count the number of inliers:
            int new_homography_points = 0;
            for (int i = 0; i < gkp_mask.size(); i++) {
                if (gkp_mask.at(i)) {
                    new_homography_points++;
                }
            }

            cout << "(" << new_homography_points << " po RANSAC)." << endl;

             /* image warping
              */
            //warpPerspective(frame, result, H_new, result.size(), INTER_LINEAR, BORDER_TRANSPARENT);
            ImageStitcher stitcher;
            Mat result_next = stitcher.stitchTwoImages(result, frame, H_new);
            result = result_next.clone();
            showResult(result);
             /* image stitching */
        }

        if (homography_points < MIN_HOMOGRAPHY_POINTS || tracked_points < MIN_TRACKED_POINTS) {
            // if the number of homography points after RANSAC is less than desired:
            if (homography_points < MIN_HOMOGRAPHY_POINTS) {
                cerr << "ERROR: Only " << homography_points << "points detected but " << MIN_HOMOGRAPHY_POINTS << " needed." << endl;
            }

            // if the number of tracked points is less than desired:
            if (tracked_points < MIN_TRACKED_POINTS) {
                cout << "Not enough tracking points. (" << tracked_points << " found but " << MIN_TRACKED_POINTS << " needed)." << endl;
            }

            // if either the number of tracking points or homography points is less than desired:
            cout << "Summary of movement: " << movement << endl;
            movement = Point2f(0.0, 0.0);

            res_keypoints = detectFP(result);
            keypoints = detectFP(frame);

            /* zde bude matching bodu
             */
            matches.clear(); // just to be sure
            matches = findMatches(result, frame, res_keypoints, keypoints);
            cout << "Found " << matches.size() << " matches ";

             /* vypocet homografie
              */
            // Get keypoints from the good matches

            gkp01.clear();
            gkp02.clear();
            for (int i = 0; i < matches.size(); i++) {
                gkp01.push_back(res_keypoints[matches[i].queryIdx].pt);
                gkp02.push_back(keypoints[matches[i].trainIdx].pt);
            }

            vector<uchar> gkp_mask;
            gkp_mask.clear();
            H_new = findHomography(gkp01, gkp02, CV_RANSAC, 1.0, gkp_mask);

            // count the number of inliers:
            int new_homography_points = 0;
            for (int i = 0; i < gkp_mask.size(); i++) {
                if (gkp_mask.at(i)) {
                    new_homography_points++;
                }
            }

            cout << "(" << new_homography_points << " po RANSAC)." << endl;

             /* image warping
              */
            warpPerspective(frame, result, H_new.inv(), result.size());
            //ImageStitcher stitcher;
            //Mat result_next = stitcher.stitchTwoImages(result, frame, H_new);
            showResult(result);
             /* image stitching */

            continue;
        }

        //warpPerspective(frame, warped_frame, H, frame.size());
        //warped_frame.copyTo(result);

        keypoints = points2keypoints(points);

        // draw lines representing points movement:
        for (int i = 0; i < status.size(); i++) {
            if (status.at(i)) {
                if (mask.at(i)) {
                    line(frame, prev_points.at(i), points.at(i), Scalar(0,255,0));
                    //line(warped_frame, prev_points.at(i), points.at(i), Scalar(0,255,0));
                }
                else {
                    line(frame, prev_points.at(i), points.at(i), Scalar(0,0,255));
                    //line(warped_frame, prev_points.at(i), points.at(i), Scalar(0,0,255));
                }
            }
        }

        imshow("img", frame);
        if(waitKey(30) >= 0) break;
    }

    showResult(result);

    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}

//http://www.cs.ucsb.edu/~holl/CS290I/Assignments/Assignments-3/Assignment3Mosaicing.html

//EMAIL:
//warpPerspective(*image, *mosaic, mosaicH, mosaicSize, INTER_LINEAR, BORDER_TRANSPARENT);
//with image being the current video frame, mosaic the bigger mosaic image, and mosaicH the Homography.
