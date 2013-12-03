#pragma once

#include "ofMain.h"
#include "ofxCv.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>

class ofApp : public ofBaseApp{

	public:
		void setup();

        vector<ofImage> images;
        vector<vector<cv::KeyPoint> > keypoints;
        vector<cv::Mat> descriptions;
        vector<vector<cv::DMatch> > matches;
        vector<vector<cv::DMatch> > filtered_matches;
};
