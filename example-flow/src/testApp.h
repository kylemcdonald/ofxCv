#pragma once

#include "ofMain.h"
#include "ofxAutoControlPanel.h"
#include "ofxCv.h"
#include "Flow.h"

using namespace ofxCv;
using namespace cv;

class testApp : public ofBaseApp{
public:
	void setup();
	void update();
	void draw();
	
	ofVideoGrabber camera;
	
	FlowFarneback farneback;
	FlowPyrLK pyrLk;
	Flow* curFlow;
	
//	ofImage prev, next;
	//cv::Mat flow;	
//	vector<cv::Point2f> prevPts, nextPts;
	
	ofxAutoControlPanel panel;
};

