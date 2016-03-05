#pragma once

#include "ofMain.h"
#include "ofxCv.h"
#include "ofxGui.h"

class ofApp : public ofBaseApp {
public:
	void setup();
	void update();
	void draw();
    
    ofVideoGrabber cam;
    ofImage img;
    
    ofxPanel gui;
    ofParameter<int> radius;
    ofParameter<bool> useGaussian;
};
