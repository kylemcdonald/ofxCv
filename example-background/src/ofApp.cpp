#include "ofApp.h"

using namespace ofxCv;
using namespace cv;

void ofApp::setup() {
	cam.setup(640, 480);
    
    gui.setup();
    gui.add(resetBackground.set("Reset Background", false));
    gui.add(learningTime.set("Learning Time", 30, 0, 30));
    gui.add(thresholdValue.set("Threshold Value", 10, 0, 255));
}

void ofApp::update() {
	cam.update();
    if(resetBackground) {
        background.reset();
        resetBackground = false;
    }
	if(cam.isFrameNew()) {
        background.setLearningTime(learningTime);
        background.setThresholdValue(thresholdValue);
		background.update(cam, thresholded);
		thresholded.update();
	}
}

void ofApp::draw() {
	cam.draw(0, 0);
    if(thresholded.isAllocated()) {
        thresholded.draw(640, 0);
    }
    gui.draw();
}
