#include "ofApp.h"

using namespace ofxCv;
using namespace cv;

void ofApp::setup() {
    ofBackground(0);
	camera.setup(640, 480);
    gui.setup();
    gui.add(size.set("size", 80, 0, 128));
    gui.add(contrast.set("contrast", 1.5, .5, 5));
}

void ofApp::update(){
	camera.update();
	
	if(camera.isFrameNew()) {
        highpass.filter(camera, filtered, size, contrast);
        filtered.update();
	}
}

void ofApp::draw(){
    camera.draw(0, 0);
    filtered.draw(ofGetWidth() / 2, 0);
    gui.draw();
}