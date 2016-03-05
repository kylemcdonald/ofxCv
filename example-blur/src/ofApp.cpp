#include "ofApp.h"

void ofApp::setup() {
    useGaussian = false;
    cam.setup(640, 480);
    
    gui.setup();
    gui.add(useGaussian.set("Use Gaussian", false));
    gui.add(radius.set("Radius", 50, 0, 100));
}

void ofApp::update() {
    cam.update();
    if(cam.isFrameNew()) {
        ofxCv::copy(cam, img);
        if(useGaussian) {
            ofxCv::GaussianBlur(img, radius);
        } else {
            ofxCv::blur(img, radius);
        }
        img.update();
    }
}

void ofApp::draw() {
    if(img.isAllocated()) {
        img.draw(0, 0);
    }
    gui.draw();
}
