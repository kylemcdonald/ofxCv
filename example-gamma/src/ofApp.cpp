#include "ofApp.h"

using namespace ofxCv;
using namespace cv;

void adjustGamma(cv::Mat& img, float gamma = 0.5) {
    cv::Mat lookUpTable(1, 256, CV_8U);
    unsigned char* p = lookUpTable.ptr();
    for (int i = 0; i < 256; i++) {
        p[i] = saturate_cast<unsigned char>(pow(i / 255.0, gamma) * 255.0);
    }
    cv::LUT(img, lookUpTable, img);
}

void ofApp::setup() {
    cam.initGrabber(640, 480);
}

void ofApp::update() {
    cam.update();
    if(cam.isFrameNew()) {
        img = toCv(cam);
        float gamma = ofMap(mouseX, 0, ofGetWidth(), 0, 2);
        adjustGamma(img, gamma);
    }
}

void ofApp::draw() {
    ofxCv::drawMat(img, 0, 0);
}
