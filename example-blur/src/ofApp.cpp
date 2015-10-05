/*
# example-blur

## Description
This example demonstrates two different blur methods (gaussian and normalized box) and a method to compute the focus amount of an image.
Focus computation is useful for objects (and especially iris) detection.
You can find more about focus computation on the appendix of [*How Iris Recognition Works* by Paul Daugman](http://www.cl.cam.ac.uk/~jgd1000/csvt.pdf) (p.29).

## Running
Console displays computed focus amount.
Use space bar to toggle between blur methods.
Use 'a' and 'z' to de- / in-crease blur amount.
See the result on focus computation.
*/

#include "ofApp.h"

void ofApp::setup() {
    useGaussian = false;
    cam.initGrabber(640, 480);
    blur=50;
}

void ofApp::update() {
    cam.update();
    if(cam.isFrameNew()) {
        ofxCv::copy(cam, img);
        if(useGaussian) {
            ofxCv::GaussianBlur(img, blur);
        } else {
            ofxCv::blur(img, blur);
        }
        img.update();

        // now we compute the high frequency quantity in the image,
        // the sharper the image is, the greater the score is
        // thus the score could be used as a focus indicator.
        cv::Mat gray, normalized, sobel;
        ofxCv::copyGray(img, gray);
        equalizeHist(gray, normalized);
        ofxCv::Sobel(normalized,sobel,-1, 1, 1);
        double score = cv::norm(cv::sum(sobel)) / sobel.total(); // need to get the norm (double) of the sum (scalar) to fix type issue
        cout << "focus : " << score << endl;
    }
}

void ofApp::draw() {
    if(img.isAllocated()) {
        img.draw(0, 0);
    }
    ofDrawBitmapStringHighlight(useGaussian ? "GaussianBlur()" : "blur()", 10, 20);
}

void ofApp::keyPressed(int key) {
    switch(key){
        case ' ':
            useGaussian = !useGaussian; break;
        case 'z':
            if(blur<255)blur++; break;
        case 'a':
            if (blur>0)blur--; break;
        default:
            break;
    }
}
