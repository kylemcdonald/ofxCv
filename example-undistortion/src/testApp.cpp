#include "testApp.h"

const float diffThreshold = 2.5; // maximum amount of movement
const float timeThreshold = 1; // minimum time between snapshots
const int startCleaning = 10; // start cleaning outliers after this many samples

void testApp::setup() {
	ofSetVerticalSync(true);
	cam.initGrabber(640, 480);
	
	imitate(undistorted, cam);
	imitate(previous, cam);
	imitate(diff, cam);
	
	lastTime = 0;
	active = true;
    
    selectedView = 0;
    viewOffset = 0;
    
    viewsPrincipal.resize(2);
    viewsPrincipal[0] = ofRectangle(0, 0, 640*2, 480);
    viewsPrincipal[1] = ofRectangle(640*2, 0, 640*2, 480);
    
    ofBackground(40, 40, 40);
}

void testApp::update() {
    
    calculateViewports();
    
	cam.update();
	if(cam.isFrameNew()) {		
		Mat camMat = toCv(cam);
		Mat prevMat = toCv(previous);
		Mat diffMat = toCv(diff);
		
		absdiff(prevMat, camMat, diffMat);	
		camMat.copyTo(prevMat);
		
		diffMean = mean(Mat(mean(diffMat)))[0];
		
		float curTime = ofGetElapsedTimef();
		if(active && curTime - lastTime > timeThreshold && diffMean < diffThreshold) {
			if(calibration.add(camMat)) {
				cout << "re-calibrating" << endl;
				calibration.calibrate();
				if(calibration.size() > startCleaning) {
					calibration.clean();
				}
				calibration.save("calibration.yml");
				lastTime = curTime;
			}
		}
		
		if(calibration.isReady) {
			calibration.undistort(toCv(cam), toCv(undistorted));
			undistorted.update();
		}
	}
}

void drawHighlightString(string text, int x, int y, ofColor background = ofColor::black, ofColor foreground = ofColor::white) {
	int textWidth =  10 + text.length() * 8;
	ofSetColor(background);
	ofFill();
	ofRect(x - 5, y - 12, textWidth, 20);
	ofSetColor(foreground);
	ofDrawBitmapString(text, x, y);
}

void testApp::draw() {
    
    ////////////////////////////////////
    // Draw camera + undistorted camera
    ////////////////////////////////////
    //
    ofPushView();
    ofViewport(viewsTransformed[0]);
    
	ofSetColor(255);
	cam.draw(0, 0);
	undistorted.draw(640, 0);
	
	stringstream intrinsics;
	intrinsics << "fov: " << toOf(calibration.getDistortedIntrinsics().getFov()) << " distCoeffs: " << calibration.getDistCoeffs();
    drawHighlightString("[SPACE] = toggle tracking[" + string(active? "x" : " ") + string("] ; [LEFT]/[RIGHT] = switch views"), 10, 20, ofColor::fromHex(0x00ec8c), ofColor::black);
    
	drawHighlightString(intrinsics.str(), 10, 40, ofColor::fromHex(0xffee00), ofColor(0));
	
    drawHighlightString("movement: " + ofToString(diffMean), 10, 60, ofColor::fromHex(0x00abec));
	
    drawHighlightString("reproj error: " + ofToString(calibration.getReprojectionError()) + " from " + ofToString(calibration.size()), 10, 80, ofColor::fromHex(0xec008c));
	
    for(int i = 0; i < calibration.size(); i++) {
		drawHighlightString(ofToString(i) + ": " + ofToString(calibration.getReprojectionError(i)), 10, 100 + 16 * i, ofColor::fromHex(0xec008c));
	}
	
    ///////////////////
    //draw found points
    //
    if (calibration.getImagePoints().size() > 0)
    {
        ofPushStyle();
        ofNoFill();
        ofSetLineWidth(3);
        ofEnableSmoothing();
        
        
        /////
        //draw raw points
        ofBeginShape();
        ofSetColor(255,50,200);
        //
        ofVec2f pt;
        const vector<Point2f > &lastImagePoints(calibration.getImagePoints().back());
        int nPoints = lastImagePoints.size();
        //
        for (int i=0; i<nPoints; i++)
        {
            pt.x = lastImagePoints[i].x;
            pt.y = lastImagePoints[i].y;
            
            ofCircle(pt.x, pt.y, 5);
            ofVertex(pt.x, pt.y);
        }
        ofSetColor(50,255,100);
        ofEndShape(false);
        //
        ////
        
        
        /////
        //draw undistorted points
        ofBeginShape();
        ofSetColor(255,50,200);
        //
        vector<ofVec2f > curvyPoints(nPoints);
        vector<ofVec2f > straightPoints(nPoints);
        //
        memcpy(&(curvyPoints[0].x), &(lastImagePoints[0].x), sizeof(float) * 2 * nPoints);
        //
        calibration.undistort(curvyPoints, straightPoints);
        //
        for (int i=0; i<nPoints; i++)
        {
            pt.x = straightPoints[i].x;
            pt.y = straightPoints[i].y;
            
            //offset the whole image
            pt += ofVec2f(640,0);
            
            ofCircle(pt.x, pt.y, 5);
            ofVertex(pt.x, pt.y);
        }
        ofSetColor(50,255,100);
        ofEndShape(false);
        //
        /////

        
        //clear style
        ofPopStyle();
    }
    //
    //////////////////
    
    ofPopView();
    //
    ////////////////////////////////////
    
    
    ////////////////////////////////////
    // Draw 3D scene
    ////////////////////////////////////
    //
    ofPushView();
    ofViewport(viewsTransformed[1]);
    
    drawHighlightString("ofEasyCam mouse controls (left button drag = orbit, right button drag = dolly)", 10, 20, ofColor::fromHex(0x00abec));
    easyCam.begin();
    
    ofDrawAxis(10.0f);
    calibration.draw3d();
    
    easyCam.end();
    
    ofPopView();
    //
    ////////////////////////////////////
    
}

void testApp::keyPressed(int key) {
	if(key == ' ') {
		active = !active;
	}
    
    if(key == 'l')
        calibration.load("calibration.yml");
    
    if(key == 356)
    {
        if (selectedView > 0)
            selectedView--;
    }
    
    if(key == 358)
    {
        if (selectedView < viewsPrincipal.size()-1)
            selectedView++;
    }
}

void testApp::calculateViewports() {
    vector<ofRectangle>::iterator it;
    
    viewOffset -= (viewOffset + viewsPrincipal[selectedView].x) * 0.1;
    
    if (viewsTransformed.size() != viewsPrincipal.size())
        viewsTransformed.resize(viewsPrincipal.size());
    
    for (int i=0; i < viewsPrincipal.size(); i++)
    {
        viewsTransformed[i] = viewsPrincipal[i];
        viewsTransformed[i].x += viewOffset;
    }
}
