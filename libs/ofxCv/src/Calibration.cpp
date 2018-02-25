#include "ofxCv/Calibration.h"
#include "ofxCv/Helpers.h"
#include "ofFileUtils.h"
#include "ofGraphics.h"
#include "ofMesh.h"
#include "ofXml.h"

namespace ofxCv {
    
    using namespace cv;
    using namespace std;
    
    void Intrinsics::setup(float focalLength, cv::Size imageSize, cv::Size2f sensorSize, cv::Point2d principalPoint) {
        float focalPixels = (focalLength / sensorSize.width) * imageSize.width;
        float fx = focalPixels; // focal length in pixels on x
        float fy = focalPixels;  // focal length in pixels on y
        float cx = imageSize.width * principalPoint.x;  // image center in pixels on x
        float cy = imageSize.height * principalPoint.y;  // image center in pixels on y
        cv::Mat cameraMatrix = (cv::Mat1d(3, 3) <<
                                fx, 0, cx,
                                0, fy, cy,
                                0, 0, 1);
        setup(cameraMatrix, imageSize, sensorSize);
    }
    void Intrinsics::setup(cv::Mat cameraMatrix, cv::Size imageSize, cv::Size2f sensorSize) {
        this->cameraMatrix = cameraMatrix;
        this->imageSize = imageSize;
        this->sensorSize = sensorSize;
        updateValues();
    }
    
    void Intrinsics::updateValues() {
        calibrationMatrixValues(cameraMatrix,
                                imageSize,
                                sensorSize.width, sensorSize.height,
                                fov.x, fov.y,
                                focalLength,
                                principalPoint, // sets principalPoint in mm
                                aspectRatio);
    }
    
    void Intrinsics::setImageSize(cv::Size imgSize) {
        imageSize = imgSize;
    }
    
    cv::Mat Intrinsics::getCameraMatrix() const {
        return cameraMatrix;
    }
    
    cv::Size Intrinsics::getImageSize() const {
        return imageSize;
    }
    
    cv::Size2f Intrinsics::getSensorSize() const {
        return sensorSize;
    }
    
    cv::Point2d Intrinsics::getFov() const {
        return fov;
    }
    
    double Intrinsics::getFocalLength() const {
        return focalLength;
    }
    
    double Intrinsics::getAspectRatio() const {
        return aspectRatio;
    }
    
    cv::Point2d Intrinsics::getPrincipalPoint() const {
        return principalPoint;
    }
    
    void Intrinsics::loadProjectionMatrix(float nearDist, float farDist, cv::Point2d viewportOffset) const {
        ofViewport(viewportOffset.x, viewportOffset.y, imageSize.width, imageSize.height);
        ofSetMatrixMode(OF_MATRIX_PROJECTION);
        ofLoadIdentityMatrix();
        float w = imageSize.width;
        float h = imageSize.height;
        float fx = cameraMatrix.at<double>(0, 0);
        float fy = cameraMatrix.at<double>(1, 1);
        float cx = principalPoint.x;
        float cy = principalPoint.y;
        
        ofMatrix4x4 frustum;
        frustum.makeFrustumMatrix(
                                  nearDist * (-cx) / fx, nearDist * (w - cx) / fx,
                                  nearDist * (cy) / fy, nearDist * (cy - h) / fy,
                                  nearDist, farDist);
        ofMultMatrix(frustum);
        
        ofSetMatrixMode(OF_MATRIX_MODELVIEW);
        ofLoadIdentityMatrix();
        
        ofMatrix4x4 lookAt;
        lookAt.makeLookAtViewMatrix(glm::vec3(0,0,0), glm::vec3(0,0,1), glm::vec3(0,-1,0));
        ofMultMatrix(lookAt);
    }
    
    Calibration::Calibration() :
    patternType(CHESSBOARD),
    patternSize(cv::Size(10, 7)), // based on Chessboard_A4.pdf, assuming world units are centimeters
    subpixelSize(cv::Size(11,11)),
    squareSize(2.5),
    reprojectionError(0),
    distCoeffs(cv::Mat::zeros(8, 1, CV_64F)),
    fillFrame(true),
    ready(false) {
        
    }
    
    void Calibration::save(const std::string& filename, bool absolute) const {
        if(!ready){
            ofLog(OF_LOG_ERROR, "Calibration::save() failed, because your calibration isn't ready yet!");
        }
        cv::FileStorage fs(ofToDataPath(filename, absolute), cv::FileStorage::WRITE);
        cv::Size imageSize = distortedIntrinsics.getImageSize();
        cv::Size sensorSize = distortedIntrinsics.getSensorSize();
        cv::Mat cameraMatrix = distortedIntrinsics.getCameraMatrix();
        fs << "cameraMatrix" << cameraMatrix;
        fs << "imageSize_width" << imageSize.width;
        fs << "imageSize_height" << imageSize.height;
        fs << "sensorSize_width" << sensorSize.width;
        fs << "sensorSize_height" << sensorSize.height;
        fs << "distCoeffs" << distCoeffs;
        fs << "reprojectionError" << reprojectionError;
        fs << "features" << "[";
        for(int i = 0; i < (int)imagePoints.size(); i++) {
            fs << imagePoints[i];
        }
        fs << "]";
    }
    
    void Calibration::load(const std::string& filename, bool absolute) {
        imagePoints.clear();
        cv::FileStorage fs(ofToDataPath(filename, absolute), cv::FileStorage::READ);
        cv::Size imageSize;
        cv::Size2f sensorSize;
        cv::Mat cameraMatrix;
        fs["cameraMatrix"] >> cameraMatrix;
        fs["imageSize_width"] >> imageSize.width;
        fs["imageSize_height"] >> imageSize.height;
        fs["sensorSize_width"] >> sensorSize.width;
        fs["sensorSize_height"] >> sensorSize.height;
        fs["distCoeffs"] >> distCoeffs;
        fs["reprojectionError"] >> reprojectionError;
        cv::FileNode features = fs["features"];
        for(cv::FileNodeIterator it = features.begin(); it != features.end(); it++) {
            std::vector<cv::Point2f> cur;
            (*it) >> cur;
            imagePoints.push_back(cur);
        }
        addedImageSize = imageSize;
        distortedIntrinsics.setup(cameraMatrix, imageSize, sensorSize);
        updateUndistortion();
        ready = true;
    }
    
    void Calibration::loadLcp(const std::string& filename, float focalLength, int imageWidth, int imageHeight, bool absolute){
        imagePoints.clear();
        
        // Load the XML
        ofXml xml;
        bool loaded = xml.load(ofToDataPath(filename, absolute));
        if(!loaded){
            ofLogError()<<"No camera profile file found at "<<filename;
            return;
        }
        
        
        // Remove the processing instruction in the top?
        
        // Find the camera profiles in the xml
        auto profiles = xml.find("//rdf:RDF/rdf:Description/photoshop:CameraProfiles/rdf:Seq");

        // Find the best matches of camera profiles
        // TODO: Not taking focus distance in account
        int bestMatchLt = -1, bestMatchGt = -1;
        float bestMatchLtVal, bestMatchGtVal;
        ofXml bestMatchLtXml, bestMatchGtXml;
        for(auto& profile : profiles) {
            int i=0;
            for(auto& child : profile.getChildren()){
                float curFocalLength = child.getChild("stCamera:FocalLength").getFloatValue();
                if(curFocalLength <= focalLength  && (bestMatchLt == -1 || curFocalLength > bestMatchLtVal)){
                    bestMatchLt = i;
                    bestMatchLtVal = curFocalLength;
                    bestMatchLtXml = child;
                }
                if(curFocalLength > focalLength && (bestMatchGt == -1 || curFocalLength < bestMatchGtVal)){
                    bestMatchGt = i;
                    bestMatchGtVal = curFocalLength;
                    bestMatchGtXml = child;
                }
                i++;
            }
        }
        
        // Get the values out of the profile
        float lcpImageWidth; // ImageWidth, pixels
        float lcpImageHeight; // ImageLength, pixels
        float cropFactor; // SensorFormatFactor, "focal length multiplier", "crop factor"
        float principalPointX = 0.5; // ImageXCenter, ratio
        float principalPointY = 0.5; // ImageYCenter, ratio
        
        float interpolation = 0;
        if(bestMatchGt != -1) {
            interpolation = ofMap(focalLength, bestMatchLtVal, bestMatchGtVal, 0, 1);
        }
        
        lcpImageWidth = bestMatchLtXml.getChild("stCamera:ImageWidth").getFloatValue();
        lcpImageHeight = bestMatchLtXml.getChild("stCamera:ImageLength").getFloatValue();
        cropFactor = bestMatchLtXml.getChild("stCamera:SensorFormatFactor").getFloatValue();
        
        float principalPointXLt = bestMatchLtXml.getChild("stCamera:PerspectiveModel").getChild("stCamera:ImageXCenter").getFloatValue();
        float principalPointYLt = bestMatchLtXml.getChild("stCamera:PerspectiveModel").getChild("stCamera:ImageYCenter").getFloatValue();
        float k1Lt = bestMatchLtXml.getChild("stCamera:PerspectiveModel").getChild("stCamera:RadialDistortParam1").getFloatValue();
        float k2Lt = bestMatchLtXml.getChild("stCamera:PerspectiveModel").getChild("stCamera:RadialDistortParam2").getFloatValue();
        float k3Lt = bestMatchLtXml.getChild("stCamera:PerspectiveModel").getChild("stCamera:RadialDistortParam3").getFloatValue();
        
        float k1 = k1Lt;
        float k2 = k2Lt;
        float k3 = k3Lt;
        
        if(bestMatchGt != -1){
            float principalPointXGt = bestMatchGtXml.getChild("stCamera:PerspectiveModel").getChild("stCamera:ImageXCenter").getFloatValue() ;
            float principalPointYGt = bestMatchGtXml.getChild("stCamera:PerspectiveModel").getChild("stCamera:ImageYCenter").getFloatValue();
            float k1Gt = bestMatchGtXml.getChild("stCamera:PerspectiveModel").getChild("stCamera:RadialDistortParam1").getFloatValue();
            float k2Gt = bestMatchGtXml.getChild("stCamera:PerspectiveModel").getChild("stCamera:RadialDistortParam2").getFloatValue();
            float k3Gt = bestMatchGtXml.getChild("stCamera:PerspectiveModel").getChild("stCamera:RadialDistortParam3").getFloatValue();
            
            k1 = k1Gt * interpolation + k1Lt * (1-interpolation);
            k2 = k2Gt * interpolation + k2Lt * (1-interpolation);
            k3 = k3Gt * interpolation + k3Lt * (1-interpolation);
        }
        
        setDistortionCoefficients(k1, k2, k3, 0);
        
        float sensorWidthMM = 35.0 / cropFactor;
        
        Intrinsics intrinsics;
        cv::Size2f sensorSize(sensorWidthMM, sensorWidthMM * lcpImageHeight / lcpImageWidth);
        
        if(imageWidth == 0) imageWidth = lcpImageWidth;
        if(imageHeight == 0) imageHeight = lcpImageHeight;
        cv::Size imageSize(imageWidth,imageHeight);
        
        intrinsics.setup(focalLength, imageSize, sensorSize);
        setIntrinsics(intrinsics);
    }
    
    
    void Calibration::setIntrinsics(Intrinsics& distortedIntrinsics){
        this->distortedIntrinsics = distortedIntrinsics;
        this->addedImageSize = distortedIntrinsics.getImageSize();
        updateUndistortion();
        this->ready = true;
    }
    void Calibration::setDistortionCoefficients(float k1, float k2, float p1, float p2, float k3, float k4, float k5, float k6) {
        distCoeffs.at<double>(0) = k1;
        distCoeffs.at<double>(1) = k2;
        distCoeffs.at<double>(2) = p1;
        distCoeffs.at<double>(3) = p2;
        distCoeffs.at<double>(4) = k3;
        distCoeffs.at<double>(5) = k4;
        distCoeffs.at<double>(6) = k5;
        distCoeffs.at<double>(7) = k6;
    }
    void Calibration::reset(){
        this->ready = false;
        this->reprojectionError = 0.0;
        this->imagePoints.clear();
        this->objectPoints.clear();
        this->perViewErrors.clear();
    }
    void Calibration::setPatternType(CalibrationPattern patternType) {
        this->patternType = patternType;
    }
    void Calibration::setPatternSize(int xCount, int yCount) {
        patternSize = cv::Size(xCount, yCount);
    }
    void Calibration::setSquareSize(float squareSize) {
        this->squareSize = squareSize;
    }
    void Calibration::setFillFrame(bool fillFrame) {
        this->fillFrame = fillFrame;
    }
    void Calibration::setSubpixelSize(int subpixelSize) {
        subpixelSize = MAX(subpixelSize,2);
        this->subpixelSize = cv::Size(subpixelSize,subpixelSize);
    }
    bool Calibration::add(cv::Mat img) {
        addedImageSize = img.size();
        
        std::vector<cv::Point2f> pointBuf;
        
        // find corners
        bool found = findBoard(img, pointBuf);
        
        if (found)
            imagePoints.push_back(pointBuf);
        else
            ofLog(OF_LOG_ERROR, "Calibration::add() failed, maybe your patternSize is wrong or the image has poor lighting?");
        return found;
    }
    bool Calibration::findBoard(cv::Mat img, std::vector<cv::Point2f>& pointBuf, bool refine) {
        bool found=false;
        if(patternType == CHESSBOARD) {
            // no CV_CALIB_CB_FAST_CHECK, because it breaks on dark images (e.g., dark IR images from kinect)
            int chessFlags = CV_CALIB_CB_ADAPTIVE_THRESH;// | CV_CALIB_CB_NORMALIZE_IMAGE;
            found = findChessboardCorners(img, patternSize, pointBuf, chessFlags);
            
            // improve corner accuracy
            if(found) {
                if(img.type() != CV_8UC1) {
                    copyGray(img, grayMat);
                } else {
                    grayMat = img;
                }
                
                if(refine) {
                    // the 11x11 dictates the smallest image space square size allowed
                    // in other words, if your smallest square is 11x11 pixels, then set this to 11x11
                    cornerSubPix(grayMat, pointBuf, subpixelSize,  cv::Size(-1,-1), cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1 ));
                }
            }
        }
#ifdef USING_OPENCV_2_3
        else {
            int flags = (patternType == CIRCLES_GRID ? CALIB_CB_SYMMETRIC_GRID : CALIB_CB_ASYMMETRIC_GRID); // + CALIB_CB_CLUSTERING
            found = findCirclesGrid(img, patternSize, pointBuf, flags);
        }
#endif
        return found;
    }
    bool Calibration::clean(float minReprojectionError) {
        int removed = 0;
        for(int i = size() - 1; i >= 0; i--) {
            if(getReprojectionError(i) > minReprojectionError) {
                objectPoints.erase(objectPoints.begin() + i);
                imagePoints.erase(imagePoints.begin() + i);
                removed++;
            }
        }
        if(size() > 0) {
            if(removed > 0) {
                return calibrate();
            } else {
                return true;
            }
        } else {
            ofLog(OF_LOG_ERROR, "Calibration::clean() removed the last object/image point pair");
            return false;
        }
    }
    bool Calibration::calibrate() {
        if(size() < 1) {
            ofLog(OF_LOG_ERROR, "Calibration::calibrate() doesn't have any image data to calibrate from.");
            if(ready) {
                ofLog(OF_LOG_ERROR, "Calibration::calibrate() doesn't need to be called after Calibration::load().");
            }
            return ready;
        }
        
        cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
        
        updateObjectPoints();
        
        int calibFlags = 0;
        float rms = calibrateCamera(objectPoints, imagePoints, addedImageSize, cameraMatrix, distCoeffs, boardRotations, boardTranslations, calibFlags);
        ofLog(OF_LOG_VERBOSE, "calibrateCamera() reports RMS error of " + ofToString(rms));
        
        ready = checkRange(cameraMatrix) && checkRange(distCoeffs);
        
        if(!ready) {
            ofLog(OF_LOG_ERROR, "Calibration::calibrate() failed to calibrate the camera");
        }
        
        distortedIntrinsics.setup(cameraMatrix, addedImageSize);
        updateReprojectionError();
        updateUndistortion();
        
        return ready;
    }
    
    bool Calibration::isReady(){
        return ready;
    }
    
    bool Calibration::calibrateFromDirectory(std::string directory) {
        ofDirectory dirList;
        ofImage cur;
        dirList.listDir(directory);
        for(std::size_t i = 0; i < dirList.size(); i++) {
            cur.load(dirList.getPath(i));
            if(!add(toCv(cur))) {
                ofLog(OF_LOG_ERROR, "Calibration::add() failed on " + dirList.getPath(i));
            }
        }
        return calibrate();
    }
    void Calibration::undistort(cv::Mat img, int interpolationMode) {
        if(img.rows != undistortMapX.rows || img.cols != undistortMapX.cols){
            ofLog(OF_LOG_ERROR, "undistort() Input image and undistort map not same size");
            return;
        }

        img.copyTo(undistortBuffer);
        undistort(undistortBuffer, img, interpolationMode);
    }
    void Calibration::undistort(cv::Mat src, cv::Mat dst, int interpolationMode) {
        remap(src, dst, undistortMapX, undistortMapY, interpolationMode);
    }
    
    glm::vec2 Calibration::undistort(glm::vec2& src) const {
        glm::vec2 dst;
        cv::Mat matSrc = cv::Mat(1, 1, CV_32FC2, &src.x);
        cv::Mat matDst = cv::Mat(1, 1, CV_32FC2, &dst.x);;
        undistortPoints(matSrc, matDst, distortedIntrinsics.getCameraMatrix(), distCoeffs);
        return dst;
    }
    
    void Calibration::undistort(std::vector<glm::vec2>& src, std::vector<glm::vec2>& dst) const {
        int n = src.size();
        dst.resize(n);
        cv::Mat matSrc = cv::Mat(n, 1, CV_32FC2, &src[0].x);
        cv::Mat matDst = cv::Mat(n, 1, CV_32FC2, &dst[0].x);
        undistortPoints(matSrc, matDst, distortedIntrinsics.getCameraMatrix(), distCoeffs);
    }
    
    bool Calibration::getTransformation(Calibration& dst, cv::Mat& rotation, cv::Mat& translation) {
        //if(imagePoints.size() == 0 || dst.imagePoints.size() == 0) {
        if(!ready) {
            ofLog(OF_LOG_ERROR, "getTransformation() requires both Calibration objects to have just been calibrated");
            return false;
        }
        if(imagePoints.size() != dst.imagePoints.size() || patternSize != dst.patternSize) {
            ofLog(OF_LOG_ERROR, "getTransformation() requires both Calibration objects to be trained simultaneously on the same board");
            return false;
        }
        cv::Mat fundamentalMatrix, essentialMatrix;
        cv::Mat cameraMatrix = distortedIntrinsics.getCameraMatrix();
        cv::Mat dstCameraMatrix = dst.getDistortedIntrinsics().getCameraMatrix();
        // uses CALIB_FIX_INTRINSIC by default
        stereoCalibrate(objectPoints,
                        imagePoints, dst.imagePoints,
                        cameraMatrix, distCoeffs,
                        dstCameraMatrix, dst.distCoeffs,
                        distortedIntrinsics.getImageSize(), rotation, translation,
                        essentialMatrix, fundamentalMatrix);
        return true;
    }
    float Calibration::getReprojectionError() const {
        return reprojectionError;
    }
    float Calibration::getReprojectionError(int i) const {
        return perViewErrors[i];
    }
    const Intrinsics& Calibration::getDistortedIntrinsics() const {
        return distortedIntrinsics;
    }
    const Intrinsics& Calibration::getUndistortedIntrinsics() const {
        return undistortedIntrinsics;
    }
    cv::Mat Calibration::getDistCoeffs() const {
        return distCoeffs;
    }
    std::size_t Calibration::size() const {
        return imagePoints.size();
    }
    cv::Size Calibration::getPatternSize() const {
        return patternSize;
    }
    float Calibration::getSquareSize() const {
        return squareSize;
    }
    void Calibration::customDraw() {
        for(int i = 0; i < size(); i++) {
            draw();
        }
    }
    void Calibration::draw() const {
        ofPushStyle();
        ofNoFill();
        ofSetColor(ofColor::red);
        for (std::size_t i = 0; i < imagePoints.size(); i++) {
           draw(i);
        }
        ofPopStyle();
    }

    void Calibration::draw(std::size_t i) const {
        for (std::size_t j = 0; j < imagePoints[i].size(); j++) {
                ofDrawCircle(toOf(imagePoints[i][j]), 5);
        }
    }
    // this won't work until undistort() is in pixel coordinates
    /*
     void Calibration::drawUndistortion() const {
     std::vector<glm::vec2> src, dst;
     cv::Point2i divisions(32, 24);
     for(int y = 0; y < divisions.y; y++) {
     for(int x = 0; x < divisions.x; x++) {
     src.push_back(glm::vec2(
					ofMap(x, -1, divisions.x, 0, addedImageSize.width),
					ofMap(y, -1, divisions.y, 0, addedImageSize.height)));
     }
     }
     undistort(src, dst);
     ofMesh mesh;
     mesh.setMode(OF_PRIMITIVE_LINES);
     for(int i = 0; i < src.size(); i++) {
     mesh.addVertex(src[i]);
     mesh.addVertex(dst[i]);
     }
     mesh.draw();
     }
     */
    void Calibration::draw3d() const {
        for(std::size_t i = 0; i < size(); i++) {
            draw3d(i);
        }
    }
    void Calibration::draw3d(std::size_t i) const {
        ofPushStyle();
        ofPushMatrix();
        ofNoFill();
        
        applyMatrix(makeMatrix(boardRotations[i], boardTranslations[i]));
        
        ofSetColor(ofColor::fromHsb(255 * i / size(), 255, 255));
        
        ofDrawBitmapString(ofToString(i), 0, 0);
        
        for(std::size_t j = 0; j < objectPoints[i].size(); j++) {
            ofPushMatrix();
            ofTranslate(toOf(objectPoints[i][j]));
            ofDrawCircle(0, 0, .5);
            ofPopMatrix();
        }
        
        ofMesh mesh;
        mesh.setMode(OF_PRIMITIVE_LINE_STRIP);
        for(std::size_t j = 0; j < objectPoints[i].size(); j++) {
            glm::vec3 cur = toOf(objectPoints[i][j]);
            mesh.addVertex(cur);
        }
        mesh.draw();
        
        ofPopMatrix();
        ofPopStyle();
    }
    void Calibration::updateObjectPoints() {
        std::vector<cv::Point3f> points = createObjectPoints(patternSize, squareSize, patternType);
        objectPoints.resize(imagePoints.size(), points);
    }
    void Calibration::updateReprojectionError() {
        std::vector<cv::Point2f> imagePoints2;
        int totalPoints = 0;
        double totalErr = 0;
        
        perViewErrors.clear();
        perViewErrors.resize(objectPoints.size());
        
        for(std::size_t i = 0; i < objectPoints.size(); i++) {
            projectPoints(cv::Mat(objectPoints[i]), boardRotations[i], boardTranslations[i], distortedIntrinsics.getCameraMatrix(), distCoeffs, imagePoints2);
            double err = norm(cv::Mat(imagePoints[i]), cv::Mat(imagePoints2), CV_L2);
            int n = objectPoints[i].size();
            perViewErrors[i] = sqrt(err * err / n);
            totalErr += err * err;
            totalPoints += n;
            ofLog(OF_LOG_VERBOSE, "view " + ofToString(i) + " has error of " + ofToString(perViewErrors[i]));
        }
        
        reprojectionError = sqrt(totalErr / totalPoints);
        
        ofLog(OF_LOG_VERBOSE, "all views have error of " + ofToString(reprojectionError));
    }
    void Calibration::updateUndistortion() {
        cv::Mat undistortedCameraMatrix = getOptimalNewCameraMatrix(distortedIntrinsics.getCameraMatrix(), distCoeffs, distortedIntrinsics.getImageSize(), fillFrame ? 0 : 1);
        initUndistortRectifyMap(distortedIntrinsics.getCameraMatrix(), distCoeffs, cv::Mat(), undistortedCameraMatrix, distortedIntrinsics.getImageSize(), CV_16SC2, undistortMapX, undistortMapY);
        undistortedIntrinsics.setup(undistortedCameraMatrix, distortedIntrinsics.getImageSize());
    }
    
    std::vector<cv::Point3f> Calibration::createObjectPoints(cv::Size patternSize, float squareSize, CalibrationPattern patternType) {
        std::vector<cv::Point3f> corners;
        switch(patternType) {
            case CHESSBOARD:
            case CIRCLES_GRID:
                for(int i = 0; i < patternSize.height; i++)
                    for(int j = 0; j < patternSize.width; j++)
                        corners.push_back(cv::Point3f(float(j * squareSize), float(i * squareSize), 0));
                break;
            case ASYMMETRIC_CIRCLES_GRID:
                for(int i = 0; i < patternSize.height; i++)
                    for(int j = 0; j < patternSize.width; j++)
                        corners.push_back(cv::Point3f(float(((2 * j) + (i % 2)) * squareSize), float(i * squareSize), 0));
                break;
        }
        return corners;
    }
}	
