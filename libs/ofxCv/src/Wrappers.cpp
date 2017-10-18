#include "ofxCv/Wrappers.h"

namespace ofxCv {
	
	using namespace cv;
	
	void loadMat(Mat& mat, std::string filename) {
		FileStorage fs(ofToDataPath(filename), FileStorage::READ);
		fs["Mat"] >> mat;
	}
	
	void saveMat(Mat mat, std::string filename) {
		FileStorage fs(ofToDataPath(filename), FileStorage::WRITE);
		fs << "Mat" << mat;
	}
	
	void saveImage(Mat& mat, std::string filename, ofImageQualityType qualityLevel) {
		if(mat.depth() == CV_8U) {
			ofPixels pix8u;
			toOf(mat, pix8u);
			ofSaveImage(pix8u, filename, qualityLevel);
		} else if(mat.depth() == CV_16U) {
			ofShortPixels pix16u;
			toOf(mat, pix16u);
			ofSaveImage(pix16u, filename, qualityLevel);
		} else if(mat.depth() == CV_32F) {
			ofFloatPixels pix32f;
			toOf(mat, pix32f);
			ofSaveImage(pix32f, filename, qualityLevel);
		}
	}
	
	Vec3b convertColor(Vec3b color, int code) {
		Mat_<Vec3b> mat(1, 1, CV_8UC3);
		mat(0, 0) = color;
		cvtColor(mat, mat, code);
		return mat(0, 0);
	}
	
	ofColor convertColor(ofColor color, int code) {
		Vec3b cvColor(color.r, color.g, color.b);
		Vec3b result = convertColor(cvColor, code);
		return ofColor(result[0], result[1], result[2], color.a);
	}	
	
	ofPolyline convexHull(const ofPolyline& polyline) {
		std::vector<cv::Point2f> contour = toCv(polyline);
		std::vector<cv::Point2f> hull;
		convexHull(Mat(contour), hull);
		return toOf(hull);
	}
	
	// this should be replaced by c++ 2.0 api style code once available
	std::vector<cv::Vec4i> convexityDefects(const std::vector<cv::Point>& contour) {
		std::vector<int> hullIndices;
		convexHull(Mat(contour), hullIndices, false, false);
		std::vector<cv::Vec4i> convexityDefects;
		if(hullIndices.size() > 0 && contour.size() > 0) {		
			CvMat contourMat = cvMat(1, contour.size(), CV_32SC2, (void*) &contour[0]);
			CvMat hullMat = cvMat(1, hullIndices.size(), CV_32SC1, (void*) &hullIndices[0]);
			CvMemStorage* storage = cvCreateMemStorage(0);
			CvSeq* defects = cvConvexityDefects(&contourMat, &hullMat, storage);
			for(int i = 0; i < defects->total; i++){
				CvConvexityDefect* cur = (CvConvexityDefect*) cvGetSeqElem(defects, i);
				cv::Vec4i defect;
				defect[0] = cur->depth_point->x;
				defect[1] = cur->depth_point->y;
				defect[2] = (cur->start->x + cur->end->x) / 2;
				defect[3] = (cur->start->y + cur->end->y) / 2;
				convexityDefects.push_back(defect);
			}
			cvReleaseMemStorage(&storage);
		}
		return convexityDefects;
	}
	
	std::vector<cv::Vec4i> convexityDefects(const ofPolyline& polyline) {
		std::vector<cv::Point2f> contour2f = toCv(polyline);
		std::vector<cv::Point2i> contour2i;
		Mat(contour2f).copyTo(contour2i);
		return convexityDefects(contour2i);
	}
	
	cv::RotatedRect minAreaRect(const ofPolyline& polyline) {
		return minAreaRect(Mat(toCv(polyline)));
	}
	
	cv::RotatedRect fitEllipse(const ofPolyline& polyline) {
		return fitEllipse(Mat(toCv(polyline)));
	}
	
	void fitLine(const ofPolyline& polyline, glm::vec2& point, glm::vec2& direction) {
		Vec4f line;
		fitLine(Mat(toCv(polyline)), line, CV_DIST_L2, 0, .01, .01);

        direction = glm::vec2(line[0], line[1]);
        point = glm::vec2(line[2], line[3]);
	}
    
	ofMatrix4x4 estimateAffine3D(std::vector<glm::vec3>& from, std::vector<glm::vec3>& to, float accuracy) {
		if(from.size() != to.size() || from.size() == 0 || to.size() == 0) {
			return ofMatrix4x4();
		}
		std::vector<unsigned char> outliers;
		return estimateAffine3D(from, to, outliers, accuracy);
	}
    
	ofMatrix4x4 estimateAffine3D(std::vector<glm::vec3>& from, std::vector<glm::vec3>& to, std::vector<unsigned char>& outliers, float accuracy) {
		Mat fromMat(1, from.size(), CV_32FC3, &from[0]);
		Mat toMat(1, to.size(), CV_32FC3, &to[0]);
		Mat affine;
		estimateAffine3D(fromMat, toMat, affine, outliers, 3, accuracy);
		ofMatrix4x4 affine4x4;
		affine4x4.set(affine.ptr<double>());
		affine4x4(3, 0) = 0;
		affine4x4(3, 1) = 0;
		affine4x4(3, 2) = 0;
		affine4x4(3, 3) = 1;
		Mat affine4x4Mat(4, 4, CV_32F, affine4x4.getPtr());
		affine4x4Mat = affine4x4Mat.t();
		affine4x4.set(affine4x4Mat.ptr<float>());
		return affine4x4;
	} 
}