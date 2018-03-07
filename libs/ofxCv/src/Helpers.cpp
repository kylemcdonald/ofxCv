#include "ofxCv/Helpers.h"
#include "ofxCv/Utilities.h"
#include "ofGraphics.h"

namespace ofxCv {
	
	using namespace cv;
	using namespace std;
	
	ofMatrix4x4 makeMatrix(Mat rotation, Mat translation) {
		Mat rot3x3;
		if(rotation.rows == 3 && rotation.cols == 3) {
			rot3x3 = rotation;
		} else {
			Rodrigues(rotation, rot3x3);
		}
		double* rm = rot3x3.ptr<double>(0);
		double* tm = translation.ptr<double>(0);
		return ofMatrix4x4(rm[0], rm[3], rm[6], 0.0f,
											 rm[1], rm[4], rm[7], 0.0f,
											 rm[2], rm[5], rm[8], 0.0f,
											 tm[0], tm[1], tm[2], 1.0f);
	}
	
	void drawMat(const Mat& mat, float x, float y) {
		drawMat(mat, x, y, mat.cols, mat.rows);
	}
	
    // special case for copying into ofTexture
    template <class S>
    void copy(const S& src, ofTexture& tex) {
        imitate(tex, src);
        int w = tex.getWidth(), h = tex.getHeight();
        int glType = tex.getTextureData().glInternalFormat;
        Mat mat = toCv(src);
		tex.loadData(mat.ptr(), w, h, glType);
    }
    
	void drawMat(const Mat& mat, float x, float y, float width, float height) {
        if(mat.empty()) {
            return;
        }
        ofTexture tex;
        copy(mat, tex);
		tex.draw(x, y, width, height);
	}
	
	void applyMatrix(const ofMatrix4x4& matrix) {
		glMultMatrixf((GLfloat*) matrix.getPtr());
	}
	
	int forceOdd(int x) {
		return (x / 2) * 2 + 1;
	}
	
	int findFirst(const Mat& arr, unsigned char target) {
		for(int i = 0; i < arr.rows; i++) {
			if(arr.at<unsigned char>(i) == target) {
				return i;
			}
		}
		return 0;
	}
	
	int findLast(const Mat& arr, unsigned char target) {
		for(int i = arr.rows - 1; i >= 0; i--) {
			if(arr.at<unsigned char>(i) == target) {
				return i;
			}
		}
		return 0;
	}
	
	float weightedAverageAngle(const std::vector<Vec4i>& lines) {
		float angleSum = 0;
		glm::vec2 start, end;
		float weights = 0;
		for(int i = 0; i < lines.size(); i++) {
            start = glm::vec2(lines[i][0], lines[i][1]);
			end = glm::vec2(lines[i][2], lines[i][3]);
			glm::vec2 diff = end - start;
            float length = glm::length(diff);
			float weight = length * length;
			float angle = atan2f(diff.y, diff.x);
			angleSum += angle * weight;
			weights += weight;
		}
		return angleSum / weights;
	}
	
    std::vector<cv::Point2f> getConvexPolygon(const std::vector<cv::Point2f>& convexHull, int targetPoints) {
		std::vector<cv::Point2f> result = convexHull;
		
		static const unsigned int maxIterations = 16;
		static const double infinity = std::numeric_limits<double>::infinity();
		double minEpsilon = 0;
		double maxEpsilon = infinity;
		double curEpsilon = 16; // good initial guess
		
		// unbounded binary search to simplify the convex hull until it's targetPoints
		if(result.size() > targetPoints) {
			for(int i = 0; i < maxIterations; i++) {
				approxPolyDP(Mat(convexHull), result, curEpsilon, true);
				if(result.size() == targetPoints) {
					break;
				}
				if(result.size() > targetPoints) {
					minEpsilon = curEpsilon;
					if(maxEpsilon == infinity) {
						curEpsilon = curEpsilon *  2;
					} else {
						curEpsilon = (maxEpsilon + minEpsilon) / 2;
					}
				}
				if(result.size() < targetPoints) {
					maxEpsilon = curEpsilon;
					curEpsilon = (maxEpsilon + minEpsilon) / 2;
				}
			}
		}
		
		return result;
	}
}
