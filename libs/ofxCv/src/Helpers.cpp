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
    
    // Code for thinning a binary image using Zhang-Suen algorithm.
    // Normally you wouldn't call this function directly from your code.
    //
    // im    Binary image with range = [0,1]
    // iter  0=even, 1=odd
    //
    // Author:  Nash (nash [at] opencv-code [dot] com)
    // https://github.com/bsdnoobz/zhang-suen-thinning
    void thinningIteration( cv::Mat & img, int iter, cv::Mat & marker )
    {
        CV_Assert(img.channels() == 1);
        CV_Assert(img.depth() != sizeof(uchar));
        CV_Assert(img.rows > 3 && img.cols > 3);
        
        int nRows = img.rows;
        int nCols = img.cols;
        
        if (img.isContinuous()) {
            nCols *= nRows;
            nRows = 1;
        }
        
        int x, y;
        uchar *pAbove;
        uchar *pCurr;
        uchar *pBelow;
        uchar *nw, *no, *ne;    // north (pAbove)
        uchar *we, *me, *ea;
        uchar *sw, *so, *se;    // south (pBelow)
        
        uchar *pDst;
        
        // initialize row pointers
        pAbove = NULL;
        pCurr  = img.ptr<uchar>(0);
        pBelow = img.ptr<uchar>(1);
        
        for (y = 1; y < img.rows-1; ++y) {
            // shift the rows up by one
            pAbove = pCurr;
            pCurr  = pBelow;
            pBelow = img.ptr<uchar>(y+1);
            
            pDst = marker.ptr<uchar>(y);
            
            // initialize col pointers
            no = &(pAbove[0]);
            ne = &(pAbove[1]);
            me = &(pCurr[0]);
            ea = &(pCurr[1]);
            so = &(pBelow[0]);
            se = &(pBelow[1]);
            
            for (x = 1; x < img.cols-1; ++x) {
                // shift col pointers left by one (scan left to right)
                nw = no;
                no = ne;
                ne = &(pAbove[x+1]);
                we = me;
                me = ea;
                ea = &(pCurr[x+1]);
                sw = so;
                so = se;
                se = &(pBelow[x+1]);
                
                // @valillon
                // Beyond this point the original Nash's code used an unified conditional at the end
                // Intermediate conditionals speeds the process up (depending on the image to be thinned).
                if (*me == 0) continue; // do not thin already zeroed pixels
                
                int A  = (*no == 0 && *ne == 1) + (*ne == 0 && *ea == 1) +
                (*ea == 0 && *se == 1) + (*se == 0 && *so == 1) +
                (*so == 0 && *sw == 1) + (*sw == 0 && *we == 1) +
                (*we == 0 && *nw == 1) + (*nw == 0 && *no == 1);
                if (A != 1) continue;
                
                int B  = *no + *ne + *ea + *se + *so + *sw + *we + *nw;
                if (B < 2 || B > 6) continue;
                
                int m1 = iter == 0 ? (*no * *ea * *so) : (*no * *ea * *we);
                if (m1) continue;
                
                int m2 = iter == 0 ? (*ea * *so * *we) : (*no * *so * *we);
                if (m2) continue;
                
                // if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
                pDst[x] = 1;
            }
        }
        
        img &= ~marker;
    }
        
}
