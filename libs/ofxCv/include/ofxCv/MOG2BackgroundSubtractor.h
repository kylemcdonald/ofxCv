#pragma once

#include "ofxCv.h"
#include <opencv2/opencv.hpp>

namespace ofxCv {
    class MOG2BackgroundSubtractor
    {
    public:
        MOG2BackgroundSubtractor(int history, float varThreshold, bool bShadowDetection = true);
        
        template<class F, class T>
        void update(F& frame, T& thresholded, double learningRate = -1) {
            ofxCv::imitate(thresholded, frame, CV_8UC1);
            cv::Mat frameMat = toCv(frame);
            cv::Mat thresholdedMat = toCv(thresholded);
            update(frameMat, thresholdedMat, learningRate);
        }
        void update(cv::Mat& image, cv::Mat& mask, double learningRate = -1);
        
    private:
        cv::BackgroundSubtractorMOG2 m_subtractor;
        cv::Mat m_erosionElement, m_dilationElement;
    };    
}
