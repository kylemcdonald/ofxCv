#include "MOG2BackgroundSubtractor.h"

namespace ofxCv {
    
    MOG2BackgroundSubtractor::MOG2BackgroundSubtractor(int history, float varThreshold, bool bShadowDetection)
    : m_subtractor(history, varThreshold, bShadowDetection)
    , m_erosionElement(cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3), cv::Point(1,1)))
    , m_dilationElement(cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7), cv::Point(3, 3)))
    {
        
    }
    
    void MOG2BackgroundSubtractor::update(cv::Mat& image, cv::Mat& mask, double learningRate) {
        m_subtractor(image, mask, learningRate);
        cv::erode(mask, mask, m_erosionElement);
        cv::dilate(mask, mask, m_dilationElement);
    }
}
