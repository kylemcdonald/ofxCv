#pragma once

#include "ofxCv.h"
#include "ofVectorMath.h"

namespace ofxCv {
	
	// Kalman filter for positioning
	template <class T>
	class KalmanPosition_ {
		cv::KalmanFilter KF;
		cv::Mat_<T> measurement, prediction, estimated;
	public:
		// smoothness, rapidness: smaller is more smooth/rapid
		// bUseAccel: set true to smooth out velocity
		void init(T smoothness = 0.1, T rapidness = 0.1, bool bUseAccel = false);
		void update(const glm::vec3&);
		glm::vec3 getPrediction();
		glm::vec3 getEstimation();
		glm::vec3 getVelocity();
	};
	
	typedef KalmanPosition_<float> KalmanPosition;
	
	// Kalman filter for orientation
	template <class T>
	class KalmanEuler_ : public KalmanPosition_<T> {
		glm::vec3 eulerPrev; // used for finding appropriate dimension
	public:
		void init(T smoothness = 0.1, T rapidness = 0.1, bool bUseAccel = false);
		void update(const ofQuaternion&);
		ofQuaternion getPrediction();
		ofQuaternion getEstimation();
		//ofQuaternion getVelocity();
	};
	
	typedef KalmanEuler_<float> KalmanEuler;	
}
