#ifndef _LABEL2D_HPP_
#define _LABEL2D_HPP_ 2014041201

#include <iostream>
#include <opencv2/core.hpp>

//#if LABEL2D_DEBUG_MODE

class Label2D {
private:
	//// Protected Fields ////

	// Input data
	int width_, height_, max_label_nr_;
	int target_value_, tolerance_;
	cv::Mat image_;

	// Internal data
	std::vector<std::vector<int>> label_mat_;
	std::vector<int> label_lut_;

	//// Private Methods ////
	void computeInitialLabel_();
	void updateLUT_();
	void mergeSameLabel_();
	void findLabelAscendingOrder_(int x, int y, std::vector<int> &label_nr_vec);
	void labelPixel_(int x, int y, std::vector<int> &label_nr_vec, bool is_valid_pixel);
	static bool compareVectorSizes(const std::vector<int> &a, const std::vector<int> &b) { return (a.size() < b.size()); }

public:
	Label2D() : label_lut_(1), max_label_nr_(0), target_value_(0), tolerance_(0) {}
	~Label2D() {}

	// Setter
	inline void setTargetValue(int value = 0) { target_value_ = value; }
	void setGrayImage(const cv::Mat &image);	// Input image must be 1 channel
	inline void setTolerance(int tolerance) { tolerance_ = tolerance; }

	// Processing
	void compute();

	// Getter
	void getIndicesList(std::vector<std::vector<int>> &indices_list, int size_threshold = 1) const;
	inline int getTolerance() const { return tolerance_; }
};

#endif		// #ifndef _LABEL2D_HPP_