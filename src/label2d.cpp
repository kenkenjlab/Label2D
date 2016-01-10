#include "label2d.hpp"

#ifdef LABEL2D_DEBUG_MODE
#include <opencv2/highgui.hpp>
#endif

void Label2D::setGrayImage(const cv::Mat &image) {
	image_ = image.clone();
	width_ = image_.cols;
	height_ = image_.rows;
	label_mat_.resize(width_);
	for(std::vector<std::vector<int>>::iterator it = label_mat_.begin(); it != label_mat_.end(); it++)
		it->resize(height_);
}

void Label2D::compute() {
#ifdef LABEL2D_DEBUG_MODE
	IplImage *test1 = cvCreateImage(cvSize(width_, height_), IPL_DEPTH_8U, 3);
	IplImage *test2 = cvCreateImage(cvSize(width_, height_), IPL_DEPTH_8U, 3);
	IplImage *test3 = cvCreateImage(cvSize(width_, height_), IPL_DEPTH_8U, 3);
	std::vector<std::vector<int>> indices_list;
#endif

	computeInitialLabel_();

#ifdef LABEL2D_DEBUG_MODE
	getIndicesList(indices_list, 1);
	for(int i = 0; i < indices_list.size(); i++) {
		for(int j = 0; j < indices_list[i].size(); j++) {
			int index = indices_list[i][j];
			test1->imageData[index*3  ] = static_cast<char>((i % 8 == 0 ? 200: 0) + 55);
			test1->imageData[index*3+1] = static_cast<char>((i % 4 == 0 ? 200: 0) + 55);
			test1->imageData[index*3+2] = static_cast<char>((i % 2 == 0 ? 200: 0) + 55);
		}
		cvShowImage("test1", test1);
		cvWaitKey(1);
	}
#endif

	updateLUT_();

#ifdef LABEL2D_DEBUG_MODE
	getIndicesList(indices_list, 1);
	for(int i = 0; i < indices_list.size(); i++) {
		for(int j = 0; j < indices_list[i].size(); j++) {
			int index = indices_list[i][j];
			test2->imageData[index*3  ] = static_cast<char>((i / 4 % 2 == 0 ? 200: 0) + 55);
			test2->imageData[index*3+1] = static_cast<char>((i / 2 % 2 == 0 ? 200: 0) + 55);
			test2->imageData[index*3+2] = static_cast<char>((i % 2 == 0 ? 200: 0) + 55);
		}
		cvShowImage("test2", test2);
		cvWaitKey(1);
	}
#endif

	mergeSameLabel_();

#ifdef LABEL2D_DEBUG_MODE
	getIndicesList(indices_list, 1);
	for(int i = 0; i < indices_list.size(); i++) {
		for(int j = 0; j < indices_list[i].size(); j++) {
			int index = indices_list[i][j];
			test3->imageData[index*3  ] = static_cast<char>((i / 4 % 2 == 0 ? 200: 0) + 55);
			test3->imageData[index*3+1] = static_cast<char>((i / 2 % 2 == 0 ? 200: 0) + 55);
			test3->imageData[index*3+2] = static_cast<char>((i % 2 == 0 ? 200: 0) + 55);
		}
		cvShowImage("test3", test3);
		cvWaitKey(1);
	}
	cvWaitKey(0);
#endif
}

void Label2D::computeInitialLabel_() {
	for(int y = 0; y < height_; y++)
		for(int x = 0; x < width_; x++) {
			std::vector<int> label_nr_vec;

			findLabelAscendingOrder_(x, y, label_nr_vec);

			if(label_nr_vec.size() > 0) {
				int pixel = image_.data[x + y * width_];
				bool is_valid_pixel = (pixel <= target_value_ + tolerance_) & (pixel >= target_value_ - tolerance_);
				labelPixel_(x, y, label_nr_vec, is_valid_pixel);
			}
		}
}

void Label2D::updateLUT_() {
	// (1) Update LUT (Change to global minimum)
	bool iterate_flag;
	do {
		iterate_flag = false;
		for(int i = 0; i < label_lut_.size(); i++) {
			// (1-1) Find smallest number
			int smallest_label_nr = i;
			while(label_lut_[smallest_label_nr] != smallest_label_nr)
				smallest_label_nr = label_lut_[smallest_label_nr];

			// (1-2) Update
			if(label_lut_[i] != smallest_label_nr) {
				label_lut_[i] = smallest_label_nr;
				iterate_flag = true;		// Flag which means updated
			}
		}
	} while(iterate_flag);
}

void Label2D::mergeSameLabel_() {
	for(int y = 0; y < height_; y++)
		for(int x = 0; x < width_; x++)
			label_mat_[x][y] = label_lut_[label_mat_[x][y]];
}

void Label2D::findLabelAscendingOrder_(int x, int y, std::vector<int> &label_nr_vec) {
	for(int dy = -1; dy < 1; dy++)
		for(int dx = -1; dx < 2; dx++) {
			if(dx == 0 && dy == 0)
				break;
			int cx = x + dx, cy = y + dy;
			if(cx < 0 || cy < 0 || cx >= width_)
				continue;
			label_nr_vec.push_back(label_mat_[cx][cy]);
		}

	// Sort (smallest one first)
	std::sort(label_nr_vec.begin(), label_nr_vec.end());
}

void Label2D::labelPixel_(int x, int y, std::vector<int> &label_nr_vec, bool is_valid_pixel) {
	if(label_nr_vec.back() == 0 && is_valid_pixel) {
		// If none of neighbors are labeled, increment label number
		max_label_nr_++;
		label_mat_[x][y] = max_label_nr_;
		label_lut_.push_back(max_label_nr_);
	} else {
		// Otherwise assign smallest number (more than 0)
		int smallest_label_nr = INT_MAX;
		for(int i = 0; i < label_nr_vec.size(); i++)
			if(label_nr_vec[i] > 0)
				smallest_label_nr = std::min(smallest_label_nr, label_lut_[label_nr_vec[i]]);

		// Set label number
		if(is_valid_pixel)
			label_mat_[x][y] = smallest_label_nr;

		// Break if every point is not a target
		if(smallest_label_nr > label_lut_.size())
			return;

		// Update LUT (Change to local minimum)
		while(label_lut_[smallest_label_nr] != smallest_label_nr)
			smallest_label_nr = label_lut_[smallest_label_nr];
		for(int i = 0; i < label_nr_vec.size(); i++)
			if(label_nr_vec[i] > 0)
				if(label_lut_[label_nr_vec[i]] >= smallest_label_nr)
					label_lut_[label_nr_vec[i]] = smallest_label_nr;
//				else
//					???label_lut_[label_nr_vec[i]] = label_lut_[label_nr_vec[i]];
	}
}

void Label2D::getIndicesList(std::vector<std::vector<int>> &indices_list, int size_threshold) const {
	// Count for all list
	std::vector<std::vector<int>> all_list;
	all_list.resize(label_lut_.size());
	int count = 0;
	for(int y = 0; y < height_; y++)
		for(int x = 0; x < width_; x++) {
			if(label_mat_[x][y] > 0)	// Do not include 0
				all_list[label_mat_[x][y]].push_back(count);
			count++;
		}

	// Sort the list based on their size (largest one first)
	std::sort(all_list.rbegin(), all_list.rend(), compareVectorSizes);
	
	// Reject small ones
	for(int i = 0; i < all_list.size(); i++)
		if(all_list[i].size() < size_threshold) {
			indices_list.assign(all_list.begin(), all_list.begin() + i);
			return;
		}

	indices_list = all_list;
}