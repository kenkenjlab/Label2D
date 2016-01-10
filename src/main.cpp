#include "label2d.hpp"
#include <iostream>
#include <vector>
#include <opencv2/highgui.hpp>

int main(int argc, char** argv) {
	// (0) Check
	if (argc < 2) {
		std::cerr << " * ERROR: Input image path" << std::endl;
		return 1;
	}

	// (1) Load image
	const std::string filepath(argv[1]);
	cv::Mat src_img = cv::imread(filepath, 0);

	// (2) Perform Labeling
	Label2D l2d;
	l2d.setGrayImage(src_img);
	l2d.setTargetValue(255);
	l2d.setTolerance(0);
	l2d.compute();

	// (3) Visualize
	cv::Mat result_img = cv::Mat(src_img.size(), CV_8UC3, cv::Scalar::all(0));
	std::vector<std::vector<int>> indices_list;
	l2d.getIndicesList(indices_list);
	for (int i = 0; i < indices_list.size(); i++) {
		const std::vector<int> &indices = indices_list[i];
		for (int j = 0; j < indices.size(); j++) {
			const int &index = indices[j];
			cv::Vec3b &color = result_img.at<cv::Vec3b>(index);
			color.val[2] = static_cast<char>((i / 4 % 2 == 0 ? 200 : 0) + 55);
			color.val[1] = static_cast<char>((i / 2 % 2 == 0 ? 200 : 0) + 55);
			color.val[0] = static_cast<char>((i % 2 == 0 ? 200 : 0) + 55);
		}
	}
	cv::imshow("(1) Input image", src_img);
	cv::imshow("(2) labelled image", result_img);
	cv::waitKey();

	return 0;
}