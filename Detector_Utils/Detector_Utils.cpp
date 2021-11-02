#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <iostream>
#include <fstream>
#include <opencv2/ml.hpp>
#include <opencv2/ximgproc/segmentation.hpp>
#include "Detector_Utils.h"

int Detector_Utils::loadFiles(cv::String path, std::vector<cv::String> pattern, std::vector<cv::String> &filenames) {

	while (filenames.empty()) {
		
		for (int i = 0; i < pattern.size(); i++) {
		
			try {

				cv::utils::fs::glob(path, pattern[i], filenames);
			}
			catch (cv::Exception e) {
				
				return -1;
			}
		}
	}
	
	if (filenames.empty()) {
		
		return -1;
	}

	return 0;
}


cv::String Detector_Utils::getImageName(cv::String filename, cv::String dir_path, cv::String ext) {

	// remove file format from name
	cv::String image_name = filename.substr(0, filename.find(ext));
	// remove path from name
	return image_name.substr(dir_path.length(), image_name.length());
}


std::vector<cv::Rect> Detector_Utils::getGroundTruth(cv::String filename) {

	std::fstream filestream(filename);
	int corners[4];
	std::vector<cv::Rect> ground_truth;

	// open annotation file stream
	if (filestream.is_open()) {

		std::string line;
		// until there is something to read
		for (int i = 0; std::getline(filestream, line); i++) {

			// extract label name (boat or hiddenboat)
			size_t pos = line.find(":");
			std::string name = line.substr(0, pos);
			line.erase(0, ++pos);

			// consider only boats
			if (name.compare("boat") == 0) {

				// parse line of the annotation file, get box corners
				getCorners(line, corners);
				ground_truth.push_back(cv::Rect(corners[0], corners[2], corners[1] - corners[0], corners[3] - corners[2]));
			}
		}
	}

	return ground_truth;
}


void Detector_Utils::getCorners(std::string line, int corners[]) {

	// get box corners coordinates
	size_t pos = 0;
	size_t i = 0;
	while ((pos = line.find(";")) != std::string::npos) {
		corners[i] = stoi(line.substr(0, pos));
		line.erase(0, ++pos);
		++i;
	}
}


void Detector_Utils::getPatches(std::vector<cv::Rect> rects, cv::Mat image, std::vector<cv::Mat> &patches) {

	patches.clear();
	cv::Mat patch;

	for (int i = 0; i < rects.size(); i++) {
		
		// crop image
		patch = image.clone();
		patch = image(rects[i]);

		patches.push_back(patch);
	}
}


void Detector_Utils::processPatches(std::vector<cv::Mat>& patches) {
	
	// switch to grayscale and perform CLAHE equalization
	int clipLimit = 40;
	cv::Size gridSize = cv::Size(8, 8);
	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clipLimit, gridSize);
	for (int i = 0; i < patches.size(); i++) {
	
		cv::cvtColor(patches[i], patches[i], cv::COLOR_BGR2GRAY);
		clahe->apply(patches[i], patches[i]);
	}
}


void Detector_Utils::savePatches(std::vector<cv::Mat> patches, cv::String image_name, cv::String patches_path) {

	for (int i = 0; i < patches.size(); i++) {
	
		// save patch to the correct position
		cv::String patchpath = patches_path + image_name + "_" + std::to_string(i) + ".png";
		cv::imwrite(patchpath, patches[i]);
	}
}


std::vector<cv::Rect> Detector_Utils::getProposals(cv::Mat image, cv::Ptr<cv::ximgproc::segmentation::SelectiveSearchSegmentation> ss, int max_n) {

	ss->setBaseImage(image);
	ss->switchToSelectiveSearchFast();

	// run selective search segmentation on input image

	std::vector<cv::Rect> rects;
	std::vector<cv::Rect> proposals;

	// run selective search
	ss->process(rects);

	int count = 0;
	
	for (int i = 0; count < max_n && i < rects.size(); i++) {

		// consider only patches with significative area
		if (rects[i].area() > 1000) {

			proposals.push_back(rects[i]);
			++count;
		}
	}
	

	return proposals;
}


float Detector_Utils::intersectionOverUnion(cv::Rect rect1, cv::Rect rect2) {

	// compute the area of intersection rectangle
	// rect1 & rect2 returns a rect which is the intersection of rect1 and rect2
	float intersection = (rect1 & rect2).area();

	// intersection over union
	return intersection / (rect1.area() + rect2.area() - intersection);

}


void Detector_Utils::nonMaximaSuppression(std::vector<cv::Rect> pred_boxes, std::vector<cv::Rect>& final_boxes, float threshold) {

	final_boxes.clear();

	if (pred_boxes.size() == 0) {
		
		return;
	}

	// sort bounding boxes according to the y-coordinate of the bottom corners
	std::multimap<int, size_t> idxs;

	for (int i = 0; i < pred_boxes.size(); i++) {
	
		idxs.emplace(pred_boxes[i].br().y, i);
	}

	while (idxs.size()) {
		
		// consider rectangle with greater y for bottom corners
		auto last = --std::end(idxs);
		cv::Rect rect1 = pred_boxes[last->second];

		// erase the element corresponding to the box we are analyzing
		idxs.erase(last);

		// loop over remaining boxes
		for (auto i = std::begin(idxs); i != std::end(idxs); i++) {
		
			// consider current box and compute IoU
			cv::Rect rect2 = pred_boxes[i->second];
			float iou = intersectionOverUnion(rect1, rect2);

			if (iou > threshold) {
				
				// suppress rect2 as non-maximum
				i = idxs.erase(i);
				i--;
			}
		}
		// rect1 is kept
		final_boxes.push_back(rect1);
	 	
	}
}


void Detector_Utils::getMaxResponseIOU(std::vector<cv::Rect> rects, cv::Rect gt_box, float &max_iou, int &max_i) {

	max_iou = 0;
	max_i = 0;
	for (int i = 0; i < rects.size(); i++) {

		float iou = Detector_Utils::intersectionOverUnion(gt_box, rects[i]);

		if (iou > max_iou) {

			max_iou = iou;
			max_i = i;
		}
	}
}






	

	






