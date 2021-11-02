#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/ximgproc/segmentation.hpp>
#include <iostream>
#include "Detector_Utils.h"

/*
* Program that prepares the dataset needed to train the classifier for boat detection.
* It builds a dataset made of positive and negative patches.
* The images classified as positive are cropped to get patches that contain only one boat each.
* Such cropping is made using the coordinates of the rectangles representing the ground truth for boat detection.
* 
* Negative patches are built using a subset of the images classified as positive. Running the selective search segmentation,
* we extract regions from each image and we use as negative patches the regions that have an intersection over union with each image's ground truth equal 
* to zero. We will generate up to 4 patches for each image.
*/


int main(int argc, char** argv) {

	if (argc < 3) {
		
		std::cout << "Some command line arguments are missing." << std::endl;
		std::cout << "Pass as arguments: path to images used to build positive samples and ";
		std::cout << "path to the annotation files." << std::endl;
		return -1;
	}

	const cv::String BOAT_PATH = argv[1];
	const cv::String ANNOTATIONS_PATH = argv[2];

	//*********************************** POSITIVE SAMPLES ************************************//

	// Load annotation files

	std::cout << "Loading annotations files..." << std::endl;

	std::vector<cv::String> filenames;
	std::vector<cv::String> pattern = { "*.txt" };

	if (Detector_Utils::loadFiles(ANNOTATIONS_PATH, pattern, filenames)) {

		std::cout << "Error occurred while loading annotations files." << std::endl;
		return -1;
	}
		
	std::cout << "Generating positive examples..." << std::endl;

	// create directory in which positive examples are going to be saved
	const cv::String BOAT_PATCHES_DIR = "../../BOATS";
	cv::utils::fs::createDirectory(BOAT_PATCHES_DIR);
	const cv::String BOAT_PATCHES_PATH = BOAT_PATCHES_DIR + "/";
	
	std::vector<cv::String> image_name;
	std::vector<std::vector<cv::Rect>> ground_truth(filenames.size());
	std::vector<cv::Mat> images;
	std::vector<cv::Mat> patches;

	for (int i = 0; i < filenames.size(); i++) {

		std::cout << "Processing " << filenames[i] << " ..." << std::endl;

		// extract the "name" of each image (e.g. image0001)
		image_name.push_back(Detector_Utils::getImageName(filenames[i], ANNOTATIONS_PATH, ".txt"));

		// build the path to the image corresponding to the current annotation file
		cv::String image_path = BOAT_PATH + image_name[i] + ".png";

		// read in image
		cv::Mat image = cv::imread(image_path);
		images.push_back(image);

		// parse annotation file and get ground truth boxes
		ground_truth[i] = Detector_Utils::getGroundTruth(filenames[i]);

		// extract boat patches according to ground truth
		Detector_Utils::getPatches(ground_truth[i], image, patches);

		// process boat patches (grayscale + CLAHE equalization)
		Detector_Utils::processPatches(patches);

		// save boat patches to the desired path
		Detector_Utils::savePatches(patches, image_name[i], BOAT_PATCHES_PATH);
	}

	std::cout << "Positive examples generated!!" << std::endl;

	//******************************** NEGATIVE SAMPLES ************************************//

	// run selective search on positive images and use as negatives the patches that have an intersection over union
	// with positive patches which is equal to zero 

	std::vector<cv::Rect> proposals;
	std::vector<cv::Rect> neg_rects;

	// create directory in which negative examples are going to be saved
	const cv::String NONBOAT_PATCHES_DIR = "../../NONBOATS";
	cv::utils::fs::createDirectory(NONBOAT_PATCHES_DIR);
	const cv::String NONBOAT_PATCHES_PATH = NONBOAT_PATCHES_DIR + "/";

	cv::Ptr<cv::ximgproc::segmentation::SelectiveSearchSegmentation> ss;
	ss = cv::ximgproc::segmentation::createSelectiveSearchSegmentation();

	for (int i = 0; i < images.size(); i+=2) {

		std::cout << "Processing image " << filenames[i] << "..." << std::endl;

		proposals = Detector_Utils::getProposals(images[i], ss, 2000);
		
		neg_rects.clear();
		int count = 0;

		for (int j = 0; count < 4 && j < proposals.size(); j++) {
			
			bool negative = true;

			//proposals that do not overlap with ground_truth are negatives
			for (int k = 0; k < ground_truth[i].size(); k++) {

				float iou = Detector_Utils::intersectionOverUnion(proposals[j], ground_truth[i][k]);

				if (iou > 0) {

					negative = false;
					break;
				}
			}

			if (negative) {
			
				neg_rects.push_back(proposals[j]);
				count++;
			}
		}
		
		// process and save negative patches
		Detector_Utils::getPatches(neg_rects, images[i], patches);
		Detector_Utils::processPatches(patches);
		Detector_Utils::savePatches(patches, image_name[i], NONBOAT_PATCHES_PATH);
	}

}