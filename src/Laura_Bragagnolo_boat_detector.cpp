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

/*
* Program that implements a boat detector, based on bag-of-words and support vector machine.
* 
* Provided some test images, it detects boats in the images, drawing bounding boxes around the objects.
* Bounding boxes depicted in green are the best boxes the boat detector has found for the boats in the image.
* Bounding boxes in red are either false positives or poorer detections with respect to the green ones.
* On the green boxes, it shows the corresponding intersection over union.
*/
int main(int argc, char** argv) {

	if (argc < 4) {
		std::cout << "Missing arguments. Provide the path to the test images, the corresponding annotations ";
		std::cout << "and the threshold for non-maxima suppression." << std::endl;
		return -1;
	}

	cv::String TEST_PATH = argv[1];
	cv::String ANNOTATIONS_PATH = argv[2];
	float NMS_THRESHOLD = std::stof(argv[3]);

	// load test images

	std::vector<cv::String> pattern = { "*.png", "*.jpg"};
	
	std::vector<cv::String> test_files;
	std::vector<cv::Mat> test_images;

	if (Detector_Utils::loadFiles(TEST_PATH, pattern, test_files)) {
	
		std::cout << "Error occurred while loading test images." << std::endl;
		return -1;
	}

	for (const auto& t : test_files) {

		cv::Mat im = cv::imread(t);
		test_images.push_back(im);
	}

	std::cout << "Test images successfully loaded." << std::endl;

	// load annotation files

	std::vector<cv::String> annot_files;
	pattern = { "*.txt" };

	if (Detector_Utils::loadFiles(ANNOTATIONS_PATH, pattern, annot_files)) {

		std::cout << "Error occurred while loading annotations files for test images." << std::endl;
		return -1;
	}

	// compute ground truth for test images

	std::vector<std::vector<cv::Rect>> ground_truth(test_images.size());

	for (int i = 0; i < test_images.size(); i++) {
	
		ground_truth[i] = Detector_Utils::getGroundTruth(annot_files[i]);
	}	

	// load vocabulary of visual words
	cv::Mat vocabulary;

	cv::FileStorage fs("../vocabulary.yml", cv::FileStorage::READ);
	fs["vocabulary"] >> vocabulary;
	fs.release();

	// load the trained svm
	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
	svm = cv::ml::SVM::load("../svm.yml");

	cv::Ptr<cv::SIFT> detector = cv::SIFT::create();

	// create a nearest neighbor matcher
	cv::Ptr<cv::DescriptorMatcher> matcher(new cv::FlannBasedMatcher);

	// create a SIFT descriptor extractor
	cv::Ptr<cv::DescriptorExtractor> extractor(new cv::SiftDescriptorExtractor);

	// create bag of words descriptor extractor
	cv::BOWImgDescriptorExtractor BOWImgDescriptor(extractor, matcher);

	// set vocabulary obtained with training
	BOWImgDescriptor.setVocabulary(vocabulary);

	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;
	cv::Mat bow_descriptors;

	std::vector<cv::Rect> proposals;
	std::vector<cv::Mat> patches;
	std::vector<cv::Rect> pred_boxes;
	std::vector<cv::Rect> final_boxes;

	cv::Mat outImage;

	// for each test image, run selective search to get proposed regions, process such patches as we processed
	// the patches used for training, compute bag of words descriptors and classify patches using the trained SVM

	for (int i = 0; i < test_images.size(); i++) {

		std::cout << "Processing image " << test_files[i] << std::endl;

		// create Selective Search Segmentation object 
		cv::Ptr<cv::ximgproc::segmentation::SelectiveSearchSegmentation> selectiveSearch;
		selectiveSearch = cv::ximgproc::segmentation::createSelectiveSearchSegmentation();

		// get regions to examine
		proposals = Detector_Utils::getProposals(test_images[i], selectiveSearch, 2000);
		
		std::cout << "Proposals successfully computed." << std::endl;

		// extract patches from test image
		Detector_Utils::getPatches(proposals, test_images[i], patches);

		// process patches 
		Detector_Utils::processPatches(patches);

		std::cout << "Classifying proposals..." << std::endl;

		outImage = test_images[i].clone();
		pred_boxes.clear();
		// for each patch extract bag of words descriptors and classify using svm
		for (int j = 0; j < patches.size(); j++) {

			// detect SIFT keypoints and compute descriptors
			detector->detectAndCompute(patches[j], cv::Mat(), keypoints, descriptors);

			if (!descriptors.empty()) {

				// compute bag of words descriptor for the patch
				BOWImgDescriptor.compute(descriptors, bow_descriptors);

				// classify patch
				float response = svm->predict(bow_descriptors);

				// if patch is classified as boat:
				if (response == 1) {

					// j-th patch is obtained from j-th proposed region
					pred_boxes.push_back(proposals[j]);
				}
			}
		}

		std::cout << "Non-maxima suppression..." << std::endl;
		std::cout << std::endl;

		Detector_Utils::nonMaximaSuppression(pred_boxes, final_boxes, NMS_THRESHOLD);

		// displaying result 
		std::cout << "Intersection over union:" << std::endl;
		outImage = test_images[i].clone();

		if (!final_boxes.empty()) {
		
			// for each ground truth box, we show in green the bounding box giving the highest response
			for (int j = 0; j < ground_truth[i].size(); j++) {

				float max_iou; int max_i;
				Detector_Utils::getMaxResponseIOU(final_boxes, ground_truth[i][j], max_iou, max_i);

				if (max_iou > 0.0f) {
					
					// show in green color the box which has maximum IOU for this ground truth box
					rectangle(outImage, final_boxes[max_i], cv::Scalar(50, 205, 50), 2);

					// write above the box the corresponding IOU
					float offset_x = final_boxes[max_i].x;
					float offset_y = final_boxes[max_i].y - 7;

					if (offset_y < 0) {
						offset_y = final_boxes[max_i].y + 21;
					}

					cv::putText(outImage, std::to_string(max_iou), cv::Point(offset_x, offset_y),
						cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(50, 205, 50), 2);

					std::cout << max_iou << std::endl;

					// erase the box we already shown 
					final_boxes.erase(final_boxes.begin() + max_i);
				
				}
			}

			// the remaining boxes are shown in red color
			for (int j = 0; j < final_boxes.size(); j++) {

				rectangle(outImage, final_boxes[j], cv::Scalar(0, 0, 255), 1);
			}
			
		}

		//show output
		cv::resize(outImage, outImage, cv::Size(1000, 600));
		cv::imshow("Test image", outImage);
		cv::waitKey(0);

		std::cout << std::endl;
	}
}