#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <iostream>
#include <fstream>
#include <opencv2/ml.hpp>
#include "Detector_Utils.h"

/*
* Program that performs the training of the boat detector (bag-of-words + SVM)
* 
* It builds the vocabulary of visual words clustering SIFT descriptors computed from positive and negative 
* patches, generated during the dataset preparation phase. Clusters centers will be the vocabulary codewords.
* 
* Once the vocabulary is built, it is used to describe the positive and the negative patches. Each patch will be
* described by a normalized histogram which represents the frequency in the patch of each visual word. This means
* that the i-th bin of the histogram will represent the frequency of the i-th word in the vocabulary in the patch.
* 
* Finally, the labelled set of bag-of-words descriptors is fed to an SVM with a non-linear kernel (RBF), which will
* come up with an hypothesis that classifies the data in two classes: boat (1) or non-boat (0).
*/
int main(int argc, char** argv) {

	if (argc < 3) {

		std::cout << "Command line arguments are missing." << std::endl;
		std::cout << "Provide path to the positive patches and the path to the negative patches." << std::endl;
		return -1;
	}

	cv::String BOAT_PATCHES_PATH = argv[1];
	cv::String NONBOAT_PATCHES_PATH = argv[2];

	//*********************************** VISUAL VOCABULARY ************************************//
	
	// Load patches to extract SIFT features from

	std::vector<cv::String> positive_files;
	std::vector<cv::String> negative_files;
	std::vector<cv::Mat> positive_patches;
	std::vector<cv::Mat> negative_patches;

	std::vector<cv::String> pattern = { "*.png" };

	// Load positive patches

	std::cout << "Loading positive patches..." << std::endl;

	if (Detector_Utils::loadFiles(BOAT_PATCHES_PATH, pattern, positive_files)) {
	
		std::cout << "Error occurred while loading positive patches.";
		return -1;
	}

	cv::Mat image;

	for (int i = 0; i < positive_files.size(); i++) {

		image = cv::imread(positive_files[i]);
		positive_patches.push_back(image);
	}

	std::cout << "Positive patches successfully loaded." << std::endl;
	std::cout << "Total number of positive patches: " << positive_patches.size() << std::endl;
	std::cout << std::endl;

	// Load negative patches

	std::cout << "Loading negative patches..." << std::endl;

	if (Detector_Utils::loadFiles(NONBOAT_PATCHES_PATH, pattern, negative_files)) {

		std::cout << "Error occurred while loading negative patches.";
		return -1;
	}

	for (int i = 0; i < negative_files.size(); i++) {

		image = cv::imread(negative_files[i]);
		negative_patches.push_back(image);
	}

	std::cout << "Negative patches successfully loaded." << std::endl;
	std::cout << "Total number of negative patches: " << negative_patches.size() << std::endl;
	std::cout << std::endl;

	// SIFT DETECTION

	std::vector<cv::KeyPoint> keypoints;
	std::vector<cv::Mat> pos_descriptors(positive_patches.size());
	std::vector<cv::Mat> neg_descriptors(negative_patches.size());
	cv::Mat descriptors;

	cv::Ptr<cv::SIFT> detector = cv::SIFT::create();

	cv::Mat all_features;

	// for positive patches

	std::cout << "Detecting SIFT features for positive patches..." << std::endl;
	std::cout << std::endl;

	for (int i = 0; i < positive_patches.size(); i++) {

		// detect sift features and compute descriptors
		detector->detectAndCompute(positive_patches[i], cv::Mat(), keypoints, descriptors);

		if (!descriptors.empty()) {

			// pos_descriptors[i] will contain SIFT descriptors computed for image positive_patches[i]
			pos_descriptors.push_back(descriptors);

			// all_features will contain ALL the computed SIFT descriptors that are going to be clustered
			all_features.push_back(descriptors);
		}
	}

	std::cout << "SIFT features successfully computed for positive patches." << std::endl;
	std::cout << std::endl;

	// for negative patches

	std::cout << "Detecting SIFT features for negative patches..." << std::endl;
	std::cout << std::endl;

	for (int i = 0; i < negative_patches.size(); i++) {

		// detect sift features and compute descriptors
		detector->detectAndCompute(negative_patches[i], cv::Mat(), keypoints, descriptors);

		if (!descriptors.empty()) {

			// neg_descriptors[i] will contain SIFT descriptors computed for image negative_patches[i]
			neg_descriptors.push_back(descriptors);

			all_features.push_back(descriptors);
		}
	}

	std::cout << "SIFT features successfully computed for negative patches." << std::endl;
	std::cout << std::endl;

	// K-MEANS CLUSTERING
	 
	// number of codewords for the visual vocabulary
	int n_words = 300;
	cv::TermCriteria term_criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 0.01);


	cv::BOWKMeansTrainer BOWTrainer(n_words, term_criteria);

	std::cout << "Clustering SIFT descriptors..." << std::endl;
	std::cout << std::endl;

	cv::Mat vocabulary = BOWTrainer.cluster(all_features);

	std::cout << "Clustering of SIFT descriptors completed successfully." << std::endl;
	std::cout << std::endl;

	cv::FileStorage fs("../../vocabulary.yml", cv::FileStorage::WRITE);
	fs << "vocabulary" << vocabulary;
	fs.release();

	//*************************************************************************************//

	//*********************************** SVM TRAINING ************************************//

	// create nearest neighbor matcher
	cv::Ptr<cv::DescriptorMatcher> matcher(new cv::FlannBasedMatcher);

	// create SIFT descriptor extractor
	cv::Ptr<cv::DescriptorExtractor> extractor(new cv::SiftDescriptorExtractor);

	// create bag of words descriptor extractor using matcher and sift features extractor
	cv::BOWImgDescriptorExtractor BOWImgDescriptor(extractor, matcher);

	// set vocabulary 
	BOWImgDescriptor.setVocabulary(vocabulary);

	cv::Mat bow_descriptors;

	cv::Mat train_samples;
	cv::Mat labels;

	// BAG OF WORDS IMAGE DESCRIPTORS COMPUTATION

	// bag of words image descriptors are computed in the following way:
	// 1. given the SIFT keypoints descriptors computed for each patch, using a nearest neighbor matcher, we find the codewords which are
	// nearest to such descriptors.
	// 2. we compute the bow descriptor, which is a normalized histogram of the frequencies of the codewords encountered in the patch.
	// The i-th bin of such histogram represents the frequency of the i-th codeword in the image.
	
	for (int i = 0; i < pos_descriptors.size(); i++) {

		if (!pos_descriptors[i].empty()) {

			// compute bow descriptor
			BOWImgDescriptor.compute(pos_descriptors[i], bow_descriptors);

			// add bow descriptor to train samples
			train_samples.push_back(bow_descriptors);

			// corresponding label is '1' since we are working with positive patches
			labels.push_back(1);
		}
	}

	std::cout << "BOW image descriptors computed for positives patches." << std::endl;

	for (int i = 0; i < neg_descriptors.size(); i++) {

		if (!neg_descriptors[i].empty()) {

			// compute bow descriptor
			BOWImgDescriptor.compute(neg_descriptors[i], bow_descriptors);

			// add bow descriptor to train samples
			train_samples.push_back(bow_descriptors);

			// corresponding label is '0' since we are working with negative patches
			labels.push_back(0);
		}
	}

	std::cout << "BOW image descriptors computed for negatives patches." << std::endl;
	std::cout << std::endl;

	// SVM TRAINING

	// given the bow descriptors and the corresponding labels, we train a support vector machine, that will learn 
	// a plane which separates positive descriptors from negative ones.
	// the SVM chosen is an SVM with a non-linear kernel, namely a radial basis function kernel
	// with respect to a linear SVM, a non-linear one is more powerful since it can capture also non-linear functions and thus, provide
	// a better classification if data is not perfectly separable.

	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();

	cv::Ptr<cv::ml::TrainData> dataset = cv::ml::TrainData::create(train_samples, cv::ml::SampleTypes::ROW_SAMPLE, labels);

	// C-Support Vector Classification
	// allows imperfect separation of classes, applying a penalty C on outliers
	svm->setType(cv::ml::SVM::C_SVC);

	// Radial Basis Function kernel
	svm->setKernel(cv::ml::SVM::RBF);

	svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 10000, 1e-6));

	// train SVM model tuning in an optimal way the different parameters. 
	// this is done performing k-fold cross validation with k = 10 (default value) using a grid of standard values for each parameter.
	// in the end, the model which performs better is chosen.

	std::cout << "Training the SVM..." << std::endl;
	std::cout << std::endl;

	svm->trainAuto(dataset);

	svm->save("../../svm.yml");

	std::cout << "Training done!" << std::endl;

}