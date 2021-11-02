#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/ximgproc/segmentation.hpp>

/*
* Class of static functions.
*/

class Detector_Utils {

public:

	/*
	* Function to load files from a specified directory. 
	* 
	* @param path						Path to the directory that contains the files we wish to load.	
	* @param pattern					Admissible formats for the files to load.
	* @param &filenames					Extracted files names. 
	* 
	* @return int						Retuns -1 if file loading encountered errors or if there were not files
	*									having the specified formats in the given directory. Returns 0 otherwise.
	*/
	static int loadFiles(cv::String path, std::vector<cv::String> pattern, std::vector<cv::String> &filenames);

	
	/*
	* Function to extract the image name, from the path of the image (e.g. extracts "image0001" from "../image0001.png").
	* 
	* @param filename					Name of the image file.
	* @param dir_path					Path to the directory of the file.
	* @param ext						Image file extension.
	* 
	* @return cv::String				Image name.
	*/
	static cv::String getImageName(cv::String filename, cv::String dir_path, cv::String ext);

	
	/*
	* Function to parse the specified annotation file. Returns ground truth boxes.
	* Only boxes for objects labelled as "boat" are considered.
	* Takes into account that a line of the annotation file .txt looks as follows:
	* boat(or hiddenboat):xmin;xmax;ymin;ymax
	*
	* @param filename					Name of the annotation file to parse.
	* 
	* @return std::vector<cv::Rect>		Returns vector of rects corresponding to ground truth boxes.
	*/
	static std::vector<cv::Rect> getGroundTruth(cv::String filename);

	
	/*
	* Function to parse a line read from the annotation file, in order to get ground truth boxes
	* corners coordinates. Takes into account that a line of the annotation file .txt looks as follows:
	* boat(or hiddenboat):xmin;xmax;ymin;ymax
	*
	* @param line			Line from file which contains box coordinates.
	* @param corners[]		Array of integers which contains the coordinates of the top-left corner of the box
	*						(corners[0], corners[3]) and of the right-bottom corner of the box (corners[2], corners[4]).
	*/
	static void getCorners(std::string line, int corners[]);

	
	/*
	* Function to extract patches from a given image.
	* 
	* @param rects			Rectangles which represent the patches contours.
	* @param image			Image to crop.
	* @param &patches		Patches cropped from the given image.
	*/
	static void getPatches(std::vector<cv::Rect> rects, cv::Mat image, std::vector<cv::Mat>& patches);


	/*
	* Function to process patches.
	* Converts to grayscale and applies CLAHE equalization.
	* 
	* @param &patches		Patches to process.
	*/
	static void processPatches(std::vector<cv::Mat>& patches);


	/*
	* Function to save the image patches to a specified path.
	* 
	* @param patches		Patches to save.
	* @param image_name		Name of the image the provided patches are created from. Used to give a unique name to each patch.
	* @param patches_path	Path for the saved patches.
	*/
	static void savePatches(std::vector<cv::Mat> patches, cv::String image_name, cv::String patches_path);


	/*
	* Function to run selective search algorithm on a image and get up to a given number of proposed regions.
	* 
	* @param image			Image on which selective search is run.
	* @param ss				Pointer to a selective seach segmentation object.
	* @param max_n			Maximum number of proposals to return.
	* 
	* @return std::vector<cv::Rect> Returns a vector containing up to max_n regions extracted from the provided image
	*								using selective search segmentation.
	*/
	static std::vector<cv::Rect> getProposals(cv::Mat image,
											cv::Ptr<cv::ximgproc::segmentation::SelectiveSearchSegmentation> ss, int max_n);


	/*
	* Function to compute the intersection over union between two boxes.
	* IOU = overlap / area of rect1 + area of rect2 - overlap
	* 
	* @param rect1			First box.
	* @param rect2			Second box.
	* 
	* @return float			Intersection over union between rect1 and rect2.
	*/
	static float intersectionOverUnion(cv::Rect rect1, cv::Rect rect2);

	
	/*
	* Function to apply non-maxima suppression to a set of bounding boxes, without prediction confidence values.
	* Takes inspiration from the blog post at the following link:
	* https://www.pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/
	* 
	* 
	* @param pred_boxes		Set of all the bounding boxes obtained for an image.
	* @param &final_boxes	Set of bounding boxes without non-maxima.
	* @param threshold		Threshold value for the non-maxima suppression. If overlap between two boxes is greater than such value, 
	*						one of the two is suppressed.
	*/
	static void nonMaximaSuppression(std::vector<cv::Rect> pred_boxes, std::vector<cv::Rect>& final_boxes, float threshold);


	/*
	* Function to get, for a ground truth box, which bounding box gives the highest intersection over union and which is such value.
	* 
	* @param rects			Set of bounding boxes.
	* @param gt_box			Ground truth box.
	* @param &max_iou		Maximum intersection over union obtained for the provided ground truth box.
	* @param &max_i			Index of the box which gives the maximum intersection over union.
	*/
	static void getMaxResponseIOU(std::vector<cv::Rect> rects, cv::Rect gt_box, float &max_iou, int &max_i);

};


















