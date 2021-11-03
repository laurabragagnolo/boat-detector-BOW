# boat-detector-BOW
Boat detection system based on bag-of-words features and SVM classification.

## Basic usage

Provided some test images, it detects boats, drawing bounding boxes around the objects.
Bounding boxes depicted in green are the best boxes the detector has found for the boats in the image.
Bounding boxes in red are either false positives or poorer detections with respect to the green ones.
On the green boxes, it shows the corresponding intersection over union.

To test this boat detector, provide to src/Laura_Bragagnolo_boat_detector.cpp the following command line arguments:

1. path to the directory containing the test images (accepted formats: png or jpg).
2. path to the directory containing the annotation files corresponding to the provided test images.
3. value of the threshold for non-maxima suppression (e.g. 0.5)

## Training
During the training phase, it builds the vocabulary of visual words clustering SIFT descriptors computed from positive and negative 
patches, generated during the dataset preparation phase. Clusters centers will be the vocabulary codewords.

Once the vocabulary is built, it is used to describe the positive and the negative patches.
Each patch will be described by a normalized histogram which represents the frequency in the patch of
each visual word. This means that the i-th bin of the histogram will represent the frequency
of the i-th word in the vocabulary in the patch.

Finally, the labelled set of bag-of-words descriptors is fed to an SVM with a non-linear kernel (RBF),
which will come up with an hypothesis that classifies the data in two classes: boat (1) or non-boat (0).

## Dataset preparation
It builds a dataset made of positive and negative patches.
The images classified as positive are cropped to get patches that contain only one boat each.
Such cropping is made using the coordinates of the rectangles representing the ground truth
for boat detection.

Negative patches are built using a subset of the images classified as positive.
Running the selective search segmentation, we extract regions from each image and we use as negative
patches the regions that have an intersection over union with each image's ground truth equal 
to zero.
