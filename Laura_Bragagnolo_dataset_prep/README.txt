Program that prepares the dataset needed to train the classifier for boat detection.
It builds a dataset made of positive and negative patches.
The images classified as positive are cropped to get patches that contain only one boat each.
Such cropping is made using the coordinates of the rectangles representing the ground truth
for boat detection.

Negative patches are built using a subset of the images classified as positive.
Running the selective search segmentation, we extract regions from each image and we use as negative
patches the regions that have an intersection over union with each image's ground truth equal 
to zero. We will generate up to 4 patches for each image.

To build the dataset, provide the following command line arguments:

1. path to the directory containing the images to be used for the generation of positive examples.
(e.g. ../data/images)
2. path to the directory containing the annotation files corresponding to the images of point 1.
Annotation files are assumed to be txt files and the structure of a generic line is assumed to be
the following:

boat:xmin;xmax;ymin;ymax