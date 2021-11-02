Program that implements a boat detector, based on bag-of-words and support vector machine.

Provided some test images, it detects boats, drawing bounding boxes around the objects.
Bounding boxes depicted in green are the best boxes the detector has found for the boats in the image.
Bounding boxes in red are either false positives or poorer detections with respect to the green ones.
On the green boxes, it shows the corresponding intersection over union.

To test this boat detector, provide the following command line arguments:

1. path to the directory containing the test images (accepted formats: png or jpg).
2. path to the directory containing the annotation files corresponding to the provided test images.
3. value of the threshold for non-maxima suppression (e.g. 0.5)