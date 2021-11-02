Program that performs the training of the boat detector (bag-of-words + SVM).

It builds the vocabulary of visual words clustering SIFT descriptors computed from positive and negative 
patches, generated during the dataset preparation phase. Clusters centers will be the vocabulary codewords.

Once the vocabulary is built, it is used to describe the positive and the negative patches.
Each patch will be described by a normalized histogram which represents the frequency in the patch of
each visual word. This means that the i-th bin of the histogram will represent the frequency
of the i-th word in the vocabulary in the patch.

Finally, the labelled set of bag-of-words descriptors is fed to an SVM with a non-linear kernel (RBF),
which will come up with an hypothesis that classifies the data in two classes: boat (1) or non-boat (0).

To train the boat detector, provide the following command line arguments:

1. path to the directory containing the positive patches extracted with Laura_Bragagnolo_dataset_prep.
2. path to the directory containing the negative patches extracted with Laura_Bragagnolo_dataset_prep.