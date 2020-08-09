The blufr_frgc_config.mat file contains the following MATLAB variables.

-imageList:
	16028x1 cell array containing the filenames of the 16,028 controlled face images from the FRGCv2 database. Note that only filenames are contained, without directories or full paths. Note also that the following variables are all based on the order of this image list. Therefore the extracted features must exactly follow the order of this list.

-labels:
	16028x1 vector containing class labels associated with the above image list. There are 466 classes and the labels range from 1 to 466.

-devTrainIndex:
	3516x1 vector containing the indexes of the 3516 face images for the training set of the development set. Note that these indexes are based on the imageList, that is, by imageList(devTrainIndex) you will get the image filenames for these 3516 images. Also, by labels(devTrainIndex) you will obtain the class labels of these 3516 images. This is similar for the following variables.

-devTestIndex:
	3580x1 vector containing the indexes of the 3580 face images for the test set of the development set.

-trainIndex:
	10x1 cell array, with each cell containing the indexes of the training images for one trial.

-testIndex:
	10x1 cell array, with each cell containing the indexes of the test images for one trial.

-galIndex:
	10x1 cell array, with each cell containing the indexes of the 266 gallery images for one trial. For each trial t, galIndex{t} is a subset of testIndex{t}.

-probIndex:
	10x1 cell array, with each cell containing the indexes of the probe images for one trial. For each trial t, probIndex{t} is a subset of testIndex{t}. More specifically, testIndex{t} is the union of galIndex{t} and probIndex{t}.
