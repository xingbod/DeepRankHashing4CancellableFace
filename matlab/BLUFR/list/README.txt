These list files can be used instead of the .mat configuration files. They are particularly useful beyond the MATLAB programming language, such as the C++ or Python. The formats of the list files are explained as follows.

-image_list.txt:
	A list containing the face image filenames. Each line contains one image file name. Note that only filenames are contained, without directories or full paths. Note also that the following list files are all based on the order of this image list. Therefore the extracted features must exactly follow the order of this list.

-labels.txt:
	A list containing class labels associated with the above image list. The labels range from 1 to C where C is the number of classes.

-dev_train_set.txt:
	A list containing the indexes, as well as the filenames of the face images for the training set of the development set. The first line gives the number, n, of images, followed by n lines, with each line containing the index of an image and the filename of the image, separated by a comma and a space. You can use the indexes along with the image list and labels to access the images, features, or labels; or, you can use the filenames to access the images.

-dev_test_set.txt:
	A list containing the indexes, as well as the filenames of the face images for the test set of the development set. The format of this file is the same as the dev_train_set.txt.

-train_set.txt:
	A list containing the indexes, as well as the filenames of the training images for 10 trials. The first line gives the number 10 meaning 10 trials, followed by 10 blocks of lines. Each block starts with the number, n, of the training images of one trial, followed by n lines, with each line containing the index of an image and the filename of the image, separated by a comma and a space.

-test_set.txt, gallery_set.txt, probe_set.txt:
	These three files contain the indexes, as well as the filenames of the test images, gallery images, and probe images, respectively, for 10 trials. The formats of these files are the same as the train_set.txt. Note that for each trial, the test set is the union of the gallery set and the probe set, where the test set is used for the performance evaluation of the face verification scenario, and the gallery and probe sets are used for performance evaluation of the open-set face identification scenario.

