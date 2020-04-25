import os
import argparse
import numpy as np
import tensorflow as tf
import blurf_config

# Save the result
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def createVerificationForBlufr(path_to_save, trails_list, labels):
    '''Create Verification for Blufr. For each trail save independent file (Memory Issue)'''
    filename = os.path.join(path_to_save, blurf_config.blufr_lfw_test_ver_file)
    create_dir(os.path.dirname(filename))
    list_filename = list()
    for num_train, trail in enumerate(trails_list):
        genuine_pairs = list()
        impostor_pairs = list()
        border_idx = len(trail) * blurf_config.blufr_lfw_ver_part
        num_part = 0
        for idx, idx_main in enumerate(trail):
            print(idx, len(trail))
            for idx_pair in trail:
                # If images have same labels, add then to genuine list
                if labels[idx_main] == labels[idx_pair]:
                    genuine_pairs.append((idx_main, idx_pair, 1))
                else:
                    # add to impostor list
                    impostor_pairs.append((idx_main, idx_pair, 0))

                if idx > border_idx:
                    # If there is enought pairs, save them to file
                    all_data = (genuine_pairs, impostor_pairs)
                    filename = os.path.join(path_to_save, blurf_config.blufr_lfw_test_ver_path.format(num_train, num_part))
                    # save_cPickle(all_data, filename)# save the file
                    filename = os.path.join(path_to_save, blurf_config.blufr_lfw_test_ver_path)
                    np.save(filename, all_data)
                    list_filename.append(filename)
                    num_part += 1
                    border_idx += len(trail) * blurf_config.blufr_lfw_ver_part
                    genuine_pairs = list()
                    impostor_pairs = list()

    filename = os.path.join(path_to_save, blurf_config.blufr_lfw_test_ver_file)
    with open(filename, 'w') as f:
        f.writelines("%s\n" % item for item in list_filename)


def loadFormatVerification(directory, data_file):
    # read data file
    path_to_file = os.path.join(directory, data_file)
    with open(path_to_file, 'r') as f:
        data = [line.strip() for line in f.readlines()]
    num_trials = int(data[0])
    curr_idx = 1
    all_trials = list()
    for num_train in range(num_trials):
        # read how many images in current trial
        how_many_img = int(data[curr_idx])
        curr_idx += 1
        new_train = list()
        print( "Trail: ", num_train, " from ", num_trials, " Num Img ", how_many_img)
        for img in range(how_many_img):
            idx_from_list = int(data[curr_idx].split(",")[0])
            # substact '1' to start from index '0' (original files was developed for MatLab)
            new_train.append(idx_from_list - 1)
            curr_idx += 1
        all_trials.append(np.asarray(new_train))

    return all_trials


def loadProtocolFiles(directory, path_to_save):
    # 1. Read labels file
    dir_lfw_labels = os.path.join(directory, blurf_config.blufr_lfw_labels)
    with open(dir_lfw_labels, 'r') as f:
        labels = np.asarray([int(line.strip()) for line in f.readlines()])
    filename = os.path.join(path_to_save, blurf_config.blufr_lfw_label_path)
    np.save(filename, labels)

    # 2. Read test_set file
    list_trials_test_set = loadFormatVerification(directory, blurf_config.blufr_lfw_test_set)
    filename = os.path.join(path_to_save, blurf_config.blufr_lfw_test_set_path)
    np.save(filename, list_trials_test_set)
    # 3. Read gallery file
    list_trials_gallery = loadFormatVerification(directory, blurf_config.blufr_lfw_gallery)
    filename = os.path.join(path_to_save, blurf_config.blufr_lfw_gallery_path)
    np.save(filename, list_trials_gallery)
    # 4. Read probe file
    list_trials_probe = loadFormatVerification(directory, blurf_config.blufr_lfw_probe)
    filename = os.path.join(path_to_save, blurf_config.blufr_lfw_probe_path)
    np.save(filename, list_trials_probe)

    # Memory Issue
    # 5. Create Verification task for Test_Set
    ver_trail_on_test_set = createVerificationForBlufr(path_to_save, list_trials_probe, labels)



'''Loading DataBase'''


def _transform_images(is_ccrop=False):
    def transform_images(x_train):
        x_train = tf.image.resize(x_train, (128, 128))
        x_train = tf.image.random_crop(x_train, (112, 112, 3))
        x_train = tf.image.random_flip_left_right(x_train)
        x_train = tf.image.random_saturation(x_train, 0.6, 1.4)
        x_train = tf.image.random_brightness(x_train, 0.4)
        x_train = x_train / 255
        return x_train
    return transform_images


def loadImage(path):
    img = tf.image.decode_jpeg(path, channels=3)
    img = _transform_images(is_ccrop=False)(img)
    return img


def loadImagesList(list_image):
    data = []
    labels = []
    for image in list_image:
        image = loadImage(image)
        data.append(image)

    return np.asarray(data).astype(np.float16)


def loadImagesDatabase(directory, color, path_to_save, name):
    # Load images to numpy array

    # 1. Read LFW path
    dir_lfw_file = os.path.join(directory, blurf_config.blufr_lfw_path_to_lfw)
    with open(dir_lfw_file, 'r') as f:
        path_to_lfw = f.readlines()[0]

    # 2. Read Images
    names_file = os.path.join(directory, blurf_config.blufr_lfw_image_file)
    filenames = []
    with open(names_file, 'r') as f:
        lines = [os.path.join(path_to_lfw, "_".join(line.strip().split("_")[:-1]), line.strip()) for line in
                 f.readlines()]
        data = loadImagesList(lines, color)
        filename = os.path.join(path_to_save, name + "_" + blurf_config.blufr_lfw_data_path)
        np.save(filename, data)
        filenames.append(filename)
        filename_data = os.path.join(path_to_save, blurf_config.name_data_filename)

        with open(filename_data, 'wb') as f:
            f.writelines("%s\n" % item for item in filenames)
        print("Data Saved")



if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Create BLUFR evaulation based on LFW ")
    parser.add_argument("--directory",default='../data/BLUFR/list/lfw/', help="BLUFR directory list of images (/BLUFT/list/lfw/)")
    # parser.add_argument("name", help="name of dataset, based on alignment method")
    args = parser.parse_args()
    path_to_save = '../data/test_dataset/LFW_BLUFR'
    create_dir(path_to_save)
    # Load the dataset and format it properly
    print("Read Images")

    # loadImagesDatabase(args.directory, args.color, path_to_save, args.name )
    loadProtocolFiles(args.directory, path_to_save)

    print("Data saved to %s" % path_to_save)
