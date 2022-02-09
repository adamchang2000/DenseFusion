import scipy
import scipy.io
import numpy as np
import numpy.ma as ma
import os
import re
import json
from PIL import Image
import shutil
import random
random.seed(123)

from dataset import standardize_image_size


def clean_idxs(idx_file, item):
    clean_idxs = []

    counter = 0
    while 1:
        input_line = idx_file.readline()
        if not input_line:
            break
        if input_line[-1:] == '\n':
            input_line = input_line[:-1]
        prefix = '{}/data/{}/{}'.format(root, item, input_line)

        with open(prefix + ".left.json") as cam_config:
            meta = json.load(cam_config)

            #remove samples with no object
            if len(meta['objects']) == 0:
                print("no obj", prefix)
                continue

            bbox = meta["objects"][0]["bounding_box"]
            rmin, rmax, cmin, cmax = int(bbox["top_left"][0]), int(bbox["bottom_right"][0]), int(bbox["top_left"][1]), int(bbox["bottom_right"][1])
            rmin, rmax, cmin, cmax = max(0, rmin), min(h, rmax), max(0, cmin), min(w, cmax)

            rmin, rmax, cmin, cmax = standardize_image_size(target_image_size, rmin, rmax, cmin, cmax, h, w)

            #IMAGE PATCH must be at least 4x4
            if (rmin + 3 >= rmax):
                print("r small", prefix)
                continue

            if (cmin + 3 >= cmax):
                print("c small", prefix)
                continue

            depth = np.array(Image.open(prefix + ".left.depth.16.png"))
            label = np.array(Image.open(prefix + ".left.cs.png"))

            with open("{}/data/{}/_object_settings.json".format(root, item)) as object_config:
                object_data = json.load(object_config)
                object_class_id = object_data["exported_objects"][0]["segmentation_class_id"]

            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(object_class_id)))

            mask = mask_label * mask_depth
            choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

            if (len(choose) == 0):
                print("choose 0", prefix)
                continue

            clean_idxs.append(input_line)
            counter += 1

            if counter % 1000 == 0:
                print("{0} clean samples".format(counter))


    return clean_idxs


def convert_data(root_dir = "custom_data/", destination_dir = "converted_custom_data/"):
    dictionary = {}
    dictionary["factor_depth"] = 65535.0 / 10

    os.makedirs(os.path.join(destination_dir, "."), exist_ok = True)
    with open(root_dir + "_object_settings.json") as file:
        data = json.load(file)
        initial_rotation = np.array(data["exported_objects"][0]["fixed_model_transform"]).T[:3, :3] / 100
        if type(data["exported_objects"][0]["segmentation_class_id"]) == int:
            dictionary["cls_indexes"] = np.array([[data["exported_objects"][0]["segmentation_class_id"]]])
        elif type(data["exported_objects"][0]["segmentation_class_id"]) == list:
            dictionary["cls_indexes"] = np.array([data["exported_objects"][0]["segmentation_class_id"]]).reshape((-1, 1))

    for f in os.listdir(root_dir):
        if re.fullmatch(r'\d{6}\.left\.cs\.png', f):
            shutil.copy(root_dir + f, destination_dir + f[:6] + "-label.png")
        elif re.fullmatch(r'\d{6}\.left\.depth\.16\.png', f):
            shutil.copy(root_dir + f, destination_dir + f[:6] + "-depth.png")
        elif re.fullmatch(r'\d{6}\.left\.png', f):
            shutil.copy(root_dir + f, destination_dir + f[:6] + "-color.png")
        elif re.fullmatch(r'\d{6}\.left\.json', f):
            with open(root_dir + f) as file:
                data = json.load(file)
            obj_class = data["objects"][0]["class"]
            if type(obj_class) == str:
                top_left = str(data["objects"][0]["bounding_box"]["top_left"][0]) + " " + str(data["objects"][0]["bounding_box"]["top_left"][1])
                bottom_right = str(data["objects"][0]["bounding_box"]["bottom_right"][0]) + " " + str(data["objects"][0]["bounding_box"]["bottom_right"][1])
                bounding_box = top_left + " " + bottom_right
                content = obj_class + " " + bounding_box
                file = open(destination_dir + f[:6] + "-box.txt", "w")
                file.write(content)

                initial_rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
                relative_rotation = np.array(data["objects"][0]["pose_transform"]).T[:3, :3]
                absolute_translation = np.array(data["objects"][0]["pose_transform"]).T[:3, -1:] / 100
                absolute_rotation = np.dot(relative_rotation, initial_rotation)
                poses = np.concatenate((absolute_rotation, absolute_translation), axis = 1)
                print(poses.shape)
                poses = poses[..., None]
                print(poses.shape)
                dictionary["poses"] = poses

            elif type(obj_class) == list:
                for i in range(len(obj_class) - 1):
                    top_left = str(data["objects"][0]["bounding_box"]["top_left"][i][0]) + " " + str(data["objects"][0]["bounding_box"]["top_left"][i][1])
                    bottom_right = str(data["objects"][0]["bounding_box"]["bottom_right"][i][0]) + " " + str(data["objects"][0]["bounding_box"]["bottom_right"][i][1])
                    bounding_box = top_left + " " + bottom_right
                    content = obj_class[i] + " " + bounding_box + "\n"
                    f = open(destination_dir + f[:6] + "-box.txt", "a")
                    f.write(content)
                top_left = str(data["objects"][0]["bounding_box"]["top_left"][len(obj_class) - 1][0]) + " " + str(data["objects"][0]["bounding_box"]["top_left"][len(obj_class) - 1][1])
                bottom_right = str(data["objects"][0]["bounding_box"]["bottom_right"][len(obj_class) - 1][0]) + " " + str(data["objects"][0]["bounding_box"]["bottom_right"][len(obj_class) - 1][1])
                bounding_box = top_left + " " + bottom_right
                content = obj_class[len(obj_class) - 1] + " " + bounding_box
                f = open(destination_dir + f[:6] + "-box.txt", "a")
                f.write(content)

            scipy.io.savemat(destination_dir + f[:6] + "-meta.mat", dictionary)

if __name__ == "__main__":
    root = "custom_preprocessed"
    target_image_size = 20
    objs = [1]
    h, w = 720, 1280

    for item in objs:
        # Split data.
        path = "custom_preprocessed/data/" + str(item)
        files = [f.split('.')[0] for f in os.listdir(path) if f.find("left.png") != -1]
        total_data_num = len(files)
        print("There are total {} data".format(total_data_num))

        ratio = 0.8
        random.shuffle(files)
        train_data = files[:  int(total_data_num * ratio)]
        test_data = files[int(total_data_num * ratio):]
        print("Split into {} training data, {} testing data".format(len(train_data), len(test_data)))

        os.makedirs(os.path.join(root + "/split/", str(item)), exist_ok = True)
        train_output_path = root + "/split/" + str(item) + "/train.txt"
        with open(train_output_path, "w") as textfile:
            for data in train_data:
                textfile.write(data + "\n")
        print("Write train data into file: {}".format(train_output_path))

        os.makedirs(os.path.join(root + "/split/", str(item)), exist_ok = True)
        test_output_path = root + "/split/" + str(item) + "/test.txt"
        with open(test_output_path, "w") as textfile:
            for data in test_data:
                textfile.write(data + "\n")
        print("Write test data into file: {}".format(test_output_path))

        # Clean data.
        train_file = open('{}/split/{}/train.txt'.format(root, item))
        test_file = open('{}/split/{}/test.txt'.format(root, item))

        clean_train_idxs = clean_idxs(train_file, item)

        clean_test_idxs = clean_idxs(test_file, item)

        print(len(clean_train_idxs), len(clean_test_idxs))

        train_file.close()
        test_file.close()

        with open("train_clean.txt", "w") as f:
            for clean_train_idx in clean_train_idxs:
                f.write(clean_train_idx + "\n")
        os.rename("train_clean.txt", '{}/split/{}/train.txt'.format(root, item))

        with open("test_clean.txt", "w") as f:
            for clean_test_idx in clean_test_idxs:
                f.write(clean_test_idx + "\n")
        os.rename("test_clean.txt", '{}/split/{}/test.txt'.format(root, item))

    # Data conversion.
    convert_data(root_dir = "custom_preprocessed/data/1/", destination_dir = "converted_custom_preprocessed/data/1/") # specify folders and root directories here!
