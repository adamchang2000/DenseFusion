import scipy
import scipy.io
import numpy as np
import os
import re
import json
from PIL import Image
import shutil

p = np.array([[ 0, 0, 1],
    [ 1, 0, 0],
    [ 0,-1, 0]])


def convert_data(root_dir = "custom_dataset/", destination_dir = "converted_custom_data/", data_paths = []):
    dictionary = {}
    dictionary["factor_depth"] = 65535.0 / 10

    for folder in data_paths:
        os.makedirs(os.path.join(destination_dir, folder), exist_ok = True)
        with open(root_dir + folder + "/_object_settings.json") as file:
            data = json.load(file)
            initial_rotation = np.array(data["exported_objects"][0]["fixed_model_transform"])[:3, :3] / 100
            if type(data["exported_objects"][0]["segmentation_class_id"]) == int:
                dictionary["cls_indexes"] = np.array([[data["exported_objects"][0]["segmentation_class_id"]]])
            elif type(data["exported_objects"][0]["segmentation_class_id"]) == list:
                dictionary["cls_indexes"] = np.array([data["exported_objects"][0]["segmentation_class_id"]]).reshape((-1, 1))

        for f in os.listdir(root_dir + folder):
            if re.fullmatch(r'\d{6}\.left\.cs\.png', f):
                shutil.copy(root_dir + folder + "/" + f, destination_dir + folder + "/" + f[:6] + "-label.png")
            elif re.fullmatch(r'\d{6}\.left\.depth\.16\.png', f):
                shutil.copy(root_dir + folder + "/" + f, destination_dir + folder + "/" + f[:6] + "-depth.png")
            elif re.fullmatch(r'\d{6}\.left\.png', f):
                shutil.copy(root_dir + folder + "/" + f, destination_dir + folder + "/" + f[:6] + "-color.png")
            elif re.fullmatch(r'\d{6}\.left\.json', f):
                with open(root_dir + folder + "/" + f) as file:
                    data = json.load(file)
                obj_class = data["objects"][0]["class"]
                if type(obj_class) == str:
                    top_left = str(data["objects"][0]["bounding_box"]["top_left"][0]) + " " + str(data["objects"][0]["bounding_box"]["top_left"][1])
                    bottom_right = str(data["objects"][0]["bounding_box"]["bottom_right"][0]) + " " + str(data["objects"][0]["bounding_box"]["bottom_right"][1])
                    bounding_box = top_left + " " + bottom_right
                    content = obj_class + " " + bounding_box
                    file = open(destination_dir + folder + "/" + f[:6] + "-box.txt", "w")
                    file.write(content)

                    relative_rotation = np.array(data["objects"][0]["pose_transform"])[:3, :3]
                    absolute_translation = np.array(data["objects"][0]["pose_transform"]).T[:3, -1:] / 100
                    absolute_rotation = relative_rotation.T @ p @ initial_rotation.T
                    poses = np.concatenate((absolute_rotation, absolute_translation), axis = 1)
                    poses = poses[..., None]
                    dictionary["poses"] = poses

                elif type(obj_class) == list:
                    for i in range(len(obj_class) - 1):
                        top_left = str(data["objects"][0]["bounding_box"]["top_left"][i][0]) + " " + str(data["objects"][0]["bounding_box"]["top_left"][i][1])
                        bottom_right = str(data["objects"][0]["bounding_box"]["bottom_right"][i][0]) + " " + str(data["objects"][0]["bounding_box"]["bottom_right"][i][1])
                        bounding_box = top_left + " " + bottom_right
                        content = obj_class[i] + " " + bounding_box + "\n"
                        f = open(destination_dir + folder + "/" + f[:6] + "-box.txt", "a")
                        f.write(content)
                    top_left = str(data["objects"][0]["bounding_box"]["top_left"][len(obj_class) - 1][0]) + " " + str(data["objects"][0]["bounding_box"]["top_left"][len(obj_class) - 1][1])
                    bottom_right = str(data["objects"][0]["bounding_box"]["bottom_right"][len(obj_class) - 1][0]) + " " + str(data["objects"][0]["bounding_box"]["bottom_right"][len(obj_class) - 1][1])
                    bounding_box = top_left + " " + bottom_right
                    content = obj_class[len(obj_class) - 1] + " " + bounding_box
                    f = open(destination_dir + folder + "/" + f[:6] + "-box.txt", "a")
                    f.write(content)

                scipy.io.savemat(destination_dir + folder + "/" + f[:6] + "-meta.mat", dictionary)

if __name__ == "__main__":
    convert_data() # specify folders and root directories here!
