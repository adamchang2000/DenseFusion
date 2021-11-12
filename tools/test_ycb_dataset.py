# --------------------------------------------------------
# DenseFusion 6D Object Pose Estimation by Iterative Dense Fusion
# Licensed under The MIT License [see LICENSE for details]
# Written by Chen
# --------------------------------------------------------

import _init_paths
import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from datasets.ycb.dataset import PoseDataset as PoseDataset_ycb
from lib.transformations import rotation_matrix_from_vectors_procedure, rotation_matrix_of_axis_angle

import open3d as o3d

def visualize_points_from_rt(model_points, r, t, label):

    print("rt r", r)

    pts = (model_points @ r.T + t).squeeze()

    pcld = o3d.geometry.PointCloud()
    pts = o3d.utility.Vector3dVector(pts)

    pcld.points = pts

    o3d.io.write_point_cloud(label + ".ply", pcld)



def visualize_points(model_points, front_orig, front, angle, t, label):
    Rf = rotation_matrix_from_vectors_procedure(front_orig, front)

    R_axis = rotation_matrix_of_axis_angle(front, angle)

    R_tot = (R_axis @ Rf)

    print("our rot rep", R_tot)

    pts = (model_points @ R_tot.T + t).squeeze()

    pcld = o3d.geometry.PointCloud()
    pts = o3d.utility.Vector3dVector(pts)

    pcld.points = pts
    
    o3d.io.write_point_cloud(label + ".ply", pcld)

def visualize_pointcloud(points, label):
    points = points.reshape((-1, 3))

    pcld = o3d.geometry.PointCloud()
    points = o3d.utility.Vector3dVector(points)

    pcld.points = points

    o3d.io.write_point_cloud(label + ".ply", pcld)

def main():
    dataset_root = r'./datasets/ycb/YCB_Video_Dataset'
    num_points = 1000
    refine_model = False
    num_rot_bins = 360
    dataset = PoseDataset_ycb('train', num_points, True, dataset_root, 0.0, refine_model, num_rot_bins)

    data = dataset.get_item_debug(1)

    points, choose, img, front_r, rot_bins, angle, R_to_front, front_orig, r, t, model_points, idx = data

    print("rot bins dist", np.unique(rot_bins, return_counts=True))
    print("selected angle", angle)

    print("OBJECT", idx[0] + 1)
    visualize_pointcloud(points, "projected_depth")
    print("true gt")
    visualize_points_from_rt(model_points, r, t, "rot_mat_gt")
    print("no theta (only r1)")
    visualize_points_from_rt(model_points, R_to_front, t, "no_theta_gt")
    print("using rot bins")
    visualize_points(model_points, front_orig, front_r, np.argmax(rot_bins) / num_rot_bins * 2 * np.pi, t, "our_rot_rep_gt")
    print("using the calculated angle")
    visualize_points(model_points, front_orig, front_r, angle, t, "no_rot_bins_gt")



if __name__ == "__main__":
    main()
