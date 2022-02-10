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
from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod
from lib.network import PoseNet
from lib.loss import Loss

from lib.transformations import rotation_matrix_from_vectors_procedure, rotation_matrix_of_axis_angle

import open3d as o3d

def dist2(x, y):
    nx, dimx = x.shape
    ny, dimy = y.shape
    assert(dimx == dimy)

    return (np.ones((ny, 1)) * np.sum((x**2).T, axis=0)).T + \
        np.ones((nx, 1)) * np.sum((y**2).T, axis=0) - \
        2 * np.inner(x, y)

def get_points(model_points, front_orig, front, angle, t):

    model_points = model_points.cpu().detach().numpy()
    front_orig = front_orig.cpu().detach().numpy().squeeze()
    front = front.cpu().detach().numpy()
    angle = angle.cpu().detach().numpy()
    t = t.cpu().detach().numpy()

    Rf = rotation_matrix_from_vectors_procedure(front_orig, front)

    R_axis = rotation_matrix_of_axis_angle(front, angle)

    R_tot = (R_axis @ Rf)

    pts = (model_points @ R_tot.T + t).squeeze()

    return pts

def visualize_points(model_points, front_orig, front, angle, t, label):

    model_points = model_points.cpu().detach().numpy()
    front_orig = front_orig.cpu().detach().numpy().squeeze()
    front = front.cpu().detach().numpy()
    angle = angle.cpu().detach().numpy()
    t = t.cpu().detach().numpy()

    Rf = rotation_matrix_from_vectors_procedure(front_orig, front)

    R_axis = rotation_matrix_of_axis_angle(front, angle)

    R_tot = (R_axis @ Rf)

    pts = (model_points @ R_tot.T + t).squeeze()

    pcld = o3d.geometry.PointCloud()
    pts = o3d.utility.Vector3dVector(pts)

    pcld.points = pts
    
    o3d.io.write_point_cloud(label + ".ply", pcld)

def visualize_pointcloud(points, label):

    points = points.cpu().detach().numpy()

    points = points.reshape((-1, 3))

    pcld = o3d.geometry.PointCloud()
    points = o3d.utility.Vector3dVector(points)

    pcld.points = points

    o3d.io.write_point_cloud(label + ".ply", pcld)

def visualize_fronts(fronts, t, label):
    fronts = fronts.cpu().detach().numpy()
    fronts = fronts.reshape((-1, 3))

    t = t.cpu().detach().numpy()
    t = t.reshape((-1, 3))

    front_points = fronts + t

    front_pcld = o3d.geometry.PointCloud()
    front_points = o3d.utility.Vector3dVector(front_points)

    front_pcld.points = front_points

    o3d.io.write_point_cloud(label + ".ply", front_pcld)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default = 'ycb', help='ycb or linemod')
    parser.add_argument('--dataset_root', type=str, default = '', help='dataset root dir (''YCB_Video_Dataset'' or ''Linemod_preprocessed'')')
    parser.add_argument('--workers', type=int, default = 1, help='number of data loading workers')
    parser.add_argument('--model', type=str, default = '',  help='PoseNet model')
    parser.add_argument('--w', default=0.015, help='regularize confidence')
    parser.add_argument('--w_rate', default=0.3, help='regularize confidence refiner decay')

    parser.add_argument('--num_rot_bins', type=int, default = 90, help='number of bins discretizing the rotation around front')
    parser.add_argument('--num_visualized', type=int, default = 5, help='number of training samples to visualize')
    parser.add_argument('--image_size', type=int, default=300, help="square side length of cropped image")
    opt = parser.parse_args()

    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if opt.dataset == 'ycb':
        opt.num_objects = 21 #number of object classes in the dataset
        opt.num_points = 1000 #number of points on the input pointcloud
        opt.outf = 'trained_models/ycb' #folder to save trained models
    else:
        print('ONLY YCB')
        exit(-1)

    
    if opt.model != '':
        estimator = PoseNet(num_points = opt.num_points, num_obj = opt.num_objects, num_rot_bins = opt.num_rot_bins)
        estimator.cuda()
        estimator = nn.DataParallel(estimator)
        estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.model)))

    if not opt.model:
        raise Exception("this is visualizer code, pls pass in a model lol")

    if opt.dataset == 'ycb':
        test_dataset = PoseDataset_ycb('train', opt.num_points, False, opt.dataset_root, 0.0, opt.num_rot_bins, opt.image_size)
    else:
        exit("only ycb supported here")
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)
    
    opt.sym_list = test_dataset.get_sym_list()
    opt.num_points_mesh = test_dataset.get_num_points_mesh()

    print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the testing set: {0}\nnumber of sample points on mesh: {1}\nsymmetry object list: {2}'.format(len(test_dataset), opt.num_points_mesh, opt.sym_list))

    criterion = Loss(opt.num_rot_bins)

    estimator.eval()

    dists = []

    for i, data in enumerate(testdataloader, 0):
        points, choose, img, front_r, rot_bins, front_orig, t, model_points, idx = data

        points, choose, img, front_r, rot_bins, front_orig, t, model_points, idx = Variable(points).cuda(), \
                                                            Variable(choose).cuda(), \
                                                            Variable(img).cuda(), \
                                                            Variable(front_r).cuda(), \
                                                            Variable(rot_bins).cuda(), \
                                                            Variable(front_orig).cuda(), \
                                                            Variable(t).cuda(), \
                                                            Variable(model_points).cuda(), \
                                                            Variable(idx).cuda()

        pred_front, pred_rot_bins, pred_t, pred_c, emb = estimator(img, points, choose, idx)
        loss, new_points, new_rot_bins, new_t, new_front_r = criterion(pred_front, pred_rot_bins, pred_t, pred_c, front_r, rot_bins, front_orig, t, idx, model_points, points, opt.w)

        #immediately shift pred_front to model coordinates
        pred_front = pred_front - pred_t

        bs, num_p, _ = pred_c.shape

        pred_c = pred_c.view(bs, num_p)
        how_max, which_max = torch.max(pred_c, 1)

        visualize_pointcloud(pred_t + points, "{0}_pred_t_points".format(i))

        visualize_pointcloud(t, "{0}_gt_t".format(i))
        visualize_pointcloud(t - new_t, "{0}_pred_t".format(i))

        visualize_fronts(front_r, t, "{0}_gt_front".format(i))
        visualize_fronts(pred_front, pred_t + points, "{0}_pred_fronts_estimator".format(i))

        visualize_pointcloud(new_points, "{0}_new_points".format(i))
        visualize_fronts(new_front_r, new_t, "{0}_new_gt_front".format(i))

        #which_max -> bs
        #pred_t -> bs * num_p * 3
        #points -> bs * num_p * 3

        which_max_3 = which_max.view(bs, 1, 1).repeat(1, 1, 3)
        which_max_rot_bins = which_max.view(bs, 1, 1).repeat(1, 1, opt.num_rot_bins)

        #best_c_pred_t -> bs * 1 * 3
        best_c_pred_t = torch.gather(pred_t, 1, which_max_3) + torch.gather(points, 1, which_max_3)

        #we need to calculate the actual transformation that our rotation rep. represents

        best_c_pred_front = torch.gather(pred_front, 1, which_max_3).squeeze(1)
        best_c_rot_bins = torch.gather(pred_rot_bins, 1, which_max_rot_bins).squeeze(1)

        #get the angle in radians based on highest histogram bin
        #angle -> bs * 1
        angle = (torch.argmax(best_c_rot_bins, axis=1) / best_c_rot_bins.shape[1] * 2 * np.pi)

        my_front = best_c_pred_front.squeeze()
        my_theta = angle.squeeze()
        my_t = best_c_pred_t.squeeze()

        gt_front = front_r.squeeze()
        gt_theta = torch.argmax(rot_bins) / opt.num_rot_bins * 2 * np.pi
        gt_t = t.squeeze()

        visualize_fronts(my_front, my_t, "{0}_selected_pred_front".format(i))

        #pts_gt = get_points(model_points, front_orig, gt_front, gt_theta, gt_t)
        #pts_pred = get_points(model_points, front_orig, my_front, my_theta, my_t)

        #dist = dist2(pts_gt, pts_pred)
        #dists.append(np.mean(np.min(dist, axis=1)))

        visualize_points(model_points, front_orig, gt_front, gt_theta, gt_t, "{0}_gt".format(i))
        visualize_points(model_points, front_orig, my_front, gt_theta, my_t, "{0}_pred".format(i))
        visualize_pointcloud(points, "{0}_projected_depth".format(i))

        if i >= opt.num_visualized:
            print("finished visualizing!")
            exit()

    #print(np.mean(dists))


if __name__ == '__main__':
    main()
