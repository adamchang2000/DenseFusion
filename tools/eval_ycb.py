import _init_paths
import argparse
import os
import copy
import random
import numpy as np
from PIL import Image
import scipy.io as scio
import scipy.misc
import numpy.ma as ma
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.autograd import Variable
from datasets.ycb.dataset import PoseDataset
from lib.network import PoseNet, PoseRefineNet
import cv2

from collections import defaultdict
from knn_cuda import KNN

def cal_auc(add_dis):
        max_dis = 0.1
        D = np.array(add_dis)
        D[np.where(D > max_dis)] = np.inf;
        D = np.sort(D)
        n = len(add_dis)
        acc = np.cumsum(np.ones((1,n)), dtype=np.float32) / n
        aps = VOCap(D, acc)
        return aps * 100

def VOCap(rec, prec):
    idx = np.where(rec != np.inf)
    if len(idx[0]) == 0:
        return 0
    rec = rec[idx]
    prec = prec[idx]
    mrec = np.array([0.0]+list(rec)+[0.1])
    mpre = np.array([0.0]+list(prec)+[prec[-1]])
    for i in range(1, prec.shape[0]):
        mpre[i] = max(mpre[i], mpre[i-1])
    i = np.where(mrec[1:] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[i] - mrec[i-1]) * mpre[i]) * 10
    return ap

try:
    from lib.tools import compute_rotation_matrix_from_ortho6d
except:
    from tools import compute_rotation_matrix_from_ortho6d

os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default = '', help='dataset root dir')
parser.add_argument('--model', type=str, default = '',  help='resume PoseNet model')
parser.add_argument('--refine_model', type=str, default = '',  help='resume PoseRefineNet model')
parser.add_argument('--output', type=str, default='visualization', help='output for point vis')
parser.add_argument('--use_normals', action="store_true", default=False, help="estimate normals and augment pointcloud")
parser.add_argument('--iteration', type=int, default = 2, help='number of refinement iterations')
opt = parser.parse_args()

if not os.path.isdir(opt.output):
    os.mkdir(opt.output)

norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
xmap = np.array([[j for i in range(640)] for j in range(480)])
ymap = np.array([[i for i in range(640)] for j in range(480)])
cam_cx = 312.9869
cam_cy = 241.3109
cam_fx = 1066.778
cam_fy = 1067.487
cam_scale = 10000.0
num_obj = 21
img_width = 480
img_length = 640
num_points = 1000
num_points_mesh = 500
iteration = 2
bs = 1
dataset_config_dir = 'datasets/ycb/dataset_config'
ycb_toolbox_dir = 'YCB_Video_toolbox'
result_wo_refine_dir = 'experiments/eval_result/ycb/Densefusion_wo_refine_result'
result_refine_dir = 'experiments/eval_result/ycb/Densefusion_iterative_result'

estimator = PoseNet(num_points = num_points, num_obj = num_obj, use_normals = opt.use_normals)
estimator.cuda()
estimator.load_state_dict(torch.load(opt.model))
estimator.eval()

if opt.refine_model != '':
    refiner = PoseRefineNet(num_points = num_points, num_obj = num_obj, use_normals = opt.use_normals)
    refiner.cuda()
    refiner.load_state_dict(torch.load(opt.refine_model))
    refiner.eval()

test_dataset = PoseDataset('test', num_points, False, opt.dataset_root, 0.0, opt.refine_model != '', use_normals = opt.use_normals)

def get_pointcloud(model_points, t, rot_mat):

    model_points = model_points.cpu().detach().numpy()

    pts = (model_points @ rot_mat.T + t).squeeze()

    return pts

def project_points(pts, cam_fx, cam_fy, cam_cx, cam_cy):
    proj_mat = np.array([[cam_fx, 0, cam_cx], [0, cam_fy, cam_cy], [0, 0, 1]])
    projected_pts = pts @ proj_mat.T
    projected_pts /= np.expand_dims(projected_pts[:,2], -1)
    projected_pts = projected_pts[:,:2]
    return projected_pts

colors = [(96, 60, 20), (156, 39, 6), (212, 91, 18), (243, 188, 46), (95, 84, 38)]

adds = defaultdict(list)
add = defaultdict(list)

knn = KNN(k=1, transpose_mode=True)

for now in range(len(test_dataset)):
    
    data_objs = test_dataset.get_all_objects(now)

    color_img_file = '{0}/{1}-color.png'.format(test_dataset.root, test_dataset.list[now])
    color_img = cv2.imread(color_img_file)

    for obj_idx, data in enumerate(data_objs):
        torch.cuda.empty_cache()

        data, intrinsics = data
        cam_fx, cam_fy, cam_cx, cam_cy = intrinsics

        data = [torch.unsqueeze(d, 0) for d in data]

        points, choose, img, target, model_points, idx = data
        points, choose, img, target, model_points, idx = Variable(points).cuda(), \
                                                                 Variable(choose).cuda(), \
                                                                 Variable(img).cuda(), \
                                                                 Variable(target).cuda(), \
                                                                 Variable(model_points).cuda(), \
                                                                 Variable(idx).cuda()

        pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)

        bs, num_p, _ = pred_c.shape
        pred_c = pred_c.view(bs, num_p)
        how_max, which_max = torch.max(pred_c, 1)
        pred_t = pred_t.view(bs * num_p, 1, 3)

        my_r = pred_r[0][which_max[0]].view(-1).unsqueeze(0).unsqueeze(0)

        my_rot_mat = compute_rotation_matrix_from_ortho6d(my_r)[0].cpu().data.numpy()

        #print('my rot mat', my_rot_mat)

        if opt.use_normals:
            points = points.contiguous().view(bs*num_p, 1, 6)
            normals = points[:,:,3:].contiguous()
            points = points[:,:,:3].contiguous()
        else:
            points = points.contiguous().view(bs*num_p, 1, 3)

        my_t = (points + pred_t)[which_max[0]].view(-1).cpu().data.numpy()

        if opt.refine_model != "":
            for ite in range(0, opt.iteration):
                T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(num_p, 1).contiguous().view(1, num_p, 3)
                rot_mat = my_rot_mat

                #print('iter!', ite, my_rot_mat)

                my_mat = np.zeros((4,4))
                my_mat[0:3,0:3] = rot_mat
                my_mat[3, 3] = 1

                #print(my_mat, my_mat.shape)

                R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
                my_mat[0:3, 3] = my_t

                points = points.view(bs, num_p, 3)
                
                new_points = torch.bmm((points - T), R).contiguous()

                if opt.use_normals:
                    normals = normals.view(bs, num_p, 3)
                    new_normals = torch.bmm((normals - T), R).contiguous()
                    new_points = torch.concat((new_points, new_normals), dim=2)

                pred_r, pred_t = refiner(new_points, emb, idx)
                pred_r = pred_r.view(1, 1, -1)
                pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
                my_r_2 = pred_r.view(-1).unsqueeze(0).unsqueeze(0)
                my_t_2 = pred_t.view(-1).cpu().data.numpy()
                rot_mat_2 = compute_rotation_matrix_from_ortho6d(my_r_2)[0].cpu().data.numpy()

                my_mat_2 = np.zeros((4, 4))
                my_mat_2[0:3,0:3] = rot_mat_2
                my_mat_2[3,3] = 1
                my_mat_2[0:3, 3] = my_t_2

                my_mat_final = np.dot(my_mat, my_mat_2)
                my_r_final = copy.deepcopy(my_mat_final)
                my_rot_mat[0:3,0:3] = my_r_final[0:3,0:3]
                my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

                #my_pred = np.append(my_r_final, my_t_final)
                my_t = my_t_final
        # Here 'my_pred' is the final pose estimation result after refinement ('my_r': quaternion, 'my_t': translation)

        my_r = copy.deepcopy(my_rot_mat)
        pred = get_pointcloud(model_points, my_t, my_r)
        pred = torch.unsqueeze(torch.from_numpy(pred.astype(np.float32)), 0).cuda()

        add_dist = torch.mean(torch.norm(target - pred, dim=2)).detach().cpu().item()

        adds_dists, inds = knn(target, pred)
        adds_dist = torch.mean(adds_dists).detach().cpu().item()
        idx = idx.detach().cpu().item()

        print("frame, idx, adds, add", now, idx, adds_dist, add_dist)

        adds[idx].append(adds_dist)
        add[idx].append(add_dist)

        for d in data:
            del d

adds_aucs = {}
add_aucs = {}

for idx, dists in adds.items():
    adds_aucs[idx] = cal_auc(dists)

for idx, dists in add.items():
    add_aucs[idx] = cal_auc(dists)

print("ADDS AUCs")
print(adds_aucs)

print("ADD AUCs")
print(add_aucs)
