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
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
from lib.transformations import rotation_matrix_from_vectors, rotation_matrix_of_axis_angle

def get_bbox(posecnn_rois, border_list, img_width, img_length, idx):
    rmin = int(posecnn_rois[idx][3]) + 1
    rmax = int(posecnn_rois[idx][5]) - 1
    cmin = int(posecnn_rois[idx][2]) + 1
    cmax = int(posecnn_rois[idx][4]) - 1
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax

def calculate_rot(front_orig, front_pred, theta_pred):
    Rf = rotation_matrix_from_vectors(front_orig, front_pred)

    R_axis = rotation_matrix_of_axis_angle(front_pred, theta_pred)

    return R_axis @ Rf

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, default = '', help='dataset root dir')
    parser.add_argument('--model', type=str, default = '',  help='resume PoseNet model')
    parser.add_argument('--refine_model', type=str, default = '',  help='resume PoseRefineNet model')

    parser.add_argument('--num_rot_bins', type=int, default = 36, help='number of bins discretizing the rotation around front')

    opt = parser.parse_args()

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



    estimator = PoseNet(num_points = num_points, num_obj = num_obj, num_rot_bins = opt.num_rot_bins)
    estimator.cuda()
    estimator.load_state_dict(torch.load(opt.model))
    estimator.eval()

    if opt.refine_model:
        refiner = PoseRefineNet(num_points = num_points, num_obj = num_obj, num_rot_bins = opt.num_rot_bins)
        refiner.cuda()
        refiner.load_state_dict(torch.load(opt.refine_model))
        refiner.eval()

    testlist = []
    input_file = open('{0}/test_data_list.txt'.format(dataset_config_dir))
    while 1:
        input_line = input_file.readline()
        if not input_line:
            break
        if input_line[-1:] == '\n':
            input_line = input_line[:-1]
        testlist.append(input_line)
    input_file.close()
    print(len(testlist))

    class_file = open('{0}/classes.txt'.format(dataset_config_dir))
    class_id = 1
    frontd = {}

    while 1:
        class_input = class_file.readline()
        if not class_input:
            break
        class_input = class_input[:-1]

        input_file = open('{0}/models/{1}/front.xyz'.format(opt.dataset_root, class_input))
        frontd[class_id] = []
        while 1:
            input_line = input_file.readline()
            if not input_line or len(input_line) <= 1:
                break
            input_line = input_line.rstrip().split(' ')
            frontd[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
        frontd[class_id] = np.array(frontd[class_id])
        input_file.close()

        class_id += 1

    for now in range(0, 2949):
        img = Image.open('{0}/{1}-color.png'.format(opt.dataset_root, testlist[now]))
        depth = np.array(Image.open('{0}/{1}-depth.png'.format(opt.dataset_root, testlist[now])))
        posecnn_meta = scio.loadmat('{0}/results_PoseCNN_RSS2018/{1}.mat'.format(ycb_toolbox_dir, '%06d' % now))
        label = np.array(posecnn_meta['labels'])
        posecnn_rois = np.array(posecnn_meta['rois'])

        lst = posecnn_rois[:, 1:2].flatten()
        my_result_wo_refine = []
        my_result = []
        
        for idx in range(len(lst)):
            itemid = lst[idx]
            try:
                rmin, rmax, cmin, cmax = get_bbox(posecnn_rois, border_list, img_width, img_length, idx)

                mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
                mask_label = ma.getmaskarray(ma.masked_equal(label, itemid))
                mask = mask_label * mask_depth

                choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
                if len(choose) > num_points:
                    c_mask = np.zeros(len(choose), dtype=int)
                    c_mask[:num_points] = 1
                    np.random.shuffle(c_mask)
                    choose = choose[c_mask.nonzero()]
                else:
                    choose = np.pad(choose, (0, num_points - len(choose)), 'wrap')

                depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
                xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
                ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
                choose = np.array([choose])

                pt2 = depth_masked / cam_scale
                pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
                pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
                cloud = np.concatenate((pt0, pt1, pt2), axis=1)

                img_masked = np.array(img)[:, :, :3]
                img_masked = np.transpose(img_masked, (2, 0, 1))
                img_masked = img_masked[:, rmin:rmax, cmin:cmax]

                cloud = torch.from_numpy(cloud.astype(np.float32))
                choose = torch.LongTensor(choose.astype(np.int32))
                img_masked = norm(torch.from_numpy(img_masked.astype(np.float32)))
                index = torch.LongTensor([itemid - 1])

                cloud = Variable(cloud).cuda()
                choose = Variable(choose).cuda()
                img_masked = Variable(img_masked).cuda()
                index = Variable(index).cuda()

                cloud = cloud.view(1, num_points, 3)
                img_masked = img_masked.view(1, 3, img_masked.size()[1], img_masked.size()[2])

                pred_front, pred_rot_bins, pred_t, pred_c, emb = estimator(img_masked, cloud, choose, index)

                pred_rot_bins = pred_rot_bins.view(bs * num_points, -1)
                pred_c = pred_c.view(bs, num_points)
                how_max, which_max = torch.max(pred_c, 1)
                pred_t = pred_t.view(bs * num_points, 1, 3)
                points = cloud.view(bs * num_points, 1, 3)
                pred_front = pred_front.view(bs * num_points, 3)
                #we need to calculate the actual transformation that our rotation rep. represents

                print("HERE")
                print(pred_t.shape, points.shape, pred_front.shape)

                #right now, we are only dealing with one "front" axis
                front_orig = frontd[itemid][0]

                best_c_pred_front = pred_front[which_max[0]]
                best_c_rot_bins = pred_rot_bins[which_max[0]]

                front_orig = torch.from_numpy(front_orig.squeeze().astype(np.float32))

                #calculate actual rotation
                front_orig = front_orig.cpu().detach().numpy()
                best_c_pred_front = best_c_pred_front.cpu().detach().numpy()
                best_c_rot_bins = best_c_rot_bins.cpu().detach().numpy()

                Rf = rotation_matrix_from_vectors(front_orig, best_c_pred_front)

                #get the angle in radians based on highest histogram bin
                angle = np.argmax(best_c_rot_bins) / best_c_rot_bins.shape[0] * 2 * np.pi

                R_axis = rotation_matrix_of_axis_angle(best_c_pred_front, angle)

                R_tot = (R_axis @ Rf).T

                R_tot = torch.from_numpy(R_tot.astype(np.float32)).cuda().contiguous().view(bs, 3, 3)

                print("HEREEREER")
                print(R_tot.shape)

                my_front = best_c_pred_front
                my_theta = np.argmax(best_c_rot_bins) / best_c_rot_bins.shape[0] * 2 * np.pi
                my_t = (points + pred_t)[which_max[0]].view(-1).cpu().data.numpy()


                print("setup my 'my's")
                print(my_front.shape, my_theta, my_t.shape)

                #my_result_wo_refine.append(my_pred.tolist())
                if opt.refine_model:
                    for ite in range(0, iteration):
                        R_tot = calculate_rot(front_orig, my_front, my_theta)

                        #we transpose since we are right multiplying it
                        R_tot = R_tot.T
                        R = torch.from_numpy(R_tot.astype(np.float32)).cuda().contiguous().view(bs, 3, 3)
                        T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(num_points, 1).contiguous().view(1, num_points, 3)
                        
                        new_cloud = torch.bmm((cloud - T), R).contiguous()
                        pred_r, pred_t = refiner(new_cloud, emb, index)
                        pred_r = pred_r.view(1, 1, -1)
                        pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
                        my_r_2 = pred_r.view(-1).cpu().data.numpy()
                        my_t_2 = pred_t.view(-1).cpu().data.numpy()
                        my_mat_2 = quaternion_matrix(my_r_2)

                        my_mat_2[0:3, 3] = my_t_2

                        my_mat_final = np.dot(my_mat, my_mat_2)
                        my_r_final = copy.deepcopy(my_mat_final)
                        my_r_final[0:3, 3] = 0
                        my_r_final = quaternion_from_matrix(my_r_final, True)
                        my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

                        my_pred = np.append(my_r_final, my_t_final)
                        my_r = my_r_final
                        my_t = my_t_final

                print("FINISHED RUNNING STUFF??")

                print(my_front)
                print(my_theta)

                output_file_name = str(now) + "_" + str(idx) + ".ply"

                cloud = model_points

                
            except ZeroDivisionError:
                print("PoseCNN Detector Lost {0} at No.{1} keyframe".format(itemid, now))
                my_result_wo_refine.append([0.0 for i in range(7)])
                my_result.append([0.0 for i in range(7)])

        

if __name__ == "__main__":
    main()