from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
import time
import numpy as np
import torch.nn as nn
import random
import torch.backends.cudnn as cudnn

from lib.transformations import rotation_matrix_from_vectors, rotation_matrix_of_axis_angle

cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')

#pred_r : batch_size * n * 4 -> batch_size * n * 6
def loss_calculation(pred_front, pred_rot_bins, pred_t, pred_c, front_r, rot_bins, front_orig, t, idx, model_points, points, w, refine, num_point_mesh, num_rot_bins):
    bs, num_p, _ = pred_c.size()

    print("loss time!", pred_front.shape, pred_rot_bins.shape, pred_t.shape, pred_c.shape, front_r.shape, rot_bins.shape, t.shape)

    front_r = front_r.view(bs, 1, 3).repeat(1, bs*num_p, 1)

    pred_rot_bins = pred_rot_bins.view(bs*num_p, num_rot_bins)
    rot_bins = rot_bins.repeat(bs*num_p, 1)

    print("loss resized...", front_r.shape, rot_bins.shape)

    #pred_front loss
    pred_front_dis = torch.norm((pred_front - front_r), dim=2)

    print("pred_front_dis", pred_front_dis.shape)

    print("going into cross entropy", pred_rot_bins.shape, rot_bins.shape)

    pred_rot_loss = cross_entropy_loss(pred_rot_bins, rot_bins).unsqueeze(0).unsqueeze(-1)

    print("pred_rot_loss", pred_rot_loss.shape, pred_c.shape)


    t = t.repeat(1, bs*num_p, 1)

    pred_t_loss = torch.norm((pred_t - t), dim=2).unsqueeze(-1)

    print("T LOSS", pred_t_loss.shape)

    print("t loss", pred_t.shape, t.shape)

    loss = torch.mean((pred_front_dis + pred_rot_loss + pred_t_loss) * pred_c - w * torch.log(pred_c))

    pred_t = pred_t.contiguous().view(bs * num_p, 1, 3)

    print("pred_t", pred_t.shape)
    

    #calculating new model_points for refiner
    ori_t = pred_t
    points = points.contiguous().view(bs * num_p, 1, 3)
    
    pred_c = pred_c.view(bs, num_p)
    how_max, which_max = torch.max(pred_c, 1)

    print(ori_t.shape)
    print(points.shape)

    t = ori_t[which_max[0]] + points[which_max[0]]
    points = points.view(1, bs * num_p, 3)


    #we need to calculate the actual transformation that our rotation rep. represents
    print("rot time")

    pred_front = pred_front.view(bs * num_p, 3)

    print(pred_front.shape, pred_rot_bins.shape)

    best_c_pred_front = pred_front[which_max[0]]
    best_c_rot_bins = pred_rot_bins[which_max[0]]

    print(best_c_pred_front.shape, best_c_rot_bins.shape)

    print(points.shape)

    front_orig = front_orig.squeeze()

    #calculate actual rotation

    front_orig = front_orig.cpu().detach().numpy()
    best_c_pred_front = best_c_pred_front.cpu().detach().numpy()
    best_c_rot_bins = best_c_rot_bins.cpu().detach().numpy()

    print(front_orig.shape, best_c_pred_front.shape)

    Rf = rotation_matrix_from_vectors(front_orig, best_c_pred_front)

    #get the angle in radians based on highest histogram bin
    angle = np.argmax(best_c_rot_bins) / best_c_rot_bins.shape[0] * 2 * np.pi

    R_axis = rotation_matrix_of_axis_angle(best_c_pred_front, angle)

    print(Rf.shape)
    print(R_axis.shape)

    R_tot = (R_axis @ Rf).T

    R_tot = torch.from_numpy(R_tot.astype(np.float32)).cuda().contiguous().view(bs, 3, 3)
    t = t.view(bs, 1, 3).repeat(1, num_p, 1)

    new_points = torch.bmm((points - t), R_tot).contiguous().detach()

    print(new_points.shape, points.shape)
    exit()

    # print('------------> ', dis[0][which_max[0]].item(), pred_c[0][which_max[0]].item(), idx[0].item())
    return loss, new_points


class Loss(_Loss):

    def __init__(self, num_points_mesh, num_rot_bins):
        super(Loss, self).__init__(True)
        self.num_pt_mesh = num_points_mesh
        self.num_rot_bins = num_rot_bins

    def forward(self, pred_front, pred_rot_bins, pred_t, pred_c, front_r, rot_bins, front_orig, t, idx, model_points, points, w, refine):

        return loss_calculation(pred_front, pred_rot_bins, pred_t, pred_c, front_r, rot_bins, front_orig, t, idx, model_points, points, w, refine, self.num_pt_mesh, self.num_rot_bins)
