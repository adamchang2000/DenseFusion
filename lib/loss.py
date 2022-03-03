from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
try:
    from .tools import compute_rotation_matrix_from_ortho6d
except:
    from tools import compute_rotation_matrix_from_ortho6d
import time
import numpy as np
import torch.nn as nn
import random
import torch.backends.cudnn as cudnn
from knn_cuda import KNN

#pred_r : batch_size * n * 4 -> batch_size * n * 6
def loss_calculation(pred_r, pred_t, pred_c, target, model_points, idx, points, w, refine, num_point_mesh, sym_list):

    #print("shapes loss regular", pred_r.shape, pred_t.shape, pred_c.shape, target.shape, model_points.shape, points.shape)

    knn = KNN(k=1, transpose_mode=True)
    bs, num_p, _ = pred_c.size()

    pred_r = pred_r / (torch.norm(pred_r, dim=2).view(bs, num_p, 1))

    base = compute_rotation_matrix_from_ortho6d(pred_r)
    base = base.view(bs*num_p, 3, 3)
    
    # base = torch.cat(((1.0 - 2.0*(pred_r[:, :, 2]**2 + pred_r[:, :, 3]**2)).view(bs, num_p, 1),\
    #                   (2.0*pred_r[:, :, 1]*pred_r[:, :, 2] - 2.0*pred_r[:, :, 0]*pred_r[:, :, 3]).view(bs, num_p, 1), \
    #                   (2.0*pred_r[:, :, 0]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 1]*pred_r[:, :, 3]).view(bs, num_p, 1), \
    #                   (2.0*pred_r[:, :, 1]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 3]*pred_r[:, :, 0]).view(bs, num_p, 1), \
    #                   (1.0 - 2.0*(pred_r[:, :, 1]**2 + pred_r[:, :, 3]**2)).view(bs, num_p, 1), \
    #                   (-2.0*pred_r[:, :, 0]*pred_r[:, :, 1] + 2.0*pred_r[:, :, 2]*pred_r[:, :, 3]).view(bs, num_p, 1), \
    #                   (-2.0*pred_r[:, :, 0]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 1]*pred_r[:, :, 3]).view(bs, num_p, 1), \
    #                   (2.0*pred_r[:, :, 0]*pred_r[:, :, 1] + 2.0*pred_r[:, :, 2]*pred_r[:, :, 3]).view(bs, num_p, 1), \
    #                   (1.0 - 2.0*(pred_r[:, :, 1]**2 + pred_r[:, :, 2]**2)).view(bs, num_p, 1)), dim=2).contiguous().view(bs * num_p, 3, 3)

    ori_base = base
    base = base.contiguous().transpose(2, 1).contiguous()

    model_points = model_points.view(bs, 1, num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs*num_p, num_point_mesh, 3)
    target = target.view(bs, 1, num_point_mesh, 3).repeat(1, num_p, 1, 1).view(bs, num_p, num_point_mesh, 3)
    ori_target = target
    pred_t = pred_t.contiguous().view(bs*num_p, 1, 3)
    ori_t = pred_t
    points = points.contiguous().view(bs*num_p, 1, 3)
    pred_c = pred_c.contiguous().view(bs, num_p)

    #print("shapes before bmm?", model_points.shape, base.shape, points.shape, pred_t.shape)

    pred = torch.add(torch.bmm(model_points, base), points + pred_t)

    pred = pred.view(bs, num_p, num_point_mesh, 3)

    #print("loss shapes now before possible knn", pred.shape, target.shape)

    #knn will happen in refiner loss
    if not refine:
        for i in range(len(idx)):
            if idx[i].item() in sym_list:

                my_target = target[i,0,:,:].contiguous().view(1, -1, 3)
                my_pred = pred[i].contiguous().view(1, -1, 3)

                dists, inds = knn(my_target, my_pred)
                inds = inds.repeat(1, 1, 3)
                my_target = torch.gather(my_target, 1, inds)

                my_target = my_target.view(num_p, num_point_mesh, 3).contiguous()

                target[i] = my_target

    dis = torch.mean(torch.norm((pred - target), dim=3), dim=2)

    #print("loss dis", dis.shape)

    loss = torch.mean((dis * pred_c - w * torch.log(pred_c)))

    #print("loss!", loss.shape)
    

    pred_c = pred_c.view(bs, num_p)
    how_max, which_max = torch.max(pred_c, 1)
    dis = dis.view(bs, num_p)

    ori_t = ori_t.view(bs, num_p, 1, 3)
    points = points.view(bs, num_p, 1, 3)

    ori_which_max = which_max

    which_max = which_max.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 3)
    #print("gotta do this stuff now", ori_t.shape, points.shape, which_max.shape)

    t = torch.gather(ori_t, 1, which_max) + torch.gather(points, 1, which_max)#ori_t[:,which_max] + points[:,which_max]

    ori_base = ori_base.view(bs, num_p, 3, 3)

    #print("more this stuff now", ori_base.shape)

    which_max = ori_which_max.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 3, 3)

    ori_base = torch.gather(ori_base, 1, which_max).view(bs, 3, 3).contiguous()


    ori_t = t.repeat(1, num_p, 1, 1).contiguous()

    #print("HERERERE", ori_t.shape, points.shape, ori_base.shape)

    ori_t = ori_t.view(bs, num_p, 3)
    points = points.view(bs, num_p, 3)

    new_points = torch.bmm((points - ori_t), ori_base).contiguous()

    new_target = ori_target[:,0].view(bs, num_point_mesh, 3).contiguous()
    ori_t = t.repeat(1, num_point_mesh, 1, 1).contiguous().view(bs, num_point_mesh, 3)
    new_target = torch.bmm((new_target - ori_t), ori_base).contiguous()

    # print('------------> ', dis[0][which_max[0]].item(), pred_c[0][which_max[0]].item(), idx[0].item())

    #print("outputting this thingy", dis.shape, ori_which_max.shape)

    which_max = ori_which_max.unsqueeze(-1)

    dis = torch.gather(dis, 1, which_max)
    dis = torch.mean(dis)

    del knn
    return loss, dis, new_points.detach(), new_target.detach()


class Loss(_Loss):

    def __init__(self, num_points_mesh, sym_list):
        super(Loss, self).__init__(True)
        self.num_pt_mesh = num_points_mesh
        self.sym_list = sym_list

    def forward(self, pred_r, pred_t, pred_c, target, model_points, idx, points, w, refine):

        return loss_calculation(pred_r, pred_t, pred_c, target, model_points, idx, points, w, refine, self.num_pt_mesh, self.sym_list)
