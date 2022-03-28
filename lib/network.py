import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import pdb
import torch.nn.functional as F
from lib.pspnet import PSPNet
from lib.RandLA.RandLANet import Network as RandLANet
from lib.randla_utils import ConfigRandLA


psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}


class ModifiedResnet(nn.Module):

    def __init__(self, usegpu=True):
        super(ModifiedResnet, self).__init__()

        self.model = psp_models['resnet18'.lower()]()
        self.model = nn.DataParallel(self.model)

    def forward(self, x):
        x = self.model(x)
        return x

class PoseNetFeat(nn.Module):
    def __init__(self, num_points, use_normals, use_colors):
        super(PoseNetFeat, self).__init__()

        pcld_dim = 3
        if use_normals:
            pcld_dim += 3

        self.conv1 = torch.nn.Conv1d(pcld_dim, 64, 1)
        #self.bn1 = torch.nn.BatchNorm1d(64)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        #self.bn2 = torch.nn.BatchNorm1d(128)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        #self.e_bn1 = torch.nn.BatchNorm1d(64)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)
        #self.e_bn2 = torch.nn.BatchNorm1d(128)

        self.conv5 = torch.nn.Conv1d(256, 512, 1)
        #self.bn5 = torch.nn.BatchNorm1d(512)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)
        #self.bn6 = torch.nn.BatchNorm1d(1024)

        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.num_points = num_points
    def forward(self, x, emb):

        #"pointnet" layer
        #XYZ -> 64 dim embedding
        x = F.relu(self.conv1(x))

        #32 dim embedding -> 64 dim embedding
        emb = F.relu(self.e_conv1(emb))

        #concating them
        pointfeat_1 = torch.cat((x, emb), dim=1)

        #64 dim embedding -> 128 dim embedding
        x = F.relu(self.conv2(x))
        #64 dim embedding -> 128 dim embedding
        emb = F.relu(self.e_conv2(emb))

        #concating them
        pointfeat_2 = torch.cat((x, emb), dim=1)

        #lifting fused 128 + 128 -> 512
        x = F.relu(self.conv5(pointfeat_2))

        #lifting fused 256 + 256 -> 1024
        x = F.relu(self.conv6(x))

        #average pooling on into 1 1024 global feature
        ap_x = self.ap1(x)

        #repeat it so they can staple it onto the back of every pixel/point
        ap_x = ap_x.view(-1, 1024, 1).repeat(1, 1, self.num_points)

        #64 + 64 (level 1), 128 + 128 (level 2), 1024 global feature
        return torch.cat([pointfeat_1, pointfeat_2, ap_x], 1) #128 + 256 + 1024

class DenseFusion(nn.Module):
    def __init__(self, num_points):
        super(DenseFusion, self).__init__()
        self.conv2_rgb = torch.nn.Conv1d(128, 256, 1)
        self.conv2_cld = torch.nn.Conv1d(128, 256, 1)

        self.conv3 = torch.nn.Conv1d(256, 512, 1)
        self.conv4 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)

    def forward(self, rgb_emb, cld_emb):
        bs, _, n_pts = cld_emb.size()
        feat_1 = torch.cat((rgb_emb, cld_emb), dim=1)
        rgb = F.relu(self.conv2_rgb(rgb_emb))
        cld = F.relu(self.conv2_cld(cld_emb))

        feat_2 = torch.cat((rgb, cld), dim=1)

        rgbd = F.relu(self.conv3(feat_1))
        rgbd = F.relu(self.conv4(rgbd))

        ap_x = self.ap1(rgbd)

        ap_x = ap_x.view(-1, 1024, 1).repeat(1, 1, n_pts)
        return torch.cat([feat_1, feat_2, ap_x], 1) # 256 + 512 + 1024 = 1792

class PoseNet(nn.Module):
    def __init__(self, num_points, num_obj, use_normals, use_colors):
        super(PoseNet, self).__init__()
        self.num_points = num_points
        self.cnn = ModifiedResnet()
        #self.feat = PoseNetFeat(num_points, use_normals)

        self.df = DenseFusion(num_points)
        
        self.conv1_r = torch.nn.Conv1d(1792, 640, 1)
        self.conv1_t = torch.nn.Conv1d(1792, 640, 1)
        self.conv1_c = torch.nn.Conv1d(1792, 640, 1)

        self.conv2_r = torch.nn.Conv1d(640, 256, 1)
        self.conv2_t = torch.nn.Conv1d(640, 256, 1)
        self.conv2_c = torch.nn.Conv1d(640, 256, 1)

        self.conv3_r = torch.nn.Conv1d(256, 128, 1)
        self.conv3_t = torch.nn.Conv1d(256, 128, 1)
        self.conv3_c = torch.nn.Conv1d(256, 128, 1)

        self.conv4_r = torch.nn.Conv1d(128, num_obj*6, 1) #6d rot
        self.conv4_t = torch.nn.Conv1d(128, num_obj*3, 1) #translation
        self.conv4_c = torch.nn.Conv1d(128, num_obj*1, 1) #confidence

        self.num_obj = num_obj

        rndla_config = ConfigRandLA
        self.rndla = RandLANet(rndla_config)

        self.use_normals = use_normals
        self.use_colors = use_colors

    def forward(self, end_points):

        out_img = self.cnn(end_points["img"])
        
        bs, di, _, _ = out_img.size()

        emb = out_img.view(bs, di, -1)
        choose = end_points["choose"].repeat(1, di, 1)
        emb = torch.gather(emb, 2, choose).contiguous()

        features = end_points["cloud"]

        if self.use_normals:
            normals = end_points["normals"]
            features = torch.cat((features, normals), dim=-1)
        
        if self.use_colors:
            colors = end_points["cloud_colors"]
            features = torch.cat((features, colors), dim=-1)

        end_points["RLA_features"] = features.transpose(1, 2)
        end_points = self.rndla(end_points)
        feat_x = end_points["RLA_embeddings"]

        #x is pointcloud
        #emb is cnn embedding
        ap_x = self.df(emb, feat_x)

        #ap_x = self.feat(x, emb)

        rx = F.relu(self.conv1_r(ap_x))
        tx = F.relu(self.conv1_t(ap_x))
        cx = F.relu(self.conv1_c(ap_x))      

        rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))
        cx = F.relu(self.conv2_c(cx))

        rx = F.relu(self.conv3_r(rx))
        tx = F.relu(self.conv3_t(tx))
        cx = F.relu(self.conv3_c(cx))

        rx = self.conv4_r(rx).view(bs, self.num_obj, 6, self.num_points)
        tx = self.conv4_t(tx).view(bs, self.num_obj, 3, self.num_points)
        cx = torch.sigmoid(self.conv4_c(cx)).view(bs, self.num_obj, 1, self.num_points)

        obj = end_points["obj_idx"].unsqueeze(-1).unsqueeze(-1)
        obj_rx = obj.repeat(1, 1, rx.shape[2], rx.shape[3])
        obj_tx = obj.repeat(1, 1, tx.shape[2], tx.shape[3])
        obj_cx = obj.repeat(1, 1, cx.shape[2], cx.shape[3])

        out_rx = torch.gather(rx, 1, obj_rx)[:,0,:,:]
        out_tx = torch.gather(tx, 1, obj_tx)[:,0,:,:]
        out_cx = torch.gather(cx, 1, obj_cx)[:,0,:,:]

        out_rx = out_rx.contiguous().transpose(2, 1).contiguous()
        out_tx = out_tx.contiguous().transpose(2, 1).contiguous()
        out_cx = out_cx.contiguous().transpose(2, 1).contiguous()

        end_points["pred_r"] = out_rx
        end_points["pred_t"] = out_tx
        end_points["pred_c"] = out_cx
        end_points["emb"] = emb.detach()

        return end_points
 


class PoseRefineNetFeat(nn.Module):
    def __init__(self, num_points, use_normals, use_colors):
        super(PoseRefineNetFeat, self).__init__()

        pcld_dim = 3
        if use_normals:
            pcld_dim += 3
        if use_colors:
            pcld_dim += 3

        self.conv1 = torch.nn.Conv1d(pcld_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(384, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.num_points = num_points

    def forward(self, x, emb):
        x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat([x, emb], dim=1)

        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = torch.cat([x, emb], dim=1)

        pointfeat_3 = torch.cat([pointfeat_1, pointfeat_2], dim=1)

        x = F.relu(self.conv5(pointfeat_3))
        x = F.relu(self.conv6(x))

        ap_x = self.ap1(x)

        ap_x = ap_x.view(-1, 1024)
        return ap_x

class PoseRefineNet(nn.Module):
    def __init__(self, num_points, num_obj, use_normals, use_colors):
        super(PoseRefineNet, self).__init__()
        self.num_points = num_points
        self.feat = PoseRefineNetFeat(num_points, use_normals, use_colors)
        
        self.conv1_r = torch.nn.Linear(1024, 512)
        self.conv1_t = torch.nn.Linear(1024, 512)

        self.conv2_r = torch.nn.Linear(512, 128)
        self.conv2_t = torch.nn.Linear(512, 128)

        self.conv3_r = torch.nn.Linear(128, num_obj*6) #6d rot
        self.conv3_t = torch.nn.Linear(128, num_obj*3) #translation

        self.num_obj = num_obj

    def forward(self, end_points, refine_iteration):

        x = end_points["new_points"]
        emb = end_points["emb"]
        obj = end_points["obj_idx"]

        bs = x.size()[0]
        
        x = x.transpose(2, 1).contiguous()
        ap_x = self.feat(x, emb)

        rx = F.relu(self.conv1_r(ap_x))
        tx = F.relu(self.conv1_t(ap_x))   

        rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))

        rx = self.conv3_r(rx).view(bs, self.num_obj, 6, 1)
        tx = self.conv3_t(tx).view(bs, self.num_obj, 3, 1)

        obj = obj.unsqueeze(-1).unsqueeze(-1)
        obj_rx = obj.repeat(1, 1, rx.shape[2], rx.shape[3])
        obj_tx = obj.repeat(1, 1, tx.shape[2], tx.shape[3])

        out_rx = torch.gather(rx, 1, obj_rx)[:,0,:,:]
        out_tx = torch.gather(tx, 1, obj_tx)[:,0,:,:]

        out_rx = out_rx.contiguous().transpose(2, 1).contiguous()
        out_tx = out_tx.contiguous().transpose(2, 1).contiguous()

        #print("shapes!", out_rx.shape, out_tx.shape)

        end_points["refiner_pred_r_" + str(refine_iteration)] = out_rx
        end_points["refiner_pred_t_" + str(refine_iteration)] = out_tx

        return end_points
