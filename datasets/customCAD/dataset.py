import torch.utils.data as data
from PIL import Image
import sys
import os
import torchvision.transforms as transforms
import numpy as np
import open3d as o3d
import numpy.ma as ma
import cv2
import random
from .project_unity_depth import UnityDepthProjector
import torch
from scipy.spatial.transform import Rotation as R

class PoseDataset(data.Dataset):
    def __init__(self, mode, num, add_noise, root, noise_trans, refine):
        self.objlist = [1] #only medical cad model
        self.mode = mode

        self.list_rgb = []
        self.list_depth = []
        self.list_label = []
        self.list_obj = []
        self.list_rank = []
        self.meta = {}
        self.pt = {}
        self.root = root
        self.noise_trans = noise_trans
        self.refine = refine

        item_count = 0
        for item in self.objlist:
            if self.mode == 'train':
                input_file = open('{0}/data/{1}/train.txt'.format(self.root, '%02d' % item))
            else:
                input_file = open('{0}/data/{1}/test.txt'.format(self.root, '%02d' % item))
            while 1:
                item_count += 1
                input_line = input_file.readline()
                if self.mode == 'test' and item_count % 10 != 0:
                    continue
                if not input_line:
                    break
                if input_line[-1:] == '\n':
                    input_line = input_line[:-1]
                self.list_rgb.append('{0}/data/{1}/rgb/FrameBuffer_{2}.png'.format(self.root, '%02d' % item, '%04d' % int(input_line)))
                self.list_depth.append('{0}/data/{1}/depth/Depth_{2}.png'.format(self.root, '%02d' % item, '%04d' % int(input_line)))
                self.list_label.append('{0}/data/{1}/mask/{2}.png'.format(self.root, '%02d' % item, '%04d' % int(input_line)))
                
                self.list_obj.append(item)

            #parse gt transforms
            self.meta[item] = {}
            meta_file = open('{0}/data/{1}/meta/transforms.txt'.format(self.root, '%02d' % item), 'r')
            
            while 1:
                try:
                    idx = int(meta_file.readline().strip())

                    pos = meta_file.readline()
                    pos = pos.replace('(', '').replace(')', '').replace(',', '').split(' ')
                    pos = np.array([float(x.rstrip()) for x in pos])

                    quat = meta_file.readline()
                    quat = quat.replace('(', '').replace(')', '').replace(',', '').split(' ')
                    quat = np.array([float(x.rstrip()) for x in quat])

                    self.meta[item][idx] = [pos, quat]

                except:
                    break
            
            self.pt[item] = ply_vtx('{0}/models/obj_{1}.ply'.format(self.root, '%02d' % item))
            
            print("Object {0} buffer loaded".format(item))

        self.length = len(self.list_rgb)

        self.num = num
        self.add_noise = add_noise
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
        self.num_pt_mesh_large = 1500
        self.num_pt_mesh_small = 1500

        self.udp = UnityDepthProjector('{0}/data/{1}/meta/proj_mat.txt'.format(self.root, '%02d' % item), (520, 1109))

    def __getitem__(self, index):
        #outputs:
        #pointcloud: xyz points of projected cropped depth image (just the object region)
        #choose: select num_points points from the cropped region of the image, this is a mask
        #img_masked: cropped region of image (RGB) with object
        #target: Rt matrix
        #model_points: just the model points, num_pt_mesh_small of them
        #obj: class (in our case, just a single number since we only have 1 object)
        img = Image.open(self.list_rgb[index])
        ori_img = np.array(img)

        depth = np.array(Image.open(self.list_depth[index]))
        label = np.array(Image.open(self.list_label[index]))
        obj = self.list_obj[index]

        gt_trans = self.meta[obj][index]

        #remove infinities
        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, np.max(depth)))

        #get mask crop
        mask_label = ma.getmaskarray(ma.masked_equal(label, 65535))
        mask = mask_label * mask_depth

        if self.add_noise:
            img = self.trancolor(img)

        img = np.array(img)[:, :, :3]
        img = np.transpose(img, (2, 0, 1))
        img_masked = img

        rmin, rmax, cmin, cmax = get_bbox(mask_label)
        img_masked = img_masked[:, rmin:rmax, cmin:cmax]


        target_r_quat = convert_quat(gt_trans[1])
        target_r = quaternion_rotation_matrix(target_r_quat)
        target_t = gt_trans[0] * 1000
        target_t[2] = -target_t[2]
        add_t = np.array([random.uniform(-self.noise_trans, self.noise_trans) for i in range(3)])

        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        if len(choose) == 0:
            cc = torch.LongTensor([0])
            return(cc, cc, cc, cc, cc, cc)

        if len(choose) > self.num:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, self.num - len(choose)), 'wrap')

        depth_projected = self.udp.project_depth(depth)[rmin:rmax, cmin:cmax].reshape((-1, 3))

        depth_masked = depth_projected[choose].astype(np.float32)
        choose = np.array([choose])
        cloud = depth_masked

        if self.add_noise:
            cloud = np.add(cloud, add_t)

        model_points = self.pt[obj] * 10
        dellist = [j for j in range(0, len(model_points))]
        dellist = random.sample(dellist, len(model_points) - self.num_pt_mesh_small)
        model_points = np.delete(model_points, dellist, axis=0)

        #want to swap the axes of the target
        swap_x = np.zeros((3, 3))
        swap_x[0,0] = 1
        swap_x[1,2] = 1
        swap_x[2,1] = 1

        swap_y = np.zeros((3, 3))
        swap_y[0,2] = 1
        swap_y[1,1] = 1
        swap_y[2,0] = -1

        target = np.copy(model_points)
        # target_mean = np.mean(target, axis=0)

        # target -= target_mean
        # target = np.dot(target, swap_left.T)
        # target += target_mean


        target = np.dot(target, (target_r @ swap_x).T)

        if self.add_noise:
            target = np.add(target, target_t + add_t * 10000)
        else:
            target = np.add(target, target_t)

        #AT THE VERY END, CONVERT EVERYTHING TO METERS
        return torch.from_numpy(cloud.astype(np.float32) / 10000.), \
               torch.LongTensor(choose.astype(np.int32)), \
               self.norm(torch.from_numpy(img_masked.astype(np.float32))), \
               torch.from_numpy(target.astype(np.float32) / 10000.), \
               torch.from_numpy(model_points.astype(np.float32) / 10000.), \
               torch.LongTensor([self.objlist.index(obj)])
        

    def __len__(self):
        return self.length

    def get_sym_list(self):
        return []

    def get_num_points_mesh(self):
        if self.refine:
            return self.num_pt_mesh_large
        else:
            return self.num_pt_mesh_small

#convert from left hand to right hand quaternion
def convert_quat(Q):
    return np.array([-Q[0], -Q[1], Q[2], Q[3]])

def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """

    r = R.from_quat(Q)
                            
    return r.as_matrix()


def get_bbox(mask):
    a = np.where(mask == True)
    return np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])

def ply_vtx(path, number_of_points=3000):
    try:
        model = o3d.io.read_triangle_mesh(path)
        if len(model.triangles) > 0:
            pts = np.array(model.sample_points_uniformly(number_of_points=number_of_points))
        else:
            model = o3d.io.read_point_cloud(path)
            pts = np.array(model.points)
            pts = pts[np.random.choice(pts.shape[0], number_of_points)]
        return pts
    except:
        print('load failed')
        raise