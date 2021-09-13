from dataset import PoseDataset
from mask_generator import UnityMaskGenerator
import open3d as o3d
import cv2
import numpy as np
import torch

#m = UnityMaskGenerator("dataset_processed")
#m.generate_masks()

p = PoseDataset("train", 1000, False, 'dataset_processed', 0.03, True)
cloud, _, _, target, model, _ = p.__getitem__(44)

o3dcloud = o3d.geometry.PointCloud()
o3dcloud.points = o3d.utility.Vector3dVector(cloud)

#o3d.visualization.draw_geometries([o3dcloud])
o3d.io.write_point_cloud('depth_projected.ply', o3dcloud)

o3dtarget = o3d.geometry.PointCloud()
o3dtarget.points = o3d.utility.Vector3dVector(target)

#o3d.visualization.draw_geometries([o3dtarget])
o3d.io.write_point_cloud('target.ply', o3dtarget)

o3dmodel = o3d.geometry.PointCloud()
o3dmodel.points = o3d.utility.Vector3dVector(model)

#o3d.visualization.draw_geometries([o3dmodel])
o3d.io.write_point_cloud('model.ply', o3dmodel)