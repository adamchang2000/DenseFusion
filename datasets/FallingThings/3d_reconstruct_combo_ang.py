import json
import cv2
import numpy as np
import open3d as o3d
from PIL import Image

def ply_vtx(path):
    f = open(path)
    assert f.readline().strip() == "ply"
    f.readline()
    f.readline()
    N = int(f.readline().split()[-1])
    while f.readline().strip() != "end_header":
        continue
    pts = []
    for _ in range(N):
        pts.append(np.float32(f.readline().split()[:3]))
    return np.array(pts)

def get_xprime(cx,cy,fx,fy, pt,depth):
    point = np.zeros(2)
    point[0] = (pt[1]-cx)/fx
    point[1] = (pt[0]-cy)/fy
    point = np.append(point, 1)
    point = point*depth
    return point


pic_dir = 'RoomDemo_static_samesize/'

cam_f = open(pic_dir+'_camera_settings.json','r')
cam_data = json.load(cam_f)
cx = float(cam_data["camera_settings"][1]["intrinsic_settings"]["cx"])
cy = float(cam_data["camera_settings"][1]["intrinsic_settings"]["cy"])
fx = float(cam_data["camera_settings"][1]["intrinsic_settings"]["fx"])
fy = float(cam_data["camera_settings"][1]["intrinsic_settings"]["fy"])

object_f = open(pic_dir+'_object_settings.json','r')
object_data = json.load(object_f)
seg_id = object_data['exported_objects'][0]['segmentation_class_id']

depth_image = cv2.imread(pic_dir+'000001.depth.16.png', cv2.IMREAD_UNCHANGED)
seg = cv2.imread(pic_dir+'000001.cs.png', cv2.IMREAD_UNCHANGED)
seg_arr = np.array(seg)
pts = []

for i in range(len(seg_arr)):
    for j in range(len(seg_arr[0])):
        if seg_arr[i][j] == seg_id:
            pts.append([i, j])

pc_whole = []

for i in range(len(pts)):
    u = pts[i][0]
    v = pts[i][1]
    depth = depth_image[u][v]
    twoD_point = np.array([u,v],dtype='float64')
    threeD_point = get_xprime(cx,cy,fx,fy,twoD_point,depth)
    pc_whole.append(threeD_point)

FOR = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000, origin=[0,0,0])
mycloud = o3d.geometry.PointCloud()
mycloud.points = o3d.utility.Vector3dVector(pc_whole)

#####################################
index = '000001'
floder = 'RoomDemo_static_samesize'
fixed_config = open('./{0}/_object_settings.json'.format(floder), )
fixed_data = json.load(fixed_config)
fixed_transform = np.array(fixed_data['exported_objects'][0]['fixed_model_transform'])
fixed_rotation = fixed_transform[:3, :3]
fixed_translation = np.zeros(3)
for i in range(3):
    fixed_translation[i] = fixed_transform[3][i]
#fixed_rotation = changeDet(fixed_rotation)

#img and related config
img = Image.open('./{1}/{0}.png'.format(index, floder))
Image._show(img)
f = open('./{1}/{0}.json'.format(index, floder), )
data = json.load(f)
target_transform = np.array(data['objects'][0]['pose_transform'])
target_rotation = target_transform[:3, :3]
print(target_rotation)
#0 2 1 n
#1 0 2 n
#1 2 0
#2 0 1
#2 1 0
# target_rotation = target_rotation[:, [2, 1, 0]]

for i in range(3):
    target_rotation[i][0] = -target_rotation[i][0]
    target_rotation[i][1] = -target_rotation[i][1]
    target_rotation[i][2] = -target_rotation[i][2]
print(target_rotation)

target_translation = np.zeros(3)
for i in range(3):
    target_translation[i] = target_transform[3][i]
p = np.array([[ 0, 0,  1],
              [ 1, 0,  0],
              [ 0,-1,  0]])
real_target_rotation = np.matmul(target_rotation.T, p)
print(np.linalg.det(real_target_rotation))
points = ply_vtx('models/1.ply')*1000
pcd = o3d.geometry.PointCloud()
pcd2  = o3d.geometry.PointCloud()
#pcd2.points = o3d.utility.Vector3dVector(points)
#target = np.dot(np.dot(points, fixed_rotation.T), (target_rotation).T)
# fixed_rotation = np.dot(fixed_rotation.T,p)

# target = np.matmul(points, np.linalg.inv(fixed_rotation))
target = np.matmul(points, fixed_rotation) + fixed_translation * 100
target = np.matmul(points, real_target_rotation.T) + target_translation * 100


#target = target+np.array([0,0,1000])

# target = np.dot(target, fixed_rotation.T)
pcd.points = o3d.utility.Vector3dVector(target)
# o3d.visualization.draw_geometries([pcd,FOR,pcd2])


o3d.visualization.draw_geometries([mycloud,FOR,pcd,pcd2])

