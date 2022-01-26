import numpy as np
from PIL import Image
import json
import open3d as o3d
import cv2

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

def xyz_vtx(path):
    f = open(path)
    pts = []
    while True:
        line = f.readline()
        if not line:
            break
        pts.append(np.float32(line.split()[:3]))
    return np.array(pts)

def changeDet(arr):
    # det to +_1
    for i in range(3):
        arr[1][i] = -arr[1][i]
    return arr

def quaternion_to_rotation_matrix(quat):
  q = quat.copy()
  n = np.dot(q, q)
  if n < np.finfo(q.dtype).eps:
    return np.identity(4)
  q = q * np.sqrt(2.0 / n)
  q = np.outer(q, q)
  rot_matrix = np.array(
    [[1.0 - q[2, 2] - q[3, 3], q[1, 2] + q[3, 0], q[1, 3] - q[2, 0],],
     [q[1, 2] - q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] + q[1, 0],],
     [q[1, 3] + q[2, 0], q[2, 3] - q[1, 0], 1.0 - q[1, 1] - q[2, 2],],
     ],
    dtype=q.dtype)
  return rot_matrix
t = [[0, -1, 0],
     [1, 0, 0],
     [0, 0, 1]]

print(np.linalg.det(t))

p = np.array([[ 0, 0,  1],
                 [ 1, 0,  0],
    [ 0, -1,  0]])

print(np.linalg.det(p))

index = '000001.right'
floder = 'power_drill_with_model'

#fixed config
fixed_config = open('./{0}/_object_settings.json'.format(floder), )
fixed_data = json.load(fixed_config)
fixed_transform = np.array(fixed_data['exported_objects'][0]['fixed_model_transform'])
fixed_rotation = fixed_transform[:3, :3]
fixed_translation = np.zeros(3)
for i in range(3):
    fixed_translation[i] = fixed_transform[3][i]
#fixed_rotation = changeDet(fixed_rotation)

#img and related config
img = Image.open('./{1}/{0}.jpg'.format(index, floder))
Image._show(img)
f = open('./{1}/{0}.json'.format(index, floder),)
data = json.load(f)

bb = data['objects'][0]['bounding_box']
top_left_x = int(bb['top_left'][0])
top_left_y = int(bb['top_left'][1])
bottom_right_x = int(bb['bottom_right'][0])
bottom_right_y = int(bb['bottom_right'][1])



cropped = img.crop((top_left_y, top_left_x, bottom_right_y, bottom_right_x))  # (left, upper, right, lower)
Image._show(cropped)
target_transform = np.array(data['objects'][0]['pose_transform_permuted'])
target_rotation = target_transform[:3, :3]
real_target_rotation = np.dot(target_rotation.T, p)
target_translation = np.zeros(3)
for i in range(3):
    target_translation[i] = target_transform[3][i]

qua = np.array(data['objects'][0]['quaternion_xyzw'])
print("real_target_rotation")
print(real_target_rotation)
print(np.linalg.det(real_target_rotation))
'''
S_rot = quaternion_to_rotation_matrix(qua)

print(np.linalg.det(S_rot))
print(np.linalg.det(target_rotation))
print(np.linalg.det(fixed_rotation))
print(S_rot)
print(target_rotation.T)
'''
#target_rotation = changeDet(target_rotation)


points = xyz_vtx('./models/points_1.xyz')
print(points.shape)
pcd = o3d.geometry.PointCloud()
pcd_o = o3d.geometry.PointCloud()

#target = np.dot(np.dot(points, fixed_rotation.T), (target_rotation).T)




def get_xprime(cx,cy,fx,fy,pt,depth):
    point = np.zeros(2)
    point[0] = (pt[1]-cx)/fx
    point[1] = (pt[0]-cy)/fy
    point = np.append(point, 1)
    point = point*depth
    # print(point)
    point = point
    return point




cam_f = open(floder + '/' +'_camera_settings.json','r')
cam_data = json.load(cam_f)
cx = float(cam_data["camera_settings"][1]["intrinsic_settings"]["cx"])
cy = float(cam_data["camera_settings"][1]["intrinsic_settings"]["cy"])
fx = float(cam_data["camera_settings"][1]["intrinsic_settings"]["fx"])
fy = float(cam_data["camera_settings"][1]["intrinsic_settings"]["fy"])

object_f = open(floder + '/' +'_object_settings.json','r')
object_data = json.load(object_f)
seg_id = object_data['exported_objects'][0]['segmentation_class_id']

image = cv2.imread(floder + '/' + index + '.depth.png', cv2.IMREAD_UNCHANGED)
print(np.array(image).shape)
seg = cv2.imread(floder + '/' + index + '.seg.png', cv2.IMREAD_UNCHANGED)
seg_arr = np.array(seg)

pts = []

for i in range(len(seg_arr)):
    for j in range(len(seg_arr[0])):
        if seg_arr[i][j] == seg_id:
            pts.append([i, j])


#print(seg_arr)
#print(seg_arr.shape)


# cv2.imshow('dep',image)
# cv2.waitKey(0) # waits until a key is pressed
# cv2.destroyAllWindows()
# print(image.shape)

pc_whole = []
pc_whole_2 = []
for i in range(len(pts)):
    u = pts[i][0]
    v = pts[i][1]
    depth = image[u][v]
    twoD_point = np.array([u,v],dtype='float64')
    threeD_point = get_xprime(cx,cy,fx,fy,twoD_point,depth)
    pc_whole.append(threeD_point)

for u in range(image.shape[0]):
    for v in range(image.shape[1]):
        depth = image[u][v]
        twoD_point = np.array([u, v], dtype='float64')
        threeD_point = get_xprime(cx, cy, fx, fy, twoD_point, depth)
        pc_whole_2.append(threeD_point)

scale = 100

FOR = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])
mycloud = o3d.geometry.PointCloud()
mycloud.points = o3d.utility.Vector3dVector(pc_whole_2)
object_cloud = o3d.geometry.PointCloud()
object_cloud.points = o3d.utility.Vector3dVector(np.array(pc_whole) / 10000)

fixed_rotation/=100
fixed_translation/=100
target_translation/=100
fixed_translation = np.matmul(fixed_translation)
print('fixed_rotation')
print(fixed_rotation)
print(fixed_translation)
print('real_target_rotation')
print(real_target_rotation)
print(target_translation)

print(np.linalg.det(fixed_rotation))
print(np.linalg.det(real_target_rotation))

target = np.dot(np.dot(points, fixed_rotation)+ fixed_translation, real_target_rotation.T) + target_translation
pcd.points = o3d.utility.Vector3dVector(target)
pcd_o.points = o3d.utility.Vector3dVector(points)

o3d.visualization.draw_geometries([FOR] + [pcd_o] + [pcd] + [object_cloud] )


'''
cam_scale = 1.0
pt2 = depth_masked / cam_scale
cam_cx =
cam_cy =
cam_fx =
cam_fy =

pt0 = []
pt1 = []


for x in range(rmin, rmax + 1):
    for y in range(cmin, cmax + 1):
        pt0.append(y - )

pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
'''








'''
target = np.dot(points, S_rot.T)
pcd.points = o3d.utility.Vector3dVector(target)
o3d.visualization.draw_geometries([pcd])

#o3d.io.write_point_cloud('target.ply', pcd)

target = np.dot(np.dot(points, S_rot.T), fixed_rotation.T)
pcd.points = o3d.utility.Vector3dVector(target)
o3d.visualization.draw_geometries([pcd])

target = np.dot(np.dot(points, fixed_rotation.T), S_rot.T)
pcd.points = o3d.utility.Vector3dVector(target)
o3d.visualization.draw_geometries([pcd])

target = np.dot(np.dot(points, fixed_rotation), target_rotation)
pcd.points = o3d.utility.Vector3dVector(target)
o3d.visualization.draw_geometries([pcd])
'''