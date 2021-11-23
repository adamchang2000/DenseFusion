import json
import cv2
import numpy as np
import open3d as o3d

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

def quaternion_to_rotation_matrix(quat):
  q = quat.copy()
  n = np.matmul(q, q)
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

def swap_rc(matrix,r_c,n1,n2):
    if r_c=='c':
        tmp = matrix[:, n1].copy()
        matrix[:, n1] = matrix[:, n2]
        matrix[:, n2] = tmp
    else:
        tmp = matrix[n1,:].copy()
        matrix[n1,:] = matrix[n2,:]
        matrix[n2,:] = tmp
    return matrix

def flip_sign(matrix,r_c,n):
    if r_c=='c':
        matrix[:,n] = -matrix[:,n]
    else:
        matrix[n,:] = -matrix[n,:]
    return matrix

def getPoseTransPermuted(quaternion):
    rot = quaternion_to_rotation_matrix(quaternion)
    rot = swap_rc(rot,'c',1,2)
    rot = rot.T
    rot = swap_rc(rot, 'c', 0, 2)
    rot = flip_sign(rot, 'c', 2)
    return rot

def get_xprime(cx,cy,fx,fy, pt,depth):
    point = np.zeros(2)
    point[0] = (pt[1]-cx)/fx
    point[1] = (pt[0]-cy)/fy
    point = np.append(point, 1)
    point = point*(depth/65535.0*100000)
    return point


index = '000051'
floder = 'RoomDemo_obj3_11'
pic_dir = floder+'/'
cam_f = open(pic_dir+'_camera_settings.json','r')
cam_data = json.load(cam_f)
cx = float(cam_data["camera_settings"][0]["intrinsic_settings"]["cx"])
cy = float(cam_data["camera_settings"][0]["intrinsic_settings"]["cy"])
fx = float(cam_data["camera_settings"][0]["intrinsic_settings"]["fx"])
fy = float(cam_data["camera_settings"][0]["intrinsic_settings"]["fy"])

object_f = open(pic_dir+'_object_settings.json','r')
object_data = json.load(object_f)
seg_id = object_data['exported_objects'][0]['segmentation_class_id']

depth_image = cv2.imread(pic_dir+index+'.left.depth.16.png', cv2.IMREAD_UNCHANGED)
print(np.array(depth_image).shape)
seg = cv2.imread(pic_dir+index+'.left.cs.png', cv2.IMREAD_UNCHANGED)
seg_arr = np.array(seg)
pts = []

for i in range(len(seg_arr)):
    for j in range(len(seg_arr[0])):
        if seg_arr[i][j] == seg_id:
            pts.append([i, j])

#print(pts)
#print(seg_arr)
#print(seg_arr.shape)


# cv2.imshow('dep',image)
# cv2.waitKey(0) # waits until a key is pressed
# cv2.destroyAllWindows()
# print(image.shape)

pc_whole = []

print("depth image values")
print(np.unique(depth_image))

for i in range(len(pts)):
    u = pts[i][0]
    v = pts[i][1]
    depth = depth_image[u][v]
    twoD_point = np.array([u,v],dtype='float64')
    threeD_point = get_xprime(cx,cy,fx,fy,twoD_point,depth)
    pc_whole.append(threeD_point)

pc_whole = np.array(pc_whole)/10000
print(pc_whole)

FOR = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])
mycloud = o3d.geometry.PointCloud()
mycloud.points = o3d.utility.Vector3dVector(pc_whole)

#####################################
# fixed_config = open(pic_dir+'_object_settings.json'.format(floder), )
fixed_config = open('RoomDemo_obj3_static/_object_settings.json'.format(floder), )
fixed_data = json.load(fixed_config)
fixed_transform = np.array(fixed_data['exported_objects'][0]['fixed_model_transform'])
fixed_rotation = fixed_transform[:3, :3]
fixed_translation = fixed_transform[3,:3]
#fixed_rotation = changeDet(fixed_rotation)
fixed_rotation = fixed_rotation.astype(np.float64)
fixed_translation = fixed_translation.astype(np.float64)
fixed_rotation /= 100
fixed_translation /= 100.

#img and related config
# img = Image.open('./{1}/{0}.left.png'.format(index, floder))
# Image._show(img)
f = open('./{1}/{0}.left.json'.format(index, floder), )
data = json.load(f)
target_tranform = np.array(data['objects'][0]['pose_transform'])
# target_quaternion = np.array(data['objects'][0]['quaternion_xyzw'])
# pose_rot_permuted = getPoseTransPermuted(target_quaternion)
p = np.array([[ 0, 0,  1],
              [ 1, 0,  0],
              [ 0,-1,  0]])
# real_target_rotation = np.matmul(pose_rot_permuted.T, p)
# target_rotation = quaternion_to_rotation_matrix(target_quaternion)
# real_target_rotation = target_rotation

target_rotation = target_tranform[:3,:3]
# real_target_rotation = target_rotation
real_target_rotation = np.matmul(target_rotation.T, p)
target_translation = np.array(data['objects'][0]['location'])/100


# ry90 = np.array([[0.,0,-1],
#                  [0,1,0],
#                  [1,0,0]])
# rz_90 = np.array([[0.,1,0],
#                  [-1,0,0],
#                  [0,0,1]])
######################read from glb##############################
# gltf = GLTF2().load_binary('models/1.glb')
#
# binary_blob = gltf.binary_blob()
#
# triangles_accessor = gltf.accessors[gltf.meshes[0].primitives[0].indices]
# triangles_buffer_view = gltf.bufferViews[triangles_accessor.bufferView]
# triangles = np.frombuffer(
#     binary_blob[
#         triangles_buffer_view.byteOffset
#         + triangles_accessor.byteOffset : triangles_buffer_view.byteOffset
#         + triangles_buffer_view.byteLength
#     ],
#     dtype="uint8",
#     count=triangles_accessor.count,
# ).reshape((-1, 3))
#
# points_accessor = gltf.accessors[gltf.meshes[0].primitives[0].attributes.POSITION]
# points_buffer_view = gltf.bufferViews[points_accessor.bufferView]
# points = np.frombuffer(
#     binary_blob[
#         points_buffer_view.byteOffset
#         + points_accessor.byteOffset : points_buffer_view.byteOffset
#         + points_buffer_view.byteLength
#     ],
#     dtype="float32",
#     count=points_accessor.count * 3,
# ).reshape((-1, 3))
# points = points*1000
#################################################

points = o3d.io.read_triangle_mesh('models/3_centered.obj')
points = np.array(points.vertices)

print(np.max(points, axis=0), np.min(points, axis=0))

pcd = o3d.geometry.PointCloud()
pcd2  = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(points)
#target = np.dot(np.dot(points, fixed_rotation.T), (target_rotation).T)
# fixed_rotation = np.dot(fixed_rotation.T,p)

# target = np.matmul(points, np.linalg.inv(fixed_rotation))
# target = np.matmul(points, fixed_rotation)
print(real_target_rotation)
print(fixed_rotation)
print(target_translation)
print(fixed_translation)

# target = np.matmul(points, real_target_rotation.T)

# target = np.dot(points, rz_90.T)
target = np.dot(points, fixed_rotation)#+ fixed_translation
target = np.dot(target, real_target_rotation.T)
target+= target_translation

# target = np.dot(target, fixed_rotation.T)
pcd.points = o3d.utility.Vector3dVector(target)
# o3d.visualization.draw_geometries([pcd,FOR,pcd2])

o3d.io.write_point_cloud("target.ply", pcd)
o3d.io.write_point_cloud("projected.ply", mycloud)
o3d.io.write_point_cloud("identity.ply", pcd2)








o3d.visualization.draw_geometries([mycloud,FOR,pcd,pcd2])

