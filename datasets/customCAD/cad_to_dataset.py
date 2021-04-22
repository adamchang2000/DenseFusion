import numpy as np
import numpy.ma as ma
import open3d as o3d #VERSION OPEN 0.12.0
import argparse
import logging
import math
import cv2

#format = '%(asctime)s.%(msecs)03d %(levelname)-.1s [%(name)s][%(threadName)s] %(message)s'
#logging.basicConfig(format=format, level=logging.DEBUG)

# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta):
    
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
        
        
                    
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                    
                    
    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R

def getRandomRotationMatrix():
	euler_angles = np.zeros(3)
	euler_angles[0] = -np.pi + np.random.uniform(0, 1) * 2 * np.pi
	euler_angles[1] = -np.pi / 2 + np.random.uniform(0, 1) * np.pi
	euler_angles[2] = -np.pi + np.random.uniform(0, 1) * 2 * np.pi
	return eulerAnglesToRotationMatrix(euler_angles)

#Get rotation matrix from axis angles.
def axisAnglesToRotationMatrix(axis_angles, theta):

	if theta == 0:
		return np.eye(3)

	assert(abs(np.linalg.norm(axis_angles) - 1) <= 0.0001)
	axis_angles_theta = np.copy(axis_angles) * theta
	return o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles_theta)

def convert_file_to_model(filename, scale = 1):
	try:
		model = o3d.io.read_triangle_mesh(filename)

		#its a pointcloud
		if np.array(model.triangles).shape[0] == 0:
			print('seems like the file is a pointcloud')
			model = o3d.io.read_point_cloud(filename)

		model.scale(scale, model.get_center())
		model.translate(np.asarray([0, 0, 0]), False)

		return model
	except:
		print('load failed xd')
		raise

def place_model(model, xyz, axis_angles, theta):
	rotate_matrix_axis_angle(axis_angles, theta, model)
	model.translate(xyz, relative=False)

def return_origin(model, xyz, axis_angles, theta):
	model.translate(np.zeros(3), relative=False)
	invert_rotate_matrix_axis_angle(axis_angles, theta, model)

def get_bb(model):
	bb = model.get_axis_aligned_bounding_box()
	return bb

# def rotate_matrix_euler(euler_angles, model):
# 	#print('euler ', eulerAnglesToRotationMatrix(euler_angles))
# 	model.rotate(eulerAnglesToRotationMatrix(euler_angles))
# 	return model

# def invert_rotate_matrix_euler(euler_angles, model):
# 	#print('euler ', eulerAnglesToRotationMatrix(euler_angles))
# 	model.rotate(eulerAnglesToRotationMatrix(euler_angles).T)
# 	return model

def rotate_matrix_axis_angle(axis_angles, theta, model):
	model.rotate(axisAnglesToRotationMatrix(axis_angles, theta))
	return model

def invert_rotate_matrix_axis_angle(axis_angles, theta, model):
	model.rotate(axisAnglesToRotationMatrix(axis_angles, theta).T)
	return model

def get_x_vec(model):
	center = np.asarray(model.get_center())

	lst = []
	for i in range(100):
		lst.append(center + np.asarray([0.01 * i, 0, 0]))

	pcld = o3d.geometry.PointCloud()
	pcld.points = o3d.utility.Vector3dVector(np.asarray(lst))
	return pcld

def get_y_vec(model):
	center = np.asarray(model.get_center())

	lst = []
	for i in range(100):
		lst.append(center + np.asarray([0, 0.01 * i, 0]))

	pcld = o3d.geometry.PointCloud()
	pcld.points = o3d.utility.Vector3dVector(np.asarray(lst))
	return pcld

def get_z_vec(model):
	center = np.asarray(model.get_center())

	lst = []
	for i in range(100):
		lst.append(center + np.asarray([0, 0, 0.01 * i]))

	pcld = o3d.geometry.PointCloud()
	pcld.points = o3d.utility.Vector3dVector(np.asarray(lst))
	return pcld

#perform augmentations
#add noise to points and color
#delete points in sections?
def augment_pointcloud(pointcloud):

	# point_noise_std = 0.01
	# color_noise_mean = -0.2
	# color_noise_std = 0.1

	# color_channel_noise_std = 0.02

	max_holes = 3
	holes_size_mean = 0.03
	holes_size_std = 0.01

	#pointcloud.points = o3d.utility.Vector3dVector(np.random.normal(np.array(pointcloud.points), point_noise_std))
	#pointcloud.colors = o3d.utility.Vector3dVector(np.array(pointcloud.colors) + np.random.normal(color_noise_mean, color_noise_std))
	#pointcloud.colors = o3d.utility.Vector3dVector(np.random.normal(np.array(pointcloud.colors), color_channel_noise_std))

	kd_tree = o3d.geometry.KDTreeFlann(pointcloud)
	points = np.array(pointcloud.points)
	lst = []

	for i in range(np.random.randint(max_holes)):
		point = points[np.random.randint(points.shape[0])]
		v = kd_tree.search_radius_vector_3d(point, max(0, np.random.normal(holes_size_mean, holes_size_std)))[1]
		lst.extend(v)

	pointcloud = pointcloud.select_by_index(lst, invert=True)

	return pointcloud


#projects pointcloud to rgb image, depth image, and mask
def project_pointcloud(model):
	assert(type(model) == type(o3d.geometry.PointCloud()))
	points = np.asarray(model.points)
	colors = (np.asarray(model.colors) * 255).astype(np.uint8)
	normals = np.asarray(model.normals)

	#compute which points are facing us
	towards_cam = -1 * model.get_center() / np.linalg.norm(model.get_center())
	towards_cam_mask = np.dot(normals, towards_cam) > 0

	points = points[towards_cam_mask]
	colors = colors[towards_cam_mask]
 
	rvec = np.array([0,0,0], np.float) # rotation vector
	tvec = np.array([0,0,0], np.float) # translation vector
	h, w = 480, 640	#resulting image size
	fx = fy = 380. #focal length
	cx, cy = 320, 240 #center pixel 
	intrMatrix = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])

	image_points, _ = cv2.projectPoints(points, rvec, tvec, intrMatrix, None)
	image_points = image_points.squeeze().astype(np.int)
	image_points = image_points[:,::-1] #reverse from x,y to y,x uv coordinates

	out_rgb = np.zeros((h, w, 3)).astype(np.uint8)
	out_depth = np.zeros((h, w)).astype(np.uint16)

	#sort all points by distance from camera
	point_dist = np.sqrt(np.sum(np.square(points), axis=1))
	p = point_dist.argsort()
	point_dist = point_dist[p]
	image_points = image_points[p]
	colors = colors[p]

	image_points[image_points[:,0] < 0] = np.array([-1, -1])
	image_points[image_points[:,0] >= h] = np.array([-1, -1])
	image_points[image_points[:,1] < 0] = np.array([-1, -1])
	image_points[image_points[:,1] >= w] = np.array([-1, -1])

	p = np.argwhere(image_points[:,0] != -1).squeeze()

	point_dist = point_dist[p]
	image_points = image_points[p]
	colors = colors[p]

	#no points on screen
	if (image_points.shape[0] == 0):
		return False, (), ()

	u, idx = np.unique(image_points, return_index=True, axis=0)

	#very few points on screen
	if (u.shape[0] < 500):
		return False, (), ()

	rows, cols = zip(*image_points)
	out_rgb[rows, cols] = colors

	#make sure black points are added last
	zero_mask = np.sum(colors, axis=1) < 20
	image_points_zero = image_points[zero_mask]
	colors_zero = colors[zero_mask]

	if image_points_zero.shape[0] > 0:
		rows, cols = zip(*image_points_zero)
		out_rgb[rows, cols] = colors_zero
	
	rows, cols = zip(*u)
	out_depth[rows, cols] = (point_dist[idx] * 1000).astype(np.uint16)

	mask = np.zeros((h, w)).astype(np.uint8)
	u = u.transpose()
	miny, maxy, minx, maxx = np.min(u[0]), np.max(u[0]), np.min(u[1]), np.max(u[1])
	mask[miny:maxy,minx:maxx] = 255

	return True, (out_rgb, out_depth, mask), (h, w, fx, fy, cx, cy)


#return pointcloud, bounding box, axis_angles, theta, images, intrinsics
def get_perspective_data_from_model(model, xyz, axis_angles, theta):

	bb = get_bb(model)
	place_model(model, xyz, axis_angles, theta)

	bb.translate(xyz, relative=False)

	#insert noise into model
	model = augment_pointcloud(model)

	success, images, intr = project_pointcloud(model)

	return_origin(model, xyz, axis_angles, theta)

	return model, bb, axis_angles, theta, success, images, intr

#return pointcloud, bounding box, axis_angles, theta, images, intrinsics
def get_perspective_data_from_model_seed(seed, model, center=np.array([0, 0, 0.5]), scene_scale = 1):
	np.random.seed(seed)

	axis_angles = np.random.uniform(-1, 1, size=3)
	axis_angles /= np.linalg.norm(axis_angles)
	theta = np.random.uniform(0, np.pi * 2)

	xyz = np.copy(center)
	xyz[0] += np.random.uniform(0, 0.5) * (-1 if np.random.rand() < 0.5 else 1) * scene_scale
	xyz[1] += np.random.uniform(0, 0.5) * (-1 if np.random.rand() < 0.5 else 1) * scene_scale
	xyz[2] += np.random.uniform(0, 0.3) * (-1 if np.random.rand() < 0.5 else 1) * scene_scale

	return get_perspective_data_from_model(model, np.asarray(xyz), axis_angles, theta)

"""
create the dataset
create a yaml with intrinsics and other metadata
Output formatted like:
	output_dir/
		data/
			object1/
				train.txt
				test.txt
				rgb/
					sample1.png
					sample2.png
					...
				depth/
					sample1.png
					sample2.png
					...
				mask/
					sample1.png
					sample2.png
					...
				gt.yaml
				...
		models/
			object1.ply
"""
def create_dataset(filename, output_dir, num_train=15000, num_test=5000):
	model = convert_file_to_model(filename, 0.001)

	for i in range(50):
		pcld, bb, axis_angles, theta, success, images, intr = get_perspective_data_from_model_seed(i, model)

		if success:
			cv2.imwrite("test_rgb" + str(i) + ".png", images[0])
			cv2.imwrite("test_depth" + str(i) + ".png", images[1])
			cv2.imwrite("test_mask" + str(i) + ".png", images[2])
		else:
			print('cad model was not on screen')


def main():
	parser = argparse.ArgumentParser(description='Extract a model and data from an obj')
	parser.add_argument('--filename', default='medical/medical_aruco_sampled.ply', help='file path to model')
	parser.add_argument('--output_dir', default='data', help='output for dataset')

	args = parser.parse_args()
	print('starting processing %s' % args.filename)

	create_dataset(args.filename, args.output_dir)


if __name__ == "__main__":
	main()