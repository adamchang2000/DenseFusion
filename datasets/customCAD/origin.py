import numpy as np
import open3d as o3d

point = np.array([[0, 0, 0]])

origin = o3d.geometry.PointCloud()
origin.points = o3d.utility.Vector3dVector(point)

o3d.io.write_point_cloud("origin.ply", origin)