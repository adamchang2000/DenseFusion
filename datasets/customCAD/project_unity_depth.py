import cv2
import numpy as np
import open3d as o3d

class UnityDepthProjector:
    def __init__(self, proj_file, image_dims):
        self.image_dims = image_dims
        self.proj_mat = np.zeros((4, 4))
        i = 0
        for line in open(proj_file, 'r'):
            if i == 4:
                break
            k = 0
            for elem in line.split("\t"):
                elem = elem.strip().rstrip()
                self.proj_mat[i,k] = float(elem)
                k += 1
            i += 1
        
        self.inverse_proj_mat = np.linalg.inv(self.proj_mat)

        x_range = np.arange(-1, 1, 2. / image_dims[1])
        y_range = np.arange(-1, 1, 2. / image_dims[0])

        #need to invert the y-axis
        pixel_map = np.array([[[x_range[i], -y_range[k]] for i in range(image_dims[1])] for k in range(image_dims[0])])

        #z_map = depth[...,np.newaxis]
        z_map = np.ones((pixel_map.shape[0], pixel_map.shape[1], 1)) * -1
        w_map = np.ones((pixel_map.shape[0], pixel_map.shape[1], 1))

        pixel_map = np.concatenate((pixel_map, z_map, w_map), axis=2)
        pixel_map = pixel_map[..., np.newaxis]

        ray_map = np.matmul(self.inverse_proj_mat, pixel_map).squeeze()

        ray_map /= ray_map[:,:,3,np.newaxis]
        ray_map /= ray_map[:,:,2,np.newaxis]

        self.ray_map = ray_map[:,:,:3]

    def project_depth(self, image):
        assert(image.shape == self.image_dims)

        depth = image.astype(np.float64) / 65534
        depth = 1-depth
        depth = -self.proj_mat[2,3] / (self.proj_mat[2,2] + depth)

        world_ray_map = np.copy(self.ray_map)
        world_ray_map *= depth[...,np.newaxis]
        return world_ray_map

    def project_depth_file(self, filename):
        image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

        assert(image.shape == self.image_dims)

        depth = image.astype(np.float64) / 65534
        depth = 1-depth
        depth = -self.proj_mat[2,3] / (self.proj_mat[2,2] + depth)

        world_ray_map = np.copy(self.ray_map)
        world_ray_map *= depth[...,np.newaxis]
        return world_ray_map



def main():
    depth_file = r"C:\Users\adam\Desktop\DenseFusion\datasets\customCAD\dataset_processed\data\01\depth\Depth_0000.png"
    proj_file = r"C:\Users\adam\Desktop\DenseFusion\datasets\customCAD\dataset_processed\data\01\meta\proj_mat.txt"

    ud_projector = UnityDepthProjector(proj_file, (520, 1109))
    projected = ud_projector.project_depth_file(depth_file)
    projected = projected.reshape((-1, 3))
    origin = np.array([[0, 0, 0]])
    
    pcld = o3d.geometry.PointCloud()
    pcld.points = o3d.utility.Vector3dVector(np.vstack((projected, origin)))

    o3d.visualization.draw_geometries([pcld])


if __name__ == "__main__":
    main()

