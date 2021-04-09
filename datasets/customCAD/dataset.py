import torch.utils.data as data

class CustomCADDataset(data.Dataset):
    def __init__(self, mode, num, add_noise, root, noise_trans, refine):
        pass

    def __getitem__(self, index):
        #outputs:
        #pointcloud: xyz points of projected cropped depth image (just the object region)
        #choose: select num_points points from the cropped region of the image, this is a mask
        #img_masked: cropped region of image (RGB) with object
        #target: Rt matrix
        #model_points: just the model points, num_pt_mesh_small of them
        #obj: class (in our case, just a single number since we only have 1 object)
        pass
        

    def __len__(self):
        pass

    def get_sym_list(self):
        pass

    def get_num_points_mesh(self):
        pass