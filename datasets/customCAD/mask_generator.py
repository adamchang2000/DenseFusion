import os
import cv2
import numpy as np

class UnityMaskGenerator:
    def __init__(self, root):
        self.root = root
        assert(os.path.isdir(root))
        assert(os.path.isdir(os.path.join(root, "data")))
    def generate_masks(self):
        data_dir = os.path.join(self.root, "data")
        for obj_dir in os.listdir(data_dir):
            depth_dir = os.path.join(self.root, "data", obj_dir, "depth")  
            mask_dir = os.path.join(self.root, "data", obj_dir, "mask")

            for image_file in os.listdir(depth_dir):
                img = cv2.imread(os.path.join(depth_dir, image_file), cv2.IMREAD_UNCHANGED)
                a = np.where(img != np.max(img))
                if (np.sum(a) > 0):
                    bbox = np.array([[np.min(a[0]), np.min(a[1])], [np.max(a[0]), np.max(a[1])]])
                else:
                    bbox = np.zeros((2,2)).astype(np.int64)
                
                mask = np.zeros(img.shape).astype(np.uint16)
                mask[bbox[0][0]:bbox[1][0],bbox[0][1]:bbox[1][1]] = 65535

                output_filename = os.path.join(mask_dir, image_file[-8:])
                cv2.imwrite(output_filename, mask)

def main():
    umg = UnityMaskGenerator('dataset_processed')
    umg.generate_masks()

if __name__ == '__main__':
    main()