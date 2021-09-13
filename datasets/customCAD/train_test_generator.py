import numpy as np
import os
import random

class TrainTestGenerator():
    def __init__(self, root, train_percent=80):
        self.root = root
        assert(os.path.isdir(root))
        assert(os.path.isdir(os.path.join(root, "data")))
    def generate_traintest(self):
        data_dir = os.path.join(self.root, "data")
        for obj_dir in os.listdir(data_dir):
            depth_dir = os.path.join(self.root, "data", obj_dir, "depth")
            train_filename = os.path.join(self.root, "data", obj_dir, "train.txt")
            test_filename = os.path.join(self.root, "data", obj_dir, "test.txt")

            image_files = list(os.listdir(depth_dir))
            image_nums = [int(x[x.find("_") + 1:x.find(".")]) for x in image_files]

            random.shuffle(image_nums)

            with open(train_filename, 'w') as f:
                for num in image_nums[:int(len(image_nums) / 100. * 80)]:
                    f.write(str(num) + "\n")

            with open(test_filename, 'w') as f:
                for num in image_nums[int(len(image_nums) / 100. * 80):]:
                    f.write(str(num) + "\n")



            

def main():
    ttg = TrainTestGenerator("dataset_processed")
    ttg.generate_traintest()

if __name__ == "__main__":
    main()