from mask_generator import UnityMaskGenerator
from train_test_generator import TrainTestGenerator

umg = UnityMaskGenerator("dataset_processed")
umg.generate_masks()


ttg = TrainTestGenerator("dataset_processed")
ttg.generate_traintest()