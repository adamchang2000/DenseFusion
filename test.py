import torch
from lib.network import ModifiedResnet
import numpy as np

model = ModifiedResnet()
model.cuda()

input = np.zeros((1, 3, 20, 5)).astype(np.float32)
x = torch.from_numpy(input).cuda()
x = model(x)

print(x.shape)