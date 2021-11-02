
import numpy as np
from PIL import Image
import paddle
import torchvision
import torch

img = Image.fromarray((np.random.rand(300, 320, 3) * 255.).astype(np.uint8))

fake_img1 = paddle.vision.transforms.CenterCrop(3)(img)
fake_img2 = torchvision.transforms.CenterCrop(3)(img)
print(""np.array(fake_img1))
print(np.array(fake_img2))
# out: (224, 224) width,height



