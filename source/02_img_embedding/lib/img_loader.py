import os
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms


class ImgLoader:

    def __init__(self, dir_images):
        self.dir_images = dir_images

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def load_image(self, file_name):

        target_path = os.path.join(self.dir_images, file_name)

        img = Image.open(target_path).convert('RGB')
        img = img.resize((224, 224))
        img = np.array(img).astype(np.float)

        input_img = self.normalize(self.to_tensor(img))
        input_tensor = torch.unsqueeze(input_img, 0)

        return input_tensor
