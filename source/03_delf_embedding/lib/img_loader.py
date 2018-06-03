import os
import numpy as np

from delf import feature_io

import torch


class ImgLoader:

    def __init__(self, dir_images):
        self.dir_images = dir_images

    def load_image(self, file_name):

        target_path = os.path.join(self.dir_images, file_name)

        try:
            locations_1, scales_1, descriptors, _, _ = feature_io.ReadFromFile(target_path)
            num_features_1 = locations_1.shape[0]
        except:
            return torch.randn(1, 40, 1000)

        descriptors = np.swapaxes(descriptors, 0, 1)
        zeros = np.zeros((1, 40, 1000))
        zeros[:, :, :num_features_1] = descriptors

        return torch.FloatTensor(zeros)
