import os
import random
from tqdm import tqdm

import numpy as np
import pandas as pd

import itertools
from annoy import AnnoyIndex

from lib.img_embedder import ImgEmbedder

import logging
logger = logging.getLogger('root')


class Evaluator:

    N = 100  # number of landmarks for evaluation
    M = 101  # number of images for each class
    K = 100  # evaluate by map@K

    F = 512  # embedding dimension

    dir_train = '../../input_large_delf/train'

    def __init__(self):

        train_csv = pd.read_csv('../../dataset/train.csv')
        logger.info('number of train images (train.csv): {:d}'.format(len(train_csv)))
        self.dict_id2landmark = {img_id: landmark for img_id, landmark in zip(train_csv.id, train_csv.landmark_id)}

        train_imgs = os.listdir('../../input/train')
        train_ids = [s[:-4] for s in train_imgs]
        train_landmarks = [self.dict_id2landmark[s] for s in train_ids]
        # print(train_landmarks)

        available_train_data = pd.DataFrame(train_ids, columns=['id'])
        available_train_data['landmark_id'] = train_landmarks

        logger.info('number of train images (avaiable): {:d}'.format(len(available_train_data)))

        landmark_count = available_train_data.landmark_id.value_counts()
        logger.info('number of landmarks: {:d}'.format(len(landmark_count)))
        landmark_topn = landmark_count[:self.N].index

        train_info = available_train_data[available_train_data.landmark_id.isin(landmark_topn)]

        series_landmark_images = train_info.groupby('landmark_id')['id'].apply(list)
        series_landmark_images = series_landmark_images.apply(lambda x: x[:self.M])

        self.valid_images = list(itertools.chain.from_iterable(series_landmark_images))
        logger.info('number of valid images: {:d}'.format(len(self.valid_images)))

    def evaluate(self, model):

        logger.info('==> evaluate model')

        t = AnnoyIndex(self.F, metric='euclidean')

        model.eval()
        embedder = ImgEmbedder(model, self.dir_train)

        logger.info('===> embed train images')

        for i, valid_image_id in tqdm(enumerate(self.valid_images), total=len(self.valid_images)):
            img_feature = embedder.get_vector('{}.delf'.format(valid_image_id))
            t.add_item(i, img_feature.tolist())

        t.build(1000)

        average_precisions = np.array([])

        logger.info('===> embed valid images and get nearest neighbors')

        for valid_image_id in tqdm(self.valid_images):

            img_feature = embedder.get_vector('{}.delf'.format(valid_image_id))

            knn_index = t.get_nns_by_vector(img_feature.tolist(), n=self.K+1)
            knn_id = [self.valid_images[i] for i in knn_index]

            try:
                knn_id.remove(valid_image_id)
            except ValueError:
                knn_id = knn_id[:-1]

            target_img_class = self.dict_id2landmark[valid_image_id]
            near_img_classes = [self.dict_id2landmark[valid_image_id] for valid_image_id in knn_id]

            hit_index = np.array(near_img_classes) == target_img_class
            precisions = np.cumsum(hit_index) / np.arange(1, self.K+1)

            precisions[~hit_index] = 0

            average_precision = np.mean(precisions)
            average_precisions = np.append(average_precisions, average_precision)

        mean_average_precision = np.mean(average_precisions)

        logger.info('')
        logger.info('mean average precision: {:.4f}'.format(mean_average_precision))
        logger.info('')
