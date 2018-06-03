import os
import time
import argparse

import json
import torch

import pandas as pd

from annoy import AnnoyIndex

from tqdm import tqdm

from lib.img_embedder import ImgEmbedder
from lib.log import Logger


def main(project_name):

    tic = time.time()

    logger = Logger('_05_make_submission_1000_{}'.format(project_name))
    logger.info('=' * 50)

    model_path = '_model/embedding_model_{}.pt'.format(project_name)
    logger.info('load model from {}'.format(model_path))
    model = torch.load(model_path)
    model.eval()

    dir_target = '../../input/test'
    embedder = ImgEmbedder(model, dir_target)

    sample_submission = pd.read_csv('../../dataset/sample_submission.csv')

    images = list()

    with open(os.path.join('_embed_index', 'index_names_{}.json'.format(project_name)), 'r') as f:
        index_names = json.load(f)

    test_id_list = sample_submission.id

    f = 512
    u = AnnoyIndex(f, metric='euclidean')
    u.load(os.path.join('_embed_index', 'index_features_{}.ann'.format(project_name)))

    logger.info('===> embed test images and get nearest neighbors')

    search_k = 1_000_000

    for test_id in tqdm(test_id_list):

        target_file = '{}.jpg'.format(test_id)

        try:
            img_feature = embedder.get_vector(target_file)
            indeces = u.get_nns_by_vector(img_feature.tolist(), n=1000, search_k=search_k)
        except:
            indeces = list(range(1000))

        names = [index_names[index] for index in indeces]

        images.append(' '.join(names))

    submission = pd.DataFrame(test_id_list, columns=['id'])
    submission['images'] = images

    output_path = '../../submission/submission_1000_{}.csv'.format(project_name)
    submission.to_csv(output_path, index=False)

    toc = time.time() - tic
    logger.info('Elapsed time: {:.1f} [min]'.format(toc / 60.0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', default='', help='project name')
    params = parser.parse_args()

    main(project_name=params.name)
