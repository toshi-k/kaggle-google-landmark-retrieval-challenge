import os
import time
import argparse

import json
import torch

from annoy import AnnoyIndex

from tqdm import tqdm

from lib.img_embedder import ImgEmbedder
from lib.log import Logger


def main(project_name):

    tic = time.time()

    logger = Logger('_03_embed_index_{}'.format(project_name))
    logger.info('=' * 50)

    model_path = '_model/embedding_model_{}.pt'.format(project_name)
    logger.info('load model from {}'.format(model_path))
    model = torch.load(model_path)
    model.eval()

    dir_target = '../../input/index'
    embedder = ImgEmbedder(model, dir_target)

    f = 512
    t = AnnoyIndex(f, metric='euclidean')

    target_files = os.listdir(dir_target)

    num_index = len(target_files)

    index_names = list()

    logger.info('===> embed index images')

    for i in tqdm(range(num_index)):

        target_file = target_files[i]
        index_names.append(target_file[:-4])

        img_feature = embedder.get_vector(target_file)
        t.add_item(i, img_feature.tolist())

    dir_index = '_embed_index'
    os.makedirs(dir_index, exist_ok=True)

    with open(os.path.join(dir_index, 'index_names_{}.json'.format(project_name)), 'w') as f:
        json.dump(index_names, f)

    t.build(100)
    t.save(os.path.join(dir_index, 'index_features_{}.ann'.format(project_name)))

    toc = time.time() - tic
    logger.info('Elapsed time: {:.1f} [min]'.format(toc / 60.0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', default='', help='project name')
    params = parser.parse_args()

    main(project_name=params.name)
