# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
This script is forked from tensorflow-delf !
https://github.com/tensorflow/models/tree/master/research/delf
"""

import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import time
import numpy as np
import pandas as pd
import warnings

from tqdm import tqdm
import multiprocessing

from scipy.spatial import cKDTree
from skimage.measure import ransac
from skimage.transform import AffineTransform
from delf import feature_io

_DISTANCE_THRESHOLD = 0.75
NUM_SUB_QUERY = 2
NUM_TARGET_PREQUERY = 50
NUM_TARGET_MAINQUERY = 200


def get_num_inlier(locations_2, descriptors_2, path1):

    num_features_2 = locations_2.shape[0]

    try:
        locations_1, _, descriptors_1, _, _ = feature_io.ReadFromFile(path1)
        num_features_1 = locations_1.shape[0]
    except:
        # print('Warning: Failed to load index feature')
        return -1

    # Find nearest-neighbor matches using a KD tree.
    d1_tree = cKDTree(descriptors_1)

    _, indices = d1_tree.query(descriptors_2, distance_upper_bound=_DISTANCE_THRESHOLD)

    # Select feature locations for putative matches.

    locations_2_to_use = np.array([locations_2[i, ] for i in range(num_features_2)
                                   if indices[i] != num_features_1])

    locations_1_to_use = np.array([locations_1[indices[i], ] for i in range(num_features_2)
                                   if indices[i] != num_features_1])

    if len(locations_1_to_use) == 0 or len(locations_2_to_use) == 0:
        return 0

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Perform geometric verification using RANSAC.
            _, inliers = ransac(
                    (locations_1_to_use, locations_2_to_use),
                    AffineTransform,
                    min_samples=3,
                    residual_threshold=80,
                    max_trials=40)
    except:
        return 0

    if inliers is None:
        return 0

    return sum(inliers)


def get_list_inliers(path1, list_images):

    try:
        locations_t, _, descriptors_t, _, _ = feature_io.ReadFromFile(path1)
    except:
        # print('Warning: Failed to load test feature')
        return range(len(list_images), 0, -1)

    list_inliers = list()

    for img in list_images:
        path2 = os.path.join('../../input_large_delf/index', img + '.delf')
        list_inliers.append(get_num_inlier(locations_t, descriptors_t, path2))

    return list_inliers


def sort_by_delf(target_row):

    query = target_row['id']
    list_images = target_row['images'].split(' ')
    list_images = list(set(list_images))

    # pre query

    path1 = os.path.join('../../input_large_delf/test', query + '.delf')

    list_inliers = get_list_inliers(path1, list_images[:max(NUM_TARGET_PREQUERY, NUM_TARGET_MAINQUERY)])
    array_inliers = np.array(list_inliers)[:NUM_TARGET_PREQUERY]

    sort_index = np.argsort(-array_inliers)

    # query expansion

    list_sub_query = np.array(list_images)[sort_index].tolist()[:NUM_SUB_QUERY]

    all_inliers = [list_inliers[:NUM_TARGET_MAINQUERY]]

    for sub_query in list_sub_query:
        path1_sub = os.path.join('../../input_large_delf/index', sub_query + '.delf')
        list_inliers = get_list_inliers(path1_sub, list_images[:NUM_TARGET_MAINQUERY])
        all_inliers.append(list_inliers)

    sort_index = np.argsort(-np.mean(np.array(all_inliers), axis=0))

    list_result = np.array(list_images)[sort_index].tolist()
    return ' '.join(list_result[:100])


def main():

    tic = time.time()

    chunk_size = 3200

    pre_submission_chunk = pd.read_csv('../../submission/*****.csv', chunksize=chunk_size)

    list_images = list()
    list_ids = list()

    num_chunk = -(-117703 // chunk_size)

    for pre_submission in tqdm(pre_submission_chunk, position=-1, desc='ALL', total=num_chunk):

        print('')

        pre_submission = pd.DataFrame(pre_submission)

        list_pre_submission = [pre_submission.iloc[i, :] for i in range(len(pre_submission))]

        pool = multiprocessing.Pool(processes=96)

        images = tqdm(pool.imap(sort_by_delf, list_pre_submission), position=0, desc='chunk',
                      total=len(list_pre_submission))

        list_images.extend(list(images))
        list_ids.extend(pre_submission.id)

        pool.close()
        pool.terminate()

    submission = pd.DataFrame(list_ids, columns=['id'])
    submission['images'] = list_images

    output_path = '../../submission/submission.csv'
    submission.to_csv(output_path, index=False)

    toc = time.time() - tic
    print('Elapsed time: {:.1f} [min]'.format(toc / 60.0))


if __name__ == '__main__':

    main()
