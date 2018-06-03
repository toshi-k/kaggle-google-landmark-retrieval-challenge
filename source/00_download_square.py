import os
import time
from urllib import request
from PIL import Image
from io import BytesIO

from tqdm import tqdm

import pandas as pd
import multiprocessing

import functools

'''
This script is forked from anokas's kaggle-kernel !
https://www.kaggle.com/anokas/py3-image-downloader-w-progress-bar
'''


def download_image(key_url, dir_output):

    (key, url) = key_url
    filename = os.path.join(dir_output, '{}.jpg'.format(key))

    if os.path.exists(filename):
        print('Image {} already exists. Skipping download.'.format(filename))
        return 0

    try:
        response = request.urlopen(url)
        image_data = response.read()
    except:
        print('Warning: Could not download image {} from {}'.format(key, url))
        return 1

    try:
        pil_image = Image.open(BytesIO(image_data))
    except:
        print('Warning: Failed to parse image {}'.format(key))
        return 1

    try:
        pil_image_rgb = pil_image.convert('RGB')
    except:
        print('Warning: Failed to convert image {} to RGB'.format(key))
        return 1

    try:
        width, height = pil_image_rgb.size
        size = min(width, height)

        w_center = width // 2
        h_center = height // 2

        pil_image_crop = pil_image_rgb.crop((w_center - size // 2, h_center - size // 2,
                                             w_center + size // 2, h_center + size // 2))

        pil_image_resize = pil_image_crop.resize((320, 320))
        pil_image_resize.save(filename, format='JPEG', quality=95)
    except:
        print('Warning: Failed to save image {}'.format(filename))
        return 1

    return 0


def save_images(csv_path):

    tic = time.time()

    data = pd.read_csv(csv_path)
    data_type = os.path.basename(csv_path)[:-4]

    data_series = data.apply(lambda row: (row['id'], row['url']), axis=1)

    dir_output = os.path.join('../input', data_type)
    os.makedirs(dir_output, exist_ok=True)

    download_image_dir = functools.partial(download_image, dir_output=dir_output)

    pool = multiprocessing.Pool(processes=8)  # Num of CPUs
    failures = sum(tqdm(pool.imap_unordered(download_image_dir, data_series), total=len(data_series)))
    print('Total number of download failures:', failures)
    pool.close()
    pool.terminate()

    toc = time.time() - tic
    print('Elapsed time: {:.0f} [min]'.format(toc / 60.0))


def main():
    save_images('../dataset/index.csv')
    save_images('../dataset/test.csv')
    save_images('../dataset/train.csv')


if __name__ == '__main__':
    main()
