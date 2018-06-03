import time
import argparse
import torch

from lib.log import Logger
from lib.train import train
from lib.build_model import build_model

# ------------------------------
# global settings

SEED = 1000
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def main(project_name):

    tic = time.time()

    logger = Logger('_01_training_{}'.format(project_name))

    logger.info('==> initialize model')
    embedding = build_model(pretrained=True)

    logger.info('==> train model')
    train(embedding, project_name=project_name)

    toc = time.time() - tic
    logger.info('Elapsed time: {:.1f} [min]'.format(toc / 60.0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', default='', help='project name')
    params = parser.parse_args()

    main(project_name=params.name)
