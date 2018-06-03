import argparse

from lib.log import Logger
from lib.evaluator import Evaluator

import torch

import logging
logger = logging.getLogger('root')


def main(project_name):

    logger = Logger('_02_valid_model_{}'.format(project_name))
    logger.info('=' * 50)

    model_path = '_model/embedding_model_{}.pt'.format(project_name)
    logger.info('load model from {}'.format(model_path))
    model = torch.load(model_path)

    evaluator = Evaluator()
    evaluator.evaluate(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', default='', help='project name')
    params = parser.parse_args()

    main(project_name=params.name)
