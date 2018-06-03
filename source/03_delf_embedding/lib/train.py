import os
import random
import torch
import torch.optim as optim
from tqdm import tqdm

from lib.img_loader import ImgLoader
from lib.get_sample import Sampler
from lib.evaluator import Evaluator

import numpy as np

from torch.autograd import Variable

from torch.nn import functional as F

import logging
logger = logging.getLogger('root')


def smooth_pairwise_loss(anchor, positive, eps=1e-6):
    d_p = F.pairwise_distance(anchor, positive, 2) + eps
    loss = torch.mean(d_p)
    return loss


def hard_negative_triplet_loss(out_anchor, hard_positive, hard_negative):
    return F.triplet_margin_loss(out_anchor, hard_positive, hard_negative, margin=0.5)


def get_apn_index(out, batch_size, group_size):

    # anchor
    anchor_indeces = list(range(0, batch_size * group_size, group_size))

    positive_indeces = list()
    negative_indeces = list()

    count_inside_positive = 0

    for i in range(batch_size):

        expanded = torch.unsqueeze(out[anchor_indeces[i]], dim=0).expand(batch_size * group_size, 512)
        dists = F.pairwise_distance(expanded, out)

        # negative
        negative_dists = dists.clone()
        negative_dists[i*group_size:(i+1)*group_size] = float('inf')
        min_value, min_index = torch.min(negative_dists, 0)
        negative_indeces.append(int(min_index.data))

        # positive
        positive_dists = dists[i*group_size:(i+1)*group_size]
        compare = positive_dists.data > min_value.data

        if compare.any():
            count_inside_positive += 1
            compare_bool = compare.cpu().numpy().flatten().astype(np.bool)
            max_index_int = i*group_size + random.choice(np.arange(group_size)[compare_bool].tolist())
        else:
            _, max_index = torch.max(positive_dists, 0)
            max_index_int = i*group_size + int(max_index.data)

        positive_indeces.append(max_index_int)

    batch_indeces = anchor_indeces + positive_indeces + negative_indeces
    return batch_indeces, count_inside_positive


def set_batch_norm_eval(model):

    bn_count = 0
    bn_training = 0

    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm.BatchNorm1d):
            if module.training:
                bn_training += 1
            module.eval()
            bn_count += 1

    for name, param in model.named_parameters():
        if 'bn' in name:
            param.requires_grad = False


def train(model, project_name):

    sampler = Sampler()
    loader = ImgLoader('../../input_large_delf/train')
    evaluator = Evaluator()

    dir_model = '_model'
    os.makedirs(dir_model, exist_ok=True)

    # for training
    batch_size = 240
    group_size = 12
    iter_outside = 10
    iter_inside = 500

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    for param_group in optimizer.param_groups:
        logger.info('start lerning rate with: {:.6f}'.format(param_group['lr']))

    for ep in range(1, iter_outside + 1):

        logger.info('-' * 30)
        logger.info('epoch: {:d}'.format(ep))

        model.train()
        if ep > 1:
            set_batch_norm_eval(model)

        train_loss1 = 0
        train_loss3 = 0
        count_sample = 0
        ave_good_index = 0

        for _ in tqdm(range(iter_inside)):

            batch = torch.FloatTensor(batch_size * group_size, 40, 1000).zero_()
            ids = sampler.get_sample(batch_size, group_size)

            for i in range(batch_size * group_size):
                batch[i] = loader.load_image('{}.delf'.format(ids[i]))

            batch_cuda = batch.cuda()

            # forward with requires_grad=False

            v_batch_no_bp = Variable(batch_cuda, volatile=True)
            optimizer.zero_grad()
            out = model.forward(v_batch_no_bp)

            batch_indeces, num_good_index = get_apn_index(out, batch_size, group_size)

            # forward with requires_grad=True

            v_batch = Variable(batch_cuda[batch_indeces, :, :])

            optimizer.zero_grad()
            out = model.forward(v_batch)

            out_anchor = out[:batch_size]
            hard_positive = out[batch_size:batch_size*2]
            hard_negative = out[batch_size*2:batch_size*3]

            # calc loss

            loss1 = smooth_pairwise_loss(out_anchor, hard_positive) * 0.1
            loss3 = hard_negative_triplet_loss(out_anchor, hard_positive, hard_negative)

            loss = loss3

            loss.backward()
            optimizer.step()

            train_loss1 += float(loss1.data.cpu().numpy()) * batch_size
            train_loss3 += float(loss3.data.cpu().numpy()) * batch_size
            ave_good_index += num_good_index * batch_size
            count_sample += batch_size

        logger.info('train loss (pair-pos): {:.6f}'.format(train_loss1 / count_sample))
        logger.info('train loss (triplet) : {:.6f}'.format(train_loss3 / count_sample))
        logger.info('average number of far negative: {:.2f} / {:d}'.format(ave_good_index / count_sample, batch_size))

        evaluator.evaluate(model)

        if ep % 4 == 0 and ep != iter_outside:

            model_name = 'embedding_model_{}_ep{}.pt'.format(project_name, ep)
            logger.info('save model: {}'.format(model_name))
            torch.save(model, os.path.join(dir_model, model_name))

            if ep % 8 == 0:

                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1
                    logger.info('change learning rate into: {:.6f}'.format(param_group['lr']))

    model_name = 'embedding_model_{}.pt'.format(project_name)
    logger.info('save model: {}'.format(model_name))
    torch.save(model, os.path.join(dir_model, model_name))
