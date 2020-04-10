#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from  model.seriesNet_torch import seriesNet #, gated_block
from CausalTrainTest import TrainManager
import numpy as np
import random
import glob
import data_set.fn_500_dataset as fn_500_dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import importlib

def argument_parser():
    """
        A parser to allow user to easily experiment different models along with datasets and differents parameters
    """
    parser = argparse.ArgumentParser(usage='\n python3 train.py [model] [dataset] [hyper_parameters]'
                                           '\n python3 train.py --model UNet [hyper_parameters]'
                                           '\n python3 train.py --model UNet --predict',
                                     description="This program allows to train different models of regression on fn_500_dataset"
                                                 )
    parser.add_argument('--company', type=str, default='AAPL_data',
                        help="Company's name")
    parser.add_argument('--batch_size', type=int, default=20,
                        help='The size of the training batch')
    parser.add_argument('--nb_causal_blk', type=int, default=8,
                        help='Nb of causal block to tack')
    parser.add_argument('--nb_filter', type=int, default=32,
                        help='Nb of filter to use in SeriesNet')
    parser.add_argument('--nb_drop_blk', type=int, default=2,
                        help='Nb of causal blk where to apply dropout')
    parser.add_argument('--channel_2_use', type=str, default='close', choices=["open", "close","high","low","volume"],
                        help='Channel to use')
    parser.add_argument('--optimizer', type=str, default="sgd", choices=["Adam", "sgd"],
                        help="The optimizer to use for training the model")
    parser.add_argument('--num_epochs', type=int, default=7,
                        help='The number of epochs')
    parser.add_argument('--validation', type=float, default=20,
                        help='Percentage of training data to use for validation')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--pts_2_pred', type=int, default=10,
                        help='Points to forecast')
    parser.add_argument('--save', action='store_true',
                        help='Learning rate')
    parser.add_argument('--figname', type=str, default='fig1.png',
                        help="Name for saving figures")
    parser.add_argument('--shuffle', action='store_true',
                        help='Shuffle or not the dataset between each epoch')
    return parser.parse_args()

if __name__ == "__main__":

    args = argument_parser()

    batch_size = args.batch_size
    num_epochs = args.num_epochs
    val_percent = args.validation
    learning_rate = args.lr
    m_causal_blk = args.nb_causal_blk
    nb_filter = args.nb_filter
    nb_drop_blk = args.nb_drop_blk
    channel_2_use = args.channel_2_use

    model = seriesNet(1, nb_causal_block=m_causal_blk, gate_nb_filter=nb_filter, nb_block_dropped=nb_drop_blk)
    model.float()

    N = model.get_pts_for_Pred()
    pts_2_pred = args.pts_2_pred
    company = args.company
    validation = args.validation
    channel_2_use = args.channel_2_use
    dataset_train, dataset_eval, test_input, test_target = fn_500_dataset.create_sliding_dataset(N,
                                                                            pts_2_pred=pts_2_pred,
                                                                            proportion= validation,
                                                                            action_name=company,
                                                                            axis=channel_2_use,
                                                                            normalise=True)
    shuffle = args.shuffle
    print(shuffle)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    valid_loader = DataLoader(dataset_eval, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    manager = TrainManager(model,
                           train_loader,
                           valid_loader,
                           lr=0.001,
                           loss_fn='MeanSquared',
                           optimizer_type='sgd',
                           pts_2pred=10)

    save = args.save
    figname = args.figname
    print("Training {} for {} epochs".format(model.__class__.__name__, args.num_epochs))
    manager.train(num_epochs, display=True, figname=figname)
    manager.plot_prediction(test_input, test_target, pts_2_pred, save=save, figname=figname)



