# coding=utf-8

"""
Module to train a NN
"""

import os
import argparse
import numpy as np
import time
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader


import hamlyn2021.unet.Unet_v2 as u
import hamlyn2021.data_reader.pytorch_data_reader as pdr
import hamlyn2021.data_processing.data_scraping as ds

def train_func(path_base):
    """
    Function to train a network using random views and depth views
    """
    device = "cuda:0"


    path_data_train = os.path.join(path_base, "/translations_random_views")
    path_data_labels = os.path.join(path_base, "/depth_random_views")

    # Check if data exists, if not, download
    if not os.path.exists(os.path.join(path_base, "translation_sequences")):
        # make a download objects
        obj_download = ds.SequenceDownloader()
        obj_download.downloadSequenceTranslations(path_base)

    # Check if data exists, if not, download
    if not os.path.exists(os.path.join(path_base, "depth_sequences")):
        # make a download objects
        obj_download = ds.SequenceDownloader()
        obj_download.downloadSequenceDepth(path_base)

    if not os.path.exists(os.path.join(path_base, "translations_random_views")):
        # make a download objects
        obj_download = ds.RandomDownloader()
        obj_download.downloadRandomTranslations(path_base)

    # Check if data exists, if not, download
    if not os.path.exists(os.path.join(path_base, "depth_random_views")):
        # make a download objects
        obj_download = ds.RandomDownloader()
        obj_download.downloadRandomDepth(path_base)


    train_data_loader, val_data_loader = pdr.get_dataloaders(input_dir=path_data_train,
                                                            depth_dir=path_data_labels)

    net = u.UNet()
    net = net.to(device)
    MSE_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-05, betas=(0.5, 0.999))
    optimizer.zero_grad()
    all_error = np.zeros(0)
    all_val_error = np.zeros(0)
    epochs=10000

    for epoch in range(epochs):

        ##########
        # Train
        ##########
        t0 = time.time()
        for i, (data, label) in enumerate(train_data_loader):
            
            # setting your network to train will ensure that parameters will be updated during training, 
            # and that dropout will be used
            net.train()
            net.zero_grad()

            batch_size = data.size()[0]
            pred = net(data)

            # loss function here
            err = MSE_loss(pred, label)
            # -------------------------------------------------------------------------------------------------------------------

            err.backward()
            optimizer.step()
            optimizer.zero_grad()

            time_elapsed = time.time() - t0
            print('[{:d}/{:d}][{:d}/{:d}] Elapsed_time: {:.0f}m{:.0f}s Loss: {:.4f} MSE: {:.4f}'
                .format(epoch, epochs, i, len(train_data_loader), time_elapsed // 60, time_elapsed % 60,
                        err.item(), err))

            error = err.item()

            all_error = np.append(all_error, error)

        for i, (data, label) in enumerate(val_data_loader):
            
            net.evaluate()
            net.zero_grad()

            batch_size = data.size()[0]
            pred = net(data)

            # loss function here
            err = MSE_loss(pred, label)
            # -------------------------------------------------------------------------------------------------------------------

            time_elapsed = time.time() - t0
            print('[{:d}/{:d}][{:d}/{:d}] Elapsed_time: {:.0f}m{:.0f}s Val Loss: {:.4f} Val MSE: {:.4f}'
                .format(epoch, epochs, i, len(train_data_loader), time_elapsed // 60, time_elapsed % 60,
                        err.item(), err))

            val_error = err.item()

            all_error = np.append(all_val_error, error)

        if epoch % 100:
            torch.save(net.state_dict(), os.path.join(path_base, "state_dict_model_unet_{}.pt".format(str(time_elapsed))))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train_12dof')

    ## ARGS
    parser.add_argument("--path_data",
                        "-d",
                        help="Path to data for simulation",
                        required=True,
                        type=str,
                        default=None)

    args = parser.parse_args()
    train_func(args.path_data)