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

def train_func(path_base, device=None, wandb_un=None, dataset_type="random"):
    """
    Function to train a network using random views and depth views
    """
    if device is None:
        device = "cuda:0"

    if wandb_un:
        wandb.init(project="Hamlyn2021", entity=wandb_un)

    path_data_train = os.path.join(path_base, "translation_random_views/random_views/")
    path_data_labels = os.path.join(path_base, "depth_random_views/random_views")


    train_data_loader, val_data_loader = pdr.get_dataloaders(input_dir=path_data_train,
                                                             depth_dir=path_data_labels,
                                                             dataset_type=dataset_type)

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
            pred = net(data.to(device).float())

            # loss function here
            err = MSE_loss(pred, label.to(device).float())
            # -------------------------------------------------------------------------------------------------------------------

            err.backward()
            optimizer.step()
            optimizer.zero_grad()

            time_elapsed = time.time() - t0
            print('[{:d}/{:d}][{:d}/{:d}] Elapsed_time: {:.0f}m{:.0f}s Loss: {:.4f} MSE: {:.4f}'
                .format(epoch, epochs, i, len(train_data_loader), time_elapsed // 60, time_elapsed % 60,
                        err.item(), err))

            error = err.detach().cpu().item()

            all_error = np.append(all_error, error)

        for i, (data, label) in enumerate(val_data_loader):
            net.eval()
            net.zero_grad()

            batch_size = data.size()[0]
            pred = net(data.to(device).float())

            # loss function here
            err = MSE_loss(pred, label.to(device).float())
            # -------------------------------------------------------------------------------------------------------------------

            time_elapsed = time.time() - t0
            print('[{:d}/{:d}][{:d}/{:d}] Elapsed_time: {:.0f}m{:.0f}s Val Loss: {:.4f} Val MSE: {:.4f}'
                .format(epoch, epochs, i, len(train_data_loader), time_elapsed // 60, time_elapsed % 60,
                        err.item(), err))

            val_error = err.detach().cpu().item()

            all_val_error = np.append(all_val_error, val_error)

        if epoch % 100:
            torch.save(net.state_dict(), os.path.join(path_base, "state_dict_model_unet_{}.pt".format(str(time_elapsed))))

    if wandb_un:
        wandb.log({"train_loss": all_error, "val_loss":all_val_error})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HamlynUNet')

    ## ARGS
    parser.add_argument("--path_data",
                        "-d",
                        help="Path to data for simulation",
                        required=True,
                        type=str,
                        default=None)
    parser.add_argument("--device",
                        "-c",
                        help="Device to use, defaults to cuda:0",
                        required=False,
                        type=str,
                        default=None)

    parser.add_argument("--wandb_un",
                        "-w",
                        help="Name of username for wandb log",
                        required=False,
                        type=str,
                        default=None)
    parser.add_argument("--dataset_type",
                        "-t",
                        help="Dataset to use, 'random' or 'sequence'",
                        required=False,
                        default="random")

    args = parser.parse_args()
    train_func(args.path_data, args.device, args.wandb_un, dataset_type=args.dataset_type)