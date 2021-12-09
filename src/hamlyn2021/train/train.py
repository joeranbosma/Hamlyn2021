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


def train_func(path_base, device=None, wandb_un=None, dataset_type="random",
               batch_size=32, epochs=10000, save_every=100, num_channels=3,
               preceding_frames=None):
    """
    Function to train a network using random views and depth views
    """
    if device is None:
        device = "cuda:0"

    if wandb_un:
        wandb.init(project="Hamlyn2021", entity=wandb_un)

    if dataset_type == "sequence":
        path_data_train = os.path.join(path_base, "translation_sequences/sequences")
        path_data_labels = os.path.join(path_base, "depth_sequences/sequences")
    elif dataset_type == "random":
        path_data_train = os.path.join(path_base, "translation_random_views/random_views")
        path_data_labels = os.path.join(path_base, "depth_random_views/random_views")
    elif dataset_type == "random_old":
        path_data_train = os.path.join(path_base, "stylernd")
        path_data_labels = os.path.join(path_base, "depths/simulated")
    else:
        raise ValueError(f"Unrecognised dataset type: {dataset_type}")


    train_data_loader, val_data_loader = pdr.get_dataloaders(input_dir=path_data_train,
                                                             depth_dir=path_data_labels,
                                                             dataset_type=dataset_type,
                                                             batch_size=batch_size,
                                                             preceding_frames=preceding_frames)

    net = u.UNet(num_channels=num_channels)
    net = net.to(device)
    MSE_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-05, betas=(0.5, 0.999))
    optimizer.zero_grad()
    all_error = np.zeros(0)
    all_val_error = np.zeros(0)

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

        if epoch % save_every == 0:
            torch.save(net.state_dict(), os.path.join(path_base, "state_dict_model_unet_{}.pt".format(str(epoch))))

        if wandb_un:
            wandb.log({"train_loss": np.mean(all_error), "val_loss": np.mean(all_val_error)})

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
                        type=str,
                        default="random")
    parser.add_argument("--batch_size",
                        "-bs",
                        help="Batch size",
                        required=False,
                        type=int,
                        default=32)
    parser.add_argument("--num_channels",
                        "-nc",
                        help="Number of input channels",
                        required=False,
                        type=int,
                        default=3)
    parser.add_argument("--save_every",
                        "-se",
                        help="Frequency to store model",
                        required=False,
                        type=int,
                        default=100)
    parser.add_argument("--preceding_frames",
                        "-pf",
                        help="Number of preceding frames for sequeence dataset",
                        required=False,
                        type=int,
                        default=3)

    args = parser.parse_args()
    train_func(args.path_data, args.device, args.wandb_un, dataset_type=args.dataset_type, 
               batch_size=args.batch_size, num_channels=args.num_channels, save_every=args.save_every,
               preceding_frames=args.preceding_frames)
