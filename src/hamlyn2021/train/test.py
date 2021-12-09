# coding=utf-8

"""
Module to test network with unseen predictions.
"""
import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import hamlyn2021.unet.Unet_v2 as u
import hamlyn2021.data_reader.pytorch_data_reader as pdr

def test_func(path_base, path_model, device=None, dataset_type="random",
               batch_size=32,):
    """
    Function to test the network with unseen data.
    """

    if dataset_type == "sequence":
        path_data_train = os.path.join(path_base, "translation_sequences/sequences")
        path_data_labels = os.path.join(path_base, "depth_sequences/sequences")
    elif dataset_type == "random":
        path_data_train = os.path.join(path_base, "translation_random_views/random_views")
        path_data_labels = os.path.join(path_base, "depth_random_views/random_views")
    
    _, val_data_loader = pdr.get_dataloaders(input_dir=path_data_train,
                                             depth_dir=path_data_labels,
                                             dataset_type=dataset_type,
                                             batch_size=batch_size)
    if dataset_type == "sequence":
        channels = 12
    else:
        channels = 3
    
    net = u.UNet(channels)
    net.load_state_dict(torch.load(path_model))
    net.to(device)

    # Test
    for i, (data, label) in enumerate(val_data_loader):
        # setting your network to train will ensure that parameters will be updated during training, 
        # and that dropout will be used
        net.eval()
        print(data.shape)
        print(label.shape)
        pred = net(data.to(device).float())
        # Plot
        fig, axes = plt.subplots(1, 2) 
        axes[0].imshow(np.transpose(data[0, 0:3, :, :].cpu().numpy(), (1, 2, 0)))
        axes[1].imshow(label[0, 0, :, :].cpu().numpy())
        plt.savefig(os.path.join(path_base,"Test_depths_{}.png".format(i)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HamlynUNetTest')

    ## ARGS
    parser.add_argument("--path_data",
                        "-d",
                        help="Path to data for simulation",
                        required=True,
                        type=str,
                        default=None)
    parser.add_argument("--path_model",
                        "-m",
                        help="Path to trained model state dict",
                        required=True,
                        type=str,
                        default=None)
    parser.add_argument("--device",
                        "-c",
                        help="Device to use, defaults to cuda:0",
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

    args = parser.parse_args()
    test_func(args.path_data, args.path_model, args.device, dataset_type=args.dataset_type, 
               batch_size=args.batch_size)
