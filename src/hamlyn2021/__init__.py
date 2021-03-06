import os
from tqdm import tqdm
import argparse

from hamlyn2021.data_reader import get_dataloaders


def test(data_dir):
    # example setup of PyTorch dataloader for random data
    batch_size = 32
    input_dir = os.path.join(data_dir, "stylernd/")
    depth_dir = os.path.join(data_dir, "simulated/")

    train_dataloader, valid_dataloader = get_dataloaders(
        input_dir=input_dir,
        depth_dir=depth_dir,
        batch_size=batch_size,
    )

    train_count, valid_count = 0, 0
    for images, labels in tqdm(train_dataloader):
        train_count += 1

    for images, labels in tqdm(valid_dataloader):
        valid_count += 1

    print(f"Have {train_count} training batches and {valid_count} valid batches")


if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Command line options')
    parser.add_argument("--data", type=str, required=True, help="Path to data folder")
    args = parser.parse_args()

    test(data_dir=args.data)
