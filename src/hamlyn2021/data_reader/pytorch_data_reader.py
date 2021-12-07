from abc import abstractmethod
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional

from hamlyn2021.data_reader import read_input_image, read_depth_map


class CustomDatasetLoader(Dataset):
    def __init__(self, input_dir: str, depth_dir: str,
                 input_files: List[str], depth_files: List[str],
                 input_shape=(256, 512, 3), depth_shape=(256, 512)):
        self.input_dir = input_dir
        self.depth_dir = depth_dir
        self.input_files = input_files
        self.depth_files = depth_files
        self.input_shape = input_shape
        self.depth_shape = depth_shape

    def __len__(self) -> int:
        return len(self.input_files)

    @abstractmethod
    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a single pair of input images and target depth map"""
        raise NotImplementedError()


class CustomDatasetLoaderRandom(CustomDatasetLoader):
    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a single pair of input image and target depth map"""
        input_fn = self.input_files[idx]
        depth_fn = self.depth_files[idx]

        if "[random_style]" in input_fn:
            # randomly choose one of the five styles
            num = np.random.randint(0, 4+1)
            input_fn = input_fn.replace("[random_style]", str(num))

        input_path = os.path.join(self.input_dir, input_fn)
        depth_path = os.path.join(self.depth_dir, depth_fn)

        img = read_input_image(input_path)
        lbl = read_depth_map(depth_path)

        return img, lbl


def setup_dataloader(input_dir, depth_dir, folders, cases=None, batch_size=32, shuffle=True) -> DataLoader:
    """Setup DataLoader for specified folders and cases"""
    # colect filenames for input images and output depth maps
    if cases is None:
        cases = [f"{i:04d}" for i in range(2000)]
    input_files = [
        f"{folder}/style_0[random_style]/img{case}.png"
        for case in cases
        for folder in folders
    ]
    depth_files = [
        f"{folder}/depths/depth{case}.exr"
        for case in cases
        for folder in folders
    ]

    # set up dataloader
    data_generator = CustomDatasetLoaderRandom(
        input_dir=input_dir,
        depth_dir=depth_dir,
        input_files=input_files,
        depth_files=depth_files
    )
    dataloader = DataLoader(
        data_generator,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=min(batch_size, 8)
    )

    return dataloader


def get_dataloaders(
    input_dir: str,
    depth_dir: str,
    train_folders: Optional[List[str]] = None,
    valid_folders: Optional[List[str]] = None,
    train_cases: Optional[List[str]] = None,
    valid_cases: Optional[List[str]] = None,
    batch_size: int = 32
) -> DataLoader:
    """
    Setup DataLoader for training and validation

    Inputs:
    :param input_dir: path to directory containing (tranformed) input images, e.g. /path/to/data/stylernd
    :param depth_dir: path to directory containing depth maps, e.g. /path/to/data/depth_random
    :param train_folders: list of folders to include for training, default: ["3Dircadb1.1", "3Dircadb1.2",
                          "3Dircadb1.8", "3Dircadb1.9", "3Dircadb1.10", "3Dircadb1.11"]
    :param train_folders: list of folders to include for validation, default: ["3Dircadb1.17", "3Dircadb1.18",
                          "3Dircadb1.19", "3Dircadb1.20"]
    :param train_cases: List of case names to include for training, default: ['00000', ..., '01999']
    :param train_cases: List of case names to include for validation, default: ['00000', ..., '01999']
    :param batch_size: number of samples per batch, default: 32

    Returns:
    - PyTorch dataloader with training samples
    - PyTorch dataloader with validation samples
    """
    # colect filenames for input images and output depth maps
    if train_folders is None:
        train_folders = [
            "3Dircadb1.1",
            "3Dircadb1.2",
            "3Dircadb1.8",
            "3Dircadb1.9",
            "3Dircadb1.10",
            "3Dircadb1.11",
        ]
    if valid_folders is None:
        valid_folders = [
            "3Dircadb1.17",
            "3Dircadb1.18",
            "3Dircadb1.19",
            "3Dircadb1.20",
        ]

    train_dataloader = setup_dataloader(
        input_dir=input_dir,
        depth_dir=depth_dir,
        folders=train_folders,
        cases=train_cases,
        batch_size=batch_size,
        shuffle=True
    )
    valid_dataloader = setup_dataloader(
        input_dir=input_dir,
        depth_dir=depth_dir,
        folders=valid_folders,
        cases=valid_cases,
        batch_size=batch_size,
        shuffle=False
    )

    return train_dataloader, valid_dataloader


def test_get_dataloaders():
    data_dir = "/Users/joeranbosma/Hamlyn2021/data/"

    # example setup of PyTorch dataloader for random data
    batch_size = 2
    input_dir = os.path.join(data_dir, "stylernd/")
    depth_dir = os.path.join(data_dir, "depth_random/")
    cases = ['00000']  # test using first case of each folder

    train_dataloader, valid_dataloader = get_dataloaders(
        input_dir=input_dir,
        depth_dir=depth_dir,
        train_folders=['3Dircadb1.1'],
        valid_folders=['3Dircadb1.1'],
        train_cases=cases,
        valid_cases=cases,
        batch_size=batch_size,
    )

    for images, labels in valid_dataloader:
        # visualise first sample of the batch
        img, lbl = images[0].numpy(), labels[0].numpy()
        f, axes = plt.subplots(1, 2, figsize=(18, 8))
        ax = axes[0]
        ax.imshow(img)
        ax = axes[1]
        ax.imshow(lbl)
        plt.show()


if __name__ == "__main__":
    test_get_dataloaders()
