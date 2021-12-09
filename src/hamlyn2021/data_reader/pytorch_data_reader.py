from abc import abstractmethod
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import argparse
from tqdm import tqdm

from hamlyn2021.data_reader import read_input_image, read_depth_map
from hamlyn2021.visualisation import show_sample


def apply_mirroring(img, lbl):
    if np.random.uniform() > 0.5:
        # horizontal flip
        img = np.flip(img, axis=-1).copy()
        lbl = np.flip(lbl, axis=-1).copy()

    if np.random.uniform() > 0.5:
        # vertical flip
        img = np.flip(img, axis=-2).copy()
        lbl = np.flip(lbl, axis=-2).copy()

    return img, lbl


class CustomDatasetLoader(Dataset):
    def __init__(self, input_dir: str, depth_dir: str,
                 input_files: List[str], depth_files: List[str],
                 input_shape=(3, 256, 512), depth_shape=(1, 256, 512),
                 augments=True):
        self.input_dir = input_dir
        self.depth_dir = depth_dir
        self.input_files = input_files
        self.depth_files = depth_files
        self.input_shape = input_shape
        self.depth_shape = depth_shape
        self.augments    = augments

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

        img = np.moveaxis(img, -1, 0)
        lbl = lbl[None,]

        if self.augments is True:
            # randomly flip images and labels (horizontal and vertical)
            img, lbl = apply_mirroring(img, lbl)

        return img, lbl


class CustomDatasetLoaderSequence(CustomDatasetLoader):
    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a single pair of input image and target depth map"""
        input_files = self.input_files[idx]
        depth_fn = self.depth_files[idx]

        depth_path = os.path.join(self.depth_dir, depth_fn)
        lbl = read_depth_map(depth_path)
        lbl = lbl[None,]

        images = []
        for fn in input_files:
            input_path = os.path.join(self.input_dir, fn)
            img = read_input_image(input_path)
            img = np.moveaxis(img, -1, 0)
            images += [img]
        images = np.concatenate(images, axis=0)

        if self.augments is True:
            # randomly flip images and labels (horizontal and vertical)
            images, lbl = apply_mirroring(images, lbl)

        return images, lbl


def setup_dataloader(input_dir, depth_dir, folders, cases=None, batch_size=32, shuffle=True, dataset_type="random") -> DataLoader:
    """Setup DataLoader for specified folders and cases"""
    # colect filenames for input images and output depth maps
    if cases is None:
        if dataset_type == "random_old":
            cases = [f"{i:05d}" for i in range(2000)]
        elif dataset_type == "random":
            cases = [f"{i:04d}" for i in range(3000)]

    folders_and_prefix = "translation/translation" if dataset_type == "random" else "style_0[random_style]/img"
    input_files = [
        f"{folder}/{folders_and_prefix}{case}.png"
        for case in cases
        for folder in folders
    ]
    folders_and_prefix = "depth/depth" if dataset_type == "random" else "depths/depth"
    depth_files = [
        f"{folder}/{folders_and_prefix}{case}.exr"
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


def setup_sequence_dataloader(input_dir, depth_dir, folders, cases=None, preceding_frames=None, 
                              batch_size=32, shuffle=True, dataset_type="sequence") -> DataLoader:
    """Setup DataLoader for specified folders and cases"""
    # colect filenames for input images and output depth maps
    if preceding_frames is None:
        preceding_frames = 3
    if cases is None:
        cases = [f"{i:04d}" for i in range(preceding_frames, 100)]
    input_files = [
        [
            f"{folder}/translation/translation{int(case)-delta:04d}.png"
            for delta in range(preceding_frames+1)
        ]
        for case in cases
        for folder in folders
    ]
    depth_files = [
        f"{folder}/depth/depth{case}.exr"
        for case in cases
        for folder in folders
    ]

    # set up dataloader
    data_generator = CustomDatasetLoaderSequence(
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
    batch_size: int = 32,
    dataset_type: str = "random",
    preceding_frames: Optional[int] = None,
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
        if dataset_type == "sequence" or dataset_type == "random":
            train_folders = [
                f"scene_{i}"
                for i in (1, 2, 3, 4)
            ]
        elif dataset_type == "random_old":
            train_folders = [
                "3Dircadb1.1",
                "3Dircadb1.2",
                "3Dircadb1.8",
                "3Dircadb1.9",
                "3Dircadb1.10",
                "3Dircadb1.11",
            ]
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    if valid_folders is None:
        if dataset_type == "sequence" or dataset_type == "random":
            valid_folders = [
                f"scene_{i}"
                for i in (5, 6)
            ]
        elif dataset_type == "random_old":
            valid_folders = [
                "3Dircadb1.17",
                "3Dircadb1.18",
                "3Dircadb1.19",
            ]
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    kwargs = dict(
        input_dir=input_dir,
        depth_dir=depth_dir,
        batch_size=batch_size,
        dataset_type=dataset_type,
    )

    if dataset_type == "random" or dataset_type == "random_old":
        dataloader_constructor = setup_dataloader
    elif dataset_type == "sequence":
        dataloader_constructor = setup_sequence_dataloader
        kwargs['preceding_frames'] = preceding_frames,

    train_dataloader = dataloader_constructor(
        folders=train_folders,
        cases=train_cases,
        shuffle=True,
        **kwargs
    )
    valid_dataloader = dataloader_constructor(
        folders=valid_folders,
        cases=valid_cases,
        shuffle=False,
        **kwargs
    )

    return train_dataloader, valid_dataloader


def test_get_dataloaders():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Command line options')
    parser.add_argument("--path_data",
                        "-d",
                        help="Path to data for simulation",
                        required=True,
                        type=str)
    parser.add_argument("--dataset_type",
                        "-t",
                        help="Dataset to use, 'random' or 'sequence'",
                        required=False,
                        default="random")
    args = parser.parse_args()

    # example setup of PyTorch dataloader
    if args.dataset_type == "sequence":
        input_dir = os.path.join(args.path_data, "translation_sequences/sequences")
        depth_dir = os.path.join(args.path_data, "depth_sequences/sequences")
    elif args.dataset_type == "random":
        input_dir = os.path.join(args.path_data, "translation_random_views/random_views")
        depth_dir = os.path.join(args.path_data, "depth_random_views/random_views")
    elif args.dataset_type == "random_old":
        input_dir = os.path.join(args.path_data, "stylernd")
        depth_dir = os.path.join(args.path_data, "depths/simulated")
    else:
        raise ValueError(f"Unrecognised dataset type: {args.dataset_type}")

    train_dataloader, valid_dataloader = get_dataloaders(
        input_dir=input_dir,
        depth_dir=depth_dir,
        dataset_type=args.dataset_type,
        batch_size=2,
    )

    # for i, (images, labels) in enumerate(valid_dataloader):
    #     try:
    #         img, lbl = images[0].numpy(), labels[0].numpy()
    #         show_sample(img, lbl, dataset_type=args.dataset_type)
    #     except Exception as e:
    #         print(f"Error: {e}")

    #     if i > 3:
    #         break

    for images, labels in tqdm(train_dataloader):
        print(images.shape, labels.shape)
    for images, labels in tqdm(valid_dataloader):
        print(images.shape, labels.shape)


if __name__ == "__main__":
    test_get_dataloaders()
