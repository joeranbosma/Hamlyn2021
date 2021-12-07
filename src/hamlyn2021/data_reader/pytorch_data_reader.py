import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import Tuple

from hamlyn2021.data_reader.data_reader import read_sequence


def random_crop(img: np.ndarray, output_shape):
    for i, size in enumerate(output_shape[::-1]):
        assert img.shape[-1-i] >= size, f"Mismatch in size: {img.shape[-1-i]} < {size}"
    start_idx = [
        np.random.randint(0, in_size - out_size)
        for in_size, out_size in zip(output_shape[::-1], list(img.shape)[::-1])
    ]
    print(start_idx)
    # img = img[y:y+height, x:x+width]
    # return img


class CustomDatasetLoader(Dataset):
    def __init__(self, input_dir, depth_dir, sequences, input_shape=(256, 512, 3), depth_shape=(256, 512)):
        self.input_dir = input_dir
        self.depth_dir = depth_dir
        self.sequences = sequences
        self.input_shape = input_shape
        self.depth_shape = depth_shape

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a single pair of input images and target depth map"""
        scene_id = self.sequences[idx]
        input_dir = os.path.join(self.input_dir, scene_id, "translation")
        depth_dir = os.path.join(self.depth_dir, scene_id, "depth")

        images, labels = read_sequence(
            input_dir=input_dir,
            depth_dir=depth_dir,
            input_shape=self.input_shape,
            depth_shape=self.depth_shape
        )
        
        return images, labels


if __name__ == "__main__":
    batch_size = 2
    input_dir = "/Users/joeranbosma/Hamlyn2021/data/translation_sequences/sequences/"
    depth_dir = "/Users/joeranbosma/Hamlyn2021/data/depth_sequences/sequences/"
    train_sequences = [
        "scene_1",
        "scene_2",
        "scene_3",
    ]
    valid_sequences = [
        "scene_4",
        "scene_5",
        "scene_6",
    ]
    train_data = CustomDatasetLoader(input_dir=input_dir, depth_dir=depth_dir, sequences=train_sequences)
    valid_data = CustomDatasetLoader(input_dir=input_dir, depth_dir=depth_dir, sequences=valid_sequences)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=min(batch_size, 8))
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=min(batch_size, 8))

    for images, labels in valid_dataloader:
        # visualise first sample of the batch
        images, labels = images[0].numpy(), labels[0].numpy()
        for img, lbl in zip(images, labels):
            f, axes = plt.subplots(1, 2, figsize=(18, 8))
            ax = axes[0]
            ax.imshow(img)
            ax = axes[1]
            ax.imshow(lbl)
            plt.show()
            break  # prevent showing a popup for all timesteps
