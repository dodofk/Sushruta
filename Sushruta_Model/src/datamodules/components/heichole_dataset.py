from typing import List, Dict, Tuple
import os

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset

from PIL import Image
from torchvision import transforms

import hydra
from hydra.utils import get_original_cwd

from numpy.random import default_rng


class HeiCholeDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "data/HeiChole_data",
        split: str = "train",
        seq_len: int = 8,
        channels: int = 3,
        np_random_seed: int = 12345,
        sample_base_on: str = None,
        sample_num: int = 300,
    ) -> None:

        assert split in ["train", "dev"], "Invalid split"

        self.data_dir = data_dir
        self.seq_len = seq_len
        self.channels = channels
        self.np_random_seed = np_random_seed

        df = pd.read_csv(
                os.path.join(
                    get_original_cwd(),
                    data_dir,
                    f"{split}.csv",
                ),
            )
        if sample_base_on is None:
            self.df = df
        else:
            self.df = self.sample_label(
                df,
                sample_base_on,
                sample_num,
            )

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def sample_label(self, df: pd.DataFrame, base_on: str, sample_num):
        return df.groupby(base_on).sample(
            n=sample_num,
            random_state=12345,
        ).reset_index()

    def __getitem__(self, index) -> Dict:
        df_row = self.df.iloc[index]
        phase = df_row["phase"]
        video_id = df_row["video_id"]
        image_id = df_row["image_id"]

        rng = default_rng(self.np_random_seed)

        if image_id < self.seq_len:
            numbers = rng.choice(list(range(1, image_id+1)), size=self.seq_len-1, replace=True)
            numbers.sort()
        else:
            numbers = rng.choice(list(range(1, image_id)), size=self.seq_len-1, replace=False)
            numbers.sort()

        numbers = np.append(numbers, image_id)
        frames = torch.FloatTensor(self.seq_len, self.channels, 224, 224)

        for i, _image_id in enumerate(numbers):
            image = Image.open(
                os.path.join(
                    get_original_cwd(),
                    self.data_dir,
                    f"HeiChole_{video_id}",
                    f"{int(_image_id)}.jpg",
                )
            )
            image = self.transform(image)
            frames[i, :, :, :] = image.to(torch.float)

        return {
            "image": torch.squeeze(frames, dim=0),
            "phase": phase,
        }

    def __len__(self):
        return len(self.df)


def heichole_collate_fn(
    inputs: List,
) -> Dict:
    image = torch.stack([data["image"] for data in inputs])
    phase = torch.Tensor([data["phase"] for data in inputs]).to(torch.long)
    return {
        "image": image,
        "phase": phase,
    }


def build_heichole_dataloader(
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    split: str,
    data_dir: str,
    seq_len: int,
    channels: int,
    sample_base_on: str = None,
    sample_num: int = None,
) -> DataLoader:
    assert split in ["train", "dev"], "Invalid Split"

    dataset = HeiCholeDataset(
        split=split,
        data_dir=data_dir,
        seq_len=seq_len,
        channels=channels,
        sample_base_on=sample_base_on,
        sample_num=sample_num,
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=heichole_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True if split == "train" else False,
    )


@hydra.main(config_path=None)
def test_dataset(cfg) -> None:
    dataset = HeiCholeDataset(
        data_dir="../../../../slue-toolkit/data/slue-voxpopuli/",
        split="fine-tune",
    )
    print(dataset.__getitem__(1))


if __name__ == "__main__":
    test_dataset()