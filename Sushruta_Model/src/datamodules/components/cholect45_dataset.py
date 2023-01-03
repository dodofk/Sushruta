from typing import List, Dict
import os
import random

import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset

from PIL import Image
import torchvision.transforms.functional as tfunc
from torchvision import transforms
import albumentations as A

import hydra
from hydra.utils import get_original_cwd

VALIDATION_VIDEOS = ["78", "43", "62", "35", "74", "01", "56", "04", "13"]

class CholecT45Dataset(Dataset):
    def __init__(
        self,
        data_dir: str = "data",
        split: str = "train",
        seq_len: int = 4,
        channels: int = 3,
        np_random_seed: int = 12345,
        img_dir: str = "data/CholecT45/data/VID01",
        triplet_file: str = "data/CholecT45/triplet/VID01.txt",
        tool_file: str = "data/CholecT45/instrument/VID01.txt",
        verb_file: str = "data/CholecT45/verb/VID01.txt",
        target_file: str = "data/CholecT45/target/VID01.txt",
        triplet_class_arg: str = "data/triplet_class_arg.npy",
        use_train_aug: bool = True,
    ) -> None:

        assert split in ["train", "dev", "test"], "Invalid split"

        self.split = split
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.use_train_aug = use_train_aug

        self.channels = channels
        self.np_random_seed = np_random_seed
        self.triplet_labels = np.loadtxt(
            os.path.join(
                get_original_cwd(),
                triplet_file,
            ),
            dtype=np.int,
            delimiter=",",
        )
        self.tool_labels = np.loadtxt(
            os.path.join(
                get_original_cwd(),
                tool_file,
            ),
            dtype=np.int,
            delimiter=",",
        )
        self.verb_labels = np.loadtxt(
            os.path.join(
                get_original_cwd(),
                verb_file,
            ),
            dtype=np.int,
            delimiter=",",
        )
        self.target_labels = np.loadtxt(
            os.path.join(
                get_original_cwd(),
                target_file,
            ),
            dtype=np.int,
            delimiter=",",
        )
        self.img_dir = img_dir
        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.A_transform = A.Compose(
            [
                A.RGBShift(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.ChannelDropout(p=0.03),
            ],
        )

        self.augment_transform = transforms.Compose(
            [
                transforms.TrivialAugmentWide(),
                # transforms.AutoAugment(),
            ]
        )

        with open(os.path.join(get_original_cwd(), triplet_class_arg), "rb") as file:
            self.triplet_sort_ind = np.load(file)[0:53]

    # def sample_label(self, df: pd.DataFrame, base_on: str, sample_num):
    #     return df.groupby(base_on).sample(
    #         n=sample_num,
    #         random_state=12345,
    #     ).reset_index()

    def __getitem__(self, index) -> Dict:
        image_id = self.triplet_labels[index, 0]
        if image_id >= self.seq_len:
            numbers = list(range(image_id + 1 - self.seq_len, image_id + 1))
        else:
            numbers = list(range(1, image_id + 1))
            while len(numbers) < self.seq_len:
                numbers.append(image_id)

        frames = torch.FloatTensor(self.seq_len, self.channels, 224, 224)

        angle = random.choice([-30, -20, -15, -5, 0, 5, 15, 20, 30])
        angle_aug = random.randint(-5, 5)

        for i, _image_id in enumerate(numbers):
            basename = "{}.png".format(str(_image_id).zfill(6))
            img_path = os.path.join(get_original_cwd(), self.img_dir, basename)

            if self.split == "train":
                image = np.array(Image.open(img_path))

                image = self.A_transform(image=image)["image"]

                image = Image.fromarray(image)
            else:
                image = Image.open(img_path)

            image = self.transform(image)

            if self.split == "train" and self.use_train_aug:
                image = tfunc.rotate(image, angle)
                angle += angle_aug
                # _index = np.where(np.array(self.triplet_labels[index, 1:] == 1))[0]
                #
                # for ind in _index:
                #     if ind in self.triplet_sort_ind:
                #         image = self.augment_transform(image.to(torch.uint8))

            frames[i, :, :, :] = image.to(torch.float)

        triplet_label = self.triplet_labels[index, 1:]
        tool_label = self.tool_labels[index, 1:]
        verb_label = self.verb_labels[index, 1:]
        target_label = self.target_labels[index, 1:]
        return {
            "frame": image_id,
            "image": torch.squeeze(frames, dim=0),
            "verb": verb_label,
            "tool": tool_label,
            "target": target_label,
            "triplet": triplet_label,
        }

    def __len__(self):
        return len(self.triplet_labels)


def default_collate_fn(
    inputs: List,
) -> Dict:
    frame = [data["frame"] for data in inputs]
    image = torch.stack([data["image"] for data in inputs])
    verb = torch.Tensor(np.array([data["verb"] for data in inputs])).to(torch.float)
    tool = torch.Tensor(np.array([data["tool"] for data in inputs])).to(torch.float)
    target = torch.Tensor(np.array([data["target"] for data in inputs])).to(torch.float)
    triplet = torch.Tensor(np.array([data["triplet"] for data in inputs])).to(
        torch.float
    )
    return {
        "frame": frame,
        "image": image,
        "verb": verb,
        "tool": tool,
        "target": target,
        "triplet": triplet,
    }


def build_dataloader(
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    split: str,
    data_dir: str,
    seq_len: int,
    channels: int,
    use_train_aug: bool,
    triplet_class_arg: str,
) -> DataLoader:
    assert split in ["train", "dev", "test"], "Invalid Split"
    iterable_dataset = []
    video_split = {
        "train": [
            79,
            2,
            51,
            6,
            25,
            14,
            66,
            23,
            50,
            80,
            32,
            5,
            15,
            40,
            47,
            26,
            48,
            70,
            31,
            57,
            36,
            18,
            52,
            68,
            10,
            8,
            73,
            42,
            29,
            60,
            27,
            65,
            75,
            22,
            49,
            12,
        ],
        "dev": [78, 43, 62, 35, 74, 1, 56, 4, 13],
        "test": [78, 43, 62, 35, 74, 1, 56, 4, 13],
    }
    train_videos = video_split[split]
    train_records = ["VID{}".format(str(v).zfill(2)) for v in train_videos]

    for video in train_records:
        dataset = CholecT45Dataset(
            img_dir=os.path.join(data_dir, "data", video),
            triplet_file=os.path.join(data_dir, "triplet", "{}.txt".format(video)),
            tool_file=os.path.join(data_dir, "instrument", "{}.txt".format(video)),
            verb_file=os.path.join(data_dir, "verb", "{}.txt".format(video)),
            target_file=os.path.join(data_dir, "target", "{}.txt".format(video)),
            split=split,
            data_dir=data_dir,
            seq_len=seq_len,
            channels=channels,
            use_train_aug=use_train_aug,
            triplet_class_arg=triplet_class_arg,
        )
        iterable_dataset.append(dataset)
    return DataLoader(
        dataset=ConcatDataset(iterable_dataset),
        batch_size=batch_size,
        collate_fn=default_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True if split == "train" else False,
    )


@hydra.main(config_path=None)
def test_dataset(cfg) -> None:
    triplet_counter = {}


    for num in VALIDATION_VIDEOS:
        print(f'processing video {num}...'.ljust(50), end='\r')
        dataset = CholecT45Dataset(
            img_dir = f"data/CholecT45/data/VID{num.zfill(2)}",
            triplet_file = f"data/CholecT45/triplet/VID{num.zfill(2)}.txt",
            tool_file = f"data/CholecT45/instrument/VID{num.zfill(2)}.txt",
            verb_file = f"data/CholecT45/verb/VID{num.zfill(2)}.txt",
            target_file = f"data/CholecT45/target/VID{num.zfill(2)}.txt",
        )
        for i in range(len(dataset)):
            arr = dataset.__getitem__(i)['triplet']
            labels = []
            for j in range(len(arr)):
                if arr[j]:
                    labels.append(j)

            for label in labels:
                if not str(label) in triplet_counter.keys():
                    triplet_counter[str(label)] = 0
                triplet_counter[str(label)] += 1

    for k, v in sorted(triplet_counter.items(), key=lambda x: x[1]):
        print(f'triplet {k.zfill(2)}:  {str(v).zfill(4)}')


if __name__ == "__main__":
    test_dataset()
