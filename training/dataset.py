import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import os
import re

from envpath.env import MANGA109_ROOT, DATA_SYNTHETIC_ROOT

def join_posix(root, rel):
    # split on / or \\, then join under root using POSIX separators
    parts = re.split(r"[\\/]+", str(rel))
    return str(root.joinpath(*parts))


class MangaDataset(Dataset):
    def __init__(
        self,
        processor,
        split,
        max_target_length,
        limit_size=None,
        augment=False,
        skip_packages=None,
    ):
        self.processor = processor
        self.max_target_length = max_target_length

        data = []

        print(f"Initializing dataset {split}...")

        if skip_packages is None:
            skip_packages = set()
        else:
            skip_packages = {f"{x:04d}" for x in skip_packages}

        for path in sorted((DATA_SYNTHETIC_ROOT / "meta").glob("*.csv")):
            if path.stem in skip_packages:
                print(f"Skipping package {path}")
                continue
            if not (DATA_SYNTHETIC_ROOT / "img" / path.stem).is_dir():
                print(f"Missing image data for package {path}, skipping")
                continue
            df = pd.read_csv(path)
            df = df.dropna()
            df["path"] = df.id.apply(lambda x: str(DATA_SYNTHETIC_ROOT / "img" / path.stem / f"{x}.jpg"))
            df = df[["path", "text"]]
            df["synthetic"] = True
            data.append(df)

        df = pd.read_csv(MANGA109_ROOT / "data.csv")
        df = df[df.split == split].reset_index(drop=True)
        df["path"] = df.crop_path.apply(lambda x: join_posix(MANGA109_ROOT, x))  # noqa: F821
        df = df[["path", "text"]]
        df["synthetic"] = False
        data.append(df)

        data = pd.concat(data, ignore_index=True)

        data["path"] = data["path"].apply(lambda p: p.replace("\\", "/"))
        missing = (~data["path"].map(os.path.exists)).sum()
        if missing:
            print(f"⚠️ Skipping {missing} rows with missing images (bad paths).")
        data = data[data["path"].map(os.path.exists)].reset_index(drop=True)

        if limit_size:
            data = data.iloc[:limit_size]
        self.data = data
        print(f"Dataset {split}: {len(self.data)}")

        self.augment = augment
        self.transform_medium, self.transform_heavy = self.get_transforms()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.loc[idx]
        text = sample.text

        if self.augment:
            medium_p = 0.8
            heavy_p = 0.02
            transform_variant = np.random.choice(
                ["none", "medium", "heavy"],
                p=[1 - medium_p - heavy_p, medium_p, heavy_p],
            )
            transform = {
                "none": None,
                "medium": self.transform_medium,
                "heavy": self.transform_heavy,
            }[transform_variant]
        else:
            transform = None

        pixel_values = self.read_image(self.processor, sample.path, transform)
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
        ).input_ids
        labels = np.array(labels)
        # important: make sure that PAD tokens are ignored by the loss function
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        encoding = {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels),
        }
        return encoding

    @staticmethod
    def read_image(processor, path, transform=None):
        # normalize path & read
        path = str(path).replace("\\", "/")
        img = cv2.imread(path, cv2.IMREAD_COLOR)  # always 3-ch BGR
        if img is None:
            raise FileNotFoundError(f"Image not found or unreadable: {path}")

        # Albumentations v2: no always_apply → use p=1.0 if you want grayscale
        transform = transform or A.ToGray(p=1.0)

        img = transform(image=img)["image"]

        # Ensure RGB 3-channel for HF processors
        if img.ndim == 2:  # single-channel
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pixel_values = processor(img, return_tensors="pt").pixel_values
        return pixel_values.squeeze()


    @staticmethod
    def get_transforms():
        t_medium = A.Compose(
            [
                A.Rotate(limit=5, border_mode=cv2.BORDER_REPLICATE, p=0.2),
                A.Perspective(scale=(0.01, 0.05), border_mode=cv2.BORDER_REPLICATE, keep_size=True, p=0.2),
                A.InvertImg(p=0.05),

                A.OneOf(
                    [
                        A.Downscale(
                            scale_range=(0.25, 0.5),
                            interpolation_pair={"downscale": cv2.INTER_LINEAR, "upscale": cv2.INTER_LINEAR},
                        ),
                        A.Downscale(
                            scale_range=(0.25, 0.5),
                            interpolation_pair={"downscale": cv2.INTER_NEAREST, "upscale": cv2.INTER_NEAREST},
                        ),
                    ],
                    p=0.1,
                ),

                A.Blur(blur_limit=(5, 9), p=0.2),
                A.Sharpen(p=0.2),

                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5,
                ),

                # noise between 1%–4% of pixel range
                A.GaussNoise(std_range=(0.01, 0.04), mean_range=(0.0, 0.0), p=0.3),

                # quality must be 1–100
                A.ImageCompression(quality_range=(10, 30), p=0.1),

                # always apply grayscale
                A.ToGray(p=1.0),
            ]
        )

        t_heavy = A.Compose(
            [
                A.Rotate(limit=10, border_mode=cv2.BORDER_REPLICATE, p=0.2),
                A.Perspective(scale=(0.01, 0.05), border_mode=cv2.BORDER_REPLICATE, keep_size=True, p=0.2),
                A.InvertImg(p=0.05),

                A.OneOf(
                    [
                        A.Downscale(
                            scale_range=(0.1, 0.2),
                            interpolation_pair={"downscale": cv2.INTER_LINEAR, "upscale": cv2.INTER_LINEAR},
                        ),
                        A.Downscale(
                            scale_range=(0.1, 0.2),
                            interpolation_pair={"downscale": cv2.INTER_NEAREST, "upscale": cv2.INTER_NEAREST},
                        ),
                    ],
                    p=0.1,
                ),

                A.Blur(blur_limit=(5, 9), p=0.5),
                A.Sharpen(p=0.5),

                A.RandomBrightnessContrast(
                    brightness_limit=0.8,
                    contrast_limit=0.8,
                    p=1.0,
                ),

                # stronger noise: ~5%–12% of pixel range
                A.GaussNoise(std_range=(0.05, 0.12), mean_range=(0.0, 0.0), p=0.3),

                A.ImageCompression(quality_range=(5, 15), p=0.5),
                A.ToGray(p=1.0),
            ]
        )

        return t_medium, t_heavy



if __name__ == "__main__":
    from .get_model import get_processor
    from .utils import tensor_to_image

    encoder_name = "facebook/deit-tiny-patch16-224"
    decoder_name = "cl-tohoku/bert-base-japanese-char-v2"

    max_length = 300

    processor = get_processor(encoder_name, decoder_name)
    ds = MangaDataset(processor, "train", max_length, augment=True)

    for i in range(20):
        sample = ds[0]
        img = tensor_to_image(sample["pixel_values"])
        tokens = sample["labels"]
        tokens[tokens == -100] = processor.tokenizer.pad_token_id
        text = "".join(processor.decode(tokens, skip_special_tokens=True).split())

        print(f"{i}:\n{text}\n")
        plt.imshow(img)
        plt.show()
