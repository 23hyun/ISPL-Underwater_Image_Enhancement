import os
import glob
import random
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data

random.seed(1143)


def populate_train_list(orig_images_path: str, hazy_images_path: str):
    train_list = []

    image_list_haze = []
    for f in glob.glob(os.path.join(hazy_images_path, "*")):
        ext = os.path.splitext(f)[1].lower()
        if ext in [".jpg", ".jpeg", ".png"]:
            image_list_haze.append(f)

    image_list_haze = sorted(set(image_list_haze))

    if "\\" in hazy_images_path:
        parts = hazy_images_path.split("\\")
        folder_token = parts[1] if len(parts) > 1 else ""
    else:
        folder_token = os.path.basename(os.path.normpath(hazy_images_path))

    tmp_dict = {}
    for fullpath in image_list_haze:
        fname = os.path.basename(fullpath)

        if folder_token == "trainA":
            sp = fname.split("_")
            if len(sp) >= 2:
                key = sp[0] + "_" + sp[1]
            else:
                key = os.path.splitext(fname)[0]
        else:
            sp = fname.split(" ")
            if len(sp) >= 2:
                key = sp[0] + " " + sp[1]
            else:
                key = os.path.splitext(fname)[0]

        if key in tmp_dict:
            tmp_dict[key].append(fname)
        else:
            tmp_dict[key] = [fname]

    for key in tmp_dict.keys():
        for hazy_fname in tmp_dict[key]:
            train_list.append([
                os.path.join(orig_images_path, hazy_fname),
                os.path.join(hazy_images_path, hazy_fname)
            ])

    random.shuffle(train_list)
    val_list = []
    return train_list, val_list


class dehazing_loader(data.Dataset):
    def __init__(
        self,
        orig_images_path: str,
        hazy_images_path: str,
        mode: str = "train",
        load_size: int = 286,
        crop_size: int = 256,
        hflip_p: float = 0.5
    ):
        self.train_list, self.val_list = populate_train_list(orig_images_path, hazy_images_path)

        if mode == "train":
            self.data_list = self.train_list
            print("Total training examples:", len(self.train_list))
        # else:
        #     self.data_list = self.val_list
        #     print("Total validation examples:", len(self.val_list))

        self.mode = mode
        self.load_size = load_size
        self.crop_size = crop_size
        self.hflip_p = hflip_p

    def __getitem__(self, index):
        data_orig_path, data_hazy_path = self.data_list[index]

        data_orig = Image.open(data_orig_path).convert("RGB")
        data_hazy = Image.open(data_hazy_path).convert("RGB")

        # 1) Resize to load_size x load_size
        data_orig = data_orig.resize((self.load_size, self.load_size), Image.Resampling.LANCZOS)
        data_hazy = data_hazy.resize((self.load_size, self.load_size), Image.Resampling.LANCZOS)

        # 2) Random crop (train mode only)
        if self.mode == "train":
            x = random.randint(0, self.load_size - self.crop_size)
            y = random.randint(0, self.load_size - self.crop_size)

            data_orig = data_orig.crop((x, y, x + self.crop_size, y + self.crop_size))
            data_hazy = data_hazy.crop((x, y, x + self.crop_size, y + self.crop_size))

            # 3) Random horizontal flip
            if random.random() < self.hflip_p:
                data_orig = data_orig.transpose(Image.FLIP_LEFT_RIGHT)
                data_hazy = data_hazy.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            # test/val에서는 center crop
            x = (self.load_size - self.crop_size) // 2
            y = (self.load_size - self.crop_size) // 2

            data_orig = data_orig.crop((x, y, x + self.crop_size, y + self.crop_size))
            data_hazy = data_hazy.crop((x, y, x + self.crop_size, y + self.crop_size))

        # 4) To Tensor [0,1]
        data_orig = np.asarray(data_orig, dtype=np.float32) / 255.0
        data_hazy = np.asarray(data_hazy, dtype=np.float32) / 255.0

        data_orig = torch.from_numpy(data_orig).permute(2, 0, 1)
        data_hazy = torch.from_numpy(data_hazy).permute(2, 0, 1)

        return data_orig, data_hazy

    def __len__(self):
        return len(self.data_list)

