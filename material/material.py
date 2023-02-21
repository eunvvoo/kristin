import os
import torch
from typing import Tuple, Sequence, Callable
from torch import nn, Tensor
from torch.utils.data import Dataset
import json
from sklearn.preprocessing import LabelEncoder
import numpy as np
from PIL import Image

def to_onehot(labels, n_categories, dtype=torch.float32):
    batch_size = len(labels)
    one_hot_labels = torch.zeros(size=(batch_size, n_categories), dtype=dtype)
    for i, label in enumerate(labels):
        label = torch.LongTensor(label)
        one_hot_labels[i] = one_hot_labels[i].scatter_(dim=0, index=label, value=1.)
    return one_hot_labels

class MyDataset(Dataset):
    def __init__(self, dir: os.PathLike, image_ids: os.PathLike, transforms: Sequence[Callable]) -> None:
        self.dir = dir
        self.transforms = transforms
        self.labels = {}
        
        material_dict = {
        'leather': 0,
        'mesh/knit': 1,
        'nylon': 2,
        'suede': 3
        }

        with open(image_ids, 'r') as f:
            reader = json.load(f)
            le = LabelEncoder()
            for row in reader:
                a = []
                try:
                    for i in row['material']:
                        a.append(material_dict[i])
                    # one hot encoding 후 list 변환
                    a = np.concatenate(to_onehot([a], n_categories=4, dtype=torch.int64).tolist()).tolist()
                    Image.open(os.path.join(self.dir, f'{row["img_name"]}.jpg'))
                    self.labels[int(row["id"])] = [row["img_name"], a]
                except (KeyError, FileNotFoundError):
                    pass
        
        self.image_ids = list(self.labels.keys())

    def __len__(self) -> str:
        return len(self.image_ids)
    
    def __getitem__(self, index: int) -> Tuple[Tensor]:
        image_id = self.image_ids[index]
        image = Image.open(os.path.join(self.dir, f'{str(self.labels.get(image_id)[0])}.jpg')).convert('RGB')
        target = np.array(self.labels.get(image_id)[1]).astype(np.float32)

        if self.transforms is not None:
            image = self.transforms(image)

        return image, target