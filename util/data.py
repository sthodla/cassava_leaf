import os
import numpy as np
from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class Cassava(Dataset):
    def __init__(self, df, data_aug, train=True, image_path='data/train_images'):
        super().__init__()
        self.image_ids = df.image_ids.values
        self.labels = df.label.values
        self.data_aug = data_aug
        self.train = train
        self.image_path = image_path


    def __len__(self):
        return len(self.image_ids)

    
    def __getitem__(self, i):
        img_id = self.image_ids[i]
        label = self.labels[i]

        # Look up if albumentation takes an array or PIL.Image
        path = os.path.join(self.image_path, img_id)
        img = np.array(Image.open(path))
        if self.train: # Data augmentation
            pass
        
        return {'img': img, 'label': label}


def make_dataloader(ds, bs, nw=4, shuffle=False):
    dl = DataLoader(
        dataset=ds,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=nw
    )
    return dl
   