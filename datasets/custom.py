import os
from PIL import Image
import numpy as np
import torch.utils.data as data


class MyCustomDataset(data.Dataset):
    def __init__(self, root, split='train', transform=None):
        self.images_dir = os.path.join(root, split, 'images')
        self.masks_dir = os.path.join(root, split, 'masks')
        self.transform = transform

        self.images = sorted([os.path.join(self.images_dir, f)
                              for f in os.listdir(self.images_dir)
                              if f.endswith('.png') or f.endswith('.jpg')])
        self.masks = sorted([os.path.join(self.masks_dir, f)
                             for f in os.listdir(self.masks_dir)
                             if f.endswith('.png')])

        assert len(self.images) == len(self.masks), "Image and mask counts do not match!"

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.masks[index])

        if self.transform:
            image, mask = self.transform(image, mask)

        mask = np.array(mask).astype(np.int64)  # (H, W), label values as class index
        return image, mask

    def __len__(self):
        return len(self.images)

    @staticmethod
    def decode_target(mask):
        mask = np.asarray(mask)
        mask = np.clip(mask, 0, 1)  # 예외 방지
        colormap = np.array([[0, 0, 0], [255, 255, 255]])
        return colormap[mask]