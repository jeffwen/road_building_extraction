from __future__ import print_function, division
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform
from skimage.color import gray2rgb

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ignore warnings
import warnings

warnings.filterwarnings("ignore")

# interactive mode
plt.ion()

local_path = '/Users/jwen/Projects/road_building_extraction/'


class MassRoadBuildingDataset(Dataset):
    """Massachusetts Road and Building dataset"""

    def __init__(self, csv_file, root_dir='mass_roads', train_valid_test='train', transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with image paths
            train_valid_test (string): 'train', 'valid', or 'test'
            root_dir (string): 'mass_roads' or 'mass_buildings'
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.train_valid_test = train_valid_test
        self.root_dir = root_dir
        self.img_path_df = self._filter_df(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.img_path_df)

    def __getitem__(self, idx):
        sat_img_name = os.path.join('data', self.root_dir, self.train_valid_test, 'sat',
                                    self.img_path_df.loc[idx, 'sat_img_path'])
        sat_image = io.imread(sat_img_name)

        map_img_name = os.path.join('data', self.root_dir, self.train_valid_test, 'map',
                                    self.img_path_df.loc[idx, 'map_img_path'])
        map_image = io.imread(map_img_name)

        # hacky way to make transforms easier later on (with PyTorch)
        map_image = gray2rgb(map_image)

        sample = {'sat_img': sat_image, 'map_img': map_image}
        # sample = {'sat_img': sat_image, 'map_img': map_image, 'sat_img_path': self.img_path_df.loc[idx, 'sat_img_path']}

        if self.transform:
            sample = {'sat_img': self.transform(sample['sat_img']), 'map_img': self.transform(sample['map_img'])}

        return sample

    def _filter_df(self, csv_file):
        df = pd.read_csv(csv_file)

        return df[(df['train_valid_test'] == self.train_valid_test) &
                  (df['sat_map'] != 'missing')].reset_index(drop=True)


# helper function for viewing images
def show_map(sat_img, map_img=None, sat_img_path=None):
    """
    Return an image with the shape mask if there is one supplied
    """

    plt.imshow(sat_img)

    if map_img is not None:
        plt.imshow(map_img, alpha=0.5, cmap='gray')


# helper function to show a batch
def show_map_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    sat_img_batch, map_img_batch = \
            sample_batched['sat_img'], sample_batched['map_img']
    batch_size = len(sat_img_batch)
    img_size = sat_img_batch.size(2)

    grid = utils.make_grid(sat_img_batch, nrow=2)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * img_size,
                    landmarks_batch[i, :, 1].numpy(),
                    s=10, marker='.', c='r')
        plt.imshow(map_img[i, :, 0].numpy() + i * img_size, alpha=0.5, cmap='gray')

        plt.title('Batch from dataloader')


data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(768),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


mass_dataset = MassRoadBuildingDataset('/Users/jwen/Projects/road_building_extraction/data/mass_roads/mass_roads.csv','mass_roads','train')


for i in range(len(mass_dataset)):
    sample = mass_dataset[i]

    print(i, sample['sat_img'].shape, sample['map_img'].shape)

    plt.figure(1, figsize=(9,9))
    ax = plt.subplot(2, 2, i + 1)
    plt.tight_layout()
    # ax.set_title('{}'.format(sample['sat_img_path']))
    ax.axis('off')
    show_map(**sample)

    if i == 3:
        plt.show()
        break


dataloader = DataLoader(mass_dataset, batch_size=4, num_workers=4)

for i_batch, sample_batched in enumerate(dataloader):
    print("{},{}".format(i_batch, len(sample_batched)))