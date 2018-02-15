from __future__ import print_function, division
from torch.utils.data import Dataset
from skimage import io

import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

if ("SSH_CONNECTION" in os.environ) or ('SSH_TTY' in os.environ):
    # dont display plot if on remote server
    matplotlib.use('Agg')


class MassRoadBuildingDataset(Dataset):
    """Massachusetts Road and Building dataset"""

    def __init__(self, csv_file, root_dir='mass_roads', train_valid_test='train', transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with image paths
            train_valid_test (string): 'train', 'valid', or 'test'
            root_dir (string): 'mass_roads' or 'mass_buildings'
            transform (callable, optional): Optional transform to be applied on a sample.
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

        sample = {'sat_img': sat_image, 'map_img': map_image}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _filter_df(self, csv_file):
        df = pd.read_csv(csv_file)

        return df[(df['train_valid_test'] == self.train_valid_test) &
                  (df['sat_map'] != 'missing')].reset_index(drop=True)


# helper function for viewing images
def show_map(sat_img, map_img=None, axis=None):
    """
    Return an image with the shape mask if there is one supplied
    """

    if axis:
        axis.imshow(sat_img)

        if map_img is not None:
            axis.imshow(map_img, alpha=0.5, cmap='gray')

    else:
        plt.imshow(sat_img)

        if map_img is not None:
            plt.imshow(map_img, alpha=0.5, cmap='gray')


# helper function to show a batch
def show_map_batch(sample_batched, img_to_show=3, save_file_path=None, as_numpy=False):
    """
    Show image with map image overlayed for a batch of samples.
    """

    # just select 6 images to show per batch
    sat_img_batch, map_img_batch = sample_batched['sat_img'][:img_to_show, :, :, :],\
                                   sample_batched['map_img'][:img_to_show, :, :, :]
    batch_size = len(sat_img_batch)

    f, ax = plt.subplots(int(np.ceil(batch_size / 3)), 3, figsize=(15, int(np.ceil(batch_size / 3)) * 5))
    f.tight_layout()
    f.subplots_adjust(hspace=.05, wspace=.05)
    ax = ax.ravel()

    for i in range(batch_size):
        print(type(ax[i]))
        ax[i].axis('off')
        show_map(sat_img=sat_img_batch.numpy()[i, :, :, :].transpose((1, 2, 0)),
                 map_img=map_img_batch.numpy()[i, 0, :, :], axis=ax[i])

    if save_file_path is not None:
        f.savefig(save_file_path)

    if as_numpy:
        f.canvas.draw()
        width, height = f.get_size_inches() * f.get_dpi()
        mplimage = np.frombuffer(f.canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

        return mplimage