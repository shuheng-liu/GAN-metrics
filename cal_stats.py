import argparse
import os
import numpy as np
import pickle as pkl
from PIL import Image
from stats import MeanStdStats
from utils import image2array, get_mean_std_stats, Trio

img_suffices = ["jpg", "jpeg", "png"]
img_suffices = ["." + suffix for suffix in img_suffices]
img_suffices.extend([suffix.upper() for suffix in img_suffices])
print("listing legal suffices", img_suffices)


class BatchHandler:
    def __init__(self, folders: (Trio, tuple), path_check=True):
        self.folders = Trio()
        if isinstance(folders, tuple):
            folders = Trio(*folders)
        self.set_folders(folders, path_check=path_check)

        self.pools = Trio(list(), list(), list())
        self.data = Trio(MeanStdStats(), MeanStdStats(), MeanStdStats())

    def check_path(self):
        real = self.folders.get_real()
        fake0 = self.folders.get_fake0()
        fake1 = self.folders.get_fake1()
        assert os.path.exists(real), "real folder does not exists {}".format(real)
        assert os.path.exists(fake0), "fake0 folder does not exists {}".format(fake0)
        assert os.path.exists(fake1), "fake1 folder does not exists {}".format(fake1)
        assert not os.path.isfile(real), "real is not a folder {}".format(real)
        assert not os.path.isfile(fake0), "fake0 is not a folder {}".format(fake0)
        assert not os.path.isfile(fake1), "fake1 is not a folder {}".format(fake1)
        print("path check passed")

    def set_folders(self, folders: Trio, path_check=True):
        assert isinstance(folders, Trio), "`folder` is not Trio: {}".format(folders)
        self.folders = folders  # type: Trio

        if path_check:
            try:
                self.check_path()
            except AssertionError as e:
                print("error in folders passed")
                print(e)

    def update_pools(self):
        real_pool = self.get_pool(self.folders.get_real())
        self.pools.set_real(real_pool)

        fake0_pool = self.get_pool(self.folders.get_fake0())
        self.pools.set_fake0(fake0_pool)

        fake1_pool = self.get_pool(self.folders.get_fake1())
        self.pools.set_fake1(fake1_pool)

    def update_data(self):
        self.data.set_real(get_mean_std_stats(self.pools.get_real()))
        self.data.set_fake0(get_mean_std_stats(self.pools.get_fake0()))
        self.data.set_fake1(get_mean_std_stats(self.pools.get_fake1()))
        self.data.update_absolute_delta()

    @staticmethod
    def get_pool(folder):
        """
        :param folder: path to a folder (as a str) that contains images
        :return: all images inside a folder as a 4-D array with shape (N, W, H, C)
        """
        pool = list()
        for name in os.listdir(folder):
            for suffix in img_suffices:
                if name.endswith(suffix):
                    dir = os.path.join(folder, name)
                    img = Image.open(dir)
                    pool.append(image2array(img))
                    break

        return np.stack(pool, axis=0)


if __name__ == '__main__':
    default_real = "images/real-images"
    default_fake0 = "images/fake-images-97"
    default_fake1 = "images/fake-images-120"

    parser = argparse.ArgumentParser()
    parser.add_argument("--real", default=default_real, help="folder containing real images")
    parser.add_argument("--fake0", default=default_fake0, help="folder containing fake images without Mute layers")
    parser.add_argument("--fake1", default=default_fake1, help="folder containing fake images with Mute layers")

    opt = parser.parse_args()
    real_folder = opt.real
    fake0_folder = opt.fake0
    fake1_folder = opt.fake1

    handler = BatchHandler((real_folder, fake0_folder, fake1_folder), path_check=False)

    mean0, mean1, std0, std1 = list(), list(), list(), list()

    for epoch in range(10, 1000, 10):
        print("epoch:", epoch)
        fake0_epoch_folder = os.path.join(default_fake0, str(epoch))
        fake1_epoch_folder = os.path.join(default_fake1, str(epoch))
        handler.set_folders(Trio(real_folder, fake0_epoch_folder, fake1_epoch_folder))
        handler.update_pools()
        handler.update_data()

        data = handler.data

        mean0.append(np.sum(data.get_abs_delta0().mean))
        mean1.append(np.sum(data.get_abs_delta1().mean))
        std0.append(np.sum(data.get_abs_delta0().std))
        std1.append(np.sum(data.get_abs_delta1().std))

    with open("stats.pkl", "wb") as f:
        pkl.dump({
            "mean0": mean0,
            "mean1": mean1,
            "std0": std0,
            "std1": std1,
        }, f)
