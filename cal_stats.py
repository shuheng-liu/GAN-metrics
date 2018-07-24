import argparse
import os
import numpy as np
import pickle as pkl
from PIL import Image
from stats import MeanStdStats

img_suffices = ["jpg", "jpeg", "png"]
img_suffices = ["." + suffix for suffix in img_suffices]
img_suffices.extend([suffix.upper() for suffix in img_suffices])
print("listing legal suffices", img_suffices)


class Trio:
    """structured data containing real-fake0-fake1 trios; can be trio of any object or data"""

    def __init__(self, real=None, fake0=None, fake1=None):
        self._real = real
        self._fake0 = fake0
        self._fake1 = fake1
        self._delta0 = None
        self._delta1 = None
        self._abs_delta0 = None
        self._abs_delta1 = None
        self.update_absolute_delta()  # which automatically calls self.update_delta()

    def update_delta(self):
        try:
            self._delta0 = self._fake0 - self._real
        except TypeError:
            self._delta0 = None

        try:
            self._delta1 = self._fake1 - self._real
        except TypeError:
            self._delta1 = None

    def update_absolute_delta(self):
        self.update_delta()
        try:
            self._abs_delta0 = abs(self._delta0)
        except TypeError:
            self._abs_delta0 = None

        try:
            self._abs_delta1 = abs(self._delta1)
        except TypeError:
            self._abs_delta1 = None

    def get_delta0(self):
        return self._delta0

    def get_delta1(self):
        return self._delta1

    def get_abs_delta0(self):
        return self._abs_delta0

    def get_abs_delta1(self):
        return self._abs_delta1

    def get_real(self):
        return self._real

    def get_fake0(self):
        return self._fake0

    def get_fake1(self):
        return self._fake1

    def set_real(self, real):
        self._real = real

    def set_fake0(self, fake0):
        self._fake0 = fake0

    def set_fake1(self, fake1):
        self._fake1 = fake1

    def __iter__(self):
        return iter([self._real, self._fake0, self._fake1])

    def __len__(self):
        return 3


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

    def set_pools(self):
        real_pool = self.get_pool(self.folders.get_real())
        self.pools.set_real(real_pool)

        fake0_pool = self.get_pool(self.folders.get_fake0())
        self.pools.set_fake0(fake0_pool)

        fake1_pool = self.get_pool(self.folders.get_fake1())
        self.pools.set_fake1(fake1_pool)

    def set_data(self):
        self.data.set_real(get_stats(self.pools.get_real()))
        self.data.set_fake0(get_stats(self.pools.get_fake0()))
        self.data.set_fake1(get_stats(self.pools.get_fake1()))
        self.data.update_absolute_delta()


def image2array(img):
    return np.array(img.getdata()).reshape(img.size[0], img.size[1], 3)


def get_stats(images, cls=MeanStdStats):
    """
    do basic stats over a bunch of images
    :param images: a 4-D numpy array in the shape of (N, W, H, C); or a list of 3-D arrays
    :return: a MeanStdStats object with mean and std being 3-D numpy array (W, H, C)
    """

    mean = np.mean(images, axis=0)
    std = np.std(images, axis=0)
    return MeanStdStats(mean=mean, std=std, sample_size=len(images))


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

    mean0 = list()
    mean1 = list()
    std0 = list()
    std1 = list()

    for epoch in range(10, 1000, 10):
        print("epoch:", epoch)
        fake0_epoch_folder = os.path.join(default_fake0, str(epoch))
        fake1_epoch_folder = os.path.join(default_fake1, str(epoch))
        handler.set_folders(Trio(real_folder, fake0_epoch_folder, fake1_epoch_folder))
        handler.set_pools()
        handler.set_data()

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

    # with open("stats.pkl", "wb") as f:
    #     pkl.dump(data, f)

    # with open("stats.pkl", "rb") as f:
    #     data = pkl.load(f)
