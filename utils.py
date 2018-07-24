import os
import argparse
import numpy as np
from PIL import Image
from datagenerator import ImageDataGenerator
from stats import MeanStdStats


# generate a txt file containing image paths and labels
def make_list(folders, flags=None, ceils=None, mode='train', store_path='/output', verbose=False):
    suffices = ('jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG')
    if ceils is None: ceils = [-1] * len(folders)  # ceil constraint not imposed
    if flags is None: flags = list(range(len(folders)))  # flags = [0, 1, ..., n-1]
    assert len(folders) == len(flags) == len(ceils), (len(folders), len(flags), len(ceils))
    assert mode in ['train', 'val', 'test']
    folders_flags_ceils = [tup for tup in zip(folders, flags, ceils)
                           if isinstance(tup[0], str) and os.path.isdir(tup[0])]
    assert folders_flags_ceils

    if verbose:
        print('Making %s list' % mode)
        for tup in folders_flags_ceils:
            print('Folder {}: flag = {}, ceil = {}'.format(*tup))
    if not os.path.isdir(store_path): os.mkdir(store_path)
    out_list = os.path.join(store_path, mode + '.txt')
    list_length = 0
    with open(out_list, 'w') as fo:
        for (folder, flag, ceil) in folders_flags_ceils:
            count = 0
            for pic_name in os.listdir(folder):
                if pic_name.split('.')[-1] not in suffices:
                    print('Ignoring non-image file {} in folder {}.'.format(pic_name, folder),
                          'Legal suffices are', suffices)
                    continue
                count += 1
                list_length += 1
                fo.write("{} {}\n".format(os.path.join(folder, pic_name), flag))
                # if ceil is imposed (ceil > 0) and count exceeds ceil, break and write next flag
                if 0 < ceil <= count: break
    if verbose:
        print('%s list made\n' % mode)
    return out_list, list_length


def get_environment_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", required=True, help="folders to the parent directory of real samples")
    parser.add_argument("--generated", required=True, help="folders to the parent directory of generated samples")
    parser.add_argument("--dumpDir", default="output/", help="dir to dump latent_tsr representations")
    parser.add_argument("--reuse", action="store_true", help="reuse dumped data")
    return parser.parse_args()


def get_init_op(iterator, some_data: ImageDataGenerator):
    return iterator.make_initializer(some_data.data)


def crop_folder(dir_raw, dir_new, grid_size=64, margin_size=2):
    assert os.path.isdir(dir_raw)
    legal_suffices = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG']

    try:
        os.mkdir(dir_new)
    except FileExistsError:
        print("%s already exists" % dir_new)

    img_names = os.listdir(dir_raw)
    img_names.sort()
    for img_name in img_names:
        if img_name.split('.')[-1] not in legal_suffices:
            continue
        print('processing', img_name)
        crop_image(os.path.join(dir_raw, img_name), dir_new, margin_size=margin_size, grid_size=grid_size)


def crop_image(path_old, dir_new, margin_size=2, grid_size=64):
    try:
        os.makedirs(dir_new)
    except FileExistsError:
        pass
    img_name = os.path.split(path_old)[-1]
    img = Image.open(path_old)
    width, height = img.size
    assert (width - margin_size) % (margin_size + grid_size) == 0, (width, margin_size, grid_size)
    assert (height - margin_size) % (margin_size + grid_size) == 0, (height, margin_size, grid_size)
    num_w = (width - margin_size) // (grid_size + margin_size)
    num_h = (height - margin_size) // (grid_size + margin_size)
    for i in range(num_w):
        for j in range(num_h):
            # Do the cropping procedure
            x0, y0 = margin_size + i * (grid_size + margin_size), margin_size + j * (grid_size + margin_size)
            x1, y1 = x0 + grid_size, y0 + grid_size
            grid = img.crop([x0, y0, x1, y1])
            save_name = os.path.splitext(img_name)[0] + "_%d_%d" % (i, j) + os.path.splitext(img_name)[1]
            grid.save(os.path.join(dir_new, save_name))


def resize_real(dir_raw, dir_new, grid_size=64):
    assert os.path.isdir(dir_raw)
    legal_suffices = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG']
    try:
        os.mkdir(dir_new)
    except FileExistsError:
        print("%s already exists" % dir_new)

    img_names = os.listdir(dir_raw)
    img_names.sort()
    for img_name in img_names:
        if img_name.split('.')[-1] not in legal_suffices:
            continue

        # UNDONE img_count undefined
        print('processing real ID: {}, name: {}'.format(img_count, img_name))
        img = Image.open(os.path.join(dir_raw, img_name))
        img_resized = img.resize((grid_size, grid_size), Image.ANTIALIAS)
        img_resized.save(os.path.join(dir_new, img_name))


def image2array(img):
    return np.array(img.getdata()).reshape(img.size[0], img.size[1], 3)


def get_mean_std_stats(images, cls=MeanStdStats):
    """
    do basic stats over a bunch of images
    :param images: a 4-D numpy array in the shape of (N, W, H, C); or a list of 3-D arrays
    :return: a MeanStdStats object with mean and std being 3-D numpy array (W, H, C)
    """

    mean = np.mean(images, axis=0)
    std = np.std(images, axis=0)
    return MeanStdStats(mean=mean, std=std, sample_size=len(images))


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
