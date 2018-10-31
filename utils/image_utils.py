import os
import numpy as np
from PIL import Image
from .. import stats


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


def get_mean_std_stats(images, cls=stats.MeanStdStats):
    """
    do basic stats over a bunch of images
    :param images: a 4-D numpy array in the shape of (N, W, H, C); or a list of 3-D arrays
    :return: a MeanStdStats object with mean and std being 3-D numpy array (W, H, C)
    """

    mean = np.mean(images, axis=0)
    std = np.std(images, axis=0)
    return stats.MeanStdStats(mean=mean, std=std, sample_size=len(images))
