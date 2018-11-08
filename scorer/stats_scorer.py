from base import BaseScorer
from utils import Duo
import numpy as np
from PIL.Image import Image
from stats import MeanStdStats


class StatsScorer(BaseScorer):
    def __init__(self, images_real, images_fake):
        super(StatsScorer, self).__init__(images_real, images_fake)
        self._latent_duo = Duo()
        self._mean = None
        self._std = None

    @classmethod
    def _convert_to_array(cls, images):
        if isinstance(images, np.ndarray):
            return images
        elif isinstance(images, (list, tuple)):
            try:
                if isinstance(images[0], Image):
                    return np.stack(np.asarray(img) for img in images)
                else:
                    return np.stack(images)
            except IndexError as e:
                print("check that `images` of {} is not empty".format(cls.__name__))
                raise e
        else:
            raise TypeError("unsupported input format {}".format(type(images)))

    @staticmethod
    def _flatten(array):
        return np.reshape(array, [len(array), -1])

    def _set_latent_duo(self):
        latent0 = self._flatten(self._convert_to_array(self._images0))  # n_images * n_channels array
        latent1 = self._flatten(self._convert_to_array(self._images1))  # n_images * n_channels array
        self._latent_duo = Duo(real=latent1, fake=latent0)

    @property
    def latent_duo(self):
        if self.latent_duo is None:
            self._set_latent_duo()
        return self._latent_duo

    def _set_mean(self):
        if self._latent_duo is None:
            self._set_latent_duo()
        pixel_mean = self._latent_duo.copy_apply(np.mean, axis=0)  # mean computed on each pixel throughout batch
        self._mean = np.sum(pixel_mean.abs_delta)

    @property
    def mean(self):
        if self._mean is None:
            self._set_mean()
        return self._mean

    def _set_std(self):
        if self._latent_duo is None:
            self._set_latent_duo()
        pixel_std = self._latent_duo.copy_apply(np.std, axis=0)  # std computed on each pixel throughout batch
        self._std = np.sum(pixel_std.abs_delta)

    @property
    def std(self):
        if self._std is None:
            self._set_std()
        return self._std

    @property
    def score(self):
        return MeanStdStats(
            mean=self.mean,
            std=self.std,
            sample_size=self.latent_duo.real.shape[1],
        )


class MeanScorer(StatsScorer):
    @property
    def score(self):
        return self.mean


class StdScorer(StatsScorer):
    @property
    def score(self):
        return self.std
