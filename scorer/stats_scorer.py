from base import BaseScorer
from utils import Duo
import numpy as np
from PIL.Image import Image


class StatsScorer(BaseScorer):
    def __init__(self, images_real, images_fake):
        super(StatsScorer, self).__init__(images_real, images_fake)
        self._latent_duo = Duo()

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
        latent0 = self._flatten(self._convert_to_array(self._images0))
        latent1 = self._flatten(self._convert_to_array(self._images1))
        self._latent_duo = Duo(real=latent1, fake=latent0)

    @property
    def latent_duo(self):
        if self.latent_duo is None:
            self._set_latent_duo()
        return self._latent_duo
