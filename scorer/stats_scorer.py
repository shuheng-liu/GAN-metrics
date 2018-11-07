from base import BaseScorer
import numpy as np
from PIL.Image import Image


class StatsScorer(BaseScorer):
    def __init__(self, images_real, images_fake):
        super(StatsScorer, self).__init__(images_real, images_fake)
        self._latent1 = None
        self._latent0 = None

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

    def _set_latents(self):
        self._latent0 = self._flatten(self._convert_to_array(self._images0))
        self._latent1 = self._flatten(self._convert_to_array(self._images1))

    @property
    def latents(self):
        if self._latent0 is None or self._latent1 is None:
            self._set_latents()
        return self._latent1, self._latent0

