from abc import ABC, abstractmethod


class BaseScorer(ABC):
    @abstractmethod
    def __init__(self, images_real, images_fake):
        self._images0 = images_fake
        self._images1 = images_real
        self._score = None

    @property
    @abstractmethod
    def score(self):
        """
        returns the score of one GAN computed against ground truths
        :rtype: float
        """
        return None