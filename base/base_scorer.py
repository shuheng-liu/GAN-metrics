from abc import ABC, abstractmethod


class BaseScorer(ABC):
    @abstractmethod
    def __init__(self, images_real, images_fake):
        pass

    @property
    @abstractmethod
    def score(self):
        """
        returns the score of one GAN computed against ground truths
        :rtype: float
        """
        pass
