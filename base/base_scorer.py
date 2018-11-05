from abc import ABC, abstractmethod


class BaseScorer(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @property
    @abstractmethod
    def score(self):
        """
        returns the score of one GAN computed against ground truths
        :rtype: float
        """
        pass


class BaseBinaryScorer:
    def __init__(self, scorer1: BaseScorer, scorer2: BaseScorer):
        self.scorer1 = scorer1
        self.scorer2 = scorer2

    def score1(self):
        return self.scorer1.score

    def score2(self):
        return self.scorer2.score
