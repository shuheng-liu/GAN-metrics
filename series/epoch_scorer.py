from abc import abstractmethod
from base import BaseScorer
from stats import MeanStdStats


class EpochScorer(BaseScorer):
    def __init__(self, batches: list, epochs=None):
        super(EpochScorer, self).__init__()  # suppresses warnings from IDE
        self.batches = batches
        if epochs is None:
            self.epochs = list(range(len(batches)))
        else:
            self.epochs = epochs
        assert len(self.epochs) == len(self.batches)

    @property
    @abstractmethod
    def scores(self):
        """
        A list of scores corresponding to the self.batches
        :rtype: list
        """
        pass

    @property
    def mean_std_stats(self):
        return MeanStdStats(self.scores)
