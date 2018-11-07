from base import BaseScorer


class BinaryScorer:
    def __init__(self, scorer1: BaseScorer, scorer2: BaseScorer):
        self.scorer1 = scorer1
        self.scorer2 = scorer2

    @property
    def score1(self):
        return self.scorer1.score

    @property
    def score2(self):
        return self.scorer2.score

    @property
    def scores(self):
        return self.scorer1, self.scorer2
