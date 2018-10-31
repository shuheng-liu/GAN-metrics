from abc import abstractmethod


class Stats:
    @abstractmethod
    @property
    def dict(self):
        pass

    @abstractmethod
    @dict.setter
    def dict(self, d: dict):
        pass

    @abstractmethod
    def extend(self, other):
        pass


class MeanStdStats(Stats):
    def __init__(self, mean=None, std=None, sample_size=1):
        self._mean = mean
        self._std = std
        self._sample_size = sample_size

    def __eq__(self, other):
        return self._mean == other.mean and self._std == other.mean

    def __sub__(self, other):
        if self._sample_size != other.sample_size:
            return None

        sample_size = self._sample_size
        mean = self._mean - other.mean
        std = self._std - other.std
        return MeanStdStats(mean=mean, std=std, sample_size=sample_size)

    def __neg__(self):
        return MeanStdStats(mean=-self._mean, std=-self._std, sample_size=self._sample_size)

    def __abs__(self):
        return MeanStdStats(mean=abs(self._mean), std=abs(self._std), sample_size=self._sample_size)

    @property
    def std(self):
        return self._std

    @property
    def mean(self):
        return self._mean

    @std.setter
    def std(self, std):
        self._std = std

    @mean.setter
    def mean(self, mean):
        self._mean = mean

    @property
    def dict(self):
        return {"mean": self._mean, "std": self._std}

    @dict.setter
    def dict(self, d: dict):
        self._std = d.get("std", None)
        self._mean = d.get("mean", None)
        self._sample_size = d.get('sample_size', None)

    def __add__(self, other):
        sample_size = self._sample_size + other.sample_size
        try:
            mean = (self._mean * self._sample_size + other.mean * other.sample_size) / sample_size
        except ValueError:
            print("force setting mean to None")
            mean = None

        try:
            std = (self._std * self._sample_size + other.std + other.sample_size) / sample_size
        except ValueError:
            print("force setting std to None")
            std = None

        return MeanStdStats(mean=mean, std=std, sample_size=sample_size)

    def extend(self, other):
        try:
            assert isinstance(other, MeanStdStats)
        except AssertionError as e:
            print(e)
            print("`other` is not a Stats object: {}".format(other))
            print("Aborting")
            return

        self._mean = self._mean * self._sample_size + other._mean * other._sample_size
        self._std = self._std * self._sample_size + other._std * other._sample_size
        self._sample_size += other._sample_size
        self._mean /= self._sample_size
        self._std /= self._sample_size
