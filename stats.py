from abc import abstractmethod


class Stats:
    @abstractmethod
    def get_dict(self):
        pass

    @abstractmethod
    def set_dict(self, d: dict):
        pass

    @abstractmethod
    def extend(self, other):
        pass


class MeanStdStats(Stats):
    def __init__(self, mean=None, std=None, sample_size=1):
        self.mean = mean
        self.std = std
        self.sample_size = sample_size

    def __eq__(self, other):
        return self.mean == other.mean and self.std == other.mean

    def __sub__(self, other):
        if self.sample_size != other.sample_size:
            return None

        sample_size = self.sample_size
        mean = self.mean - other.mean
        std = self.std - other.std
        return MeanStdStats(mean=mean, std=std, sample_size=sample_size)

    def __neg__(self):
        return MeanStdStats(mean=-self.mean, std=-self.std, sample_size=self.sample_size)

    def __abs__(self):
        return MeanStdStats(mean=abs(self.mean), std=abs(self.std), sample_size=self.sample_size)

    def get_std(self):
        return self.std

    def get_mean(self):
        return self.mean

    def set_std(self, std):
        self.std = std

    def set_mean(self, mean):
        self.mean = mean

    def get_dict(self):
        return {"mean": self.mean, "std": self.std}

    def set_dict(self, d: dict):
        if "std" not in d:
            d["std"] = None
        if "mean" not in d:
            d["mean"] = None
        if "sample_size" not in d:
            d["sample_size"] = None

        self.std = d["std"]
        self.mean = d["mean"]
        self.sample_size = d['sample_size']

    def __add__(self, other):
        sample_size = self.sample_size + other.sample_size
        try:
            mean = (self.mean * self.sample_size + other.mean * other.sample_size) / sample_size
        except ValueError:
            print("force setting mean to None")
            mean = None

        try:
            std = (self.std * self.sample_size + other.std + other.sample_size) / sample_size
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

        self.mean = self.mean * self.sample_size + other.mean * other.sample_size
        self.std = self.std * self.sample_size + other.std * other.sample_size
        self.sample_size += other.sample_size
        self.mean /= self.sample_size
        self.std /= self.sample_size
