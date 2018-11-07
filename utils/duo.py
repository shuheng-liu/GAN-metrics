class Duo:
    def __init__(self, real, fake):
        self._real = real
        self._fake = fake
        self._delta = None
        self._abs_delta = None
        self._to_reset = True

    @property
    def real(self):
        return self._real

    @property
    def fake(self):
        return self._fake

    @real.setter
    def real(self, real):
        self._real = real

    @fake.setter
    def fake(self, fake):
        self._fake = fake

    @property
    def delta(self):
        return self._delta

    @property
    def abs_delta(self):
        return self._abs_delta
