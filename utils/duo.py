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
        self._to_reset = True
        self._real = real

    @fake.setter
    def fake(self, fake):
        self._to_reset = True
        self._fake = fake

    @property
    def delta(self):
        if self._to_reset:
            self._set_delta()
        return self._delta

    @property
    def abs_delta(self):
        if self._to_reset:
            self._set_abs_delta()
        return self._abs_delta

    def _set_delta(self):
        try:
            self._delta = self._real - self._fake
        except ValueError as e:
            print(e)
            self._delta = None
            print("assigning `delta` of {} to {}".format(self, self._delta))

    def _set_abs_delta(self):
        try:
            self._abs_delta = abs(self._delta)
        except ValueError as e:
            print(e)
            self._abs_delta = None
            print("assigning `abs_delta` of {} to {}".format(self, self._abs_delta))

    @property
    def to_reset(self):
        return self._to_reset

    @to_reset.setter
    def to_reset(self, value):
        if not value:
            print("Warning: manually overriding resetting procedure")
        self._to_reset = value

