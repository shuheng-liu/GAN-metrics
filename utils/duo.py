from itertools import chain


class Duo:
    def __init__(self, real=None, fake=None):
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
        self._to_reset = False

    def _set_abs_delta(self):
        try:
            self._abs_delta = abs(self._delta)
        except ValueError as e:
            print(e)
            self._abs_delta = None
            print("assigning `abs_delta` of {} to {}".format(self, self._abs_delta))
        self._to_reset = False

    @property
    def to_reset(self):
        return self._to_reset

    @to_reset.setter
    def to_reset(self, value):
        if not value:
            print("Warning: manually overriding resetting procedure")
        self._to_reset = value

    @property
    def duo(self):
        return self._real, self._fake

    @duo.setter
    def duo(self, tup):
        self._real, self._fake = tup
        self._to_reset = True

    def apply(self, func):
        self._real = func(self._real)
        self._fake = func(self._fake)
        self._to_reset = True

    def copy_apply(self, func):
        return Duo(func(self._real), func(self._fake))

    def __iter__(self):
        return chain(self._real, self._fake)

    def __len__(self):
        try:
            return len(self._real) + len(self._fake)
        except TypeError as e:
            print(e)
            return 0

    def __str__(self):
        return "Duo(real={}, fake={})".format(self._real, self._fake)

    def __repr__(self):
        return "<" + str(self) + " at {}>".format(id(self))

    


