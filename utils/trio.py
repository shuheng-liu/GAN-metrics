from itertools import chain


class Trio:
    """structured data containing real-fake0-fake1 trios; can be trio of any object or data"""

    def __init__(self, real=None, fake0=None, fake1=None):
        self._real = real
        self._fake0 = fake0
        self._fake1 = fake1
        self._delta0 = None
        self._delta1 = None
        self._abs_delta0 = None
        self._abs_delta1 = None
        self._to_reset = True
        self.update_absolute_delta()  # which automatically calls self.update_delta()

    def update_delta(self):
        try:
            self._delta0 = self._fake0 - self._real
        except TypeError:
            self._delta0 = None

        try:
            self._delta1 = self._fake1 - self._real
        except TypeError:
            self._delta1 = None

        self._to_reset = False

    def update_absolute_delta(self):
        self.update_delta()
        try:
            self._abs_delta0 = abs(self._delta0)
        except TypeError:
            self._abs_delta0 = None

        try:
            self._abs_delta1 = abs(self._delta1)
        except TypeError:
            self._abs_delta1 = None

        self._to_reset = False

    @property
    def delta0(self):
        if self._to_reset:
            self.update_delta()
        return self._delta0

    @property
    def delta1(self):
        if self._to_reset:
            self.update_delta()
        return self._delta1

    @property
    def abs_delta0(self):
        if self._to_reset:
            self.update_absolute_delta()
        return self._abs_delta0

    @property
    def abs_delta1(self):
        if self._to_reset:
            self.update_absolute_delta()
        return self._abs_delta1

    @property
    def to_reset(self):
        return self._to_reset

    @property
    def real(self):
        return self._real

    @property
    def fake0(self):
        return self._fake0

    @property
    def fake1(self):
        return self._fake1

    @property
    def trio(self):
        return self._real, self._fake0, self._fake1

    @real.setter
    def real(self, real):
        self._real = real
        self._to_reset = True

    @fake0.setter
    def fake0(self, fake0):
        self._fake0 = fake0
        self._to_reset = True

    @fake1.setter
    def fake1(self, fake1):
        self._fake1 = fake1
        self._to_reset = True

    @trio.setter
    def trio(self, tup):
        self._real, self._fake0, self._fake1 = tup
        self._to_reset = True

    @to_reset.setter
    def to_reset(self, value):
        self._to_reset = value

    def apply(self, func, *args, **kwargs):
        self._real = func(self._real, *args, **kwargs)
        self._fake0 = func(self._fake0, *args, **kwargs)
        self._fake1 = func(self._fake1, *args, **kwargs)
        self._to_reset = True

    def copy_apply(self, func, *args, **kwargs):
        return Trio(
            func(self._real, *args, **kwargs),
            func(self._fake0, *args, **kwargs),
            func(self._fake1, *args, **kwargs)
        )

    def __iter__(self):
        return chain(self._real, self._fake0, self._fake1)

    def __len__(self):
        try:
            return len(self._real) + len(self._fake0) + len(self._fake1)
        except TypeError as e:
            print(e)
            print("falling back to default length 0 for {}".format(self))
            return 0

    def __str__(self):
        return "Trio(real={}, fake0={}, fake1={})".format(self._real, self._fake0, self.fake1)

    def __repr__(self):
        return "<" + str(self) + " at {}>".format(id(self))
