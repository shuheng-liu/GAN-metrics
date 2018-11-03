import os
import pickle as pkl


class Dumper:
    def __init__(self, obj, dump_dir):
        self.obj = obj
        self.dump_dir = dump_dir
        # last placeholder is for offset when dumping
        self.filename = "{}-at-{}".format(obj.__class__.__name__, id(obj)) + "-{}.pkl"

    def _make_dump_dirs(self):
        try:
            os.makedirs(self.dump_dir)
        except FileExistsError as e:
            print(e)
            print("aborting dumping dirs")

    def _dumpable(self, path, force=False):
        if not os.path.exists(self.dump_dir):
            try:
                self._make_dump_dirs()
            except PermissionError as e:
                print(e)
                return False

        if not os.path.exists(path):
            return True

        if os.path.isfile(path) and force:
            return True

        return False

    def dump(self, offset=0, force=False):
        path = os.path.join(self.dump_dir, self.filename.format(offset))
        if not self._dumpable(path, force=force):
            print("cannot dump file at {}".format(path))
            return False

        with open(path, "wb") as f:
            pkl.dump(self.obj, f)

        print("dump successful at {}".format(path))
