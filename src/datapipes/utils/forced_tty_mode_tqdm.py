from tqdm import tqdm
import sys

class tty_tqdm(tqdm):
    def display(self, msg=None, pos=None):
        # Force carriage return and flush to same line
        if msg is None:
            msg = self.__str__()
        sys.stdout.write('\r' + msg)
        sys.stdout.flush()

    def clear(self, *args, **kwargs):
        pass