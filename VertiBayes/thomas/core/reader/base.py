import os.path

from . import net
from . import oobn


class UnknownFileTypeError(Exception):
    pass


def read(filename):
    _, ext = os.path.splitext(filename)
    ext = ext.lower()

    if ext == '.oobn':
        return oobn.read(filename)

    if ext == '.net':
        return net.read(filename)

    msg = f"Don't know how to handle the '{ext}' extension."
    raise UnknownFileTypeError(msg)