"""utilities"""
import os
import numpy as np

# def sep(default='-', nchar=80):
#     print(default * nchar)


def get_pkg_filename(filename, path='data'):
    """Return filename for data contained in this pacakge."""
    directory = os.path.dirname(__file__)
    return os.path.join(directory, path, filename)


def flatten(list_):
    """Via https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists"""
    return [item for sublist in list_ for item in sublist]


def isiterable(obj):
    """Return True iff an object is iterable."""
    try:
        iter(obj)
    except Exception:
        return False
    else:
        return True


def remove_none_values_from_dict(dict_):
    """Remove none values, like `None` and `np.nan` from the dict."""
    def t(x):
        return (x is None) or (isinstance(x, float) and np.isnan(x))

    result = {k: v for k, v in dict_.items() if not t(v)}
    return result
