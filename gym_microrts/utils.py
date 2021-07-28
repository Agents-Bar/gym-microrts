from typing import List

import numpy


def to_list(obj: object) -> List:
    if isinstance(obj, list):
        return obj
    elif isinstance(obj, numpy.ndarray):
        return obj.tolist()

    # Just try...
    return list(obj)

def to_numpy(l: List) -> numpy.ndarray:
    return numpy.array(l)
