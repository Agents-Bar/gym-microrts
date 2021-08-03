from typing import Any, Dict, List

import gym
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

def extract_space_info(space) -> Dict[str, Any]:
    if isinstance(space, gym.spaces.multi_discrete.MultiDiscrete):
        return dict(dtype=str(space.dtype), shape=to_list(space.nvec))
    elif "Discret" in str(space):
        return dict(dtype=str(space.dtype), shape=(space.n,))
    else:
        return {
            "low": space.low.tolist(),  # TODO: If all are equal replace with a single value
            "high": space.high.tolist(),  # TODO: If all are equal, replace with a single value
            "shape": space.shape,
            "dtype": str(space.dtype),
        }
