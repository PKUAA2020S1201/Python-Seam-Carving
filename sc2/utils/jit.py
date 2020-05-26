"""
sc2.utils.jit

just in time
"""

import numba
import sc2


class JitConfiguration(object):
    enable = False
    target = "cpu"

 
config = JitConfiguration()


def configurate(name: str, value):
    """
    modify config
    """

    if hasattr(config, name):
        setattr(config, name, value)
    else:
        raise ValueError(f"JitConfiguration has no attribute {name}")


def enable():
    """
    enable just in time globally
    """

    configurate("enable", True)


def disable():
    """
    disable just in time globally
    """

    configurate("enable", False)


def target(tar: str):
    """
    set the target device for numba.jit
    """

    configurate("target", tar)


def just_in_time(*args, **kwargs):
    """
    the replacement of numba.jit wrapper

    NOTE:
        if jit is enabled,
            the jit function will be used, 
        otherwise,
            the original function will be used.
    """

    if "target" in kwargs or len(args) >= 3:
        raise ValueError(f"jit arguments should not contain target")
    
    ori_func = args[0]
    jit_func = numba.jit(*args, **kwargs, target=config.target)

    def wrapper(*args, **kwargs):
        if config.enable:
            return jit_func(*args, **kwargs)
        else:
            return ori_func(*args, **kwargs)
    
    return wrapper
    
