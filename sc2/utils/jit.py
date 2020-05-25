import numba
import sc2


class JitConfiguration(object):
    enable = False
    target = "cpu"

 
config = JitConfiguration()


def configurate(name: str, value):
    if hasattr(config, name):
        setattr(config, name, value)
    else:
        raise ValueError(f"JitConfiguration has no attribute {name}")


def enable():
    configurate("enable", True)


def disable():
    configurate("enable", False)


def target(tar: str):
    configurate("target", tar)


def just_in_time(*args, **kwargs):
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
    
