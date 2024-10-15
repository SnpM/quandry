from threading import Lock
from typing import *

target_locks:dict[Type,Lock] = {}
target_flags:set[Type] = set()
_loader_lock = Lock()
def static_init(cls:Callable, **kwargs) -> Callable:
    with _loader_lock:
        if cls not in target_locks:
            target_locks[cls] = Lock()
        target_lock = target_locks[cls]
    with target_lock:
        if cls not in target_flags:
            target_flags.add(cls)
            func = cls.__dict__.get('static_init',None)
            if func is None:
                raise TypeError(f"{cls} must define static_init classmethod")
            else: 
                # Ensure that static_init is a @classmethod
                if not isinstance(func, classmethod):
                    raise TypeError(f"{cls}.static_init must be a classmethod")
                else:
                    cls.static_init(**kwargs)
    return cls