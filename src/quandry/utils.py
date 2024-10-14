from threading import Lock
from typing import *

_class_locks:dict[Type,Lock] = {}
_loader_lock = Lock()
def static_init(cls):
    with _loader_lock:
        if cls not in _class_locks:
            _class_locks[cls] = Lock()
        class_lock = _class_locks[cls]
    with class_lock:
        if getattr(cls, "static_init", None):
            if not getattr("static_init_done"):
                cls.static_init()
                cls.static_init_done = True
    return cls