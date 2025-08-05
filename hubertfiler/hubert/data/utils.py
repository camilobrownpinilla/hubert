from typing import Any, Dict, List, Optional, Tuple, Union
from os import PathLike
from threading import Thread
from queue import Queue
from itertools import cycle, islice

import os
import torch
import torch.distributed as dist


PathOrStr = Union[str, PathLike]

def file_size(path: PathOrStr) -> int:
    """
    Get the size of a local or remote file in bytes.
    """
    return os.stat(path).st_size

def get_bytes_range(source: PathOrStr, bytes_start: int, num_bytes: int) -> bytes:
    with open(source, "rb") as f:
        f.seek(bytes_start)
        return f.read(num_bytes)
    
def barrier() -> None:
    if is_distributed():
        dist.barrier()

def get_global_rank() -> int:
    if is_distributed():
        return int(os.environ.get("RANK") or dist.get_rank())
    else:
        return 0
    
def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_world_size() -> int:
    if is_distributed():
        return dist.get_world_size()
    else:
        return 1
    
def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK") or 0)

def get_fs_local_rank() -> int:
    """Get the local rank per filesystem, meaning that, regardless of the number of nodes,
    if all ranks share the same filesystem then `get_fs_local_rank()` will be equivalent to `get_global_rank()`,
    but if nodes do not share the same filesystem then `get_fs_local_rank()` will be equivalent to `get_local_rank()`.
    """
    return int(os.environ.get("FS_LOCAL_RANK") or get_local_rank())

def threaded_generator(g, maxsize: int = 16, thread_name: Optional[str] = None):
    q: Queue = Queue(maxsize=maxsize)

    sentinel = object()

    def fill_queue():
        try:
            for value in g:
                q.put(value)
        except Exception as e:
            q.put(e)
        finally:
            q.put(sentinel)

    thread_name = thread_name or repr(g)
    thread = Thread(name=thread_name, target=fill_queue, daemon=True)
    thread.start()

    for x in iter(q.get, sentinel):
        if isinstance(x, Exception):
            raise RuntimeError(f"generator thread {thread_name} failed") from x
        else:
            yield x

def roundrobin(*iterables):
    """
    Call the given iterables in a round-robin fashion. For example:
    ``roundrobin('ABC', 'D', 'EF') --> A D E B F C``
    """
    # Adapted from https://docs.python.org/3/library/itertools.html#itertools-recipes
    num_active = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = cycle(islice(nexts, num_active))

