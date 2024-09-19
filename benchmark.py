import time
from collections.abc import Callable


def benchmark(func: Callable, *args, **kwargs) -> int:
    start = time.perf_counter_ns()
    func(*args, **kwargs)
    return time.perf_counter_ns() - start
