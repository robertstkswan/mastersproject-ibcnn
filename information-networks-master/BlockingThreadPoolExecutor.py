from concurrent.futures import ThreadPoolExecutor
import queue


class BlockingThreadPoolExecutor(ThreadPoolExecutor):
    def __init__(self, max_workers=None, thread_name_prefix=''):
        super().__init__(max_workers=max_workers, thread_name_prefix=thread_name_prefix)
        self._work_queue = queue.Queue(maxsize=max_workers)
