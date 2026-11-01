"""Background-thread prefetch of stacked batches, overlapping batch handoff with the train step."""
import copy
import logging
import queue
import threading

import gunpowder as gp
from gunpowder.profiling import Timing

logger = logging.getLogger(__name__)


class ThreadedPrefetch(gp.BatchFilter):
    """Request the next batch upstream in a background thread.

    PreCache workers build samples in parallel, but pulling them out of the
    multiprocessing queue (unpickling ~35 MiB per sample) and gp.Stack-ing
    them runs serially in the main process: ~1 s per batch at batch_size=18
    (sbatch/ppp/tools/bench_ppp_dataloader.py), with the GPU idle. Place this after
    gp.Stack so that work overlaps the train step.

    Requests equal to the first one seen (random seed ignored) are served from
    the prefetch queue; any other request is served synchronously. Upstream
    access is serialized with a lock, so upstream nodes never run concurrently.

    Args:

        num_batches (``int``):

            How many stacked batches to hold ready.
    """

    def __init__(self, num_batches=2):
        self.num_batches = num_batches
        self.request = None
        self.queue = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None

    def teardown(self):
        self.stop_event.set()
        if self.thread is not None:
            # unblock a producer waiting on a full queue
            try:
                while True:
                    self.queue.get_nowait()
            except queue.Empty:
                pass
            self.thread.join(timeout=60)
            self.thread = None
        self.request = None

    def provide(self, request):
        timing = Timing(self)
        timing.start()

        if self.request is None:
            self.request = copy.deepcopy(request)
            self.queue = queue.Queue(maxsize=self.num_batches)
            self.stop_event.clear()
            self.thread = threading.Thread(
                target=self._run, name="ThreadedPrefetch", daemon=True)
            self.thread.start()
            logger.info("prefetching %d batches in a background thread",
                        self.num_batches)

        if request == self.request:
            item = self._get()
            if isinstance(item, BaseException):
                raise item
            batch = item
        else:
            logger.debug("request differs from prefetched one, serving synchronously")
            with self.lock:
                batch = self.get_upstream_provider().request_batch(request)

        timing.stop()
        batch.profiling_stats.add(timing)
        return batch

    def _get(self):
        while True:
            try:
                return self.queue.get(timeout=10)
            except queue.Empty:
                if not self.thread.is_alive():
                    raise RuntimeError("ThreadedPrefetch thread died")

    def _run(self):
        upstream = self.get_upstream_provider()
        base = copy.deepcopy(self.request)
        while not self.stop_event.is_set():
            base._update_random_seed()
            request = copy.deepcopy(base)
            try:
                with self.lock:
                    item = upstream.request_batch(request)
            except Exception as e:
                item = e
            while not self.stop_event.is_set():
                try:
                    self.queue.put(item, timeout=1)
                    break
                except queue.Full:
                    continue
            if isinstance(item, Exception):
                return
