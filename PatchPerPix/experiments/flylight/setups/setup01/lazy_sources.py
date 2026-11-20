"""Lazy random file source for training on many volumes (opt-in: [data] lazy_sources = true).

train.get_sources normally builds one source chain per training file (ZarrSource +
CsvPointsSource + Pad + RandomLocation + ...) and picks one per request with
gp.RandomProvider. Building a chain costs ~85 ms (RandomLocation loads the file's fg
points), so 50,000 files took ~85-90 min before the first iteration (experiment_v4
Phase 0, 2026-11-18).

LazyRandomSource replaces ``tuple(chains) + gp.RandomProvider()`` with one node that
picks a file uniformly at random per request (seeded by the request's random seed, as
RandomProvider does) and builds that file's chain on first use, keeping the most recent
chains in a per-process LRU cache. The sampling distribution is the same (uniform over
files, then the chain's own RandomLocation); startup costs one chain. Each PreCache worker
process keeps its own cache.

Validation never uses this: get_sources only takes the lazy path for training sources,
so the fixed validation crops (seed chain, see train._cast_raw_if_f16) are unchanged.
"""
import collections
import logging

import gunpowder as gp
import numpy as np

logger = logging.getLogger(__name__)


class LazyRandomSource(gp.BatchProvider):
    """Uniform random choice over ``files``; ``make_chain(fn)`` builds a file's pipeline."""

    def __init__(self, files, make_chain, cache_size=32, name="lazy"):
        self.files = list(files)
        if not self.files:
            raise ValueError("LazyRandomSource needs at least one file")
        self.make_chain = make_chain
        self.cache_size = max(1, int(cache_size))
        self.label = name
        self._cache = collections.OrderedDict()
        self.n_built = 0

    def _chain(self, i):
        chain = self._cache.get(i)
        if chain is not None:
            self._cache.move_to_end(i)
            return chain
        chain = self.make_chain(self.files[i])
        chain.setup()
        self.n_built += 1
        self._cache[i] = chain
        while len(self._cache) > self.cache_size:
            _, old = self._cache.popitem(last=False)
            old.internal_teardown()
        return chain

    def setup(self):
        # all chains provide the same keys / specs (RandomLocation makes the ROIs unbounded,
        # raw is float32 after the per-file cast); declare them from the first file
        spec = self._chain(0).spec
        for key, s in spec.items():
            self.provides(key, s.copy())
        logger.info("LazyRandomSource[%s]: %d files, cache %d chains per process",
                    self.label, len(self.files), self.cache_size)

    def provide(self, request):
        i = int(np.random.RandomState(request.random_seed % (2 ** 32)).randint(len(self.files)))
        return self._chain(i).request_batch(request)

    def teardown(self):
        while self._cache:
            _, chain = self._cache.popitem(last=False)
            chain.internal_teardown()
