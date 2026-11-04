"""Test-session setup: cap CPU threads (min(8, usable cores), or OMP_NUM_THREADS if set).

The tests run small networks and simulations; on many-core machines PyTorch's default of one thread per
core makes them slower and occupies the whole machine (see ``supermimic.vehicles.threads``)."""

from supermimic.vehicles.threads import limit_threads

limit_threads()
