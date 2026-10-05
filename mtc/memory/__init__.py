"""Memory package: store protocols, in-memory implementations and store plumbing.

Modules reach persistence only through the protocols in
``mtc.memory.protocols``; ``mtc.memory.inmemory`` holds the default
implementations, and a None store always means the in-memory implementation.
``mtc.memory.clients`` builds the optional network clients (imported lazily, a
missing client names its extra), ``mtc.memory.redis_manager`` is the Redis
working memory and ``mtc.memory.resilient_wrappers`` the opt-in circuit-breaker
layers over real clients. Importing this package imports no database driver.
"""
