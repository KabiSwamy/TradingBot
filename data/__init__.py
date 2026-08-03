"""Market data: sources, validation, and the local parquet cache.

The cached parquet files live in `data/cache/`, which is gitignored — they are
regenerable, and committing them would put a rescalable price history into
version control where it would drift out of step with the adjustment convention
that produced it.
"""
