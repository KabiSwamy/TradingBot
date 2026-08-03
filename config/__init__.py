"""Configuration package: settings.yaml plus its loader.

The YAML file lives inside this package deliberately, so `config` is both the
importable loader and the single home of every tunable threshold.
"""

from config.loader import Settings, load_settings

__all__ = ["Settings", "load_settings"]
