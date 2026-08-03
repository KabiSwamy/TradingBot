"""Load and validate config/settings.yaml.

Validation here is not defensive boilerplate — several of the CLAUDE.md rules
are only as strong as the config that feeds them. A zero-cost backtest (rule 2)
or a missing kill switch (rule 5) is a rule violation that would otherwise be a
one-character YAML edit away, so the loader refuses to produce such a config.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parent / "settings.yaml"


class ConfigError(ValueError):
    """Raised when settings.yaml is missing a key or violates a project rule."""


@dataclass(frozen=True)
class StrategySettings:
    rsi_period: int
    sma_period: int
    entry_rsi: float
    exit_rsi: float


@dataclass(frozen=True)
class RiskSettings:
    max_positions: int
    time_stop_days: int
    kill_switch_drawdown: float
    long_only: bool
    allow_margin: bool


@dataclass(frozen=True)
class Settings:
    universe: list[str]
    benchmark: str
    strategy: StrategySettings
    risk: RiskSettings
    cost_per_side: float
    initial_capital: float
    train_start: str
    train_end: str
    test_start: str
    cache_dir: str
    max_gap_business_days: int
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def fingerprint(self) -> tuple[str, str]:
        """Return (short_hash, canonical_json) identifying this configuration.

        Used by results/experiments.csv so runs are groupable after the fact
        without having to reconstruct which settings produced them (rule 4).
        The raw YAML is excluded — only the resolved values matter.
        """
        payload = asdict(self)
        payload.pop("raw", None)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return sha256(canonical.encode()).hexdigest()[:12], canonical


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"settings.yaml: missing required key {context}.{key}")
    return mapping[key]


def load_settings(path: str | Path | None = None, **overrides: Any) -> Settings:
    """Load settings.yaml into a validated, frozen Settings object.

    `overrides` patches top-level resolved fields (used by Phase 2 sweeps and by
    tests) and is validated exactly like the file itself — an override cannot
    smuggle in a rule violation either.
    """
    path = Path(path) if path is not None else DEFAULT_SETTINGS_PATH
    if not path.exists():
        raise ConfigError(f"settings file not found: {path}")

    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"settings file must contain a YAML mapping: {path}")

    strategy_raw = _require(raw, "strategy", "root")
    risk_raw = _require(raw, "risk", "root")
    costs_raw = _require(raw, "costs", "root")
    capital_raw = _require(raw, "capital", "root")
    windows_raw = _require(raw, "windows", "root")
    data_raw = raw.get("data", {}) or {}

    settings = Settings(
        universe=list(_require(raw, "universe", "root")),
        benchmark=str(_require(raw, "benchmark", "root")),
        strategy=StrategySettings(
            rsi_period=int(_require(strategy_raw, "rsi_period", "strategy")),
            sma_period=int(_require(strategy_raw, "sma_period", "strategy")),
            entry_rsi=float(_require(strategy_raw, "entry_rsi", "strategy")),
            exit_rsi=float(_require(strategy_raw, "exit_rsi", "strategy")),
        ),
        risk=RiskSettings(
            max_positions=int(_require(risk_raw, "max_positions", "risk")),
            time_stop_days=int(_require(risk_raw, "time_stop_days", "risk")),
            kill_switch_drawdown=float(
                _require(risk_raw, "kill_switch_drawdown", "risk")
            ),
            long_only=bool(risk_raw.get("long_only", True)),
            allow_margin=bool(risk_raw.get("allow_margin", False)),
        ),
        cost_per_side=float(_require(costs_raw, "per_side_pct", "costs")),
        initial_capital=float(_require(capital_raw, "initial", "capital")),
        train_start=str(_require(windows_raw, "train_start", "windows")),
        train_end=str(_require(windows_raw, "train_end", "windows")),
        test_start=str(_require(windows_raw, "test_start", "windows")),
        cache_dir=str(data_raw.get("cache_dir", "data/cache")),
        max_gap_business_days=int(data_raw.get("max_gap_business_days", 5)),
        raw=raw,
    )

    if overrides:
        settings = _apply_overrides(settings, overrides)

    _validate(settings)
    return settings


def _apply_overrides(settings: Settings, overrides: dict[str, Any]) -> Settings:
    """Return a copy of `settings` with nested fields replaced.

    Accepts both flat strategy/risk field names (entry_rsi=5) and top-level
    Settings fields (cost_per_side=0.001), so sweep code can stay terse.
    """
    from dataclasses import replace

    strategy_fields = {f for f in StrategySettings.__dataclass_fields__}
    risk_fields = {f for f in RiskSettings.__dataclass_fields__}

    strategy_patch: dict[str, Any] = {}
    risk_patch: dict[str, Any] = {}
    top_patch: dict[str, Any] = {}

    for key, value in overrides.items():
        if key in strategy_fields:
            strategy_patch[key] = value
        elif key in risk_fields:
            risk_patch[key] = value
        elif key in Settings.__dataclass_fields__:
            top_patch[key] = value
        else:
            raise ConfigError(f"unknown settings override: {key}")

    if strategy_patch:
        top_patch["strategy"] = replace(settings.strategy, **strategy_patch)
    if risk_patch:
        top_patch["risk"] = replace(settings.risk, **risk_patch)
    return replace(settings, **top_patch)


def _validate(s: Settings) -> None:
    """Enforce the CLAUDE.md rules that a config file could otherwise break."""
    if not s.universe:
        raise ConfigError("universe must not be empty")
    if len(set(s.universe)) != len(s.universe):
        raise ConfigError("universe contains duplicate symbols")
    if s.benchmark not in s.universe:
        # Not fatal in principle, but rule 12 requires the benchmark on every
        # report, and fetching it separately is a silent extra data dependency.
        raise ConfigError(
            f"benchmark {s.benchmark!r} must be part of the universe so it is "
            "always fetched alongside it"
        )

    # Rule 2: COSTS ALWAYS ON. A zero-cost backtest is not producible from this
    # loader, "just to see" or otherwise.
    if s.cost_per_side <= 0:
        raise ConfigError(
            "costs.per_side_pct must be > 0 — CLAUDE.md rule 2 forbids "
            "zero-cost backtests"
        )
    if s.cost_per_side > 0.05:
        raise ConfigError("costs.per_side_pct > 5% is almost certainly a typo")

    # Rule 5: risk limits must actually constrain something.
    if s.risk.max_positions < 1:
        raise ConfigError("risk.max_positions must be >= 1")
    if not 0 < s.risk.kill_switch_drawdown < 1:
        raise ConfigError(
            "risk.kill_switch_drawdown must be a fraction in (0, 1), e.g. 0.15"
        )
    if not s.risk.long_only:
        raise ConfigError("CLAUDE.md rule 5: this project is long-only")
    if s.risk.allow_margin:
        raise ConfigError("CLAUDE.md rule 5: cash account only, no margin")

    # Rule 6: the time stop is not optional.
    if s.risk.time_stop_days < 1:
        raise ConfigError("risk.time_stop_days must be >= 1 (rule 6)")

    if s.initial_capital <= 0:
        raise ConfigError("capital.initial must be > 0")

    # Indicator sanity.
    if s.strategy.rsi_period < 1:
        raise ConfigError("strategy.rsi_period must be >= 1")
    if s.strategy.sma_period < 1:
        raise ConfigError("strategy.sma_period must be >= 1")
    if not 0 <= s.strategy.entry_rsi <= 100:
        raise ConfigError("strategy.entry_rsi must be within [0, 100]")
    if not 0 <= s.strategy.exit_rsi <= 100:
        raise ConfigError("strategy.exit_rsi must be within [0, 100]")
    if s.strategy.entry_rsi >= s.strategy.exit_rsi:
        raise ConfigError(
            "strategy.entry_rsi must be below exit_rsi, otherwise every entry "
            "would immediately satisfy its own exit condition"
        )

    # Rule 4: the windows must not overlap, or out-of-sample stops being out.
    if s.train_end >= s.test_start:
        raise ConfigError(
            "windows.train_end must precede windows.test_start — overlapping "
            "windows destroy the out-of-sample discipline of rule 4"
        )
