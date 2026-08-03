"""The loader is the last line of defence for rules that a YAML edit could break."""

from __future__ import annotations

import pytest
import yaml

from config.loader import ConfigError, load_settings


def test_ships_with_the_strategy_spec_from_claude_md():
    s = load_settings()
    assert s.universe == [
        "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLV", "XLE", "XLI", "XLP",
    ]
    assert s.strategy.rsi_period == 2
    assert s.strategy.sma_period == 200
    assert s.strategy.entry_rsi == 10.0
    assert s.strategy.exit_rsi == 70.0
    assert s.risk.max_positions == 3
    assert s.risk.time_stop_days == 7
    assert s.risk.kill_switch_drawdown == 0.15
    assert s.cost_per_side == 0.0005


def _write(tmp_path, patch: dict):
    """Write the shipped settings with a nested patch applied."""
    from config.loader import DEFAULT_SETTINGS_PATH

    raw = yaml.safe_load(DEFAULT_SETTINGS_PATH.read_text())
    for section, values in patch.items():
        if isinstance(values, dict):
            raw.setdefault(section, {}).update(values)
        else:
            raw[section] = values
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


@pytest.mark.parametrize("bad_cost", [0.0, -0.001])
def test_rule_2_zero_or_negative_cost_is_rejected(tmp_path, bad_cost):
    """Rule 2: never produce a zero-cost backtest, even 'just to see'."""
    path = _write(tmp_path, {"costs": {"per_side_pct": bad_cost}})
    with pytest.raises(ConfigError, match="rule 2"):
        load_settings(path)


def test_rule_2_cannot_be_bypassed_by_an_override():
    with pytest.raises(ConfigError, match="rule 2"):
        load_settings(cost_per_side=0.0)


def test_rule_5_margin_and_shorting_are_rejected(tmp_path):
    with pytest.raises(ConfigError, match="no margin"):
        load_settings(_write(tmp_path, {"risk": {"allow_margin": True}}))
    with pytest.raises(ConfigError, match="long-only"):
        load_settings(_write(tmp_path, {"risk": {"long_only": False}}))


def test_rule_5_kill_switch_must_be_a_sane_fraction(tmp_path):
    for bad in (0.0, 1.0, 15):
        with pytest.raises(ConfigError, match="kill_switch_drawdown"):
            load_settings(_write(tmp_path, {"risk": {"kill_switch_drawdown": bad}}))


def test_rule_6_time_stop_must_exist(tmp_path):
    with pytest.raises(ConfigError, match="rule 6"):
        load_settings(_write(tmp_path, {"risk": {"time_stop_days": 0}}))


def test_rule_4_windows_must_not_overlap(tmp_path):
    path = _write(tmp_path, {"windows": {"train_end": "2020-06-30"}})
    with pytest.raises(ConfigError, match="rule 4"):
        load_settings(path)


def test_entry_threshold_below_exit_threshold(tmp_path):
    """entry_rsi >= exit_rsi means every fill instantly satisfies its own exit."""
    path = _write(tmp_path, {"strategy": {"entry_rsi": 75.0}})
    with pytest.raises(ConfigError, match="below exit_rsi"):
        load_settings(path)


def test_benchmark_must_be_in_universe(tmp_path):
    path = _write(tmp_path, {"benchmark": "VTI"})
    with pytest.raises(ConfigError, match="benchmark"):
        load_settings(path)


def test_duplicate_symbols_rejected(tmp_path):
    path = _write(tmp_path, {"universe": ["SPY", "QQQ", "SPY"]})
    with pytest.raises(ConfigError, match="duplicate"):
        load_settings(path)


def test_overrides_reach_nested_fields():
    s = load_settings(entry_rsi=5.0, time_stop_days=10)
    assert s.strategy.entry_rsi == 5.0
    assert s.risk.time_stop_days == 10
    assert s.strategy.exit_rsi == 70.0  # untouched


def test_unknown_override_is_an_error():
    with pytest.raises(ConfigError, match="unknown settings override"):
        load_settings(entry_rsl=5.0)  # typo


def test_fingerprint_is_stable_and_sensitive():
    """experiments.csv groups runs by this hash, so it must track real changes."""
    a_hash, a_json = load_settings().fingerprint()
    assert a_hash == load_settings().fingerprint()[0]
    assert a_hash != load_settings(entry_rsi=5.0).fingerprint()[0]
    assert '"entry_rsi":10.0' in a_json


def test_settings_are_frozen():
    """Nothing should be able to mutate config mid-run."""
    import dataclasses

    s = load_settings()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.cost_per_side = 0.0  # type: ignore[misc]
