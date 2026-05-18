"""YAML configuration file watcher with hot-reload support.

Monitors a YAML configuration file for changes and triggers reload callbacks.
Uses file modification time polling (zero external dependencies).
"""

import time
from pathlib import Path
from typing import Callable, Optional

import yaml

from .param_map import get_param_description


def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten nested dict into dot-separated keys."""
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep))
        else:
            items[new_key] = v
    return items


def _diff_configs(old_config: dict, new_config: dict) -> list:
    """Find changed parameters between two configs.

    Returns:
        List of (param_path, old_value, new_value) tuples
    """
    old_flat = _flatten_dict(old_config)
    new_flat = _flatten_dict(new_config)

    changes = []
    all_keys = set(old_flat.keys()) | set(new_flat.keys())
    for key in sorted(all_keys):
        old_val = old_flat.get(key)
        new_val = new_flat.get(key)
        if old_val != new_val:
            changes.append((key, old_val, new_val))
    return changes


class ConfigWatcher:
    """Watches a YAML config file and detects changes.

    Usage:
        watcher = ConfigWatcher("config.yaml")

        # In your main loop:
        changed, new_config = watcher.check()
        if changed:
            # new_config contains the updated configuration
            retargeter.reload(new_config)
    """

    def __init__(
        self,
        config_path: str,
        poll_interval: float = 0.5,
        on_change: Optional[Callable] = None,
        verbose: bool = True,
    ):
        """Initialize config watcher.

        Args:
            config_path: Path to YAML configuration file
            poll_interval: How often to check for changes (seconds)
            on_change: Optional callback(new_config, changes) called on change
            verbose: Print change details to terminal
        """
        self.config_path = Path(config_path).resolve()
        self.poll_interval = poll_interval
        self.on_change = on_change
        self.verbose = verbose

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        self._last_mtime = self.config_path.stat().st_mtime
        self._last_check_time = time.time()
        self._current_config = self._load_config()

    def _load_config(self) -> dict:
        """Load and return config from YAML file."""
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f) or {}

    @property
    def config(self) -> dict:
        """Current configuration dict."""
        return self._current_config

    def check(self) -> tuple:
        """Check if config file has changed.

        Returns:
            (changed: bool, config: dict)
            - changed is True if the file was modified since last check
            - config is the current (possibly updated) configuration
        """
        now = time.time()
        if now - self._last_check_time < self.poll_interval:
            return False, self._current_config

        self._last_check_time = now

        try:
            mtime = self.config_path.stat().st_mtime
        except OSError:
            return False, self._current_config

        if mtime <= self._last_mtime:
            return False, self._current_config

        # File changed - reload
        self._last_mtime = mtime
        try:
            new_config = self._load_config()
        except Exception as e:
            if self.verbose:
                print(f"[ConfigWatcher] Failed to parse config: {e}")
            return False, self._current_config

        old_config = self._current_config
        self._current_config = new_config

        if self.verbose:
            self._print_changes(old_config, new_config)

        if self.on_change is not None:
            changes = _diff_configs(old_config, new_config)
            try:
                self.on_change(new_config, changes)
            except Exception as e:
                if self.verbose:
                    print(f"[ConfigWatcher] on_change callback failed: {e}")

        return True, new_config

    def _print_changes(self, old_config: dict, new_config: dict):
        """Print changed parameters to terminal with tuning guidance."""
        changes = _diff_configs(old_config, new_config)
        if not changes:
            print("[ConfigWatcher] File timestamp changed but no parameter differences detected.")
            return

        print(f"\n{'=' * 60}")
        print(f"[ConfigWatcher] Config reloaded: {self.config_path.name}")
        print(f"{'=' * 60}")

        for param_path, old_val, new_val in changes:
            # Skip non-retarget params
            if not any(param_path.startswith(p) for p in ("retarget.", "video_input.")):
                continue

            # Strip "retarget." prefix for param_map lookup
            lookup_key = param_path
            if param_path.startswith("retarget."):
                lookup_key = param_path[len("retarget."):]

            desc = get_param_description(lookup_key)
            print(f"  {param_path}: {old_val} -> {new_val}")
            if desc:
                print(f"    {desc}")
        print(f"{'=' * 60}\n")
