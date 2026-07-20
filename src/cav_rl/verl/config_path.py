from __future__ import annotations

from pathlib import Path


def resolve_verl_config_dir(explicit: str | None = None) -> Path:
    """Locate Hydra configs from an installed `verl` package.

    No sibling repositories (T3/EnvRL/HiPER) are required. Override with an
    explicit path when needed.
    """
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"VERL config directory not found: {path}")
        return path

    try:
        import verl
    except ImportError as exc:  # pragma: no cover - runtime dependency hint
        raise ImportError(
            "Optional veRL backend requires the `verl` package. "
            "Install with `pip install 'cav-rl[verl]'` or `pip install verl`."
        ) from exc

    path = Path(verl.__file__).resolve().parent / "trainer" / "config"
    if not path.is_dir():
        raise FileNotFoundError(
            f"Installed verl package has no trainer/config directory at {path}. "
            "Upgrade verl or pass --config-path explicitly."
        )
    return path
