from __future__ import annotations

from src.config import get_settings


def main() -> None:
    """Entry point for the `sicilian-word-grouping` console script.

    Currently a placeholder; the full ETVL pipeline orchestration will be wired
    here in a later layer. Verifies configuration loads at startup.
    """
    settings = get_settings()
    print(
        f"sicilian-word-grouping ready: volume={settings.volume} "
        f"model={settings.model} layout={settings.column_layout}"
    )


if __name__ == "__main__":
    main()