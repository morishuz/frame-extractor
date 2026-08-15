from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from frame_extractor import output


def test_run_directories_are_unique_within_the_same_second(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FixedDatetime:
        @staticmethod
        def now() -> datetime:
            return datetime(2026, 8, 15, 12, 34, 56)

    monkeypatch.setattr(output, "datetime", FixedDatetime)

    first = output.make_run_paths(str(tmp_path), save_debug_video=False)
    second = output.make_run_paths(str(tmp_path), save_debug_video=False)

    assert first.run_dir.name == "20260815_123456"
    assert second.run_dir.name == "20260815_123456_01"
    assert first.run_dir != second.run_dir
