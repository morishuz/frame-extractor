from __future__ import annotations

import pytest

from frame_extractor import cli


@pytest.mark.parametrize(
    "arguments",
    [
        ["--start-frame", "-1"],
        ["--max-frames", "0"],
        ["--max-frames", "-1"],
        ["--duration-frames", "0"],
    ],
)
def test_cli_rejects_invalid_processing_ranges(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.build_parser().parse_args(["input.mp4", *arguments])

    assert exc_info.value.code == 2


def test_cli_reports_conflicting_frame_limits_as_an_argument_error() -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "input.mp4",
                "--max-frames",
                "10",
                "--duration-frames",
                "20",
            ]
        )

    assert exc_info.value.code == 2


def test_cli_accepts_matching_frame_limit_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, int | None] = {}

    def fake_run_experiment(*_args, **kwargs) -> None:
        observed["max_frames"] = kwargs["max_frames"]

    monkeypatch.setattr(cli, "run_experiment", fake_run_experiment)

    cli.main(
        [
            "input.mp4",
            "--max-frames",
            "10",
            "--duration-frames",
            "10",
        ]
    )

    assert observed["max_frames"] == 10
