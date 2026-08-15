from __future__ import annotations

import numpy as np

from frame_extractor.config import parse_config
from frame_extractor.tracking import FlowStepDiagnostics
from frame_extractor.tracking import TrackingState
from frame_extractor.tracking import _beyond_lost_border
from frame_extractor.tracking import _bilinear_sample_flow
from frame_extractor.tracking import _clip_vectors
from frame_extractor.tracking import _inside_image
from frame_extractor.tracking import compute_frame_scores
from frame_extractor.tracking import initialize_tracking_state
from frame_extractor.tracking import step_tracking


class FixedFlowSolver:
    def __init__(self, flow: np.ndarray) -> None:
        self.flow = flow
        self.calls: list[tuple[np.ndarray, np.ndarray, None]] = []

    def calc(
        self,
        previous: np.ndarray,
        current: np.ndarray,
        initial_flow: None,
    ) -> np.ndarray:
        self.calls.append((previous, current, initial_flow))
        return self.flow


def test_bilinear_sample_flow_interpolates_vector_components() -> None:
    flow = np.array(
        [
            [[0.0, 0.0], [2.0, 0.0]],
            [[0.0, 4.0], [2.0, 4.0]],
        ],
        dtype=np.float32,
    )

    sampled, valid = _bilinear_sample_flow(
        flow,
        np.array([[0.25, 0.75]], dtype=np.float32),
    )

    np.testing.assert_allclose(sampled, [[0.5, 3.0]])
    np.testing.assert_array_equal(valid, [True])


def test_bilinear_sample_flow_accepts_edges_and_rejects_invalid_points() -> None:
    flow = np.array(
        [
            [[1.0, 10.0], [2.0, 20.0]],
            [[3.0, 30.0], [4.0, 40.0]],
        ],
        dtype=np.float32,
    )
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [-0.01, 0.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [np.nan, 0.0],
            [0.0, np.inf],
        ],
        dtype=np.float32,
    )

    sampled, valid = _bilinear_sample_flow(flow, points)

    np.testing.assert_allclose(sampled[:2], [[1.0, 10.0], [4.0, 40.0]])
    np.testing.assert_array_equal(valid, [True, True, False, False, False, False, False])


def test_clip_vectors_caps_norms_without_changing_direction_or_input() -> None:
    vectors = np.array(
        [[3.0, 4.0], [-6.0, 8.0], [1.0, 1.0], [0.0, 0.0]],
        dtype=np.float64,
    )
    original = vectors.copy()

    clipped = _clip_vectors(vectors, max_norm=5.0)

    np.testing.assert_array_equal(vectors, original)
    np.testing.assert_allclose(clipped, [[3.0, 4.0], [-3.0, 4.0], [1.0, 1.0], [0.0, 0.0]])
    assert clipped.dtype == np.float32
    assert np.all(np.linalg.norm(clipped, axis=1) <= 5.0)
    assert np.dot(vectors[1], clipped[1]) > 0.0
    cross_product = vectors[1, 0] * clipped[1, 1] - vectors[1, 1] * clipped[1, 0]
    assert np.isclose(cross_product, 0.0)


def test_clip_vectors_is_disabled_by_a_nonpositive_limit_and_still_returns_a_copy() -> None:
    vectors = np.array([[3.0, 4.0]], dtype=np.float64)

    for max_norm in (0.0, -1.0):
        result = _clip_vectors(vectors, max_norm=max_norm)

        np.testing.assert_array_equal(result, vectors)
        assert result.dtype == np.float32
        assert not np.shares_memory(result, vectors)


def test_image_and_lost_border_masks_have_distinct_boundaries() -> None:
    points = np.array(
        [
            [0.0, 0.0],
            [9.0, 7.0],
            [-0.01, 3.0],
            [9.01, 3.0],
            [-2.0, 3.0],
            [11.0, 3.0],
            [-2.01, 3.0],
            [11.01, 3.0],
            [3.0, -2.01],
            [3.0, 9.01],
            [np.nan, 3.0],
        ],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(
        _inside_image(points, width=10, height=8),
        [True, True, False, False, False, False, False, False, False, False, False],
    )
    np.testing.assert_array_equal(
        _beyond_lost_border(points, width=10, height=8, lost_border_px=2.0),
        [False, False, False, False, False, False, True, True, True, True, True],
    )


def test_tracking_grid_preserves_original_pixel_placement_when_downsampled() -> None:
    full_config = parse_config(
        {
            "n_downsample": 0,
            "sampling": {
                "grid_step_original_px": 160,
                "min_margin_original_px": 16,
            },
        }
    )
    quarter_config = parse_config(
        {
            "n_downsample": 2,
            "sampling": {
                "grid_step_original_px": 160,
                "min_margin_original_px": 16,
            },
        }
    )

    full_state = initialize_tracking_state(np.zeros((480, 640), dtype=np.uint8), full_config)
    quarter_state = initialize_tracking_state(np.zeros((120, 160), dtype=np.uint8), quarter_config)

    assert full_state.origin_points.shape == (12, 2)
    np.testing.assert_array_equal(full_state.origin_points[0], [16.0, 16.0])
    np.testing.assert_array_equal(full_state.origin_points[-1], [496.0, 336.0])
    np.testing.assert_array_equal(quarter_state.origin_points * 4.0, full_state.origin_points)
    np.testing.assert_array_equal(full_state.current_points, full_state.origin_points)
    np.testing.assert_array_equal(full_state.alive_mask, np.ones(12, dtype=bool))


def test_compute_frame_scores_uses_valid_displacements_in_original_pixels() -> None:
    config = parse_config(
        {
            "n_downsample": 2,
            "scoring": {"percentile": 50.0},
        }
    )
    state = TrackingState(
        origin_points=np.zeros((4, 2), dtype=np.float32),
        current_points=np.array(
            [
                [0.0, 1.0],  # Valid: 1 processing px = 4 original px.
                [0.0, 2.0],  # Excluded because it is out of bounds.
                [0.0, 3.0],  # Excluded because the point is no longer alive.
                [0.0, 4.0],  # Valid: 4 processing px = 16 original px.
            ],
            dtype=np.float32,
        ),
        alive_mask=np.array([True, True, False, True]),
    )
    diagnostics = FlowStepDiagnostics(
        in_bounds_mask=np.array([True, False, True, True])
    )

    scores = compute_frame_scores(
        state,
        diagnostics,
        frame_index=17,
        timestamp_sec=1.25,
        config=config,
    )

    assert scores.frame_index == 17
    assert scores.timestamp_sec == 1.25
    assert scores.global_score == 10.0
    assert scores.in_bounds_points == 2
    assert scores.in_bounds_ratio == 0.5


def test_step_tracking_applies_fixed_flow_clipping_and_lost_border() -> None:
    config = parse_config(
        {
            "n_downsample": 0,
            "max_step_norm_original_px": 2.0,
            "sampling": {"lost_border_original_px": 1.0},
        }
    )
    points = np.array([[1.0, 1.0], [6.0, 5.0]], dtype=np.float32)
    state = TrackingState(
        origin_points=points.copy(),
        current_points=points.copy(),
        alive_mask=np.array([True, True]),
    )
    flow = np.empty((6, 7, 2), dtype=np.float32)
    flow[...] = [3.0, 4.0]
    solver = FixedFlowSolver(flow)
    previous = np.zeros((6, 7), dtype=np.uint8)
    current = np.ones((6, 7), dtype=np.uint8)

    diagnostics = step_tracking(state, previous, current, solver, config)  # type: ignore[arg-type]

    assert len(solver.calls) == 1
    assert solver.calls[0][0] is previous
    assert solver.calls[0][1] is current
    assert solver.calls[0][2] is None
    np.testing.assert_allclose(state.current_points, [[2.2, 2.6], [7.2, 6.6]])
    np.testing.assert_array_equal(diagnostics.in_bounds_mask, [True, False])
    np.testing.assert_array_equal(state.alive_mask, [True, False])
