from __future__ import annotations

import numpy as np
from vizdoom_turbo._vizdoom_turbo import preprocess_into


def test_area_resize_grayscale_and_maxpool_are_batched() -> None:
    current = np.asarray(
        [
            [
                [[0, 0, 0], [40, 80, 120]],
                [[80, 40, 0], [255, 255, 255]],
            ],
            [
                [[10, 20, 30], [10, 20, 30]],
                [[10, 20, 30], [10, 20, 30]],
            ],
        ],
        dtype=np.uint8,
    )
    previous = np.zeros_like(current)
    previous[1] = 200
    output = np.empty((2, 1, 1, 1), dtype=np.uint8)

    preprocess_into(current, output, [0, 0, 0, 0], False, 0, "area", previous)

    pooled_lane_zero = current[0].mean(axis=(0, 1))
    expected_zero = int(
        round(
            (
                pooled_lane_zero[0] * 77
                + pooled_lane_zero[1] * 150
                + pooled_lane_zero[2] * 29
            )
            / 256
        )
    )
    assert output[:, 0, 0, 0].tolist() == [expected_zero, 200]


def test_crop_mask_preserves_geometry_and_uses_fill() -> None:
    current = np.full((1, 4, 4, 3), 100, dtype=np.uint8)
    output = np.empty_like(current)

    preprocess_into(current, output, [1, 1, 1, 1], True, 7, "nearest")

    assert np.all(output[:, 0] == 7)
    assert np.all(output[:, -1] == 7)
    assert np.all(output[:, :, 0] == 7)
    assert np.all(output[:, :, -1] == 7)
    assert np.all(output[:, 1:3, 1:3] == 100)
