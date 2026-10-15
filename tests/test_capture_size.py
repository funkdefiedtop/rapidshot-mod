"""`capture_size`: the region a `capture_scale` captures. Pure arithmetic, no GPU.

The two invariants every result must hold -- the output's aspect ratio
exactly, and both sides multiples of ``align`` -- are checked over a sweep, not
only at hand-picked values, because the failure that matters is a size that
breaks one of them by a pixel.
"""

import itertools
import math

import pytest

from rapidshot.converter import capture_size


@pytest.mark.parametrize(
    "scale, expected",
    [
        (1.0, (288, 384)),
        (1.01, (384, 512)),   # the next valid scale up, not the nearest
        (1.3, (384, 512)),
        (4 / 3, (384, 512)),  # exact scales land on themselves despite float noise
        (1.34, (480, 640)),
        (5 / 3, (480, 640)),
        (1.67, (576, 768)),
        (2.0, (576, 768)),
    ],
)
def test_288x384_steps_by_a_third(scale, expected):
    assert capture_size((288, 384), scale) == expected


@pytest.mark.parametrize(
    "size, scale, expected",
    [
        ((320, 320), 1.05, (352, 352)),   # steps of 1/10
        ((640, 640), 1.3, (832, 832)),    # steps of 1/20, 1.3 itself valid
        ((300, 300), 1.0, (320, 320)),    # an unaligned model size rounds up too
        ((288, 384), 1.3, (384, 512)),
    ],
)
def test_other_sizes(size, scale, expected):
    assert capture_size(size, scale) == expected


def test_align_1_keeps_only_the_ratio():
    # 96 * 1.3 = 124.8 -> 125 units of 3:4.
    assert capture_size((288, 384), 1.3, align=1) == (375, 500)


@pytest.mark.parametrize(
    "size, align",
    list(itertools.product([(288, 384), (320, 320), (640, 384), (416, 256), (300, 200)], [1, 16, 32])),
)
def test_invariants_over_a_sweep(size, align):
    width, height = size
    for hundredths in range(100, 301):
        scale = hundredths / 100
        w, h = capture_size(size, scale, align)
        assert w * height == h * width, f"ratio broken at {scale}: {w}x{h}"
        assert w % align == 0 and h % align == 0, f"unaligned at {scale}: {w}x{h}"
        assert w >= width * scale - 1e-6, f"rounded down at {scale}: {w}x{h}"
        # And it is the smallest such region: one step smaller is too small.
        unit = w // math.gcd(w, h)
        step_w = unit * math.lcm(*(align // math.gcd(u, align)
                                   for u in (width // math.gcd(width, height),
                                             height // math.gcd(width, height))))
        assert w - step_w < width * scale - 1e-6 or w - step_w <= 0


@pytest.mark.parametrize("scale", [0.5, 0, -1, float("nan"), float("inf"), "big", True])
def test_bad_scales_are_refused(scale):
    with pytest.raises(ValueError, match="capture_scale"):
        capture_size((288, 384), scale)


@pytest.mark.parametrize("align", [0, -32, 1.5, True])
def test_bad_aligns_are_refused(align):
    with pytest.raises(ValueError, match="align"):
        capture_size((288, 384), 1.3, align)
