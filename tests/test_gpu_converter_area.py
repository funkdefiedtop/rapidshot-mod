"""Area sampling and ``capture_scale`` for `GpuConverter`.

**The reference is exact.** ``area_reference`` computes the overlap-weighted
mean of a crop in float64 from the captured bytes (read back at identity size
with nearest sampling, as ``test_gpu_converter_crop.py`` does). The shader's
weights are rational with integer boundaries, so the only error left is float32
accumulation: float output must match to ~1e-6 and BGRA8 to one code.

``cv2.INTER_AREA`` is checked separately and only when OpenCV is installed. It
is the promise made to anyone building a dataset with OpenCV, so it gets its
own test rather than standing in for the exact reference.
"""

import numpy as np
import pytest

import rapidshot
from rapidshot import native

pytestmark = pytest.mark.skipif(
    not native.is_available(), reason="native extension not built"
)


@pytest.fixture(scope="module")
def live_frame():
    camera = rapidshot.create(output_color="BGRA")
    frame = None
    for _ in range(600):
        frame = camera.grab_frame()
        if frame is not None:
            break
    if frame is None:
        camera.release()
        pytest.skip("no frame captured — the screen must be changing")
    yield frame
    frame.release()
    camera.release()


@pytest.fixture(scope="module")
def surface(live_frame):
    """(H, W, 4) BGRA of the whole frame: the captured bytes."""
    return bgra(live_frame, (live_frame.width, live_frame.height), "nearest")


def bgra(frame, size, sampling, crop=None, **options):
    converter = rapidshot.GpuConverter(
        frame, size, dtype="uint8", layout="nhwc", sampling=sampling, **options
    )
    return converter.process(frame, crop=crop).numpy()[0]


def busy_crop(surface, width, height):
    """(left, top, right, bottom) of the highest-variance window, or a skip."""
    h, w = surface.shape[:2]
    best, best_std = None, -1.0
    grey = surface[..., :3].astype(np.float32).mean(axis=-1)
    for top in range(1, h - height, max(1, height // 2)):
        for left in range(1, w - width, max(1, width // 2)):
            std = grey[top:top + height, left:left + width].std()
            if std > best_std:
                best, best_std = (left | 1, top | 1), std
    if best is None or best_std < 8.0:
        pytest.skip("screen content too flat to test resampling")
    left, top = best
    left = min(left, w - width)
    top = min(top, h - height)
    return left, top, left + width, top + height


def texels(surface, crop):
    left, top, right, bottom = crop
    return surface[top:bottom, left:right]


def area_weights(src, out):
    """(out, src) matrix: row x holds each texel's share of output pixel x."""
    weights = np.zeros((out, src))
    for x in range(out):
        a, b = x * src, (x + 1) * src
        for i in range(a // out, (b - 1) // out + 1):
            weights[x, i] = min((i + 1) * out, b) - max(i * out, a)
    return weights / src


def area_reference(patch, out_w, out_h):
    """Overlap-weighted mean of *patch*, in float64. Separable, so two matmuls."""
    h, w = patch.shape[:2]
    wy, wx = area_weights(h, out_h), area_weights(w, out_w)
    rows = np.tensordot(wy, patch.astype(np.float64), axes=(1, 0))   # (out_h, w, c)
    return np.einsum("xw,ywc->yxc", wx, rows, optimize=True)


# --------------------------------------------------------------------------
# the filter
# --------------------------------------------------------------------------


def test_identity_size_area_is_exactly_the_slice(live_frame, surface):
    crop = busy_crop(surface, 96, 64)
    np.testing.assert_array_equal(
        bgra(live_frame, (96, 64), "area", crop), texels(surface, crop)
    )


def test_integer_reduction_is_the_block_mean(live_frame, surface):
    crop = busy_crop(surface, 192, 128)
    got = bgra(live_frame, (96, 64), "area", crop).astype(np.float64)
    patch = texels(surface, crop).astype(np.float64)
    expected = patch.reshape(64, 2, 96, 2, 4).mean(axis=(1, 3))
    assert np.abs(got - expected).max() <= 0.5 + 1e-6


@pytest.mark.parametrize(
    "crop_size, out_size",
    [
        ((384, 512), (288, 384)),   # 4:3, the model-input case
        ((374, 499), (288, 384)),   # x1.3, unequal ratios per axis
        ((500, 300), (96, 64)),     # past 4x, several texels per output
        ((97, 65), (96, 64)),       # barely reduced
    ],
)
def test_area_matches_the_exact_reference(live_frame, surface, crop_size, out_size):
    crop = busy_crop(surface, *crop_size)
    expected = area_reference(texels(surface, crop), *out_size)
    got = bgra(live_frame, out_size, "area", crop).astype(np.float64)
    # Rounded to a code from a float32 accumulation of an exact mean.
    assert np.abs(got - expected).max() <= 0.5 + 1e-3


def test_float_output_matches_the_exact_reference(live_frame, surface):
    crop = busy_crop(surface, 384, 512)
    got = rapidshot.GpuConverter(
        live_frame, (288, 384), dtype="float32", sampling="area"
    ).process(live_frame, crop=crop).numpy()[0]
    mean = area_reference(texels(surface, crop), 288, 384) / 255.0
    expected = np.stack([mean[..., 2], mean[..., 1], mean[..., 0]])
    np.testing.assert_allclose(got, expected, atol=2e-6)


def test_area_matches_opencv_inter_area(live_frame, surface):
    cv2 = pytest.importorskip("cv2")
    for crop_size in ((384, 512), (374, 499)):
        crop = busy_crop(surface, *crop_size)
        patch = np.ascontiguousarray(texels(surface, crop))
        expected = cv2.resize(patch, (288, 384), interpolation=cv2.INTER_AREA)
        got = bgra(live_frame, (288, 384), "area", crop)
        diff = np.abs(got.astype(np.int64) - expected.astype(np.int64))
        assert diff.max() <= 1, f"{crop_size}: off by up to {diff.max()} codes"


def test_area_differs_from_bilinear_at_4_to_3(live_frame, surface):
    """Guards against the area define silently compiling as bilinear."""
    crop = busy_crop(surface, 384, 512)
    assert not np.array_equal(
        bgra(live_frame, (288, 384), "area", crop),
        bgra(live_frame, (288, 384), "bilinear", crop),
    )


def test_enlarging_falls_back_to_bilinear(live_frame, surface):
    crop = busy_crop(surface, 24, 16)
    np.testing.assert_array_equal(
        bgra(live_frame, (96, 64), "area", crop),
        bgra(live_frame, (96, 64), "bilinear", crop),
    )


def test_sampling_reports_area(live_frame):
    assert rapidshot.GpuConverter(live_frame, (32, 32), sampling="area").sampling == "area"


# --------------------------------------------------------------------------
# capture_scale
# --------------------------------------------------------------------------


def centred(frame, width, height):
    left = (frame.width - width) // 2
    top = (frame.height - height) // 2
    return left, top, left + width, top + height


@pytest.mark.parametrize(
    "scale, region_size",
    [(1.0, (288, 384)), (4 / 3, (384, 512)), (1.3, (384, 512)), (1.34, (480, 640))],
)
def test_capture_scale_is_a_centred_crop(live_frame, scale, region_size):
    if region_size[0] > live_frame.width or region_size[1] > live_frame.height:
        pytest.skip("frame too small for this region")
    region = centred(live_frame, *region_size)
    converter = rapidshot.GpuConverter(
        live_frame, (288, 384), dtype="uint8", layout="nhwc",
        sampling="area", capture_scale=scale,
    )
    got = converter.process(live_frame).numpy()[0]
    assert converter.capture_region == region
    assert converter.capture_size == region_size
    assert converter.capture_scale == region_size[0] / 288
    np.testing.assert_array_equal(got, bgra(live_frame, (288, 384), "area", region))


def test_capture_scale_can_change_between_calls(live_frame):
    converter = rapidshot.GpuConverter(live_frame, (96, 64), sampling="area", capture_scale=1.0)
    converter.process(live_frame)
    assert converter.capture_region == centred(live_frame, 96, 64)
    converter.capture_scale = 2.0
    converter.process(live_frame)
    assert converter.capture_region == centred(live_frame, 192, 128)
    converter.capture_scale = None
    converter.process(live_frame)
    assert converter.capture_region == (0, 0, live_frame.width, live_frame.height)


def test_per_call_crop_overrides_capture_scale(live_frame):
    converter = rapidshot.GpuConverter(live_frame, (32, 32), sampling="area", capture_scale=1.5)
    converter.process(live_frame, crop=(0, 0, 64, 64))
    assert converter.capture_region == (0, 0, 64, 64)
    converter.process(live_frame)
    # 1.5 x 32 = 48 is not a multiple of 32, so the region rounds up to 64.
    assert converter.capture_region == centred(live_frame, 64, 64)
    assert converter.capture_scale == 2.0


def test_to_frame_maps_corners_onto_the_region(live_frame):
    converter = rapidshot.GpuConverter(live_frame, (288, 384), sampling="area", capture_scale=1.3)
    with pytest.raises(RuntimeError, match="process"):
        converter.to_frame(0, 0)
    converter.process(live_frame)
    left, top, right, bottom = converter.capture_region
    assert converter.to_frame(0, 0) == (left, top)
    assert converter.to_frame(288, 384) == pytest.approx((right, bottom))
    assert converter.to_frame(144, 192) == pytest.approx(((left + right) / 2, (top + bottom) / 2))


@pytest.mark.parametrize("scale", [0.5, 0.999, 0, -1, float("nan"), float("inf"), "big", True])
def test_bad_scales_are_refused(live_frame, scale):
    with pytest.raises(ValueError, match="capture_scale"):
        rapidshot.GpuConverter(live_frame, (32, 32), capture_scale=scale)
    converter = rapidshot.GpuConverter(live_frame, (32, 32))
    with pytest.raises(ValueError, match="capture_scale"):
        converter.capture_scale = scale


def test_a_region_larger_than_the_frame_is_refused_not_shrunk(live_frame):
    converter = rapidshot.GpuConverter(live_frame, (288, 384), capture_scale=100.0)
    with pytest.raises(ValueError, match="does not fit"):
        converter.process(live_frame)


def test_crop_and_capture_scale_together_are_refused(live_frame):
    with pytest.raises(ValueError, match="crop or capture_scale"):
        rapidshot.GpuConverter(live_frame, (32, 32), crop=(0, 0, 64, 64), capture_scale=1.5)


def test_multi_region_call_has_no_single_region(live_frame):
    converter = rapidshot.GpuConverter(live_frame, (32, 32), batch=2, capture_scale=1.0)
    converter.process(live_frame)
    converter.process(live_frame, regions=[(0, 0, 64, 64), (64, 0, 128, 64)])
    assert converter.capture_region is None
