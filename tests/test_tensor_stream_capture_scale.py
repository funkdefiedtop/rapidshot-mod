"""`TensorStream` with ``capture_scale``: the region follows the setting.

Its own module because it needs its own camera: the converter tests hold a
live frame for the whole module, and a camera refuses to acquire while one is
outstanding. The motion source keeps frames coming on an otherwise idle desktop.
"""

import pytest

import rapidshot
from rapidshot import native

pytestmark = pytest.mark.skipif(
    not native.is_available(), reason="native extension not built"
)


@pytest.fixture(scope="module")
def camera(motion):
    cam = rapidshot.create(output_color="BGRA")
    yield cam
    cam.release()


def centred(frame, width, height):
    left = (frame.width - width) // 2
    top = (frame.height - height) // 2
    return left, top, left + width, top + height


def test_capture_scale_sets_and_changes_the_region(camera):
    with rapidshot.TensorStream(
        camera, (288, 384), sampling="area", capture_scale=4 / 3, timeout=10
    ) as stream:
        assert stream.capture_scale == pytest.approx(4 / 3)
        assert stream.capture_region is None

        tensor = next(stream)
        assert tensor.shape == (1, 3, 384, 288)
        assert stream.capture_region == centred(stream.frame, 384, 512)

        stream.capture_scale = 1.0
        next(stream)
        assert stream.capture_region == centred(stream.frame, 288, 384)
        left, top, _, _ = stream.capture_region
        assert stream.to_frame(0, 0) == (left, top)

        with pytest.raises(ValueError, match="capture_scale"):
            stream.capture_scale = 0.5
        assert stream.capture_scale == 1.0

        # Not a valid scale for 288x384: rounds up to the next, 4/3.
        stream.capture_scale = 1.3
        assert stream.capture_size == (384, 512)
        assert stream.capture_scale == pytest.approx(4 / 3)
        next(stream)
        assert stream.capture_region == centred(stream.frame, 384, 512)
