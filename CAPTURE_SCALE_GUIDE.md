# Capture scale: integration guide

**Audience: an LLM/coding agent wiring this fork of RapidShot into an
application or a training pipeline.** Read it end to end before writing code.
Every number below was produced by `rapidshot.capture_size` in this repository,
not by hand. If something here disagrees with the code, the code wins — check
`rapidshot/converter.py` and the tests named in §12.

---

## 1. What the feature does

A model takes a fixed input, e.g. 288×384 (width×height). Instead of capturing
exactly 288×384 screen pixels, the application captures a **larger region
around the centre of the screen** and **reduces it on the GPU** to the model
input. The model therefore sees more of the screen ("zoom out") at the same
input size and the same inference cost.

```
monitor frame (e.g. 1920×1080, physical pixels)
   │  1. centred region, exactly the input's aspect ratio  ← capture_size(size, scale)
   ▼
region 432×576            (never seen by the model)
   │  2. area filter on the GPU (= cv2.INTER_AREA)          ← sampling="area"
   ▼
tensor (1, 3, 384, 288)   ← the only thing the model sees; exactly `size`
```

Nothing about sizes, aspect ratios or filters is hard-coded in the library.
The application passes:

| parameter | meaning | who decides |
|---|---|---|
| `size=(width, height)` | the model input = the produced tensor, **exactly**, never rounded | the application (from the model config) |
| `sampling="area"` | reduction filter | the application — always `"area"` for this feature |
| `capture_scale=s` | region size / input size, `s >= 1.0` | the application (a setting) |
| `capture_align=a` | both *region* sides are multiples of `a`; default `1` | leave at default (see §4) |

---

## 2. Hard invariants — never violate these

1. **The tensor is exactly `size`.** If the model needs multiples of 32 (stride-32
   backbones), the *application* must pass such a `size`. The library does not
   check it, because it is a property of the model, not of capture. Assert it
   yourself (§7).
2. **The region has exactly the aspect ratio of `size`.** The library
   guarantees it: the region is always `n × (unit_w, unit_h)` where
   `(unit_w, unit_h)` is `size` divided by `gcd(width, height)`. Because of
   this, the reduction is uniform on both axes and objects are not distorted.
3. **No letterboxing, no padding, no stretching.** None of these occur anywhere
   in this path. Do not add any.
4. **The region is never smaller than requested.** The scale is rounded *up* to
   the nearest exact one, never down. The *effective* scale is
   `capture_size[0] / size[0]`; read it back from `.capture_scale`.
5. **The region is centred on the frame:** `left = (W - w) // 2`,
   `top = (H - h) // 2` (integer floor division). Datasets must use the same
   formula (§8).
6. **Never enlarges.** `capture_scale < 1.0` is refused with `ValueError`.
   A region larger than the frame is refused at `process()` time with
   `ValueError("... does not fit ...")` — it is never silently shrunk, because
   that would change the scale the model was trained at.
7. **Training and inference must produce the same pixels.** Same region size,
   same centring, same filter (area / `cv2.INTER_AREA`), same channel order,
   same value range. §8 gives the recipe.

---

## 3. The math (exact, so you can reproduce it anywhere)

`rapidshot.capture_size(size, scale, align=1) -> (region_w, region_h)`:

```python
import math

def capture_size(size, scale, align=1):
    width, height = size
    g = math.gcd(width, height)
    unit_w, unit_h = width // g, height // g           # smallest exact-ratio rectangle
    step = math.lcm(align // math.gcd(unit_w, align),  # n must keep both sides
                    align // math.gcd(unit_h, align))  # multiples of `align`
    n = math.ceil(g * scale / step - 1e-9) * step      # round UP; 1e-9 absorbs float noise
    return unit_w * n, unit_h * n                       # effective scale = n / g
```

This is a faithful copy of the library function (validation omitted). Prefer
calling `rapidshot.capture_size` directly; use the copy only where `rapidshot`
cannot be imported (e.g. a Linux training box) and keep a test asserting both
agree over a sweep of scales.

Consequences:

- With the default `align=1`, effective scales step by **`1 / gcd(width, height)`**:
  - 288×384 → gcd 96 → step 1/96 ≈ 0.0104
  - 384×512 → gcd 128 → step 1/128 ≈ 0.0078
  - 576×768 → gcd 192 → step 1/192 ≈ 0.0052
  - 640×640 → gcd 640 → step 1/640
- A scale that is a multiple of the step is used **exactly** (no rounding).
  Everything else rounds up to the next multiple.

---

## 4. `capture_align` — leave it at 1

`capture_align` constrains the **region on screen**, not the tensor. The model
never sees the region, so its multiple is irrelevant to the model. Raising
`capture_align` only coarsens the available scales:

| align (288×384 input) | region grid | scale step | request 1.5 → region (effective) |
|---|---|---|---|
| **1 (default)** | 3k×4k | 1/96 | **432×576 (1.5)** |
| 8 | 24k×32k | 1/12 | 432×576 (1.5) |
| 16 | 48k×64k | 1/6 | 432×576 (1.5) |
| 32 | 96k×128k | 1/3 | 480×640 (1.667) — rounded up |

Do **not** set `capture_align=32` "because the model needs 32". That
requirement is satisfied by `size`. (Before this fork's latest change the
default was 32; any code or notes assuming 384×512 for `capture_scale=1.3` on a
288×384 input are from that era — today 1.3 gives 375×500.)

Fractional reduction ratios (e.g. 375→288) are fine: the area filter weights
every covered source pixel by its exact overlap, computed in integers on the
GPU. Integer ratios (2.0) are marginally "cleaner" (each output pixel is a
plain 2×2 mean), but as long as the dataset is built the same way (§8) the
model sees identical statistics.

---

## 5. The three-model setup in this project

The project ships three models, one per monitor resolution. Their inputs have
the same 3:4 shape and cover the same share of their monitor (1×, 4/3×, 2× of
the 1080p input). All three are multiples of 32.

| monitor | model input `size` | gcd → scale step |
|---|---|---|
| 1920×1080 | 288×384 | 96 → 1/96 |
| 2560×1440 | 384×512 | 128 → 1/128 |
| 3840×2160 | 576×768 | 192 → 1/192 |

**Keep the scale identical across the three models, and choose it as a
multiple of 1/32.** 1/32 is the coarsest step common to all three (96, 128 and
192 are all multiples of 32), so such scales are exact for every model and the
three capture precisely the same share of their screen. Otherwise rounding
differs per model, e.g. `1.3` → 1.3021 for 288×384 and 576×768, but 1.3047 for
384×512.

Exact scales common to all three: 1.0, 1.03125, 1.0625, …, **1.25, 1.5, 1.75,
2.0, 2.5**, … (`k / 32`).

| scale | 1080p: region of 1920×1080 | 1440p: region of 2560×1440 | 2160p: region of 3840×2160 |
|---|---|---|---|
| 1.0 | 288×384 | 384×512 | 576×768 |
| 1.25 | 360×480 | 480×640 | 720×960 |
| 1.5 | 432×576 | 576×768 | 864×1152 |
| 1.75 | 504×672 | 672×896 | 1008×1344 |
| 2.0 | 576×768 | 768×1024 | 1152×1536 |
| 2.5 | 720×960 | 960×1280 | 1440×1920 |
| **2.8125 (max)** | 810×1080 | 1080×1440 | 1620×2160 |

**Maximum scale is 2.8125 (= 90/32) for all three**: the region's height hits
the monitor height (384 × 2.8125 = 1080). Anything above fails at `process()`
with "does not fit" (e.g. 2.82 → 813×1084 on 1080p). Width never binds for a
3:4 region on a 16:9 monitor.

---

## 6. Installation — the native build is mandatory

`sampling="area"` lives in the Rust extension. The prebuilt
`rapidshot-native` wheel from PyPI **does not have it** and still reports
version 0.2.2, the same version string as this fork's build, so the version
cannot tell them apart.

The fork: <https://github.com/g-osypenko/rapidshot-mod> (branch `main`).
The built extension `rapidshot/_rapidshot_native.pyd` is git-ignored, so a
plain `pip install git+…` gives Python code **without** the area shader.

Correct setup (Windows, Rust toolchain installed):

```powershell
git clone https://github.com/g-osypenko/rapidshot-mod.git
cd rapidshot-mod
cd native; cargo build --release; cd ..
python native/install_dev.py          # copies the DLL to rapidshot/_rapidshot_native.pyd
pip install -e .                       # editable, so the .pyd in the package is used
```

The development `.pyd` inside the package is preferred over an installed
wheel. Confirm which one is loaded:

```python
import rapidshot.native
print(rapidshot.native.build_info()["source"])
# must be: "development build (rapidshot/_rapidshot_native)"
```

After pulling changes to `native/`, rebuild and re-run `install_dev.py`.

---

## 7. Inference

### 7.1 Recommended: `TensorStream`

```python
import rapidshot

MODELS = {                        # monitor height -> model config
    1080: dict(size=(288, 384), weights="model_1080.engine"),
    1440: dict(size=(384, 512), weights="model_1440.engine"),
    2160: dict(size=(576, 768), weights="model_2160.engine"),
}
CAPTURE_SCALE = 1.5               # one setting for all three; a multiple of 1/32

camera = rapidshot.create()       # full monitor, no region= (see §10)
cfg = MODELS[camera.height]       # KeyError = unsupported resolution: fail loudly
w, h = cfg["size"]
assert w % 32 == 0 and h % 32 == 0, cfg["size"]   # the model's requirement, checked here

stream = rapidshot.TensorStream(
    camera, cfg["size"],
    sampling="area",
    capture_scale=CAPTURE_SCALE,
    dtype="float16",              # or "float32"; must match the model
    layout="nchw",                # (1, 3, H, W)
    # bgr=False (default): RGB channel order
    # normalize=True (default): values 0..1
    timeout=None,                 # or seconds; idle screens yield no frames
)
assert abs(stream.capture_scale - CAPTURE_SCALE) < 1e-9, stream.capture_scale  # exact, not rounded

with stream:
    for tensor in stream:                     # GpuTensor, shape (1, 3, h, w)
        x = tensor.to_torch()                 # zero-copy CUDA tensor; aliases a reused buffer
        boxes = model(x)                      # boxes in tensor pixels: 0..w, 0..h
        for x0, y0, x1, y1 in boxes:
            fx0, fy0 = stream.to_frame(x0, y0)   # -> frame (= monitor) pixels
            fx1, fy1 = stream.to_frame(x1, y1)
            ...
```

Rules:

- **`to_frame()` must be called before the next iteration.** It maps through the
  region of the *latest* tensor (`stream.capture_region`).
- `to_frame` uses continuous coordinates: `(0, 0)` is the top-left *corner*
  of the tensor, `(w, h)` the bottom-right corner, so detector box corners map
  directly. It returns frame coordinates; for a full-monitor camera these are
  monitor pixels. If the camera has a `region=`, add its left/top offset.
- The tensor aliases a buffer that the next frame overwrites. `.clone()` it if it
  must outlive the iteration. `TensorStream` already syncs CUDA work before
  overwriting it (`sync=True`).
- Model-specific normalisation (mean/std) is applied by the application on the
  GPU after `to_torch()`. The library produces 0..1 RGB.
- Changing the scale at runtime: `stream.capture_scale = 1.75`. It is validated
  on assignment and takes effect on the next frame. Only change it if the model
  was trained for that scale (§9).
- Read back what is actually applied: `stream.capture_scale` (effective),
  `stream.capture_size` (region w, h), `stream.capture_region`
  (left, top, right, bottom of the latest tensor).

### 7.2 Manual: `GpuConverter`

```python
conv = rapidshot.GpuConverter(frame, (288, 384), sampling="area",
                              capture_scale=1.5, dtype="float16")
tensor = conv.process(frame)             # centred 432×576 -> (1, 3, 384, 288)
conv.capture_region                      # region of this call
conv.to_frame(x, y)                      # same semantics as above
```

Build the converter once and reuse it. Release each frame promptly
(`with camera.grab_frame() as frame:`); a held frame stalls capture.
`TensorStream` handles that, plus the other pitfalls listed in its docstring —
prefer it.

### 7.3 Do not combine

- `capture_scale` with `crop=` → `ValueError("give crop or capture_scale, not both")`.
- `capture_scale` with `regions=[...]` (multi-ROI): regions win and
  `capture_region` becomes `None`. Do not mix them.
- A per-call `process(frame, crop=...)` overrides `capture_scale` for that call
  only. That is the way to do an arbitrary moving ROI; it is a separate feature
  and the caller is then responsible for the region's aspect ratio.

---

## 8. Dataset preparation — must mirror inference

Source images: full screenshots at the **native resolution of the monitor the
model is for** (1080p screenshots for the 1080 model, etc.).

```python
import cv2
import rapidshot     # or the copy of capture_size from §3

def make_sample(shot_bgr, size, scale):
    """shot_bgr: full screenshot as loaded by cv2.imread (H, W, 3) BGR uint8.
    Returns the model-input image (RGB uint8, size) and the region used."""
    H, W = shot_bgr.shape[:2]
    w, h = rapidshot.capture_size(size, scale)      # the same function inference uses
    if w > W or h > H:
        raise ValueError(f"scale {scale} needs {w}x{h}, screenshot is {W}x{H}")
    left, top = (W - w) // 2, (H - h) // 2          # the same centring (floor)
    crop = shot_bgr[top:top + h, left:left + w]
    img = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)   # dsize is (width, height)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      # the library emits RGB by default
    return img, (left, top, w, h)
```

Then convert to float and divide by 255 (the library's 0..1), then apply the
same mean/std as at inference.

Labels (boxes in screenshot pixels → tensor pixels):

```python
def box_to_tensor(box, region, size):
    left, top, w, h = region
    sx, sy = size[0] / w, size[1] / h               # equal: same aspect ratio
    x0, y0, x1, y1 = box
    return ((x0 - left) * sx, (y0 - top) * sy, (x1 - left) * sx, (y1 - top) * sy)
# Clip to [0, size] and drop boxes that end up (mostly) outside the region.
```

Agreement: `sampling="area"` matches `cv2.INTER_AREA` to within one 8-bit code
on reduction (pinned by `tests/test_gpu_converter_area.py`, including
non-integer ratios such as 374×499 → 288×384); in a live check it was
bit-identical (§12, step 5). At scale 1.0 the area path is an identity and
reproduces the captured pixels bit-for-bit. Float outputs (`float16`/`float32`)
carry the unrounded mean, so they differ from `uint8 / 255` by under half a
code plus fp16 precision — irrelevant to a model, but don't assert exact
equality between the dataset's uint8 and the float tensor.

A model trained on 1:1 crops (scale 1.0) and run at `capture_scale=1.5` sees
every object 1.5× smaller than it was trained on. That costs accuracy and is
not a library bug. Retrain or augment (§9).

---

## 9. One scale vs. a range (augmentation)

- **Single scale:** generate the dataset at exactly the deployed
  `capture_scale`. Simplest and most accurate at that one scale.
- **Range:** to allow changing `capture_scale` later without retraining, sample
  the scale per training image. Sample the **integer `n`**, not a float, so
  every scale is exact and on the same grid inference uses:

```python
import math, random

def random_region(size, s_min, s_max):
    g = math.gcd(*size)
    uw, uh = size[0] // g, size[1] // g
    n = random.randint(math.ceil(g * s_min), math.floor(g * s_max))
    return uw * n, uh * n          # == rapidshot.capture_size(size, n / g)
```

For the three-model setup, sample in steps of 1/32 instead (`n32` in
`[ceil(32*s_min), floor(32*s_max)]`, scale = `n32 / 32`) so all three models
see the same set of scales. Deploy only scales inside the trained range.

---

## 10. Pitfalls

- **Camera with `region=`**: the "centre" becomes the centre of that region,
  and `to_frame` returns coordinates relative to it. For this feature create
  the camera on the full monitor.
- **Choosing the model by resolution**: use the full-monitor camera's
  `camera.height` / `frame.height` (physical pixels — Windows DPI scaling does
  not apply to Desktop Duplication). Fail loudly on an unknown resolution
  instead of falling back to a mismatched model.
- **Rotated (portrait) displays**: crops are refused when
  `frame.rotation_angle != 0`. Not supported.
- **HDR desktop**: an scRGB (linear light) frame is refused unless
  `allow_linear=True`, and models are not trained on linear values anyway. Run
  with HDR off.
- **Hybrid GPUs / capture on another adapter**: `to_torch()` raises
  `CrossAdapterRequired`. See "Hybrid GPU laptops" in `README.md`;
  `rapidshot.diagnose()` reports the machine type.
- **Idle screen**: Desktop Duplication delivers only changed frames.
  With `timeout=` set, `TensorStream` raises `TimeoutError` on a static screen;
  with `timeout=None` it waits.
- **Cost**: the area filter reads about `scale²` source pixels per output pixel.
  Measured: 384×512 → 288×384 fp16, 0.083 ms on an RTX 3080 (bilinear 0.076 ms).
  Re-measure at your largest scale and resolution if latency matters.
- **Don't**: resize on the CPU, compute the region by hand with a different
  rounding, use `sampling="bilinear"` for inference while training with
  INTER_AREA, set `capture_align=32`, or pad/letterbox.

---

## 11. Errors and what they mean

| error | cause | fix |
|---|---|---|
| `ValueError: sampling must be 'nearest' or 'bilinear', got 'area'` | the PyPI wheel (or a stale `.pyd`) is loaded | §6: build, `install_dev.py`, check `build_info()["source"]` |
| `rapidshot.capture_size` is `None` / missing | converter module failed to import, or upstream RapidShot installed | `from rapidshot.converter import capture_size` to see the error; install this fork |
| `ValueError: capture_scale must be at least 1.0 …` | scale < 1, NaN, inf, bool, non-number | pass a number ≥ 1.0 |
| `ValueError: … needs a WxH region, which does not fit the … frame` | scale too large for this monitor | ≤ 2.8125 for the three models (§5) |
| `ValueError: give crop or capture_scale, not both` | both passed to the constructor | use one |
| `RuntimeError: no tensor yet: iterate the stream first` | `to_frame` before the first frame | iterate first |
| `RuntimeError: no single-region conversion to map from …` | `to_frame` after a multi-region call | don't mix with `regions=` |
| `CrossAdapterRequired` | capture and CUDA on different adapters | README, "Hybrid GPU laptops" |
| `TimeoutError: no changed frame in … s` | static screen | `timeout=None`, or handle it |

---

## 12. Verification checklist for the integrating agent

Run all of these and report the results. Do not report success without them.

```python
import math, rapidshot, rapidshot.native

# 1. The fork's native build is loaded (area shader available).
assert rapidshot.native.build_info()["source"].startswith("development build"), \
    rapidshot.native.build_info()

# 2. The fork's Python with the align=1 default.
assert rapidshot.capture_size((288, 384), 1.5) == (432, 576)
assert rapidshot.capture_size((288, 384), 1.3) == (375, 500)
assert rapidshot.capture_size((288, 384), 1.3, align=32) == (384, 512)

# 3. The three models: exact at the deployed scale, same share of the screen.
for size, screen in [((288, 384), (1920, 1080)), ((384, 512), (2560, 1440)), ((576, 768), (3840, 2160))]:
    w, h = rapidshot.capture_size(size, CAPTURE_SCALE)
    assert (w, h) == (size[0] * CAPTURE_SCALE, size[1] * CAPTURE_SCALE), (size, w, h)
    assert w * size[1] == h * size[0]                       # exact ratio
    assert w <= screen[0] and h <= screen[1]                # fits
    assert size[0] % 32 == 0 and size[1] % 32 == 0          # model input

# 4. Live: shape, effective scale, centred region.
camera = rapidshot.create()
cfg = MODELS[camera.height]
with rapidshot.TensorStream(camera, cfg["size"], sampling="area",
                            capture_scale=CAPTURE_SCALE, dtype="float16", timeout=10) as s:
    t = next(s)
    w, h = cfg["size"]
    assert t.shape == (1, 3, h, w)
    rw, rh = s.capture_size
    F = s.frame
    assert s.capture_region == ((F.width - rw) // 2, (F.height - rh) // 2,
                                (F.width - rw) // 2 + rw, (F.height - rh) // 2 + rh)
    assert s.to_frame(0, 0) == s.capture_region[:2]
```

5. **Train/inference parity** (strongly recommended). One frame, read twice:
   once at identity size as the "screenshot" a dataset would start from, and
   once through the area path. The dataset recipe applied to the first must
   reproduce the second.

```python
import cv2, numpy as np, rapidshot

camera = rapidshot.create(output_color="BGRA")
frame = None
while frame is None:                       # needs a changing screen
    frame = camera.grab_frame()
with frame:
    W, H = frame.width, frame.height
    # The captured bytes, untouched: identity size + nearest = exact copy. BGRA.
    shot = rapidshot.GpuConverter(frame, (W, H), dtype="uint8", layout="nhwc",
                                  sampling="nearest").process(frame).numpy()[0].copy()
    size, scale = (288, 384), 1.5
    conv = rapidshot.GpuConverter(frame, size, dtype="uint8", layout="nhwc",
                                  sampling="area", capture_scale=scale)
    gpu = conv.process(frame).numpy()[0][..., :3].astype(int)       # BGR
w, h = rapidshot.capture_size(size, scale)
left, top = (W - w) // 2, (H - h) // 2
assert conv.capture_region == (left, top, left + w, top + h)
ref = cv2.resize(shot[top:top + h, left:left + w, :3], size,
                 interpolation=cv2.INTER_AREA).astype(int)          # BGR, the §8 recipe
assert np.abs(gpu - ref).max() <= 1, np.abs(gpu - ref).max()
camera.release()
```

   Measured on this fork (RTX 3080, 3840×2160 desktop with real content,
   2026-10-16): maximum difference **0** — bit-identical — for all three model
   inputs at scales 1.0, 1.25, 1.3, 1.5, 2.0 and 2.8125. For contrast,
   `cv2.INTER_LINEAR` on the same crops differed by up to 100 codes. A
   difference above 1 in your pipeline means it diverges (centring, filter,
   channel order or region), not the library.

Library tests covering this feature (run with `python -m pytest`):
`tests/test_capture_size.py`, `tests/test_gpu_converter_area.py` (needs
`opencv-python` for the cv2 comparison), `tests/test_tensor_stream_capture_scale.py`.
