"""Plan-owned decode-resolution policy for Bladeworks seek and scan (interactive preview).

Architecture map
================

    build_tensor_plan(..., decode_policy=DecodePolicy.VISIBLE)      (preview only)
        -> per video leaf, after its LayerSpec is lowered:
           resolve_decode_raster(layer, probe=..., policy=...)
              -> fallback rules (below) -> DecodeRaster.native(reason)     or
              -> static homography      -> DecodeRaster(display + encoded dims)
        -> LayerSpec.decode_raster
              -> decode.open_source     : ClipDecoder(decode_size=encoded dims)
                                          (libswscale downscale of the owning
                                          frame's planes, same pixel format,
                                          BEFORE pack / upload / planes_to_rgb)
              -> renderer.placed        : samples the smaller raster through
                                          ``decoded_to_native_matrix`` so every
                                          authored geometry stays evaluated
                                          against the NATIVE FrameGeometry

Why this exists
---------------
Bladeworks decodes every video leaf at its native raster, uploads and colour
converts that raster, and only then samples it onto the 720p / 540p / 480p
composition canvas.  For a 4K leaf on a 720p canvas that is a 12 MB upload, a
4K yuv->RGB kernel and the calibrated whole-raster minification per frame
(measured ~110 ms/frame source path on MPS; ~9 ms/frame when libswscale
produces the 1280x720 planes first -- see
``scripts/final_cut/fcpxml_tensor_decode_stage_bench.py``).  Decoding "near the
maximum visible output contribution" removes that cost without touching the
codec (the decoder still reconstructs the native picture; only the planes it
hands on are smaller).

What stays exact
----------------
* Geometry.  The plan keeps the native ``FrameGeometry`` and every exact kernel
  (crop percentages, camera placement, conform, transform) evaluates on it.
  The renderer only pre-multiplies the sampling homography with the fixed
  ``decoded -> native`` scale, so placement is identical to native decoding up
  to the resampling filter.
* Colour.  The downscale runs on the source's own planes in its own pixel
  format (yuv420p stays yuv420p, 10-bit stays 10-bit); ``planes_to_rgb`` and
  the HDR reject run unchanged on the smaller planes with the same tags.
* Time.  Seeking, forward ownership, the reverse GOP cache and temporal
  preroll all operate on the native decoded ``av.VideoFrame``; scaling happens
  after the owning frame is chosen.

Native-resolution fallback rules (documented product rules)
------------------------------------------------------------
``resolve_decode_raster`` returns a native raster with a named ``reason``
whenever any rule below applies.  Every rule is deliberately conservative: a
layer whose visible footprint cannot be bounded by ONE static axis-aligned
scale keeps native decoding in this first version.

    export           policy is DecodePolicy.NATIVE (render jobs never downscale)
    raster           the layer is a still / title / solid (no ClipDecoder)
    hdr              PQ / HLG sources (the float LUT path is not measured here)
    nested scope     the leaf sits inside ANY scope (inert or rendered):
                     compound, multicam, sync-clip, audition, retimed group
    animated         transform / corner-pin keyframes or Ken Burns (pan crop)
    rotation         a non-zero transform rotation
    corner pin       a non-identity corner pin (perspective homography)
    conform none     1:1 pixel placement depends on the native raster
    uncertain        the resolved homography is not a finite axis-aligned
                     affine (any shear / perspective / non-finite term)
    magnified        the layer needs >= native resolution on either axis
                     (zooms and crops that magnify past 1:1)

Supported shapes: root-level Fit and Fill leaves, static Trim / Crop windows,
static uniform or anisotropic zooms and position offsets (they request the
extra source resolution their magnification needs), and container display
rotation (the request is issued in encoded orientation).

Never above native: the requested raster is capped at the probed source
raster on both axes and rounded UP to even dimensions (chroma-block aligned
for 4:2:0 / 4:2:2), so the visible footprint is always covered.

Known preview-only deviations (accepted for seek / scan, never export)
----------------------------------------------------------------------
* The first spatial resample is libswscale's bilinear minification of the
  YUV planes instead of the calibrated encoded-RGB port
  (``sampler.resize_exact_aspect_opaque``); pixels differ by resampling noise.
* Trim / Crop alpha windows are re-expressed on the decoded grid with
  round-half-up edges: at most one output pixel at a window edge.

Main callers:
- ``plan._lower_leaf`` (via ``build_tensor_plan(decode_policy=...)``).
- ``renderer._FrameComposer.placed`` (``decoded_to_native_matrix``,
  ``scale_alpha_window``).
- ``decode.open_source`` (``DecodeRaster.encoded_size``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional

import numpy as np

from .errors import TensorRenderError
from .sampler import affine_matrix, layer_homography

if TYPE_CHECKING:  # pragma: no cover
    from .decode import VideoProbe
    from .plan import LayerSpec


class DecodePolicy(str, Enum):
    """Which decode raster ``build_tensor_plan`` requests for video leaves.

    ``NATIVE`` is the export contract (every source decoded at its own raster).
    ``VISIBLE`` is the interactive seek / scan contract described in the module
    docstring.
    """

    NATIVE = "native"
    VISIBLE = "visible"


# Tolerance for "axis-aligned affine": shear / perspective terms below this are
# treated as zero (float64 homography solves leave ~1e-12 residue).
_AXIS_ALIGNED_TOLERANCE = 1e-6


@dataclass(frozen=True)
class DecodeRaster:
    """The raster ``ClipDecoder`` should produce for one layer, plus why.

    ``display_width`` / ``display_height`` are what the renderer sees after
    container display rotation (the same orientation as
    ``LayerSpec.frame.source_*``); ``encoded_width`` / ``encoded_height`` is
    the request handed to the decoder (swapped for 90 / 270 degree sources).
    ``native_*`` are the probed display dimensions.  ``reason`` names the rule
    that produced this raster (``"fit"``, ``"fill"``, ``"crop"``, ``"trim"``,
    ``"zoom"`` or a ``native: ...`` fallback).
    """

    display_width: int
    display_height: int
    encoded_width: int
    encoded_height: int
    native_width: int
    native_height: int
    reason: str

    def __post_init__(self) -> None:
        for name in ("display_width", "display_height", "encoded_width", "encoded_height", "native_width", "native_height"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise TensorRenderError(f"DecodeRaster.{name} must be a positive integer, got {value!r}")
        if self.display_width > self.native_width or self.display_height > self.native_height:
            raise TensorRenderError(
                f"DecodeRaster {self.display_width}x{self.display_height} exceeds native "
                f"{self.native_width}x{self.native_height} ({self.reason})"
            )

    @property
    def is_native(self) -> bool:
        return (self.display_width, self.display_height) == (self.native_width, self.native_height)

    @property
    def encoded_size(self) -> tuple[int, int]:
        """``(width, height)`` in the decoder's (encoded) orientation."""

        return (self.encoded_width, self.encoded_height)

    @property
    def display_size(self) -> tuple[int, int]:
        return (self.display_width, self.display_height)

    def describe(self) -> str:
        return (
            f"{self.display_width}x{self.display_height} of native "
            f"{self.native_width}x{self.native_height} ({self.reason})"
        )


def native_raster(*, native_width: int, native_height: int, rotation_degrees: int, reason: str) -> DecodeRaster:
    """A ``DecodeRaster`` that requests the probed source raster (display orientation given)."""

    encoded_width, encoded_height = _encoded_orientation(native_width, native_height, rotation_degrees)
    return DecodeRaster(
        display_width=native_width,
        display_height=native_height,
        encoded_width=encoded_width,
        encoded_height=encoded_height,
        native_width=native_width,
        native_height=native_height,
        reason=reason,
    )


def _encoded_orientation(display_width: int, display_height: int, rotation_degrees: int) -> tuple[int, int]:
    """Display-oriented ``(w, h)`` -> the encoded ``(w, h)`` the decoder produces before rotation."""

    if rotation_degrees % 360 in {90, 270}:
        return display_height, display_width
    return display_width, display_height


def _even_ceil(value: float) -> int:
    """Smallest even integer >= ``value`` (at least 2)."""

    return max(2, int(math.ceil(value / 2.0 - 1e-9)) * 2)


def resolve_decode_raster(
    layer: "LayerSpec",
    *,
    probe: "VideoProbe",
    policy: DecodePolicy,
) -> Optional[DecodeRaster]:
    """Decide the decode raster for one lowered video leaf (``None`` for raster layers).

    What it does (Pythonese)
    ------------------------
    1. Raster layers (stills / titles / solids) have no decoder: return None.
    2. Under ``DecodePolicy.NATIVE`` return the native raster ("native: export").
    3. Apply the fallback rules from the module docstring in order (hdr, nested
       scope, animation, conform none, rotation / corner pin / uncertain
       geometry) -- each returns a native raster naming the rule.
    4. Otherwise evaluate the layer's homography at its first local frame
       (native source edge coordinates -> owner canvas edge coordinates), read
       the two axis scales, take the larger one as the uniform decode scale
       (source pixels must stay square), and if it is below 1 request
       ``even_ceil(scale * native)`` on both axes, capped at native.

    Main callers: ``plan._lower_leaf``.
    """

    if layer.source_kind != "video":
        return None
    native_width, native_height = layer.frame.source_width, layer.frame.source_height
    rotation = layer.source_rotation_degrees

    def native(reason: str) -> DecodeRaster:
        return native_raster(
            native_width=native_width,
            native_height=native_height,
            rotation_degrees=rotation,
            reason=f"native: {reason}",
        )

    if policy == DecodePolicy.NATIVE:
        return native("export")
    if policy != DecodePolicy.VISIBLE:
        raise TensorRenderError(f"unknown decode policy {policy!r}")
    if probe.color_transfer in {"smpte2084", "arib-std-b67"}:
        return native("hdr")
    if layer.nearest_scope_id is not None or layer.owner_id is not None or layer.local_clock is not None:
        return native("nested scope")
    if layer.geometry.has_animation:
        return native("ken burns" if layer.crop_mode == "pan" else "animated transform")
    if layer.conform == "none":
        return native("conform none")
    if layer.geometry.corners is not None:
        return native("corner pin")

    snapshot = layer.geometry_at(layer.first_frame, layer.frame_duration)
    if snapshot.transform.rotation_degrees % 360.0 != 0.0:
        return native("rotation")
    canvas_to_owner = np.array(layer.canvas_to_owner, dtype=np.float64).reshape(3, 3)
    homography = layer_homography(
        snapshot, frame=layer.frame, conform=layer.conform, canvas_to_project=canvas_to_owner
    )
    if not np.all(np.isfinite(homography)) or abs(homography[2, 2]) < 1e-12:
        return native("uncertain geometry")
    homography = homography / homography[2, 2]
    off_axis = (
        abs(homography[0, 1]), abs(homography[1, 0]),
        abs(homography[2, 0]), abs(homography[2, 1]),
    )
    if max(off_axis) > _AXIS_ALIGNED_TOLERANCE:
        return native("uncertain geometry")
    scale_x, scale_y = abs(float(homography[0, 0])), abs(float(homography[1, 1]))
    if scale_x <= 0.0 or scale_y <= 0.0:
        return native("uncertain geometry")
    scale = max(scale_x, scale_y)
    if scale >= 1.0:
        return native("magnified")

    display_width = min(native_width, _even_ceil(scale * native_width))
    display_height = min(native_height, _even_ceil(scale * native_height))
    if (display_width, display_height) == (native_width, native_height):
        return native("magnified")
    encoded_width, encoded_height = _encoded_orientation(display_width, display_height, rotation)
    return DecodeRaster(
        display_width=display_width,
        display_height=display_height,
        encoded_width=encoded_width,
        encoded_height=encoded_height,
        native_width=native_width,
        native_height=native_height,
        reason=_describe_shape(layer, snapshot),
    )


def _describe_shape(layer: "LayerSpec", snapshot) -> str:
    """Name the geometry shape a downscaled request came from (for diagnostics / tests)."""

    if layer.crop_mode == "crop":
        return "crop"
    if layer.crop_mode == "trim":
        return "trim"
    scale = snapshot.transform.scale
    if scale != (1.0, 1.0):
        return "zoom"
    return layer.conform


def decoded_to_native_matrix(raster: DecodeRaster) -> np.ndarray:
    """3x3 affine mapping decoded display edge coordinates onto native display edge coordinates.

    ``layer_homography`` maps NATIVE source edge coordinates to the canvas; the
    renderer right-multiplies it by this matrix so the same homography samples
    the smaller decoded raster.  Identity for a native raster.
    """

    return affine_matrix(
        raster.native_width / raster.display_width,
        raster.native_height / raster.display_height,
        0.0,
        0.0,
    )


def scale_alpha_window(
    window: Optional[tuple[int, int, int, int]],
    raster: DecodeRaster,
) -> Optional[tuple[int, int, int, int]]:
    """Re-express a native pixel-index alpha window ``(x0, x1, y0, y1)`` on the decoded grid.

    Edges are scaled by the decode ratio and rounded half up, then clamped to
    the decoded raster; a window that covers the whole native raster stays None.
    """

    if window is None or raster.is_native:
        return window
    x0, x1, y0, y1 = window
    sx = raster.display_width / raster.native_width
    sy = raster.display_height / raster.native_height

    def scaled(value: int, ratio: float, limit: int) -> int:
        return max(0, min(limit, int(math.floor(value * ratio + 0.5))))

    return (
        scaled(x0, sx, raster.display_width),
        scaled(x1, sx, raster.display_width),
        scaled(y0, sy, raster.display_height),
        scaled(y1, sy, raster.display_height),
    )


__all__ = [
    "DecodePolicy",
    "DecodeRaster",
    "decoded_to_native_matrix",
    "native_raster",
    "resolve_decode_raster",
    "scale_alpha_window",
]
