"""One homography per layer: crop / conform / corner pin / transform -> ``grid_sample``.

Architecture map
================

    GeometryPlan.snapshot(t)            (geometry.py -- the exact semantics)
        composed_quad, camera_placement, crop_rect, source_rect
            -> conform_matrix       : source raster (edge coords) -> clip canvas
            -> composed matrix      : clip canvas -> clip canvas (corner pin, then affine)
            -> canvas_to_project    : clip canvas -> project (inert group placement)
            -> layer_homography     : the product, source edge coords -> project edge coords
        source_alpha_window         : integer pixel-index window (trim / crop / pan / conform none)
    resize_exact_aspect_opaque(...) : encoded-domain legacy-swscale bilinear minification
                                      for the calibrated whole-raster Fit / Fill fast path
    warp(source_rgba, H, out_size)  : inverse-map every output pixel centre through H,
                                      bilinear ``grid_sample`` with transparent (zeros) padding

Conventions (the trap list in ``plan_research/xyzt_inventory.md`` §5)
--------------------------------------------------------------------
* Everything here is in *edge* coordinates: a raster of ``W`` pixels spans
  ``[0, W]`` and pixel ``i`` is centred at ``i + 0.5``.  ``geometry.py`` quads
  are edge coordinates too (``compose_spatial_quad`` maps the project rectangle
  ``(0,0)-(W,H)``).  FFmpeg's ``perspective`` works on pixel *indices*; the
  reference converts with ``correct_quad_for_pixel_centers`` -- applying the
  edge homography to output pixel centres ``(x+0.5, y+0.5)`` and sampling with
  ``align_corners=False`` is that same correction, so no half-pixel shift is
  applied here.  (Proof: for a pure scale ``s``, edge ``x = x0 + s*u`` at
  centres gives index ``x_i = x0 + s*u_i + (s-1)/2`` -- exactly the reference's
  ``(scale-1)/2`` conform term.)
* Rotation sign, +Y-up flip, project-height units, and the anchor rule are
  *not* re-derived here: ``composed_quad`` already contains them.
* Crop edges are percent of *source* height and arrive resolved in
  ``source_rect``/``crop_rect``; the alpha window uses the reference's
  inclusive integer-index rule (``between(X, left, W-right)``).
* Sampling is bilinear in calibrated linear light on *straight* RGBA with a
  transparent border, like the reference's ``lutrgb pow(1.94) -> perspective``;
  callers premultiply *after* the warp.

Main callers:
- ``renderer.render_document`` (per layer, per frame).
- ``experimental_tests/core/test_tensor_sampler.py`` (corner-error test).
"""

from __future__ import annotations

from collections import OrderedDict

import math
from functools import lru_cache
from typing import Final, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..core.geometry import FrameGeometry, GeometrySnapshot, Quad
from .swscale_fixedpoint import c_div as _c_div, finalize_swscale_filter

Matrix = np.ndarray  # 3x3 float64 homography, row vector convention h @ [x, y, 1]^T


# --------------------------------------------------------------------------- matrices


def identity_matrix() -> Matrix:
    return np.eye(3, dtype=np.float64)


def affine_matrix(scale_x: float, scale_y: float, translate_x: float, translate_y: float) -> Matrix:
    """``x' = scale_x * x + translate_x`` (and likewise for y)."""

    return np.array(
        [[scale_x, 0.0, translate_x], [0.0, scale_y, translate_y], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def homography_from_points(source: Quad, destination: Quad) -> Matrix:
    """Return the 3x3 homography mapping the four ``source`` points onto ``destination``.

    Quad order is (top-left, top-right, bottom-left, bottom-right) as in
    ``geometry.py``.  Four exact correspondences pin a projective map; the
    standard direct linear transform with ``h33 = 1`` is solved with numpy.
    """

    rows: list[list[float]] = []
    rhs: list[float] = []
    for (x, y), (u, v) in zip(source, destination):
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        rhs.append(u)
        rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        rhs.append(v)
    solution = np.linalg.solve(np.array(rows, dtype=np.float64), np.array(rhs, dtype=np.float64))
    return np.array(
        [[solution[0], solution[1], solution[2]],
         [solution[3], solution[4], solution[5]],
         [solution[6], solution[7], 1.0]],
        dtype=np.float64,
    )


def apply_matrix(matrix: Matrix, points: Quad) -> Quad:
    result = []
    for x, y in points:
        hx, hy, hw = matrix @ np.array([x, y, 1.0])
        result.append((float(hx / hw), float(hy / hw)))
    return tuple(result)  # type: ignore[return-value]


def is_identity(matrix: Matrix, *, tolerance: float = 1e-9) -> bool:
    return bool(np.allclose(matrix / matrix[2, 2], np.eye(3), atol=tolerance, rtol=0.0))


def project_rect_quad(width: int, height: int) -> Quad:
    return ((0.0, 0.0), (float(width), 0.0), (0.0, float(height)), (float(width), float(height)))


# --------------------------------------------------------------------------- semantics


def conform_matrix(snapshot: GeometrySnapshot, frame: FrameGeometry, conform: str) -> Matrix:
    """Source raster (edge coords) -> clip canvas (edge coords) for one frame.

    Mirrors ``GeometryPlan._conform_filters`` / ``resolve_camera_placement``:
    * Crop and Pan replace conform by the calibrated camera (``exact_scale``,
      ``exact_origin``) computed from the full source and the reference window;
    * Fit / Fill: ``scale = min/max(P/S)`` and ``x = (Pw - s*Ws)/2 + s*u``;
    * None: 1:1 pixels, integer-centred like ``crop``/``pad`` (truncation).
    """

    camera = snapshot.camera_placement
    if camera is not None:
        scale = camera.exact_scale
        return affine_matrix(scale, scale, camera.exact_origin_x, camera.exact_origin_y)
    source_width, source_height = frame.source_width, frame.source_height
    project_width, project_height = frame.project_width, frame.project_height
    if conform == "none":
        return affine_matrix(
            1.0, 1.0,
            float(_truncated_half(project_width - source_width)),
            float(_truncated_half(project_height - source_height)),
        )
    if conform == "fit":
        scale = min(project_width / source_width, project_height / source_height)
    elif conform == "fill":
        scale = max(project_width / source_width, project_height / source_height)
    else:
        raise ValueError(f"unknown conform {conform!r}")
    return affine_matrix(
        scale, scale,
        (project_width - scale * source_width) / 2.0,
        (project_height - scale * source_height) / 2.0,
    )


def _truncated_half(delta: int) -> int:
    """FFmpeg ``crop``/``pad`` evaluate ``(a-b)/2`` as a double and store it in an int."""

    return int(delta / 2.0)


def composed_matrix(snapshot: GeometrySnapshot, frame: FrameGeometry) -> Matrix:
    """Clip canvas -> clip canvas: corner pin then anchor transform (``composed_quad``)."""

    return homography_from_points(
        project_rect_quad(frame.project_width, frame.project_height), snapshot.composed_quad
    )


def layer_homography(
    snapshot: GeometrySnapshot,
    *,
    frame: FrameGeometry,
    conform: str,
    canvas_to_project: Matrix,
) -> Matrix:
    """Source edge coords -> project edge coords: ``canvas_to_project @ composed @ conform``."""

    return canvas_to_project @ composed_matrix(snapshot, frame) @ conform_matrix(snapshot, frame, conform)


def source_alpha_window(
    snapshot: GeometrySnapshot,
    *,
    frame: FrameGeometry,
    conform: str,
    crop_mode: Optional[str],
) -> Optional[tuple[int, int, int, int]]:
    """Return ``(x0, x1, y0, y1)`` pixel-index bounds (half-open) that stay opaque, or None.

    * Trim: the integer extraction rectangle (``crop`` then transparent ``pad``).
    * Crop / Pan: the reference's ``geq`` window keeps integer sample indices
      with ``left <= X <= W - right`` (inclusive, fractional edges).
    * Conform None: ``crop`` to the project size, centred by truncation.
    Windows intersect; a source without any window returns None.
    """

    source_width, source_height = frame.source_width, frame.source_height
    x0, x1, y0, y1 = 0, source_width, 0, source_height
    if crop_mode == "trim" and snapshot.crop_rect is not None:
        rect = snapshot.crop_rect
        x0, x1 = max(x0, rect.x), min(x1, rect.x + rect.width)
        y0, y1 = max(y0, rect.y), min(y1, rect.y + rect.height)
    elif crop_mode in {"crop", "pan"} and snapshot.source_rect is not None:
        rect = snapshot.source_rect
        x0 = max(x0, math.ceil(rect.x - 1e-9))
        x1 = min(x1, math.floor(rect.x + rect.width + 1e-9) + 1)
        y0 = max(y0, math.ceil(rect.y - 1e-9))
        y1 = min(y1, math.floor(rect.y + rect.height + 1e-9) + 1)
    if conform == "none" and snapshot.camera_placement is None:
        if source_width > frame.project_width:
            left = _truncated_half(source_width - frame.project_width)
            x0, x1 = max(x0, left), min(x1, left + frame.project_width)
        if source_height > frame.project_height:
            top = _truncated_half(source_height - frame.project_height)
            y0, y1 = max(y0, top), min(y1, top + frame.project_height)
    if (x0, x1, y0, y1) == (0, source_width, 0, source_height):
        return None
    return (max(0, x0), max(0, x1), max(0, y0), max(0, y1))


def uses_exact_aspect_minification(
    snapshot: GeometrySnapshot,
    *,
    frame: FrameGeometry,
    conform: str,
    crop_mode: Optional[str],
    source_is_opaque: bool,
) -> bool:
    """Whether Final Cut uses its calibrated whole-raster resize path.

    The retained Final Cut witnesses use an encoded-domain, support-scaled
    bilinear filter when an opaque decoder raster is reduced to an exactly
    matching Fit / Fill canvas. Crop and Pan create a camera, Trim creates an
    alpha edge, and mismatched aspect ratios create transparent or clipped
    edges, so all of those stay on the general homography sampler.

    Main callers:
    - ``renderer._FrameComposer.placed`` before linear-light conversion.

    Why this exists:
    A homography sampler reads only a 2x2 source neighbourhood. That aliases a
    4K or 1080p raster reduced to a small proxy, while Final Cut's resize filter
    widens its footprint as the reduction ratio grows.
    """

    source_width, source_height = frame.source_width, frame.source_height
    output_width, output_height = frame.project_width, frame.project_height
    return (
        source_is_opaque
        and crop_mode is None
        and snapshot.camera_placement is None
        and conform in {"fit", "fill"}
        and output_width >= 3
        and output_height >= 3
        and source_width > output_width
        and source_height > output_height
        and source_width * output_height == source_height * output_width
    )


_SWS_H_ALIGN = 4
_SWS_V_ALIGN = 2
_SWS_ONE_H = 1 << 14
_SWS_ONE_V = 1 << 12

# Legacy swscale's default RGB matrix when an RGBA link has no colourspace tag.
# These are ``fill_rgb2yuv_table`` and ``ff_yuv2rgb_c_init_tables`` for
# SWS_CS_DEFAULT (BT.601), limited range, in FFmpeg n8.0.1.
_RGB2YUV_601 = (8414, 16519, 3208, -4865, -9528, 14392, 14392, -12061, -2332)
_YUV2RGB_601 = (8192, 9539, 13075, -6660, -3209, 16525)


@lru_cache(maxsize=256)
def _swscale_bilinear_filter(
    source_length: int,
    output_length: int,
    *,
    alignment: int,
    one: int,
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    """Port legacy swscale ``initFilter`` for one bilinear minification axis.

    This includes its 16.16 output-centre positions, support-scaled triangle,
    near-zero tap reduction, SIMD-width padding, border folding, 14/12-bit
    coefficient quantisation, and residual-error diffusion. The returned rows
    each sum to ``one``.

    An identity axis (``output_length == source_length``, i.e. no scaling on
    that axis) is a special case: swscale performs no resampling there, so we
    short-circuit to a copy filter instead of running the minification maths.

    Main callers:
    - ``resize_exact_aspect_opaque`` for the horizontal and vertical passes.

    Why this exists:
    PyTorch's antialiased bilinear kernel has different positions and weights.
    On sharp content that difference is tens of 8-bit codes, so row 23 requires
    swscale's actual legacy coefficients rather than a generic antialias flag.
    """

    # Identity axis: an unscaled axis reaches this builder on exact-2x opaque
    # downscales, where the packed-RGBA path derives a half-width chroma plane
    # (e.g. source 1920 -> chroma 960) whose length already equals the output
    # width (960). swscale does no filtering on an unscaled axis, so the correct
    # behaviour is a straight copy: output sample ``i`` selects source sample
    # ``i`` with the full weight ``one``. Feeding this single-tap-per-row filter
    # through the shared horizontal/vertical gather reproduces the copy exactly
    # while staying in the same fixed-point domain a real minified row uses --
    # every row still sums to ``one``, so the downstream >>13 / >>10 rescale is
    # identical. We must NOT relax the guard below to ``<=`` and run the general
    # maths for this case: at output == source it produces a degenerate,
    # non-identity filter, and at output < 3 the tap geometry is undefined.
    if output_length == source_length:
        positions = tuple(range(output_length))
        rows = tuple((one,) for _ in range(output_length))
        return positions, rows

    if not (3 <= output_length < source_length):
        raise ValueError(f"legacy bilinear minification requires 3 <= output < source, got {source_length}->{output_length}")
    increment = ((source_length << 16) + (output_length >> 1)) // output_length
    size = 1 + (2 * source_length + output_length - 1) // output_length
    size = max(1, min(size, source_length - 2))
    fone = 1 << (54 - min((source_length // output_length).bit_length() - 1, 8))
    position = ((128 * increment) >> 7) - ((128 * 0x10000) >> 7)
    positions: list[int] = []
    rows: list[list[int]] = []

    # The bilinear kernel: a support-scaled triangle (``distance`` shrinks by the
    # reduction ratio) quantised to ``fone``. This is the only part that differs
    # from the bicubic port; ``finalize_swscale_filter`` runs the identical
    # reduce / align / border-fix / normalise tail on the raw taps. Bilinear is
    # strictly minifying, so it never needs the size-1 vertical align quirk
    # (``apply_align_quirk=False``) and its historical reform only ever pads
    # (``reform_truncates=False``).
    for _ in range(output_length):
        source_index = _c_div(position - (size - 2) * (1 << 16), 1 << 17)
        positions.append(source_index)
        row: list[int] = []
        for _tap in range(size):
            distance = abs(source_index * (1 << 17) - position) << 13
            distance = distance * output_length // source_length
            coefficient = max(0, (1 << 30) - distance) * (fone >> 30)
            row.append(coefficient)
            source_index += 1
        rows.append(row)
        position += 2 * increment

    return finalize_swscale_filter(
        positions,
        rows,
        size,
        fone,
        filter_align=alignment,
        one=one,
        apply_align_quirk=False,
        reform_truncates=False,
        src_len=source_length,
    )


def _filter_tensors(
    source_length: int,
    output_length: int,
    *,
    alignment: int,
    one: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions, taps = _swscale_bilinear_filter(
        source_length,
        output_length,
        alignment=alignment,
        one=one,
    )
    return (
        torch.tensor(positions, dtype=torch.long, device=device),
        torch.tensor(taps, dtype=torch.int32, device=device),
    )


def _floor_shift(values: torch.Tensor, bits: int) -> torch.Tensor:
    return torch.div(values, 1 << bits, rounding_mode="floor")


def resize_exact_aspect_opaque(source_rgb: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    """Minify opaque encoded RGB through FFmpeg n8.0.1 legacy swscale bilinear.

    The port follows the packed-RGBA path, not an independent resize of R, G,
    and B. Legacy swscale first converts RGB to 14-bit BT.601 limited-range
    Y/U/V, derives half-width chroma when reducing by at least 2x, applies its
    14-bit horizontal and 12-bit vertical bilinear filters, then uses the
    fixed-point full-chroma YUV-to-RGB writer. Input and output remain encoded
    8-bit code values, matching the calibrated plan-emitter stage order.

    Main callers:
    - ``renderer._FrameComposer.placed`` for the bounded exact-aspect case.
    """

    if source_rgb.ndim != 3 or source_rgb.shape[0] != 3:
        raise ValueError(f"source_rgb must have shape [3,H,W], got {tuple(source_rgb.shape)}")
    _, source_height, source_width = source_rgb.shape
    if not (3 <= width < source_width and 3 <= height < source_height):
        raise ValueError(
            f"resize_exact_aspect_opaque only supports minification with output dimensions >= 3, "
            f"got {source_width}x{source_height}->{width}x{height}"
        )

    code = torch.clamp(torch.round(source_rgb * 255.0), 0.0, 255.0).to(torch.int32)
    red, green, blue = code[0], code[1], code[2]
    ry, gy, by, ru, gu, bu, rv, gv, bv = _RGB2YUV_601
    y14 = _floor_shift(ry * red + gy * green + by * blue + (32 << 14) + (1 << 8), 9)

    chroma_half = source_width % 2 == 0 and width <= source_width // 2
    if chroma_half:
        red_pair = red[:, 0::2] + red[:, 1::2]
        green_pair = green[:, 0::2] + green[:, 1::2]
        blue_pair = blue[:, 0::2] + blue[:, 1::2]
        u14 = _floor_shift(ru * red_pair + gu * green_pair + bu * blue_pair + (256 << 15) + (1 << 9), 10)
        v14 = _floor_shift(rv * red_pair + gv * green_pair + bv * blue_pair + (256 << 15) + (1 << 9), 10)
        chroma_width = source_width // 2
    else:
        u14 = _floor_shift(ru * red + gu * green + bu * blue + (256 << 14) + (1 << 8), 9)
        v14 = _floor_shift(rv * red + gv * green + bv * blue + (256 << 14) + (1 << 8), 9)
        chroma_width = source_width

    h_positions, h_taps = _filter_tensors(
        source_width, width, alignment=_SWS_H_ALIGN, one=_SWS_ONE_H, device=source_rgb.device
    )
    c_positions, c_taps = _filter_tensors(
        chroma_width, width, alignment=_SWS_H_ALIGN, one=_SWS_ONE_H, device=source_rgb.device
    )
    v_positions, v_taps = _filter_tensors(
        source_height, height, alignment=_SWS_V_ALIGN, one=_SWS_ONE_V, device=source_rgb.device
    )

    def horizontal(plane14: torch.Tensor, positions: torch.Tensor, taps: torch.Tensor) -> torch.Tensor:
        value = torch.zeros((source_height, width), dtype=torch.int32, device=source_rgb.device)
        for tap_index in range(taps.shape[1]):
            indices = torch.clamp(positions + tap_index, max=plane14.shape[1] - 1)
            value += torch.index_select(plane14, 1, indices) * taps[:, tap_index].view(1, -1)
        return torch.clamp(_floor_shift(value, 13), max=32767)

    horizontal_y = horizontal(y14, h_positions, h_taps)
    horizontal_u = horizontal(u14, c_positions, c_taps)
    horizontal_v = horizontal(v14, c_positions, c_taps)

    def vertical(plane15: torch.Tensor, *, center_chroma: bool) -> torch.Tensor:
        initial = (1 << 9) - (128 << 19) if center_chroma else (1 << 9)
        value = torch.full((height, width), initial, dtype=torch.int32, device=source_rgb.device)
        for tap_index in range(v_taps.shape[1]):
            indices = torch.clamp(v_positions + tap_index, max=source_height - 1)
            value += torch.index_select(plane15, 0, indices) * v_taps[:, tap_index].view(-1, 1)
        return _floor_shift(value, 10)

    y17 = vertical(horizontal_y, center_chroma=False)
    u17 = vertical(horizontal_u, center_chroma=True)
    v17 = vertical(horizontal_v, center_chroma=True)
    y_offset, y_coeff, v2r, v2g, u2g, u2b = _YUV2RGB_601
    luma = (y17 - y_offset) * y_coeff + (1 << 21)
    limit = (1 << 30) - 1
    result = torch.stack(
        (
            torch.clamp(luma + v17 * v2r, 0, limit),
            torch.clamp(luma + v17 * v2g + u17 * u2g, 0, limit),
            torch.clamp(luma + u17 * u2b, 0, limit),
        )
    )
    return _floor_shift(result, 22).to(source_rgb.dtype) / 255.0


def apply_display_rotation(source: torch.Tensor, rotation_degrees: int) -> torch.Tensor:
    """Apply container display rotation to ``[C,H,W]`` before timeline geometry.

    FFmpeg's autorotate and Final Cut interpret a positive display-matrix angle
    as a counter-clockwise image rotation. ``torch.rot90`` uses that same sign.

    Main callers:
    - ``renderer._FrameComposer.placed`` immediately after source decode.
    """

    normalized = rotation_degrees % 360
    if normalized not in {0, 90, 180, 270}:
        raise ValueError(f"display rotation must be a quarter turn, got {rotation_degrees}")
    return source if normalized == 0 else torch.rot90(source, k=normalized // 90, dims=(-2, -1))


# --------------------------------------------------------------------------- torch


# How many distinct (height, width, device) centre grids ``GridCache`` keeps.
# The steady state of a render is a handful of sizes (the output raster, each
# retimed pad grid, a few fixed clip canvases); an effects-bearing clip with an
# ANIMATED pan / zoom produces a differently-sized overscan surface on nearly
# every frame (``renderer._overscan_surface``), and each 1920x1080-class centre
# tensor is ~25 MB, so an unbounded dict would grow by the frame count.
CENTRES_CACHE_LIMIT: Final[int] = 8


class GridCache:
    """Homogeneous output pixel-centre grids per (height, width, device), plus per-key grids.

    ``grid_for(key, matrix, ...)`` returns the normalized sampling grid for
    ``matrix`` and reuses the previous one while the matrix is unchanged --
    static layers pay the grid build once, animated layers once per frame.

    ``centres`` is a small LRU (``CENTRES_CACHE_LIMIT`` entries): the recurring
    sizes stay hot, while the one-off sizes of a frame-varying overscan surface
    are rebuilt (a cheap ``meshgrid``) instead of pinning device memory forever.
    """

    def __init__(self) -> None:
        self._centres: OrderedDict[tuple[int, int, str], torch.Tensor] = OrderedDict()
        self._grids: dict[str, tuple[bytes, torch.Tensor]] = {}

    def centres(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        key = (height, width, str(device))
        centres = self._centres.get(key)
        if centres is not None:
            self._centres.move_to_end(key)
            return centres
        ys = torch.arange(height, dtype=torch.float32, device=device) + 0.5
        xs = torch.arange(width, dtype=torch.float32, device=device) + 0.5
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        centres = torch.stack((grid_x, grid_y, torch.ones_like(grid_x)), dim=-1).reshape(-1, 3)
        self._centres[key] = centres
        while len(self._centres) > CENTRES_CACHE_LIMIT:
            self._centres.popitem(last=False)
        return centres

    def grid_for(
        self,
        key: str,
        matrix: Matrix,
        *,
        out_height: int,
        out_width: int,
        source_height: int,
        source_width: int,
        device: torch.device,
    ) -> torch.Tensor:
        signature = np.ascontiguousarray(matrix, dtype=np.float64).tobytes() + bytes(
            f"{out_height}x{out_width}<{source_height}x{source_width}", "ascii"
        )
        cached = self._grids.get(key)
        if cached is not None and cached[0] == signature:
            return cached[1]
        grid = sampling_grid(
            matrix,
            centres=self.centres(out_height, out_width, device),
            out_height=out_height,
            out_width=out_width,
            source_height=source_height,
            source_width=source_width,
        )
        self._grids[key] = (signature, grid)
        return grid

    def forget(self, key: str) -> None:
        self._grids.pop(key, None)


def sampling_grid(
    matrix: Matrix,
    *,
    centres: torch.Tensor,
    out_height: int,
    out_width: int,
    source_height: int,
    source_width: int,
) -> torch.Tensor:
    """Inverse-map output pixel centres through ``matrix`` into normalized source coords.

    ``matrix`` maps source edge coords to output edge coords; the grid holds
    the source position of every output pixel centre, normalized so that the
    source raster spans ``[-1, 1]`` (``align_corners=False``).  Solved in
    float64 on the CPU for the 3x3 inverse, evaluated in float32 on device
    (fp16 grids are ~1 px wrong; never use them).
    """

    inverse = np.linalg.inv(matrix)
    inverse_t = torch.from_numpy(inverse.T.astype(np.float32)).to(centres.device)
    mapped = centres @ inverse_t  # [N, 3]
    w = mapped[:, 2:3]
    # A point behind the camera / at infinity (degenerate perspective) is sent
    # far outside the source so zeros padding makes it transparent.
    safe_w = torch.where(w.abs() < 1e-12, torch.full_like(w, 1e-12), w)
    u = mapped[:, 0:1] / safe_w
    v = mapped[:, 1:2] / safe_w
    grid_x = u * (2.0 / source_width) - 1.0
    grid_y = v * (2.0 / source_height) - 1.0
    return torch.cat((grid_x, grid_y), dim=1).reshape(1, out_height, out_width, 2)


def warp(source_rgba: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Bilinear straight-RGBA warp with a transparent border; ``[4,h,w]`` -> ``[4,H,W]``."""

    return F.grid_sample(
        source_rgba.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=False
    ).squeeze(0)


def apply_alpha_window(alpha: torch.Tensor, window: Optional[tuple[int, int, int, int]]) -> torch.Tensor:
    """Zero ``alpha`` (``[1,h,w]``) outside the half-open pixel-index window."""

    if window is None:
        return alpha
    x0, x1, y0, y1 = window
    masked = torch.zeros_like(alpha)
    if x1 > x0 and y1 > y0:
        masked[:, y0:y1, x0:x1] = alpha[:, y0:y1, x0:x1]
    return masked


def premultiply(straight_rgba: torch.Tensor) -> torch.Tensor:
    alpha = straight_rgba[3:4]
    return torch.cat((straight_rgba[:3] * alpha, alpha), dim=0)
