"""Independent CPU pixel evaluator for frozen raster camera plans.

Architecture map
================

frozen JSON plan -> typed ``RasterCameraSamplingPlan``
    -> inverse destination quad
    -> inclusive integer source-alpha ownership
    -> full support plus a two-pixel transparent border
    -> power-1.94 RGB / linear-alpha bilinear sampling
    -> RGBA8 camera boundary
    -> typed scope source-over on transparent or opaque black

This is evidence-side code.  It deliberately does not import a CPU emitter,
Vulkan lowering module, FCPXML parser, geometry planner, or FFmpeg graph
builder.  The only production types it uses are the frozen plan records that
the evaluator is required to consume.  The legacy comparison is a separately
written stock-FFmpeg command line, executed through the ``ffmpeg`` binary.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
from typing import TypeAlias

from bladeworks.core.composition_ir import (
    RasterCameraSamplingPlan,
    ResolvedCameraSample,
    SamplingRect,
    SourceAlphaWindowPlan,
    SurfaceSpec,
)


RGBA: TypeAlias = tuple[float, float, float, float]
Frame: TypeAlias = tuple[RGBA, ...]


@dataclass(frozen=True)
class ReferenceFrame:
    """One expected frame, retaining camera alpha before underlay compositing."""

    index: int
    pts: Fraction
    camera: Frame
    composited: Frame

    def rgba8(self, *, composited: bool = False) -> bytes:
        pixels = self.composited if composited else self.camera
        return bytes(
            channel
            for pixel in pixels
            for channel in (_round8(value) for value in pixel)
        )

    def sha256(self, *, composited: bool = False) -> str:
        return hashlib.sha256(self.rgba8(composited=composited)).hexdigest()

    def typed_module_rgba8(self) -> bytes:
        """Return the camera after its transparent-black scope fold.

        Main callers:
        - Vulkan input-compositor physical evidence.

        Why this exists:
        ``camera`` is an internal sampler result whose hidden RGB is needed
        during bilinear interpolation.  A reusable module handle is a later
        stage: Normal source-over onto the scope's transparent-black clear.
        Pinned FFmpeg therefore returns black RGB when the completed module
        pixel has zero alpha.
        """

        return self.rgba8(composited=True)


@dataclass(frozen=True)
class LegacyRender:
    """Decoded bytes and exact clock metadata from the legacy CPU graph."""

    frames: tuple[bytes, ...]
    camera_frames: tuple[bytes, ...]
    pts: tuple[int, ...]
    time_base: Fraction
    width: int
    height: int


def _round8(value: float) -> int:
    if not math.isfinite(value):
        raise ValueError("non-finite reference channel")
    return max(0, min(255, int(math.floor(value + 0.5))))


def ffmpeg_rgba8_opacity_alpha(alpha_code: int, wire_opacity: float) -> int:
    """Apply pinned FFmpeg GEQ's RGBA8 alpha truncation exactly.

    The v2 wire owns a binary32 opacity.  Python's ``as_integer_ratio`` gives
    that rounded wire value's exact rational meaning, so this evidence oracle
    can perform ``uint8_t(alpha * opacity)`` without host floating ambiguity.
    """

    if isinstance(alpha_code, bool) or not isinstance(alpha_code, int):
        raise ValueError("alpha_code must be an integer")
    if not 0 <= alpha_code <= 255:
        raise ValueError("alpha_code must be between 0 and 255")
    if isinstance(wire_opacity, bool) or not isinstance(wire_opacity, float):
        raise ValueError("wire_opacity must be a binary32-compatible float")
    if not math.isfinite(wire_opacity) or not 0.0 <= wire_opacity <= 1.0:
        raise ValueError("wire_opacity must be finite and between 0 and 1")
    rounded = struct.unpack("<f", struct.pack("<f", wire_opacity))[0]
    numerator, denominator = rounded.as_integer_ratio()
    return alpha_code * numerator // denominator


def ffmpeg_fast_div255(value: int) -> int:
    """Return pinned ``FAST_DIV255`` nearest-integer division."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("FAST_DIV255 input must be a non-negative integer")
    return ((value + 128) * 257) >> 16


def ffmpeg_rgba8_source_over_alpha(lower: int, source: int) -> int:
    """Fold one already-opacity-scaled alpha code over another."""

    for name, value in (("lower", lower), ("source", source)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} alpha must be an integer")
        if not 0 <= value <= 255:
            raise ValueError(f"{name} alpha must be between 0 and 255")
    if source == 0:
        return lower
    if source == 255:
        return 255
    return lower + ffmpeg_fast_div255((255 - lower) * source)


def load_frozen_plan(path: Path, *, name: str) -> RasterCameraSamplingPlan:
    """Load one plan from evidence JSON without resolving any authored input."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    item = payload["plans"][name]

    def surface(value: dict[str, int]) -> SurfaceSpec:
        return SurfaceSpec(value["width"], value["height"], value["origin_x"], value["origin_y"])

    samples: list[ResolvedCameraSample] = []
    for raw in item["samples"]:
        rect = SamplingRect(*raw["source_alpha_window"]["rect"])
        alpha = SourceAlphaWindowPlan(rect, raw["source_alpha_window"]["behavior"])
        samples.append(
            ResolvedCameraSample(
                frame_index=raw["frame_index"],
                source_alpha_window=alpha,
                scale=raw["scale"],
                origin_x=raw["origin_x"],
                origin_y=raw["origin_y"],
                camera_quad=tuple(tuple(point) for point in raw["camera_quad"]),  # type: ignore[arg-type]
                pixel_center_quad=tuple(tuple(point) for point in raw["pixel_center_quad"]),  # type: ignore[arg-type]
                transparent_border_quad=tuple(tuple(point) for point in raw["transparent_border_quad"]),  # type: ignore[arg-type]
            )
        )
    return RasterCameraSamplingPlan(
        operation=item["operation"],
        frame_count=item["frame_count"],
        support_surface=surface(item["support_surface"]),
        padded_support_surface=surface(item["padded_support_surface"]),
        output_clip=surface(item["output_clip"]),
        pixel_center_convention=item["pixel_center_convention"],
        interpolation_kernel=item["interpolation_kernel"],
        rgb_interpolation_space=item["rgb_interpolation_space"],
        alpha_interpolation=item["alpha_interpolation"],
        transparent_border_behavior=item["transparent_border_behavior"],
        transparent_border_pixels=item["transparent_border_pixels"],
        samples=tuple(samples),
    )


def owns_integer_source_pixel(rect: SamplingRect, x: int, y: int) -> bool:
    """Apply the legacy GEQ rule to integer X/Y samples, inclusively."""

    return rect.left <= x <= rect.right and rect.top <= y <= rect.bottom


def _inverse_source_edge(
    sample: ResolvedCameraSample,
    support: SurfaceSpec,
    destination_x: float,
    destination_y: float,
) -> tuple[float, float]:
    """Invert the frozen affine destination quad at one output pixel center."""

    top_left, top_right, bottom_left, _ = sample.pixel_center_quad
    ax = ((top_right[0] - top_left[0]) / support.width, (top_right[1] - top_left[1]) / support.width)
    ay = ((bottom_left[0] - top_left[0]) / support.height, (bottom_left[1] - top_left[1]) / support.height)
    dx = destination_x - top_left[0]
    dy = destination_y - top_left[1]
    determinant = ax[0] * ay[1] - ax[1] * ay[0]
    if abs(determinant) < 1e-12:
        raise ValueError("singular frozen camera quad")
    return (
        (dx * ay[1] - dy * ay[0]) / determinant,
        (ax[0] * dy - ax[1] * dx) / determinant,
    )


def _source_pixel(source: Frame, width: int, height: int, rect: SamplingRect, x: int, y: int) -> RGBA:
    if not (0 <= x < width and 0 <= y < height):
        return (0.0, 0.0, 0.0, 0.0)
    pixel = source[y * width + x]
    if not owns_integer_source_pixel(rect, x, y):
        return pixel[0], pixel[1], pixel[2], 0.0
    return pixel


def _sample_bilinear(plan: RasterCameraSamplingPlan, sample: ResolvedCameraSample, source: Frame, source_size: tuple[int, int], sx: float, sy: float) -> RGBA:
    """Sample the padded support in independent power/alpha domains."""

    width, height = source_size
    border = plan.transparent_border_pixels
    padded_x = sx - 0.5 + border
    padded_y = sy - 0.5 + border
    x0, y0 = math.floor(padded_x), math.floor(padded_y)
    fx, fy = padded_x - x0, padded_y - y0
    rgb = [0.0, 0.0, 0.0]
    alpha = 0.0
    for dy in (0, 1):
        for dx in (0, 1):
            px, py = x0 + dx, y0 + dy
            if 0 <= px < plan.padded_support_surface.width and 0 <= py < plan.padded_support_surface.height:
                value = _source_pixel(source, width, height, sample.source_alpha_window.rect, px - border, py - border)
            else:
                value = (0.0, 0.0, 0.0, 0.0)
            weight = (1.0 - fx if dx == 0 else fx) * (1.0 - fy if dy == 0 else fy)
            for channel in range(3):
                rgb[channel] += weight * (max(0.0, value[channel]) / 255.0) ** 1.94
            alpha += weight * value[3]
    return tuple(255.0 * max(0.0, value) ** (1.0 / 1.94) for value in rgb) + (alpha,)


def _over(source: RGBA, underlay: RGBA) -> RGBA:
    source_alpha = max(0.0, min(1.0, source[3] / 255.0))
    underlay_alpha = max(0.0, min(1.0, underlay[3] / 255.0))
    output_alpha = source_alpha + underlay_alpha * (1.0 - source_alpha)
    if output_alpha == 0.0:
        return (0.0, 0.0, 0.0, 0.0)
    return tuple(
        (source[c] * source_alpha + underlay[c] * underlay_alpha * (1.0 - source_alpha)) / output_alpha
        for c in range(3)
    ) + (255.0 * output_alpha,)


def evaluate(plan: RasterCameraSamplingPlan, source: Frame, *, source_size: tuple[int, int], frame_duration: Fraction, pts_origin: Fraction, underlay: RGBA) -> tuple[ReferenceFrame, ...]:
    """Evaluate every output frame using only the frozen plan records."""

    if len(source) != source_size[0] * source_size[1]:
        raise ValueError("source dimensions do not match source frame")
    samples = {sample.frame_index: sample for sample in plan.samples}
    if plan.operation != "two_rect_pan":
        samples = {index: plan.samples[0] for index in range(plan.frame_count)}
    if tuple(sorted(samples)) != tuple(range(plan.frame_count)):
        raise ValueError("frozen samples do not cover every output frame")
    result: list[ReferenceFrame] = []
    for index in range(plan.frame_count):
        sample = samples[index]
        camera: list[RGBA] = []
        composited: list[RGBA] = []
        for y in range(plan.output_clip.height):
            for x in range(plan.output_clip.width):
                sx, sy = _inverse_source_edge(sample, plan.support_surface, x + 0.5, y + 0.5)
                value = _sample_bilinear(plan, sample, source, source_size, sx, sy)
                camera.append(value)
                composited.append(_over(value, underlay))
        result.append(ReferenceFrame(index, pts_origin + index * frame_duration, tuple(camera), tuple(composited)))
    return tuple(result)


def _quad_options(quad: tuple[tuple[float, float], ...]) -> str:
    names = ("x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3")
    values = [coordinate for point in quad for coordinate in point]
    return ":".join(f"{name}={value:.12g}" for name, value in zip(names, values, strict=True))


def legacy_camera_graph(plan: RasterCameraSamplingPlan, sample_index: int) -> str:
    """Return the frozen legacy CPU perspective graph, without production calls."""

    sample = plan.samples[0] if plan.operation != "two_rect_pan" else plan.samples[sample_index]
    rect = sample.source_alpha_window.rect
    border = plan.transparent_border_pixels
    alpha_window = (
        "geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':"
        f"a='alpha(X,Y)*between(X,{rect.left:.12g},{rect.right:.12g})*between(Y,{rect.top:.12g},{rect.bottom:.12g})'"
    )
    sampler = (
        "format=rgba64le,"
        "lutrgb=r='maxval*pow(val/maxval,1.94)':g='maxval*pow(val/maxval,1.94)':b='maxval*pow(val/maxval,1.94)',"
        f"pad=w=iw+{2 * border}:h=ih+{2 * border}:x={border}:y={border}:color=black@0,"
        "setparams=range=full,perspective=sense=destination:eval=init:interpolation=linear:"
        + _quad_options(sample.transparent_border_quad)
        + f",lutrgb=r='maxval*pow(val/maxval,{1.0 / 1.94:.12g})':g='maxval*pow(val/maxval,{1.0 / 1.94:.12g})':b='maxval*pow(val/maxval,{1.0 / 1.94:.12g})',"
        f"setparams=range=full,crop=w={plan.output_clip.width}:h={plan.output_clip.height}:x={border}:y={border},format=rgba"
    )
    return f"format=rgba64le,{alpha_window},{sampler}"


def _write_pam(path: Path, source: Frame, width: int, height: int) -> None:
    body = bytes(channel for pixel in source for channel in (_round8(value) for value in pixel))
    path.write_bytes((f"P7\nWIDTH {width}\nHEIGHT {height}\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\n").encode() + body)


def run_legacy_cpu_graph(plan: RasterCameraSamplingPlan, source: Frame, *, source_size: tuple[int, int], workdir: Path) -> LegacyRender | None:
    """Execute the direct legacy graph if stock FFmpeg is installed."""

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        return None
    source_path = workdir / "landmarks.pam"
    output_path = workdir / "legacy.nut"
    camera_path = workdir / "legacy-camera.nut"
    _write_pam(source_path, source, *source_size)
    branches = []
    branch_count = plan.frame_count if plan.operation == "two_rect_pan" else 1
    input_labels = [f"i{index}" for index in range(branch_count)]
    split = (
        f"[0:v]split={branch_count}"
        + "".join(f"[{label}]" for label in input_labels)
        + ";"
        if branch_count > 1
        else ""
    )
    for index in range(branch_count):
        label = f"c{index}"
        branches.append(f"[{input_labels[index]}]trim=start_frame=0:end_frame=1,setpts=PTS-STARTPTS,{legacy_camera_graph(plan, index)}[{label}]")
    if plan.operation == "two_rect_pan":
        graph = split + ";".join(branches) + ";" + "".join(f"[c{i}]" for i in range(plan.frame_count)) + f"concat=n={plan.frame_count}:v=1:a=0[camera]"
    else:
        graph = branches[0] + f";[c0]tpad=stop_mode=clone:stop={plan.frame_count - 1}[camera]"
    graph += ";[camera]settb=expr=1/4,setpts=N,split=2[camera_clock][camera_for_overlay];[1:v]format=rgba[underlay];[underlay][camera_for_overlay]overlay=eof_action=repeat:shortest=0:format=auto[mixed];[mixed]settb=expr=1/4,setpts=N,format=rgba[out]"
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-framerate", "4", "-loop", "1", "-i", str(source_path), "-f", "lavfi", "-i", f"color=c=0x17436d:s={plan.output_clip.width}x{plan.output_clip.height}:r=4", "-filter_complex", graph, "-map", "[out]", "-frames:v", str(plan.frame_count), "-vsync", "0", "-c:v", "ffv1", "-pix_fmt", "rgba", str(output_path), "-map", "[camera_clock]", "-frames:v", str(plan.frame_count), "-vsync", "0", "-c:v", "ffv1", "-pix_fmt", "rgba", str(camera_path)]
    completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
    if completed.returncode != 0:
        raise RuntimeError(f"legacy FFmpeg graph failed: {completed.stderr[-2000:]}")
    probe = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,time_base:frame=pts", "-of", "json", str(output_path)], capture_output=True, text=True, check=True, timeout=30)
    metadata = json.loads(probe.stdout)
    stream = metadata["streams"][0]
    numerator, denominator = (int(part) for part in stream["time_base"].split("/"))
    raw = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(output_path), "-f", "rawvideo", "-pix_fmt", "rgba", "-",], capture_output=True, check=True, timeout=30).stdout
    camera_raw = subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(camera_path), "-f", "rawvideo", "-pix_fmt", "rgba", "-",], capture_output=True, check=True, timeout=30).stdout
    frame_bytes = plan.output_clip.width * plan.output_clip.height * 4
    return LegacyRender(tuple(raw[offset:offset + frame_bytes] for offset in range(0, len(raw), frame_bytes)), tuple(camera_raw[offset:offset + frame_bytes] for offset in range(0, len(camera_raw), frame_bytes)), tuple(int(item["pts"]) for item in metadata.get("frames", [])), Fraction(numerator, denominator), int(stream["width"]), int(stream["height"]))


def ssim(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("SSIM inputs must be nonempty and equal")
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    variance_left = sum((value - mean_left) ** 2 for value in left) / len(left)
    variance_right = sum((value - mean_right) ** 2 for value in right) / len(right)
    covariance = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right, strict=True)) / len(left)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return ((2 * mean_left * mean_right + c1) * (2 * covariance + c2)) / ((mean_left**2 + mean_right**2 + c1) * (variance_left + variance_right + c2))


def decoded_metrics(reference: ReferenceFrame, observed: bytes) -> dict[str, float | int]:
    expected = list(reference.rgba8(composited=True))
    if len(observed) != len(expected):
        raise ValueError("legacy frame size differs from reference")
    rgb_expected = [float(value) for index, value in enumerate(expected) if index % 4 != 3]
    rgb_observed = [float(value) for index, value in enumerate(observed) if index % 4 != 3]
    luma_expected = [0.2126 * expected[i] + 0.7152 * expected[i + 1] + 0.0722 * expected[i + 2] for i in range(0, len(expected), 4)]
    luma_observed = [0.2126 * observed[i] + 0.7152 * observed[i + 1] + 0.0722 * observed[i + 2] for i in range(0, len(observed), 4)]
    return {"rgb_ssim": ssim(rgb_expected, rgb_observed), "luma_ssim": ssim(luma_expected, luma_observed), "max_rgb_abs_error": max(abs(a - b) for a, b in zip(rgb_expected, rgb_observed, strict=True)), "alpha_exact": int(all(observed[i] == expected[i] for i in range(3, len(observed), 4)))}


__all__ = [
    "Frame",
    "ReferenceFrame",
    "decoded_metrics",
    "evaluate",
    "ffmpeg_fast_div255",
    "ffmpeg_rgba8_opacity_alpha",
    "ffmpeg_rgba8_source_over_alpha",
    "load_frozen_plan",
    "legacy_camera_graph",
    "owns_integer_source_pixel",
    "run_legacy_cpu_graph",
]
