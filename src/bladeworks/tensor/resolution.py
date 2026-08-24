"""Typed output-resolution policy for Bladeworks render, seek, and scan modes.

Architecture map
================

    API profile name (``1080p`` / ``720p`` / ``540p`` / ``480p``)
        -> ``ResolutionProfile`` validates the public value
        -> ``resolve_output_resolution`` fits the project's authored raster
           inside the profile envelope
        -> ``OutputResolution`` carries both the integer output raster and the
           exact X/Y coordinate transforms needed by tensor-plan lowering

Product rules
-------------
* Export render defaults to 1080p and supports all four profiles.
* Interactive seek and scan use a fixed user-selected 720p, 540p, or 480p
  profile. Slow graphs buffer instead of silently changing profiles.
* A profile is an aspect-preserving envelope, not a request to stretch the
  project to one fixed raster.
* Output dimensions are even for YUV 4:2:0 encoders.
* Smaller projects are never enlarged. Odd source dimensions may lose one
  edge pixel because an even encoder raster is required.

Plan integration boundary
-------------------------
This module does not mutate ``RenderDocument`` or an already-built
``TensorRenderPlan``. Preview resolution changes the coordinate space in
which composition happens, so changing only the root canvas would be wrong.
``build_tensor_plan(..., output_resolution=...)`` therefore consistently:

1. use ``width`` / ``height`` for the root canvas and transition contexts;
2. transform project-edge coordinates with ``scale_x`` / ``scale_y`` in every
   root-owned ``FrameGeometry`` and ``canvas_to_owner`` matrix;
3. scale rendered scope canvases relative to their parent coordinate space;
4. samples authored-resolution title, caption, and generator rasters into the
   selected output while scaling geometric mask coordinates;
5. keeps explicitly fixed-pixel effect kernels literal in the selected output
   raster, as documented by the preview PRD.

Source media dimensions and authored timeline clocks remain unchanged. This
keeps conform, crop, transform, animation, retime, and transition timing
semantics owned by the existing exact kernels.

Main callers:
- The localhost API when validating mode/profile combinations.
- ``build_tensor_plan(..., output_resolution=...)`` and Tensor render sessions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from types import MappingProxyType
from typing import Final, Mapping


class ResolutionProfile(str, Enum):
    """Stable public names for supported output envelopes."""

    P1080 = "1080p"
    P720 = "720p"
    P540 = "540p"
    P480 = "480p"


class RenderMode(str, Enum):
    """The three API operations that select a resolution policy."""

    RENDER = "render"
    SEEK = "seek"
    SCAN = "scan"


@dataclass(frozen=True)
class ResolutionEnvelope:
    """Maximum landscape raster for one named profile.

    Portrait projects use the transposed envelope. Square projects use the
    landscape envelope, which has the same limiting short edge.
    """

    width: int
    height: int

    def for_source(self, source_width: int, source_height: int) -> tuple[int, int]:
        if source_width >= source_height:
            return self.width, self.height
        return self.height, self.width


@dataclass(frozen=True)
class OutputResolution:
    """Resolved encoder raster and project-to-output coordinate transform.

    ``scale_x`` and ``scale_y`` are kept separately because forcing dimensions
    to even integers can make them differ by a tiny amount. Plan lowering must
    use these actual ratios instead of the nominal floating-point fit scale.
    """

    profile: ResolutionProfile
    source_width: int
    source_height: int
    width: int
    height: int

    @property
    def scale_x(self) -> float:
        return self.width / self.source_width

    @property
    def scale_y(self) -> float:
        return self.height / self.source_height

    @property
    def was_downscaled(self) -> bool:
        return self.width < self.source_width or self.height < self.source_height


PROFILE_ENVELOPES: Final[Mapping[ResolutionProfile, ResolutionEnvelope]] = MappingProxyType(
    {
        ResolutionProfile.P1080: ResolutionEnvelope(1920, 1080),
        ResolutionProfile.P720: ResolutionEnvelope(1280, 720),
        ResolutionProfile.P540: ResolutionEnvelope(960, 540),
        ResolutionProfile.P480: ResolutionEnvelope(854, 480),
    }
)

SUPPORTED_PROFILES: Final[Mapping[RenderMode, tuple[ResolutionProfile, ...]]] = MappingProxyType(
    {
        RenderMode.RENDER: (
            ResolutionProfile.P1080,
            ResolutionProfile.P720,
            ResolutionProfile.P540,
            ResolutionProfile.P480,
        ),
        RenderMode.SEEK: (
            ResolutionProfile.P720,
            ResolutionProfile.P540,
            ResolutionProfile.P480,
        ),
        RenderMode.SCAN: (
            ResolutionProfile.P720,
            ResolutionProfile.P540,
            ResolutionProfile.P480,
        ),
    }
)

DEFAULT_PROFILE: Final[Mapping[RenderMode, ResolutionProfile]] = MappingProxyType(
    {
        RenderMode.RENDER: ResolutionProfile.P1080,
        RenderMode.SEEK: ResolutionProfile.P720,
        RenderMode.SCAN: ResolutionProfile.P720,
    }
)


def profile_for_mode(
    mode: RenderMode | str,
    profile: ResolutionProfile | str | None = None,
) -> ResolutionProfile:
    """Return a validated explicit profile or the mode's product default.

    Main callers:
    - HTTP request parsing before a render session or plan is allocated.
    """

    resolved_mode = RenderMode(mode)
    resolved_profile = DEFAULT_PROFILE[resolved_mode] if profile is None else ResolutionProfile(profile)
    if resolved_profile not in SUPPORTED_PROFILES[resolved_mode]:
        supported = ", ".join(item.value for item in SUPPORTED_PROFILES[resolved_mode])
        raise ValueError(
            f"{resolved_profile.value} is not supported for {resolved_mode.value}; "
            f"supported profiles: {supported}"
        )
    return resolved_profile


def resolve_output_resolution(
    source_width: int,
    source_height: int,
    profile: ResolutionProfile | str,
) -> OutputResolution:
    """Fit a project raster inside a named envelope without upscaling it.

    Dimensions are floored to an even integer after the uniform aspect fit.
    Flooring guarantees the result never exceeds either the profile envelope
    or the source raster. The aspect ratio is preserved to integer-pixel
    precision; ``OutputResolution.scale_x`` / ``scale_y`` capture the exact
    coordinate mapping after even rounding.

    Main callers:
    - Tensor-plan/session creation after API profile validation.
    """

    if isinstance(source_width, bool) or isinstance(source_height, bool):
        raise TypeError("source dimensions must be integers, not bool")
    if not isinstance(source_width, int) or not isinstance(source_height, int):
        raise TypeError("source dimensions must be integers")
    if source_width < 2 or source_height < 2:
        raise ValueError("source dimensions must each be at least 2 pixels")

    resolved_profile = ResolutionProfile(profile)
    maximum_width, maximum_height = PROFILE_ENVELOPES[resolved_profile].for_source(
        source_width, source_height
    )
    scale = min(
        Fraction(1),
        Fraction(maximum_width, source_width),
        Fraction(maximum_height, source_height),
    )

    width = _floor_even(source_width * scale)
    height = _floor_even(source_height * scale)
    if width < 2 or height < 2:
        raise ValueError(
            f"{source_width}x{source_height} cannot fit inside the "
            f"{resolved_profile.value} envelope as an even-pixel raster"
        )
    return OutputResolution(
        profile=resolved_profile,
        source_width=source_width,
        source_height=source_height,
        width=width,
        height=height,
    )


def _floor_even(value: Fraction) -> int:
    """Largest positive even integer no greater than ``value``."""

    return (value.numerator // value.denominator) // 2 * 2


__all__ = [
    "DEFAULT_PROFILE",
    "PROFILE_ENVELOPES",
    "SUPPORTED_PROFILES",
    "OutputResolution",
    "RenderMode",
    "ResolutionEnvelope",
    "ResolutionProfile",
    "profile_for_mode",
    "resolve_output_resolution",
]
