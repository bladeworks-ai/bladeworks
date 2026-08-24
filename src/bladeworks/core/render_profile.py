"""Render profiles for the legacy CPU builder (``ffmpeg.py``).

Architecture map
----------------
A ``RenderProfile`` bundles the pixel-pipeline decisions that trade Final Cut
parity evidence for speed.  Two profiles exist:

* ``reference`` -- the calibrated evidence pipeline: 16-bit linear-light
  working domain, geometry resampled in linear light through one affine
  ``perspective`` sampler at source resolution, two graph workers.  Its graph
  text is the byte-stable baseline every calibration test was frozen against.
* ``fast8`` -- the product/MVP pipeline: 8-bit encoded working domain (no
  per-composite gamma round trips), geometry scaled to the canvas first with
  stock ``scale``/``crop``/``pad`` (``perspective`` only for genuine
  rotations/corner pins, at canvas resolution), more graph workers, and a
  **two-format pixel policy**: every graph node carries ``gbrp`` (opaque) or
  ``gbrap`` (alpha), decoders enter as ``yuv420p`` through the one ``scale``
  pass, and the encoder converts once at the end.  ``check_filter_script``
  enforces the policy on the emitted graph text and fails the build loudly on
  any other pixel format, any ``lutrgb`` gamma pass, or an ``overlay`` left on
  ``format=auto`` (auto negotiation can pick subsampled 10-bit YUV).

The active profile is process-local state scoped by ``render_profile_scope``;
``build_invocation`` enters the scope for the whole build so the deeply nested
graph helpers (geometry, working-domain selection, compositor formats) do not
need a new parameter each.  Reading it is explicit: ``current_render_profile()``.

Product rules
-------------
* ``reference`` output must not change when ``fast8`` changes.
* ``fast8`` never silently falls back to ``reference`` behavior for a
  construct; a construct it cannot express is emitted at ``fast8`` precision
  through the shared perspective path, or fails loudly.

Main callers:
- ``ffmpeg.build_invocation`` / ``build_planned_invocation`` (scope entry).
- ``geometry.GeometryPlan`` (conform/perspective strategy).
- ``ffmpeg._working_domain`` and the compositor (working precision).
- ``executor.execute_render`` / ``cli`` (selection by name).
"""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator, Literal, Optional

RenderProfileName = Literal["reference", "fast8"]
Transfer = Literal["fcp_encoded", "fcp_linear"]
Precision = Literal[8, 16]
GeometryStrategy = Literal["perspective_linear_light", "scale_first"]

DEFAULT_ENCODER_PRESET = "medium"

# ``fast8`` pixel policy: the only formats a graph node may carry.
FAST8_OPAQUE_PIXEL_FORMAT = "gbrp"
FAST8_ALPHA_PIXEL_FORMAT = "gbrap"
FAST8_ALLOWED_PIXEL_FORMATS = frozenset(
    {FAST8_OPAQUE_PIXEL_FORMAT, FAST8_ALPHA_PIXEL_FORMAT}
)
# Tokens that can never appear in a ``fast8`` graph, wherever they occur.
_FAST8_BANNED_TOKENS = ("rgba64le", "gbrap16le", "gbrp16le", "rgb48le")
# Gamma-transfer LUT passes (``lutrgb`` with a ``pow`` curve) are the
# linear-light round trips the policy exists to remove.  A plain 8-bit
# per-channel ``lutrgb`` (e.g. posterize ``floor(val/4)*4``) is an ordinary
# one-pass 8-bit effect and stays allowed.
_FAST8_TRANSFER_LUT = re.compile(r"lutrgb=[^,\[;]*pow\(")


class RenderProfileError(RuntimeError):
    """The emitted graph violates the active profile's pixel policy."""


@dataclass(frozen=True)
class RenderProfile:
    """One complete set of legacy-builder pixel-pipeline decisions."""

    name: RenderProfileName
    working_transfer: Transfer
    working_precision: Precision
    geometry_strategy: GeometryStrategy
    filter_complex_threads: int
    encoder_preset: str = DEFAULT_ENCODER_PRESET
    # ``None`` = no policy (``reference`` keeps its calibrated formats);
    # otherwise the closed set of pixel formats a graph node may carry.
    allowed_pixel_formats: Optional[frozenset[str]] = None

    @property
    def working_pixel_format(self) -> str:
        """FFmpeg pixel format that carries the working domain."""

        return "rgba64le" if self.working_precision == 16 else FAST8_ALPHA_PIXEL_FORMAT

    @property
    def layer_pixel_format(self) -> str:
        """Pixel format a decoded layer chain ends in before compositing.

        ``reference`` keeps packed ``rgba`` (the calibrated chains convert to
        16-bit right after); ``fast8`` goes straight to planar ``gbrap`` so the
        ``overlay=format=gbrp`` composite consumes it with no conversion.
        """

        return "rgba" if self.allowed_pixel_formats is None else FAST8_ALPHA_PIXEL_FORMAT

    @property
    def overlay_format_option(self) -> str:
        """``overlay=format=...`` option for placement overlays.

        ``auto`` (reference) lets FFmpeg negotiate; ``fast8`` pins ``gbrp`` so
        planar RGB with alpha is composited without a hidden conversion.
        """

        return "auto" if self.allowed_pixel_formats is None else "gbrp"

    def normalize_alpha_format(self, script: str) -> str:
        """Map legacy packed ``rgba`` module tails onto this profile's alpha format.

        Why this exists:
        The legacy builder's 8-bit stock transition / effect modules were all
        authored to end in ``format=rgba`` -- 8-bit straight alpha, packed.
        Under the ``fast8`` two-format policy the same nodes must carry planar
        ``gbrap`` (identical meaning, different memory layout, and the layout
        the ``overlay=format=gbrp`` composite consumes without conversion).
        Rewriting the emitted text once here is one auditable rule instead of
        fifty scattered emitters; ``reference`` (no policy) is returned
        untouched, so its graph text stays byte-stable.

        Main callers:
        - ``ffmpeg._build_invocation_in_profile`` right before
          ``check_filter_script``.
        """

        if self.allowed_pixel_formats is None:
            return script
        return re.sub(
            r"format=rgba(?![0-9A-Za-z_])",
            f"format={FAST8_ALPHA_PIXEL_FORMAT}",
            script,
        )

    def check_filter_script(
        self, script: str, *, encoder_output_label: Optional[str] = None
    ) -> None:
        """Fail loudly when ``script`` breaks this profile's pixel policy.

        What it does (only when the profile has ``allowed_pixel_formats``):
        1. Any banned token (16-bit formats) or gamma-transfer ``lutrgb``
           (``pow`` curve) anywhere -> error.
        2. Every ``format=<fmt>`` *filter node* (start of a chain, after ``,``
           or after a ``[label]``) must name an allowed format.
        3. Every ``overlay`` ``:format=<opt>`` option must be ``gbrp`` --
           ``auto`` is rejected.
        4. The one chain that ends in ``[encoder_output_label]`` is the
           sanctioned encoder exit and may carry the encoder's ``yuv420p``.
        The error lists the offending graph lines so the construct that
        emitted them can be ported instead of silently degraded.

        Main callers:
        - ``ffmpeg._build_invocation_in_profile`` right after the graph text
          is assembled, before FFmpeg is invoked.
        """

        if self.allowed_pixel_formats is None:
            return
        offenders: list[str] = []
        encoder_suffix = (
            None if encoder_output_label is None else f"[{encoder_output_label}]"
        )
        for line in script.split(";"):
            text = line.strip()
            if not text:
                continue
            if encoder_suffix is not None and text.endswith(encoder_suffix):
                continue
            reasons: list[str] = []
            for token in _FAST8_BANNED_TOKENS:
                if token in text:
                    reasons.append(f"banned token {token!r}")
            if _FAST8_TRANSFER_LUT.search(text):
                reasons.append("gamma transfer lutrgb")
            for match in re.finditer(r"(?:^|[,\]])format=([A-Za-z0-9_]+)", text):
                if match.group(1) not in self.allowed_pixel_formats:
                    reasons.append(f"pixel format {match.group(1)!r}")
            for match in re.finditer(r":format=([A-Za-z0-9_]+)", text):
                if match.group(1) != FAST8_OPAQUE_PIXEL_FORMAT:
                    reasons.append(f"overlay format option {match.group(1)!r}")
            if reasons:
                offenders.append(f"{'; '.join(sorted(set(reasons)))}: {text[:200]}")
        if offenders:
            shown = "\n  ".join(offenders[:12])
            more = "" if len(offenders) <= 12 else f"\n  ... {len(offenders) - 12} more"
            raise RenderProfileError(
                f"render profile {self.name!r} allows only pixel formats "
                f"{sorted(self.allowed_pixel_formats)} but the graph emits "
                f"{len(offenders)} violating node(s):\n  {shown}{more}"
            )

    def with_encoder_preset(self, preset: str | None) -> "RenderProfile":
        if preset is None or preset == self.encoder_preset:
            return self
        return RenderProfile(
            name=self.name,
            working_transfer=self.working_transfer,
            working_precision=self.working_precision,
            geometry_strategy=self.geometry_strategy,
            filter_complex_threads=self.filter_complex_threads,
            encoder_preset=preset,
            allowed_pixel_formats=self.allowed_pixel_formats,
        )

    def manifest(self) -> dict[str, object]:
        return {
            "name": self.name,
            "working_transfer": self.working_transfer,
            "working_precision": self.working_precision,
            "geometry_strategy": self.geometry_strategy,
            "filter_complex_threads": self.filter_complex_threads,
            "encoder_preset": self.encoder_preset,
            "allowed_pixel_formats": (
                None
                if self.allowed_pixel_formats is None
                else sorted(self.allowed_pixel_formats)
            ),
        }


REFERENCE_PROFILE = RenderProfile(
    name="reference",
    working_transfer="fcp_linear",
    working_precision=16,
    geometry_strategy="perspective_linear_light",
    # Large real projects contain hundreds of format/scaler nodes. FFmpeg's
    # automatic complex-graph worker count can initialize many of them at once
    # and exhaust macOS thread/scaler resources before frame one. Two graph
    # workers retain parallelism while keeping initialization deterministic.
    filter_complex_threads=2,
)


def _fast8_threads() -> int:
    return max(2, min(8, os.cpu_count() or 2))


FAST8_PROFILE = RenderProfile(
    name="fast8",
    working_transfer="fcp_encoded",
    working_precision=8,
    geometry_strategy="scale_first",
    filter_complex_threads=_fast8_threads(),
    allowed_pixel_formats=FAST8_ALLOWED_PIXEL_FORMATS,
)

_PROFILES: dict[str, RenderProfile] = {
    REFERENCE_PROFILE.name: REFERENCE_PROFILE,
    FAST8_PROFILE.name: FAST8_PROFILE,
}

_ACTIVE: ContextVar[RenderProfile] = ContextVar(
    "fcpxml_render_profile", default=REFERENCE_PROFILE
)


def resolve_render_profile(name: str | RenderProfile) -> RenderProfile:
    """Return the named profile, failing loudly on unknown names."""

    if isinstance(name, RenderProfile):
        return name
    try:
        return _PROFILES[name]
    except KeyError as error:
        raise ValueError(
            f"unknown render profile {name!r}; expected one of {sorted(_PROFILES)}"
        ) from error


def current_render_profile() -> RenderProfile:
    """Return the profile active for the current build (default ``reference``)."""

    return _ACTIVE.get()


@contextmanager
def render_profile_scope(profile: RenderProfile) -> Iterator[RenderProfile]:
    """Activate ``profile`` for the dynamic extent of one graph build."""

    if not isinstance(profile, RenderProfile):
        raise TypeError("profile must be RenderProfile")
    token = _ACTIVE.set(profile)
    try:
        yield profile
    finally:
        _ACTIVE.reset(token)


__all__ = [
    "DEFAULT_ENCODER_PRESET",
    "FAST8_ALLOWED_PIXEL_FORMATS",
    "FAST8_ALPHA_PIXEL_FORMAT",
    "FAST8_OPAQUE_PIXEL_FORMAT",
    "FAST8_PROFILE",
    "REFERENCE_PROFILE",
    "RenderProfile",
    "RenderProfileError",
    "RenderProfileName",
    "current_render_profile",
    "render_profile_scope",
    "resolve_render_profile",
]
