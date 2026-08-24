"""Compile explicit pixel-domain boundaries for FFmpeg filter graphs.

Architecture map
================

``PixelDomain``
    -> describes what the pixel values mean, not merely their FFmpeg format
    -> includes transfer, alpha association, precision, dimensions, and clock

``PixelDomainCompiler.adapt``
    -> connects two semantic modules directly when their domains match
    -> emits the smallest reviewed conversion when their pixel meanings differ
    -> rejects implicit resizing or retiming because those are semantic stages

``ConversionRecord``
    -> records every genuine boundary for graph audits and real-project metrics

Composition modules remain independent: each module declares its input and
output contracts.  The compiler may fuse their transport representation, but
it never removes or reorders the semantic operation itself.  This separates
Final Cut's composition tree from FFmpeg's pixel-format plumbing.

Important invariants
--------------------

* Equal contracts connect without a filter or a new label.
* Dimensions and frame clocks never change as an accidental format conversion.
* The calibrated Final Cut transfer is applied to RGB only; alpha is unchanged.
* Straight/premultiplied changes are explicit and are never inferred from an
  FFmpeg pixel-format name.
* Every emitted conversion is inspectable and countable.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import re
from typing import Literal, TypeAlias


ColorTransfer: TypeAlias = Literal["fcp_encoded", "fcp_linear"]
AlphaAssociation: TypeAlias = Literal["straight", "premultiplied"]
ChannelPrecision: TypeAlias = Literal[8, 16]


class PixelDomainError(ValueError):
    """A module requested an invalid or implicit pixel-domain transition."""


def _label(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_]+", value):
        raise PixelDomainError(
            f"{name} must contain only ASCII letters, digits, or underscores"
        )
    return value


def _number(value: float) -> str:
    if not math.isfinite(value):
        raise PixelDomainError("transfer exponent must be finite")
    return format(value, ".12g")


def _fraction_manifest(value: Fraction) -> dict[str, int]:
    """Return an exact rational without converting through binary float."""

    return {"numerator": value.numerator, "denominator": value.denominator}


@dataclass(frozen=True)
class FrameClock:
    """The exact dimensions of a stream's timeline coordinate system."""

    frame_duration: Fraction
    duration: Fraction
    pts_origin: Fraction = Fraction(0)

    def __post_init__(self) -> None:
        for name in ("frame_duration", "duration", "pts_origin"):
            value = getattr(self, name)
            if not isinstance(value, Fraction):
                raise PixelDomainError(f"{name} must be an exact Fraction")
        if self.frame_duration <= 0 or self.duration <= 0:
            raise PixelDomainError("frame duration and stream duration must be positive")
        if self.duration / self.frame_duration != int(
            self.duration / self.frame_duration
        ):
            raise PixelDomainError("stream duration must contain an exact frame count")

    @property
    def frame_count(self) -> int:
        return int(self.duration / self.frame_duration)

    def manifest(self) -> dict[str, object]:
        """Return one JSON-safe exact clock contract."""

        return {
            "frame_duration": _fraction_manifest(self.frame_duration),
            "duration": _fraction_manifest(self.duration),
            "pts_origin": _fraction_manifest(self.pts_origin),
            "frame_count": self.frame_count,
        }


@dataclass(frozen=True)
class PixelDomain:
    """The complete contract carried by one video link between modules."""

    transfer: ColorTransfer
    alpha: AlphaAssociation
    precision: ChannelPrecision
    width: int
    height: int
    clock: FrameClock

    def __post_init__(self) -> None:
        if self.transfer not in {"fcp_encoded", "fcp_linear"}:
            raise PixelDomainError(f"unknown color transfer {self.transfer!r}")
        if self.alpha not in {"straight", "premultiplied"}:
            raise PixelDomainError(f"unknown alpha association {self.alpha!r}")
        if self.precision not in {8, 16}:
            raise PixelDomainError("channel precision must be 8 or 16 bits")
        if (
            isinstance(self.width, bool)
            or isinstance(self.height, bool)
            or not isinstance(self.width, int)
            or not isinstance(self.height, int)
            or min(self.width, self.height) <= 0
        ):
            raise PixelDomainError("pixel dimensions must be positive integers")
        if not isinstance(self.clock, FrameClock):
            raise PixelDomainError("clock must be FrameClock")

    def with_pixels(
        self,
        *,
        transfer: ColorTransfer | None = None,
        alpha: AlphaAssociation | None = None,
        precision: ChannelPrecision | None = None,
    ) -> "PixelDomain":
        """Return the same surface and clock with a different pixel meaning."""

        return PixelDomain(
            transfer=transfer or self.transfer,
            alpha=alpha or self.alpha,
            precision=precision or self.precision,
            width=self.width,
            height=self.height,
            clock=self.clock,
        )

    def manifest(self) -> dict[str, object]:
        """Return the complete JSON-safe link contract."""

        return {
            "transfer": self.transfer,
            "alpha": self.alpha,
            "precision": self.precision,
            "width": self.width,
            "height": self.height,
            "clock": self.clock.manifest(),
        }


@dataclass(frozen=True)
class ConversionRecord:
    """One explicit representation boundary inserted between semantic modules."""

    source_label: str
    output_label: str
    source: PixelDomain
    target: PixelDomain
    filters: tuple[str, ...]

    def manifest(self) -> dict[str, object]:
        """Return an auditable JSON-safe representation barrier."""

        return {
            "source_label": self.source_label,
            "output_label": self.output_label,
            "source": self.source.manifest(),
            "target": self.target.manifest(),
            "filters": self.filters,
        }


@dataclass(frozen=True)
class CompositionModuleContract:
    """Pixel contracts for one independent two-input composition operation.

    The contract is deliberately separate from graph optimization.  Tests can
    inspect one Final Cut composition node even when its links are fused with
    compatible parents and children in the emitted FFmpeg graph.
    """

    module_id: str
    lower: PixelDomain
    foreground: PixelDomain
    output: PixelDomain

    def __post_init__(self) -> None:
        _label(self.module_id, name="module_id")
        domains = (self.lower, self.foreground, self.output)
        if not all(isinstance(domain, PixelDomain) for domain in domains):
            raise PixelDomainError("composition contracts require PixelDomain values")
        surfaces = {(domain.width, domain.height) for domain in domains}
        clocks = {domain.clock for domain in domains}
        if len(surfaces) != 1:
            raise PixelDomainError(
                "composition module inputs and output must share one surface"
            )
        if len(clocks) != 1:
            raise PixelDomainError(
                "composition module inputs and output must share one frame clock"
            )

    def manifest(self) -> dict[str, object]:
        """Return one independently inspectable semantic composition module."""

        return {
            "module_id": self.module_id,
            "lower": self.lower.manifest(),
            "foreground": self.foreground.manifest(),
            "output": self.output.manifest(),
        }


class PixelDomainCompiler:
    """Connect semantic graph modules using the minimum valid conversions.

    Main callers:
    - The FCPXML FFmpeg graph builder while folding layers and groups.
    - Structural regression tests and Yunah benchmark instrumentation.

    Why this exists:
    A composition node used to normalize, linearize, composite, encode, and
    normalize again even when its parent immediately repeated the inverse
    operations.  This compiler owns those boundaries globally.  Semantic
    modules remain separate objects/calls, while compatible links stay in one
    shared working representation.
    """

    def __init__(self, *, calibrated_gamma: float) -> None:
        if not math.isfinite(calibrated_gamma) or calibrated_gamma <= 0:
            raise PixelDomainError("calibrated_gamma must be positive and finite")
        self.calibrated_gamma = calibrated_gamma
        self._domains: dict[str, PixelDomain] = {}
        self._conversions: list[ConversionRecord] = []
        self._composition_modules: list[CompositionModuleContract] = []
        self._fused_connections = 0

    @property
    def conversions(self) -> tuple[ConversionRecord, ...]:
        return tuple(self._conversions)

    @property
    def composition_modules(self) -> tuple[CompositionModuleContract, ...]:
        return tuple(self._composition_modules)

    @property
    def fused_connection_count(self) -> int:
        return self._fused_connections

    def register_composition(self, contract: CompositionModuleContract) -> None:
        """Record one semantic module independently of representation fusion."""

        if not isinstance(contract, CompositionModuleContract):
            raise PixelDomainError("contract must be CompositionModuleContract")
        if any(
            existing.module_id == contract.module_id
            for existing in self._composition_modules
        ):
            raise PixelDomainError(
                f"composition module {contract.module_id!r} was already registered"
            )
        self._composition_modules.append(contract)

    def manifest(self) -> dict[str, object]:
        """Return stable metrics plus every auditable module and boundary."""

        return {
            "composition_module_count": len(self._composition_modules),
            "conversion_count": len(self._conversions),
            "transfer_conversion_count": sum(
                record.source.transfer != record.target.transfer
                for record in self._conversions
            ),
            "alpha_conversion_count": sum(
                record.source.alpha != record.target.alpha
                for record in self._conversions
            ),
            "precision_conversion_count": sum(
                record.source.precision != record.target.precision
                for record in self._conversions
            ),
            "fused_connection_count": self._fused_connections,
            "composition_modules": tuple(
                contract.manifest() for contract in self._composition_modules
            ),
            "conversions": tuple(
                conversion.manifest() for conversion in self._conversions
            ),
        }

    def register(self, label: str, domain: PixelDomain) -> None:
        """Bind one module output label to its declared contract."""

        clean = _label(label, name="label")
        if not isinstance(domain, PixelDomain):
            raise PixelDomainError("domain must be PixelDomain")
        previous = self._domains.get(clean)
        if previous is not None and previous != domain:
            raise PixelDomainError(
                f"label {clean!r} was already registered with another pixel domain"
            )
        self._domains[clean] = domain

    def domain(self, label: str) -> PixelDomain:
        clean = _label(label, name="label")
        try:
            return self._domains[clean]
        except KeyError as error:
            raise PixelDomainError(f"label {clean!r} has no pixel-domain contract") from error

    def adapt(
        self,
        lines: list[str],
        *,
        source_label: str,
        target: PixelDomain,
        output_label: str,
    ) -> str:
        """Return a label satisfying ``target``, emitting a real barrier if needed.

        Equal domains return ``source_label`` directly.  Resizing and retiming
        are rejected because the corresponding semantic geometry/timing module
        must own them; silently hiding either inside a color conversion would
        make composition fusion change project behavior.
        """

        source_name = _label(source_label, name="source_label")
        output_name = _label(output_label, name="output_label")
        source = self.domain(source_name)
        if source == target:
            self._fused_connections += 1
            return source_name
        if (source.width, source.height) != (target.width, target.height):
            raise PixelDomainError(
                "pixel-domain adaptation cannot resize a semantic surface"
            )
        if source.clock != target.clock:
            raise PixelDomainError(
                "pixel-domain adaptation cannot retime a semantic stream"
            )

        filters: list[str] = []
        current_alpha = source.alpha
        current_transfer = source.transfer
        current_precision = source.precision

        # Transfer functions operate on straight RGB.  This ordering avoids
        # applying a nonlinear curve to premultiplied color values.
        if current_alpha == "premultiplied" and (
            current_transfer != target.transfer
            or current_precision != target.precision
        ):
            filters.append("unpremultiply=inplace=1:planes=7")
            current_alpha = "straight"

        if current_transfer != target.transfer:
            filters.append("format=rgba64le")
            current_precision = 16
            exponent = (
                self.calibrated_gamma
                if target.transfer == "fcp_linear"
                else 1.0 / self.calibrated_gamma
            )
            expression = f"maxval*pow(val/maxval,{_number(exponent)})"
            filters.append(
                f"lutrgb=r='{expression}':g='{expression}':b='{expression}'"
            )
            current_transfer = target.transfer

        if current_precision != target.precision:
            filters.append("format=rgba64le" if target.precision == 16 else "format=gbrap")
            current_precision = target.precision

        if current_alpha != target.alpha:
            filters.append(
                "premultiply=inplace=1:planes=7"
                if target.alpha == "premultiplied"
                else "unpremultiply=inplace=1:planes=7"
            )
            current_alpha = target.alpha

        if target.precision == 16 and not any(
            item == "format=rgba64le" for item in filters
        ):
            filters.append("format=rgba64le")
        if target.precision == 8 and not any(
            item == "format=gbrap" for item in filters
        ):
            filters.append("format=gbrap")

        if not filters:
            raise PixelDomainError("different pixel domains produced no conversion")
        lines.append(f"[{source_name}]{','.join(filters)}[{output_name}]")
        self.register(output_name, target)
        self._conversions.append(
            ConversionRecord(
                source_label=source_name,
                output_label=output_name,
                source=source,
                target=target,
                filters=tuple(filters),
            )
        )
        return output_name


__all__ = [
    "AlphaAssociation",
    "ChannelPrecision",
    "ColorTransfer",
    "CompositionModuleContract",
    "ConversionRecord",
    "FrameClock",
    "PixelDomain",
    "PixelDomainCompiler",
    "PixelDomainError",
]
