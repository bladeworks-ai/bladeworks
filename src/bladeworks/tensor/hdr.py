"""GPU application of the renderer-owned HDR-to-SDR color-conform LUTs.

Architecture map
================

1. Load and validate the frozen 17x17x17 ``.cube`` artifact once on the CPU.
2. Cache one flattened copy per Torch device and HDR transfer.
3. Apply FFmpeg-compatible tetrahedral interpolation to each RGB frame.

The LUT artifacts are shared with ``core.spatial_intrinsics``. They encode the
declared semantic approximation used by the renderer: Rec.2020 HLG or PQ to a
100-nit Rec.709 SDR target. This module only evaluates those frozen values; it
does not choose a tone-mapping policy or regenerate the artifacts.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import torch

from .errors import TensorRenderError


HDRTransfer = Literal["hlg", "pq"]
_LUT_ROOT = Path(__file__).resolve().parents[1] / "spatial_luts"
_LUT_PATHS: dict[HDRTransfer, Path] = {
    "hlg": _LUT_ROOT / "rec2020_hlg_to_rec709_sdr_v1.cube",
    "pq": _LUT_ROOT / "rec2020_pq_to_rec709_sdr_v1.cube",
}
_DEVICE_LUTS: dict[tuple[HDRTransfer, str], tuple[torch.Tensor, int]] = {}


@lru_cache(maxsize=2)
def _load_cube(transfer: HDRTransfer) -> tuple[torch.Tensor, int]:
    """Load one renderer-owned cube as ``[blue, green, red, RGB]`` values."""

    path = _LUT_PATHS[transfer]
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except OSError as exc:
        raise TensorRenderError(f"could not read HDR color-conform LUT {path}: {exc}") from exc
    size = None
    values: list[list[float]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("LUT_3D_SIZE "):
            size = int(stripped.split()[1])
            continue
        if stripped[0].isdigit() or stripped[0] in "+-.":
            parts = stripped.split()
            if len(parts) == 3:
                values.append([float(part) for part in parts])
    if size is None or size < 2:
        raise TensorRenderError(f"HDR color-conform LUT {path} has no valid LUT_3D_SIZE")
    if len(values) != size**3:
        raise TensorRenderError(
            f"HDR color-conform LUT {path} has {len(values)} entries; expected {size**3}"
        )
    return torch.tensor(values, dtype=torch.float32).view(size, size, size, 3), size


def _device_lut(transfer: HDRTransfer, device: torch.device) -> tuple[torch.Tensor, int]:
    """Return the flattened LUT cached on ``device`` for frame-time gathers."""

    key = (transfer, str(device))
    cached = _DEVICE_LUTS.get(key)
    if cached is None:
        cube, size = _load_cube(transfer)
        cached = (cube.reshape(-1, 3).to(device), size)
        _DEVICE_LUTS[key] = cached
    return cached


def hdr_to_sdr(rgb: torch.Tensor, transfer: HDRTransfer) -> torch.Tensor:
    """Map code-space Rec.2020 HDR RGB ``[3,H,W]`` to code-space Rec.709 SDR.

    Main callers:
    - ``decode.ClipDecoder.frame_at`` in the direct render path.
    - ``pipeline._LayerWorker.pop`` in the prefetched render path.

    Tetrahedral interpolation follows the same four-vertex construction as
    FFmpeg's ``lut3d=interp=tetrahedral``. Sorting each pixel's three fractional
    coordinates identifies its tetrahedron without constructing six complete
    candidate frames.
    """

    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise TensorRenderError(f"HDR LUT expects RGB [3,H,W], got {tuple(rgb.shape)}")
    lut, size = _device_lut(transfer, rgb.device)
    coordinates = rgb.clamp(0.0, 1.0) * (size - 1)
    lower = coordinates.floor().to(torch.int64).clamp_(0, size - 2)
    fractions = coordinates - lower.to(coordinates.dtype)
    order = torch.argsort(fractions, dim=0, descending=True)
    sorted_fractions = torch.gather(fractions, 0, order)

    strides = torch.tensor((1, size, size * size), dtype=torch.int64, device=rgb.device)
    first_stride = strides[order[0]]
    second_stride = strides[order[1]]
    base = lower[0] + size * lower[1] + size * size * lower[2]
    corner0 = lut[base]
    corner1 = lut[base + first_stride]
    corner2 = lut[base + first_stride + second_stride]
    corner3 = lut[base + 1 + size + size * size]

    first = sorted_fractions[0].unsqueeze(-1)
    second = sorted_fractions[1].unsqueeze(-1)
    third = sorted_fractions[2].unsqueeze(-1)
    result = (
        corner0
        + first * (corner1 - corner0)
        + second * (corner2 - corner1)
        + third * (corner3 - corner2)
    )
    return result.permute(2, 0, 1).contiguous().clamp_(0.0, 1.0)


__all__ = ["HDRTransfer", "hdr_to_sdr"]
