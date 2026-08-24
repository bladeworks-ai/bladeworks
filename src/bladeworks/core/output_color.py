"""Define the one color boundary shared by every render backend.

Architecture map
================

CPU/Vulkan full-range RGB working pixels
    -> Rec.709 RGB-to-YUV matrix
    -> limited-range delivery or mezzanine YUV
    -> encoder receives matching pixel values and frame metadata.

Why this exists
---------------
The semantic compositors intentionally work in full-range RGB. If the encoder
exit merely requests a YUV pixel format, FFmpeg may preserve the incoming
``full`` frame flag and make libx264 emit deprecated ``yuvj`` output. Keeping
the value conversion and range tag in one filter prevents CPU and Vulkan from
silently choosing different output contracts.
"""

from __future__ import annotations

from typing import Literal


OutputProfile = Literal["delivery", "oracle_mezzanine", "oracle_rgba"]


def limited_rec709_encoder_exit(output_profile: OutputProfile) -> str:
    """Return the explicit full-RGB to limited-Rec.709 encoder boundary.

    Main callers:
    - The CPU graph builder after its final semantic pixel module.
    - The Vulkan lowerer immediately after its one GPU download.
    """

    if output_profile == "delivery":
        pixel_format = "yuv420p"
    elif output_profile == "oracle_mezzanine":
        pixel_format = "yuv422p10le"
    elif output_profile == "oracle_rgba":
        return "format=rgba"
    else:  # pragma: no cover - public callers validate the profile first.
        raise ValueError(f"unknown output profile {output_profile!r}")
    return (
        "scale=in_range=full:out_range=limited:out_color_matrix=bt709,"
        f"format={pixel_format},"
        "setparams=range=limited:color_primaries=bt709:"
        "color_trc=bt709:colorspace=bt709"
    )


__all__ = ["OutputProfile", "limited_rec709_encoder_exit"]
