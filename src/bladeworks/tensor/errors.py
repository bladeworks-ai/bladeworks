"""Errors for the tensor renderer.

Why these derive from the renderer's shared error tree
-----------------------------------------------------
``cli.main`` catches ``FCPXMLRenderError`` and ``executor.execute_render``
catches it to write the ``failed`` manifest and unlink the partial output.
Bare ``RuntimeError``s escape both, so a tensor rejection would exit with a
traceback and leave the pre-flight ``preparing`` manifest on disk. Deriving
from the existing types puts tensor failures on exactly the same reporting
path as every other backend:

- ``TensorRenderUnsupported`` is a *capability* verdict decided before any
  frame is decoded, so it derives from ``RenderCapabilityError`` (the same
  class the Vulkan gate and the segmentation gate raise).
- ``TensorRenderError`` is an *execution* failure (decode, encode, device),
  so it derives from ``FFmpegExecutionError`` -- the class the executor
  already treats as "the render ran and produced nothing usable".
"""

from ..core.errors import FFmpegExecutionError, RenderCapabilityError


class TensorRenderUnsupported(RenderCapabilityError):
    """A compiled construct is outside the tensor renderer's supported class.

    Raised at plan time, before any frame is decoded, so a project either
    renders fully or fails loudly with the construct named.
    """


class TensorRenderError(FFmpegExecutionError):
    """A runtime failure inside the tensor renderer (decode/encode/device)."""
