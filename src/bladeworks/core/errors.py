"""Explicit failure types for the portable FCPXML renderer."""


class FCPXMLRenderError(RuntimeError):
    """Base class for errors that should become a concise CLI failure."""


class FCPXMLParseError(FCPXMLRenderError):
    """The XML is unsafe, malformed, or outside the required document shape."""


class FCPXMLCompileError(FCPXMLRenderError):
    """The document parsed, but its references or timing are invalid."""


class RenderCapabilityError(FCPXMLRenderError):
    """The host FFmpeg/Pillow installation cannot execute the render plan."""


class FFmpegExecutionError(FCPXMLRenderError):
    """FFmpeg failed or produced an empty/unreadable output."""
