"""Explicit failure types for the portable FCPXML renderer."""


class FCPXMLRenderError(RuntimeError):
    """Base class for errors that should become a concise CLI failure."""


class FCPXMLParseError(FCPXMLRenderError):
    """The XML is unsafe, malformed, or outside the required document shape."""


class FCPXMLCompileError(FCPXMLRenderError):
    """The document parsed, but its references or timing are invalid."""


class RenderCapabilityError(FCPXMLRenderError):
    """The host FFmpeg/Pillow installation cannot execute the render plan."""


class MediaConsolidationError(FCPXMLRenderError):
    """`--symlink-media` could not consolidate a bundle's referenced media.

    Raised for a hard, unrecoverable condition (input is not a bundle, or two
    distinct source files would collide on one ``Media/<name>`` link). Never
    raised for merely-offline media: that is reported as a warning and the rest
    of the bundle is still consolidated.
    """


class FCPXMLWriteConflictError(FCPXMLRenderError):
    """A document rewrite was refused because the file changed since it was read.

    Raised by ``core/media_consolidate.write_fcpxml_document`` when the bytes on
    disk no longer match the bytes the mutating command (``fcpxml proxy``,
    ``--symlink-media``) parsed. Publishing the stale tree would silently drop
    whatever Final Cut, Studio, or another command wrote in between.
    """


class FFmpegExecutionError(FCPXMLRenderError):
    """FFmpeg failed or produced an empty/unreadable output."""
