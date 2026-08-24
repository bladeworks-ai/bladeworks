"""Backend-agnostic front end and shared semantics: FCPXML text in, a typed plan out.

Architecture map
================

    .fcpxml text
        -> parser.py         : XML -> source model (no semantics, no rendering)
        -> capabilities.py   : the bounded registry of Final Cut UIDs we accept
        -> compiler.py       : source model -> RenderDocument (+ CompatibilityReport)
        -> composition_ir.py : the recursive CompositionPlan both pixel backends lower from
        -> model.py          : the frozen types every backend consumes

Alongside the front end sit the rules that are *shared* rather than emitter-specific:
``compositor.py`` (alpha / blend / source-over), ``geometry.py`` and
``spatial_intrinsics.py``, ``retime*.py``, ``animation*.py``, ``audio_*.py``,
``text*.py``, ``color.py`` / ``output_color.py``, ``pixel_domains.py``,
``render_profile.py``, and ``filter_text.py`` -- the one definition of how a number or
a duration is spelled in emitted FFmpeg text, which both backends must agree on
character for character.

Invariant
---------
**``core`` imports no backend.**  Not ``tensor``, not ``legacy_ffmpeg``, not ``vulkan``,
not ``research``.  The arrow points one way, from a backend into ``core``; a helper only
one backend needs belongs in that backend.  ``experimental_tests/core/
test_package_boundaries.py`` fails the build if this stops being true.

Why this exists
---------------
Before the split, backend-agnostic semantics lived at the package root next to a
12k-line FFmpeg emitter, so "is this a shared rule or an FFmpeg detail?" could only be
answered by reading the file.  The package boundary answers it now, and the boundary
test keeps the answer honest.
"""
