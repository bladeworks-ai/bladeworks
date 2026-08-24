"""Packaged sample ``.fcpxmld`` projects for the ``bladeworks`` CLI.

Architecture / why this exists
------------------------------
These five tiny, self-contained ``.fcpxmld`` bundles are the ONE source of
truth for both:

1. The ``bladeworks examples ls`` / ``examples cp`` commands, which let a user
   discover and copy a renderable sample project out of the installed wheel.
2. The ``experimental_tests`` sanity fixtures that render one core mechanic
   each (``test_fcpxmld_fixtures.py``), which import :data:`EXAMPLES_DIR` and
   :data:`EXAMPLES` from here instead of pointing at a private fixtures path.

Each bundle is a directory ``<Name>.fcpxmld`` holding ``Info.fcpxml`` at its
root plus a ``Media/`` subfolder with BUNDLE-RELATIVE media ``src`` (e.g.
``src="Media/a.mp4"``). The media is committed (a few KB each, 160x90 / ~2s)
so the bundles are fully self-contained.

``EXAMPLES`` maps each sample name to a one-line human description and its
expected decoded frame count (sequence duration x 30 fps). The frame count
lives here so the CLI test and the fixtures test agree by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# Directory that physically holds the ``<name>.fcpxmld`` bundles. This is the
# single location every reader (CLI + tests) resolves samples from.
EXAMPLES_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Example:
    """One packaged sample project.

    - ``name``: the manifest key and the bundle stem (``<name>.fcpxmld``).
    - ``description``: one-line summary of the core mechanic it exercises.
    - ``expected_frames``: decoded video frame count when rendered at 30 fps,
      used by the sanity tests to assert a frame-exact render.
    """

    name: str
    description: str
    expected_frames: int

    @property
    def bundle(self) -> Path:
        """Absolute path to this sample's ``.fcpxmld`` bundle directory."""

        return EXAMPLES_DIR / f"{self.name}.fcpxmld"


# The manifest. Order is display order for ``bladeworks examples ls``.
EXAMPLES: dict[str, Example] = {
    example.name: example
    for example in (
        Example("single_clip", "One audio+video clip on the spine.", 30),
        Example("spine_and_lane", "Two spine clips plus a connected lane-1 clip.", 60),
        Example("transform", "A single clip with an adjust-transform (scale/position).", 30),
        Example("crop_or_transition", "Two clips joined by a cross-dissolve transition.", 60),
        Example("color", "A single clip with Color Adjustments applied.", 30),
    )
}


def example_bundle(name: str) -> Path:
    """Return the ``.fcpxmld`` bundle path for ``name`` or raise loudly.

    Main callers: ``bladeworks examples cp`` and the CLI tests. Raises
    ``KeyError`` with the known names when ``name`` is not in the manifest so
    the CLI can turn it into a loud user-facing error (never a silent miss).
    """

    if name not in EXAMPLES:
        known = ", ".join(sorted(EXAMPLES))
        raise KeyError(f"unknown example {name!r}; known examples: {known}")
    return EXAMPLES[name].bundle
