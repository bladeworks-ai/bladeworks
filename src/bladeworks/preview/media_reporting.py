"""Translate compiled missing-media facts into the public SSE payload."""

from __future__ import annotations

from typing import Any

from ..core.missing_media import missing_media_basename
from ..core.model import RenderDocument


def missing_media_event_data(
    document: RenderDocument,
    *,
    source_version: str,
) -> dict[str, object] | None:
    """Group exact offline locators while retaining every timeline reference.

    Main callers:
    - source compilation and preview session create/sync flows before emitting
      the replayable ``missing_media`` SSE event.
    """

    grouped: dict[str, dict[str, Any]] = {}
    for reference in document.missing_media_references:
        entry = grouped.setdefault(
            reference.locator,
            {
                "locator": reference.locator,
                "basename": missing_media_basename(reference.locator),
                "hasVideo": False,
                "hasAudio": False,
                "references": {},
            },
        )
        entry["hasVideo"] = bool(entry["hasVideo"] or reference.has_video)
        entry["hasAudio"] = bool(entry["hasAudio"] or reference.has_audio)
        key = (
            reference.fcpxml_path,
            reference.timeline_start,
            reference.timeline_duration,
        )
        entry["references"][key] = {
            "fcpxmlPath": reference.fcpxml_path,
            "start": float(reference.timeline_start),
            "duration": float(reference.timeline_duration),
        }
    if not grouped:
        return None
    paths: list[dict[str, object]] = []
    for locator in sorted(grouped):
        entry = grouped[locator]
        references = entry.pop("references")
        entry["references"] = [references[key] for key in sorted(references)]
        paths.append(entry)
    return {"sourceVersion": source_version, "paths": paths}
