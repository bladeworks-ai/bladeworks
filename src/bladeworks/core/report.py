"""Structured compatibility reporting for every approximation and omission."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Optional

from .model import fraction_json


@dataclass(frozen=True)
class Finding:
    outcome: str
    portable_status: str
    fcpxml_path: str
    construct: str
    disposition: str
    timeline_start: Optional[Fraction] = None
    timeline_duration: Optional[Fraction] = None
    uid: Optional[str] = None

    def to_json(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "portable_status": self.portable_status,
            "fcpxml_path": self.fcpxml_path,
            "construct": self.construct,
            "uid": self.uid,
            "timeline_start": fraction_json(self.timeline_start),
            "timeline_duration": fraction_json(self.timeline_duration),
            "disposition": self.disposition,
        }


@dataclass
class CompatibilityReport:
    schema_version: int = 1
    source_path: Optional[str] = None
    source_sha256: Optional[str] = None
    project_name: Optional[str] = None
    timeline_start: Optional[Fraction] = None
    timeline_duration: Optional[Fraction] = None
    findings: list[Finding] = field(default_factory=list)

    def add(
        self,
        *,
        outcome: str,
        portable_status: str,
        fcpxml_path: str,
        construct: str,
        disposition: str,
        timeline_start: Optional[Fraction] = None,
        timeline_duration: Optional[Fraction] = None,
        uid: Optional[str] = None,
    ) -> None:
        self.findings.append(
            Finding(
                outcome=outcome,
                portable_status=portable_status,
                fcpxml_path=fcpxml_path,
                construct=construct,
                disposition=disposition,
                timeline_start=timeline_start,
                timeline_duration=timeline_duration,
                uid=uid,
            )
        )

    @property
    def degraded(self) -> bool:
        return any(
            item.outcome in {"approximated", "omitted"}
            or item.portable_status == "degraded"
            for item in self.findings
        )

    @property
    def has_strict_failures(self) -> bool:
        return any(
            item.outcome in {"approximated", "omitted", "failed"}
            for item in self.findings
        )

    def counts(self) -> dict[str, int]:
        values = {"exact": 0, "approximated": 0, "omitted": 0, "failed": 0, "info": 0}
        for item in self.findings:
            values[item.outcome] = values.get(item.outcome, 0) + 1
        return values

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "project_name": self.project_name,
            "timeline_start": fraction_json(self.timeline_start),
            "timeline_duration": fraction_json(self.timeline_duration),
            "degraded": self.degraded,
            "counts": self.counts(),
            "findings": [item.to_json() for item in self.findings],
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def human_summary(self) -> str:
        if not self.findings:
            return "FCPXML compatibility: no findings"
        lines = [f"FCPXML compatibility: {self.counts()}" ]
        order = {"failed": 0, "omitted": 1, "approximated": 2, "info": 3, "exact": 4}
        for item in sorted(self.findings, key=lambda value: (order.get(value.outcome, 99), value.fcpxml_path)):
            lines.append(f"[{item.outcome}] {item.construct} ({item.fcpxml_path}): {item.disposition}")
        return "\n".join(lines)
