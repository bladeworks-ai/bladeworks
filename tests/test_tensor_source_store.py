"""Opened-source, atomic replacement, and immutable-history contract tests."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

from bladeworks.preview.contracts import PreviewAPIError
from bladeworks.preview.source import OpenedSourceStore, source_version
from bladeworks.core.parser import enumerate_library_projects, read_fcpxml_root


def _xml(name: str, *, color_space: str = "1-1-1 (Rec. 709)") -> bytes:
    return f'''<?xml version="1.0"?><fcpxml version="1.14">
<resources>
  <format id="fmt" frameDuration="1/30s" width="160" height="90" colorSpace="{color_space}"/>
</resources>
<library><event name="Event"><project name="{name}">
<sequence format="fmt" duration="1s"><spine>
  <gap offset="0s" start="0s" duration="1s"/>
</spine></sequence></project></event></library></fcpxml>'''.encode()


def _multi_xml(*, second_format: str = "fmt", second_name: str = "Duplicate") -> bytes:
    return f'''<?xml version="1.0"?><fcpxml version="1.14">
<resources><format id="fmt" frameDuration="1/30s" width="160" height="90"/></resources>
<library name="Library"><event name="First Event">
  <project name="Duplicate"><sequence format="fmt" duration="1s"><spine>
    <gap offset="0s" duration="1s"/>
  </spine></sequence></project>
</event><event name="Second Event">
  <project name="{second_name}"><sequence format="{second_format}" duration="2s"><spine>
    <gap offset="0s" duration="2s"/>
  </spine></sequence></project>
</event></library></fcpxml>'''.encode()


def _bundle(tmp_path: Path, xml: bytes | None = None) -> Path:
    bundle = tmp_path / "Project.fcpxmld"
    bundle.mkdir()
    (bundle / "Info.fcpxml").write_bytes(xml or _xml("initial"))
    return bundle


def _store(
    tmp_path: Path,
    *,
    xml: bytes | None = None,
    strict: bool = False,
    history_limit: int = 50,
) -> OpenedSourceStore:
    return OpenedSourceStore.open(
        _bundle(tmp_path, xml),
        history_directory=tmp_path / "history",
        history_limit=history_limit,
        strict=strict,
    )


def test_open_reads_exact_bytes_hashes_and_creates_initial_snapshot(tmp_path: Path) -> None:
    xml = _xml("initial")
    store = _store(tmp_path, xml=xml)

    result = store.read_source()

    assert result.xml == xml
    assert result.status.disk_version == source_version(xml)
    assert result.status.loaded_version == source_version(xml)
    assert result.status.compile_status == "ready"
    assert (result.status.history_index, result.status.history_length) == (0, 1)
    assert list((tmp_path / "history").glob("*.fcpxml"))[0].read_bytes() == xml
    assert store.require_current(
        source_version(xml),
        "library[1]/event[1]/project[1]",
    ).document.source_path == (
        tmp_path / "Project.fcpxmld" / "Info.fcpxml"
    ).resolve()


def test_open_rejects_non_bundle_missing_info_and_invalid_startup(tmp_path: Path) -> None:
    plain = tmp_path / "plain.fcpxml"
    plain.write_bytes(_xml("plain"))
    with pytest.raises(PreviewAPIError) as not_bundle:
        OpenedSourceStore.open(plain, history_directory=tmp_path / "h1")
    assert not_bundle.value.code == "source_not_found"

    missing = tmp_path / "Missing.fcpxmld"
    missing.mkdir()
    with pytest.raises(PreviewAPIError) as no_info:
        OpenedSourceStore.open(missing, history_directory=tmp_path / "h2")
    assert no_info.value.code == "source_not_found"

    malformed = tmp_path / "Malformed.fcpxmld"
    malformed.mkdir()
    (malformed / "Info.fcpxml").write_bytes(b"<fcpxml>")
    with pytest.raises(PreviewAPIError) as invalid:
        OpenedSourceStore.open(malformed, history_directory=tmp_path / "h3")
    assert invalid.value.code == "source_invalid"

    unsupported = tmp_path / "Unsupported.fcpxmld"
    unsupported.mkdir()
    (unsupported / "Info.fcpxml").write_bytes(
        _xml("unsupported", color_space="6-1-6 (Rec. 2020)")
    )
    with pytest.raises(PreviewAPIError) as strict:
        OpenedSourceStore.open(
            unsupported,
            history_directory=tmp_path / "h4",
            strict=True,
        )
    assert strict.value.code == "unsupported_construct"


def test_conditional_replace_is_atomic_versioned_and_idempotent(tmp_path: Path) -> None:
    initial = _xml("initial")
    updated = _xml("updated")
    store = _store(tmp_path, xml=initial)

    result = store.replace(updated, expected_version=source_version(initial))

    assert store.source_path.read_bytes() == updated
    assert result.disk_version == source_version(updated)
    assert (result.history_index, result.history_length) == (1, 2)
    duplicate = store.replace(updated, expected_version=source_version(updated))
    assert (duplicate.history_index, duplicate.history_length) == (1, 2)
    assert not list(store.source_path.parent.glob(".Info.fcpxml.*.tmp"))


def test_history_write_failure_does_not_replace_the_live_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    initial = _xml("initial")
    updated = _xml("updated")
    store = _store(tmp_path, xml=initial)

    def fail_history(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected history failure")

    monkeypatch.setattr(store.history, "append", fail_history)
    with pytest.raises(OSError, match="injected history failure"):
        store.replace(updated, expected_version=source_version(initial))

    assert store.source_path.read_bytes() == initial
    assert store.read_source().status.disk_version == source_version(initial)


def test_live_replace_failure_rolls_back_the_prepared_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    initial = _xml("initial")
    updated = _xml("updated")
    store = _store(tmp_path, xml=initial)
    real_replace = os.replace

    def fail_live_replace(source: Path, destination: Path) -> None:
        if Path(destination) == store.source_path:
            raise OSError("injected live replace failure")
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_live_replace)
    with pytest.raises(OSError, match="injected live replace failure"):
        store.replace(updated, expected_version=source_version(initial))

    assert store.source_path.read_bytes() == initial
    assert store.history.position.index == 0
    assert store.history.position.length == 1
    assert store.history.selected_bytes() == initial


def test_stale_malformed_and_strict_rejected_puts_do_not_mutate(tmp_path: Path) -> None:
    initial = _xml("initial")
    store = _store(tmp_path, xml=initial, strict=True)

    with pytest.raises(PreviewAPIError) as stale:
        store.replace(_xml("stale"), expected_version="sha256:" + "0" * 64)
    assert stale.value.code == "source_version_conflict"

    with pytest.raises(PreviewAPIError) as malformed:
        store.replace(b"<fcpxml>", expected_version=source_version(initial))
    assert malformed.value.code == "source_invalid"

    unsupported = _xml("unsupported", color_space="6-1-6 (Rec. 2020)")
    with pytest.raises(PreviewAPIError) as strict:
        store.replace(unsupported, expected_version=source_version(initial))
    assert strict.value.code == "unsupported_construct"
    assert "sequence" in strict.value.message
    assert store.source_path.read_bytes() == initial
    assert store.status().history_length == 1


def test_undo_redo_branching_and_history_boundaries(tmp_path: Path) -> None:
    first = _xml("first")
    second = _xml("second")
    third = _xml("third")
    branch = _xml("branch")
    store = _store(tmp_path, xml=first)
    store.replace(second, expected_version=source_version(first))
    store.replace(third, expected_version=source_version(second))

    assert store.undo().disk_version == source_version(second)
    assert store.undo().disk_version == source_version(first)
    with pytest.raises(PreviewAPIError) as at_start:
        store.undo()
    assert at_start.value.code == "history_at_start"

    assert store.redo().disk_version == source_version(second)
    branched = store.replace(branch, expected_version=source_version(second))
    assert (branched.history_index, branched.history_length) == (2, 3)
    with pytest.raises(PreviewAPIError) as at_end:
        store.redo()
    assert at_end.value.code == "history_at_end"
    assert store.undo().disk_version == source_version(second)


def test_history_limit_removes_oldest_snapshots(tmp_path: Path) -> None:
    values = [_xml(f"version-{index}") for index in range(4)]
    store = _store(tmp_path, xml=values[0], history_limit=3)
    for previous, current in zip(values, values[1:]):
        store.replace(current, expected_version=source_version(previous))

    status = store.status()
    assert (status.history_index, status.history_length) == (2, 3)
    assert store.undo().disk_version == source_version(values[2])
    assert store.undo().disk_version == source_version(values[1])
    with pytest.raises(PreviewAPIError) as at_start:
        store.undo()
    assert at_start.value.code == "history_at_start"


def test_external_valid_edit_is_unlogged_until_managed_departure(tmp_path: Path) -> None:
    initial = _xml("initial")
    external = _xml("external")
    managed = _xml("managed")
    store = _store(tmp_path, xml=initial)
    store.source_path.write_bytes(external)

    adopted = store.reload()
    assert adopted.disk_version == source_version(external)
    assert adopted.loaded_version == source_version(external)
    assert (adopted.history_index, adopted.history_length) == (0, 1)

    saved = store.replace(managed, expected_version=source_version(external))
    assert (saved.history_index, saved.history_length) == (2, 3)
    assert store.undo().disk_version == source_version(external)
    assert store.undo().disk_version == source_version(initial)


def test_undo_discards_an_unlogged_external_edit_without_moving_cursor(tmp_path: Path) -> None:
    initial = _xml("initial")
    second = _xml("second")
    external = _xml("external")
    store = _store(tmp_path, xml=initial)
    store.replace(second, expected_version=source_version(initial))
    store.source_path.write_bytes(external)

    restored = store.undo()
    assert restored.disk_version == source_version(second)
    assert restored.history_index == 1
    previous = store.undo()
    assert previous.disk_version == source_version(initial)
    assert previous.history_index == 0


def test_external_invalid_bytes_remain_readable_but_never_claim_current_pixels(tmp_path: Path) -> None:
    initial = _xml("initial")
    repaired = _xml("repaired")
    store = _store(tmp_path, xml=initial)
    malformed = b"<fcpxml>"
    store.source_path.write_bytes(malformed)

    observed = store.read_source()
    assert observed.xml == malformed
    assert observed.status.disk_version == source_version(malformed)
    assert observed.status.loaded_version == source_version(initial)
    assert observed.status.compile_status == "source_invalid"
    with pytest.raises(PreviewAPIError) as invalid:
        store.require_current(
            source_version(malformed),
            "library[1]/event[1]/project[1]",
        )
    assert invalid.value.code == "source_invalid"

    recovered = store.replace(repaired, expected_version=source_version(malformed))
    assert recovered.compile_status == "ready"
    assert recovered.history_length == 2
    assert store.undo().disk_version == source_version(initial)


def test_same_base_concurrent_put_allows_exactly_one_writer(tmp_path: Path) -> None:
    initial = _xml("initial")
    store = _store(tmp_path, xml=initial)
    barrier = threading.Barrier(3)
    successes: list[str] = []
    failures: list[str] = []

    def write(xml: bytes) -> None:
        barrier.wait()
        try:
            successes.append(store.replace(xml, expected_version=source_version(initial)).disk_version)
        except PreviewAPIError as error:
            failures.append(error.code)

    threads = [threading.Thread(target=write, args=(_xml("a"),)), threading.Thread(target=write, args=(_xml("b"),))]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert len(successes) == 1
    assert failures == ["source_version_conflict"]


def test_non_strict_report_is_retained_and_copied(tmp_path: Path) -> None:
    xml = _xml("wide-gamut", color_space="6-1-6 (Rec. 2020)")
    store = _store(tmp_path, xml=xml, strict=False)
    loaded = store.require_current(
        source_version(xml),
        "library[1]/event[1]/project[1]",
    )

    first = store.report_for(loaded.document)
    second = store.report_for(loaded.document)
    assert first is not second
    assert first.degraded
    assert any(item.fcpxml_path == "sequence" for item in first.findings)


def test_multi_project_library_uses_structural_refs_even_with_duplicate_names(tmp_path: Path) -> None:
    xml = _multi_xml()
    store = _store(tmp_path, xml=xml)

    status = store.status()
    assert [item.project_ref for item in status.projects] == [
        "library[1]/event[1]/project[1]",
        "library[1]/event[2]/project[1]",
    ]
    assert [item.project_name for item in status.projects] == ["Duplicate", "Duplicate"]
    first = store.require_current(source_version(xml), status.projects[0].project_ref)
    second = store.require_current(source_version(xml), status.projects[1].project_ref)
    assert first.document.duration == 1
    assert second.document.duration == 2
    assert first.project_ref != second.project_ref


def test_project_ref_errors_are_stable_and_version_scoped(tmp_path: Path) -> None:
    xml = _multi_xml()
    store = _store(tmp_path, xml=xml)

    with pytest.raises(PreviewAPIError) as malformed:
        store.require_current(source_version(xml), "event/one/project/two")
    assert (malformed.value.code, malformed.value.status) == ("invalid_project_ref", 400)

    with pytest.raises(PreviewAPIError) as absent:
        store.require_current(source_version(xml), "library[1]/event[9]/project[1]")
    assert (absent.value.code, absent.value.status) == ("project_not_found", 404)

    with pytest.raises(PreviewAPIError) as stale:
        store.require_current("sha256:" + "0" * 64, "library[1]/event[9]/project[1]")
    assert (stale.value.code, stale.value.status) == ("source_version_conflict", 409)


def test_replacement_is_rejected_when_any_project_is_invalid(tmp_path: Path) -> None:
    initial = _multi_xml()
    store = _store(tmp_path, xml=initial)
    invalid = _multi_xml(second_format="missing")

    with pytest.raises(PreviewAPIError) as failure:
        store.replace(invalid, expected_version=source_version(initial))

    assert failure.value.code == "source_invalid"
    assert "library[1]/event[2]/project[1]" in failure.value.message
    assert store.source_path.read_bytes() == initial
    assert store.status().history_length == 1


def test_project_refs_use_tag_specific_indexes_across_multiple_libraries(tmp_path: Path) -> None:
    path = tmp_path / "Library.fcpxml"
    path.write_text(
        """<fcpxml version="1.14"><resources/>
<metadata/><library name="One"><metadata/><event><metadata/><project name="A"/></event></library>
<metadata/><library name="Two"><event><project name="B"/><metadata/><project name="C"/></event></library>
</fcpxml>""",
        encoding="utf-8",
    )

    projects = enumerate_library_projects(read_fcpxml_root(path))

    assert [project.project_ref for project in projects] == [
        "library[1]/event[1]/project[1]",
        "library[2]/event[1]/project[1]",
        "library[2]/event[1]/project[2]",
    ]
