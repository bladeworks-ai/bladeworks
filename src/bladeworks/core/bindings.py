"""Load explicit asset and font bindings without guessing filenames."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlparse

from .errors import FCPXMLCompileError
from .model import AssetBinding, AssetResource, Bindings, FontBinding


def load_bindings(path: Optional[Path]) -> Bindings:
    if path is None:
        return Bindings()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FCPXMLCompileError(f"could not read bindings JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise FCPXMLCompileError("bindings JSON root must be an object")

    assets: list[AssetBinding] = []
    for index, raw in enumerate(payload.get("assets", [])):
        if not isinstance(raw, dict) or not raw.get("path"):
            raise FCPXMLCompileError(f"bindings.assets[{index}] must contain path")
        resource_id = raw.get("resource_id")
        uid = raw.get("uid")
        if not resource_id and not uid:
            raise FCPXMLCompileError(f"bindings.assets[{index}] must contain resource_id or uid")
        assets.append(
            AssetBinding(
                resource_id=str(resource_id) if resource_id else None,
                uid=str(uid) if uid else None,
                path=Path(str(raw["path"])).expanduser().resolve(),
            )
        )

    fonts: list[FontBinding] = []
    for index, raw in enumerate(payload.get("fonts", [])):
        if not isinstance(raw, dict) or not raw.get("name") or not raw.get("path"):
            raise FCPXMLCompileError(f"bindings.fonts[{index}] must contain name and path")
        fonts.append(
            FontBinding(
                name=str(raw["name"]),
                path=Path(str(raw["path"])).expanduser().resolve(),
                index=int(raw.get("index", 0)),
            )
        )
    return Bindings(assets=tuple(assets), fonts=tuple(fonts))


def file_url_path(raw: Optional[str]) -> Optional[Path]:
    """Return the absolute local path named by an absolute ``file://`` URI.

    Only an ABSOLUTE ``file://`` URI resolves here (``scheme == "file"`` and an
    absolute path). A scheme-less or relative ``src`` is handled by
    ``relative_src_path`` against the document/bundle base dir instead.
    """

    if not raw:
        return None
    parsed = urlparse(raw)
    if parsed.scheme != "file":
        return None
    local = Path(unquote(parsed.path)).expanduser()
    # A relative ``file:`` src (e.g. ``file:Media/a.mp4``) is NOT an absolute
    # media locator; defer it to the base-dir resolver.
    if not local.is_absolute():
        return None
    return local


def relative_src_path(raw: Optional[str], base_dir: Optional[Path]) -> Optional[Path]:
    """Resolve a BUNDLE-RELATIVE media ``src`` against ``base_dir``.

    A media-rep ``src`` is bundle-relative when it names a path with no scheme
    (``Media/a.mp4``, ``./Media/a.mp4``) or a relative ``file:`` path. It is
    resolved as ``(base_dir / rel).resolve()`` and returned only when it points
    at an existing file. There is NO guessing and NO filename search: the src
    explicitly names the path relative to the bundle root, so a miss returns
    ``None`` and the caller fails loudly downstream.

    Absolute paths and absolute ``file://`` URIs are handled by
    ``file_url_path`` and skipped here.
    """

    candidate = _relative_src_candidate(raw, base_dir)
    if candidate is None:
        return None
    resolved = candidate.resolve()
    if resolved.is_file():
        return resolved
    return None


def _relative_src_candidate(raw: Optional[str], base_dir: Optional[Path]) -> Optional[Path]:
    """The path a bundle-relative ``src`` NAMES under ``base_dir`` -- unresolved, unchecked."""

    if not raw or base_dir is None:
        return None
    parsed = urlparse(raw)
    # Reject anything that is not a plain path or a relative ``file:`` path;
    # a real remote scheme (http, https, ...) is never a local bundle file.
    if parsed.scheme not in ("", "file"):
        return None
    rel = Path(unquote(parsed.path))
    if rel.is_absolute():
        return None
    return base_dir / rel


def lexical_src_path(raw: Optional[str], base_dir: Optional[Path]) -> Optional[Path]:
    """The absolute path a media-rep ``src`` names, WITHOUT following symlinks.

    ``file_url_path`` / ``relative_src_path`` answer "which real file is this?"
    (the latter symlink-resolves). This answers "where does the document say
    the file lives?" -- for a consolidated ``Media/clip.mov`` link that is the
    bundle's ``Media/`` directory, not the external directory the link points
    at. Existence is not checked here.

    Main callers: ``core/proxy_media._choose_original_rep`` -- a generated proxy
    must be PLACED beside the location the src names (so the injected
    ``Media/clip.proxy.mp4`` locator actually exists), even though it is
    ENCODED from the resolved target.
    """

    local = file_url_path(raw)
    if local is not None:
        return local
    return _relative_src_candidate(raw, base_dir)


def _representation_rank(kind: Optional[str], prefer: str) -> int:
    """Preference rank for one ``media-rep`` (lower = more preferred).

    Ties are broken by document order at the call site, so this only decides
    which ``kind`` families jump the queue.

    ``prefer == "original"`` (the default): a rep explicitly tagged
    ``original-media`` outranks everything else, so a *disambiguated* original
    always wins. When NO rep carries the ``original-media`` tag every rep ranks
    the same and document order alone decides -- i.e. the historical "first
    representation that resolves" behavior is preserved unchanged.

    ``prefer == "proxy"``: proxy first, then original, then anything else --
    proxy with a graceful fallback to the original, then to whatever resolves.
    """

    if prefer == "proxy":
        return {"proxy-media": 0, "original-media": 1}.get(kind, 2)
    return {"original-media": 0}.get(kind, 1)


def resolve_asset(
    asset: AssetResource,
    bindings: Bindings,
    base_dir: Optional[Path] = None,
    *,
    prefer: str = "original",
) -> tuple[Optional[Path], Optional[AssetBinding]]:
    """Resolve one resource by explicit identity, then a concrete local file.

    Precedence, highest first:

    1. An explicit ``--bindings`` entry that matches the resource identity.
    2. Among the ``media-rep`` children whose ``src`` resolves to an existing
       file, the one preferred by ``prefer`` (see ``_representation_rank``),
       ties broken by document order. An ABSOLUTE ``file://`` src and a
       BUNDLE-RELATIVE src (resolved against ``base_dir``) are both eligible.

    ``prefer`` selects between an ``original-media`` and a ``proxy-media``
    representation when an asset carries both. It only ever chooses among reps
    that ALREADY resolve to a real file, so it never trades a present file for a
    missing one -- the fallback is always "whatever we can actually find."

    Both fields on a binding are constraints. A binding that provides a UID and
    resource ID must match both, preventing a stale resource ID from silently
    redirecting a different stable media identity.

    ``base_dir`` is threaded in from ``SourceDocument.media_base_dir``. When no
    binding matches and no representation resolves to an existing file, this
    returns ``(None, None)`` exactly as before -- NO guessing, NO silent
    fallback -- and the compiler reports the missing media loudly.
    """

    matches: list[AssetBinding] = []
    for binding in bindings.assets:
        if binding.resource_id is not None and binding.resource_id != asset.id:
            continue
        if binding.uid is not None and binding.uid != asset.uid:
            continue
        matches.append(binding)
    if len(matches) > 1:
        raise FCPXMLCompileError(f"multiple explicit bindings match asset {asset.id}")
    if matches:
        return matches[0].path, matches[0]

    # Collect every representation that resolves to a real file, tagged with its
    # preference rank and document index, then pick the best (rank, index).
    resolved: list[tuple[int, int, Path]] = []
    for index, representation in enumerate(asset.media_representations):
        candidate = file_url_path(representation.src)
        if candidate is None:
            candidate = relative_src_path(representation.src, base_dir)
        if candidate is None or not candidate.is_file():
            continue
        resolved.append((_representation_rank(representation.kind, prefer), index, candidate.resolve()))

    if resolved:
        resolved.sort(key=lambda item: (item[0], item[1]))
        chosen = resolved[0][2]
        return chosen, AssetBinding(resource_id=asset.id, uid=asset.uid, path=chosen)
    return None, None


def unresolved_asset_locators(
    asset: AssetResource,
    bindings: Bindings,
    base_dir: Optional[Path] = None,
) -> tuple[str, ...]:
    """Return the exact locations attempted for an unresolved asset.

    Main callers:
    - ``compiler._compile_clip`` when ``resolve_asset`` cannot produce a file.

    Why this exists:
    Missing media must remain visible and reportable. Resolution intentionally
    refuses filename guessing, but the authored location is still useful for
    the placeholder label, compatibility report, and editor relink UI.
    Explicit bindings retain precedence over the FCPXML representations.
    """

    matching_bindings = tuple(
        binding
        for binding in bindings.assets
        if (binding.resource_id is None or binding.resource_id == asset.id)
        and (binding.uid is None or binding.uid == asset.uid)
    )
    if matching_bindings:
        return tuple(dict.fromkeys(str(binding.path) for binding in matching_bindings))

    locators: list[str] = []
    for representation in asset.media_representations:
        raw = representation.src
        if not raw:
            continue
        absolute = file_url_path(raw)
        if absolute is not None:
            locators.append(str(absolute.resolve(strict=False)))
            continue
        parsed = urlparse(raw)
        if parsed.scheme in ("", "file") and base_dir is not None:
            relative = Path(unquote(parsed.path))
            if not relative.is_absolute():
                locators.append(str((base_dir / relative).resolve(strict=False)))
                continue
        locators.append(raw)
    if not locators:
        return (f"asset:{asset.id}",)
    return tuple(dict.fromkeys(locators))
