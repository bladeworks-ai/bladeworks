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
    candidate = (base_dir / rel).resolve()
    if candidate.is_file():
        return candidate
    return None


def resolve_asset(
    asset: AssetResource,
    bindings: Bindings,
    base_dir: Optional[Path] = None,
) -> tuple[Optional[Path], Optional[AssetBinding]]:
    """Resolve one resource by explicit identity, then a concrete local file.

    Precedence, highest first:

    1. An explicit ``--bindings`` entry that matches the resource identity.
    2. An ABSOLUTE ``file://`` media ``src`` that points at an existing file
       (the pre-existing plain-``.fcpxml`` behavior).
    3. A BUNDLE-RELATIVE media ``src`` resolved against ``base_dir`` (the
       ``.fcpxmld`` bundle root or the plain file's parent directory).

    Both fields on a binding are constraints. A binding that provides a UID and
    resource ID must match both, preventing a stale resource ID from silently
    redirecting a different stable media identity.

    ``base_dir`` is threaded in from ``SourceDocument.media_base_dir``. When no
    binding matches and neither an absolute nor a relative src resolves to an
    existing file, this returns ``(None, None)`` exactly as before -- NO
    guessing, NO silent fallback -- and the compiler reports the missing media
    loudly.
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

    for representation in asset.media_representations:
        candidate = file_url_path(representation.src)
        if candidate is None:
            candidate = relative_src_path(representation.src, base_dir)
        if candidate is not None and candidate.is_file():
            return candidate.resolve(), AssetBinding(resource_id=asset.id, uid=asset.uid, path=candidate.resolve())
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
