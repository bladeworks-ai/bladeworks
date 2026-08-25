"""Generate downscaled proxy media for an FCPXML document and inject proxy reps.

Architecture map
----------------
This module implements the ``bladeworks proxy`` command. Given a plain
``.fcpxml`` file or a ``.fcpxmld`` bundle, it walks every VIDEO asset, transcodes
its ORIGINAL media into a small, fast-to-decode proxy, drops the proxy file
BESIDE the original, and injects a ``<media-rep kind="proxy-media" src=...>`` into
that asset so the renderer can later pick it up with ``--prefer proxy`` (see
``core/bindings.py:resolve_asset`` and ``compile_fcpxml``'s ``media_preference``).

The pass is DOCUMENT-WIDE over ``<resources>`` and never selects a Project, so
a full library export holding many Projects is fine without ``--project``
(``core/parser.py:parse_fcpxml_resources`` loads the resource tree alone).

Pipeline per asset (all-or-nothing per asset, never a partial file):

    1. pick the asset's ORIGINAL media-rep (``_choose_original_rep``: an
       ``original-media`` rep first, else the first resolvable NON-proxy rep;
       proxy reps are never an encode source, even under ``overwrite``) and
       resolve that rep's ``src`` to the file to transcode
    2. ffprobe it for raster + pixel format + codec + rotation
    3. decide codec by ALPHA:
         opaque -> H.264 / .mp4  (h264_videotoolbox on macOS when present,
                                  else libx264), low target bitrate
         alpha  -> ProRes 4444 / .mov (prores_ks -profile:v 4, yuva444p10le),
                   because H.264 cannot carry an alpha channel
    4. scale so the SHORTER side of the DISPLAYED picture becomes
       ``target_short_side`` (default 480). A 90/270 display rotation swaps the
       coded width/height first (``_display_dimensions``): ffmpeg autorotates
       on decode, so the proxy stores the upright picture with no rotation tag.
       Frame rate is kept (ffmpeg's default -- we pass no -r / fps filter).
    5. refuse an already-occupied destination unless ``overwrite=True``
       (``_assert_destination_free``), then encode to a temp file and atomically
       move it into place
    6. inject the proxy media-rep, mirroring the original rep's src STYLE
       (relative stays relative, absolute ``file://`` stays absolute) with the
       new filename segment percent-encoded (``encode_src_segment``)

Then the document is written back once, atomically, preserving the ``<?xml?>`` /
``<!DOCTYPE>`` prolog (ElementTree drops it) via
``core/media_consolidate.write_fcpxml_document``.

Idempotent: an asset that already carries a RESOLVABLE ``proxy-media`` rep is
skipped unless ``overwrite=True``. Stills (single-image codecs, or an asset
whose declared duration is not positive) and audio-only assets are skipped -- a
"proxy" only makes sense for real video. A missing CONTAINER duration is not
evidence of a still (fragmented / streamed MP4s report none) and does not skip.

Why a local ffprobe instead of ``tensor.decode.probe_video``
------------------------------------------------------------
``tensor/decode.py`` imports ``torch`` at module load, and this command is a pure
media-transcode utility that has no business dragging the whole tensor render
stack (or its multi-second import) into a CLI that just shells out to ffmpeg. The
proxy only needs raster / pixel-format / rotation facts, so ``_probe_source``
does one focused ffprobe call. Rotation extraction mirrors ``probe_video`` so the
two agree on what the container declares.

No silent failures: a probe or ffmpeg error raises ``ProxyGenerationError`` naming
the asset; unresolvable media becomes a WARNING on the returned result (never a
silently skipped asset). Nothing is overwritten unless ``overwrite=True`` -- an
occupied proxy destination is a loud ``ProxyGenerationError``.

Main callers: ``cli.main`` for the ``proxy`` subcommand.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import platform
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .bindings import file_url_path, lexical_src_path, relative_src_path
from .errors import FCPXMLCompileError
from .media_consolidate import _split_prolog, encode_src_segment, write_fcpxml_document
from .parser import parse_fcpxml_resources

# Container/codec defaults. The alpha branch is a deliberate exception to
# "low bitrate": preserving an alpha channel requires ProRes 4444, which is not
# a small format, but flattening alpha would corrupt the render.
DEFAULT_SHORT_SIDE = 480
DEFAULT_H264_BITRATE_KBPS = 1200

# ffprobe reports these as the video "codec" for single-image / still sources.
# A proxy of a still is pointless, so we skip them.
_STILL_CODECS = frozenset(
    {"png", "mjpeg", "bmp", "gif", "tiff", "webp", "jpeg2000", "apng", "jpegls"}
)


class ProxyGenerationError(FCPXMLCompileError):
    """A proxy could not be probed or encoded (loud, names the asset/media)."""


@dataclass
class _SourceProbe:
    """The handful of source facts proxy sizing + codec choice actually need."""

    width: int
    height: int
    pixel_format: str
    codec_name: str
    duration: Optional[float]
    rotation_degrees: int


@dataclass
class ProxyGenerationResult:
    """What one ``bladeworks proxy`` pass did, for a concise human report.

    - ``document`` / ``source_path``: the resolved bundle-or-file and the exact
      FCPXML file that was (or would be) rewritten.
    - ``generated``: ``(asset_id, proxy_path)`` for each proxy actually written.
    - ``skipped_existing`` / ``skipped_still`` / ``skipped_small``: asset ids that
      were intentionally left alone, by reason.
    - ``warnings``: assets whose original media could not be resolved/probed.
    - ``changed``: whether the FCPXML needs to be written back.
    """

    document: Path
    source_path: Path
    generated: list[tuple[str, Path]] = field(default_factory=list)
    skipped_existing: list[str] = field(default_factory=list)
    skipped_still: list[str] = field(default_factory=list)
    skipped_small: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    changed: bool = False

    def summary_lines(self) -> list[str]:
        """Deterministic multi-line report for stdout."""

        lines = [f"proxy: {self.document}"]
        for asset_id, path in self.generated:
            lines.append(f"  generated  {asset_id} -> {path}")
        for asset_id in self.skipped_existing:
            lines.append(f"  exists     {asset_id} already has a proxy media-rep")
        for asset_id in self.skipped_small:
            lines.append(f"  small      {asset_id} source shorter side already <= target")
        for asset_id in self.skipped_still:
            lines.append(f"  still      {asset_id} is a still image, not video")
        for warning in self.warnings:
            lines.append(f"  warning    {warning}")
        lines.append(
            f"  wrote {len(self.generated)} proxy file(s); "
            f"{'updated' if self.changed else 'no change to'} {self.source_path.name}"
        )
        return lines


def _local_tag(element: ET.Element) -> str:
    """Local element name, stripping any XML namespace (mirrors parser._tag)."""

    return element.tag.rsplit("}", 1)[-1]


def _require_tool(name: str) -> str:
    """Resolve an ffmpeg-family binary via PATH (house convention) or fail loud."""

    resolved = shutil.which(name)
    if resolved is None:
        raise ProxyGenerationError(
            f"{name} is required to generate proxies but was not found on PATH "
            f"(install ffmpeg, e.g. `brew install ffmpeg`)"
        )
    return resolved


def _probe_source(ffprobe: str, media_path: Path) -> Optional[_SourceProbe]:
    """One ffprobe call for the first video stream; ``None`` if there is none.

    Extracts width/height/pix_fmt/codec/duration and the container display
    rotation (side-data matrix or the legacy ``rotate`` tag), the same rotation
    the FFmpeg CLI autorotates by. Raises ``ProxyGenerationError`` if ffprobe
    itself fails -- a missing video stream is a legitimate ``None`` (audio-only
    asset), an ffprobe crash is not.
    """

    completed = subprocess.run(
        (
            ffprobe, "-v", "error", "-select_streams", "v:0",
            "-show_streams", "-show_format", "-of", "json", str(media_path),
        ),
        capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0:
        raise ProxyGenerationError(
            f"ffprobe could not read {media_path}: {completed.stderr.strip()}"
        )
    payload = json.loads(completed.stdout)
    streams = payload.get("streams") or []
    if not streams:
        return None
    stream = streams[0]

    rotation = 0
    for side in stream.get("side_data_list") or ():
        if "rotation" in side:
            rotation = int(round(float(side["rotation"]))) % 360
    tag_rotation = (stream.get("tags") or {}).get("rotate")
    if tag_rotation not in (None, "0"):
        rotation = int(round(float(tag_rotation))) % 360

    duration_raw = stream.get("duration") or (payload.get("format") or {}).get("duration")
    duration = (
        float(duration_raw) if duration_raw not in (None, "", "N/A") else None
    )

    return _SourceProbe(
        width=int(stream["width"]),
        height=int(stream["height"]),
        pixel_format=str(stream.get("pix_fmt") or ""),
        codec_name=str(stream.get("codec_name") or ""),
        duration=duration,
        rotation_degrees=rotation,
    )


def _has_alpha(pixel_format: str) -> bool:
    """True when a pixel format carries an alpha channel.

    Uses PyAV's ``VideoFormat`` component metadata (the same surface
    ``tensor/decode.py`` reads for bit depth), which is authoritative. Only if
    PyAV cannot parse the format name do we fall back to a conservative name
    heuristic -- and that fallback errs toward PRESERVING alpha (ProRes) rather
    than silently flattening it.
    """

    if not pixel_format:
        return False
    try:
        import av

        return any(getattr(c, "is_alpha", False) for c in av.VideoFormat(pixel_format).components)
    except Exception:
        name = pixel_format.lower()
        return name.startswith(("rgba", "bgra", "argb", "abgr", "yuva", "ya")) or name.endswith("a")


def _even(value: float) -> int:
    """Round to the nearest positive even integer (H.264/ProRes need even dims)."""

    return max(2, int(round(value / 2.0)) * 2)


def _display_dimensions(width: int, height: int, rotation_degrees: int) -> tuple[int, int]:
    """The picture size a viewer SEES after the container's display rotation.

    A 90/270 rotation swaps the coded width and height; 0/180 (or any
    non-quarter-turn angle, which ffmpeg autorotates without resizing) keeps
    the coded raster.

    Why this exists: the ffmpeg argv has no ``-noautorotate``, so ffmpeg
    rotates portrait phone footage upright BEFORE our ``scale=W:H`` filter.
    Sizing from the coded (landscape) raster would squash that upright picture
    into a landscape proxy; sizing from the displayed raster produces a proxy
    that stores the upright picture with no rotation tag -- the same image.
    """

    if rotation_degrees % 180 == 90:
        return height, width
    return width, height


def _scaled_dimensions(
    width: int,
    height: int,
    target_short_side: int,
    rotation_degrees: int = 0,
) -> Optional[tuple[int, int]]:
    """Target (w, h) so the shorter DISPLAYED side becomes ``target_short_side``.

    ``width``/``height`` are the coded raster; ``rotation_degrees`` is the
    container display rotation (see ``_display_dimensions``). Returns ``None``
    when the source is already at or below the target on its shorter side --
    we downscale only, never UPscale a proxy.
    """

    shown_w, shown_h = _display_dimensions(width, height, rotation_degrees)
    short = min(shown_w, shown_h)
    if short <= target_short_side:
        return None
    scale = target_short_side / short
    return _even(shown_w * scale), _even(shown_h * scale)


def _videotoolbox_h264_available(ffmpeg: str) -> bool:
    """True on macOS when ffmpeg lists the ``h264_videotoolbox`` encoder.

    Bladeworks does not ship ``legacy_ffmpeg.probe_host`` (it is internal-only),
    so this does a small, self-contained ``ffmpeg -encoders`` scan instead.
    """

    if platform.system() != "Darwin":
        return False
    completed = subprocess.run(
        (ffmpeg, "-hide_banner", "-encoders"),
        capture_output=True, text=True, check=False,
    )
    return completed.returncode == 0 and "h264_videotoolbox" in completed.stdout


def _build_encode_argv(
    ffmpeg: str,
    src: Path,
    dst: Path,
    *,
    dimensions: tuple[int, int],
    has_alpha: bool,
    bitrate_kbps: int,
    use_videotoolbox: bool,
) -> list[str]:
    """Assemble the ffmpeg argv for one proxy encode.

    Frame rate is intentionally NOT set: with no ``-r`` and no fps filter ffmpeg
    preserves the source rate, which is exactly "keep FPS". Audio is re-encoded
    to AAC so the proxy stays a faithful stand-in for rendering (a source with no
    audio simply yields no audio stream). ``dimensions`` are DISPLAY-oriented
    (see ``_scaled_dimensions``) because ffmpeg autorotates before ``scale``.
    """

    target_w, target_h = dimensions
    argv = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src),
        # ``setsar=1`` is load-bearing: ``scale`` preserves the source DISPLAY
        # aspect, so when ``_scaled_dimensions`` rounds (1280x720 -> 854x480,
        # not 853.33) ffmpeg stores a compensating sample aspect (1280:1281).
        # The tensor renderer rejects any non-square pixel aspect, which would
        # make ``render --prefer proxy`` fail for every 16:9 source. Square
        # pixels at the rounded raster are the intended proxy geometry.
        "-vf", f"scale={target_w}:{target_h},setsar=1",
    ]
    if has_alpha:
        # ProRes 4444 is the standard alpha-preserving proxy codec; VideoToolbox
        # has no low-bitrate alpha path, so we always encode alpha in software.
        argv += ["-c:v", "prores_ks", "-profile:v", "4", "-pix_fmt", "yuva444p10le"]
    elif use_videotoolbox:
        argv += ["-c:v", "h264_videotoolbox", "-b:v", f"{bitrate_kbps}k", "-allow_sw", "1", "-pix_fmt", "yuv420p"]
    else:
        argv += [
            "-c:v", "libx264", "-preset", "veryfast",
            "-b:v", f"{bitrate_kbps}k", "-maxrate", f"{bitrate_kbps}k",
            "-bufsize", f"{bitrate_kbps * 2}k", "-pix_fmt", "yuv420p",
        ]
    argv += ["-c:a", "aac", "-b:a", "160k", str(dst)]
    return argv


def _proxy_src(original_src: str, proxy_name: str) -> str:
    """Rewrite an original rep ``src`` to point at ``proxy_name`` in the SAME dir.

    Mirrors the original's style automatically: ``file:///a/b/c.mov`` ->
    ``file:///a/b/c.proxy.mp4``, ``Media/a.mp4`` -> ``Media/a.proxy.mp4``,
    ``a.mp4`` -> ``a.proxy.mp4``. Only the final path segment is swapped, so the
    scheme and directory (relative or absolute) are preserved VERBATIM; the new
    segment is percent-encoded (``clip#1.proxy.mp4`` -> ``clip%231.proxy.mp4``)
    because ``src`` is later urlparsed and a raw ``#``/``?``/``%`` would be
    read as a fragment/query and never resolve.
    """

    encoded = encode_src_segment(proxy_name)
    if "/" in original_src:
        prefix = original_src.rsplit("/", 1)[0]
        return f"{prefix}/{encoded}"
    return encoded


def _resolve_rep_path(raw_src: Optional[str], base_dir: Optional[Path]) -> Optional[Path]:
    """Resolve one media-rep ``src`` to an existing local file, or ``None``."""

    if not raw_src:
        return None
    candidate = file_url_path(raw_src)
    if candidate is None:
        candidate = relative_src_path(raw_src, base_dir)
    if candidate is not None and candidate.is_file():
        return candidate.resolve()
    return None


def _resolved_proxy_rep_paths(asset_element: ET.Element, base_dir: Optional[Path]) -> set[Path]:
    """Existing files this asset's ``proxy-media`` reps already point at."""

    paths: set[Path] = set()
    for child in asset_element:
        if _local_tag(child) != "media-rep" or child.get("kind") != "proxy-media":
            continue
        resolved = _resolve_rep_path(child.get("src"), base_dir)
        if resolved is not None:
            paths.add(resolved)
    return paths


@dataclass(frozen=True)
class _OriginalRep:
    """The media-rep a proxy is based on, with the two paths the pass needs.

    - ``element``: the ``<media-rep>`` whose ``src`` the proxy locator mirrors.
    - ``source_path``: the resolved real file to probe and encode from.
    - ``lexical_path``: the path the src NAMES (``lexical_src_path``); the
      proxy is written beside it, named after ITS stem. For a plain file the
      two paths coincide; for a consolidation symlink
      (``Media/clip.mov -> /ext/clip.mov``) they differ, and the proxy must land
      in ``Media/`` so that the injected ``Media/clip.proxy.mp4`` locator
      resolves -- writing it beside the link TARGET would leave the locator
      dangling and ``--prefer proxy`` silently falling back to the original.
    """

    element: ET.Element
    source_path: Path
    lexical_path: Path


def _choose_original_rep(
    asset_element: ET.Element, base_dir: Optional[Path]
) -> Optional[_OriginalRep]:
    """Pick the media-rep to base the proxy on, with its resolved file.

    Order: an ``original-media`` rep first, else the first resolvable rep in
    document order. ``proxy-media`` reps are NEVER candidates -- this is the
    one place the proxy pass deliberately diverges from
    ``resolve_asset(prefer="original")``, which ranks proxy and untagged reps
    equally; here an old proxy must never become the encode SOURCE of the new
    one (it would re-encode an already-degraded picture under ``overwrite``).

    Returns an ``_OriginalRep`` or ``None`` when nothing resolves.
    """

    resolvable: list[tuple[int, int, _OriginalRep]] = []
    for index, child in enumerate(asset_element):
        if _local_tag(child) != "media-rep":
            continue
        if child.get("kind") == "proxy-media":
            continue
        src = child.get("src")
        resolved = _resolve_rep_path(src, base_dir)
        if resolved is None:
            continue
        lexical = lexical_src_path(src, base_dir)
        if lexical is None:
            raise AssertionError(f"media-rep src {src!r} resolved but has no lexical path")
        rank = 0 if child.get("kind") == "original-media" else 1
        resolvable.append(
            (rank, index, _OriginalRep(element=child, source_path=resolved, lexical_path=lexical))
        )
    if not resolvable:
        return None
    resolvable.sort(key=lambda item: (item[0], item[1]))
    return resolvable[0][2]


def _assert_destination_free(
    asset_id: str,
    proxy_path: Path,
    own_proxy_paths: set[Path],
    overwrite: bool,
) -> None:
    """Refuse to clobber a file at ``proxy_path`` unless it may be replaced.

    Replacing is allowed when ``overwrite`` is set, or when the file is already
    this asset's own proxy (one of its ``proxy-media`` reps resolves to it).
    Anything else at that path -- a stale proxy whose rep was removed, another
    asset's proxy sharing the stem, a user's file -- is a loud error, never a
    silent overwrite and never a silent skip.
    """

    if overwrite or not proxy_path.exists():
        return
    if proxy_path.resolve() in own_proxy_paths:
        return
    raise ProxyGenerationError(
        f"{asset_id}: proxy destination {proxy_path} already exists and is not "
        f"this asset's proxy-media rep; refusing to overwrite it "
        f"(pass --overwrite to replace it)"
    )


def _insert_proxy_rep(asset_element: ET.Element, proxy_src: str) -> None:
    """Insert ``<media-rep kind="proxy-media" src=...>`` after the last media-rep."""

    rep = ET.Element("media-rep")
    rep.set("kind", "proxy-media")
    rep.set("src", proxy_src)

    last_rep_index = -1
    for index, child in enumerate(asset_element):
        if _local_tag(child) == "media-rep":
            last_rep_index = index
    asset_element.insert(last_rep_index + 1, rep)


def generate_proxies(
    input_path: Path,
    *,
    target_short_side: int = DEFAULT_SHORT_SIDE,
    bitrate_kbps: int = DEFAULT_H264_BITRATE_KBPS,
    overwrite: bool = False,
) -> ProxyGenerationResult:
    """Generate proxy media for every video asset and inject proxy reps.

    Procedurally:
      1. Load the document's ``<resources>`` (bundle or plain file) WITHOUT
         selecting a Project -- validates the XML, gives the typed assets, the
         media base dir, the exact source file to rewrite, and a MUTABLE tree.
      2. Index the tree's ``<asset>`` elements by id.
      3. For each VIDEO asset: skip audio-only, assets with a resolvable proxy
         (unless ``overwrite``), stills, and sources already at/below the
         target. Otherwise pick the original rep -> probe -> check the
         destination is free -> encode -> move into place -> inject the rep.
      4. Write the FCPXML back once, atomically, only if anything changed.

    Main callers: ``cli.main`` for the ``proxy`` subcommand.
    """

    ffmpeg = _require_tool("ffmpeg")
    ffprobe = _require_tool("ffprobe")

    source = parse_fcpxml_resources(input_path)
    base_dir = source.media_base_dir
    result = ProxyGenerationResult(
        document=Path(input_path).expanduser().resolve(),
        source_path=source.source_path,
    )

    prolog, _ = _split_prolog(source.data)
    root = source.root
    if root.tag.startswith("{"):
        uri = root.tag[1:].split("}", 1)[0]
        ET.register_namespace("", uri)

    asset_elements = {
        element.get("id"): element
        for element in root.iter()
        if _local_tag(element) == "asset" and element.get("id")
    }

    use_videotoolbox = _videotoolbox_h264_available(ffmpeg)

    for asset in source.assets.values():
        if not asset.has_video:
            continue
        asset_element = asset_elements.get(asset.id)
        if asset_element is None:
            result.warnings.append(f"{asset.id}: <asset> element not found in document")
            continue
        own_proxy_paths = _resolved_proxy_rep_paths(asset_element, base_dir)
        if not overwrite and own_proxy_paths:
            result.skipped_existing.append(asset.id)
            continue

        chosen = _choose_original_rep(asset_element, base_dir)
        if chosen is None:
            result.warnings.append(f"{asset.id}: no resolvable original media-rep to base a proxy on")
            continue
        original_rep, original_path = chosen.element, chosen.source_path

        probe = _probe_source(ffprobe, original_path)
        if probe is None:
            result.warnings.append(f"{asset.id}: {original_path} has no video stream")
            continue
        # A still is a single-image codec or a declared non-positive duration.
        # A missing CONTAINER duration is NOT a still (fragmented/streamed
        # sources legitimately report none) and must not skip real video.
        if probe.codec_name in _STILL_CODECS or (asset.duration is not None and asset.duration <= 0):
            result.skipped_still.append(asset.id)
            continue

        dimensions = _scaled_dimensions(
            probe.width, probe.height, target_short_side, probe.rotation_degrees
        )
        if dimensions is None:
            result.skipped_small.append(asset.id)
            continue

        has_alpha = _has_alpha(probe.pixel_format)
        extension = ".mov" if has_alpha else ".mp4"
        # Placed beside the location the src NAMES (see ``_OriginalRep``), named
        # after that location's stem so the locator mirrors the original's.
        lexical = chosen.lexical_path
        proxy_path = lexical.parent / f"{lexical.stem}.proxy{extension}"
        _assert_destination_free(asset.id, proxy_path, own_proxy_paths, overwrite)

        _encode_proxy(
            ffmpeg,
            original_path,
            proxy_path,
            dimensions=dimensions,
            has_alpha=has_alpha,
            bitrate_kbps=bitrate_kbps,
            use_videotoolbox=use_videotoolbox,
            asset_id=asset.id,
        )

        proxy_src = _proxy_src(str(original_rep.get("src") or ""), proxy_path.name)
        if overwrite:
            _remove_existing_proxy_reps(asset_element)
        _insert_proxy_rep(asset_element, proxy_src)
        result.generated.append((asset.id, proxy_path))
        result.changed = True

    if result.changed:
        # Compare-and-replace against the bytes we parsed: an edit that landed
        # while ffmpeg was encoding must not be clobbered by the stale tree.
        write_fcpxml_document(source.source_path, prolog, root, expected_bytes=source.data)

    return result


def _remove_existing_proxy_reps(asset_element: ET.Element) -> None:
    """Drop any existing proxy-media reps (used only on ``overwrite``)."""

    for child in list(asset_element):
        if _local_tag(child) == "media-rep" and child.get("kind") == "proxy-media":
            asset_element.remove(child)


def _encode_proxy(
    ffmpeg: str,
    src: Path,
    dst: Path,
    *,
    dimensions: tuple[int, int],
    has_alpha: bool,
    bitrate_kbps: int,
    use_videotoolbox: bool,
    asset_id: str,
) -> None:
    """Encode one proxy to a temp file next to ``dst``, then atomically move it in.

    Encoding to a sibling temp file and ``os.replace``-ing keeps the operation
    all-or-nothing: a failed/interrupted ffmpeg never leaves a half-written proxy
    that a later resolve would treat as real media. The caller has already
    decided (``_assert_destination_free``) that replacing ``dst`` is allowed.
    """

    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".proxy-", suffix=dst.suffix, dir=str(dst.parent))
    os.close(fd)
    temp_path = Path(temp_name)
    argv = _build_encode_argv(
        ffmpeg, src, temp_path,
        dimensions=dimensions, has_alpha=has_alpha,
        bitrate_kbps=bitrate_kbps, use_videotoolbox=use_videotoolbox,
    )
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            tail = (completed.stderr or completed.stdout or "ffmpeg failed without output")[-4000:]
            raise ProxyGenerationError(
                f"{asset_id}: ffmpeg proxy encode failed (exit {completed.returncode}):\n{tail}"
            )
        os.replace(temp_path, dst)
    finally:
        if temp_path.exists():
            temp_path.unlink()
