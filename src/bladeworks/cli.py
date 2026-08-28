"""Command-line interface for inspecting and rendering portable FCPXML.

Architecture map
----------------
This module is the ``bladeworks`` Python package front door. The public CLI is
installed as the ``fcpxml`` console script, while the Python module entry point
remains available for development inside Bladeworks via ``__main__.py``. It
hosts a small family of subcommands:

- ``render``   -- compile an ``.fcpxml`` / ``.fcpxmld`` and execute a render.
                  Defaults to the ``tensor`` backend; ``-o/--output`` is
                  optional (``<stem>.mp4``, or ``.mov`` under ``--alpha``).
- ``inspect``  -- parse + classify a project without rendering.
- ``projects`` -- list the projects a library/bundle holds (browse the names
                  and UIDs you can pass to ``--project``), without compiling.
- ``examples`` -- ``ls`` the packaged sample projects or ``cp`` one out.
- ``doctor``   -- check runtime prerequisites (``ffprobe`` on PATH, the torch
                  device, and component versions).
- ``server``   -- run one opened bundle as a foreground local API, or check a
                  running server's health.
- ``studio``   -- run the same API plus the packaged browser editor.

Why the ``--oracle-mezzanine`` flag is hidden rather than removed
-----------------------------------------------------------------
``--oracle-mezzanine`` (a 10-bit ProRes comparison master) is a dev / evidence
path, not a user-facing delivery, so it is suppressed from ``--help``. It is
NOT deleted: the CPU evidence harness and the tensor-backend rejection test
still drive it by name, so the option keeps working -- it just no longer
advertises itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fcpxml",
        description="Bladeworks FCPXML renderer: portable FCPXML to video",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="parse and classify an FCPXML document without rendering")
    inspect_parser.add_argument("input", type=Path)
    inspect_parser.add_argument("--project", help="exact project name or UID to select from a full library export")
    inspect_parser.add_argument("--bindings", type=Path)
    inspect_parser.add_argument("--report", type=Path)
    inspect_parser.add_argument("--emit-plan", type=Path, help="write the resolved render document as JSON")
    inspect_parser.add_argument("--strict", action="store_true", help="fail when any construct is approximated or omitted")

    projects_parser = subparsers.add_parser(
        "projects",
        help="list the projects a .fcpxml file / .fcpxmld bundle contains (for --project selection)",
    )
    projects_parser.add_argument("input", type=Path, help="a plain .fcpxml file OR a .fcpxmld bundle directory")
    projects_parser.add_argument(
        "--json",
        action="store_true",
        help="emit a machine-readable [{event, name, uid}, ...] array instead of the grouped listing",
    )

    render_parser = subparsers.add_parser("render", help="compile and execute a render (default backend: tensor)")
    render_parser.add_argument("input", type=Path, help="a plain .fcpxml file OR a .fcpxmld bundle directory")
    render_parser.add_argument("--project", help="exact project name or UID to select from a full library export")
    render_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="output path; default <input-stem>.mp4 (or <input-stem>.mov with --alpha)",
    )
    render_parser.add_argument("--bindings", type=Path)
    render_parser.add_argument("--report", type=Path)
    render_parser.add_argument("--manifest", type=Path)
    render_parser.add_argument("--emit-plan", type=Path)
    render_parser.add_argument(
        "--backend",
        choices=("tensor",),
        default="tensor",
        help=(
            "pixel backend (default: tensor): tensor is the PyTorch renderer "
            "(video rendered on tensors, audio still the calibrated FFmpeg "
            "graph); cpu is the correctness reference; vulkan is strict; auto "
            "falls back for the complete render with a reason and never picks "
            "tensor"
        ),
    )
    render_parser.add_argument(
        "--device",
        choices=("auto", "mps", "cuda", "cpu"),
        default="auto",
        help=(
            "torch device for --backend tensor (default: auto = mps then cuda then cpu). "
            "Fleet/gym must pass mps; an explicit mps/cuda pin fails loudly if missing"
        ),
    )
    render_parser.add_argument(
        "--video-only",
        action="store_true",
        help="render video with a silent output track and explicitly omit source audio",
    )
    # Hidden dev/evidence path: a 10-bit 4:2:2 ProRes comparison master. Kept
    # functional (the CPU evidence harness and the tensor rejection test drive
    # it by name) but removed from --help via argparse.SUPPRESS.
    render_parser.add_argument(
        "--oracle-mezzanine",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    render_parser.add_argument(
        "--alpha",
        action="store_true",
        help=(
            "tensor backend only: write a ProRes 4444 straight-alpha .mov "
            "(transparent root canvas) instead of delivery H.264"
        ),
    )
    render_parser.add_argument("--strict", action="store_true", help="fail before FFmpeg when any construct is approximated or omitted")
    render_parser.add_argument(
        "--prefer",
        choices=("original", "proxy"),
        default="original",
        help=(
            "which media representation to decode when an asset carries both an "
            "original and a proxy: original (default) prefers original-media and "
            "falls back to the first that resolves; proxy prefers proxy-media, "
            "then original, then whatever resolves"
        ),
    )
    _add_symlink_media_option(render_parser)
    render_parser.add_argument(
        "--encoder-preset",
        default=None,
        help="x264 preset override for delivery encodes (e.g. veryfast); default medium",
    )
    render_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable the tensor backend's completed-frame progress bar",
    )

    examples_parser = subparsers.add_parser("examples", help="list or copy out the packaged sample projects")
    examples_sub = examples_parser.add_subparsers(dest="examples_command", required=True)
    examples_sub.add_parser("ls", help="list the packaged sample projects (name + one-line description)")
    examples_cp = examples_sub.add_parser("cp", help="copy a packaged sample .fcpxmld bundle to DEST")
    examples_cp.add_argument("name", help="sample project name (see 'examples ls')")
    examples_cp.add_argument(
        "dest",
        nargs="?",
        type=Path,
        default=Path("."),
        help="destination directory (default: current directory)",
    )

    subparsers.add_parser("doctor", help="check runtime prerequisites (ffmpeg, ffprobe, torch device, versions)")

    proxy_parser = subparsers.add_parser(
        "proxy",
        help="generate downscaled proxy media for each video asset and inject proxy media-reps",
    )
    proxy_parser.add_argument("input", type=Path, help="a .fcpxml file or a .fcpxmld bundle directory")
    proxy_parser.add_argument(
        "--height",
        type=_positive_int,
        default=480,
        help="target for the SHORTER side of the proxy in pixels (default: 480); sources already this small are skipped",
    )
    proxy_parser.add_argument(
        "--bitrate",
        type=_positive_int,
        default=1200,
        help="target H.264 proxy bitrate in kbps for opaque sources (default: 1200); alpha sources use ProRes 4444",
    )
    proxy_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="regenerate proxies even for assets that already carry a proxy media-rep",
    )

    server_parser = subparsers.add_parser("server", help="serve one .fcpxmld bundle on localhost")
    server_sub = server_parser.add_subparsers(dest="server_command", required=True)
    server_run = server_sub.add_parser("run", help="run one foreground Bladeworks server")
    server_run.add_argument("input", type=Path, help="an existing .fcpxmld bundle directory")
    _add_local_runtime_options(server_run, default_port=8765)
    server_health = server_sub.add_parser("health", help="check liveness and readiness")
    server_health.add_argument("--url", required=True, help="server base URL from its ready record")
    server_health.add_argument("--timeout", type=float, default=2.0)

    studio_parser = subparsers.add_parser(
        "studio",
        help="open one .fcpxmld bundle in the packaged local web editor",
    )
    studio_parser.add_argument("input", type=Path, help="an existing .fcpxmld bundle directory")
    _add_local_runtime_options(studio_parser, default_port=0)
    studio_parser.add_argument(
        "--no-open",
        action="store_true",
        help="serve Studio without launching a browser",
    )

    return parser


def _positive_int(raw: str) -> int:
    """argparse ``type`` for the ``proxy`` sizing flags: an integer >= 1.

    Why this exists: ``--height 0`` would make every source "already small"
    and silently generate nothing, and a non-positive ``--bitrate`` is an
    ffmpeg error deep in the encode. Rejecting both at parse time names the
    bad value up front.
    """

    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {raw!r}") from exc
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer (>= 1), got {value}")
    return value


def _add_local_runtime_options(command: argparse.ArgumentParser, *, default_port: int) -> None:
    """Add the identical API runtime controls to server and Studio commands."""

    command.add_argument("--host", default="127.0.0.1", choices=("127.0.0.1",))
    command.add_argument(
        "--port",
        type=int,
        default=default_port,
        help="listen port; 0 selects an available port",
    )
    command.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    command.add_argument("--decoder-threads", type=int, default=2)
    command.add_argument("--history-limit", type=int, default=50)
    command.add_argument("--render-dir", type=Path, default=None)
    command.add_argument(
        "--log-level",
        choices=("debug", "info", "warning", "error"),
        default="info",
    )
    command.add_argument("--strict", action="store_true")
    command.add_argument(
        "--allow-origin",
        action="append",
        default=[],
        help="explicit loopback browser origin; repeat for multiple origins",
    )
    _add_symlink_media_option(command)


def _add_symlink_media_option(command: argparse.ArgumentParser) -> None:
    """Add the identical ``--symlink-media`` opt-in to every open-a-bundle command.

    When set, opening the ``.fcpxmld`` bundle first symlinks every EXTERNAL
    media file referenced by ``Info.fcpxml`` into the bundle's ``Media/`` folder
    and repoints each ``src`` at ``Media/<name>`` -- making the bundle
    self-contained by reference before the command runs. Shared here (render,
    server run, studio) so the flag reads and behaves identically everywhere.
    """

    command.add_argument(
        "--symlink-media",
        action="store_true",
        help=(
            "before opening, symlink every external media file referenced in "
            "Info.fcpxml into the bundle's Media/ folder and rewrite each src to "
            "Media/<name> (requires a .fcpxmld bundle)"
        ),
    )


def _output_profile(parser: argparse.ArgumentParser, args: argparse.Namespace) -> str:
    """Resolve the render command's output profile from its flags, loudly on conflicts.

    ``--alpha`` is the tensor backend's ProRes 4444 straight-alpha exit
    (``executor._TENSOR_OUTPUT_PROFILES``); no other backend has an
    alpha-carrying delivery, so it is refused here by name rather than
    surfacing as an unknown-profile error deep in the CPU graph builder.
    """

    if args.alpha and args.oracle_mezzanine:
        parser.error("--alpha and --oracle-mezzanine are mutually exclusive output profiles")
    if args.alpha and args.backend != "tensor":
        parser.error("--alpha (ProRes 4444 straight-alpha delivery) is implemented by --backend tensor only")
    if args.alpha:
        return "delivery_alpha"
    return "oracle_mezzanine" if args.oracle_mezzanine else "delivery"


def _resolve_render_output(args: argparse.Namespace) -> Path:
    """Return the render command's output path, defaulting from the input stem.

    Why this exists: ``-o/--output`` is optional. When omitted we deliver next
    to the input using the container the chosen profile implies -- ``.mov`` for
    the tensor ``--alpha`` (ProRes 4444) exit, ``.mp4`` otherwise. A
    ``.fcpxmld`` bundle directory's stem is the bundle name (``Info.fcpxml``
    lives INSIDE it), so ``single_clip.fcpxmld`` -> ``single_clip.mp4``.
    """

    if args.output is not None:
        return args.output
    suffix = ".mov" if args.alpha else ".mp4"
    return args.input.with_suffix(suffix)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # ``--symlink-media`` consolidates the bundle's referenced media into its
    # Media/ folder BEFORE the command opens it, so render/server/studio all see
    # the same self-contained bundle. It runs ahead of every dispatch branch
    # below (server/studio return early from main), and any hard failure aborts
    # here rather than surfacing mid-open.
    if getattr(args, "symlink_media", False):
        symlink_status = _run_symlink_media(args)
        if symlink_status != 0:
            return symlink_status

    # ``examples`` and ``doctor`` never compile a project, so they are handled
    # before the compile path (they do not carry input/project/bindings).
    if args.command == "examples":
        return _run_examples(parser, args)
    if args.command == "doctor":
        return _run_doctor()
    if args.command == "projects":
        return _run_projects(args)
    if args.command == "server":
        return _run_server_command(parser, args)
    if args.command == "studio":
        return _run_studio_command(parser, args)
    if args.command == "proxy":
        return _run_proxy(args)

    if args.command == "render":
        args.output = _resolve_render_output(args)

    from .core.compiler import compile_fcpxml
    from .core.errors import FCPXMLRenderError
    from .core.model import dataclass_json, fraction_json
    from .executor import execute_render

    compiled = None
    try:
        compiled = compile_fcpxml(
            args.input,
            project=args.project,
            bindings_path=args.bindings,
            media_preference=getattr(args, "prefer", "original"),
        )
        if args.command == "inspect":
            report_path = args.report.expanduser().resolve() if args.report else args.input.expanduser().resolve().with_suffix(".compatibility.json")
            compiled.report.write(report_path)
            if args.emit_plan:
                plan_path = args.emit_plan.expanduser().resolve()
                plan_path.parent.mkdir(parents=True, exist_ok=True)
                plan_path.write_text(
                    json.dumps(dataclass_json(compiled.render), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            payload = {
                "schema_version": 1,
                "project_name": compiled.render.project_name,
                "source_sha256": compiled.render.source_sha256,
                "format": {
                    "width": compiled.render.width,
                    "height": compiled.render.height,
                    "frame_duration": fraction_json(compiled.render.frame_duration),
                    "frame_count": compiled.render.frame_count,
                },
                "clip_count": len(compiled.render.clips),
                "transition_count": len(compiled.render.transitions),
                "compatibility": compiled.report.to_json(),
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
            if compiled.report.findings:
                print(compiled.report.human_summary(), file=sys.stderr)
            if args.strict and compiled.report.has_strict_failures:
                return 1
            return 0

        if args.backend == "tensor" and args.device != "auto":
            from .tensor.renderer import require_torch_device

            try:
                require_torch_device(args.device)
            except Exception as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
        result = execute_render(
            compiled.render,
            compiled.report,
            output_path=args.output,
            report_path=args.report,
            manifest_path=args.manifest,
            emit_plan_path=args.emit_plan,
            strict=args.strict,
            video_only=args.video_only,
            output_profile=_output_profile(parser, args),
            backend=args.backend,
            device=None if args.device == "auto" else args.device,
            cpu_segmentation=None,
            cpu_segment_parallelism=1,
            render_profile="reference",
            encoder_preset=args.encoder_preset,
            show_progress=args.backend == "tensor" and not args.no_progress,
        )
        if compiled.report.findings:
            print(compiled.report.human_summary(), file=sys.stderr)
        print(
            json.dumps(
                {
                    "output": str(result.output_path),
                    "report": str(result.report_path),
                    "manifest": str(result.manifest_path),
                    "degraded": result.degraded,
                    "requested_backend": result.requested_backend,
                    "selected_backend": result.selected_backend,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except FCPXMLRenderError as exc:
        if args.command == "render" and compiled is None:
            _write_compile_failure_artifacts(args, exc)
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _run_symlink_media(args: argparse.Namespace) -> int:
    """Consolidate the bundle's referenced media, printing a concise report.

    Main callers: ``main`` when ``--symlink-media`` is set on ``render``,
    ``server run``, or ``studio``.

    Returns 0 on success (even a no-op is success), or 1 when consolidation
    hits a hard, unrecoverable condition (input is not a bundle, a malformed or
    unsafe ``Info.fcpxml``, two distinct files colliding on one ``Media/<name>``
    link, or the document changing on disk mid-run). Offline media is a
    warning inside the report, never a failure here.

    Every failure type derives from ``FCPXMLRenderError`` (``MediaConsolidationError``,
    ``FCPXMLParseError``, ``FCPXMLWriteConflictError``), so the shared base is
    caught here: a parse failure is a concise ``error:`` line like any other,
    never a traceback.
    """

    from .core.errors import FCPXMLRenderError
    from .core.media_consolidate import consolidate_bundle_media

    try:
        result = consolidate_bundle_media(args.input)
    except FCPXMLRenderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for line in result.summary_lines():
        print(line, file=sys.stderr)
    return 0


def _run_proxy(args: argparse.Namespace) -> int:
    """Generate proxy media for a document's video assets, printing a report.

    Main callers: ``main`` for the ``proxy`` subcommand.

    Returns 0 on success (a no-op -- e.g. every asset already proxied -- is still
    success), or 1 when generation hits a hard condition (a bad input, an
    ffmpeg/ffprobe failure on a source, or the document changing on disk while
    proxies were encoding). Unresolvable media is surfaced as a warning in the
    report, never a hard failure.

    ``ProxyGenerationError``, ``FCPXMLParseError`` and ``FCPXMLWriteConflictError``
    all derive from ``FCPXMLRenderError``, so the shared base is caught.
    """

    from .core.errors import FCPXMLRenderError
    from .core.proxy_media import generate_proxies

    try:
        result = generate_proxies(
            args.input,
            target_short_side=args.height,
            bitrate_kbps=args.bitrate,
            overwrite=args.overwrite,
        )
    except FCPXMLRenderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for line in result.summary_lines():
        print(line, file=sys.stderr)
    return 0


def _write_compile_failure_artifacts(args: argparse.Namespace, error: Exception) -> None:
    """Persist render diagnostics when parsing/compilation never produced an IR.

    Main callers:
    - ``main`` for failures before ``execute_render`` takes ownership.
    """

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    report_path = args.report.expanduser().resolve() if args.report else output_path.with_suffix(".compatibility.json")
    manifest_path = args.manifest.expanduser().resolve() if args.manifest else output_path.with_suffix(".manifest.json")
    try:
        digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    except OSError:
        digest = None
    finding = {
        "outcome": "failed",
        "portable_status": "unsupported",
        "fcpxml_path": "fcpxml",
        "construct": "portable render compilation",
        "uid": None,
        "timeline_start": None,
        "timeline_duration": None,
        "disposition": str(error),
    }
    report = {
        "schema_version": 1,
        "source_path": str(input_path),
        "source_sha256": digest,
        "project_name": None,
        "timeline_start": None,
        "timeline_duration": None,
        "degraded": False,
        "counts": {"exact": 0, "approximated": 0, "omitted": 0, "failed": 1, "info": 0},
        "findings": [finding],
    }
    manifest = {
        "schema_version": 1,
        "engine": "bladeworks-portable-fcpxml",
        "engine_version": 1,
        "status": "failed",
        "error": str(error),
        "source_path": str(input_path),
        "source_sha256": digest,
        "project_name": None,
        "ffmpeg_version": None,
        "render_backend": {
            "requested": getattr(args, "backend", "cpu"),
            "selected": None,
            "fallback_reason": None,
        },
        "output": {"path": str(output_path)},
        "asset_bindings": [],
        "font_bindings": [],
        "compatibility": {
            "report_path": str(report_path),
            "degraded": False,
            "counts": report["counts"],
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _project_label(project) -> str:
    """Format one project as ``Name [uid]`` with loud fallbacks.

    Mirrors ``parser._project_choices`` so the browse listing and the parser's
    ambiguity/selection errors name the very same projects the same way.
    """

    name = project.get("name") or "Untitled Project"
    uid = project.get("uid") or "no uid"
    return f"{name} [{uid}]"


def _run_projects(args: argparse.Namespace) -> int:
    """List the projects a ``.fcpxml`` file / ``.fcpxmld`` bundle contains.

    Why this exists: ``render``/``inspect`` take ``--project NAME_OR_UID`` and
    the parser's ambiguity error already names the choices, but there was no
    NON-error way to browse them. This gives one.

    Steps, procedurally:
    1. Resolve + securely read the input into its ``<fcpxml>`` root
       (``read_fcpxml_root`` handles the ``.fcpxmld`` bundle -> ``Info.fcpxml``
       resolution), catching parse errors as a loud non-zero exit.
    2. Enumerate ``(event, project)`` pairs via ``list_library_projects`` -- the
       SAME source of truth ``_select_project`` uses, so the browse view and
       the selectable set never drift apart.
    3. Fail loudly (non-zero, no silent fallback) when the document holds no
       project inside a library event, using the parser's exact message.
    4. Print either a ``--json`` array or a human listing grouped by
       library -> event, ending with a copy-pasteable ``render`` hint.

    Main callers: ``main`` for the ``projects`` subcommand.
    """

    from .core.errors import FCPXMLParseError
    from .core.parser import list_library_projects, read_fcpxml_root

    try:
        root = read_fcpxml_root(args.input)
    except FCPXMLParseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    candidates = list_library_projects(root)
    if not candidates:
        # Match the parser's message verbatim; no silent fallback.
        print(
            "error: document does not contain a project inside a library event",
            file=sys.stderr,
        )
        return 1

    if args.json:
        payload = [
            {"event": event.get("name"), "name": project.get("name"), "uid": project.get("uid")}
            for event, project in candidates
        ]
        print(json.dumps(payload, indent=2))
        return 0

    # Human listing grouped by library -> event. ``candidates`` is in document
    # order, so an event's projects are contiguous; we map each event back to
    # its owning library element to print the two levels of headers. Identity
    # (``id(...)``) is the group key so libraries/events that share a name still
    # print as distinct groups.
    library_of_event = {}
    for library in root:
        if _tag_name(library) != "library":
            continue
        for event in library:
            if _tag_name(event) == "event":
                library_of_event[id(event)] = library

    last_library_id = None
    last_event_id = None
    for event, project in candidates:
        library = library_of_event[id(event)]
        if id(library) != last_library_id:
            print(f"library: {library.get('name') or 'Untitled Library'}")
            last_library_id = id(library)
            last_event_id = None
        if id(event) != last_event_id:
            print(f"  event: {event.get('name') or 'Untitled Event'}")
            last_event_id = id(event)
        print(f"    {_project_label(project)}")

    print(f"render one with: fcpxml render {args.input} --project NAME_OR_UID")
    return 0


def _tag_name(element) -> str:
    """Local-name of an element, stripping any XML namespace (like parser._tag)."""

    return element.tag.rsplit("}", 1)[-1]


def _run_examples(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Handle ``examples ls`` and ``examples cp`` against the packaged manifest.

    - ``ls`` prints each sample's name and one-line description (display order
      is the manifest order).
    - ``cp <name> [DEST]`` copies the sample ``.fcpxmld`` bundle into DEST
      (default the current directory). It errors loudly -- never silently -- on
      an unknown name (surfaces the known names) or when the target bundle
      already exists at DEST (no silent overwrite of a user's file).

    Main callers: ``main`` for the ``examples`` subcommand.
    """

    from .examples import EXAMPLES, example_bundle

    if args.examples_command == "ls":
        width = max(len(name) for name in EXAMPLES)
        for name, example in EXAMPLES.items():
            print(f"{name.ljust(width)}  {example.description}")
        return 0

    # examples cp
    try:
        source = example_bundle(args.name)
    except KeyError as exc:
        # str(KeyError) wraps the message in quotes; use the raw argument.
        print(f"error: {exc.args[0]}", file=sys.stderr)
        return 1

    dest_dir = args.dest.expanduser()
    target = dest_dir / source.name
    if target.exists():
        print(f"error: destination already exists: {target} (refusing to overwrite)", file=sys.stderr)
        return 1
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    print(str(target))
    return 0


def _run_doctor() -> int:
    """Report runtime prerequisites and exit non-zero if a hard one is missing.

    Checks, in order:
    - ``ffmpeg`` and ``ffprobe`` on PATH (HARD prerequisites: live preview
      audio and probing shell out to them). If either is absent, print a loud
      install hint and make the overall exit non-zero.
    - The torch compute device the tensor backend would select (mps > cuda >
      cpu), matching ``renderer._select_device``'s preference order.
    - Pillow's version and Raqm text shaping support. Raqm is a hard
      prerequisite because rendering text without it changes layout.
    - Versions of python, torch, av (PyAV), and ffprobe, so a bug report can
      pin the toolchain.

    Main callers: ``main`` for the ``doctor`` subcommand.
    """

    ok = True

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        print(f"ffmpeg: OK ({ffmpeg})")
    else:
        ok = False
        print("ffmpeg: MISSING -- required for Studio preview audio.", file=sys.stderr)
        print("  install hint: `brew install ffmpeg` (macOS) or your distro's ffmpeg package.", file=sys.stderr)

    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        print(f"ffprobe: OK ({ffprobe})")
    else:
        ok = False
        print("ffprobe: MISSING -- required for source probing.", file=sys.stderr)
        print("  install hint: `brew install ffmpeg` (macOS) or your distro's ffmpeg package.", file=sys.stderr)

    print(f"python: {sys.version.split()[0]} ({sys.executable})")

    try:
        from PIL import __version__ as pillow_version
        from PIL import features

        raqm_available = bool(features.check("raqm"))
        print(f"Pillow: {pillow_version}")
        if raqm_available:
            print("RAQM: OK")
        else:
            ok = False
            print(
                "RAQM: MISSING -- required for correct text shaping.",
                file=sys.stderr,
            )
            print(
                "  install hint: use the supported Bladeworks installer; "
                "developer installs must provide FriBiDi, HarfBuzz, and a "
                "Pillow build with RAQM enabled.",
                file=sys.stderr,
            )
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"Pillow: UNAVAILABLE ({exc})", file=sys.stderr)
        print(
            "  install hint: reinstall Bladeworks with its supported installer.",
            file=sys.stderr,
        )

    try:
        import torch  # local import: torch is heavy and only needed here / in the tensor backend

        device = "cpu"
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        print(f"torch: {torch.__version__} (device: {device})")
    except Exception as exc:  # noqa: BLE001 - report, don't crash the doctor
        ok = False
        print(f"torch: UNAVAILABLE ({exc})", file=sys.stderr)

    try:
        import av  # PyAV: decode/encode backing the tensor renderer

        print(f"av (PyAV): {av.__version__}")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"av (PyAV): UNAVAILABLE ({exc})", file=sys.stderr)

    if ffprobe:
        print(f"ffprobe version: {_ffprobe_version(ffprobe)}")

    return 0 if ok else 1


def _run_server_command(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Dispatch the lazy server commands without burdening render startup."""

    from .core.errors import FCPXMLRenderError
    from .preview.runner import ServerConfig, check_health, run_server

    if args.server_command == "health":
        if args.timeout <= 0:
            parser.error("server health --timeout must be greater than zero")
        return check_health(args.url, timeout=args.timeout)
    if not 0 <= args.port <= 65535:
        parser.error("server run --port must be between 0 and 65535")
    if args.decoder_threads < 1:
        parser.error("server run --decoder-threads must be at least 1")
    if args.history_limit < 1:
        parser.error("server run --history-limit must be at least 1")
    try:
        return run_server(
            ServerConfig(
                source=args.input,
                host=args.host,
                port=args.port,
                device=args.device,
                decoder_threads=args.decoder_threads,
                history_limit=args.history_limit,
                render_directory=args.render_dir,
                log_level=args.log_level,
                strict=args.strict,
                allowed_origins=tuple(args.allow_origin),
            )
        )
    except (OSError, RuntimeError, ValueError, FCPXMLRenderError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def _run_studio_command(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Run the same local API with packaged static assets mounted at root."""

    from .core.errors import FCPXMLRenderError
    from .preview.runner import ServerConfig, run_studio

    if not 0 <= args.port <= 65535:
        parser.error("studio --port must be between 0 and 65535")
    if args.decoder_threads < 1:
        parser.error("studio --decoder-threads must be at least 1")
    if args.history_limit < 1:
        parser.error("studio --history-limit must be at least 1")
    try:
        return run_studio(
            ServerConfig(
                source=args.input,
                host=args.host,
                port=args.port,
                device=args.device,
                decoder_threads=args.decoder_threads,
                history_limit=args.history_limit,
                render_directory=args.render_dir,
                log_level=args.log_level,
                strict=args.strict,
                allowed_origins=tuple(args.allow_origin),
            ),
            open_browser=not args.no_open,
        )
    except (OSError, RuntimeError, ValueError, FCPXMLRenderError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def _ffprobe_version(ffprobe: str) -> str:
    """Return the ffprobe version string, or a loud placeholder on failure."""

    try:
        result = subprocess.run(
            (ffprobe, "-v", "error", "-show_entries", "program_version=version",
             "-of", "default=noprint_wrappers=1:nokey=1"),
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip() or "unknown"
    except Exception as exc:  # noqa: BLE001
        return f"unknown ({exc})"


if __name__ == "__main__":
    raise SystemExit(main())
