"""Generate disposable, self-contained Studio libraries for the two-hour QA pass.

Architecture map
----------------
1. ``build_media_corpus`` asks the system FFmpeg for four synthetic A/V files.
   Hard edges, grids, motion, and stepped audio levels make visual defects easy
   to see and waveform defects easy to hear.
2. ``BUNDLES`` stores six small FCPXML documents.  Each document isolates one
   product area instead of combining every feature into one ambiguous Project.
3. ``generate`` writes each ``Info.fcpxml`` and copies only the media that the
   bundle references.  Every resulting ``.fcpxmld`` can therefore be copied,
   edited, restarted, and deleted independently.

The output is deliberately disposable because Studio saves edits back into
``Info.fcpxml``.  The committed source of truth is this generator, not a bundle
that a previous manual test may already have mutated.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


HEADER = """<?xml version="1.0"?><!DOCTYPE fcpxml><fcpxml version="1.14">
  <resources>
    <format id="land" frameDuration="1/30s" width="640" height="360" colorSpace="1-1-1 (Rec. 709)"/>
    <format id="portrait" frameDuration="1/30s" width="360" height="640" colorSpace="1-1-1 (Rec. 709)"/>
    <format id="square" frameDuration="1/30s" width="540" height="540" colorSpace="1-1-1 (Rec. 709)"/>
    <asset id="grid" name="Moving Grid" start="0s" duration="12s" hasVideo="1" hasAudio="1" videoSources="1" audioSources="1" audioChannels="1" audioRate="48000" format="land"><media-rep kind="original-media" src="Media/grid.mp4"/></asset>
    <asset id="bars" name="Reference Bars" start="0s" duration="12s" hasVideo="1" hasAudio="1" videoSources="1" audioSources="1" audioChannels="1" audioRate="48000" format="land"><media-rep kind="original-media" src="Media/bars.mp4"/></asset>
    <asset id="clock" name="Motion Clock" start="0s" duration="12s" hasVideo="1" hasAudio="1" videoSources="1" audioSources="1" audioChannels="1" audioRate="48000" format="land"><media-rep kind="original-media" src="Media/clock.mp4"/></asset>
    <asset id="pulse" name="Stepped Audio" start="0s" duration="12s" hasVideo="1" hasAudio="1" videoSources="1" audioSources="1" audioChannels="1" audioRate="48000" format="land"><media-rep kind="original-media" src="Media/pulse.mp4"/></asset>
    <effect id="cross" name="Cross Dissolve" uid="FxPlug:4731E73A-8DAC-4113-9A30-AE85B1761265"/>
    <effect id="color" name="Color Adjustments" uid="FxPlug:7E2022A5-202B-4EEB-A311-AC2B585D01B0"/>
    <effect id="negative" name="Negative" uid="FxPlug:8BBE307B-F5BC-4704-AD31-10CA5CC9B12B"/>
    <effect id="basic" name="Basic Title" uid=".../Titles.localized/Bumper:Opener.localized/Basic Title.localized/Basic Title.moti"/>
  </resources>
"""


def document(library: str, body: str) -> str:
    """Wrap Event and Project XML in the shared resources and Library.

    Main callers: the ``BUNDLES`` declarations below.
    """

    return f'{HEADER}  <library name="{library}">\n{body}\n  </library>\n</fcpxml>\n'


BUNDLES: dict[str, str] = {
    "qa_01_navigation_formats.fcpxmld": document(
        "QA 01 Navigation and Formats",
        """    <event name="A. Landscape">
      <project name="Shared Name" uid="qa-nav-land"><sequence format="land" duration="6s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k"><spine>
        <asset-clip ref="grid" name="Landscape Grid" offset="0s" start="0s" duration="6s" audioRole="dialogue"/>
      </spine></sequence></project>
      <project name="Second Cut" uid="qa-nav-second"><sequence format="land" duration="8s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k"><spine>
        <asset-clip ref="bars" name="Bars First" offset="0s" start="0s" duration="4s" audioRole="effects"/>
        <asset-clip ref="clock" name="Clock Second" offset="4s" start="2s" duration="4s" audioRole="dialogue"/>
      </spine></sequence></project>
    </event>
    <event name="B. Alternate Shapes">
      <project name="Shared Name" uid="qa-nav-portrait"><sequence format="portrait" duration="6s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k"><spine>
        <asset-clip ref="clock" name="Portrait Fill" offset="0s" start="0s" duration="6s" audioRole="dialogue"><adjust-conform type="fill"/></asset-clip>
      </spine></sequence></project>
      <project name="Square Reference" uid="qa-nav-square"><sequence format="square" duration="6s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k"><spine>
        <asset-clip ref="grid" name="Square Fit" offset="0s" start="1s" duration="6s" audioRole="dialogue"><adjust-conform type="fit"/></asset-clip>
      </spine></sequence></project>
    </event>
    <event name="C. Collapse Me">
      <project name="Short Project" uid="qa-nav-short"><sequence format="land" duration="2s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k"><spine>
        <asset-clip ref="bars" name="Two Seconds" offset="0s" start="0s" duration="2s" audioRole="effects"/>
      </spine></sequence></project>
    </event>""",
    ),
    "qa_02_geometry_controls.fcpxmld": document(
        "QA 02 Geometry Controls",
        """    <event name="Geometry">
      <project name="Landscape Geometry" uid="qa-geo-land"><sequence format="land" duration="8s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k"><spine>
        <asset-clip ref="bars" name="Background Bars" offset="0s" start="0s" duration="8s" audioRole="effects">
          <asset-clip ref="grid" name="Connected Grid" lane="1" offset="1s" start="2s" duration="6s" audioRole="dialogue">
            <adjust-transform position="18 9" scale="0.52 0.52" rotation="-12" anchor="8 -5"/>
            <adjust-crop mode="trim"><trim-rect left="8" right="14" top="10" bottom="5"/></adjust-crop>
            <adjust-blend amount="0.82" mode="normal"/>
          </asset-clip>
        </asset-clip>
      </spine></sequence></project>
      <project name="Portrait Geometry" uid="qa-geo-portrait"><sequence format="portrait" duration="6s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k"><spine>
        <asset-clip ref="clock" name="Portrait Clock" offset="0s" start="0s" duration="6s" audioRole="dialogue"><adjust-conform type="fill"/><adjust-transform position="-12 15" scale="1.18 1.18" rotation="7"/></asset-clip>
      </spine></sequence></project>
      <project name="Square Geometry" uid="qa-geo-square"><sequence format="square" duration="6s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k"><spine>
        <asset-clip ref="grid" name="Square Grid" offset="0s" start="0s" duration="6s" audioRole="dialogue"><adjust-conform type="fit"/><adjust-transform position="0 0" scale="0.86 0.86" rotation="0"/></asset-clip>
      </spine></sequence></project>
    </event>""",
    ),
    "qa_03_audio_waveforms.fcpxmld": document(
        "QA 03 Audio and Waveforms",
        """    <event name="Audio">
      <project name="Waveform Steps" uid="qa-audio-steps"><sequence format="land" duration="12s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k"><spine>
        <asset-clip ref="pulse" name="Quiet Loud Medium" offset="0s" start="0s" duration="12s" audioRole="dialogue">
          <adjust-volume amount="0dB"><param name="amount"><fadeIn type="linear" duration="1s"/><fadeOut type="easeOut" duration="1s"/></param></adjust-volume>
          <audio ref="clock" name="Connected Tone" lane="-1" offset="2s" start="1s" duration="7s" role="music"><adjust-volume amount="-12dB"/></audio>
        </asset-clip>
      </spine></sequence></project>
      <project name="Mute Isolation" uid="qa-audio-mute"><sequence format="land" duration="8s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k"><spine>
        <asset-clip ref="grid" name="Visible While Muted" offset="0s" start="0s" duration="8s" audioRole="dialogue"/>
      </spine></sequence></project>
    </event>""",
    ),
    "qa_04_timeline_editing.fcpxmld": document(
        "QA 04 Timeline Editing",
        """    <event name="Editing">
      <project name="Ripple Split Retime" uid="qa-edit-main"><sequence format="land" duration="12s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k"><spine>
        <asset-clip ref="grid" name="A Grid" offset="0s" start="0s" duration="4s" audioRole="dialogue"><title ref="basic" name="Attached Title" lane="1" offset="1s" start="0s" duration="2s"><text><text-style ref="edit-title-style">ATTACHED TO A</text-style></text><text-style-def id="edit-title-style"><text-style font="Helvetica" fontFace="Bold" fontSize="40" fontColor="1 1 1 1" alignment="center"/></text-style-def></title></asset-clip>
        <asset-clip ref="bars" name="B Bars" offset="4s" start="1s" duration="4s" audioRole="effects"><asset-clip ref="clock" name="B Connected" lane="1" offset="5s" start="2s" duration="2s" audioRole="dialogue"><adjust-transform position="20 10" scale="0.4 0.4"/></asset-clip></asset-clip>
        <asset-clip ref="clock" name="C Clock" offset="8s" start="2s" duration="4s" audioRole="music"/>
      </spine></sequence></project>
    </event>""",
    ),
    "qa_05_inspector_metadata.fcpxmld": document(
        "QA 05 Inspector and Metadata",
        """    <event name="Inspector">
      <project name="Color Blend Ratings" uid="qa-meta-main"><sequence format="land" duration="10s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k"><spine>
        <asset-clip ref="grid" name="Graded Favorite" offset="0s" start="0s" duration="5s" audioRole="dialogue">
          <rating start="0s" duration="5s" value="favorite"/>
          <marker start="1s" duration="1/30s" value="Check Color"/>
          <filter-video ref="color" name="Color Adjustments"><param name="Exposure" key="3" value="10"/><param name="Contrast" key="17" value="8"/><param name="Saturation" key="16" value="12"/></filter-video>
          <asset-clip ref="bars" name="Multiply Overlay" lane="1" offset="1s" start="0s" duration="3s" audioRole="effects"><adjust-transform position="12 0" scale="0.62 0.62"/><adjust-blend amount="0.72" mode="multiply"/><filter-video ref="negative" name="Negative" enabled="0"/></asset-clip>
        </asset-clip>
        <transition name="Cross Dissolve" offset="9/2s" duration="1s"><filter-video ref="cross" name="Cross Dissolve"/></transition>
        <asset-clip ref="clock" name="Titled Clock" offset="5s" start="1s" duration="5s" audioRole="music"><title ref="basic" name="QA Title" lane="1" offset="6s" start="0s" duration="2s"><text><text-style ref="meta-title-style">EDIT ME</text-style></text><text-style-def id="meta-title-style"><text-style font="Helvetica" fontFace="Bold" fontSize="48" fontColor="1 0.9 0.2 1" alignment="center"/></text-style-def></title></asset-clip>
      </spine></sequence></project>
    </event>""",
    ),
    "qa_06_export_and_faults.fcpxmld": document(
        "QA 06 Export and Faults",
        """    <event name="Export">
      <project name="Export Reference" uid="qa-export"><sequence format="land" duration="8s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k"><spine>
        <asset-clip ref="grid" name="Export Grid" offset="0s" start="0s" duration="4s" audioRole="dialogue"><adjust-transform position="-8 4" scale="0.9 0.9"/><adjust-crop mode="trim"><trim-rect left="4" right="9" top="6" bottom="2"/></adjust-crop><adjust-volume amount="-5dB"/></asset-clip>
        <asset-clip ref="pulse" name="Export Pulse" offset="4s" start="2s" duration="4s" audioRole="music"><timeMap frameSampling="floor" preservesPitch="1"><timept time="2s" value="2s" interp="linear"/><timept time="6s" value="4s" interp="linear"/></timeMap><adjust-blend amount="0.75" mode="normal"/></asset-clip>
      </spine></sequence></project>
    </event>
    <event name="Faults">
      <project name="Missing Middle" uid="qa-fault-missing"><sequence format="land" duration="8s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k"><spine>
        <asset-clip ref="bars" name="Online Before" offset="0s" start="0s" duration="3s" audioRole="effects"/>
        <asset-clip ref="missing" name="INTENTIONALLY MISSING" offset="3s" start="0s" duration="2s" audioRole="dialogue"/>
        <asset-clip ref="clock" name="Online After" offset="5s" start="1s" duration="3s" audioRole="music"/>
      </spine></sequence></project>
    </event>""",
    ).replace(
        "    <effect id=\"basic\"",
        "    <asset id=\"missing\" name=\"Missing Camera\" start=\"0s\" duration=\"2s\" hasVideo=\"1\" hasAudio=\"1\" videoSources=\"1\" audioSources=\"1\" audioChannels=\"1\" audioRate=\"48000\" format=\"land\"><media-rep kind=\"original-media\" src=\"Media/intentionally_missing.mov\"/></asset>\n    <effect id=\"basic\"",
    ),
}


MEDIA_NAMES = ("grid.mp4", "bars.mp4", "clock.mp4", "pulse.mp4")


def run_ffmpeg(output: Path, video_source: str, tone_hz: int, audio_filter: str | None = None) -> None:
    """Create one short reference clip with deterministic video and audio.

    Main callers: ``build_media_corpus``.

    Why this exists: generated sources prove the QA pass is testing the current
    code, not stale media or a previously edited example bundle.
    """

    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", video_source,
        "-f", "lavfi", "-i", f"sine=frequency={tone_hz}:sample_rate=48000:duration=12",
    ]
    if audio_filter:
        command.extend(("-af", audio_filter))
    command.extend((
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264",
        "-preset", "veryfast", "-crf", "25", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "96k", "-shortest", str(output),
    ))
    subprocess.run(command, check=True)


def build_media_corpus(directory: Path) -> None:
    """Generate the four visual and audible references shared by the bundles.

    Main callers: ``generate``.
    """

    run_ffmpeg(directory / "grid.mp4", "testsrc2=size=640x360:rate=30:duration=12", 440)
    run_ffmpeg(directory / "bars.mp4", "smptehdbars=size=640x360:rate=30:duration=12", 660)
    run_ffmpeg(directory / "clock.mp4", "testsrc=size=640x360:rate=30:duration=12", 880)
    run_ffmpeg(
        directory / "pulse.mp4",
        "gradients=size=640x360:rate=30:duration=12:speed=0.08:c0=0x15223a:c1=0xd14d72:c2=0x29b6a8:c3=0xf0c75e:nb_colors=4",
        220,
        "volume='if(lt(mod(t,4),1),0.05,if(lt(mod(t,4),2),0.25,if(lt(mod(t,4),3),0.8,0.12)))':eval=frame",
    )


def generate(output_root: Path) -> None:
    """Create a clean suite and refuse to overwrite an existing destination.

    Main callers: ``main``.
    """

    if output_root.exists():
        raise SystemExit(f"refusing to overwrite existing path: {output_root}")
    output_root.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="bladeworks-qa-media-") as temporary:
        corpus = Path(temporary)
        build_media_corpus(corpus)
        for bundle_name, xml in BUNDLES.items():
            bundle = output_root / bundle_name
            media = bundle / "Media"
            media.mkdir(parents=True)
            (bundle / "Info.fcpxml").write_text(xml, encoding="utf-8")
            for media_name in MEDIA_NAMES:
                shutil.copy2(corpus / media_name, media / media_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new directory that will contain six .fcpxmld bundles")
    arguments = parser.parse_args()
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required; install it in your normal shell environment first")
    generate(arguments.output.expanduser().resolve())
    print(arguments.output.expanduser().resolve())


if __name__ == "__main__":
    main()
