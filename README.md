# Bladeworks FCPXML

Bladeworks FCPXML renders portable Final Cut Pro XML projects (`.fcpxml` files and
`.fcpxmld` bundles) to video.

## Install

### macOS

Homebrew installs Bladeworks FCPXML in an isolated Python environment and provides
FFmpeg, ffprobe, FriBiDi, HarfBuzz, and RAQM:

```bash
brew install bladeworks-ai/tap/fcpxml
fcpxml doctor
```

### Linux

The supported installer detects Ubuntu 22.04+ or Debian 12+ and installs native
dependencies:

```bash
curl -fsSL https://raw.githubusercontent.com/bladeworks-ai/fcpxml/main/install-linux.sh | bash
```

### Developer installation

```bash
python -m pip install bladeworks
```

The PyPI package does not install native libraries. Developers must separately
provide FFmpeg/ffprobe and a Pillow build with RAQM text shaping. Confirm the
resolved executable paths and libraries after installation:

```bash
fcpxml doctor
```

## Quick start

No project of your own yet? Render a packaged sample end to end:

```bash
fcpxml doctor                      # verify ffmpeg, ffprobe, and the torch device
fcpxml examples ls                 # list the packaged sample projects
fcpxml examples cp single_clip .   # copy single_clip.fcpxmld into the current directory
fcpxml render single_clip.fcpxmld --output out.mp4
```

## AI agent skill

The repository includes [`skills/bladeworks-fcpxml/SKILL.md`](skills/bladeworks-fcpxml/SKILL.md),
a portable agent skill for authoring and validating FCPXML against Bladeworks'
supported render surface. Its bundled references include the complete authoring
specification, focused domain guides, FCPXML 1.14 DTD, examples, and the
machine-readable capability registry.

Render your own project the same way:

```bash
fcpxml render path/to/project.fcpxmld --output output.mp4
```

> **Tip:** Want to use a library exported from Final Cut Pro? Choose
> **File > Export XML...**, then add `--symlink-media` when opening the exported
> `.fcpxmld` bundle:
>
> ```bash
> fcpxml studio path/to/library.fcpxmld --symlink-media
> ```
>
> Bladeworks symlinks the referenced media from your `.fcpbundle` into the
> exported bundle's `Media/` directory, so you do not need to copy the media.

## Preview & edit locally

Both open one `.fcpxmld` bundle on `127.0.0.1`, using the same render engine:

```bash
fcpxml studio path/to/project.fcpxmld   # interactive web editor (opens in a browser)
fcpxml server run path/to/project.fcpxmld   # headless HTTP API, no UI
```

Use `studio` to tweak and preview a bundle by hand; use `server` to drive the
renderer programmatically (`fcpxml server health --url <url>` checks readiness).

## More commands

```bash
fcpxml inspect path/to/project.fcpxml    # classify what a document uses, without rendering
fcpxml projects path/to/project.fcpxml   # list the projects a file/bundle contains
fcpxml proxy path/to/project.fcpxmld     # generate downscaled proxy media
fcpxml --help                            # all commands and options
```

## Capabilities

Bladeworks treats real Final Cut Pro as its correctness oracle. Our fidelity
goal differs by area:

- **Core mechanics**: spine & lanes, transform, crop, distort, conform,
  retiming, and compound clips. We aim for near-100% parity with Final Cut Pro,
  up to numeric differences in rendering arithmetic.
- **Effects & transitions**: we aim to cover every *default-included* Final Cut
  effect and transition template, at *semantic* parity, so the mechanism,
  timing, and endpoints match, while fine texture may differ.
- **Color**: we aim for 100% of Final Cut's *color-adjustment* functionality at
  strong approximate visual similarity, so a viewer shouldn't be able to tell
  the difference.

Where each capability stands today
(✅ supported · 🟡 partial · ⛔ out of scope near-term):

| Capability | Status | Notes |
|---|---|---|
| Spine, lanes & z-index | ✅ | Full connected-clip / lane compositing |
| Compositing (blend / opacity / alpha) | ✅ | 17 RGB blend modes, 4 matte modes; HSL modes on roadmap |
| Transform / Crop / Distort | ✅ | Keyframed; shear expressed via four-corner Distort |
| Color adjustments | ✅ | Basic grade exact; Color Board & Wheels approximated |
| Color curves / Hue-Sat curves / LUTs | ⛔ | Out of scope near-term |
| Effects & transitions | 🟡 | 32 effect ports, 46 transitions; parameter support varies by effect |
| Retime | 🟡 | Constant / reverse / hold / linear ramp; smooth ramps & frame-blend on roadmap |
| Compound & multicam clips | ✅ | Genuinely recursive |
| Text / titles | 🟡 | Composited from caller-supplied rasters; no native font engine |
| Masking & keying | ✅ | Numeric masks + green-screen keyer; ML / tracked masks out of scope |
| Media import | 🟡 | Broad codec support; HDR is tone-mapped to SDR today (native HDR & more formats on roadmap) |
| Export | 🟡 | H.264 delivery + ProRes 4444 alpha; HEVC / 10-bit / HDR / image-sequence exits on roadmap |
| Audio | 🟡 | Mono / stereo deliver; surround 5.1 on roadmap |
| Motion templates | ⛔ | Proprietary Motion rigs; out of scope near-term |

## Under the hood

We love tensors. As it turns out, PyTorch is a highly capable tensor-operation
package that is now portable across a range of GPU-accelerated platforms such as
NVIDIA graphics, Apple Silicon, and AMD. So we implemented the entire rendering system
in the Python ecosystem, on PyTorch: every pixel operation is a tensor op, and
the same code runs on whatever accelerator (or CPU) you have.

Core dependencies: **PyTorch** (rendering), **PyAV** (media decode / audio /
mux), NumPy, Pillow, and fonttools; the local server and Studio add FastAPI,
Uvicorn, and aiortc.

## Contributing

At this time, while in early development, we are not accepting external
source-code contributions. We do gladly welcome detailed **bug reports** and
**feature requests** in the Issues section. Using your coding agent to give
detailed feedback against the source will greatly increase the speed and
likelihood of your request being incorporated.

## Test a checkout

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pytest -q
```

## License

Bladeworks is licensed under the GNU Affero General Public License v3.0 only.
See [LICENSE](LICENSE).
