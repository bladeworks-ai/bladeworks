# Bladeworks Core: Document, Resource, and Media Model

This page is how to author the FCPXML document itself — structure, resource
identity, formats, assets, source media, and delivery — for the Bladeworks
renderer (`bladeworks render`). Timeline placement belongs in
[TIMELINE.md](TIMELINE.md); geometry and compositing in [GEOMETRY.md](GEOMETRY.md).

Bladeworks renders a subset of Final Cut's FCPXML, and the document scaffold
authors almost exactly as it does natively — so this page teaches the same
structure and expressions. A few base-FCPXML constructs are **not part of
Bladeworks**; they are called out inline where they come up and collected in
[What Bladeworks Does Not Render](#what-bladeworks-does-not-render), each with the
supported way to get the same result. Two document-wide requirements govern every
format the renderer touches: the project colour space must be **Rec.709 SDR**, and
formats must use **square pixels**.

Anchoring (backend): source decode `tensor/decode.py`; the structural construct
gate `tensor/support.py`; effect/title/transition matching `core/capabilities.py`;
colour `core/color.py`; delivery `executor._TENSOR_OUTPUT_PROFILES`. Paths under
[`../../../../src/bladeworks/`](../../../../src/bladeworks/).

## Document Structure

An FCPXML document has one root, a resource table, and a body such as a library:

```xml
<fcpxml version="1.14">
  <resources>...</resources>
  <library>...</library>
</fcpxml>
```

Resource declarations precede the body so later `ref` attributes can resolve them.
Bladeworks renders **one** `<project><sequence>`, which you select by name or UID:

```bash
bladeworks render <doc>.fcpxml --project NAME_OR_UID
```

Pass either a plain `.fcpxml` file or a `.fcpxmld` bundle directory. For a
bundle, Bladeworks reads `Info.fcpxml` from its root and preserves media paths
relative to the bundle.

Sequence `duration` covers its timeline. Bladeworks uses that authored value as
the output clock; it does not remeasure the spine. After an edit that changes
storyline length, write `<sequence duration>` to the new exact total. `tcStart`
and `tcFormat` define sequence timecode. `audioLayout` and `audioRate` define
sequence audio context.

Two things inside the sequence carry no pixels and therefore never draw:
`<marker>` and `<chapter-marker>` are timeline annotations, not picture. Author
them freely for downstream metadata; do not expect them to appear in the render.

## Authoring Profile Versus DTD Grammar

The default Spellshot profile uses one library, event, project, sequence, and
spine. These are authoring defaults, not grammar requirements — the DTD also
permits multiple events and projects, browser clips, collections, and reusable
media. A document may legally carry structure Bladeworks does not turn into pixels
(sibling projects, browser clips); that structure is not an error. Only the
sequence you name with `--project` produces a render; the rest of the document is
resolved for its resources and otherwise ignored.

## FCPXML Versions

`fcpxml/@version` selects a grammar. Bladeworks targets **1.14**; validate against
the DTD matching the document's declared version. Do not raise the version merely
because a newer grammar exists on the machine — upgrade only when the document
requires new syntax and the full document validates against the target DTD.

## Well-Formedness and DTD Validation

The target grammar (FCPXML 1.14) is checked into this reference next to this page:

```text
FCPXMLv1_14.dtd
```

This is Apple's own DTD, copied verbatim from Final Cut's Interchange framework so
an authoring agent can validate without a Final Cut install. Validate syntax
first, then grammar:

```bash
xmllint --noout document.fcpxml
xmllint --noout --dtdvalid FCPXMLv1_14.dtd document.fcpxml
```

DTD validity proves well-formedness and element structure only. It says nothing
about whether Bladeworks renders a given construct — that is a separate question,
decided when the document compiles. A perfectly DTD-valid document can still be
refused at plan time if it uses a construct outside Bladeworks; such refusals name
the construct loudly rather than dropping it silently, so treat a clean `xmllint`
run as a floor, not a guarantee.

The proof that the picture is right is a **render**, not a validation pass. Use
one-question microfixtures the way you would for Final Cut — two participants with
a gap to test transition adjacency, unequal left/right channels to test routing,
an asymmetric corner pin to reveal coordinate interpretation — and read the
rendered frames. A fixture that looks the same under competing interpretations
proves neither.

## Child Ordering

DTD content models are authoritative and Bladeworks requires DTD-valid input: a
child in the wrong position is a hard validation failure before the renderer ever
sees it. Rows that quote a model in backticks give the declaration verbatim:

| Parent | Declared content model |
| --- | --- |
| `fcpxml` | `import-options?`, `resources?`, then the body |
| `resources` | Resource declarations in any DTD-permitted sequence |
| `asset` | `(media-rep+, metadata?)` — at least one `media-rep`, **first** |
| `sequence` | `(note?, spine, metadata?)` — `metadata` **after** `spine` |
| `asset-clip` | `note?`, timing params, intrinsic adjustments, **anchored children**, markers, `audio-channel-source*`, video filters, `filter-audio*`, `metadata?` |
| `title` | `param*`, `text*`, `text-style-def*`, `note?`, video adjustments, **anchored children**, markers, video filters, `metadata?` |

Two orderings are easy to get backwards and are silent in review but fatal at
validation: `metadata` follows the content it describes (`media-rep`, `spine`),
never precedes it; and anchored children come **before** the source and filter
children on every story element except `mc-clip`, which declares `mc-source*`
first. Story elements other than `asset-clip` and `title` differ — see
[Intrinsic Adjustment Order](GEOMETRY.md#intrinsic-adjustment-order).

## IDs and References

- Every resource `id` must be unique in one document.
- Every `ref` must resolve to a compatible resource. An unresolved `ref` fails the
  document at compile — Bladeworks never renders a placeholder for a dangling
  reference.
- IDs are document-local; do not infer resource type from an ID prefix such as `r`.
- Names are labels, not identity.

## Formats

A `<format>` defines a media or sequence context. Keep distinct resources when
raster, frame duration, pixel aspect, colour space, or media type differs. Two
constraints apply to any format Bladeworks renders into:

- **Rec.709 SDR only.** The project/sequence colour space must be Rec.709
  (`core/color.py`). Author a Rec.709 SDR project; a wide-gamut or HDR project has
  no supported delivery path. (HDR *sources* are a separate matter — they are
  tone-mapped in on decode; see [Source Media](#source-media-which-sources-decode).)
- **Square pixels only.** Author square-pixel formats. There is no display-versus-pixel
  geometry gap to reconcile, so a non-square pixel aspect has no meaning here.

### Sequence Formats

```xml
<format id="project-format" name="FFVideoFormat1080p2997"
        frameDuration="1001/30000s" width="1920" height="1080"
        colorSpace="1-1-1 (Rec. 709)"/>
```

`sequence/@format` selects the project or reusable sequence context.

### Source-Media Formats

```xml
<format id="source-format" name="FFVideoFormat1080p5994"
        frameDuration="1001/60000s" width="1920" height="1080"
        colorSpace="1-1-1 (Rec. 709)"/>
```

Source cadence is independent of project cadence — do not rewrite source-time
rationals using the project frame duration. A source or container frame rate that
differs from the project is honoured, and frame ownership is floor / nearest-earlier
([TIMELINE.md](TIMELINE.md#frame-ownership-is-floor--nearest-earlier)).

### Rate-Undefined Still Formats

Stills commonly use a width/height format without `frameDuration`; the asset uses
`duration="0s"` and a timeline instance supplies a positive duration.

```xml
<format id="still-format" name="FFVideoFormatRateUndefined"
        width="4000" height="3000" colorSpace="1-13-1"/>
```

`1-13-1` is the colorSpace Final Cut writes for a rate-undefined still; it is not
the `1-1-1 (Rec. 709)` used for video formats, and the two are not interchangeable.

## Assets

An `<asset>` declares source timing, capabilities, format, identity, and media
representations:

```xml
<asset id="camera" name="camera.mov" uid="EXAMPLE-CAMERA-UID"
       start="3600s" duration="10s" hasVideo="1" videoSources="1"
       hasAudio="1" audioSources="1" audioChannels="2" audioRate="48000"
       format="source-format">
  <media-rep kind="original-media" src="file:///tmp/camera.mov"/>
</asset>
```

Bladeworks decodes pixels from `media-rep/@src`, so that URL must point at a file
that exists and is readable on disk — a missing or unreadable file fails the clip
loudly rather than rendering black. Encode spaces and special characters in the
URL correctly. `asset/@start` anchors source timecode: a clip with `start="3602s"`
on an asset whose `start="3600s"` selects two seconds in; this says nothing about
where the clip sits in the project (see [TIMELINE.md](TIMELINE.md)).

Omit video-only fields for audio-only media, and write `videoSources` on every
video-bearing asset — a single-stream file is `1`. Validate every attribute
against the target DTD.

### Still-Image Assets

A still asset is `duration="0s"`; the timeline element decides how long the still
appears.

```xml
<asset id="photo" name="photo.jpg" uid="EXAMPLE-PHOTO-UID"
       start="0s" duration="0s" hasVideo="1" videoSources="1"
       format="still-format">
  <media-rep kind="original-media" src="file:///tmp/photo.jpg"/>
</asset>
```

The canonical untrimmed source start is `start="3600s"`. When left-trimming a
still, advance its source start and reduce its duration by the same amount:

```text
trimmed source start = 3600s + left trim amount
```

### Stable Media Identity

`asset/@uid` is the stable identity of a source file. Bladeworks resolves the asset
by this identity but reads pixels from `media-rep/@src`, so both must be correct.
Reuse the exact uid from an exported Final Cut document; for brand-new file-backed
media, mint one stable identity per file. A missing `uid`, a placeholder
(`example-*`, `placeholder-*`), or a positional stand-in such as `r2` is invalid.
The `EXAMPLE-*-UID` values in these pages are illustrative placeholders, never
literals to copy. `media-rep/@sig` is optional Final Cut bookkeeping — do not
invent one.

In a Spellshot workspace the brand-new-file identity is derived from the canonical
file URL by the bundled validator; that derivation is documented in the Final Cut
agent instructions, not here.

## Source Media: Which Sources Decode

The gate on a source clip is its **decoded pixel format plus colour tags**, not its
codec. Any stream PyAV/libav can demux and decode — H.264, HEVC, ProRes 422/4444,
in `.mov` / `.mp4` / etc. — is accepted, then read against the surface below
(`tensor/decode.py`).

### Pixel formats

Planar YUV, 8 / 10 / 12-bit (`decode.py::_PIXEL_FORMATS`):

- 8-bit: `yuv420p`, `yuv422p`, `yuv444p` and their full-range `yuvj*` variants.
- 10-bit: `yuv420p10le`, `yuv422p10le`, `yuv444p10le`.
- 12-bit: `yuv444p12le` — ProRes 4444 **without** alpha decodes as 12-bit 4:4:4.

Anything outside this planar set — `nv12` / `p010` hwaccel surfaces, gray, packed
RGB — has no decode path; deliver the source as one of the planar formats above.

### Colour matrices

`bt709` → BT.709; `bt470bg` / `smpte170m` → BT.601; `bt2020nc` → BT.2020
non-constant. An **unspecified** matrix is read as BT.601, matching swscale's
default (including for subsampled sources regardless of resolution), so tag a
source explicitly if BT.601 would be wrong for it. `yuvj*` is full-range by
definition; `pc` → full range, `tv` / unspecified → limited. Exotic matrices
(`fcc`, `smpte240m`, `ycgco`, constant-luminance `bt2020c`, chroma-derived,
`ictcp`, RGB) are not read — transcode to a Rec.709/601/2020-nc source.

### HDR sources

An HLG (`arib-std-b67`) or PQ (`smpte2084`) source is accepted only when it carries
a Rec.2020 matrix and Rec.2020 primaries, and is tone-mapped in to 100-nit Rec.709
SDR through a fixed `.cube` LUT (`tensor/hdr.py`). This is a fixed tone-map, not a
grade — author around that if you need precise HDR highlights. HLG/PQ with any
other matrix or primaries is malformed and is refused.

### Source rotation and intrinsics

Source display-rotation metadata is honoured — validate transforms against the
displayed (rotated) source. Other spatial intrinsics baked into a source —
360°/stereo clip intrinsics, stabilization, rolling-shutter — are not applied.
Bake the intended result into the source frames before importing, or author the
equivalent motion in [GEOMETRY.md](GEOMETRY.md). (360° *transitions* are a
separate, supported construct — see
[EFFECTS_AND_TRANSITIONS.md](EFFECTS_AND_TRANSITIONS.md).)

### Alpha-carrying sources

Alpha is an *output* format (see [Delivery](#delivery-output-profiles)), not an
input one: an alpha pixel format on a **source** is refused rather than
compositing its own transparency. To key in transparency, matte the source ahead
of time, or drive opacity with `<adjust-blend>`
([GEOMETRY.md](GEOMETRY.md#opacity-and-blend)).

## Media Resources

`<media>` contains reusable Final Cut structures. Compound sequences, multicam,
sync-clip, and audition containers all render — they lower to recursive group
scopes, and nesting is genuinely recursive (a compound inside a compound, a
multicam angle scope). Every reusable `<sequence>` owns its own format, raster,
cadence, pixel aspect, colour space, and timecode domain; its descendants are
interpreted in that local context, and a parent `ref-clip` receives the completed
surface. See
[TIMELINE.md#compounds-and-scopes](TIMELINE.md#compounds-and-scopes) for the scope
model and its geometry/retime consequences.

When editing an existing compound or multicam, preserve the complete nested subtree
— source-format references, clip timing, sync data, `media/@uid`,
`mc-angle/@angleID` — because a partial subtree resolves incorrectly.

## Effect and Title Resources

An `<effect>` gives a reusable title, filter, or transition identity:

```xml
<effect id="basic-title" name="Basic Title"
        uid=".../Basic Title.localized/Basic Title.moti"/>
```

An `<effect>` is matched by `uid` (exact, then `uid_glob`) and, only for a UID-less
resource, by an explicit registry `alias` — it is never guessed from display text
(`core/capabilities.py::match`). A known UID proves identity only, not the validity
of arbitrary `<param>` keys; whether a matched effect/title/transition renders is
covered in [EFFECTS_AND_TRANSITIONS.md](EFFECTS_AND_TRANSITIONS.md) and
[INVENTORY.md](INVENTORY.md).

A supported `<title>` is laid out from its `<text>` runs by Bladeworks' runtime
rasterizer and then composited as an image layer. Author the text and supported
styles directly in FCPXML — see
[TITLES_AND_CAPTIONS.md](TITLES_AND_CAPTIONS.md).

## Notes, Metadata, and Opaque Data

`<note>`, `<metadata>`, `effectConfig`, `effectData`, keyed archives, tracking
payloads, and browser-clip analysis carry no pixels. Their presence or absence
never changes the picture, so author them where the DTD allows (obeying child
ordering) as breadcrumbs or downstream metadata; they are neither required for a
render nor a reason it fails. The one place such a payload matters to the picture
is when it is the *only* faithful form of an active Apple-owned effect — that kind
of construct renders when it arrives in genuine Final Cut XML but cannot be
authored fresh.

## Delivery: Output Profiles

Bladeworks writes one of a fixed set of masters, chosen by flag
(`executor._TENSOR_OUTPUT_PROFILES`, selected by `cli.py::_output_profile`). All
containers carry `+faststart+write_colr`; audio is always **AAC**, muxed in-process
by the PyAV graph (see [AUDIO.md](AUDIO.md)).

| Profile | Flag | Codec / pixels | Container | Notes |
| --- | --- | --- | --- | --- |
| `delivery` (default) | — | H.264 `libx264`, 8-bit `yuv420p`, Rec.709 **limited** | `.mp4` | CRF 18 / preset `medium`, overridable via `--encoder-preset`. |
| `delivery_alpha` | `--alpha` | ProRes 4444 **straight** (un-premultiplied) alpha, `prores_ks` profile 4, `yuva444p10le` | `.mov` (enforced) | The one path that exports alpha; `--alpha` requires a `.mov`. |
| silent video | `--video-only` | as `delivery`, silent audio track | `.mp4` | Explicitly omits source audio; fails `--strict` per the fidelity policy. |

These are the whole delivery surface. HEVC/H.265 delivery, 10-bit H.264, HDR
(PQ/HLG) delivery, and image-sequence export are not written and are never
silently swapped for a lookalike — pick one of the profiles above.

## What Bladeworks Does Not Render

These base-FCPXML constructs are outside Bladeworks. Each is refused loudly at
compile/plan time rather than rendered wrong or dropped silently; author the
supported alternative instead.

| Not rendered | Author instead |
| --- | --- |
| `<marker>` / `<chapter-marker>` picture | Nothing to render — they stay as metadata only. |
| Non-Rec.709 / wide-gamut / HDR **project** | A Rec.709 SDR project (HDR *sources* are tone-mapped in on decode). |
| Non-square pixel aspect (`paspH`/`paspV` ≠ 1) | Square-pixel formats. |
| Alpha-carrying **source** | Matte the source ahead of time, or drive opacity with `<adjust-blend>`. |
| Exotic pixel formats (`nv12`, `p010`, gray, packed RGB) | A planar YUV source (8/10/12-bit) from the [supported set](#pixel-formats). |
| Exotic colour matrices (`fcc`, `smpte240m`, `ycgco`, `bt2020c`, `ictcp`, RGB) | A `bt709` / BT.601 / `bt2020nc` source. |
| Malformed HDR (HLG/PQ without Rec.2020 matrix + primaries) | A well-formed Rec.2020 HDR source, or a Rec.709 SDR source. |
| Spatial intrinsics — 360°/stereo clip, stabilization, rolling-shutter | Bake the result into the source, or author motion in [GEOMETRY.md](GEOMETRY.md). |
| HEVC / 10-bit H.264 / HDR / image-sequence **delivery** | A supported [output profile](#delivery-output-profiles). |

## Core Pitfalls

- Treating a clean `xmllint` run as proof of renderability — DTD validity is
  well-formedness only; the render gate is separate.
- Authoring a non-Rec.709 project, a non-square pixel aspect, or an alpha-carrying
  source and expecting a render.
- Expecting HDR delivery; Bladeworks only tone-maps HDR *in* to Rec.709 SDR.
- Selecting a codec or container outside the delivery profiles and expecting a
  substitution.
- Forgetting that `--alpha` requires a `.mov` container.
- Pointing `media-rep/@src` at a missing or unreadable file.
- Treating a `<title>` as text Bladeworks will lay out — it composites a raster you
  supply.
- Using one format for unrelated source and sequence contexts, or reordering
  children for readability without re-validating the DTD.
