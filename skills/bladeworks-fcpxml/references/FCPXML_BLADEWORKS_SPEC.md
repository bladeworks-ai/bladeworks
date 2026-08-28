# Bladeworks FCPXML Authoring Reference

This is the stable entrypoint for authoring FCPXML that will be **rendered by
Bladeworks FCPXML** - Spellshot's portable PyTorch tensor renderer (the `fcpxml`
CLI, `src/bladeworks/`). It describes the document you hand to
`fcpxml render`: how the timeline is structured, the expressions that place
and transform media, the compositing and colour model the renderer commits to,
and how to validate a document before you render it.

The reference is self-contained. An agent authoring for Bladeworks should not
need any other document open. FCPXML is Apple's XML interchange for a Final Cut
timeline — resources declared once at the top, then a `<library>` holding an
`<event>`, a `<project>`, and the `<sequence><spine>` that lays media out in
time. Bladeworks reads that same grammar.

Bladeworks renders a **subset** of Final Cut's FCPXML. Almost everything authors
exactly as it does natively, so each sub-page teaches the same expressions the
native reference does — and, where a base-FCPXML construct is *not part of*
Bladeworks, states so plainly and names the supported way to get the same
result. There is no guessing and no silent substitution: a construct Bladeworks
cannot render is refused **loudly** at plan time (`TensorRenderUnsupported`), or,
for an effect, omitted and reported. It is never quietly degraded to an
identity, a black frame, or a hard cut you did not ask for.

## Architecture Map

```text
FCPXML document  (rendered by `fcpxml render`)
+- resources
|  +- format                 raster, cadence, colour space (Rec.709 SDR, square pixels)
|  +- asset                  external media, gated on decoded pixel format + colour tags
|  +- media                  reusable compound or multicam sequence (recursive scope)
|  `- effect                 title, filter, or transition resource
`- library
   `- event
      `- project
         `- sequence
            `- spine         primary storyline and connected story elements

Bladeworks authoring surface
+- document, source media, delivery profiles       -> FCPXML_BLADEWORKS/CORE.md
+- clocks, placement, scopes, transitions, retime  -> FCPXML_BLADEWORKS/TIMELINE.md
+- conform, crop, transforms, compositing, blend    -> FCPXML_BLADEWORKS/GEOMETRY.md
+- roles, routing, gain, fades, panning             -> FCPXML_BLADEWORKS/AUDIO.md
+- runtime-rasterized titles, captions, Custom Solid -> FCPXML_BLADEWORKS/TITLES_AND_CAPTIONS.md
+- effect / transition / mask expressions           -> FCPXML_BLADEWORKS/EFFECTS_AND_TRANSITIONS.md
+- exhaustive capability tables                      -> FCPXML_BLADEWORKS/INVENTORY.md
`- complete copyable documents                       -> FCPXML_BLADEWORKS/EXAMPLES.md
```

## The Authoring Model

Bladeworks renders **one** `<project><sequence>` per invocation, selected by
name or UID:

```bash
fcpxml render <doc>.fcpxml --project NAME_OR_UID
```

A multi-project or browser-clip document is legal; only the selected sequence
produces pixels. Author your timeline as the native reference teaches — an exact
rational clock, a primary spine with connected lanes, reusable compounds as
their own recursive scopes — and the renderer honours it. The render model
itself commits to a few things worth stating up front, because they shape what
you author:

- **Video and audio are separate backends.** `tensor/` renders video only. Audio
  is resolved by an independent in-process PyAV graph and muxed beside the video
  as stereo AAC. Roles, gain, fades, pan, and routing all live on the audio side
  — see [AUDIO.md](FCPXML_BLADEWORKS/AUDIO.md).
- **Compositing is premultiplied and linear-light.** Layers are composited with
  premultiplied alpha in linear light, then delivered as Rec.709 SDR. This is why
  a container completes internally before its outer transform/opacity/clip apply,
  and why blend results are calibrated approximations of Final Cut rather than
  bit copies.
- **Supported text is rasterized automatically.** The public executor lays out
  supported titles and captions with FreeType/HarfBuzz and generates temporary
  project-space RGBA images before building the tensor plan. Custom Solid uses
  the same runtime-raster path. Unsupported templates or unresolved fonts are
  reported rather than silently replaced — see
  [TITLES_AND_CAPTIONS.md](FCPXML_BLADEWORKS/TITLES_AND_CAPTIONS.md).

## Global Bladeworks Authoring Rules

These hold across every document and every sub-page. They are authoring
requirements, not preferences — a document that breaks one does not render.

1. **Rec.709 SDR only.** Bladeworks grades and delivers in Rec.709 SDR.
   Non-Rec.709 grading and HDR / wide-gamut *delivery* reject; HDR *sources*
   (HLG/PQ) are accepted and tone-mapped to Rec.709 SDR on decode.
2. **Square pixels only.** Every format the renderer touches must have a square
   pixel aspect (`paspH`/`paspV` = 1). Author square-pixel formats.
3. **Compositing is premultiplied linear-light.** Author transparency and blend
   expecting premultiplied-alpha compositing in linear light; internal container
   children compose before the container's outer transform, opacity, and clip.
4. **Audio is a separate stereo/AAC backend.** Video carries no audio; author
   roles and audio adjustments for the independent PyAV graph, which delivers
   mono or stereo (5.1 / surround reject).
5. **Supported titles and captions are runtime-rasterized.** Author their text
   and styles in FCPXML; `fcpxml render` resolves fonts and rasterizes them
   automatically. Use only the templates and controls certified in the registry.
6. **One `<project>` renders.** Select it with `--project`; a document may hold
   many, but only the selected sequence produces pixels.
7. **A DTD-valid `.fcpxml` or `.fcpxmld` bundle.** Bladeworks renders a plain
   `.fcpxml` document or a `.fcpxmld` directory whose root contains
   `Info.fcpxml`. Bundle-relative media paths resolve from inside the bundle, so
   pass the bundle directory directly instead of extracting its XML.

The standard FCPXML timing and composition invariants all hold on top of these —
exact rational times; distinct project / containing-story / source-timecode /
parameter clocks; recursive connected timing
(`child_absolute = parent_absolute + child.offset − parent.start`); each reusable
`<media><sequence>` owning its own format; connected children anchored in time
but not inheriting the parent transform; transitions binding only immediately
adjacent story elements; keyframes advancing in local parameter time. They are
detailed in [TIMELINE.md](FCPXML_BLADEWORKS/TIMELINE.md) and
[GEOMETRY.md](FCPXML_BLADEWORKS/GEOMETRY.md).

## FCPXML Structural Requirements

The Apple DTD defines the full grammar. At minimum:

- The root is `<fcpxml version="1.14">`. Bladeworks targets grammar 1.14.
- `<resources>` precedes the document body.
- Resource IDs are unique XML IDs and every `ref` resolves to one; an unresolved
  `ref` rejects at compile — it never renders a placeholder.
- Elements and children follow DTD content models and ordering.
- Times use exact FCPXML rational syntax, such as `0s`, `1/2s`, or `1001/30000s`.
- Referenced media must exist and be decodable; matched effects and transitions
  must resolve to a construct Bladeworks renders.

## Documentation Map

Each page is a self-contained authoring reference for its slice of the document.
Start at the one that owns your construct.

- [Core document, media, and delivery](FCPXML_BLADEWORKS/CORE.md): the document
  scaffold, resource identity, formats, assets, accepted source pixel formats and
  colour tags, and export profiles.
- [Timeline and timing](FCPXML_BLADEWORKS/TIMELINE.md): clocks, spine and lanes,
  compounds and multicam as recursive scopes, transitions as composited sides, and
  the retime surface (forward / reverse / freeze / piecewise).
- [Geometry and compositing](FCPXML_BLADEWORKS/GEOMETRY.md): conform, transform,
  crop / Ken Burns, corner pinning, opacity, blend modes, nested composition, and
  the coordinate units.
- [Audio](FCPXML_BLADEWORKS/AUDIO.md): the separate PyAV audio backend — roles,
  gain, fades, mutes, pan, routing, retime, multicam; mono / stereo only.
- [Titles, text, and captions](FCPXML_BLADEWORKS/TITLES_AND_CAPTIONS.md): runtime
  text rasterization, supported styles, Custom Solid, and template boundaries.
- [Effects, masks, and transitions](FCPXML_BLADEWORKS/EFFECTS_AND_TRANSITIONS.md):
  the filter / transition / mask XML an agent authors, per-parameter fidelity, and
  what each construct does at render time.
- [Capability inventory](FCPXML_BLADEWORKS/INVENTORY.md): the exhaustive certified
  table of every effect, transition, mask, title, and story element, mirrored from
  the machine-readable registry.
- [Complete examples](FCPXML_BLADEWORKS/EXAMPLES.md): copyable documents that
  render exactly (or as noted) on Bladeworks.

## The Exhaustive Reference

Prose orients you; two machine-readable artifacts are the exhaustive, binding
record of what Bladeworks renders.

- **`FCPXML_RENDER_CAPABILITIES.yaml`** (this directory) is the registry the
  compiler actually loads and enforces at render time
  (`core/capabilities.py::CapabilityRegistry`). Each row names an effect,
  transition, mask, or title, the `uid`/`aliases` that match it, the `handler`
  that renders it, its calibrated `parameters`, and the exact rule for what
  happens when it cannot be rendered. Structural constructs (geometry, time,
  compositing, scopes, media, colour) are gated in
  [`tensor/support.py`](../../../src/bladeworks/tensor/support.py).
- **[INVENTORY.md](FCPXML_BLADEWORKS/INVENTORY.md)** is the human-readable mirror
  of that registry — the full table with fidelity notes.

When this prose and those sources disagree, **the sources win.** Treat any
divergence as a documentation bug.

## Grammar Validation

The DTD for grammar 1.14 is checked in at
[`FCPXML_BLADEWORKS/FCPXMLv1_14.dtd`](FCPXML_BLADEWORKS/FCPXMLv1_14.dtd) —
Apple's own DTD, copied verbatim so an agent can validate without a Final Cut
install:

```bash
xmllint --noout --dtdvalid FCPXML_BLADEWORKS/FCPXMLv1_14.dtd doc.fcpxml
```

**DTD validity proves grammar, not renderability.** It confirms tags,
required attributes, and child order are legal. It does *not* confirm the
document is inside the surface Bladeworks renders — a perfectly DTD-valid
document can still reject at plan time. Grammar is the first gate; the capability
registry and `tensor/support.py` are the one that decides whether pixels come
out. Prove the last mile the way you would for Final Cut: render a
one-question microfixture (two participants with a gap for transition adjacency,
unequal left/right channels for audio routing, an asymmetric corner pin for
coordinate interpretation) and read the frames.

## Minimal Project That Renders

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.14">
  <resources>
    <format id="r1" name="FFVideoFormat1080p30"
            frameDuration="1/30s" width="1920" height="1080" colorSpace="1-1-1 (Rec. 709)"/>
    <asset id="r2" name="intro.mp4" uid="EXAMPLE-CLIP-UID"
           start="0s" duration="240/30s" hasVideo="1" videoSources="1"
           hasAudio="1" audioSources="1" audioChannels="2" audioRate="48000"
           format="r1">
      <media-rep kind="original-media" src="file:///tmp/intro.mp4"/>
    </asset>
  </resources>
  <library>
    <event name="Example Event">
      <project name="Minimal Project">
        <sequence format="r1" duration="3s" tcStart="0s" tcFormat="NDF"
                  audioLayout="stereo" audioRate="48k">
          <spine>
            <asset-clip ref="r2" name="intro.mp4" offset="0s"
                        start="0s" duration="3s"/>
          </spine>
        </sequence>
      </project>
    </event>
  </library>
</fcpxml>
```

Render it with:

```bash
fcpxml render project.fcpxml --project "Minimal Project"
```

The canonical copy and broader cases live in
[EXAMPLES.md](FCPXML_BLADEWORKS/EXAMPLES.md).

## Common Mistakes

1. Stopping at DTD validity. Grammar is necessary, not sufficient — check the
   construct against the capability registry and `tensor/support.py` before you
   trust it to render.
2. Authoring an HDR, wide-gamut, or non-Rec.709 project. Deliver Rec.709 SDR; let
   HDR *sources* tone-map in on decode.
3. Authoring a non-square pixel aspect. Use square-pixel formats.
4. Using an unsupported Motion title template or a font unavailable on the
   render machine. Author a certified template and an installed font, then let
   `fcpxml render` rasterize it automatically.
5. Authoring `audioLayout="surround"` or 5.1 output. Only mono and stereo deliver.
6. Reaching for a base-FCPXML construct a sub-page marks as not-rendered instead
   of the supported alternative it names beside it.
7. Expecting a blend or graded look to be a pixel copy of Final Cut. Premultiplied
   linear-light compositing gives calibrated approximations — verify a frame.
8. Extracting `Info.fcpxml` from a `.fcpxmld` bundle before rendering. Pass the
   bundle directory directly so bundle-relative media paths keep working.

Use the [documentation map](#documentation-map) to continue to the page that owns
your construct.
