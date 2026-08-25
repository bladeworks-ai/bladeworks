# Bladeworks Titles, Text, and Captions

This page is how to author title resources, editable text, template controls,
native captions, and the Custom Solid generator for the Bladeworks renderer
(`bladeworks render`). Geometry and placement are in [GEOMETRY.md](GEOMETRY.md);
document, resource, and media structure is in [CORE.md](CORE.md). Validated
title parameter keys are in [INVENTORY.md](INVENTORY.md).

The one thing to understand before authoring anything on this page:
**the public Bladeworks executor rasterizes supported text automatically.** It
reads each supported title or caption's text and styling, lays out glyphs with
FreeType/HarfBuzz, writes a temporary project-space RGBA image, and composites
that image at the element's geometry and time. You author valid `<title>` /
`<caption>` / `<generator>` XML with its full grammar, geometry, and timing;
`bladeworks render` performs the raster preparation internally. A few base-FCPXML
constructs are not part of Bladeworks; they are called out where they come up and collected in
[What Bladeworks Does Not Render](#what-bladeworks-does-not-render), each with
the supported way to get the same result.

Backend anchoring: `executor.py::_prepare_runtime_rasters` (CLI
orchestration), `core/text.py` (text and generator rasterization), `tensor/plan.py`
(raster placement), `core/capabilities.py` (template resolution),
`core/compositor.py` (compositing),
[`tensor/support.py`](../../../../src/bladeworks/tensor/support.py)
(the construct gate).

## Bladeworks Rasterizes, Then Composites

The division of labour is fixed:

1. The public executor rasterizes each title / caption / Custom-Solid clip to a
   straight-alpha RGBA PNG (`core/text.py`). **All text layout — font, size,
   colour, stroke, shadow, baseline, and the published Position / Size / Scale /
   Tracking controls — is baked into that PNG** at a 1920×1080 Motion design
   space, scaled by `project_height / 1080`.
2. The executor hands the resulting `clip → raster` mapping to the internal
   tensor plan (`tensor/plan.py`). This mapping is not a public CLI input.
3. Bladeworks places each raster as an ordinary layer. Everything downstream —
   conform, geometry, opacity, blend, effects — runs the **same code path as a
   video leaf**.

The practical consequence for authoring: **you decide the look through the XML,
and Bladeworks' rasterizer produces it before the tensor compositor places the
finished pixels.**
Whether a rasterized Basic Title *matches* Final Cut's own rendering is a
property of the upstream rasterizer (a calibrated approximation of the template),
not of the compositing step. Everything in the grammar below is still authored in
full — it is simply read by the rasterizer, not by the pixel renderer. The one
generator Bladeworks renders is **Custom Solid**, which you author directly (see
[Custom Solid Generator](#custom-solid-generator)).

## Title Resources and Title Instances

You still author the title grammar in full so the rasterizer can read it. An
`<effect>` resource identifies a Motion title template; a `<title ref="...">`
story element places one instance and owns its text, published parameters,
styles, timing, role, and clip-level adjustments.

Title instances carry `start="3600s"`, not `start="0s"` — Final Cut's Basic Title
source-time sentinel, on the Motion-template timebase. The examples below use it
throughout.

```xml
<effect id="title-basic" name="Basic Title" uid=".../Titles.localized/Bumper:Opener.localized/Basic Title.localized/Basic Title.moti"/>
```

Bladeworks resolves the template by UID through the capability registry
(`core/capabilities.py`). **Basic Title (`title-basic`) is the one authorable
title template.** Any other Motion title template — Drifting, Standard, Lower
Thirds, the Essential families — has no render path; to place that content,
re-author it as Basic Title, or produce the design upstream and supply it as a
still image or Custom Solid (see
[What Bladeworks Does Not Render](#what-bladeworks-does-not-render)). The full UID
must match an installed template; the instance `name` is only a timeline label.

## Motion-Template UID Rules

- Copy exact UIDs from validated Final Cut resources or the inventory.
- Do not reconstruct a localized path from a display name.
- Keep the literal `.../Titles.localized/.../*.moti` shape. Never write the
  installed absolute file path, the browser package path, or a `.motntitle`
  package path as the UID.
- A known UID proves resource identity, not arbitrary parameter keys.
- A missing template can pass DTD validation but has no render path.

## Editable Text Fields

One template can expose multiple text fields. Preserve field order and any
template-specific identifiers — the rasterizer reads them positionally. Do not
assume every `<text>` node is the visible headline, or that a field count implies
a parameter contract.

## text, text-style, and text-style-def

A title or caption references style definitions by ID; the rasterizer consumes
them to lay out the PNG:

```xml
<text>
  <text-style ref="ts1">Editable text</text-style>
</text>
<text-style-def id="ts1">
  <text-style font="DejaVu Sans" fontSize="64" fontFace="Regular"
              fontColor="1 1 1 1" alignment="center"/>
</text-style-def>
```

Style IDs are local to the owning story element; do not repeat one ID across
unrelated titles. Preserve text runs when styles change within one field.

## Text Colours, Stroke, Shadow, Kerning, and Alignment

These `text-style` attributes are read by the rasterizer:

`font`, `fontSize`, `fontFace`, `fontColor`, `alignment`, `strokeColor`,
`strokeWidth`, `bold`, `kerning`, `shadowColor`, `shadowOffset`,
`shadowBlurRadius`. Captions may also carry `backgroundColor`, which the DTD
declares specifically for them.

The font family and face must exist on the machine that runs the render. An
unresolved font does not fall back to another face — the title is dropped from
the composite with a compatibility finding, so choose a font you know is
installed.

Colours use normalized RGBA components:

```text
white           -> 1 1 1 1
black           -> 0 0 0 1
red             -> 1 0 0 1
50% alpha white -> 1 1 1 0.5
```

### Basic Title stroke width must be non-positive

For Basic Title outlines, `strokeWidth` must be `<= 0`. Serialize every nonzero
outline thickness as `-abs(width)`, for example `strokeWidth="-4"`. Omit
`strokeWidth`, or use `0`, when no outline is requested.

A positive width is valid FCPXML and imports without error — which is what makes
this trap expensive. The rasterizer normalizes the outline as
`round(abs(signed_width) * project_height / 1080)` (`core/text.py`), and a
positive value produces an outline-only treatment: the outline renders but the
requested Face colour is lost.

```xml
<text-style-def id="style-1">
  <text-style font="Helvetica" fontSize="48" fontColor="1 1 1 1"
              alignment="center" strokeColor="0 0 0 1" strokeWidth="-4"/>
</text-style-def>
```

## Basic Title

Basic Title is the primary detailed recipe because its common text and published
control contract is known and calibrated.

### Published parameters

Every key below is a validated Basic Title contract; the exact keys and evidence
are the Basic Title rows in [INVENTORY.md](INVENTORY.md). The rasterizer reads
them at 1920×1080 design space:

```xml
<title ref="title-basic" name="Intro" offset="0s" start="3600s" duration="3s">
  <param name="Position" key="9999/999166631/999166633/1/100/101" value="0 -500"/>
  <param name="Flatten" key="9999/999166631/999166633/2/351" value="1"/>
  <param name="Alignment" key="9999/999166631/999166633/2/354/999169573/401" value="1 (Center)"/>
  <param name="Size" key="9999/999166631/999166633/5/999166635/3" value="54"/>
  <text><text-style ref="ts1">Intro</text-style></text>
  <text-style-def id="ts1">
    <text-style font="DejaVu Sans" fontSize="54" fontFace="Regular"
                fontColor="1 1 1 1" alignment="center"/>
  </text-style-def>
</title>
```

`Position`, `Flatten`, and `Alignment` are all expected on an authored Basic
Title; omitting any of them is a validation warning, not a silent default.

### Position

The published `Position` control uses 1920×1080 Motion design units, not media
transform percentages or project pixels. The rasterizer maps it as
`anchor_x = width/2 + posX * height/1080` and
`baseline = height/2 - posY * height/1080` (`core/text.py`), so a rough
conversion is:

```text
template_value = desired_project_pixels * 1080 / project_height
```

Baseline anchoring means an apparently centered numeric position can still land
optically shifted — render a representative frame to confirm exact placement.

### Size

Basic Title's published `Size` key is `9999/999166631/999166633/5/999166635/3`.
Mirror `text-style/@fontSize` into it — the two must agree. Do not guess a
published Size key from another template; every template has its own.

### Scale — two distinct stages

Basic Title has two scaling paths, and under Bladeworks they run in different
stages. Prefer the first:

1. A clip-level `<adjust-transform scale="...">` scales the **finished raster**
   at composite time. It uses Bladeworks's generic intrinsic transform and needs
   no Motion-template key — see [GEOMETRY.md](GEOMETRY.md#transform).
2. Basic Title's published `Scale` control, key
   `9999/999166631/999166633/1/100/105`, is read by the rasterizer and baked into
   the PNG:

   ```xml
   <param name="Scale" key="9999/999166631/999166633/1/100/105" value="0.5 0.5"/>
   ```

   Use it only when deliberately editing or preserving that template control. The
   key belongs to the Basic Title template; do not reuse it for any other title.

### Tracking

Basic Title's Tracking key is `9999/999166631/999166633/5/999166635/81/79`. A
title can instead carry per-run tracking nested under a Motion text-style param:

```xml
<param name="MotionSimpleValues" key="MotionTextStyle:SimpleValues">
  <param name="motionTextTracking" key="tracking" value="-1.9125"/>
</param>
```

Preserve that nested shape when it appears in a genuine export, but do not
synthesize it or infer a tracking key by display name. For generated documents,
prefer `text-style/@kerning`.

### Baseline and design-space coordinates

Text baseline, glyph metrics, and Motion design coordinates all affect the
visual bounds, and they are resolved in the rasterizer. Render representative
frames for exact layout rather than trusting the numeric position alone.

## Custom Solid Generator

Custom Solid is the one generator Bladeworks renders, and you author it directly
as a flat colour fill. Its raster is produced on the same runtime path as a title
(`core/text.py`), guarded on the solid-colour execution, and a flat fill
reproduces exactly. Any generator that is not Custom Solid has no render path —
author a Custom Solid, or supply the design upstream as a still image (see
[What Bladeworks Does Not Render](#what-bladeworks-does-not-render)).

Because a Custom Solid resolves to a raster, everything downstream — geometry,
opacity, blend, effects — is the ordinary raster path described in
[Downstream Geometry and Effects](#downstream-geometry-and-effects-on-rasters).

## Native Captions

`<caption>` represents subtitle / caption semantics in a caption lane or role. It
owns timing, text, and the styles the DTD allows. Under Bladeworks a caption is
rasterized automatically and composited as burned-in pixels — the visual caption
renders on the raster path, exactly like a title.

Bladeworks produces only the pixels. It does not emit caption interchange
(ITT / CEA608 / SRT sidecars); that delivery-metadata layer is outside the pixel
renderer. If you need caption interchange *and* burn-in, keep the caption
representation for the interchange tool and accept that Bladeworks renders the
burn-in.

Every native caption role must name its caption format and language:

```text
roleName?captionFormat=captionFormat.language
```

For example, `caption?captionFormat=ITT.en` means role name `caption`, ITT format,
English. Final Cut defines three caption formats: `ITT`, `CEA608`, and `SRT`. A
value such as `caption.English` is not a valid native caption role because it
omits `captionFormat`.

### Native caption placement

`<caption>` is an anchored item, not a clip item, so it cannot be a direct child
of a primary or secondary `<spine>`:

```xml
<!-- Invalid: a spine cannot contain caption directly. -->
<spine>
  <caption name="Caption" duration="2s" role="caption?captionFormat=ITT.en"/>
</spine>
```

Attach the caption to a clip item that is its natural timeline anchor:

```xml
<asset-clip ref="video" offset="0s" start="0s" duration="5s">
  <caption name="Caption" lane="1" offset="1s" start="0s" duration="2s"
           role="caption?captionFormat=ITT.en">
    <text><text-style ref="caption-style">Hello</text-style></text>
    <text-style-def id="caption-style">
      <text-style font="Helvetica"/>
    </text-style-def>
  </caption>
</asset-clip>
```

If a caption needs a storyline slot with no natural clip anchor, put a
same-duration `<gap>` in the spine and attach the caption to that gap:

```xml
<spine>
  <gap name="Caption Anchor" offset="0s" duration="2s">
    <caption name="Caption" lane="1" offset="0s" start="0s" duration="2s"
             role="caption?captionFormat=ITT.en">
      <text><text-style ref="caption-style">Hello</text-style></text>
      <text-style-def id="caption-style">
        <text-style font="Helvetica"/>
      </text-style-def>
    </caption>
  </gap>
</spine>
```

### Native caption content model

The complete allowed child order is:

```text
caption = text* text-style-def* note?
```

In plain language: zero or more `<text>` blocks, then zero or more
`<text-style-def>` declarations, then one optional `<note>`. Do not put Motion
parameters, transforms, crops, retimes, markers, filters, or metadata inside
`<caption>`. Keep durations positive and avoid overlaps the chosen caption format
cannot represent.

## Visual Captions Authored as Titles

Timed titles suit visual design and precise screen placement more than caption
semantics. Under Bladeworks each is a Basic Title instance (or an upstream still),
composited on the raster path — a normal title with title role, duration,
editable text, and styles.

Do not call title-based graphics native captions. They are visible title clips
and do not export through caption workflows.

## Choosing title Versus caption

| Need | Choose | Bladeworks result |
| --- | --- | --- |
| Accessibility or subtitle interchange | `caption` | Pixels only; the interchange sidecar is not emitted here. |
| Native caption tools downstream | `caption` | Same; keep the caption for the external tool. |
| Motion-template animation | `title` (Basic Title only) | Composited on the raster path; non-Basic templates have no render path. |
| Precise branded visual treatment | `title` or an upstream still | Composited on the raster path. |
| Both semantics and branded burn-in | Maintain both representations | Bladeworks renders the burn-in pixels. |

## Downstream Geometry and Effects on Rasters

A resolved raster reaches the same pipeline as a video leaf, so its geometry and
effect support is exactly the leaf support elsewhere in this reference: conform
fit / fill / none, static and animated `adjust-transform`, crop and pan, corner
pin, opacity and fades, the supported blend modes, and any supported effect —
subject to the same constraints. See [GEOMETRY.md](GEOMETRY.md) for placement and
[CORE.md](CORE.md) for document and format rules.

One raster-specific timing rule: a raster has no temporal content, so a retime
map on a title / caption / Custom-Solid clip is ignored — dropping the map is the
identity. Do not author a non-unit `speed` on a title or caption; instead author a
static raster at unit speed (see
[What Bladeworks Does Not Render](#what-bladeworks-does-not-render)).

## What Bladeworks Does Not Render

These constructs have no render path; author the supported alternative instead.

| Not rendered | Author instead |
| --- | --- |
| A non-Basic Motion title template (Drifting / Standard / Lower Thirds / Essential) | Re-author as Basic Title, or produce the design upstream as a normal still-image asset / Custom Solid. |
| A generator that is not Custom Solid | A Custom Solid fill, or an upstream still image. |
| A title / caption with `speed != 1` (a retimed raster) | A static raster at unit speed — the raster has no temporal content to retime. |
| A title / caption / Custom-Solid clip the runtime rasterizer cannot resolve | Use a supported template, valid styling, and an installed font; inspect the compatibility report for the exact failure. |
| A raster supplied for a clip that is not a raster | Remove the stale mapping; a raster belongs only to a title / caption / Custom Solid. |
| A runtime raster that cannot be written or read | Fix the reported render environment or font/template error; Bladeworks does not invent replacement pixels. |
| A title whose font did not resolve on the render machine | Use a font family and face installed where the render runs. |

## Pitfalls

- Expecting an unsupported template or unavailable font to fall back silently.
  The runtime rasterizer reports the incompatibility instead.
- Authoring a non-Basic Motion title template and expecting it to render — see
  [What Bladeworks Does Not Render](#what-bladeworks-does-not-render).
- Authoring a generator other than Custom Solid.
- Using a display name as a Motion-template UID.
- Guessing a published parameter key because two templates expose similarly named
  controls.
- Mixing media-transform percentages with 1920×1080 title design units.
- Serializing a positive Basic Title `strokeWidth` — the outline renders and the
  Face colour is lost.
- Authoring `speed != 1` on a title or caption.
- Treating title-based visual captions as native caption semantics, or expecting
  a `<caption>` to emit ITT / CEA608 / SRT interchange — Bladeworks renders the
  burn-in pixels only.
- Repeating style IDs across unrelated owning title elements.
- Assuming DTD-valid text will render with a missing font or an unsupported
  template.
