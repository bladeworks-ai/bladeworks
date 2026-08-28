# Bladeworks Geometry and Compositing

This page is how to author FCPXML geometry — coordinate systems, spatial
adjustments, animation, nesting, and compositing — for the Bladeworks renderer
(`fcpxml render`). Timing is in [TIMELINE.md](TIMELINE.md).

Bladeworks renders a subset of Final Cut's FCPXML. Almost all geometry authors
exactly as it does natively, so this page teaches the same expressions the native
reference does. A few base-FCPXML constructs are **not part of Bladeworks** — they
are called out where they come up, and collected in
[What Bladeworks Does Not Render](#what-bladeworks-does-not-render) — with the
supported way to get the same result. Two document-wide requirements: the project
colour space must be **Rec.709 SDR**, and formats must use **square pixels**.

## Geometry Is Not One Coordinate System

Identify the owning element before interpreting a number. Media transform
position, crop edges, Motion-template controls, and raster dimensions use
different units.

### Source Raster

An asset's format defines its source width, height, cadence, pixel aspect, and
colour context. Source raster is not automatically the project raster.

### Project Raster

The project sequence format defines the output canvas. Media transform position
uses project-height-relative units for both axes.

### Compound-Local Raster

Every reusable `<media><sequence>` owns a local raster. Its descendants compose
there first; a parent `ref-clip` then places the completed local surface in the
parent raster. Author a reusable composition in its own sequence format and place
its `ref-clip` in the parent — see [Nested Composition](#nested-composition).

## Coordinate and Unit Reference

| Target | Unit | How to write it |
| --- | --- | --- |
| Format `width`, `height` | raster pixels | Read from the owning format. |
| `adjust-transform/@position` | percent of project height, both axes | `xmlX = rightPixels / projectHeight * 100`; `xmlY = -downPixels / projectHeight * 100`. |
| `adjust-transform/@scale` | multiplier | `1 1` is identity; keep axes equal unless distortion is intended. |
| `adjust-transform/@rotation` | degrees | Positive rotates counter-clockwise around the anchor. |
| `adjust-transform/@anchor` | transform-local position units | Animate in local parameter time. |
| `crop-rect` / `trim-rect` edges | percent-like frame-height family | Use values around `0..100`, not pixels. Account for aspect ratio on horizontal conversion. |
| `adjust-corners` coordinates | normalized corner displacement | See [Corner Pinning](#corner-pinning). |

Positive transform X moves right; positive transform Y moves **up**, opposite the
usual screen direction.

## Intrinsic Adjustment Order

The DTD declares each story element's children as a strict sequence, so order is a
hard failure, not a warning. `asset-clip` carries the fullest set:

```text
note?
conform-rate?          timing params
timeMap?
object-tracker?        video intrinsics, in this order
adjust-crop?
adjust-corners?
adjust-conform?
adjust-transform?
adjust-blend?
adjust-volume?         audio intrinsics
adjust-panner?
anchored children      connected clips, titles, captions
markers
audio-channel-source*
video filters
filter-audio*
metadata?
```

`adjust-corners` sits between `adjust-crop` and `adjust-conform`; appending it
after `adjust-transform` is the common mistake, and the DTD rejects it. The order
holds for `ref-clip`, `clip`, and `sync-clip`, but the source slot is
element-specific — `asset-clip`/`clip` use `audio-channel-source*`, `ref-clip`
uses `audio-role-source*`, `sync-clip` uses `sync-source*`, and `mc-clip` puts
`mc-source*` **before** its anchored children. `title`, `video`, `audio`, `gap`,
and `caption` each have a narrower model — do not carry the `asset-clip` list onto
them.

## Conform

Conform decides how a source of one shape fills an output of another. Omitting
`<adjust-conform>` means `fit` (the DTD default); write it explicitly for `fill`
or `none`. Give the story element a `format` so conform knows the source shape — a
video `asset-clip` with no `format` can conform Fill as if it were Fit.

```xml
<adjust-conform type="fit"/>    <!-- scale to fit inside the frame; letterbox/pillarbox may remain -->
<adjust-conform type="fill"/>   <!-- scale to cover the frame; overflow is clipped -->
<adjust-conform type="none"/>   <!-- place at native size, centered; overscan clipped -->
```

All three render. After `fill`, use `position` to choose which off-frame region
stays visible — derive the legal window from the conformed display size:

```text
xBias = (renderedWidth  - projectWidth)  / (2 * projectHeight) * 100
yBias = (renderedHeight - projectHeight) / (2 * projectHeight) * 100
```

A 1280×720 source filling a 720×1280 project has `xBias = 60.763889`; a 720×1280
source filling 1280×720 has `yBias = 108.024691`. Positive X keeps the left
region, negative X the right; negative Y keeps the top, positive Y the bottom.

**Frame-rate conform.** A passive `<conform-rate>` (cadence bookkeeping,
`scaleEnabled="0"`) is fine. An *active* `conform-rate` (`scaleEnabled="1"`, real
rate retiming) is not part of Bladeworks — to change a clip's rate, retime it with
a `<timeMap>` ([TIMELINE.md](TIMELINE.md#timemap)) or author the source at the
project cadence.

## Crop, Spatial Trim, and Pan / Ken Burns

These are three different operations. The `mode` and the rect child must agree —
`mode="trim"` needs a `<trim-rect>`, `mode="crop"` a `<crop-rect>`, `mode="pan"`
`<pan-rect>` children; a mismatched child silently fails to apply.

**Crop** selects a region and enlarges it to fill (a tighter shot):

```xml
<adjust-crop mode="crop"><crop-rect left="0" top="12" right="0" bottom="12"/></adjust-crop>
```

**Trim** removes edges without enlarging the survivor (panels, split screens):

```xml
<adjust-crop mode="trim"><trim-rect left="0" top="25" right="0" bottom="25"/></adjust-crop>
```

Two half-height panels share the trim and use positions `0 25` / `0 -25`. A value
of `480` means 480 percent of project height, not pixels.

**Pan / Ken Burns** interpolates between two viewports. Author it only for a
still-image `<video>` whose source and project share an aspect ratio, with two
ordered `<pan-rect>` children whose viewports keep that aspect:

```xml
<adjust-crop mode="pan">
  <pan-rect left="0" top="0" right="0" bottom="0"/>
  <pan-rect left="88.888889" top="50" right="0" bottom="0"/>
</adjust-crop>
```

Each edge is an inward distance as a percentage of the **original source height**
(`left = viewportLeft/sourceHeight*100`, `right =
(sourceWidth-viewportRight)/sourceHeight*100`, etc.). Horizontal values may exceed
`100`. The camera easing is supplied automatically; do not add keyframes.

## Transform

Move, scale, and rotate a clip with `<adjust-transform>`:

```xml
<adjust-transform position="12 -6" scale="0.8 0.8" rotation="-5" anchor="0 0"/>
```

- **Position** is project-height-relative (see the unit table).
- **Scale** is multiplicative and may be non-uniform. A negative scale mirrors
  that axis — author the explicit mirror when you intend a flip.
- **Rotation** is degrees around the anchor. Source display-rotation metadata is
  honoured, so validate against the displayed source.
- **Anchor** moves the pivot for scale and rotation; it is not a second position.

### Skewing or distorting a clip

`<adjust-transform>` has no shear axis — do not add a `shear`/`skew` param under
it; it is not part of Bladeworks and rejects at plan time. To skew or apply a
perspective distortion, author the four-corner Distort ([Corner
Pinning](#corner-pinning)) and move the corners.

### Transform Keyframes

Animate a channel with a `<param>` child holding a `<keyframeAnimation>`. An
animated channel is owned by its keyframes — omit the matching static attribute
and `param/@value` for that channel; leave other channels static:

```xml
<adjust-transform position="0 0">
  <param name="anchor">
    <keyframeAnimation>
      <keyframe time="0s" value="-8 0" curve="linear"/>
      <keyframe time="2s" value="8 0" curve="linear"/>
    </keyframeAnimation>
  </param>
</adjust-transform>
```

- Keyframe `time` is in the story element's **parameter-time** domain — the same
  coordinate as the element's `start`, not clip-relative. For motion beginning
  `T` seconds into a clip, use `start + T`. Parameter time continues forward
  through source freeze and reverse.
- Values hold at the first/last keyframe outside the animated span; do not add
  redundant boundary keyframes.
- Use `start + duration - one frame` when the terminal value must land on the
  final visible frame.
- Author `curve="linear"` or `curve="smooth"` — both render (smooth is a
  per-component monotone curve); `ease-in`/`ease-out` interpolation also renders.
- Do **not** author the `interp` attribute on a transform `<keyframe>` — Final Cut
  strips it and the exporter omits it, so it only causes divergence. (`interp` on
  a `<timept>` in a `<timeMap>` is a different, legal attribute.)

## Corner Pinning

Corner pinning is first-class Bladeworks geometry — a full four-corner projective
pin, static or per-corner animated. The DTD spells the lower corners `botRight` /
`botLeft`:

```xml
<adjust-corners enabled="1"
                topLeft="-0.08 0.04" topRight="0.06 0.02"
                botRight="0.04 -0.06" botLeft="-0.05 -0.03"/>
```

Animate a corner the same way any channel is animated:

```xml
<adjust-corners enabled="1">
  <param name="topLeft">
    <keyframeAnimation>
      <keyframe time="0s" value="0 0" curve="linear"/>
      <keyframe time="2s" value="-0.1 0.08" curve="linear"/>
    </keyframeAnimation>
  </param>
</adjust-corners>
```

## Opacity and Blend

Set a steady opacity with the scalar `amount`, or animate/edge-fade it through a
`<param name="amount">`:

```xml
<adjust-blend amount="0.75">
  <param name="amount">
    <fadeIn type="linear" duration="15/30s"/>
    <fadeOut type="linear" duration="15/30s"/>
  </param>
</adjust-blend>
```

`amount` runs from transparent `0` to opaque `1`. A fade may coexist with a scalar
`amount`; fade curve types `linear`, `easeIn`, `easeInOut` render.

**Blend modes.** Author a `blendMode` from the supported set: Normal, Behind, the
separable RGB modes (add, subtract, darken, lighten, multiply, screen, overlay,
soft-light, hard-light, difference, exclusion, color-burn, color-dodge, divide,
linear-light, pin-light, hard-mix), and the stencil/silhouette mattes. The
cross-channel modes **Hue, Saturation, Color, and Luminosity are not part of
Bladeworks** — choose a supported mode instead. (RGB and luma-matte modes render
as calibrated approximations of Final Cut's; Normal/Behind and the alpha mattes
are exact.)

## Nested Composition

A container completes internally, then its own transform/opacity/clip apply to
that finished surface. Author child motion first, then the outer clip:

```xml
<clip offset="0s" start="0s" duration="4s">
  <adjust-transform position="15 -6" scale="0.82 1.12" rotation="-9" anchor="7 -4"/>
  <adjust-blend amount="0.78"/>
  <asset-clip ref="camera" offset="0s" start="0s" duration="4s">
    <adjust-transform position="-11 8" scale="0.72 0.66" rotation="14" anchor="-8 5"/>
  </asset-clip>
</clip>
```

The inner transform applies first; the outer clip then transforms and fades the
completed composition. A **connected** child (a `lane` sibling) is temporally
anchored but spatially independent — give it its own transform if it must follow
the parent visually. An internal child is clipped by its container; a connected
sibling is not.

## Canonical Layout Recipes

- Full bleed across unlike aspect ratios: `fill`, then a uniform position bias.
- Tighter full-frame shot: `crop` with measured edge distances.
- Split-screen panels: `trim`, then transform each panel.
- Reusable portrait composition in landscape: author the portrait compound in its
  own sequence format, then place its `ref-clip` in the parent.

Panel constants for sources matching the project aspect ratio:

| Layout | Spatial Trim per clip | Positions |
| --- | --- | --- |
| Two columns in 16:9 | `left="44.444444" right="44.444444"` | X = `-44.444444`, `44.444444` |
| Three columns in 16:9 | `left="59.259259" right="59.259259"` | X = `-59.259259`, `0`, `59.259259` |
| Two rows in 9:16 | `top="25" bottom="25"` | Y = `25`, `-25` |
| Three rows in 9:16 | `top="33.333333" bottom="33.333333"` | Y = `33.333333`, `0`, `-33.333333` |

Use uniform scale so panels do not distort.

## What Bladeworks Does Not Render

These base-FCPXML geometry constructs are outside Bladeworks; each rejects at plan
time rather than rendering wrong. Author the supported alternative instead.

| Not rendered | Author instead |
| --- | --- |
| `shear` / `skew` param under `adjust-transform` | Four-corner Distort ([Corner Pinning](#corner-pinning)). |
| A scale animation that crosses through zero | Keep the sign stable; use the explicit mirror for a flip. |
| Active `conform-rate` (`scaleEnabled="1"`) | Retime with `<timeMap>` ([TIMELINE.md](TIMELINE.md#timemap)). |
| Hue / Saturation / Color / Luminosity blend modes | A supported RGB or matte blend mode. |
| Non-square pixel aspect (`paspH`/`paspV` ≠ 1) | Author square-pixel formats. |
| Non-Rec.709 / wide-gamut / HDR project | Author a Rec.709 SDR project (HDR *sources* are tone-mapped in on decode). |

## Pitfalls

- Pasting pixel values into percent-like crop or transform fields.
- Treating Fill, Crop, and spatial Trim as synonyms.
- Writing a `mode`/rect mismatch (`mode="trim"` with a `<crop-rect>`) — the crop
  silently does not apply.
- Interpreting compound descendants in the parent raster instead of their local
  raster.
- Expecting a connected sibling to inherit its anchor's transform — give it its
  own.
- Authoring a `shear`/`skew` or a Hue/Saturation/Color/Luminosity blend and
  expecting a render — see [What Bladeworks Does Not Render](#what-bladeworks-does-not-render).
