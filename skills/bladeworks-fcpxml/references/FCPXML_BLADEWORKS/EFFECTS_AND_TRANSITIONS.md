# Bladeworks Effects, Masks, and Transitions

This page is how to author FCPXML filters, masks, and transitions for the
Bladeworks renderer (`bladeworks render`). Storyline adjacency and transition
timing are in [TIMELINE.md](TIMELINE.md#transitions); the paired audio crossfade
is in [AUDIO.md](AUDIO.md).

Bladeworks renders a subset of Final Cut's effect surface, and it renders it
differently from Final Cut — from the UID-matched port driven by scalar `<param>`
values, not by executing Apple's FxPlug. That makes authoring simpler: you supply
the correct UID and the scalar params and skip the opaque init blobs (see
[How Bladeworks Reads an Effect](#how-bladeworks-reads-an-effect)). A handful of
base-FCPXML effect constructs are **not part of Bladeworks** — they are called out
where they come up and collected in
[What Bladeworks Does Not Render](#what-bladeworks-does-not-render), each with the
supported way to get the same result. The certified per-op parameter detail lives
in [INVENTORY.md](INVENTORY.md#effects-and-transitions); this page teaches the
shapes and the authoring rules.

Anchoring: `tensor/effects.py`, `tensor/transitions.py`, `tensor/fx_*.py`,
`tensor/fx_mask.py`, `core/effect_parameters.py`, `core/capabilities.py`
([backend](../EFFECT_PARAMETER_SUPPORT.md)).

## Effect Resources

Filters reference `<effect>` resources by ID; the UID identifies an installed
native or Motion-template effect. Use exact known UIDs — Bladeworks matches the UID
against its capability registry (`core/capabilities.py`), and a UID it does not
recognize does not render:

```xml
<effect id="sharp" name="Sharpen" uid=".../Effects.localized/Blur.localized/Sharpen.localized/Sharpen.moef"/>
```

A UID identifies the effect; it does not by itself say which parameter keys reach
the render. That is a per-op fact — read it from [INVENTORY.md](INVENTORY.md#effects).

## Filter Placement and DTD Order

`filter-video`, `filter-audio`, and `filter-video-mask` must appear in the owning
story element's DTD-defined order relative to adjustments, sources, connected
children, and metadata (see
[GEOMETRY.md#intrinsic-adjustment-order](GEOMETRY.md#intrinsic-adjustment-order)).
A mis-ordered filter is a hard DTD failure before the renderer ever runs, so
validate the complete document against the matching DTD; visually neat ordering is
not necessarily legal ordering.

## How Bladeworks Reads an Effect

Bladeworks renders from the **matched port** driven by the scalar `<param>` values.
It does not run Apple's FxPlug, so the opaque `effectConfig` / `effectData`
initialization blobs Final Cut needs are **not required** to author an effect for
Bladeworks. For a scalar effect you author the UID plus the scalar params and
Bladeworks renders — no blob, no base64 property list to copy. This is a genuine
simplification over native authoring.

The one exception is a genuine **preserved payload** effect — the Green Screen
Keyer and opaque colour/mask archives. Those render only from a real preserved
payload carried in from an export; Bladeworks renders a bounded approximation of
them but cannot synthesize the payload from a UID alone (see
[Green Screen Keyer](#green-screen-keyer)).

Two things follow for the author:

- **A key Bladeworks does not know for an op has no effect** — it either fails the
  effect closed or is ignored, depending on the op. So author only the controls the
  inventory lists for that op; for an op that renders its fixed default, an authored
  control simply does not change the render. This is authoring advice, not a
  fidelity grade: pick the op whose controls you actually need.
- **A `filter-video/@enabled="0"` filter does not render** — it stays attached and
  inert. Author it when you want the settings preserved but the look off.

## Native Video Effects

### Color Adjustments

Color Adjustments is the native basic grade — brightness, exposure, contrast,
saturation, shadows/highlights, black point, warmth/tint. Author the UID plus the
scalar `<param>` children you want to set; the `effectConfig` init blob native
Final Cut requires is not part of the Bladeworks render path, so omit it:

```xml
<effect id="color-adjustments" name="Color Adjustments"
        uid="FxPlug:7E2022A5-202B-4EEB-A311-AC2B585D01B0"/>
...
<filter-video ref="color-adjustments" name="Color Adjustments">
  <param name="Brightness" key="2" value="200"/>
  <param name="Saturation" key="16" value="-100"/>
</filter-video>
```

Author each control with its exact key from
[INVENTORY.md](INVENTORY.md#effects); neutral controls may be omitted. The reviewed
grade controls read your static values.

### Motion-Template Filters — Gaussian, Sharpen, Vignette

Gaussian, Sharpen, and Vignette are Motion-template `.moef` resources. They take
`<param>` children only — never a `<data key="effectConfig">` child. Author the
one or two reviewed controls each exposes:

```xml
<effect id="fx-effect-blur-gaussian" name="Gaussian"
        uid=".../Effects.localized/Blur.localized/Gaussian.localized/Gaussian.moef"/>
<effect id="fx-effect-stylize-vignette" name="Vignette"
        uid=".../Effects.localized/Stylize.localized/Vignette.localized/Vignette.moef"/>

<filter-video ref="fx-effect-blur-gaussian" name="Gaussian">
  <param name="Amount" key="9999/986883370/100/986883376/2/100" value="0.1875"/>
</filter-video>
<filter-video ref="fx-effect-stylize-vignette" name="Vignette">
  <param name="Strength" key="9999/987213582/3001385021/1/200/202" value="1"/>
  <param name="Size" key="9999/987213582/3001385021/3/987213589/1" value="1.5"/>
</filter-video>
```

- **Gaussian** reads `Amount` (clamped to its tested envelope). Author that key with
  a tested value; other Gaussian controls are unproven — do not guess a key.
- **Sharpen** reads `Amount` with the UID and key above.
- **Vignette** reads `Strength` and `Size`; softness and centre have no proven key,
  so leave them out. Vignette also drops alpha, so the clip becomes opaque — a
  reproduced reference departure worth knowing before you stack it.

The exact tested keys and value envelopes are in [INVENTORY.md](INVENTORY.md#effects).

### Other Scalar Effects

Radial Blur, Droplet, Crop & Feather, and Vibrancy each read their one or two
reviewed controls the same way — author the UID plus the listed key(s). Several
stylized effects (Cartoon, Camcorder, Focus Blur, Earthquake) read some controls
and hold the rest fixed; a few (Drop Shadow, Callout, Directional Blur, Fisheye,
Vignette Mask, Kaleidoscope, Perspective Tile) render their calibrated default look
and an authored control on them does not take. And the simple built-ins —
Negative, Threshold, Mirror, Colorize, Tint, Flipped, Add Noise, Pixellate — render
only at their fixed default: **authoring any `<param>` on one of these omits the
whole effect**, so author them as a bare, parameterless `<filter-video>`. Read the
per-op control list from [INVENTORY.md](INVENTORY.md#effects) and author only what
it lists; where it lists nothing, author the bare effect for its default look.

**Generated-drill circular alpha ABI.** BladeFrame's generated FCPGym tasks may
use `SpellEffect:VisualUnshuffleAlphaCrop:v1`. This Spellshot-only numeric
effect accepts Crop & Feather's Width, Height, and Feather keys plus static
`Roundness` key `9999/988494964/100/988494966/2/353/144` (`0..1`) and
`Position` key `9999/988494964/100/988494966/1/100/101` (two normalized frame
offsets in `[-0.5, 0.5]`). Equal Width and Height, Roundness `1`, and Feather
`1` create a hard circular alpha cut. Do not use this UID for native Final Cut
delivery or represent its controls as keyframes or opaque Motion payloads.

> **Earthquake caveat.** Earthquake reads Amount and Layers but **ignores the
> authored Epicenter** — it always shakes about frame centre. Author Earthquake for
> the shake, but do not rely on epicenter placement; there is no off-centre form.

### Green Screen Keyer

The keyer is a preserved-payload effect: it renders a bounded approximation only
from a **genuine** preserved payload carried in from a Final Cut export — the
opaque `effectConfig` archive plus the per-instance `effectData` scene. Bladeworks
cannot build a working key from a UID and `<param>` children alone; a keyer emitted
that way is inert. To key fresh footage, matte it with a **numeric Range or Color
mask** ([Masks](#masks)) instead of synthesizing a keyer.

## Color Corrections

- **Color Board** renders, but per-zone saturation collapses to one global
  saturation and the exposure pucks become a bounded master curve — a look, not a
  pixel copy. Author it when a rough board grade is enough.
- **Color Wheels** read the temperature / tint / hue scalars; the proprietary wheel
  channel data is not read. Author the scalars.

For a precise grade, prefer **Color Adjustments** — its reviewed controls read
exactly.

## Masks

Wrap any supported effect in an explicit **numeric** mask to composite it through an
inside/outside matte. The native nesting is the authoring shape: the mask geometry
first, then the inside `filter-video`:

```xml
<filter-video-mask>
  <mask-shape name="Shape Mask 1" blendMode="add">
    <param name="Radius" key="160" value="60 45"/>
    <param name="Curvature" key="159" value="1"/>
    <param name="Feather" key="102" value="5"/>
    <param name="Transforms" key="200"><param name="Position" key="201" value="0 0"/></param>
  </mask-shape>
  <filter-video ref="sharp" name="Sharpen">
    <param name="Amount" key="9999/986883553/100/986883554/2/100" value="1"/>
  </filter-video>
</filter-video-mask>
```

You can author these numeric mask forms (`tensor/fx_mask.py`):

- **Shape** mask — radius / position / rotation / curvature / feather / opacity /
  falloff superellipse. This is the **one place keyframed geometry is honoured**:
  animate the matte shape with `<keyframeAnimation>` on any of those channels the
  same way geometry channels animate
  ([GEOMETRY.md#transform-keyframes](GEOMETRY.md#transform-keyframes)).
- **Draw** mask — a convex polygon.
- **Color** mask — an RGB colour key.
- **Range** mask — a luma keyer.
- Matte blend modes: `add`, `subtract`, `multiply`, plus group invert.

Author the exact `mask-shape` names and keys shown, and render a frame to confirm
the matte before relying on it. The numeric Spellshot color-range ABI renders; an
Apple opaque colour-isolation archive is preserved, not decoded.

## Transitions

Author a transition as a `<transition>` between two adjacent story elements. It
references the video filter, and — when both participants carry audio — the paired
audio crossfade filter. The immediately adjacent elements are the participants; the
transition never searches descendants or crosses a gap
([TIMELINE.md#transitions](TIMELINE.md#transitions)):

```xml
<transition name="Cross Dissolve" offset="75/30s" duration="30/30s">
  <filter-video ref="fx-transition-cross-dissolve" name="Cross Dissolve">
    <param name="Look" key="1" value="11 (Video)"/>
    <param name="Amount" key="2" value="50"/>
  </filter-video>
  <filter-audio ref="fx-transition-audio-crossfade" name="Audio Crossfade"/>
</transition>
```

Declare both effect resources, the audio one included:

```xml
<effect id="fx-transition-cross-dissolve" name="Cross Dissolve"
        uid="FxPlug:4731E73A-8DAC-4113-9A30-AE85B1761265"/>
<effect id="fx-transition-audio-crossfade" name="Audio Crossfade"
        uid="FFAudioTransition"/>
```

**Always pair the audio crossfade over audio-bearing participants.** A video-only
transition across audio clips cuts the audio hard where Final Cut would have
crossfaded it. Bladeworks's video handler renders the visual transition; the audio
crossfade is carried out by the separate PyAV audio backend ([AUDIO.md](AUDIO.md)).

### What you can author per transition family

- **Cross Dissolve** — a linear-light premultiplied dissolve; author it plainly.
- **Fade to Color / Black / White** — the colour comes from the `Color` key
  (default black). Author a well-formed colour value; a malformed or short one is
  refused rather than silently defaulting.
- **Wipe** — 4 directions via key `13`. Author an in-set integer direction.
- **Slide / Push** — 4 directions (key `4`) × mode Slide/Push (key `5`), with
  resolution-scaled motion blur. This is a **calibrated approximation** of Final
  Cut (luma SSIM 0.80–0.91), not a pixel copy — fine for the move, not for exact
  parity.
- **Motion "cohort" transitions** — Bloom, Flash, Lens Flare, Deco, Center, Clock,
  Page Curl, Swap, Rotate, Swing, Switch, Arrows, Curtains, Veil, Divide, Spin,
  Clothesline, Flip, Scale, Multi-flip, Pinwheel, Reflection, Fall, Black Hole,
  Cross Zoom, and more, plus the temporal-kernel family (Earthquake, Flashback,
  Drop In, Smear, Directional Blur, Gaussian, Radial). Most of these render their
  **default motion** — author one **for its kind** and expect the calibrated
  default look; fine parameter control is not available unless the inventory marks
  the control calibrated.
- **360° transitions** — Bloom, Circle Wipe, Divide, Gaussian Blur, Push, Reveal
  Wipe, Slide, Wipe.

**Authoring the enum values.** For Motion-template transitions, author enum values
as **bare numeric tags** (Final Cut supplies the label on export). For FxPlug
transitions, author the **label with the value** exactly as recorded —
`value="11 (Video)"`. The full per-row parameter surface is in
[INVENTORY.md](INVENTORY.md#transitions).

## What Bladeworks Does Not Render

These base-FCPXML effect and transition constructs are outside Bladeworks; each is
refused loudly at plan time rather than rendering wrong. Author the supported
alternative instead.

| Not rendered | Author instead |
| --- | --- |
| Color Curves / Hue-Saturation Curves authored curve data | A **Color Adjustments** or **Color Wheels** grade with scalar controls. |
| Corner mask, Auto mask, Magnetic mask, tracked masks, ML masks | A numeric **Shape / Draw / Color / Range** mask ([Masks](#masks)). |
| A non-convex polygon or unbounded-numeric mask | A convex Draw mask, or bounded numeric mask values. |
| Animated effect parameters (any op) | Keep effect params static; animate only the **mask-shape** matte geometry. |
| A keyer synthesized from a UID alone (no genuine payload) | A numeric **Range** or **Color** mask; or preserve a real keyer payload. |
| A transition resolving to a **native mode** or carrying an unported **prefilter** | A supported cohort / 360° / FxPlug transition of the same kind. |
| An **unadmitted** or `unsupported` transition (e.g. `spell_*` Vulkan, Squares, Cross Blur, Circle) | A supported transition; the unsupported ones become an explicit hard cut with a finding, never a silent swap. |
| A transition **missing a participant** (one side has no video) | Give both sides a real video participant before adding the transition. |

## Pitfalls

- Authoring keyframed effect parameters (outside the mask-shape matte) — refused;
  keep effect params static.
- Authoring controls on a default-only op and expecting them to take — you get the
  calibrated default, and on a simple built-in an authored param omits the whole
  effect.
- Copying an opaque `effectConfig` / `effectData` blob expecting it to drive the
  render — Bladeworks renders from the port and scalar params, not the blob.
- Trying to build a keyer, colour-mask, or tracked-mask from a UID — inert or
  refused; those need a genuine preserved payload.
- Relying on Earthquake's Epicenter — it always shakes about centre.
- Dropping the paired `FFAudioTransition` over audio clips — the audio hard-cuts.
- Expecting Slide/Push or a cohort transition to be a pixel copy of Final Cut —
  they are calibrated approximations of the move.
