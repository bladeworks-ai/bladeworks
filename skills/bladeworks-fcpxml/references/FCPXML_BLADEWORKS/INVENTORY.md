# Bladeworks Capability Inventory

This is the exhaustive certified surface: every effect, transition, mask, title,
and story-element capability Bladeworks knows, with its two status axes and (for
effects/transitions) its authored-parameter tier. It mirrors the machine-readable
registry the compiler actually loads:

[`FCPXML_RENDER_CAPABILITIES.yaml`](../FCPXML_RENDER_CAPABILITIES.yaml)

The registry is the source of truth. If this table and the YAML disagree, the
YAML wins. The model behind the columns is in
[EFFECTS_AND_TRANSITIONS.md](EFFECTS_AND_TRANSITIONS.md); the structural
(non-effect) constructs are in the second table below, from
[`tensor/support.py`](../../../../src/bladeworks/tensor/support.py).

## Reading the Columns

- **Authoring** (`authoring_status`): `authorable` = synthesize fresh;
  `preserve_only` = renders a genuine payload but do not synthesize it;
  `out_of_scope` = not covered.
- **Portable status** (`portable_status`) — this is the render disposition in the
  three-outcome vocabulary the rest of this spec uses: `exact_portable` = **exact**
  (renders exactly); `calibrated_portable` = **approximate** (renders as a
  calibrated pixel- *or* semantic approximation — the semantic default of a
  parameter-blocked op still counts as supported); `unsupported` / `apple_only` =
  **reject** (omit / hard-cut / raise loudly, never a silent substitution).
- **Param tier** (effects/transitions): from the parameter-calibration evidence
  (`parameter_inventory/*.v1.json`). `calibrated` = your static controls render;
  `partial` = some static controls render (animation rejects); `default-only` =
  the calibrated default renders, your controls are omitted; `reject` = the
  standalone op is not implemented; `—` = no per-parameter evidence row (the
  effect's control support, where it has any, is documented in
  [`EFFECT_PARAMETER_SUPPORT.md`](../EFFECT_PARAMETER_SUPPORT.md)).

**Catch-all rows.** Rows whose name is a bare capability id
(`effect-fxplug-unsupported`, `effect-motion-unsupported`,
`transition-motion-unsupported`, `transition-fxplug-unsupported`) and the
`*-spell-*` / experimental Vulkan artifact rows are **generic fallbacks**: any
Motion/FxPlug effect or transition that matches no specific capability lands here
and is omitted (effect) or hard-cut (transition) with a finding. They are not
authorable named resources.

**A note on `default-only` vs. the runtime.** The param tier reflects the
*certified parameter-calibration evidence*, which is intentionally conservative;
the tensor runtime may render a `default-only` op's default look faithfully while
still refusing authored controls. Author to the tier: if it is not `calibrated`,
do not depend on your parameter values.

## Effects and Transitions

<!-- BEGIN GENERATED: FCPXML_RENDER_CAPABILITIES.yaml × parameter_inventory -->

### Effects

FCPXML `kind: video_filter`.

| Name | Authoring | Portable status | Param tier | Capability id |
| --- | --- | --- | --- | --- |
| Callout | `authorable` | `calibrated_portable` | default-only | `effect-callout-reframe` |
| Camcorder | `authorable` | `calibrated_portable` | partial | `effect-camcorder-cohort` |
| Cartoon | `authorable` | `calibrated_portable` | partial | `effect-cartoon-cohort` |
| Color Adjustments | `authorable` | `calibrated_portable` | — | `effect-color-adjustments` |
| Color Curves | `preserve_only` | `calibrated_portable` | default-only | `effect-color-curves` |
| Crop & Feather | `authorable` | `calibrated_portable` | partial | `effect-crop-feather-cohort` |
| Directional Blur | `authorable` | `calibrated_portable` | default-only | `effect-directional-blur-cohort` |
| Draw Mask | `preserve_only` | `calibrated_portable` | reject | `effect-draw-mask-opaque` |
| Drop Shadow | `authorable` | `calibrated_portable` | default-only | `effect-drop-shadow-cohort` |
| Droplet | `authorable` | `calibrated_portable` | partial | `effect-droplet-cohort` |
| Earthquake | `authorable` | `calibrated_portable` | partial | `effect-earthquake-cohort` |
| Fisheye | `authorable` | `calibrated_portable` | default-only | `effect-fisheye-cohort` |
| Focus Blur | `authorable` | `calibrated_portable` | partial | `effect-focus-blur-cohort` |
| Gaussian | `authorable` | `calibrated_portable` | — | `effect-gaussian` |
| Green Screen Keyer | `preserve_only` | `calibrated_portable` | — | `effect-green-screen-keyer` |
| Hue/Saturation Curves | `preserve_only` | `calibrated_portable` | default-only | `effect-hue-saturation-curves` |
| Kaleidoscope | `authorable` | `calibrated_portable` | default-only | `effect-kaleidoscope-cohort` |
| Perspective Tile | `authorable` | `calibrated_portable` | default-only | `effect-perspective-tile-cohort` |
| Radial Blur | `authorable` | `calibrated_portable` | partial | `effect-radial-blur-cohort` |
| Sharpen | `authorable` | `calibrated_portable` | — | `effect-sharpen` |
| Vibrancy | `authorable` | `calibrated_portable` | partial | `effect-vibrancy-cohort` |
| Vignette | `authorable` | `calibrated_portable` | — | `effect-vignette` |
| Vignette Mask | `authorable` | `calibrated_portable` | default-only | `effect-vignette-mask-cohort` |
| Add Noise | `authorable` | `unsupported` | — | `effect-add-noise-default` |
| Color Board | `preserve_only` | `unsupported` | — | `effect-color-board` |
| Color Wheels | `preserve_only` | `unsupported` | — | `effect-color-wheels` |
| Colorize | `authorable` | `unsupported` | — | `effect-colorize-default` |
| Corner Mask | `preserve_only` | `unsupported` | reject | `effect-corner-mask-opaque` |
| Flipped | `authorable` | `unsupported` | — | `effect-flipped-default` |
| Mirror | `authorable` | `unsupported` | — | `effect-mirror-default` |
| Negative | `authorable` | `unsupported` | — | `effect-negative-default` |
| Pixellate | `preserve_only` | `unsupported` | — | `effect-motion-pixellate-default` |
| Threshold | `authorable` | `unsupported` | — | `effect-threshold-default` |
| Tint | `authorable` | `unsupported` | — | `effect-tint-default` |
| Auto Mask | `preserve_only` | `apple_only` | — | `effect-auto-mask-excluded` |
| Magnetic Mask | `preserve_only` | `apple_only` | — | `effect-magnetic-mask-excluded` |
| _catch-all: Motion effect_ | `authorable` | `unsupported` | — | `effect-motion-unsupported` |
| _catch-all: FxPlug effect_ | `authorable` | `unsupported` | — | `effect-fxplug-unsupported` |
| _experimental Vulkan artifact_ | `preserve_only` | `unsupported` | — | `effect-spell-radial-blur-v1` |
| _experimental Vulkan artifact_ | `preserve_only` | `unsupported` | — | `effect-spell-pixellate-v1` |

Note: the simple built-ins (Negative, Threshold, Mirror, Colorize, Tint,
Flipped, Add Noise, Pixellate) are `unsupported` in the *capability* sense — no
calibrated parameter contract — yet the tensor runtime renders their bare,
parameterless default. Author them **without** parameters, or not at all; any
authored param omits the whole effect.

### Transitions

| Name | Authoring | Portable status | Param tier | Capability id |
| --- | --- | --- | --- | --- |
| 360° Bloom | `authorable` | `calibrated_portable` | calibrated | `transition-360-bloom-cohort` |
| 360° Circle Wipe | `authorable` | `calibrated_portable` | partial | `transition-360-circle-wipe-cohort` |
| 360° Divide | `authorable` | `calibrated_portable` | partial | `transition-360-divide-cohort` |
| 360° Gaussian Blur | `authorable` | `calibrated_portable` | partial | `transition-360-gaussian-blur-cohort` |
| 360° Push | `authorable` | `calibrated_portable` | partial | `transition-360-push-cohort` |
| 360° Reveal Wipe | `authorable` | `calibrated_portable` | partial | `transition-360-reveal-wipe-cohort` |
| 360° Slide | `authorable` | `calibrated_portable` | partial | `transition-360-slide-cohort` |
| 360° Wipe | `authorable` | `calibrated_portable` | partial | `transition-360-wipe-cohort` |
| Arrows | `authorable` | `calibrated_portable` | calibrated | `transition-motion-arrows-cohort` |
| Black Hole | `authorable` | `calibrated_portable` | default-only | `transition-motion-black-hole-xfade-default` |
| Bloom | `authorable` | `calibrated_portable` | calibrated | `transition-motion-bloom-cohort` |
| Center | `authorable` | `calibrated_portable` | partial | `transition-fx-center-cohort` |
| Clock | `authorable` | `calibrated_portable` | partial | `transition-fx-clock-cohort` |
| Clothesline | `authorable` | `calibrated_portable` | calibrated | `transition-motion-clothesline-cohort` |
| Color Planes | `authorable` | `calibrated_portable` | calibrated | `transition-motion-color-planes-cohort` |
| Cross Dissolve | `authorable` | `calibrated_portable` | — | `transition-cross-dissolve` |
| Cross Zoom | `authorable` | `calibrated_portable` | partial | `transition-fx-cross-zoom-xfade-default` |
| Curtains | `authorable` | `calibrated_portable` | partial | `transition-motion-curtains-cohort` |
| Deco | `authorable` | `calibrated_portable` | default-only | `transition-motion-deco-cohort` |
| Divide | `authorable` | `calibrated_portable` | calibrated | `transition-motion-divide-cohort` |
| Drop In | `authorable` | `calibrated_portable` | partial | `transition-motion-drop-in-cohort` |
| Earthquake | `authorable` | `calibrated_portable` | partial | `transition-motion-earthquake-cohort` |
| Fade to Color | `authorable` | `calibrated_portable` | — | `transition-fade-color` |
| Fall | `authorable` | `calibrated_portable` | calibrated | `transition-motion-fall-xfade-default` |
| Flash | `authorable` | `calibrated_portable` | default-only | `transition-motion-flash-cohort` |
| Flashback | `authorable` | `calibrated_portable` | calibrated | `transition-motion-flashback-cohort` |
| Flip | `authorable` | `calibrated_portable` | partial | `transition-motion-flip-cohort` |
| Gaussian | `authorable` | `calibrated_portable` | default-only | `transition-motion-gaussian-cohort` |
| Leaves | `authorable` | `calibrated_portable` | partial | `transition-motion-leaves-cohort` |
| Lens Flare | `authorable` | `calibrated_portable` | default-only | `transition-motion-lens-flare-cohort` |
| Light Noise | `authorable` | `calibrated_portable` | default-only | `transition-motion-light-noise-cohort` |
| Multi-flip | `authorable` | `calibrated_portable` | default-only | `transition-motion-multi-flip-cohort` |
| Page Curl | `authorable` | `calibrated_portable` | partial | `transition-fx-page-curl-cohort` |
| Pinwheel | `authorable` | `calibrated_portable` | calibrated | `transition-motion-pinwheel-cohort` |
| Radial | `authorable` | `calibrated_portable` | default-only | `transition-motion-radial-cohort` |
| Reflection | `authorable` | `calibrated_portable` | partial | `transition-motion-reflection-cohort` |
| Rotate | `authorable` | `calibrated_portable` | calibrated | `transition-motion-rotate-cohort` |
| Scale | `authorable` | `calibrated_portable` | calibrated | `transition-motion-scale-cohort` |
| Slide / Push | `authorable` | `calibrated_portable` | — | `transition-slide-push` |
| Smear | `authorable` | `calibrated_portable` | calibrated | `transition-motion-smear-cohort` |
| Spin | `authorable` | `calibrated_portable` | partial | `transition-fx-spin-cohort` |
| Static | `authorable` | `calibrated_portable` | partial | `transition-motion-static-cohort` |
| Swap | `authorable` | `calibrated_portable` | calibrated | `transition-fx-swap-cohort` |
| Swing | `authorable` | `calibrated_portable` | calibrated | `transition-motion-swing-cohort` |
| Switch | `authorable` | `calibrated_portable` | calibrated | `transition-motion-switch-cohort` |
| Veil | `authorable` | `calibrated_portable` | default-only | `transition-motion-veil-cohort` |
| Wipe | `authorable` | `calibrated_portable` | — | `transition-wipe` |
| Zoom | `authorable` | `calibrated_portable` | partial | `transition-motion-zoom-spell-v1` |
| Circle | `authorable` | `unsupported` | — | `transition-fx-circle-spell-v1` |
| Cross Blur | `authorable` | `unsupported` | — | `transition-fx-cross-blur-xfade-default` |
| Directional | `authorable` | `unsupported` | — | `transition-motion-directional-spell-v1` |
| Squares | `authorable` | `unsupported` | — | `transition-motion-squares-xfade-default` |
| _catch-all: Motion transition_ | `authorable` | `unsupported` | — | `transition-motion-unsupported` |
| _catch-all: FxPlug transition_ | `authorable` | `unsupported` | — | `transition-fxplug-unsupported` |
| _experimental Vulkan artifact_ | `authorable` | `unsupported` | — | `transition-spell-crossfade-v1` |

Cross Dissolve, Fade to Color, Wipe, and Slide/Push are hand-written torch
kernels with their own key handling (see
[EFFECTS_AND_TRANSITIONS.md](EFFECTS_AND_TRANSITIONS.md#transitions)); their
`—` param tier means "not evidenced in the parameter-calibration corpus," not
"no controls." An `unsupported` transition renders as an explicit **hard cut**,
never a silent substitution.

### Masks

| Name | Authoring | Portable status | Capability id |
| --- | --- | --- | --- |
| Color Mask (Range / HSL) | `preserve_only` | `calibrated_portable` | `mask-isolation-bounded` |
| Shape Mask | `authorable` | `calibrated_portable` | `mask-shape-standard` |

Shape Mask supports keyframed geometry. Color/Range mask renders the numeric
Spellshot color-range form; Apple's opaque color-isolation archive is preserved,
not decoded. Auto / Magnetic / tracked / ML masks are `apple_only` /
`unsupported` (see the effects table).

### Titles

| Name | Authoring | Portable status | Capability id |
| --- | --- | --- | --- |
| Basic Title | `authorable` | `calibrated_portable` | `title-basic` |
| _catch-all: Motion title template_ | `authorable` | `apple_only` | `title-motion-unsupported` |

Supported titles render as **automatically generated runtime rasters**; see
[TITLES_AND_CAPTIONS.md](TITLES_AND_CAPTIONS.md).

### Story elements

| Name | Authoring | Portable status | Capability id |
| --- | --- | --- | --- |
| mc-clip (multicam angle selection) | `authorable` | `exact_portable` | `story-multicam-selection` |

<!-- END GENERATED -->

## Structural Constructs (`tensor/support.py`)

The capability registry above governs *effects, transitions, masks, and titles*.
The rest of the render surface — clip kinds, geometry, time, compositing, scopes,
media, colour — is governed by the tensor renderer's single construct table,
[`tensor/support.py`](../../../../src/bladeworks/tensor/support.py). Each
row is either `supported` (renders) or `rejected` (raises
`TensorRenderUnsupported` naming the construct and its owning port task).

### Supported (renders exactly, unless a page notes an approximation)

- **Clips / sources:** asset-clip, still image, title/caption/Custom-Solid
  raster, spine gap, source display-rotation metadata, supported source pixel
  formats and colour matrices, HLG/PQ source transfer (tone-mapped to SDR).
- **Geometry:** conform fit/fill/none, static & animated `adjust-transform`,
  `adjust-crop` trim/crop/pan, corner pin.
- **Time:** retime forward/variable/freeze, reverse, forward/freeze inside a
  retimed group, transition endpoint holds.
- **Opacity / blend:** opacity + fades + animation, reviewed standard blend modes.
- **Effects/transitions:** ported effect handlers (leaf or group), explicit
  numeric shape/draw/color/luma masks, admitted xfade / phase-5 / equirect
  transition ids and ported handlers, cross-dissolve, hard-cut for a
  handler-less transition, multi-participant and overlapping transition sides.
- **Scopes:** inert and rendered group scopes, group transform/crop/opacity,
  group conform, group effects, retimed rendered groups, nested rendered scopes,
  transitions on/inside retimed groups, group blend, group retime maps,
  container-frame-rate-differs-from-project.

### Rejected (loud, at plan time)

- **Clips / sources:** unresolved/mismatched runtime raster, non-Custom-Solid
  generator, raster speed on a title/caption, missing/unreadable media,
  non-square pixel aspect, spatial intrinsics (360/stereo/stabilization/rolling
  shutter), unsupported source pixel format / colour matrix, malformed HDR
  metadata.
- **Geometry / blend:** conform (other modes), active conform-rate, shear/skew
  under transform, zero-crossing scale, unknown/uncalibrated blend mode
  (incl. Hue/Saturation/Color/Luminosity).
- **Time:** smooth/eased speed ramps, frame-blend / optical-flow retime.
- **Effects / transitions:** unported effect handler, unsupported effect
  parameter, animated effect parameter (outside the mask matte), opaque/tracked/
  Magnetic/Auto/ML masks, unadmitted / native-mode / prefiltered transition,
  transition missing a participant.
- **Audio:** 5.1 / surround output, non-executable audio enhancement, missing
  audio binding (see [AUDIO.md](AUDIO.md)).

## Provenance

The effect/transition/mask/title/story tables are generated from
`FCPXML_RENDER_CAPABILITIES.yaml` joined with `parameter_inventory/*.v1.json`.
Regenerate them in the same change whenever the registry changes; do not
hand-edit rows. The structural table summarizes `tensor/support.py` — the
authoritative construct table, whose rows carry the owning port task for each
rejection.
