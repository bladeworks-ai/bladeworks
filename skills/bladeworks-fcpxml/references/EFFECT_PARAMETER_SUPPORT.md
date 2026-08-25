# Effect & transition parameter-support catalog

This is the per-effect / per-transition breakdown of **which authored parameters each port actually
honors** in the tensor renderer, versus which run at a fixed calibrated default or reject.

## The two-layer model (why "default" is not one thing)

Parameter *meaning* is decided by the CPU emitter (`core/color.py`, `core/cohort_effects.py`):
it reads authored keys, clamps them, and emits an ffmpeg string / numeric constants. The
tensor port then re-evaluates that emitted string (color / geq / vibrancy) or re-freezes the
emitter's constants (cohort). **Authoring is gated upstream** by
`core/effect_parameters.py::unsupported_parameter_reason`: a key not in the effect's
registry/calibration dict — *or any animated control* — rejects the whole effect closed
(`effect_parameters.py:71`). So an effect can be "default-only" for three different reasons,
distinguished below.

**Support legend**

| Mark | Meaning |
|---|---|
| ✅ | **Parameterized** — reads authored keys, clamps to range, renders the authored look |
| ◐ | **Partial** — honors some controls; others are fixed or ignored |
| ○ | **Default-only** — always renders the fixed calibrated default (see the *why* column) |
| ⊘ | **No-op** — identity passthrough; rejects only opaque template data |
| ⚠ | **Divergence** — silently ignores an authored control the reference honors (a defect) |

Cross-cutting: **animated effect parameters reject** everywhere except the masked-effect
matte (which supports keyframed mask shapes). Static authored values are what the ✅/◐ ports read.

---

## Effects

### Simple built-ins — ○ default-only (any authored param omits the whole effect)

`negative`, `threshold`, `mirror`, `colorize`, `tint`, `flipped`, `add_noise`, `pixellate`.
Shared guard `_require_no_params` → reject `"effect (unsupported parameters)"` at
`fx_basic.py:134`. *Why:* the corpus never pinned these templates' control keys/ranges, so
a parameterized instance is reported rather than rendered with a guessed default
(`BASIC_EFFECTS.md:53`). `add_noise` additionally rejects canvas width > 4096; `tint` also
needs a ported YUV bridge link.

### Blur / adjust — ✅ parameterized

| Effect | Registry key | Honored params (key → effect) | Notes | file:line |
|---|---|---|---|---|
| Gaussian blur | `gaussian` | Amount `…/986883376/2/100`, Boost `…/986884620/2/100` → sigma | clamps; unknown params ignored; never rejects | `fx_basic.py:633` |
| Sharpen | `sharpen` | Amount `…/986883554/2/100` → unsharp | rejects only on unported YUV bridge link | `fx_basic.py:695` |
| Vignette | `vignette` | Strength `…/200/202` (def 0.65), Size `…/987213589/1` (def 1.5) → angle | clamps to [0.01, π/2] | `fx_basic.py:746` |

### Color — ✅ parameterized (Board & keyer are bounded approximations)

| Effect | Registry key | Honored params | Notes | file:line |
|---|---|---|---|---|
| Color Adjustments | `color_adjustments` | Brightness `2`, Exposure `3`, Contrast `17` (0.05–3.0), Saturation `16` (0–3), Shadows `4`+Highlights `7`→gamma, Black Point `1` (0–0.2), Warmth `14/12/10`+Tint `15/13/11` | full basic grade, bit-exact bridge | `core/color.py:80` |
| Color Board | `color_board` | Color pucks `2000/2003/2002/2001`, Saturation `2004–2007`, Exposure `2008–2011` | zones collapsed to one global sat + one master curve (approx, SSIM ~0.91) | `core/color.py:124` |
| Color Wheels | `color_wheels` | Temperature `8890`, Tint `8891`, Hue `8892` | wheel channels 1-4 proprietary → dropped | `fx_basic.py:885` |
| Color Curves | `cohort_color_curves` | ⊘ none | identity no-op; rejects opaque data at `fx_basic.py:856` | — |
| Hue/Sat Curves | `cohort_hue_saturation_curves` | ⊘ none | identity no-op; rejects opaque data | `fx_basic.py:856` |

### Keying & masks — ✅ parameterized

| Effect | Registry key | Honored params | file:line |
|---|---|---|---|
| Green Screen Keyer | `green_screen_keyer` | key_color 3×[0,1], softness [0,20], strength [0,2], spill_level [0,1], chroma/luma rolloff, green/blue chroma & min/max, mix [0,1] | `fx_keyer.py:89` |
| Masked Effect | (matte, via `apply_masked_effect`) | Shape: radius `160`, position `201`, rotation `202`, curvature `159`, feather `102`, opacity `103`, falloff `104` — **keyframe-animated**; Color/Range/Draw mask controls; blend add/subtract/multiply + invert | `fx_mask.py:255` |

### Cohort stylized — ◐ partial / ○ default-only

| Effect | Registry key | Support | Honored params | file:line |
|---|---|---|---|---|
| Cartoon | `cohort_cartoon` | ◐ | Amount `…/100310/2/100` → blur/poster/unsharp (only control) | `fx_cohort.py:160` |
| Camcorder | `cohort_camcorder` | ◐ | Amount, Recording (≥0.5 toggles HUD), Size, Battery | `fx_cohort.py:198` |
| Focus Blur | `cohort_focus_blur` | ◐ | Amount, Softness, Emphasis, Width, Height | `fx_cohort.py:336` |
| Drop Shadow | `cohort_drop_shadow` | ○ | none — fixed payload (opacity 0.75, sigma 3.0, offset 4,4); controls not in reviewed contract | `fx_cohort.py:377` |
| Callout | `cohort_callout` | ○ | none — fixed graph; rejects any authored control | `fx_branched.py:246` |

### Geq distort (`fx_warp.py`) — mixed

| Effect | Registry key | Support | Honored params | file:line |
|---|---|---|---|---|
| Radial Blur | `cohort_radial_blur` | ✅ | Amount `…/986883376/2/100`, Center | `core/cohort_effects.py:319` |
| Droplet | `cohort_droplet` | ✅ | Intensity `…/10013/2/100` (+center/radius/thickness) | `core/cohort_effects.py:399` |
| Crop & Feather | `cohort_crop_feather` | ✅ | Width `…/989379746`, Height `…/989379838`, Feather `…/989379995` | `core/cohort_effects.py:369` |
| Vibrancy | `cohort_vibrancy` | ✅ | Amount `…/987152515/2/100` (+Protect Skin) | `core/cohort_effects.py:99` |
| Directional Blur | `cohort_directional_blur` | ○ | *emitter reads Amount/Direction but the registry admits no keys → authoring fails closed; runs at default* | `fx_warp.py:40` |
| Fisheye | `cohort_fisheye` | ○ | same (no admitted keys) | `fx_warp.py:40` |
| Vignette Mask | `cohort_vignette_mask` | ○ | same | `fx_warp.py:40` |
| Kaleidoscope | `cohort_kaleidoscope` | ○ | same | `fx_warp.py:40` |
| Perspective Tile | `cohort_perspective_tile` | ○ | same | `fx_warp.py:40` |

### ⚠ Earthquake — partial, with a silent-divergence defect

`cohort_earthquake` (`tensor/effects.py:266`) reads **Amount** `…/10063/2/100` and **Layers**
`…/10044/4`, but **hardcodes `phase_x = phase_y = π/2` and never reads the Epicenter key**
`9999/10039/100/10044/5`. The CPU reference computes `phase = epicenter × π`
(`core/cohort_effects.py:428-431`), so an authored off-center epicenter renders as centered
**with no error**, and the port skips `_validate`, so it does not fail closed on unexpected
params the way the `fx_cohort` handlers do. This violates "no silent failures" and is a
tracked fix candidate (read the vector, or reject when an authored epicenter is present).

---

## Transitions

**Dispatch is unified (one interface).** Every transition is one `Transition` object —
`apply(payload, A, B, ctx) -> frame` — registered by `kind` in the single `TRANSITIONS`
registry (`transitions.py`). "xfade=custom" is not a separate class of transition: it is one
implementation *strategy* (an ffmpeg registry expression evaluated in torch by `expr.py`,
never leaving PyTorch), the same interface as the hand-written torch kernels (Cross Dissolve,
Wipe, Slide/Push). The only real axis of difference is `Lowered.needs_history` — temporal
kernels (Phase-5 earthquake / flashback) that read a per-side 3-frame window; the renderer
builds that history purely off the flag. So the parameter buckets below are about *how each
transition's authored controls reach it*, not about different dispatch paths.

### ○ Fixed-default (parameterless)

Cross Dissolve (`transitions.py:207`); Phase-5 `directional_blur`, `cross_blur` (hblur),
`cohort_flashback`, `cohort_gaussian`, `cohort_radial` — the strict validator allows **zero
keys** and rejects any param present (`tr_phase5.py:151`).

### ✅ Parameterized (native key reading)

| Transition | Handler | Honored params | Rejects on | file:line |
|---|---|---|---|---|
| Cross Dissolve | `cross_dissolve` | fixed Final Cut video-dissolve default; no authored controls | any future non-default control is outside the published Studio contract | `transitions.py:387` |
| Wipe | `wipe` | Direction key `13` ∈ {0=L,1=U,2=R,3=D}; duration divisor | non-int / out of set | `tr_handlers.py:254` |
| Slide / Push | `slide_push` | Direction key `4` ∈ {0–3}; Mode key `5` ∈ {0=Slide, 2=Push} (mode 1 rejects). Studio exposes separate Slide and Push entries backed by the shared resource UID. | non-int / out of set | `tr_handlers.py:389` |
| Fade to Color | `fade_color` | Color key `3` (else "Color") → RGB triplet, def `0,0,0` | non-numeric / < 3 components | `tr_handlers.py:175` |
| Phase-5 Earthquake | `xfade`→phase5 | Smoke bool (`EARTHQUAKE_SMOKE_KEY`) → smoke vs no-smoke | any other key; value not bool | `tr_phase5.py:142` |
| Phase-5 Drop In | `xfade`→phase5 | Smoke bool (`DROP_IN_SMOKE_KEY`) | any other key; not bool | `tr_phase5.py:144` |
| Phase-5 Smear | `xfade`→phase5 | Direction numeric ∈ {0,1} (`SMEAR_DIRECTION_KEY`) | other key; bool / non-numeric / ∉{0,1} | `tr_phase5.py:146` |
| Equirect Gaussian Blur | `equirect_gaussian_blur` | `plan.strength`→h-sigma, `plan.spread`→v-sigma | profile mismatch | `tr_equirect.py:374` |

### ◑ Registry pass-through (parameter-aware, admission per-id, no tensor-side key handling)

The **27 admitted `xfade=custom` ids** (`ADMITTED_XFADE_IDS`, `transitions.py:107`) and the **6
equirect expression ids** (`ADMITTED_EQUIRECT_IDS`) fold authored `parameter_values` into the
registry-built expression string (`transitions.py:304`, `tr_equirect.py:363`). The tensor
layer evaluates whatever text results — it neither enumerates nor rejects specific keys;
admitting an id admits every registry parameter branch for it. (`cohort_static_default` is
forced to CPU float64.)

### ⛔ Rejected / unported transitions

`equirect_bloom_default` (prefilter `equirectangular_bloom`) is not ported → loud reject
(`tr_equirect.py:367`). Any xfade/equirect id resolving to a native mode or carrying a
prefilter, and any unadmitted id, reject at plan time.

---

*Provenance: assembled 2026-08-18 from a read-across audit of the tensor effect/transition
ports on branch `codex/tensorfcp-combined`; every row is anchored to a `file:line` citation.
Update in the same PR when a port's parameter surface changes.*
