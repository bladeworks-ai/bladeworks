# Bladeworks Timeline and Timing

This page is how to author FCPXML timing — clocks, placement, story
relationships, containers, retiming, and transition topology — for the Bladeworks
renderer (`fcpxml render`). Geometry and compositing are in
[GEOMETRY.md](GEOMETRY.md); resource declarations are in [CORE.md](CORE.md).

Bladeworks renders a subset of Final Cut's FCPXML. Almost all timing authors
exactly as it does natively, so this page teaches the same time syntax, placement
arithmetic, and transition geometry the native reference does. A few base-FCPXML
timing constructs are **not part of Bladeworks** — they are called out where they
come up, and collected in [What Bladeworks Does Not Render](#what-bladeworks-does-not-render)
with the supported way to get the same result. This is a self-contained authoring
reference: targeting `fcpxml render`, you should not need another timing
document.

Anchoring: `tensor/support.py`, `tensor/decode.py` (frame ownership),
`tensor/plan.py` (scopes / transitions), `core/retime.py`, `tensor/renderer.py`.

## Time Syntax and Exact Rational Values

FCPXML times are rational seconds. Use `0s`, an integer such as `3s`, or a
fraction such as `1001/30000s`. Reduce fractions when practical and never pass
them through floating point. A frame at 29.97 fps is `1001/30000s`, not `0.033s`.
Bladeworks evaluates the exact rationals; a decimal approximation moves frame
ownership and lands the cut on the wrong frame.

## The Four Time Domains

Identify which clock a number lives in before you write it. Placement, source
selection, and animation each use a different domain.

### Project Timeline Time

Positions in the project sequence. A spine item's `offset` is interpreted here.

### Containing-Storyline Time

A connected or nested story element is placed relative to its containing story,
not the project root. Resolve placement recursively:

```text
child_absolute = parent_absolute + child.offset - parent.start
```

```xml
<asset-clip ref="base" offset="10s" start="20s" duration="4s">
  <video ref="overlay" lane="1" offset="21s" start="3600s" duration="1s"/>
</asset-clip>
```

The overlay begins at `10s + 21s - 20s = 11s`. Apply the same equation at every
nested or connected boundary. **Forgetting `- parent.start` is the most common
placement error** — Bladeworks faithfully renders the clip at the wrong time you
actually authored, so the mistake is silent.

### Source-Timecode Time

`start` on an asset or reusable sequence anchors its source timecode; a story
element's `start` selects from that source domain:

```xml
<asset id="a1" start="3600s" duration="10s" .../>
<asset-clip ref="a1" offset="0s" start="3602s" duration="3s"/>
```

Placement begins at project `0s`; source selection begins two seconds into the
asset. Source selections must stay within the asset's available media bounds — an
out-of-bounds selection is refused at plan time rather than rendering black.

### Parameter-Animation Time

Transform, corner, anchor, and opacity keyframes use the containing story
element's local parameter-time domain, numerically anchored at that element's
`start`. This clock advances independently of the source clock: it keeps moving
forward through a source freeze or reverse (see
[Animation Through Retime](#animation-through-retime)). It is never evaluated by
substituting decoded source time from a `timeMap`.

## Frame Ownership Is Floor / Nearest-Earlier

Beyond the exact rational clocks, one rendering rule governs which source frame a
given output frame shows: **frame ownership is floor / nearest-earlier**. Output
frame `j` shows the source frame whose display interval contains the output
frame's start instant — the last decoded frame with presentation time `≤` that
instant. This is the same rule for spine leaves and for leaves inside a container,
and it matches the witnessed `frameSampling="floor"` semantic
(see [conform-rate](#conform-rate)).

A source whose frame rate differs from the container/project rate (e.g. a 60 fps
angle in a 29.97 multicam) is floor-owned by the same rule: output frame `j` owns
source frame `floor(j · srcRate / seqRate)`. You do not author around this — it is
the calibrated behaviour, identical whether or not the rates match.

## offset, start, duration, audioStart, audioDuration

- `offset`: placement in the containing storyline's coordinate system.
- `start`: first selected source or local-sequence time.
- `duration`: timeline extent of the story element.
- `audioStart` / `audioDuration`: audio source start / extent when split from
  video (J/L edits). See [AUDIO.md](AUDIO.md).

For J/L edits, audio can begin before the visible edit or continue after it.
Every source selection must stay within available media bounds.

## Primary Storyline, Lanes, and Secondary Storylines

`<spine>` contains ordered story elements and transitions; its direct children
form one adjacency chain. Positive lanes composite above the primary story,
negative below (compositing z-order is `(lane, document_order)`). A connected item
uses its parent as a **temporal anchor** but stays spatially beside it — it does
not inherit the parent's transform, opacity, crop, or clipping.

`<spine lane="...">` is a secondary storyline: its own children form a separate
adjacency chain, so transitions inside it straddle the secondary storyline's
immediate neighbours, not the primary spine's.

A `<transition>` must be a direct child of a `<spine>` (primary or secondary). You
cannot attach a transition to a lone connected clip — it sits on a lane, not a
storyline, and has no adjacency chain to straddle. To place a transition on
connected content, first promote both participants into a secondary storyline
(`<spine lane="...">`) and write the transition between them. A transition that is
not a spine child fails DTD validation before it reaches Bladeworks.

Importing a document does not rewrite later `offset`s. If you shorten, lengthen,
insert, or delete a primary-storyline clip, later items stay where their
`offset` attributes say. Packed storyline (ripple) means you write those
`offset`s so items stay back-to-back from `0s`, and you set `<sequence duration>`
to the new end. A leftover hole is an explicit `<gap>` occupying that window,
used only when the edit is supposed to hold later clips still.

## Roles

Roles describe editorial meaning (`video`, `dialogue`, `music`, `effects`,
`titles`, or a subrole such as `dialogue.interview-1`), not track numbers. Keep
role strings consistent on clips and audio sources; Bladeworks carries them
through for audio routing (see [AUDIO.md](AUDIO.md)). Which attribute holds the
role is element-specific: a story element that owns both a video and an audio
stream splits them across two attributes; a single-stream element uses one `role`.

| Element | Role attributes |
| --- | --- |
| `asset-clip` | `videoRole` and `audioRole` |
| `video`, `title` | `role` |
| `audio`, `audio-role-source` | `role` |
| `caption` | `role`, which must also name a caption format |
| `mc-clip`, `ref-clip`, `sync-clip` | none — the DTD declares no role attribute here. Roles live on the elements inside the referenced media; writing `videoRole` here fails DTD validation. |

```xml
<asset-clip videoRole="video.main" audioRole="dialogue.main" .../>
<video role="video.overlay" .../>
<title role="titles.main" .../>
```

Keep role suffixes simple, stable, and human-readable — letters, numbers,
underscores, or hyphens.

## Story Elements

`asset-clip`, `video`, `audio`, `gap`, `clip`, `sync-clip`, `audition`,
`ref-clip`, and `mc-clip` all place and composite. The container elements
(`clip`, `sync-clip`, `audition`, `ref-clip`, `mc-clip`) compose genuinely
recursively — see [Compounds and Scopes](#compounds-and-scopes).

### asset-clip

References an `<asset>` and carries media timing, adjustments, audio components,
filters, connected children, notes, and metadata. It is the default editable media
edit. Required when authoring one: `ref`, `name`, `offset`, `start`, `duration`.

Geometry-critical: `format`, the source format ID declared by the referenced
`<asset>`, is **required whenever the asset format differs from the sequence
format**, and optional when they match (which is why most examples omit it).
Omitting it in the differing case can make `<adjust-conform type="fill"/>` render
with Fit-style letterboxing even though the XML is DTD-valid — Bladeworks conforms
from whatever format it can resolve, so state the format the asset already
declares. When in doubt, write it: it is never wrong.

`asset/@name` (the source media) and `asset-clip/@name` (one use of that media in
the timeline) describe different things. Several clips may share one `ref` and
carry distinct instance names; prefer distinct names so repeated segments do not
all appear under the same source filename.

### video (and stills)

A video-bearing story element, often an internal or connected child. Use
`<video>` for still-image placement and for video-like connected overlays that do
not need the audio-bearing `<asset-clip>` shape. Stills use the `start="3600s"`
source sentinel (see [stills](CORE.md#rate-undefined-still-formats)); a still
placed at source `0s` can fail to produce canvas pixels. `<video>` has no `format`
attribute — its geometry comes from the referenced asset's format, so the `format`
trap above cannot arise.

### audio

References an audio-bearing resource with `srcCh`, `outCh`, role, and adjustments.
Audio routing renders through a separate PyAV backend — see [AUDIO.md](AUDIO.md).

### gap

Occupies timeline time without media and is a real story element and transition
participant. A transition adjacent to a gap does not search past it for a later
clip. A gap side with **no** video participant makes the transition one-sided,
which Bladeworks refuses (see [Transitions](#transitions)).

### clip, sync-clip, audition, ref-clip, mc-clip

These are story containers. Their internal children compose within the container's
local context; the container's own outer adjustments apply to the completed
internal surface. Each renders as a recursive group scope
([Compounds and Scopes](#compounds-and-scopes)):

- `clip` — a story container; internal children compose in the clip's local
  context.
- `sync-clip` — groups synchronized sources; internal `sync-source` children and
  connected children keep distinct ownership.
- `audition` — owns choices, one active; the **active choice** renders and choices
  are never synthesized or switched.
- `ref-clip` — references a reusable `<media><sequence>`; its `start` selects from
  that sequence's local timecode, its `offset` places the completed result in the
  parent story, and outer adjustments operate in the parent context.
- `mc-clip` — `mc-clip/@ref` resolves to `<media><multicam>`, each
  `mc-source/@angleID` resolves to an existing `mc-angle/@angleID`, and
  `srcEnable="video"|"audio"|"all"` selects components. Never invent an angle ID.
  See [Multicam Audio](AUDIO.md#multicam-audio).

## Reusable Story Resources

Every reusable `<media><sequence>` owns its own local format, raster, cadence, and
timecode, and Bladeworks honours a container frame rate that differs from the
project (native cadence conversion). A 1080x1920 30 fps compound with
`tcStart="10s"` stays vertical and 30 fps when referenced from a 1920x1080 24 fps
parent; a `ref-clip start="12s"` begins two seconds into that compound. The parent
places and transforms the completed compound surface. Multicam angles are internal
resources; timeline selections never rewrite angle synchronization.

## Compounds and Scopes

Compound (`ref-clip`), multicam (`mc-clip`), sync-clip, and audition containers
lower to **group scopes** and composite recursively. A scope is classified as one
of two kinds, and the kind has real authoring consequences:

- **Inert scope** — a Fit-same-aspect / None-same-size container with no
  transform, crop, opacity, blend, effects, or retime. Its leaves fold directly
  onto the parent canvas, pixel-identical to placing the leaves themselves.
- **Rendered scope** — a container that owns a transform, crop, a conform to a
  different size, opacity, a blend mode, effects, a retime, or a cadence boundary.
  It composes on **its own container surface** and is then placed like a leaf.

The practical rule: **a container with a transform/crop/opacity/effect/retime
composes on its own surface first, then that finished surface is placed into the
parent.** Group transform / crop / opacity, group conform onto a different
container size, group blend mode, group effects, constant-speed retimed groups,
effects on a retimed group, transitions on or inside a retimed group, and nested
rendered scopes (compound-in-compound, multicam angle scope) all render. Recursion
depth is unbounded but untested at extreme depth, so treat a very deeply nested
compound as a theoretical, not certified, limit.

## Temporal Anchoring Versus Spatial Scope

An **internal container child** belongs to the container's internal composition,
is evaluated in the container's local context, and receives the container's outer
transform, opacity, and clipping after internal composition finishes. A
**connected timeline child** uses the parent as a temporal anchor, stays spatially
beside it, and does not inherit the parent's transform. Both render faithfully;
give a connected child its own transform if it must follow the parent visually.

## Transitions

Effect and parameter fidelity for each transition lives in
[EFFECTS_AND_TRANSITIONS.md](EFFECTS_AND_TRANSITIONS.md#transitions); this section
is the **timing and participant** model. A transition uses the immediately
adjacent story elements in the same storyline and never searches through a gap or
into a neighbouring container.

### Centered Placement

A transition straddles the cut, so its `offset` is half its duration *before* the
edit point:

```text
transition.offset = cut - transition.duration / 2
```

The cut is the incoming element's `offset`. The participants keep their own
offsets; only the transition reaches back. A transition written at the cut itself
plays entirely inside the incoming clip — DTD-valid, imports without a warning,
and visibly wrong.

An even frame count divides cleanly onto the frame grid; an odd one does not, and
the answer is to keep the exact rational rather than round it. At 30 fps an
11-frame transition centred on a 4s cut is `229/60s` — half a frame before
`115/30s`, which is what rounding would give. A half-frame offset is a legal,
intended rational value; rounding shifts the transition off the cut. A transition
at the very start of a spine is the one exception: clamp the offset to `0s` rather
than computing a negative one. Bladeworks renders the endpoint holds exactly.

### Participants and Sides

`clip`, `sync-clip`, `audition`, `ref-clip`, and `mc-clip` participate as complete
story elements — the container is the participant, not a descendant in its
reusable sequence. In `asset-clip -> transition -> gap`, the gap is the
participant; the transition never searches beyond it.

Each **side** of the transition is every marked participant layer (connected
lanes, group leaves) composed full-canvas, so a side with several participants
renders. Overlapping transitions are independent items and render. A transition
on a rendered group takes the group's finished surface as its side; a transition
*inside* a retimed group renders on the group clock.

Give every transition **two video participants**. A transition with only one video
side — authored into a gap-only boundary, or off the end of a spine — is
one-sided; Bladeworks refuses it at plan time rather than silently degrading it to
a cut. (A transition whose effect has no portable handler is a different case: it
degrades to the honoured hard cut, which is a supported outcome.)

### Source Handles

Because the transition straddles the cut, **each participant with a bounded source
domain must have at least half the transition's duration of unused source beyond
its edge** — the outgoing clip a tail, the incoming clip a head. A 1s dissolve
needs `1/2s` on each side. Measure the head as the participant's `start` minus its
source domain's origin (an asset's own `start`, or a compound's inner `tcStart`),
and the tail as whatever source remains past `start + duration`.

Three kinds have no bounded source and are exempt: a `<gap>`; a still
(`duration="0s"`, held indefinitely); and a generator-backed element — a
`<title>`, or a `<video>` referencing an `<effect>` — which synthesizes its
frames. A dissolve between two stills or two titles needs no handles. Container
descendants do not donate handles to an outer transition; validate available
source after `start`, `duration`, retiming, and reusable-sequence bounds.

## conform-rate

`conform-rate` reconciles a clip's source frame rate with the containing
sequence's rate. Keep the source `<format>` and the containing
`<sequence format="...">` distinct; `conform-rate` belongs on the story element
whose source cadence is being reconciled.

```xml
<asset-clip ref="camera" offset="0s" start="3602s" duration="3s">
  <conform-rate scaleEnabled="0" srcFrameRate="59.94" frameSampling="floor"/>
</asset-clip>
```

- `srcFrameRate` records the source cadence — the FCPXML 1.14 set, `23.98`
  through `120`.
- `frameSampling` selects the frame-conversion method; its default is `floor`.
  Author `floor`: it renders as the nearest-earlier ownership rule above (output
  frame `j` owns source frame `floor(j · srcRate / seqRate)`). The frame-blending
  and optical-flow methods synthesize or reselect intermediate frames and are not
  part of Bladeworks.
- `scaleEnabled` records Final Cut's cadence-scaling state. A **passive**
  `conform-rate` (`scaleEnabled="0"`, cadence bookkeeping) is fine and renders as
  native cadence conversion. An **active** `conform-rate` (`scaleEnabled="1"`,
  real rate retiming) is not part of Bladeworks — active rate conform is a
  retime, so express it through a `<timeMap>` ([timeMap](#timemap)) or author the
  source at the project cadence instead.

When both `conform-rate` and `timeMap` are present, the DTD requires
`conform-rate` first.

## timeMap

`timeMap` maps output-local time (`time`) to selected source time (`value`).
Points must be ordered by output time, and source values must stay inside the
asset or reusable-sequence bounds. `<timeMap>` also accepts an optional
`frameSampling` attribute (same value set as `conform-rate`); author `floor` or
leave it off.

The retime you can author is constant speed, reverse, freeze, and piecewise-linear
ramps — every segment joined at `interp="linear"` points.

Two-point constant map (2x playback — four source seconds in two output seconds):

```xml
<timeMap>
  <timept time="0s" value="0s" interp="linear"/>
  <timept time="2s" value="4s" interp="linear"/>
</timeMap>
```

Piecewise-linear forward / freeze / reverse:

```xml
<timeMap>
  <timept time="0s" value="0s" interp="linear"/>
  <timept time="2s" value="4s" interp="linear"/>
  <timept time="3s" value="4s" interp="linear"/>
  <timept time="5s" value="2s" interp="linear"/>
</timeMap>
```

- output `0s`–`2s`: forward at 2x;
- output `2s`–`3s`: **freeze** at source `4s` (equal adjacent `value` points hold
  source time);
- output `3s`–`5s`: **reverse** from source `4s` to `2s` (a later `value` smaller
  than the earlier one plays source backward).

Confirm both endpoints of every segment stay inside the selected source bounds.
Video freezes on the owned frame; an **audio** freeze is calibrated silence for
the held interval, not a sustained sample — if a hold needs audible sound, hold
the picture but keep audio running, or lay in a separate audio clip
([AUDIO.md](AUDIO.md)).

A true speed **ramp** — a continuously accelerating segment authored with
non-linear (`smooth`/eased) `timept` interpolation — is not part of Bladeworks.
This includes the DTD's default `smooth2` interpolation, so set `interp="linear"`
explicitly on every `<timept>`; approximate the curve as a chain of short linear
segments if you need an accelerating feel.

### Source Bounds and Transition Handles

Retiming does not create extra transition handles: the map must reach back into
the media the transition needs. For a visible body starting at source frame `S`
with `H` incoming handle frames, the first time point lands before the visible
start:

```text
first.value = S - H
first.time  = S - output_frames_for_source_frames(H)
```

For slow motion, reserve enough incoming headroom — at least `ceil(H / speed)`
source frames before the visible start when `speed < 1`. Too little produces a
negative `timept/@time`, which is invalid on import.

## Animation Through Retime

```xml
<adjust-transform>
  <param name="position">
    <keyframeAnimation>
      <keyframe time="0s" value="-32 0" curve="linear"/>
      <keyframe time="2s" value="0 18" curve="linear"/>
      <keyframe time="4s" value="28 -12" curve="linear"/>
    </keyframeAnimation>
  </param>
</adjust-transform>
```

Transform, anchor, corner, and opacity keyframes advance in the containing story
element's **local parameter time**, even while media freezes or reverses. Combined
with the piecewise map above, this parameter track keeps moving forward during the
freeze and the reverse: the parameter clock is independent of the source clock. Do
not reverse a visual parameter animation merely because the media reverses — if
you want the motion to reverse too, say so in its own keyframes. (Both `linear`
and `smooth` keyframe curves render — see
[GEOMETRY.md](GEOMETRY.md#transform-keyframes).)

## What Bladeworks Does Not Render

These base-FCPXML timing constructs are outside Bladeworks; each is refused at
plan time rather than rendering wrong. Author the supported alternative instead.

| Not rendered | Author instead |
| --- | --- |
| Smooth / eased `timept` speed ramps (including the DTD default `smooth2`) | Explicitly-linear `timept` ramps (`interp="linear"`); approximate a curve as short linear segments. |
| `frameSampling="frame-blending"` / optical-flow retiming or conform | `frameSampling="floor"` (nearest-earlier frame ownership). |
| Active `conform-rate` (`scaleEnabled="1"`, real rate retiming) | Retime through a `<timeMap>` ([timeMap](#timemap)), or author the source at the project cadence. |
| A transition with only one video participant (into a gap-only boundary, off the end of a spine) | Give the transition two video participants on both sides of the cut. |

## Pitfalls

- Rounding frame rationals to decimal seconds — shifts floor frame ownership.
- Forgetting `- parent.start` in connected placement.
- Treating source timecode as project placement.
- Assuming a connected child is spatially inside its temporal anchor — give it its
  own transform.
- Looking through a container or gap for transition participants.
- Writing a transition at the cut instead of half its duration before it.
- Authoring a `smooth2` (or any eased) speed ramp — linearize it.
- Expecting frame-blended or optical-flow slow motion — author `floor`.
- Authoring active `conform-rate` (`scaleEnabled="1"`) — retime via `timeMap`.
- Authoring a transition into a gap-only or off-the-end boundary — give it two
  video participants.
- Treating freeze audio as a held sample — it is calibrated silence.
- Reversing a visual parameter animation merely because the media reverses.
