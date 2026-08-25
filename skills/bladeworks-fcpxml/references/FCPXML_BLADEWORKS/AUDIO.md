# Bladeworks Audio

This page is how to author FCPXML audio — roles, channel routing, gain, fades,
mutes, panning, retiming, multicam audio, and calibrated enhancements — for the
Bladeworks renderer (`bladeworks render`). Placement and clocks are in
[TIMELINE.md](TIMELINE.md).

Bladeworks renders a subset of Final Cut's FCPXML. Almost all audio authors
exactly as it does natively, so this page teaches the same expressions native
authoring uses. A few base-FCPXML constructs are **not part of Bladeworks** — they
are called out where they come up, and collected in
[What Bladeworks Does Not Render](#what-bladeworks-does-not-render) — with the
supported way to get the same result.

Anchoring: `core/audio_execution.py` (semantics), `core/audio_enhancements.py`
(enhancement ports), `tensor/audio_delivery.py` (layout resolution),
`tensor/audio_pyav.py` (graph execution), `tensor/encode.py` (mux).

## Audio Is a Separate Backend

Context worth holding while authoring: Bladeworks's video and audio travel
different paths. The frame grid never carries sound — audio is resolved from the
same story elements, run through an independent **PyAV** filtergraph, and muxed
into the container beside the video at encode. Delivery is always **AAC**, mono or
stereo. This is transparent to authoring, with one practical consequence: there is
**no limiter and no automatic loudness normalization**. Author the levels you
actually want; nothing downstream will rescue a hot or quiet mix. (If you want
measured loudness, author it explicitly with `<adjust-loudness>` — see
[Audio Enhancements](#audio-enhancements).)

## Audio-Bearing Assets and Story Elements

An audio-bearing asset declares `hasAudio`, `audioSources`, `audioChannels`, and
`audioRate` as appropriate. Author audio on `asset-clip`, `audio`, `sync-clip`,
`mc-clip`, and `ref-clip`; channel counts and source timing are honoured.

`<video>` cannot expose audio of its own — it declares no `audioStart`,
`audioDuration`, or `audioRole`, and an `<adjust-volume>` inside a `<video>` fails
DTD validation. Place audio-bearing media as an `<asset-clip>` or `<audio>`. To
hang music beneath a still, connect an `<audio lane="-1">` under the `<video>` —
that is the normal pattern.

## Roles and Subroles

Roles identify editorial meaning, not physical channels. Common roles are
`dialogue`, `music`, and `effects`; a dot adds a subrole, for example
`dialogue.dialogue-1`. The same role can appear on a clip, a direct `<audio>`
element, or a source component. Preserve custom role names and keep one consistent
hierarchy. A role window (a role active over a time span) authors as an ordinary
time-windowed volume expression.

## Split Audio and J/L Edits

Select an audio interval independently of the visible video edit with `audioStart`
and `audioDuration`:

```xml
<asset-clip ref="a" name="J and L extension"
            offset="4s" start="4s" duration="2s"
            audioStart="3s" audioDuration="4s"
            audioRole="dialogue"/>
```

The visible source interval is `[4s, 6s)`; the selected audio interval is
`[3s, 7s)`, giving one second of leading and one second of trailing audio around
the visible edit. `audioStart` picks the first audio source time regardless of the
video `start`; `audioDuration` may exceed video `duration` when source handles
exist. `offset` still places the visible edit at project time `4s`. Verify both
intervals against source bounds — an out-of-bounds audio window is an authoring
error.

## Channel Routing and Source Selection

Route physical source channels to output channels with `audio-channel-source`,
which may carry its own gain:

```xml
<audio-channel-source srcCh="1, 2" outCh="L, R" role="dialogue.dialogue-1">
  <adjust-volume amount="-6dB"/>
</audio-channel-source>
```

- **`srcCh`** uses **one-based** source channel numbers (`1`, or `1, 2`). Inspect
  the asset's real channel layout rather than assuming a label from position.
- **`outCh`** names destination channels (`L, R`); the count and layout must be
  compatible with the sequence and the source mapping.
- Omitting `srcCh`/`outCh` resolves through the documented default routing matrix
  — it is derived, not guessed.

Direct routing on an `<audio>` element is equally valid where the DTD permits:

```xml
<audio ref="a" lane="-1" offset="0s" start="0s" duration="3s"
       srcCh="1, 2" outCh="L, R" role="dialogue"/>
```

Two more source-selection shapes author normally:

- **`audio-role-source`** selects a role component and can own its own gain or
  other adjustments; it is common inside multicam selections.
- **`sync-source`** carries the source of a synchronized container and renders as
  part of it.

Two attributes govern component state and are **not** interchangeable — they live
on different shapes:

```xml
<audio-channel-source srcCh="3" outCh="C" role="dialogue.alt" active="0"/>
<audio ref="a" offset="0s" start="0s" duration="3s" enabled="1"/>
```

`active="0"` disables one source-component selection; `enabled="1"` keeps a whole
audio story element enabled. Both are honoured.

**One placement rule.** A mute or source-window belongs at the **clip level**, not
on the source-instance layer of a mixed multicam or compound pad. A mute/window
placed on that inner layer stops graph construction for the item rather than being
silently ignored — apply it clip-level instead.

## Gain

`adjust-volume/@amount` is in **decibels**, and the unit is mandatory:

```xml
<adjust-volume amount="-6dB"/>
```

Zero dB is unity. Static and keyframed gain both author normally. Convert any
linear multiplier to dB before writing it:

```text
linear 1.0 -> 0dB      linear 0.5 -> -6.02dB      linear 0.0 -> -96dB (a floor)
```

There is no `-infinity` to write, so silence needs an explicit floor value. Only
emit volume adjustments on audio-bearing story elements. Author gain in dB — a
non-`dB` unit is not something Bladeworks coerces (see
[What Bladeworks Does Not Render](#what-bladeworks-does-not-render)).

## Edge Fades

Fades nest under the volume `amount` parameter:

```xml
<adjust-volume amount="-6dB">
  <param name="amount">
    <fadeIn type="linear" duration="1s"/>
    <fadeOut type="easeOut" duration="1s"/>
  </param>
</adjust-volume>
```

Curve types render as: `linear` → triangular, `easeIn` / `easeOut` → `qsin`,
`easeInOut` / `smooth` → `hsin`. Final Cut's own defaults are `easeIn` for a
`fadeIn` and `easeOut` for a `fadeOut`, so the `easeOut` above is the native
default shape. Keep each fade's `duration` within the owning component's audio
extent — a fade longer than the item, an unknown fade kind, or an unsupported
curve is refused at plan time rather than clamped.

## Mute Ranges

```xml
<mute start="2s" duration="1/2s"/>
```

A `<mute>` authors as a time-windowed volume expression in the owning source's
local domain; multiple ranges are fine. This is the way to silence a range in new
work — author the mute (or a clip-level `<adjust-volume>` floor) directly.

## Stereo Panning

Author stereo pan with `adjust-panner` in Stereo Left/Right mode. The `amount`
unit is **normalized**: `-100` is left, `0` centre, `100` right. Stereo Left/Right
is constant-power, so centre is not a naive equal-amplitude mix.

```xml
<adjust-panner mode="1 (Stereo Left/Right)" amount="-100"/>
```

Animate the pan by keyframing `amount`:

```xml
<adjust-panner mode="1 (Stereo Left/Right)">
  <param name="amount">
    <keyframeAnimation>
      <keyframe time="3s" value="-100" curve="linear"/>
      <keyframe time="6s" value="100" curve="linear"/>
    </keyframeAnimation>
  </param>
</adjust-panner>
```

Keyframe `time` is the audio component's local parameter time — on a split-audio
clip that domain is anchored by `audioStart`, not the video `start`, so an audio
keyframe time can legitimately fall outside the visible video interval. Author the
`amount` in normalized units; a non-normalized unit is not coerced. Surround and
spatial panner modes (`Ambience`, `Create Space`, …) are outside Bladeworks — see
[What Bladeworks Does Not Render](#what-bladeworks-does-not-render).

## Audio Through timeMap

Audio retime rides the same `<timeMap>` you author for video
([TIMELINE.md](TIMELINE.md)):

- **Fast / slow / variable forward** playback follows source-time progression;
  validate audible quality at extreme rates.
- **Reverse** plays the interval backwards.
- **`preservesPitch`** is threaded into the retime path and honoured. In practice
  Final Cut omits it by default; preserve an exported value rather than adding it
  speculatively.
- **Freeze / hold** is the one approximation that changes an authoring choice: a
  freeze renders **silence** for the exact held source-time interval, with normal
  audio before and after. It does **not** sustain the last sample or synthesize
  time-stretched audio. If you need sound under a held frame, author it as a
  separate connected `<audio>` item rather than expecting the freeze to carry it.

## Multicam Audio

An `mc-clip` can take video from one angle and audio from another:

```xml
<mc-source angleID="angle-a" srcEnable="audio">
  <audio-role-source role="dialogue.dialogue-1"><adjust-volume amount="6dB"/></audio-role-source>
</mc-source>
<mc-source angleID="angle-b" srcEnable="video">…</mc-source>
```

`srcEnable` selects `audio`, `video`, or `all`, and every `angleID` must exist in
the referenced multicam resource. Descendant sources mix into one local pad, which
is then retimed and controlled once. The outer `mc-clip` `audioStart` /
`audioDuration` drive the timeline split; the inner `mc-source` nodes choose and
adjust angle components — keep those two stages distinct. (Remember the placement
rule from [Channel Routing](#channel-routing-and-source-selection): a
source-instance-layer mute/window on the mixed pad is refused — apply it at the
clip level.)

## Audio Enhancements

Three enhancement nodes are **calibrated and authorable**, ported against Final
Cut A/B references (`core/audio_enhancements.py`). They sit ahead of any volume or
panner adjustment on the same element:

```xml
<adjust-loudness amount="6.0" uniformity="7.0"/>
<adjust-noiseReduction amount="20.0"/>
<adjust-voiceIsolation amount="80.0"/>
```

- **`adjust-loudness`** and **`adjust-noiseReduction`** author directly.
- **`adjust-voiceIsolation`** authors only when the frozen model registered as
  `voice_isolation.v1.json` is present; without that model the node is
  unavailable, so do not author voice isolation into an environment that lacks it.

Any other enhancement payload is opaque — its parameter contract is unknown, so
Bladeworks cannot execute it and will not guess or silently drop it. Author the
supported calibrated form above, or omit the enhancement (see
[What Bladeworks Does Not Render](#what-bladeworks-does-not-render)).

## Output Layout and Delivery

Author **mono or stereo** output. If a sequence has no audible audio (or you pass
`--video-only`), a sequence-length silent AAC bed is muxed so the container always
carries an audio track; `--video-only` additionally records a loud `omitted`
finding for the dropped source audio and fails under `--strict`. Declared audio
that cannot be decoded plays silent for its interval and records a verbatim
`omitted` finding — never a quiet substitution.

## What Bladeworks Does Not Render

These base-FCPXML audio constructs are outside Bladeworks; each is refused loudly
at plan time rather than being coerced, downmixed, or silently dropped. Author the
supported alternative instead.

| Not rendered | Author instead |
| --- | --- |
| `sequence/@audioLayout="surround"` (5.1 delivery) and surround panner modes (`Ambience`, `Create Space`) | Author stereo output and Stereo Left/Right panning. |
| `adjust-volume/@amount` in a non-dB unit | Author gain in dB (convert linear → dB). |
| `adjust-panner/@amount` in a non-normalized unit | Author a normalized pan amount (`-100`…`100`). |
| Opaque / non-executable enhancement payloads (e.g. an opaque Match EQ) | Author the supported calibrated enhancement (`adjust-loudness` / `adjust-noiseReduction` / `adjust-voiceIsolation`), or omit it. |
| `adjust-voiceIsolation` without the frozen `voice_isolation.v1.json` model | Author it only where the model is registered, otherwise omit voice isolation. |

## Pitfalls

- Relying on an implicit limiter or loudness normalizer — there is none; author
  final levels directly.
- Authoring `audioLayout="surround"` and expecting an automatic stereo downmix.
- Treating dB values as linear multipliers, or panner amounts as anything but
  normalized.
- Using zero-based `srcCh` numbering, or treating role names as physical channel
  mappings.
- Authoring a fade longer than its clip, or an unsupported fade curve.
- Expecting a freeze/hold to sustain audio — it renders as silence; hang a
  separate `<audio>` item if you need sound there.
- Applying a source-instance-layer mute/window on a mixed multicam pad — apply it
  clip-level.
- Handing an opaque enhancement (or a voice-isolation node with no frozen model)
  and expecting a render.
- Placing `<adjust-volume>` inside a `<video>` — it fails DTD validation; use an
  `<asset-clip>` or `<audio>`.
