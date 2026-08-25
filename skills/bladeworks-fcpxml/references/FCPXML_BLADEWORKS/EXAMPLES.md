# Bladeworks Examples

This page is a set of copyable FCPXML authoring templates for the Bladeworks
renderer (`bladeworks render`). Every XML block labelled a **complete document**
is a full, DTD-valid FCPXML 1.14 document you can render as-is — start from the
one closest to what you want to build, then swap in your own media and edit the
constructs you need. The narrower rules each example relies on live in the
sibling pages: [CORE.md](CORE.md), [TIMELINE.md](TIMELINE.md),
[GEOMETRY.md](GEOMETRY.md), [AUDIO.md](AUDIO.md),
[TITLES_AND_CAPTIONS.md](TITLES_AND_CAPTIONS.md),
[EFFECTS_AND_TRANSITIONS.md](EFFECTS_AND_TRANSITIONS.md),
[INVENTORY.md](INVENTORY.md).

Render any complete document with:

```bash
bladeworks render project.fcpxml --project "PROJECT NAME"
```

Replace the example file URLs and stable UIDs with real media identities before
you render. Add `--strict` to have the renderer flag anything it would render as
a calibrated approximation rather than an exact reproduction — a clean `--strict`
run confirms your document sits entirely inside the exactly-reproduced surface.

A handful of DTD-valid constructs are deliberately not rendered by Bladeworks;
they are collected in
[What Not To Author](#what-not-to-author-and-what-to-author-instead) with the
supported construct to author in their place.

## 1. Minimal One-Clip Project

The smallest renderable document, and the skeleton every other example builds on:
one Rec.709 square-pixel format, one source, one spine clip. Copy this first and
grow it.

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.14">
  <resources>
    <format id="r1" name="FFVideoFormat1080p30" frameDuration="1/30s"
            width="1920" height="1080" colorSpace="1-1-1 (Rec. 709)"/>
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

## 2. Transform, Crop, and an Opacity Fade

How to move, scale, window, and fade a single clip. This authors a static
transform, a trim window, and a keyframed opacity fade-in on one `asset-clip`.
Author opacity as `adjust-blend`'s `amount` parameter (range `0..1`) — there is no
`adjust-opacity` element in the DTD. Keep the intrinsic children in DTD order,
crop → transform → blend (see
[GEOMETRY.md#intrinsic-adjustment-order](GEOMETRY.md#intrinsic-adjustment-order)).

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.14">
  <resources>
    <format id="r1" name="FFVideoFormat1080p30" frameDuration="1/30s"
            width="1920" height="1080" colorSpace="1-1-1 (Rec. 709)"/>
    <asset id="r2" name="clip.mp4" uid="EXAMPLE-TRANSFORM-UID" start="0s"
           duration="300/30s" hasVideo="1" videoSources="1" format="r1">
      <media-rep kind="original-media" src="file:///tmp/clip.mp4"/>
    </asset>
  </resources>
  <library>
    <event name="Example Event">
      <project name="Transform and Fade">
        <sequence format="r1" duration="5s" tcStart="0s" tcFormat="NDF"
                  audioLayout="stereo" audioRate="48k">
          <spine>
            <asset-clip ref="r2" name="clip.mp4" offset="0s" start="0s"
                        duration="5s">
              <adjust-crop mode="trim">
                <trim-rect left="5" right="5" top="5" bottom="5"/>
              </adjust-crop>
              <adjust-transform position="100 0" scale="1.1 1.1" rotation="0"/>
              <adjust-blend>
                <param name="amount">
                  <keyframeAnimation>
                    <keyframe time="0s" value="0" curve="linear"/>
                    <keyframe time="15/30s" value="1" curve="linear"/>
                  </keyframeAnimation>
                </param>
              </adjust-blend>
            </asset-clip>
          </spine>
        </sequence>
      </project>
    </event>
  </library>
</fcpxml>
```

## 3. Still Image, Conform Fill, and Ken Burns Pan

How to author a Ken Burns move over a still. Use a rate-undefined still `<video>`,
conform it to Fill, and give it exactly two ordered `<pan-rect>` children for the
start and end viewports. Pan requires a still source whose aspect matches the
project — see
[GEOMETRY.md#crop-spatial-trim-and-pan--ken-burns](GEOMETRY.md#crop-spatial-trim-and-pan--ken-burns).
The camera easing between the two viewports is supplied for you; do not add
keyframes.

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.14">
  <resources>
    <format id="project" name="FFVideoFormat1080p30" frameDuration="1/30s"
            width="1920" height="1080" colorSpace="1-1-1 (Rec. 709)"/>
    <format id="still" name="FFVideoFormatRateUndefined"
            width="3840" height="2160" colorSpace="1-1-1 (Rec. 709)"/>
    <asset id="photo" name="photo.jpg" uid="EXAMPLE-PHOTO-UID"
           start="0s" duration="0s" hasVideo="1" videoSources="1" format="still">
      <media-rep kind="original-media" src="file:///tmp/photo.jpg"/>
    </asset>
  </resources>
  <library>
    <event name="Example Event">
      <project name="Ken Burns">
        <sequence format="project" duration="3s" tcStart="0s" tcFormat="NDF"
                  audioLayout="stereo" audioRate="48k">
          <spine>
            <video ref="photo" name="Photo" offset="0s" start="3600s"
                   duration="3s">
              <adjust-crop mode="pan">
                <pan-rect left="0" top="0" right="0" bottom="0"/>
                <pan-rect left="30" top="16.875" right="0" bottom="0"/>
              </adjust-crop>
              <adjust-conform type="fill"/>
            </video>
          </spine>
        </sequence>
      </project>
    </event>
  </library>
</fcpxml>
```

## 4. Piecewise Retime with Animation Through It

How to retime a clip and animate its geometry at the same time. The `<timeMap>`
advances the source, holds it (freeze), then reverses it, while position, corner,
and opacity keyframes advance independently in **local parameter time**. Keep the
timeMap piecewise-linear (`interp="linear"`); to ramp speed smoothly, see
[What Not To Author](#what-not-to-author-and-what-to-author-instead).

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.14">
  <resources>
    <format id="f" name="FFVideoFormat1080p30" frameDuration="1/30s"
            width="1920" height="1080" colorSpace="1-1-1 (Rec. 709)"/>
    <asset id="a" name="retime.mov" uid="EXAMPLE-RETIME-UID" start="0s"
           duration="6s" hasVideo="1" videoSources="1" hasAudio="1" format="f"
           audioSources="1" audioChannels="2" audioRate="48000">
      <media-rep kind="original-media" src="file:///tmp/retime.mov"/>
    </asset>
  </resources>
  <library>
    <event name="Example Event">
      <project name="Retime and Animation">
        <sequence format="f" duration="5s" tcStart="0s" tcFormat="NDF"
                  audioLayout="stereo" audioRate="48k">
          <spine>
            <asset-clip ref="a" offset="0s" start="0s" duration="5s">
              <timeMap frameSampling="floor">
                <timept time="0s" value="0s" interp="linear"/>
                <timept time="2s" value="4s" interp="linear"/>
                <timept time="3s" value="4s" interp="linear"/>
                <timept time="5s" value="2s" interp="linear"/>
              </timeMap>
              <adjust-corners topRight="0 0" botRight="0 0" botLeft="0 0">
                <param name="topLeft">
                  <keyframeAnimation>
                    <keyframe time="0s" value="0 0" curve="linear"/>
                    <keyframe time="4s" value="-0.1 0.08" curve="linear"/>
                  </keyframeAnimation>
                </param>
              </adjust-corners>
              <adjust-transform>
                <param name="position">
                  <keyframeAnimation>
                    <keyframe time="0s" value="-32 0" curve="linear"/>
                    <keyframe time="2s" value="0 18" curve="linear"/>
                    <keyframe time="4s" value="28 -12" curve="linear"/>
                  </keyframeAnimation>
                </param>
              </adjust-transform>
              <adjust-blend>
                <param name="amount">
                  <keyframeAnimation>
                    <keyframe time="0s" value="0.2" curve="linear"/>
                    <keyframe time="4s" value="1" curve="linear"/>
                  </keyframeAnimation>
                </param>
              </adjust-blend>
            </asset-clip>
          </spine>
        </sequence>
      </project>
    </event>
  </library>
</fcpxml>
```

## 5. Cross Dissolve Between Two Clips

How to author a Cross Dissolve. Place the `<transition>` on the boundary between
two spine clips; it consumes source handles from both participants (timing model
in [TIMELINE.md#transitions](TIMELINE.md#transitions)). When the clips carry
audio, pair the video dissolve with an `FFAudioTransition` — a video-only
transition over audio cuts the audio hard.

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.14">
  <resources>
    <format id="f" name="FFVideoFormat1080p30" frameDuration="1/30s"
            width="1920" height="1080" colorSpace="1-1-1 (Rec. 709)"/>
    <asset id="a" name="a.mov" uid="EXAMPLE-XD-A" start="0s" duration="5s"
           hasVideo="1" videoSources="1" format="f">
      <media-rep kind="original-media" src="file:///tmp/a.mov"/>
    </asset>
    <asset id="b" name="b.mov" uid="EXAMPLE-XD-B" start="0s" duration="5s"
           hasVideo="1" videoSources="1" format="f">
      <media-rep kind="original-media" src="file:///tmp/b.mov"/>
    </asset>
    <effect id="dissolve" name="Cross Dissolve"
            uid="FxPlug:4731E73A-8DAC-4113-9A30-AE85B1761265"/>
  </resources>
  <library>
    <event name="Example Event">
      <project name="Cross Dissolve">
        <sequence format="f" duration="6s" tcStart="0s" tcFormat="NDF">
          <spine>
            <asset-clip ref="a" name="a.mov" offset="0s" start="0s"
                        duration="3s"/>
            <transition name="Cross Dissolve" offset="5/2s" duration="1s">
              <filter-video ref="dissolve" name="Cross Dissolve"/>
            </transition>
            <asset-clip ref="b" name="b.mov" offset="3s" start="0s"
                        duration="3s"/>
          </spine>
        </sequence>
      </project>
    </event>
  </library>
</fcpxml>
```

## 6. A Motion-Catalogue Transition (Bloom)

How to author one of the Motion-catalogue transitions (Bloom, Flash, Spin, and
the rest — see [INVENTORY.md#transitions](INVENTORY.md#transitions)). Reference the
transition's effect by its `uid` and place it on the clip boundary, exactly like
Cross Dissolve. Author the transition for the look you want; the individual
parameter controls are not driveable unless the inventory marks that op
`calibrated`.

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.14">
  <resources>
    <format id="f" name="FFVideoFormat1080p30" frameDuration="1/30s"
            width="1920" height="1080" colorSpace="1-1-1 (Rec. 709)"/>
    <asset id="a" name="a.mov" uid="EXAMPLE-BLOOM-A" start="0s" duration="5s"
           hasVideo="1" videoSources="1" format="f">
      <media-rep kind="original-media" src="file:///tmp/a.mov"/>
    </asset>
    <asset id="b" name="b.mov" uid="EXAMPLE-BLOOM-B" start="0s" duration="5s"
           hasVideo="1" videoSources="1" format="f">
      <media-rep kind="original-media" src="file:///tmp/b.mov"/>
    </asset>
    <effect id="bloom" name="Bloom"
            uid=".../Transitions.localized/Lights.localized/Bloom.localized/Bloom.motr"/>
  </resources>
  <library>
    <event name="Example Event">
      <project name="Bloom Transition">
        <sequence format="f" duration="6s" tcStart="0s" tcFormat="NDF">
          <spine>
            <asset-clip ref="a" name="a.mov" offset="0s" start="0s"
                        duration="3s"/>
            <transition name="Bloom" offset="5/2s" duration="1s">
              <filter-video ref="bloom" name="Bloom"/>
            </transition>
            <asset-clip ref="b" name="b.mov" offset="3s" start="0s"
                        duration="3s"/>
          </spine>
        </sequence>
      </project>
    </event>
  </library>
</fcpxml>
```

## 7. Color Adjustments Grade

How to author a Color Adjustments grade. Attach the effect to the clip as a
`<filter-video>`, copy the opaque `effectConfig` blob **verbatim** from a genuine
Final Cut export (never synthesize it), and drive the grade by editing only the
scalar `<param>` values, within the envelopes in
[INVENTORY.md#effects](INVENTORY.md#effects). The authored `<param>` values read
through a bit-exact bridge, so they control the grade precisely.

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.14">
  <resources>
    <format id="f" name="FFVideoFormat1080p30" frameDuration="1/30s"
            width="1080" height="1920" colorSpace="1-1-1 (Rec. 709)"/>
    <asset id="a" name="clip.mov" uid="EXAMPLE-COLOR-ADJUSTMENTS-CLIP"
           start="0s" duration="8s" hasVideo="1" videoSources="1" format="f">
      <media-rep kind="original-media" src="file:///tmp/clip.mov"/>
    </asset>
    <effect id="color-adjustments" name="Color Adjustments"
            uid="FxPlug:7E2022A5-202B-4EEB-A311-AC2B585D01B0"/>
  </resources>
  <library>
    <event name="Example Event">
      <project name="Color Adjustments Example">
        <sequence format="f" duration="8s" tcStart="0s" tcFormat="NDF"
                  audioLayout="stereo" audioRate="48k">
          <spine>
            <asset-clip ref="a" name="clip.mov" offset="0s" start="0s"
                        duration="8s" format="f" videoRole="video.main">
              <adjust-conform type="fill"/>
              <filter-video ref="color-adjustments" name="Color Adjustments">
                <data key="effectConfig">YnBsaXN0MDDUAQIDBAUGBwpYJHZlcnNpb25ZJGFyY2hpdmVyVCR0b3BYJG9iamVjdHMSAAGGoF8QD05TS2V5ZWRBcmNoaXZlctEICVRyb290gAGlCwwVFhdVJG51bGzTDQ4PEBIUV05TLmtleXNaTlMub2JqZWN0c1YkY2xhc3OhEYACoROAA4AEXXBsdWdpblZlcnNpb24QAtIYGRobWiRjbGFzc25hbWVYJGNsYXNzZXNfEBNOU011dGFibGVEaWN0aW9uYXJ5oxocHVxOU0RpY3Rpb25hcnlYTlNPYmplY3QIERokKTI3SUxRU1lfZm55gIKEhoiKmJqfqrPJzdoAAAAAAAABAQAAAAAAAAAeAAAAAAAAAAAAAAAAAAAA4w==</data>
                <param name="Control Range" key="19" value="0 (SDR)"/>
                <param name="Brightness" key="2" value="200"/>
                <param name="Saturation" key="16" value="-100"/>
              </filter-video>
            </asset-clip>
          </spine>
        </sequence>
      </project>
    </event>
  </library>
</fcpxml>
```

## 8. Shape-Masked Sharpen with an Animated Mask

How to author an animated shape mask that drives an effect. The `mask-shape`
matte is the one place keyframed geometry moves an effect region: animate the
mask's Position while the inside filter (Sharpen) reads a static Amount. The
`filter-video-mask` model is on
[EFFECTS_AND_TRANSITIONS.md](EFFECTS_AND_TRANSITIONS.md).

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.14">
  <resources>
    <format id="f" name="FFVideoFormat1080p30" frameDuration="1/30s"
            width="1920" height="1080" colorSpace="1-1-1 (Rec. 709)"/>
    <asset id="a" name="camera.mov" uid="EXAMPLE-MASK-UID" start="0s"
           duration="4s" hasVideo="1" videoSources="1" format="f">
      <media-rep kind="original-media" src="file:///tmp/camera.mov"/>
    </asset>
    <effect id="sharp" name="Sharpen"
            uid=".../Effects.localized/Blur.localized/Sharpen.localized/Sharpen.moef"/>
  </resources>
  <library>
    <event name="Example Event">
      <project name="Animated Shape Mask">
        <sequence format="f" duration="4s" tcStart="0s" tcFormat="NDF">
          <spine>
            <asset-clip ref="a" offset="0s" start="0s" duration="4s">
              <filter-video-mask>
                <mask-shape name="Shape Mask 1" blendMode="add">
                  <param name="Radius" key="160" value="60 45"/>
                  <param name="Curvature" key="159" value="1"/>
                  <param name="Feather" key="102" value="5"/>
                  <param name="Transforms" key="200">
                    <param name="Position" key="201">
                      <keyframeAnimation>
                        <keyframe time="0s" value="0 0" curve="linear"/>
                        <keyframe time="60/30s" value="200 0" curve="linear"/>
                      </keyframeAnimation>
                    </param>
                  </param>
                </mask-shape>
                <filter-video ref="sharp" name="Sharpen">
                  <param name="Amount"
                         key="9999/986883553/100/986883554/2/100" value="1"/>
                </filter-video>
              </filter-video-mask>
            </asset-clip>
          </spine>
        </sequence>
      </project>
    </event>
  </library>
</fcpxml>
```

## 9. Compound Clip with a Group Transform

How to place a reusable composition and transform it as a unit. A `ref-clip`
places a completed reusable `<media><sequence>`; because the outer instance
carries its own `<adjust-transform>`, the compound composes internally first and
then the whole surface is transformed together. Author the inner child motion in
the compound, then the outer instance transform in the parent (scope model in
[TIMELINE.md#compounds-and-scopes](TIMELINE.md#compounds-and-scopes)). A title
composited on top would be a connected sibling — temporally anchored but spatially
outside the compound; Bladeworks rasterizes its glyphs at runtime (see example 11).

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.14">
  <resources>
    <format id="parent-f" name="FFVideoFormat1080p24" frameDuration="1/24s"
            width="1920" height="1080" colorSpace="1-1-1 (Rec. 709)"/>
    <format id="compound-f" name="FFVideoFormat1080p30" frameDuration="1/30s"
            width="1080" height="1920" colorSpace="1-1-1 (Rec. 709)"/>
    <asset id="portrait" name="portrait.mov" uid="EXAMPLE-PORTRAIT-UID"
           start="10s" duration="6s" hasVideo="1" videoSources="1"
           format="compound-f">
      <media-rep kind="original-media" src="file:///tmp/portrait.mov"/>
    </asset>
    <media id="compound" name="Portrait Compound" uid="EXAMPLE-COMPOUND-UID">
      <sequence format="compound-f" duration="6s" tcStart="10s" tcFormat="NDF"
                audioLayout="stereo" audioRate="48k">
        <spine>
          <asset-clip ref="portrait" offset="10s" start="10s" duration="6s">
            <adjust-transform position="0 8" scale="0.9 0.9"/>
          </asset-clip>
        </spine>
      </sequence>
    </media>
  </resources>
  <library>
    <event name="Example Event">
      <project name="Compound Ownership">
        <sequence format="parent-f" duration="4s" tcStart="0s" tcFormat="NDF"
                  audioLayout="stereo" audioRate="48k">
          <spine>
            <ref-clip ref="compound" name="Compound edit" offset="0s"
                      start="12s" duration="4s">
              <adjust-transform position="20 0" scale="0.5 0.5"/>
            </ref-clip>
          </spine>
        </sequence>
      </project>
    </event>
  </library>
</fcpxml>
```

## 10. Independent Multicam Selection

How to author a multicam clip that draws video from one angle and audio from
another. Define the angles in a `<media><multicam>`, then in the `<mc-clip>` add
one `<mc-source srcEnable="audio">` (here angle A, with a dB gain) and one
`<mc-source srcEnable="video">` (here angle B, with a transform). Clip-level audio
— role and dB gain — flows through the PyAV audio backend (see
[AUDIO.md](AUDIO.md)).

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.14">
  <resources>
    <format id="f" name="FFVideoFormat1080p30" frameDuration="1/30s"
            width="1920" height="1080" colorSpace="1-1-1 (Rec. 709)"/>
    <asset id="cam-a" name="camera-a.mov" uid="EXAMPLE-MC-A" start="0s"
           duration="8s" hasVideo="1" videoSources="1" hasAudio="1" format="f"
           audioSources="1" audioChannels="2" audioRate="48000">
      <media-rep kind="original-media" src="file:///tmp/camera-a.mov"/>
    </asset>
    <asset id="cam-b" name="camera-b.mov" uid="EXAMPLE-MC-B" start="0s"
           duration="8s" hasVideo="1" videoSources="1" hasAudio="1" format="f"
           audioSources="1" audioChannels="2" audioRate="48000">
      <media-rep kind="original-media" src="file:///tmp/camera-b.mov"/>
    </asset>
    <media id="mc" name="Interview Multicam" uid="EXAMPLE-MULTICAM-UID">
      <multicam format="f" duration="8s" tcStart="0s" tcFormat="NDF">
        <mc-angle name="Camera A" angleID="angle-a">
          <asset-clip ref="cam-a" offset="0s" start="0s" duration="8s"/>
        </mc-angle>
        <mc-angle name="Camera B" angleID="angle-b">
          <asset-clip ref="cam-b" offset="0s" start="0s" duration="8s"/>
        </mc-angle>
      </multicam>
    </media>
  </resources>
  <library>
    <event name="Example Event">
      <project name="Independent Multicam Selection">
        <sequence format="f" duration="4s" tcStart="0s" tcFormat="NDF"
                  audioLayout="stereo" audioRate="48k">
          <spine>
            <mc-clip ref="mc" offset="0s" start="1s" duration="4s"
                     audioStart="0s" audioDuration="5s">
              <mc-source angleID="angle-a" srcEnable="audio">
                <audio-role-source role="dialogue.dialogue-1">
                  <adjust-volume amount="6dB"/>
                </audio-role-source>
              </mc-source>
              <mc-source angleID="angle-b" srcEnable="video">
                <adjust-transform position="0 0" scale="1 1" rotation="0"/>
              </mc-source>
            </mc-clip>
          </spine>
        </sequence>
      </project>
    </event>
  </library>
</fcpxml>
```

## 11. Runtime-Rasterized Title and Caption

How to place a `<title>` or `<caption>` for its geometry and timing. The public
executor reads the authored text and styles, lays out supported content through
FreeType/HarfBuzz, generates a temporary RGBA image, and composites that image
at the element's geometry and time. No out-of-band raster argument is exposed
by the CLI. See [TITLES_AND_CAPTIONS.md](TITLES_AND_CAPTIONS.md) for supported
templates, styles, and font-resolution behavior.

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.14">
  <resources>
    <format id="project" name="FFVideoFormat1080p30" frameDuration="1/30s"
            width="1920" height="1080" colorSpace="1-1-1 (Rec. 709)"/>
    <asset id="camera" name="camera.mov" uid="EXAMPLE-TITLE-CAMERA-UID"
           start="0s" duration="8s" hasVideo="1" videoSources="1" hasAudio="1"
           format="project" audioSources="1" audioChannels="2" audioRate="48000">
      <media-rep kind="original-media" src="file:///tmp/camera.mov"/>
    </asset>
    <effect id="basic" name="Basic Title"
            uid=".../Titles.localized/Bumper:Opener.localized/Basic Title.localized/Basic Title.moti"/>
  </resources>
  <library>
    <event name="Example Event">
      <project name="Rasterized Title">
        <sequence format="project" duration="5s" tcStart="0s" tcFormat="NDF"
                  audioLayout="stereo" audioRate="48k">
          <spine>
            <asset-clip ref="camera" name="Camera" offset="0s" start="0s"
                        duration="5s" audioRole="dialogue">
              <title ref="basic" name="Headline" lane="1" offset="1s"
                     start="3600s" duration="2s" role="titles">
                <param name="Position"
                       key="9999/999166631/999166633/1/100/101" value="0 -400"/>
                <text><text-style ref="ts1">Rasterized by Bladeworks</text-style></text>
                <text-style-def id="ts1">
                  <text-style font="DejaVu Sans" fontSize="64"
                              fontFace="Bold" fontColor="1 1 1 1"
                              alignment="center"/>
                </text-style-def>
              </title>
              <caption name="Caption" lane="2" offset="2s" start="0s"
                       duration="1s" role="caption?captionFormat=ITT.en">
                <text><text-style ref="cs1">Editable caption</text-style></text>
                <text-style-def id="cs1">
                  <text-style font="DejaVu Sans" fontSize="42"
                              fontColor="1 1 1 1" backgroundColor="0 0 0 0.8"/>
                </text-style-def>
              </caption>
            </asset-clip>
          </spine>
        </sequence>
      </project>
    </event>
  </library>
</fcpxml>
```

## What Not To Author (and What To Author Instead)

Each fragment below is valid FCPXML — it passes the DTD and Final Cut would accept
it — but Bladeworks does not render it, and refuses it loudly at plan time rather
than guessing. Author the supported construct in its place. (The fragments are
kept DTD-valid on purpose: they are refused at render time, not at parse time.)

```xml
<!-- Don't author shear/skew under transform: adjust-transform has no shear DOF.
     To skew or distort, author the four-corner Distort (adjust-corners). -->
<adjust-transform position="0 0" scale="1 1">
  <param name="shear" value="0.2 0"/>
</adjust-transform>

<!-- Don't animate scale through zero: keep the sign stable and use the explicit
     mirror (a negative scale on the axis you want flipped) for a flip. -->
<adjust-transform>
  <param name="scale">
    <keyframeAnimation>
      <keyframe time="0s" value="1 1" curve="linear"/>
      <keyframe time="1s" value="-1 1" curve="linear"/>
    </keyframeAnimation>
  </param>
</adjust-transform>

<!-- Don't author a Hue/Saturation/Color/Luminosity blend: these cross-channel
     modes are not rendered. Author a supported RGB or matte blend mode instead. -->
<adjust-blend mode="hue"/>

<!-- Don't ramp speed with a smooth/eased timeMap: keep the timeMap
     piecewise-linear (interp="linear"), as in example 4. -->
<timeMap>
  <timept time="0s" value="0s" interp="smooth2"/>
  <timept time="2s" value="4s" interp="smooth2"/>
</timeMap>

<!-- Don't author an active conform-rate (scaleEnabled="1"): retime with a
     timeMap, or author the source at the project cadence. A passive
     conform-rate (scaleEnabled="0") is fine. -->
<conform-rate scaleEnabled="1" srcFrameRate="59.94" frameSampling="floor"/>

<!-- Don't author 5.1 / surround output: author audioLayout="stereo". -->
<sequence format="f" audioLayout="surround" audioRate="48k"> ... </sequence>
```

A few more constructs to author around, because they live in the resource table
rather than in a clip:

| Don't author | Author instead |
| --- | --- |
| An **alpha-carrying source** pixel format (alpha is exportable, not importable). | A source without an embedded alpha channel; key or matte inside the document. |
| A **non-square pixel aspect** format (`paspH`/`paspV` ≠ 1). | A square-pixel format. |
| A **non-Rec.709** / wide-gamut / HDR **project** colour space. | A Rec.709 SDR project (HDR *sources* are tone-mapped in on decode). |
| An **HDR delivery** request. | An SDR delivery — there is no HDR exit. |
| **Optical-flow / frame-blending** retime methods. | A `floor`-sampled `<timeMap>` (example 4). |

Finally, three constructs Bladeworks renders around and reports rather than
failing on — author the supported form to keep the report clean:

- An **unmatched transition** (a `uid` not in the registry) becomes an explicit
  hard cut. Author a transition whose `uid` is in
  [INVENTORY.md#transitions](INVENTORY.md#transitions).
- A **parameterized simple built-in effect** (e.g. Negative with an authored
  param) has its parameter dropped. Author the bare default, or author a
  parameter-driven effect the inventory supports (like Color Adjustments,
  example 7).
- A source whose **audio Bladeworks cannot decode** plays silent for that
  interval. Author a source in a decodable audio codec.

`--strict` turns each of these reports into a hard error, so a clean `--strict`
render is your proof the document is authored entirely inside the
exactly-reproduced surface.
