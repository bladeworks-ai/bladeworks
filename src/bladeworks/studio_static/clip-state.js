/**
 * Default inspector/control state and dotted-path helpers.
 *
 * Architecture map:
 * TimelineClip visual fields (transform / video / audio / keyframes)
 *   -> complete defaults so older snapshots still render
 *   -> getPath / setPath as the inspector ABI
 *   -> clampClipControls before a snapshot is accepted
 *
 * Main callers:
 * - magnetic-timeline normalize and updateClipPath
 * - fixtures when constructing synthetic clips
 * - templates/app when reading inspector values
 *
 * Why this exists:
 * The magnetic reducer owns editorial structure. This module owns the extra
 * FCP-style control surface that the polished UI needs without turning every
 * transform tweak into a one-off clip field.
 */
import { clamp } from "./ui.js";
export function defaultTransform() {
    return {
        enabled: true,
        x: 0,
        y: 0,
        scale: 1,
        scaleX: 1,
        scaleY: 1,
        rotation: 0,
        anchorX: 0,
        anchorY: 0,
        opacity: 1,
    };
}
export function defaultCrop() {
    return {
        enabled: false,
        type: "trim",
        left: 0,
        right: 0,
        top: 0,
        bottom: 0,
        kenStart: { x: 50, y: 50, width: 86, height: 86 },
        kenEnd: { x: 50, y: 50, width: 68, height: 68 },
        activeKenWindow: "start",
        easing: "ease-in-out",
    };
}
export function defaultDistort() {
    return {
        enabled: false,
        topLeftX: 0,
        topLeftY: 0,
        topRightX: 0,
        topRightY: 0,
        bottomLeftX: 0,
        bottomLeftY: 0,
        bottomRightX: 0,
        bottomRightY: 0,
    };
}
export function defaultPuck() {
    return { x: 0, y: 0 };
}
/** Convert an HTML RGB picker value without discarding stored transparency. */
export function colorFromHex(value, alpha = 1) {
    if (!/^#[0-9a-fA-F]{6}$/.test(value))
        throw new Error(`Invalid RGB color ${value}.`);
    return {
        red: Number.parseInt(value.slice(1, 3), 16) / 255,
        green: Number.parseInt(value.slice(3, 5), 16) / 255,
        blue: Number.parseInt(value.slice(5, 7), 16) / 255,
        alpha,
    };
}
export function defaultVideo() {
    return {
        blendEnabled: true,
        blendMode: "normal",
        crop: defaultCrop(),
        distort: defaultDistort(),
        stabilization: false,
        rollingShutter: false,
        spatialConform: "fit",
        colorConform: true,
        colorConformType: "automatic",
        color: {
            exposure: 0,
            brightness: 0,
            contrast: 0,
            saturation: 0,
            temperature: 0,
            tint: 0,
            highlights: 0,
            midtones: 0,
            shadows: 0,
            blackPoint: 0,
            hue: 0,
            pucks: {
                shadows: defaultPuck(),
                midtones: defaultPuck(),
                highlights: defaultPuck(),
            },
        },
    };
}
export function defaultAudio() {
    return {
        gainDb: 0,
        muted: false,
        solo: false,
        fadeIn: 0,
        fadeOut: 0,
        pan: 0,
        loudness: 0,
        noiseRemoval: 0,
    };
}
/** Reset the visible Volume controls without erasing separate enhancements. */
export function resetAudioVolume(audio) {
    const baseline = defaultAudio();
    return {
        ...audio,
        gainDb: baseline.gainDb,
        muted: baseline.muted,
        solo: baseline.solo,
        fadeIn: baseline.fadeIn,
        fadeOut: baseline.fadeOut,
        pan: baseline.pan,
    };
}
export function defaultKeyframes() {
    return {};
}
/**
 * Fill any missing inspector/control fields on a clip.
 *
 * Main callers:
 * - normalizeProject before lane allocation
 * - clip factories in fixtures
 */
export function normalizeClipVisualState(clip) {
    const transform = { ...defaultTransform(), ...clip.transform };
    const videoDefaults = defaultVideo();
    const video = {
        ...videoDefaults,
        ...clip.video,
        crop: {
            ...videoDefaults.crop,
            ...clip.video?.crop,
            kenStart: { ...videoDefaults.crop.kenStart, ...clip.video?.crop?.kenStart },
            kenEnd: { ...videoDefaults.crop.kenEnd, ...clip.video?.crop?.kenEnd },
        },
        distort: { ...videoDefaults.distort, ...clip.video?.distort },
        color: {
            ...videoDefaults.color,
            ...clip.video?.color,
            pucks: {
                ...videoDefaults.color.pucks,
                ...clip.video?.color?.pucks,
                shadows: { ...videoDefaults.color.pucks.shadows, ...clip.video?.color?.pucks?.shadows },
                midtones: { ...videoDefaults.color.pucks.midtones, ...clip.video?.color?.pucks?.midtones },
                highlights: { ...videoDefaults.color.pucks.highlights, ...clip.video?.color?.pucks?.highlights },
            },
        },
    };
    const audio = { ...defaultAudio(), ...clip.audio };
    return {
        ...clip,
        transform,
        video,
        audio,
        keyframes: Object.fromEntries(Object.entries({ ...defaultKeyframes(), ...clip.keyframes }).map(([path, frames]) => [
            path,
            frames.map((frame) => ({
                ...frame,
                time: { ...frame.time },
                value: typeof frame.value === "object" && frame.value !== null
                    ? structuredClone(frame.value)
                    : frame.value,
            })),
        ])),
        timeMap: clip.timeMap
            ? {
                ...clip.timeMap,
                points: clip.timeMap.points.map((point) => ({
                    ...point,
                    time: { ...point.time },
                    value: { ...point.value },
                })),
            }
            : null,
        textStyle: clip.textStyle ? structuredClone(clip.textStyle) : null,
        caption: clip.caption ? { ...clip.caption } : null,
        generatorColor: clip.generatorColor ? { ...clip.generatorColor } : null,
        markers: (clip.markers ?? []).map((marker) => ({ ...marker })),
        effects: (clip.effects ?? []).map((effect) => ({
            ...effect,
            parameters: structuredClone(effect.parameters),
            parameterNames: { ...effect.parameterNames },
            parameterKeyframes: structuredClone(effect.parameterKeyframes),
        })),
        effectStack: structuredClone(clip.effectStack ?? (clip.effects ?? []).map((effect) => ({ kind: "effect", effect }))),
    };
}
export function getPath(source, path) {
    let value = source;
    for (const key of path.split(".")) {
        if (!value || typeof value !== "object") {
            return undefined;
        }
        value = value[key];
    }
    return value;
}
export function setPath(target, path, value) {
    const keys = path.split(".");
    let cursor = target;
    for (const key of keys.slice(0, -1)) {
        const next = cursor[key];
        if (!next || typeof next !== "object") {
            cursor[key] = {};
        }
        else {
            cursor[key] = Array.isArray(next) ? [...next] : { ...next };
        }
        cursor = cursor[key];
    }
    const finalKey = keys.at(-1);
    if (!finalKey) {
        throw new Error(`Parameter path ${path} is empty.`);
    }
    cursor[finalKey] = value;
}
/**
 * Clamp inspector values to the ranges the FCP-style controls advertise.
 *
 * Main callers:
 * - updateClipPath / updateClip after a parameter write
 */
export function clampClipControls(clip) {
    const signedScale = (raw) => {
        const value = Number(raw);
        if (!Number.isFinite(value))
            return 1;
        if (value === 0)
            return 0.1;
        return Math.sign(value) * clamp(Math.abs(value), 0.1, 8);
    };
    const transform = {
        ...clip.transform,
        scale: signedScale(clip.transform.scale),
        scaleX: signedScale(clip.transform.scaleX),
        scaleY: signedScale(clip.transform.scaleY),
        opacity: clamp(Number(clip.transform.opacity) || 0, 0, 1),
        x: clamp(Number(clip.transform.x) || 0, -400, 400),
        y: clamp(Number(clip.transform.y) || 0, -400, 400),
        anchorX: clamp(Number(clip.transform.anchorX) || 0, -200, 200),
        anchorY: clamp(Number(clip.transform.anchorY) || 0, -200, 200),
    };
    const cropLeft = clamp(Number(clip.video.crop.left) || 0, 0, 99);
    const cropRight = clamp(Number(clip.video.crop.right) || 0, 0, 99 - cropLeft);
    const cropTop = clamp(Number(clip.video.crop.top) || 0, 0, 99);
    const cropBottom = clamp(Number(clip.video.crop.bottom) || 0, 0, 99 - cropTop);
    const crop = {
        ...clip.video.crop,
        left: cropLeft,
        right: cropRight,
        top: cropTop,
        bottom: cropBottom,
        kenStart: {
            x: clamp(Number.isFinite(Number(clip.video.crop.kenStart.x)) ? Number(clip.video.crop.kenStart.x) : 50, 0, 100),
            y: clamp(Number.isFinite(Number(clip.video.crop.kenStart.y)) ? Number(clip.video.crop.kenStart.y) : 50, 0, 100),
            width: clamp(Number(clip.video.crop.kenStart.width) || 10, 5, 100),
            height: clamp(Number(clip.video.crop.kenStart.height) || 10, 5, 100),
        },
        kenEnd: {
            x: clamp(Number.isFinite(Number(clip.video.crop.kenEnd.x)) ? Number(clip.video.crop.kenEnd.x) : 50, 0, 100),
            y: clamp(Number.isFinite(Number(clip.video.crop.kenEnd.y)) ? Number(clip.video.crop.kenEnd.y) : 50, 0, 100),
            width: clamp(Number(clip.video.crop.kenEnd.width) || 10, 5, 100),
            height: clamp(Number(clip.video.crop.kenEnd.height) || 10, 5, 100),
        },
    };
    const color = {
        ...clip.video.color,
        exposure: clamp(Number(clip.video.color.exposure) || 0, -100, 100),
        contrast: clamp(Number(clip.video.color.contrast) || 0, -100, 100),
        saturation: clamp(Number(clip.video.color.saturation) || 0, -100, 100),
        temperature: clamp(Number(clip.video.color.temperature) || 0, -100, 100),
        tint: clamp(Number(clip.video.color.tint) || 0, -100, 100),
        highlights: clamp(Number(clip.video.color.highlights) || 0, -100, 100),
        midtones: clamp(Number(clip.video.color.midtones) || 0, -100, 100),
        shadows: clamp(Number(clip.video.color.shadows) || 0, -100, 100),
        pucks: {
            shadows: {
                x: clamp(Number(clip.video.color.pucks.shadows.x) || 0, -100, 100),
                y: clamp(Number(clip.video.color.pucks.shadows.y) || 0, -100, 100),
            },
            midtones: {
                x: clamp(Number(clip.video.color.pucks.midtones.x) || 0, -100, 100),
                y: clamp(Number(clip.video.color.pucks.midtones.y) || 0, -100, 100),
            },
            highlights: {
                x: clamp(Number(clip.video.color.pucks.highlights.x) || 0, -100, 100),
                y: clamp(Number(clip.video.color.pucks.highlights.y) || 0, -100, 100),
            },
        },
    };
    const distort = {
        ...clip.video.distort,
        topLeftX: clamp(Number(clip.video.distort.topLeftX) || 0, -100, 100),
        topLeftY: clamp(Number(clip.video.distort.topLeftY) || 0, -100, 100),
        topRightX: clamp(Number(clip.video.distort.topRightX) || 0, -100, 100),
        topRightY: clamp(Number(clip.video.distort.topRightY) || 0, -100, 100),
        bottomLeftX: clamp(Number(clip.video.distort.bottomLeftX) || 0, -100, 100),
        bottomLeftY: clamp(Number(clip.video.distort.bottomLeftY) || 0, -100, 100),
        bottomRightX: clamp(Number(clip.video.distort.bottomRightX) || 0, -100, 100),
        bottomRightY: clamp(Number(clip.video.distort.bottomRightY) || 0, -100, 100),
    };
    const audio = {
        ...clip.audio,
        gainDb: clamp(Number(clip.audio.gainDb) || 0, -96, 24),
        pan: clamp(Number(clip.audio.pan) || 0, -1, 1),
        fadeIn: clamp(Number(clip.audio.fadeIn) || 0, 0, clip.duration),
        fadeOut: clamp(Number(clip.audio.fadeOut) || 0, 0, clip.duration),
        loudness: clamp(Number(clip.audio.loudness) || 0, 0, 100),
        noiseRemoval: clamp(Number(clip.audio.noiseRemoval) || 0, 0, 100),
    };
    return {
        ...clip,
        transform,
        video: { ...clip.video, crop, distort, color },
        audio,
    };
}
