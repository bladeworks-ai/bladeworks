/**
 * Pure magnetic-timeline reducer.
 *
 * Architecture map:
 * ProjectSnapshot + one EditOperation
 *   -> operation-specific structural mutation
 *   -> normalize primary storyline into a contiguous sequence
 *   -> resolve connected-clip anchor positions
 *   -> deterministically allocate non-overlapping visual lanes
 *   -> increment revision and return a new snapshot
 *
 * Main callers:
 * - MockEditorRuntime.commitEdit
 * - reducer unit tests
 * - the future localhost FCPXML adapter's browser-contract tests
 *
 * Why this exists:
 * UI gesture handling is transient and platform-specific. Magnetic timeline
 * semantics must be deterministic, testable, and independent of DOM layout so
 * the same operation produces the same timeline regardless of input device.
 */
import { clampClipControls, getPath, normalizeClipVisualState, setPath, } from "./clip-state.js";
export const MIN_CLIP_DURATION = 0.25;
const EPSILON = 1e-6;
function roundTime(value) {
    return Math.round(value * 1_000_000) / 1_000_000;
}
function clamp(value, minimum, maximum) {
    return Math.min(maximum, Math.max(minimum, value));
}
function cloneClip(clip) {
    const visual = normalizeClipVisualState(clip);
    return {
        ...visual,
        colors: { ...visual.colors },
        transform: { ...visual.transform },
        video: structuredClone(visual.video),
        audio: { ...visual.audio },
        keyframes: Object.fromEntries(Object.entries(visual.keyframes).map(([path, frames]) => [path, structuredClone(frames)])),
        timeMap: visual.timeMap ? structuredClone(visual.timeMap) : null,
    };
}
function cloneConnectedClip(clip) {
    return {
        ...cloneClip(clip),
        anchorId: clip.anchorId,
        anchorOffset: clip.anchorOffset,
        lane: clip.lane,
    };
}
/**
 * Keep keyframes that belong to one surviving clip fragment and express their
 * times relative to that fragment's new start.
 *
 * Main callers:
 * - clipFragment while splitting or overwriting a clip.
 *
 * Why this exists:
 * Inspector, filter, and mask keyframe times are clip-local. Copying an entire
 * keyframe collection into both fragments duplicates animation and leaves
 * timestamps outside the new clip durations.
 */
function fragmentKeyframes(keyframes, fragmentStart, fragmentEnd, includeEnd) {
    return Object.fromEntries(Object.entries(keyframes).map(([path, frames]) => [
        path,
        frames
            .filter((frame) => (frame.time.seconds >= fragmentStart - EPSILON
            && (includeEnd
                ? frame.time.seconds <= fragmentEnd + EPSILON
                : frame.time.seconds < fragmentEnd - EPSILON)))
            .map((frame) => ({
            ...structuredClone(frame),
            time: { seconds: roundTime(frame.time.seconds - fragmentStart), raw: "" },
        })),
    ]));
}
function fragmentEffect(effect, fragmentStart, fragmentEnd, includeEnd) {
    return {
        ...structuredClone(effect),
        parameterKeyframes: fragmentKeyframes(effect.parameterKeyframes, fragmentStart, fragmentEnd, includeEnd),
    };
}
function fragmentEffectStackItem(item, fragmentStart, fragmentEnd, includeEnd) {
    if (item.kind === "effect") {
        return {
            kind: "effect",
            effect: fragmentEffect(item.effect, fragmentStart, fragmentEnd, includeEnd),
        };
    }
    return {
        kind: "masked-effect",
        maskedEffect: {
            ...structuredClone(item.maskedEffect),
            masks: item.maskedEffect.masks.map((mask) => ({
                ...structuredClone(mask),
                parameterKeyframes: fragmentKeyframes(mask.parameterKeyframes, fragmentStart, fragmentEnd, includeEnd),
            })),
            filters: item.maskedEffect.filters.map((effect) => (fragmentEffect(effect, fragmentStart, fragmentEnd, includeEnd))),
        },
    };
}
/** Return an editorially equivalent slice of one clip. */
function clipFragment(clip, fragmentStart, fragmentEnd) {
    const cloned = cloneClip(clip);
    const duration = roundTime(fragmentEnd - fragmentStart);
    const includeEnd = fragmentEnd >= clip.duration - EPSILON;
    return {
        ...cloned,
        sourceStart: roundTime(clip.sourceStart + fragmentStart),
        duration,
        audio: {
            ...cloned.audio,
            fadeIn: fragmentStart <= EPSILON ? cloned.audio.fadeIn : 0,
            fadeOut: fragmentEnd >= clip.duration - EPSILON ? cloned.audio.fadeOut : 0,
        },
        markers: cloned.markers
            .filter((marker) => (marker.offset >= fragmentStart - EPSILON
            && marker.offset < fragmentEnd - EPSILON))
            .map((marker) => ({ ...marker, offset: roundTime(marker.offset - fragmentStart) })),
        keyframes: fragmentKeyframes(cloned.keyframes, fragmentStart, fragmentEnd, includeEnd),
        effects: cloned.effects.map((effect) => (fragmentEffect(effect, fragmentStart, fragmentEnd, includeEnd))),
        effectStack: cloned.effectStack.map((item) => (fragmentEffectStackItem(item, fragmentStart, fragmentEnd, includeEnd))),
    };
}
function clipEnd(clip) {
    return clip.timelineStart + clip.duration;
}
function overlaps(left, right) {
    return left.timelineStart < clipEnd(right) - EPSILON && right.timelineStart < clipEnd(left) - EPSILON;
}
function normalizeSpine(spine) {
    let cursor = 0;
    return spine.map((clip) => {
        if (!Number.isFinite(clip.duration) || clip.duration <= 0) {
            throw new Error(`Clip ${clip.id} has a non-positive duration.`);
        }
        const duration = clip.duration;
        const normalized = {
            ...cloneClip(clip),
            duration: roundTime(duration),
            timelineStart: roundTime(cursor),
        };
        cursor += duration;
        return normalized;
    });
}
function laneDirection(clip) {
    return clip.kind === "audio" || clip.role === "connected-audio" ? -1 : 1;
}
function roleForClip(clip, inSpine) {
    if (inSpine) {
        return "storyline";
    }
    if (clip.kind === "audio") {
        return "connected-audio";
    }
    if (clip.kind === "title") {
        return "title";
    }
    return "connected-video";
}
/**
 * Assign the closest free lane on each side of the storyline.
 *
 * Main callers:
 * - normalizeProject after connected clips receive absolute positions.
 *
 * Why this exists:
 * FCPXML lanes express editorial ownership, while the browser needs stable
 * visual rows. Recomputing lanes from temporal overlap prevents stale row
 * numbers after trimming, ripple edits, or re-anchoring.
 */
function allocateConnectedLanes(clips) {
    const ordered = [...clips].sort((left, right) => {
        const startDifference = left.timelineStart - right.timelineStart;
        if (Math.abs(startDifference) > EPSILON) {
            return startDifference;
        }
        return left.id.localeCompare(right.id);
    });
    const placed = [];
    for (const clip of ordered) {
        const direction = laneDirection(clip);
        const authoredLane = Math.trunc(clip.lane);
        const authoredDirection = authoredLane < 0 ? -1 : 1;
        if (authoredLane !== 0 && authoredDirection === direction) {
            const collision = placed.some((placedClip) => placedClip.lane === authoredLane && overlaps(placedClip, clip));
            if (!collision) {
                placed.push({ ...cloneConnectedClip(clip), lane: authoredLane });
                continue;
            }
        }
        let laneMagnitude = 1;
        while (true) {
            const candidateLane = direction * laneMagnitude;
            const collision = placed.some((placedClip) => placedClip.lane === candidateLane && overlaps(placedClip, clip));
            if (!collision) {
                placed.push({ ...cloneConnectedClip(clip), lane: candidateLane });
                break;
            }
            laneMagnitude += 1;
        }
    }
    return placed;
}
export function normalizeProject(project) {
    const spine = normalizeSpine(project.spine).map((clip) => ({
        ...clip,
        role: "storyline",
    }));
    const anchors = new Map(spine.map((clip) => [clip.id, clip]));
    const storylineOrder = new Map(spine.map((clip, index) => [clip.id, index]));
    const positionedConnected = [];
    for (const clip of project.connected) {
        const anchor = anchors.get(clip.anchorId);
        if (!anchor) {
            continue;
        }
        // A connected clip stays anchored to its parent but may sit BEYOND the
        // parent's out-point -- Final Cut allows this, and the Python compiler
        // (the render oracle) never clamps it. Clamping to anchor.duration
        // collapsed every caption whose offset exceeded the (often short) anchor
        // clip onto the anchor's end, piling them on one timeline x. Only the
        // non-negative floor is kept, so a connected clip never renders before
        // t=0; the upper bound is removed to match the render.
        const anchorOffset = Math.max(0, clip.anchorOffset);
        if (!Number.isFinite(clip.duration) || clip.duration <= 0) {
            throw new Error(`Connected clip ${clip.id} has a non-positive duration.`);
        }
        positionedConnected.push({
            ...cloneConnectedClip(clip),
            role: roleForClip(clip, false),
            anchorOffset: roundTime(anchorOffset),
            timelineStart: roundTime(anchor.timelineStart + anchorOffset),
            duration: roundTime(clip.duration),
        });
    }
    const transitions = (project.transitions ?? []).filter((transition) => {
        const left = storylineOrder.get(transition.leftItemId);
        const right = storylineOrder.get(transition.rightItemId);
        return left !== undefined && right === left + 1;
    }).map((transition) => {
        if (!Number.isFinite(transition.duration) || transition.duration <= 0) {
            throw new Error(`Transition ${transition.id} has a non-positive duration.`);
        }
        return { ...transition, duration: roundTime(transition.duration) };
    });
    return {
        ...project,
        spine,
        connected: allocateConnectedLanes(positionedConnected),
        transitions,
    };
}
export function projectDuration(project) {
    const spineEnd = project.spine.length === 0 ? 0 : clipEnd(project.spine[project.spine.length - 1]);
    const connectedEnd = project.connected.reduce((maximum, clip) => Math.max(maximum, clipEnd(clip)), 0);
    return roundTime(Math.max(spineEnd, connectedEnd));
}
export function findTimelineClip(project, clipId) {
    return project.spine.find((clip) => clip.id === clipId)
        ?? project.connected.find((clip) => clip.id === clipId)
        ?? null;
}
function requireSpineIndex(spine, clipId) {
    const index = spine.findIndex((clip) => clip.id === clipId);
    if (index < 0) {
        throw new Error(`Primary-storyline clip ${clipId} does not exist.`);
    }
    return index;
}
function nearestAnchor(spine, absoluteTime) {
    if (spine.length === 0) {
        return null;
    }
    let bestClip = spine[0];
    let bestDistance = Number.POSITIVE_INFINITY;
    let bestOffset = 0;
    for (const clip of spine) {
        const offset = clamp(absoluteTime - clip.timelineStart, 0, clip.duration);
        const representedTime = clip.timelineStart + offset;
        const distance = Math.abs(absoluteTime - representedTime);
        if (distance < bestDistance - EPSILON) {
            bestClip = clip;
            bestDistance = distance;
            bestOffset = offset;
        }
    }
    return { clip: bestClip, offset: bestOffset };
}
function insertClip(project, clip, index) {
    const spine = [...project.spine.map(cloneClip)];
    spine.splice(clamp(Math.trunc(index), 0, spine.length), 0, cloneClip(clip));
    return { ...project, spine };
}
function reorderClip(project, clipId, toIndex) {
    const spine = [...project.spine.map(cloneClip)];
    const fromIndex = requireSpineIndex(spine, clipId);
    const [clip] = spine.splice(fromIndex, 1);
    if (!clip) {
        throw new Error(`Primary-storyline clip ${clipId} disappeared during reorder.`);
    }
    const target = clamp(Math.trunc(toIndex), 0, spine.length);
    spine.splice(target, 0, clip);
    return { ...project, spine };
}
function trimClip(project, clipId, edge, delta) {
    const update = (clip) => {
        if (clip.id !== clipId) {
            return cloneClip(clip);
        }
        if (edge === "start") {
            const maximumTrim = clip.duration - MIN_CLIP_DURATION;
            const sourceRangeStart = clip.sourceRangeStart ?? Math.min(0, clip.sourceStart);
            const effectiveDelta = clamp(delta, sourceRangeStart - clip.sourceStart, maximumTrim);
            return clipFragment(clip, effectiveDelta, clip.duration);
        }
        const minimumDelta = MIN_CLIP_DURATION - clip.duration;
        const maximumDelta = clip.sourceDuration === undefined
            ? Number.POSITIVE_INFINITY
            : (clip.sourceRangeStart ?? 0) + clip.sourceDuration - clip.sourceStart - clip.duration;
        const effectiveDelta = clamp(delta, minimumDelta, maximumDelta);
        return {
            ...cloneClip(clip),
            duration: roundTime(clip.duration + effectiveDelta),
        };
    };
    const inSpine = project.spine.some((clip) => clip.id === clipId);
    const inConnected = project.connected.some((clip) => clip.id === clipId);
    if (!inSpine && !inConnected) {
        throw new Error(`Clip ${clipId} does not exist.`);
    }
    return {
        ...project,
        spine: project.spine.map(update),
        connected: project.connected.map((clip) => ({
            ...cloneConnectedClip(update(clip)),
            anchorId: clip.anchorId,
            anchorOffset: clip.anchorOffset,
            lane: clip.lane,
        })),
    };
}
function splitClip(project, clipId, offset, trailingClipId) {
    const index = requireSpineIndex(project.spine, clipId);
    const original = project.spine[index];
    const splitOffset = clamp(offset, MIN_CLIP_DURATION, original.duration - MIN_CLIP_DURATION);
    if (splitOffset <= MIN_CLIP_DURATION - EPSILON || splitOffset >= original.duration - MIN_CLIP_DURATION + EPSILON) {
        throw new Error(`Clip ${clipId} is too short to split at ${offset.toFixed(2)}s.`);
    }
    const leading = clipFragment(original, 0, splitOffset);
    const trailing = {
        ...clipFragment(original, splitOffset, original.duration),
        id: trailingClipId,
        xmlOriginId: original.xmlOriginId ?? original.id,
        name: `${original.name} · 2`,
    };
    const spine = [...project.spine.map(cloneClip)];
    spine.splice(index, 1, leading, trailing);
    const connected = project.connected.map((clip) => {
        if (clip.anchorId !== clipId || clip.anchorOffset < splitOffset - EPSILON) {
            return cloneConnectedClip(clip);
        }
        return {
            ...cloneConnectedClip(clip),
            anchorId: trailingClipId,
            anchorOffset: roundTime(clip.anchorOffset - splitOffset),
        };
    });
    const transitions = (project.transitions ?? []).map((transition) => transition.leftItemId === clipId
        ? { ...transition, leftItemId: trailingClipId }
        : transition);
    return { ...project, spine, connected, transitions };
}
function deleteClips(project, clipIds) {
    const deleteSet = new Set(clipIds);
    const normalizedBeforeDelete = normalizeProject(project);
    const deletedAnchorTimes = new Map();
    for (const connected of normalizedBeforeDelete.connected) {
        if (deleteSet.has(connected.anchorId)) {
            deletedAnchorTimes.set(connected.id, connected.timelineStart);
        }
    }
    const spine = normalizedBeforeDelete.spine
        .filter((clip) => !deleteSet.has(clip.id))
        .map(cloneClip);
    const normalizedSurvivors = normalizeSpine(spine);
    const connected = [];
    for (const clip of normalizedBeforeDelete.connected) {
        if (deleteSet.has(clip.id)) {
            continue;
        }
        if (!deleteSet.has(clip.anchorId)) {
            connected.push(cloneConnectedClip(clip));
            continue;
        }
        const absoluteTime = deletedAnchorTimes.get(clip.id) ?? clip.timelineStart;
        const replacement = nearestAnchor(normalizedSurvivors, absoluteTime);
        if (!replacement) {
            continue;
        }
        connected.push({
            ...cloneConnectedClip(clip),
            anchorId: replacement.clip.id,
            anchorOffset: roundTime(replacement.offset),
        });
    }
    return { ...project, spine: normalizedSurvivors, connected };
}
function connectClip(project, clip) {
    if (!project.spine.some((anchor) => anchor.id === clip.anchorId)) {
        throw new Error(`Connected clip ${clip.id} references missing anchor ${clip.anchorId}.`);
    }
    return {
        ...project,
        connected: [...project.connected.map(cloneConnectedClip), cloneConnectedClip(clip)],
    };
}
/**
 * Lift a primary-storyline clip onto a connected lane and close the hole.
 *
 * What it does: take the clip off the spine so later clips ripple left, then
 * attach that same clip to whichever spine clip now sits at `timelineStart`.
 *
 * Main callers: timeline drag preview and drop when the pointer leaves the
 * storyline shelf into the connected-video or connected-audio zone.
 *
 * Why this exists: Final Cut's Select-tool vertical drag is a ripple
 * conversion. Lift from Storyline is a different command because it leaves a
 * Gap and does not shorten the project. Preview and commit must share this
 * one operation so the hover layout is the layout that drop writes.
 */
function spineToConnected(project, clipId, timelineStart, lane) {
    const index = requireSpineIndex(project.spine, clipId);
    const original = project.spine[index];
    if (project.connected.some((clip) => clip.anchorId === clipId)) {
        throw new Error("Move or remove clips connected to this storyline clip first.");
    }
    const remaining = deleteClips(project, [clipId]);
    if (remaining.spine.length === 0) {
        throw new Error("A connected clip cannot attach without a primary-storyline clip.");
    }
    const anchor = nearestAnchor(remaining.spine, timelineStart);
    if (!anchor) {
        throw new Error("A connected clip cannot attach without a primary-storyline clip.");
    }
    const connected = {
        ...cloneClip(original),
        role: roleForClip(original, false),
        anchorId: anchor.clip.id,
        anchorOffset: roundTime(anchor.offset),
        lane: lane < 0 ? -1 : 1,
    };
    return connectClip(remaining, connected);
}
/**
 * Drop a connected clip onto the primary storyline as an insert.
 *
 * What it does: take the clip off its connected lane, then splice it into the
 * spine at `toIndex` so later storyline clips ripple right.
 *
 * Main callers: timeline drag preview and drop when a connected clip is dragged
 * onto the storyline shelf.
 *
 * Why this exists: Final Cut's Select-tool reverse of spine-to-connected is an
 * insert at an edit point, not Overwrite to Primary Storyline (which would
 * cover existing spine media and keep duration). Preview and commit share this
 * one operation so the hover gap is the gap that drop writes.
 */
function connectedToSpine(project, clipId, toIndex) {
    const original = project.connected.find((clip) => clip.id === clipId);
    if (!original) {
        throw new Error(`Connected clip ${clipId} does not exist.`);
    }
    const connected = project.connected
        .filter((clip) => clip.id !== clipId)
        .map(cloneConnectedClip);
    const cloned = cloneClip(original);
    const { anchorId: _anchorId, anchorOffset: _anchorOffset, lane: _lane, ...storyline } = cloned;
    return insertClip({ ...project, connected }, storyline, toIndex);
}
function replaceClip(project, clipId, replacement) {
    const index = requireSpineIndex(project.spine, clipId);
    const original = project.spine[index];
    const spine = project.spine.map((clip, clipIndex) => {
        if (clipIndex !== index) {
            return cloneClip(clip);
        }
        return {
            ...cloneClip(replacement),
            id: original.id,
            timelineStart: original.timelineStart,
            duration: original.duration,
        };
    });
    return { ...project, spine };
}
function updateClip(project, clipId, patch) {
    let found = false;
    const applyPatch = (clip) => {
        if (clip.id !== clipId) {
            return cloneClip(clip);
        }
        found = true;
        const effectStack = patch.effectStack === undefined
            ? structuredClone(clip.effectStack)
            : structuredClone(patch.effectStack);
        const next = clampClipControls({
            ...cloneClip(clip),
            ...patch,
            effectStack,
            effects: effectStack.flatMap((entry) => entry.kind === "effect" ? [entry.effect] : []),
            transform: patch.transform ? { ...clip.transform, ...patch.transform } : { ...clip.transform },
            video: patch.video ? structuredClone({ ...clip.video, ...patch.video }) : structuredClone(clip.video),
            audio: patch.audio ? { ...clip.audio, ...patch.audio } : { ...clip.audio },
            keyframes: patch.keyframes
                ? Object.fromEntries(Object.entries(patch.keyframes).map(([path, frames]) => [path, structuredClone(frames)]))
                : Object.fromEntries(Object.entries(clip.keyframes).map(([path, frames]) => [path, structuredClone(frames)])),
            timeMap: patch.timeMap === undefined
                ? (clip.timeMap ? structuredClone(clip.timeMap) : null)
                : (patch.timeMap ? structuredClone(patch.timeMap) : null),
            duration: roundTime(patch.duration === undefined
                ? clip.duration
                : Math.max(MIN_CLIP_DURATION, patch.duration)),
            sourceStart: roundTime(patch.sourceStart ?? clip.sourceStart),
        });
        return next;
    };
    const spine = project.spine.map(applyPatch);
    const connected = project.connected.map((clip) => {
        const patched = applyPatch(clip);
        return {
            ...cloneConnectedClip(patched),
            anchorId: clip.anchorId,
            anchorOffset: clip.anchorOffset,
            lane: clip.lane,
        };
    });
    if (!found) {
        throw new Error(`Clip ${clipId} does not exist.`);
    }
    return { ...project, spine, connected };
}
function updateClipPath(project, clipId, path, value) {
    const clip = findTimelineClip(project, clipId);
    if (!clip) {
        throw new Error(`Clip ${clipId} does not exist.`);
    }
    const draft = cloneClip(clip);
    setPath(draft, path, value);
    return updateClip(project, clipId, {
        transform: draft.transform,
        video: draft.video,
        audio: draft.audio,
        keyframes: draft.keyframes,
        timeMap: draft.timeMap,
        name: draft.name,
        text: draft.text,
    });
}
function toggleKeyframe(project, clipId, path, timelineTime) {
    const clip = findTimelineClip(project, clipId);
    if (!clip) {
        throw new Error(`Clip ${clipId} does not exist.`);
    }
    const localTime = clamp(timelineTime - clip.timelineStart, 0, clip.duration);
    const existing = [...(clip.keyframes[path] ?? [])];
    const threshold = 1 / Math.max(1, project.fps);
    const index = existing.findIndex((frame) => Math.abs(frame.time.seconds - localTime) < threshold);
    if (index === -1) {
        existing.push({
            time: { seconds: localTime, raw: "" },
            value: keyframeValue(clip, path),
            interpolation: "linear",
        });
    }
    else {
        existing.splice(index, 1);
    }
    existing.sort((left, right) => left.time.seconds - right.time.seconds);
    return updateClip(project, clipId, {
        keyframes: { ...clip.keyframes, [path]: existing },
    });
}
function clearKeyframes(project, clipId, path) {
    const clip = findTimelineClip(project, clipId);
    if (!clip) {
        throw new Error(`Clip ${clipId} does not exist.`);
    }
    return updateClip(project, clipId, {
        keyframes: { ...clip.keyframes, [path]: [] },
    });
}
/**
 * Capture the exact current parameter value for a new keyframe.
 *
 * Main callers:
 * - toggleKeyframe after the user clicks a diamond control
 *
 * Why this exists:
 * FCPXML stores position, scale, anchor, and corner controls as vector values.
 * Keeping those controls under one canonical path prevents the UI from writing
 * half of a vector and silently losing the other coordinate.
 */
function keyframeValue(clip, path) {
    const pointValues = {
        "transform.position": { x: clip.transform.x, y: clip.transform.y },
        "transform.scale": {
            x: clip.transform.scale * clip.transform.scaleX,
            y: clip.transform.scale * clip.transform.scaleY,
        },
        "transform.anchor": { x: clip.transform.anchorX, y: clip.transform.anchorY },
        "video.distort.topleft": {
            x: clip.video.distort.topLeftX,
            y: clip.video.distort.topLeftY,
        },
        "video.distort.topright": {
            x: clip.video.distort.topRightX,
            y: clip.video.distort.topRightY,
        },
        "video.distort.bottomleft": {
            x: clip.video.distort.bottomLeftX,
            y: clip.video.distort.bottomLeftY,
        },
        "video.distort.bottomright": {
            x: clip.video.distort.bottomRightX,
            y: clip.video.distort.bottomRightY,
        },
    };
    const point = pointValues[path];
    if (point) {
        return point;
    }
    const scalarValues = {
        "transform.rotation": clip.transform.rotation,
        "transform.opacity": clip.transform.opacity,
        "video.crop.left": clip.video.crop.left,
        "video.crop.right": clip.video.crop.right,
        "video.crop.top": clip.video.crop.top,
        "video.crop.bottom": clip.video.crop.bottom,
        "audio.gainDb": clip.audio.gainDb,
        "audio.pan": clip.audio.pan,
    };
    const scalar = scalarValues[path];
    if (scalar !== undefined) {
        return scalar;
    }
    throw new Error(`Keyframes are not supported for parameter path ${path}.`);
}
function moveConnected(project, clipId, timelineStart) {
    const clip = project.connected.find((candidate) => candidate.id === clipId);
    if (!clip) {
        throw new Error(`Connected clip ${clipId} does not exist.`);
    }
    const maxStart = Math.max(0, projectDuration(project) - MIN_CLIP_DURATION);
    const clampedStart = clamp(timelineStart, 0, maxStart);
    const replacement = nearestAnchor(project.spine, clampedStart);
    if (!replacement) {
        throw new Error("A connected clip cannot move without a primary-storyline anchor.");
    }
    const connected = project.connected.map((candidate) => {
        if (candidate.id !== clipId) {
            return cloneConnectedClip(candidate);
        }
        return {
            ...cloneConnectedClip(candidate),
            anchorId: replacement.clip.id,
            anchorOffset: roundTime(replacement.offset),
        };
    });
    return { ...project, connected };
}
/**
 * Re-anchor connected clips after the primary storyline is rebuilt in place.
 *
 * Connected clips that still reference a surviving anchor keep their local
 * offset; clips whose anchor was fully removed re-anchor to the storyline clip
 * nearest their former absolute time. This is the shared settling step for edits
 * that can delete or replace spine clips without a downstream ripple (overwrite).
 *
 * Main callers:
 * - overwriteClip
 */
function settleConnected(preNormalized, newSpine, splitTails = new Map()) {
    const survivors = normalizeSpine(newSpine);
    const survivorIds = new Set(survivors.map((clip) => clip.id));
    const connected = [];
    for (const clip of preNormalized.connected) {
        const tailId = splitTails.get(clip.anchorId);
        const tail = tailId ? survivors.find((candidate) => candidate.id === tailId) : undefined;
        if (tail && clip.timelineStart >= tail.timelineStart - EPSILON) {
            connected.push({
                ...cloneConnectedClip(clip),
                anchorId: tail.id,
                anchorOffset: roundTime(clip.timelineStart - tail.timelineStart),
            });
            continue;
        }
        if (survivorIds.has(clip.anchorId)) {
            connected.push(cloneConnectedClip(clip));
            continue;
        }
        const replacement = nearestAnchor(survivors, clip.timelineStart);
        if (!replacement) {
            continue;
        }
        connected.push({
            ...cloneConnectedClip(clip),
            anchorId: replacement.clip.id,
            anchorOffset: roundTime(replacement.offset),
        });
    }
    return { spine: survivors, connected };
}
/**
 * Overwrite: write `incoming` over the storyline starting at `startTime` for its
 * own duration, WITHOUT rippling downstream clips.
 *
 * In natural language: walk the contiguous storyline; keep every clip fully
 * before the write window untouched; drop the slice covered by the window,
 * trimming clips that straddle either edge (the tail piece keeps the source
 * media beyond the window); and drop the incoming clip into the hole. Because we
 * remove exactly `duration` seconds and insert `duration` seconds, the storyline
 * stays contiguous and every downstream clip keeps its absolute time — this is
 * the defining difference from `insert`, which ripples.
 *
 * The write point is clamped to the storyline end, so overwrite never has to
 * synthesize a gap clip in this prototype.
 */
function overwriteClip(project, incoming, startTime) {
    const spineEnd = project.spine.length === 0
        ? 0
        : clipEnd(project.spine[project.spine.length - 1]);
    const duration = roundTime(Math.max(MIN_CLIP_DURATION, incoming.duration));
    const start = roundTime(clamp(startTime, 0, spineEnd));
    const end = roundTime(start + duration);
    const occupiedIds = new Set([...project.spine, ...project.connected, incoming].map((clip) => clip.id));
    const allocateTailId = (clipId) => {
        const base = `${clipId}~ow`;
        let candidate = base;
        let suffix = 2;
        while (occupiedIds.has(candidate)) {
            candidate = `${base}~${suffix}`;
            suffix += 1;
        }
        occupiedIds.add(candidate);
        return candidate;
    };
    const written = {
        ...cloneClip(incoming),
        role: "storyline",
        duration,
    };
    const result = [];
    const splitTails = new Map();
    let inserted = false;
    const insertHere = () => {
        if (!inserted) {
            result.push(written);
            inserted = true;
        }
    };
    for (const clip of project.spine) {
        const clipStart = clip.timelineStart;
        const clipFinish = clipEnd(clip);
        if (clipFinish <= start + EPSILON) {
            result.push(cloneClip(clip));
            continue;
        }
        if (clipStart >= end - EPSILON) {
            insertHere();
            result.push(cloneClip(clip));
            continue;
        }
        // This clip straddles the write window. Keep any left remainder, then the
        // incoming clip, then any right remainder carved from the same source media.
        // A clip that only loses its head (no left remainder) keeps its id; only a
        // clip split into two pieces needs a fresh id for its trailing half.
        const hasLeft = clipStart < start - EPSILON;
        const hasRight = clipFinish > end + EPSILON;
        if (hasLeft) {
            result.push(clipFragment(clip, 0, start - clipStart));
        }
        insertHere();
        if (hasRight) {
            const tailId = hasLeft ? allocateTailId(clip.id) : clip.id;
            if (hasLeft) {
                splitTails.set(clip.id, tailId);
            }
            result.push({
                ...clipFragment(clip, end - clipStart, clip.duration),
                id: tailId,
            });
        }
    }
    insertHere();
    const settled = settleConnected(project, result, splitTails);
    const transitions = project.transitions.map((transition) => ({
        ...transition,
        leftItemId: splitTails.get(transition.leftItemId) ?? transition.leftItemId,
    }));
    return { ...project, spine: settled.spine, connected: settled.connected, transitions };
}
/**
 * Roll: move the shared edit point between `clipId` and its right-hand neighbor.
 * A positive delta lengthens the left clip's outgoing edge and shortens the
 * right clip's incoming edge (advancing its source start) by the same amount, so
 * the storyline's total length never changes.
 */
function rollTrim(project, clipId, delta) {
    const index = requireSpineIndex(project.spine, clipId);
    if (index >= project.spine.length - 1) {
        throw new Error(`Clip ${clipId} has no edit to its right to roll.`);
    }
    const left = project.spine[index];
    const right = project.spine[index + 1];
    const rightRangeStart = right.sourceRangeStart ?? 0;
    const leftRangeEnd = left.sourceDuration === undefined
        ? Number.POSITIVE_INFINITY
        : (left.sourceRangeStart ?? 0) + left.sourceDuration;
    const minDelta = Math.max(MIN_CLIP_DURATION - left.duration, rightRangeStart - right.sourceStart);
    const maxDelta = Math.min(right.duration - MIN_CLIP_DURATION, leftRangeEnd - left.sourceStart - left.duration);
    const effective = clamp(delta, minDelta, maxDelta);
    const spine = project.spine.map((clip, clipIndex) => {
        if (clipIndex === index) {
            return { ...cloneClip(clip), duration: roundTime(left.duration + effective) };
        }
        if (clipIndex === index + 1) {
            return {
                ...cloneClip(clip),
                sourceStart: roundTime(right.sourceStart + effective),
                duration: roundTime(right.duration - effective),
            };
        }
        return cloneClip(clip);
    });
    return { ...project, spine };
}
/**
 * Slip: shift a clip's visible source window by `delta` while its timeline
 * position and duration stay fixed. Neighbors and total length are untouched.
 * Works on both storyline and connected clips.
 */
function slipClip(project, clipId, delta) {
    const clip = findTimelineClip(project, clipId);
    if (!clip) {
        throw new Error(`Clip ${clipId} does not exist.`);
    }
    const minimum = clip.sourceRangeStart ?? 0;
    const maximum = clip.sourceDuration === undefined
        ? Number.POSITIVE_INFINITY
        : Math.max(minimum, minimum + clip.sourceDuration - clip.duration);
    const nextSourceStart = roundTime(clamp(clip.sourceStart + delta, minimum, maximum));
    const effectiveDelta = nextSourceStart - clip.sourceStart;
    const timeMap = clip.timeMap ? {
        ...clip.timeMap,
        points: clip.timeMap.points.map((point) => ({
            ...point,
            value: { ...point.value, seconds: roundTime(point.value.seconds + effectiveDelta) },
        })),
    } : null;
    return updateClip(project, clipId, { sourceStart: nextSourceStart, timeMap });
}
/**
 * Slide: move `clipId` along the storyline by `delta`, absorbing the shift into
 * its two neighbors. The previous clip grows (or shrinks) by delta and the next
 * clip's incoming edge moves by the same amount, so the moved clip's duration
 * and the storyline's total length are both preserved.
 */
function slideClip(project, clipId, delta) {
    const index = requireSpineIndex(project.spine, clipId);
    if (index <= 0 || index >= project.spine.length - 1) {
        throw new Error(`Clip ${clipId} needs a neighbor on each side to slide.`);
    }
    const previous = project.spine[index - 1];
    const next = project.spine[index + 1];
    const nextRangeStart = next.sourceRangeStart ?? 0;
    const previousRangeEnd = previous.sourceDuration === undefined
        ? Number.POSITIVE_INFINITY
        : (previous.sourceRangeStart ?? 0) + previous.sourceDuration;
    const minDelta = Math.max(MIN_CLIP_DURATION - previous.duration, nextRangeStart - next.sourceStart);
    const maxDelta = Math.min(next.duration - MIN_CLIP_DURATION, previousRangeEnd - previous.sourceStart - previous.duration);
    const effective = clamp(delta, minDelta, maxDelta);
    const spine = project.spine.map((clip, clipIndex) => {
        if (clipIndex === index - 1) {
            return { ...cloneClip(clip), duration: roundTime(previous.duration + effective) };
        }
        if (clipIndex === index + 1) {
            return {
                ...cloneClip(clip),
                sourceStart: roundTime(next.sourceStart + effective),
                duration: roundTime(next.duration - effective),
            };
        }
        return cloneClip(clip);
    });
    return { ...project, spine };
}
/**
 * Apply a pure transformation to one clip found anywhere (storyline or
 * connected), preserving the connected-clip anchor/lane fields. Marker and
 * effect operations funnel through here so they share one lookup + not-found
 * guard instead of duplicating the spine/connected split.
 */
function mutateClip(project, clipId, mutate) {
    let found = false;
    const applyTo = (clip) => {
        if (clip.id !== clipId) {
            return cloneClip(clip);
        }
        found = true;
        return cloneClip(mutate(cloneClip(clip)));
    };
    const spine = project.spine.map(applyTo);
    const connected = project.connected.map((clip) => ({
        ...cloneConnectedClip(applyTo(clip)),
        anchorId: clip.anchorId,
        anchorOffset: clip.anchorOffset,
        lane: clip.lane,
    }));
    if (!found) {
        throw new Error(`Clip ${clipId} does not exist.`);
    }
    return { ...project, spine, connected };
}
function addMarker(project, clipId, marker) {
    return mutateClip(project, clipId, (clip) => {
        const offset = roundTime(clamp(marker.offset, 0, clip.duration));
        const markers = [...clip.markers, { ...marker, offset }].sort((a, b) => a.offset - b.offset);
        return { ...clip, markers };
    });
}
function updateMarker(project, clipId, markerId, patch) {
    return mutateClip(project, clipId, (clip) => {
        let touched = false;
        const markers = clip.markers
            .map((marker) => {
            if (marker.id !== markerId) {
                return marker;
            }
            touched = true;
            const offset = patch.offset === undefined
                ? marker.offset
                : roundTime(clamp(patch.offset, 0, clip.duration));
            return { ...marker, ...patch, offset };
        })
            .sort((a, b) => a.offset - b.offset);
        if (!touched) {
            throw new Error(`Marker ${markerId} does not exist on clip ${clipId}.`);
        }
        return { ...clip, markers };
    });
}
function deleteMarker(project, clipId, markerId) {
    return mutateClip(project, clipId, (clip) => ({
        ...clip,
        markers: clip.markers.filter((marker) => marker.id !== markerId),
    }));
}
function addEffect(project, clipId, effect) {
    return mutateClip(project, clipId, (clip) => ({
        ...clip,
        effects: [...clip.effects, { ...effect }],
    }));
}
function updateEffect(project, clipId, effectId, patch) {
    return mutateClip(project, clipId, (clip) => {
        let touched = false;
        const effects = clip.effects.map((effect) => {
            if (effect.id !== effectId) {
                return effect;
            }
            touched = true;
            return {
                ...effect,
                ...patch,
                parameters: patch.parameters ? structuredClone(patch.parameters) : effect.parameters,
                parameterKeyframes: patch.parameterKeyframes
                    ? structuredClone(patch.parameterKeyframes)
                    : effect.parameterKeyframes,
            };
        });
        if (!touched) {
            throw new Error(`Effect ${effectId} does not exist on clip ${clipId}.`);
        }
        return { ...clip, effects };
    });
}
function removeEffect(project, clipId, effectId) {
    return mutateClip(project, clipId, (clip) => ({
        ...clip,
        effects: clip.effects.filter((effect) => effect.id !== effectId),
    }));
}
function reorderEffect(project, clipId, effectId, toIndex) {
    return mutateClip(project, clipId, (clip) => {
        const effects = [...clip.effects];
        const fromIndex = effects.findIndex((effect) => effect.id === effectId);
        if (fromIndex < 0) {
            throw new Error(`Effect ${effectId} does not exist on clip ${clipId}.`);
        }
        const [moved] = effects.splice(fromIndex, 1);
        effects.splice(clamp(Math.trunc(toIndex), 0, effects.length), 0, moved);
        return { ...clip, effects };
    });
}
function addTransition(project, transition) {
    // One transition per edit point: drop any existing transition on the same
    // left/right pair before adding the new one. normalizeProject drops any
    // transition that no longer bridges adjacent storyline clips.
    const filtered = (project.transitions ?? []).filter((existing) => existing.leftItemId !== transition.leftItemId);
    return { ...project, transitions: [...filtered, { ...transition }] };
}
function updateTransition(project, transitionId, patch) {
    let touched = false;
    const transitions = (project.transitions ?? []).map((transition) => {
        if (transition.id !== transitionId) {
            return transition;
        }
        touched = true;
        return { ...transition, ...patch };
    });
    if (!touched) {
        throw new Error(`Transition ${transitionId} does not exist.`);
    }
    return { ...project, transitions };
}
function removeTransition(project, transitionId) {
    return {
        ...project,
        transitions: (project.transitions ?? []).filter((transition) => transition.id !== transitionId),
    };
}
export function applyEdit(project, operation) {
    const normalized = normalizeProject(project);
    let edited;
    switch (operation.type) {
        case "insert":
            edited = insertClip(normalized, operation.clip, operation.index);
            break;
        case "reorder":
            edited = reorderClip(normalized, operation.clipId, operation.toIndex);
            break;
        case "trim":
            edited = trimClip(normalized, operation.clipId, operation.edge, operation.delta);
            break;
        case "split":
            edited = splitClip(normalized, operation.clipId, operation.offset, operation.trailingClipId);
            break;
        case "delete":
            edited = deleteClips(normalized, operation.clipIds);
            break;
        case "connect":
            edited = connectClip(normalized, operation.clip);
            break;
        case "spineToConnected":
            edited = spineToConnected(normalized, operation.clipId, operation.timelineStart, operation.lane);
            break;
        case "connectedToSpine":
            edited = connectedToSpine(normalized, operation.clipId, operation.toIndex);
            break;
        case "replace":
            edited = replaceClip(normalized, operation.clipId, operation.replacement);
            break;
        case "updateClip":
            edited = updateClip(normalized, operation.clipId, operation.patch);
            break;
        case "updateClipPath":
            edited = updateClipPath(normalized, operation.clipId, operation.path, operation.value);
            break;
        case "toggleKeyframe":
            edited = toggleKeyframe(normalized, operation.clipId, operation.path, operation.time);
            break;
        case "clearKeyframes":
            edited = clearKeyframes(normalized, operation.clipId, operation.path);
            break;
        case "moveConnected":
            edited = moveConnected(normalized, operation.clipId, operation.timelineStart);
            break;
        case "overwrite":
            edited = overwriteClip(normalized, operation.clip, operation.timelineStart);
            break;
        case "rollTrim":
            edited = rollTrim(normalized, operation.clipId, operation.delta);
            break;
        case "slip":
            edited = slipClip(normalized, operation.clipId, operation.delta);
            break;
        case "slide":
            edited = slideClip(normalized, operation.clipId, operation.delta);
            break;
        case "addMarker":
            edited = addMarker(normalized, operation.clipId, operation.marker);
            break;
        case "updateMarker":
            edited = updateMarker(normalized, operation.clipId, operation.markerId, operation.patch);
            break;
        case "deleteMarker":
            edited = deleteMarker(normalized, operation.clipId, operation.markerId);
            break;
        case "addEffect":
            edited = addEffect(normalized, operation.clipId, operation.effect);
            break;
        case "updateEffect":
            edited = updateEffect(normalized, operation.clipId, operation.effectId, operation.patch);
            break;
        case "removeEffect":
            edited = removeEffect(normalized, operation.clipId, operation.effectId);
            break;
        case "reorderEffect":
            edited = reorderEffect(normalized, operation.clipId, operation.effectId, operation.toIndex);
            break;
        case "addTransition":
            edited = addTransition(normalized, operation.transition);
            break;
        case "updateTransition":
            edited = updateTransition(normalized, operation.transitionId, operation.patch);
            break;
        case "removeTransition":
            edited = removeTransition(normalized, operation.transitionId);
            break;
        default: {
            const exhaustive = operation;
            throw new Error(`Unsupported edit operation: ${JSON.stringify(exhaustive)}`);
        }
    }
    return normalizeProject({ ...edited, revision: normalized.revision + 1 });
}
export function clipAtTime(project, time) {
    return project.spine.find((clip) => time >= clip.timelineStart - EPSILON && time < clipEnd(clip) - EPSILON) ?? project.spine[project.spine.length - 1] ?? null;
}
export function insertionIndexAtTime(project, time) {
    const normalized = normalizeProject(project);
    for (let index = 0; index < normalized.spine.length; index += 1) {
        const clip = normalized.spine[index];
        if (time < clip.timelineStart + clip.duration / 2) {
            return index;
        }
    }
    return normalized.spine.length;
}
export function itemHasKeyframeAt(item, path, timelineTime, fps) {
    const local = timelineTime - item.timelineStart;
    return (item.keyframes[path] ?? []).some((frame) => Math.abs(frame.time.seconds - local) < 1 / Math.max(1, fps));
}
export function itemParameter(item, path) {
    return getPath(item, path);
}
/**
 * Return human-readable invariant violations. Runtime adapters should reject
 * commits with any entries rather than silently repairing unexpected state.
 */
export function validateProject(project) {
    const errors = [];
    let cursor = 0;
    const ids = new Set();
    for (const item of project.spine) {
        if (ids.has(item.id)) {
            errors.push(`duplicate item id: ${item.id}`);
        }
        ids.add(item.id);
        if (Math.abs(item.timelineStart - cursor) > EPSILON) {
            errors.push(`storyline item ${item.id} starts at ${item.timelineStart}, expected ${cursor}`);
        }
        if (!Number.isFinite(item.duration) || item.duration <= 0) {
            errors.push(`item ${item.id} has a non-positive duration`);
        }
        cursor += item.duration;
    }
    const anchorIds = new Set(project.spine.map((item) => item.id));
    const order = new Map(project.spine.map((item, index) => [item.id, index]));
    for (const transition of project.transitions ?? []) {
        const left = order.get(transition.leftItemId);
        const right = order.get(transition.rightItemId);
        if (left === undefined || right !== left + 1) {
            errors.push(`transition ${transition.id} does not bridge adjacent storyline clips`);
        }
        if (!(transition.duration > 0)) {
            errors.push(`transition ${transition.id} has an invalid duration`);
        }
    }
    for (const item of project.connected) {
        if (ids.has(item.id)) {
            errors.push(`duplicate item id: ${item.id}`);
        }
        ids.add(item.id);
        if (!Number.isFinite(item.duration) || item.duration <= 0) {
            errors.push(`connected item ${item.id} has a non-positive duration`);
        }
        if (!item.anchorId || !anchorIds.has(item.anchorId)) {
            errors.push(`connected item ${item.id} has no valid anchor`);
        }
        if (item.role === "connected-audio" && item.lane >= 0) {
            errors.push(`audio item ${item.id} must use a negative lane`);
        }
        if (item.role !== "connected-audio" && item.lane <= 0) {
            errors.push(`visual item ${item.id} must use a positive lane`);
        }
    }
    return errors;
}
