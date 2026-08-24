/**
 * Exact, non-drag retime editing for Studio's portable linear time maps.
 *
 * Architecture map:
 * TimelineClip + playhead/segment command
 *   -> normalized linear time-map points
 *   -> one immutable TimeMapState plus the resulting clip duration
 *   -> ordinary updateClip commit through the runtime
 *
 * Product rules:
 * - New points carry `raw: ""` so the FCPXML codec quantizes them against the
 *   Project's exact frame duration.
 * - A hold is represented by adjacent output-time points with equal source
 *   values. Smooth ramps and optical-flow interpolation are never authored.
 * - Segment-rate edits keep source boundaries fixed and move this and every
 *   downstream output boundary, which gives precise ripple behavior.
 */
const EPSILON = 1e-6;
function authored(seconds) {
    return { seconds, raw: "" };
}
function clonePoint(point) {
    return { ...point, time: { ...point.time }, value: { ...point.value } };
}
export function editableTimeMap(clip) {
    if (clip.timeMap && clip.timeMap.points.length >= 2) {
        return {
            frameSampling: "floor",
            preservesPitch: clip.timeMap.preservesPitch ?? true,
            points: clip.timeMap.points.map(clonePoint).sort((left, right) => left.time.seconds - right.time.seconds),
        };
    }
    return {
        frameSampling: "floor",
        preservesPitch: true,
        points: [
            { time: authored(clip.sourceStart), value: authored(clip.sourceStart), interpolation: "linear" },
            {
                time: authored(clip.sourceStart + clip.duration),
                value: authored(clip.sourceStart + clip.duration),
                interpolation: "linear",
            },
        ],
    };
}
export function sourceValueAt(map, outputTime) {
    const points = map.points;
    if (outputTime <= points[0].time.seconds)
        return points[0].value.seconds;
    for (let index = 1; index < points.length; index += 1) {
        const left = points[index - 1];
        const right = points[index];
        if (outputTime <= right.time.seconds + EPSILON) {
            const span = right.time.seconds - left.time.seconds;
            if (span <= EPSILON)
                return right.value.seconds;
            const progress = Math.max(0, Math.min(1, (outputTime - left.time.seconds) / span));
            return left.value.seconds + (right.value.seconds - left.value.seconds) * progress;
        }
    }
    return points.at(-1).value.seconds;
}
/** Insert one linear segment boundary, matching Final Cut's Shift-B action. */
export function splitRetimeAt(clip, localTime) {
    const map = editableTimeMap(clip);
    const time = clip.sourceStart + Math.max(0, Math.min(clip.duration, localTime));
    if (map.points.some((point) => Math.abs(point.time.seconds - time) <= EPSILON)) {
        return { timeMap: map, duration: clip.duration };
    }
    const point = {
        time: authored(time),
        value: authored(sourceValueAt(map, time)),
        interpolation: "linear",
    };
    return {
        timeMap: { ...map, points: [...map.points.map(clonePoint), point].sort((a, b) => a.time.seconds - b.time.seconds) },
        duration: clip.duration,
    };
}
/** Add an exact-duration hold beginning at the playhead. */
export function addRetimeHold(clip, localTime, holdDuration) {
    if (!Number.isFinite(holdDuration) || holdDuration <= 0) {
        throw new Error("Hold duration must be positive.");
    }
    const split = splitRetimeAt(clip, localTime);
    const start = clip.sourceStart + Math.max(0, Math.min(clip.duration, localTime));
    const source = sourceValueAt(split.timeMap, start);
    const shifted = split.timeMap.points.map((point) => point.time.seconds > start + EPSILON
        ? { ...clonePoint(point), time: authored(point.time.seconds + holdDuration) }
        : clonePoint(point));
    shifted.push({ time: authored(start + holdDuration), value: authored(source), interpolation: "linear" });
    shifted.sort((left, right) => left.time.seconds - right.time.seconds);
    return {
        timeMap: { ...split.timeMap, points: shifted },
        duration: clip.duration + holdDuration,
    };
}
/** Freeze the complete clip at the source frame under the playhead. */
export function freezeRetime(clip, localTime) {
    const map = editableTimeMap(clip);
    const source = sourceValueAt(map, clip.sourceStart + Math.max(0, Math.min(clip.duration, localTime)));
    return {
        timeMap: {
            ...map,
            points: [
                { time: authored(clip.sourceStart), value: authored(source), interpolation: "linear" },
                { time: authored(clip.sourceStart + clip.duration), value: authored(source), interpolation: "linear" },
            ],
        },
        duration: clip.duration,
    };
}
/** Set one segment's exact rate while preserving all source boundaries. */
export function setRetimeSegmentRate(clip, segmentIndex, rate) {
    if (!Number.isFinite(rate) || rate <= 0)
        throw new Error("Segment rate must be positive.");
    const map = editableTimeMap(clip);
    const left = map.points[segmentIndex];
    const right = map.points[segmentIndex + 1];
    if (!left || !right)
        throw new Error(`Retime segment ${segmentIndex + 1} does not exist.`);
    const sourceSpan = Math.abs(right.value.seconds - left.value.seconds);
    if (sourceSpan <= EPSILON)
        throw new Error("A hold segment has no playback rate. Change its duration instead.");
    const oldDuration = right.time.seconds - left.time.seconds;
    const nextDuration = sourceSpan / rate;
    const delta = nextDuration - oldDuration;
    const points = map.points.map((point, index) => index <= segmentIndex
        ? clonePoint(point)
        : { ...clonePoint(point), time: authored(point.time.seconds + delta) });
    return {
        timeMap: { ...map, points },
        duration: Math.max(0, clip.duration + delta),
    };
}
/** Set a segment's output duration directly, including zero-source-span holds. */
export function setRetimeSegmentDuration(clip, segmentIndex, duration) {
    if (!Number.isFinite(duration) || duration <= 0)
        throw new Error("Segment duration must be positive.");
    const map = editableTimeMap(clip);
    const left = map.points[segmentIndex];
    const right = map.points[segmentIndex + 1];
    if (!left || !right)
        throw new Error(`Retime segment ${segmentIndex + 1} does not exist.`);
    const current = right.time.seconds - left.time.seconds;
    const delta = duration - current;
    const points = map.points.map((point, index) => index <= segmentIndex
        ? clonePoint(point)
        : { ...clonePoint(point), time: authored(point.time.seconds + delta) });
    return { timeMap: { ...map, points }, duration: Math.max(0, clip.duration + delta) };
}
