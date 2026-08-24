/** Shared formatting and DOM-safe rendering helpers for the editor UI. */
export function escapeHtml(value) {
    return value
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}
export function formatTime(seconds, includeFrames = false, fps = 30) {
    const safe = Math.max(0, seconds);
    const wholeSeconds = Math.floor(safe);
    const minutes = Math.floor(wholeSeconds / 60);
    const remainingSeconds = wholeSeconds % 60;
    if (!includeFrames) {
        return `${minutes}:${remainingSeconds.toString().padStart(2, "0")}`;
    }
    return formatTimecode(seconds, fps);
}
export function formatTimecode(seconds, fps = 30) {
    const clamped = Math.max(0, seconds);
    const hours = Math.floor(clamped / 3600);
    const minutes = Math.floor((clamped % 3600) / 60);
    const wholeSeconds = Math.floor(clamped % 60);
    const frames = Math.floor((clamped - Math.floor(clamped)) * fps + 1e-6);
    return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(wholeSeconds).padStart(2, "0")}:${String(frames).padStart(2, "0")}`;
}
export function formatDuration(seconds) {
    if (seconds < 10) {
        return `${seconds.toFixed(1)}s`;
    }
    return formatTime(seconds);
}
export function clamp(value, minimum, maximum) {
    return Math.min(maximum, Math.max(minimum, value));
}
/**
 * Return an asynchronous result only if the selection that requested it is
 * still current when the request finishes.
 *
 * Main callers:
 * - BladeworksEditorApp.toggleFavorite for complete-library Favorite saves.
 *
 * Why this exists:
 * Event switches and localhost saves can overlap. A slower result for Event A
 * must not replace the browser contents after the user has selected Event B.
 */
export async function resultForCurrentSelection(requestedSelectionId, request, currentSelectionId) {
    const result = await request();
    return currentSelectionId() === requestedSelectionId ? result : null;
}
/**
 * Return the horizontal timeline scale that keeps the complete Project visible.
 *
 * Main callers:
 * - BladeworksEditorApp.fitTimeline, including the Shift-Z shortcut.
 *
 * Why this exists:
 * Timeline fitting is a viewport calculation, not a manual-zoom operation. A
 * fixed zoom clamp makes short Projects occupy only a sliver of the viewport
 * and prevents long Projects from fitting at all.
 */
export function fitTimelinePixelsPerSecond(viewportWidth, projectDuration, endPadding = 80) {
    if (!Number.isFinite(viewportWidth) || viewportWidth <= 0) {
        throw new Error(`Timeline viewport width must be positive, received ${viewportWidth}.`);
    }
    if (!Number.isFinite(projectDuration) || projectDuration < 0) {
        throw new Error(`Project duration cannot be negative, received ${projectDuration}.`);
    }
    const usableWidth = Math.max(1, viewportWidth - Math.max(0, endPadding));
    const fittedDuration = projectDuration > 0 ? projectDuration : 1;
    return usableWidth / fittedDuration;
}
/** Return the last time the ruler may render without enlarging its canvas. */
export function timelineRulerDuration(projectDuration, pixelsPerSecond, canvasWidth) {
    if (pixelsPerSecond <= 0) {
        throw new Error(`Timeline pixels per second must be positive, received ${pixelsPerSecond}.`);
    }
    return Math.max(projectDuration, canvasWidth / pixelsPerSecond);
}
export function randomId(prefix) {
    return `${prefix}-${crypto.randomUUID()}`;
}
export function numberFromInput(element) {
    const value = Number.parseFloat(element.value);
    if (!Number.isFinite(value)) {
        throw new Error(`Input ${element.name || element.id || "value"} is not numeric.`);
    }
    return value;
}
export function requiredElement(root, selector) {
    const element = root.querySelector(selector);
    if (!element) {
        throw new Error(`Required UI element ${selector} is missing.`);
    }
    return element;
}
