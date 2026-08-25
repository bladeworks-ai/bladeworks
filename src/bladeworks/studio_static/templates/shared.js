/**
 * Tiny HTML helpers shared by more than one panel template.
 *
 * Panel-local helpers stay in that panel file. This module is only for
 * functions that media, inspector, viewer, and timeline all need.
 */
import { escapeHtml } from "../ui.js";
export function capitalize(value) {
    if (!value)
        return "";
    return value.charAt(0).toUpperCase() + value.slice(1).replaceAll("-", " ");
}
export function waveBars(count) {
    return Array.from({ length: count }, (_, i) => `<i style="height:${18 + ((i * 17) % 67)}%"></i>`).join("");
}
export function mediaVisualAttributes(asset, start, duration, thumbnails, audioBands) {
    // Titles, captions, and generators have no file to decode. Emitting
    // data-media-visual without a path made the loader stamp "Media unavailable".
    if (!asset?.sourcePath) {
        return "";
    }
    return `data-media-visual="true" data-media-path="${escapeHtml(asset.sourcePath)}" data-media-start="${Math.max(0, start)}" data-media-duration="${Math.min(300, Math.max(0.01, duration))}" data-media-thumbnails="${Math.min(12, thumbnails)}" data-media-audio-bands="${Math.min(256, audioBands)}"`;
}
export function assetVisualIndex(assetId) {
    if (!assetId)
        return 0;
    let hash = 0;
    for (const character of assetId)
        hash = (hash * 31 + character.charCodeAt(0)) % 11;
    return hash;
}
export function studioBuildId() {
    const build = globalThis.__BLADEWORKS_STUDIO_BUILD__;
    return build && /^[a-f0-9]{10}$/.test(build) ? build : "development";
}
export function createPopover(activePopover, hasLibrary) {
    if (activePopover !== "create")
        return "";
    return `<div class="create-popover" role="menu" aria-label="Create">
    <button data-action="new-event" role="menuitem" ${hasLibrary ? "" : "disabled"}>New Event</button>
    <button data-action="new-project" role="menuitem">New Project</button>
  </div>`;
}
