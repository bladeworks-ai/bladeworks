/**
 * Viewer chrome: toolbar, onscreen transform/crop/distort, control strip, transport.
 *
 * Architecture map:
 * selected clip + viewer tool
 *   -> toolbar / canvas controls / transport HTML
 *   -> same dotted parameter paths as the inspector
 *
 * Main callers: BladeworksEditorApp.renderViewer
 */
import { itemHasKeyframeAt } from "../magnetic-timeline.js";
import { escapeHtml, formatTimecode } from "../ui.js";
import { icon } from "./icons.js";
import { capitalize } from "./shared.js";
export function viewerToolbarTemplate(state) {
    const itemName = state.selectedItem?.name ?? state.project.name;
    return `
    <div class="viewer-format">${state.project.width}×${state.project.height} | ${state.project.fps.toFixed(2)} fps, ${state.project.audioLayout === "mono" ? "Mono" : "Stereo"}</div>
    <div class="viewer-project-title">${icon("project")}<strong>${escapeHtml(itemName)}</strong></div>
    <div class="viewer-top-actions">
      <button class="viewer-popup-button ${state.activePopover === "viewer-zoom" ? "active" : ""}" data-action="viewer-zoom-menu">${state.viewerZoom === 0 ? "Fit" : `${state.viewerZoom}%`} ${icon("chevron-down")}</button>
      <button class="viewer-popup-button ${state.activePopover === "viewer-view" ? "active" : ""}" data-action="viewer-view-menu">View ${icon("chevron-down")}</button>
      ${viewerZoomPopover(state)}
      ${viewerViewPopover(state)}
    </div>
  `;
}
function viewerZoomPopover(state) {
    if (state.activePopover !== "viewer-zoom")
        return "";
    const values = [
        ["fit", "Fit"],
        ["25", "25%"],
        ["50", "50%"],
        ["100", "100%"],
        ["200", "200%"],
    ];
    return `<div class="viewer-popover viewer-zoom-popover">${values.map(([value, label]) => {
        const active = String(state.viewerZoom) === value || (value === "fit" && state.viewerZoom === 0);
        return `<button class="${active ? "active" : ""}" data-action="set-viewer-zoom" data-value="${value}"><span>${label}</span>${active ? icon("check") : ""}</button>`;
    }).join("")}</div>`;
}
function viewerViewPopover(state) {
    if (state.activePopover !== "viewer-view")
        return "";
    return `<div class="viewer-popover viewer-view-popover">
    <div class="viewer-popover-title">Bladeworks Preview</div>
    ${[["better-quality", "Better Quality", "720p"], ["better-performance", "Better Performance", "540p"], ["best-performance", "Best Performance", "480p"]].map(([value, label, resolution]) => `<button data-action="set-view-option" data-option="quality" data-value="${value}" class="check-row ${state.viewerView.quality === value ? "active" : ""}">${state.viewerView.quality === value ? icon("check") : '<span class="check-placeholder"></span>'}<span>${label}</span><span class="viewer-quality-resolution">${resolution}</span></button>`).join("")}
    <div class="viewer-popover-separator"></div>
    <label class="viewer-background-row"><span>Player Background</span><select data-action="viewer-background"><option value="black" ${state.viewerView.background === "black" ? "selected" : ""}>Black</option><option value="checker" ${state.viewerView.background === "checker" ? "selected" : ""}>Checkerboard</option><option value="white" ${state.viewerView.background === "white" ? "selected" : ""}>White</option></select></label>
  </div>`;
}
/**
 * Onscreen controls are rendered inside the program frame. They duplicate the
 * same transform/crop/distort paths used by the inspector and disappear during
 * playback, matching Final Cut's control lifecycle.
 */
export function canvasControlsTemplate(state) {
    const item = state.selectedItem;
    if (!item || item.kind === "audio" || state.playing || state.viewerTool === "none")
        return "";
    if (state.viewerTool === "transform" && !item.transform.enabled)
        return "";
    if (state.viewerTool === "crop") {
        if (!item.video.crop.enabled)
            return "";
        if (state.connectionMode === "localhost" && item.video.crop.type === "ken-burns")
            return "";
        return cropControlsTemplate(item);
    }
    if (state.viewerTool === "distort") {
        return item.video.distort.enabled ? distortControlsTemplate(item) : "";
    }
    return transformControlsTemplate(item);
}
function transformControlsTemplate(item) {
    return `
    <div class="transform-box" id="transform-box">
      ${["nw", "n", "ne", "e", "se", "s", "sw", "w"].map((handle) => `<span class="onscreen-handle handle-${handle}" data-canvas-handle="${handle}" title="Scale"></span>`).join("")}
      <span class="anchor-arm"></span>
      <span class="anchor-center" data-canvas-handle="move" title="Move"></span>
      <span class="rotation-arm"></span>
      <span class="rotation-handle" data-canvas-handle="rotate" title="Rotate"></span>
      <span class="onscreen-name">${escapeHtml(item.name)}</span>
    </div>
  `;
}
function cropControlsTemplate(item) {
    const crop = item.video.crop;
    if (crop.type === "ken-burns") {
        return `
      <div class="ken-burns-overlay">
        <div class="ken-window start ${crop.activeKenWindow === "start" ? "active" : ""}" data-ken-window="start"><span>Start</span>${cornerHandles("ken-start")}</div>
        <div class="ken-window end ${crop.activeKenWindow === "end" ? "active" : ""}" data-ken-window="end"><span>End</span>${cornerHandles("ken-end")}</div>
        <svg class="ken-arrow" viewBox="0 0 100 100" preserveAspectRatio="none"><path d="M31 61 L67 39"/><path d="M59 37 L67 39 L64 47"/></svg>
      </div>
    `;
    }
    return `
    <div class="crop-shade top"></div><div class="crop-shade right"></div><div class="crop-shade bottom"></div><div class="crop-shade left"></div>
    <div class="crop-window ${crop.type}">
      ${["nw", "n", "ne", "e", "se", "s", "sw", "w"].map((handle) => `<span class="onscreen-handle handle-${handle}" data-crop-handle="${handle}"></span>`).join("")}
      <span class="crop-move-target" data-crop-handle="move"></span>
      <span class="crop-label">${crop.type === "crop" ? "Crop" : "Trim"}</span>
    </div>
  `;
}
function cornerHandles(prefix) {
    return ["nw", "ne", "se", "sw"].map((handle) => `<i class="ken-handle ${handle}" data-ken-handle="${prefix}-${handle}"></i>`).join("");
}
function distortControlsTemplate(_item) {
    return `
    <svg class="distort-outline" id="distort-outline" viewBox="0 0 100 100" preserveAspectRatio="none"><path id="distort-path" d="M0 0 L100 0 L100 100 L0 100 Z"/></svg>
    ${["top-left", "top-right", "bottom-right", "bottom-left"].map((corner) => `<span class="distort-handle ${corner}" data-distort-handle="${corner}"></span>`).join("")}
    <span class="distort-center" data-distort-handle="move"></span>
  `;
}
export function viewerControlStripTemplate(state) {
    const item = state.selectedItem;
    const canEdit = Boolean(item && item.kind !== "audio");
    const currentPath = state.viewerTool === "crop"
        ? "video.crop.enabled"
        : state.viewerTool === "distort"
            ? "video.distort.enabled"
            : "transform.enabled";
    const keyed = item ? itemHasKeyframeAt(item, currentPath, state.currentTime, state.fps) : false;
    const cropModes = state.connectionMode === "localhost"
        ? ["trim", "crop"]
        : ["trim", "crop", "ken-burns"];
    const toolRows = [
        ["transform", "transform", "Transform", "⇧T"],
        ["crop", "crop", "Crop", "⇧C"],
        ["distort", "distort", "Distort", "⌥D"],
        ["none", "pointer", "Hide Controls", ""],
    ];
    return `
    <div class="viewer-control-left">
      <button class="viewer-tool-popup ${canEdit && state.viewerTool !== "none" ? "active" : ""}" data-action="toggle-viewer-tools" ${canEdit ? "" : "disabled"}>
        ${icon(state.viewerTool === "crop" ? "crop" : state.viewerTool === "distort" ? "distort" : "transform")}
        <span>${capitalize(state.viewerTool)}</span>${icon("chevron-down")}
      </button>
      ${state.viewerTool === "crop" && item ? `<div class="crop-mode-switch">
        ${cropModes.map((mode) => `<button class="${item.video.crop.type === mode ? "active" : ""}" data-action="crop-mode" data-mode="${mode}">${mode === "ken-burns" ? "Ken Burns" : capitalize(mode)}</button>`).join("")}
      </div>` : ""}
    </div>
    <div class="viewer-control-center">
      ${canEdit ? `<button class="viewer-small-button ${keyed ? "keyed" : ""}" data-action="toggle-keyframe" data-path="${currentPath}" title="Add or remove keyframe">${icon("diamond")}</button>` : ""}
      ${state.viewerTool === "transform" ? `<button class="viewer-small-button" data-action="toggle-overscan" title="Show overscan">${icon("overscan")}</button>` : ""}
      ${state.viewerTool === "crop" && item?.video.crop.type === "ken-burns" ? `
        <button class="viewer-small-button" data-action="swap-ken-burns" title="Swap Start and End">${icon("swap")}</button>
        <button class="viewer-small-button ${state.kenBurnsLoop ? "active" : ""}" data-action="loop-ken-burns" title="Loop preview">${icon("loop")}</button>
      ` : ""}
    </div>
    <div class="viewer-control-right">${canEdit && state.viewerTool !== "none" ? `<button class="done-button" data-action="viewer-done">Done</button>` : ""}</div>
    <div class="viewer-tools-popover" id="viewer-tools-popover" hidden>
      ${toolRows.map(([tool, ico, label, key]) => `<button data-action="viewer-tool" data-tool="${tool}" class="${state.viewerTool === tool ? "active" : ""}">${icon(ico)}<span>${label}</span><kbd>${key}</kbd></button>`).join("")}
    </div>
  `;
}
export function transportTemplate(state) {
    return `
    <div class="transport-left">
      ${state.connectionMode === "localhost" ? "" : `<button class="transport-button" data-action="viewer-tool" data-tool="transform" title="Transform (Shift-T)">${icon("frame")}</button>`}
      <button class="transport-button" data-action="viewer-color" title="Color controls">${icon("color")}</button>
    </div>
    <div class="transport-center">
      <button class="transport-button" data-action="step-back" title="Previous frame">${icon("step-back")}</button>
      <button class="transport-button play" data-action="toggle-play" title="Play / Pause (Space)">${icon(state.playing ? "pause" : "play")}</button>
      <button class="transport-button" data-action="step-forward" title="Next frame">${icon("step-forward")}</button>
      <div class="timecode-readout"><span id="transport-current">${formatTimecode(state.currentTime, 29.97)}</span></div>
    </div>
    <div class="transport-right"></div>
  `;
}
