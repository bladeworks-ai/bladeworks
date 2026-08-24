/**
 * Inspector: Video / Color / Audio / Info plus effect stack and masks.
 *
 * Architecture map:
 * selected clip or transition + InspectorTab
 *   -> inspectorTemplate HTML
 *   -> data-parameter-path and data-action controls
 *
 * Main callers: BladeworksEditorApp.renderInspector
 */
import { itemHasKeyframeAt } from "../magnetic-timeline.js";
import { supportDetail, supportLabel } from "../capability-ui.js";
import { escapeHtml, formatTimecode } from "../ui.js";
import { maskKeyframeKey, maskUiValues } from "../mask-ui.js";
import { icon } from "./icons.js";
import { capitalize, waveBars } from "./shared.js";
export function inspectorTemplate(state) {
    const item = state.selectedItem;
    const transition = state.selectedTransition;
    const inspectorTabs = [
        ["video", "film"],
        ["color", "color"],
        ["audio", "speaker"],
        ["info", "info"],
    ];
    return `
    <div class="inspector-tabs">
      ${inspectorTabs.map(([tab, ico]) => `<button class="inspector-tab ${state.inspectorTab === tab ? "active" : ""}" data-action="inspector-tab" data-tab="${tab}" title="${capitalize(tab)} Inspector">${icon(ico)}</button>`).join("")}
      <span class="inspector-clip-name">${escapeHtml(state.selectionCount > 1 ? `${state.selectionCount} Clips Selected` : transition?.name ?? item?.name ?? "Nothing Selected")}</span>
      <span class="inspector-source-time" id="inspector-source-time">${item ? formatTimecode(item.sourceStart + Math.max(0, state.currentTime - item.timelineStart), state.fps) : ""}</span>
    </div>
    <div class="inspector-scroll">
      ${transition ? transitionInspector(transition, state) : item ? inspectorBody(item, state) : `<div class="empty-inspector">${icon("inspector")}<strong>Select a timeline clip</strong><span>Video, audio, and metadata controls appear here.</span></div>`}
    </div>
    ${state.connectionMode === "mock" ? '<div class="inspector-footer"><button data-action="save-effects-preset">Save Effects Preset</button></div>' : ""}
  `;
}
function inspectorBody(item, state) {
    if (state.inspectorTab === "audio")
        return audioInspector(item, state);
    if (state.inspectorTab === "color")
        return colorInspector(item, state);
    if (state.inspectorTab === "info")
        return infoInspector(item, state.fps);
    return videoInspector(item, state);
}
/**
 * The applied-effect stack for the selected clip. Effects render top-to-bottom
 * in array order (matching FCP). Each row exposes enable, reorder, an intensity
 * amount, and remove; the "+" opens the Effects browser. The pixel result is a
 * labeled mock, but placement and ordering are real reducer state.
 */
function effectsSection(item, state, visibleEffects = item.effects) {
    const isFilteredProjection = visibleEffects !== item.effects;
    const rows = isFilteredProjection
        ? visibleEffects.map((effect, index) => effectRow(effect, index, visibleEffects.length, state, false)).join("")
        : item.effectStack.map((entry, index) => entry.kind === "effect"
            ? effectRow(entry.effect, index, item.effectStack.length, state, state.connectionMode !== "localhost")
            : state.connectionMode === "localhost"
                ? `<div class="effect-row disabled"><div class="effect-meta"><strong>Masked Effect</strong><small>Preserved, not editable in Studio</small></div></div>`
                : maskedEffectRow(entry.maskedEffect, index, item.effectStack.length, state)).join("");
    const body = `
    ${rows || '<p class="effect-empty">No effects. Add one from the Effects browser.</p>'}
    <button class="effect-add" data-action="open-effects-browser">${icon("plus")}<span>Add Effect</span></button>
  `;
    const enabled = item.effectStack.every((entry) => entry.kind === "effect" ? entry.effect.enabled : entry.maskedEffect.enabled);
    return section(`Effects${rows ? ` (${isFilteredProjection ? visibleEffects.length : item.effectStack.length})` : ""}`, "effects-stack", enabled, "", body, Boolean(rows), "toggle-effects-section");
}
function effectRow(effect, index, total, state, allowMask) {
    const capability = state.capabilities.effects.find((candidate) => candidate.handler === effect.handler || candidate.resource.uid === effect.resourceUid || candidate.name === effect.name);
    const parameters = capability?.parameters ?? [];
    const maskKinds = state.capabilities.mechanics.find((mechanic) => mechanic.id === "masks")?.sourceKinds ?? [];
    return `<div class="effect-row ${effect.enabled ? "" : "disabled"}" data-effect-id="${escapeHtml(effect.id)}">
    <label class="effect-toggle"><input type="checkbox" data-action="toggle-effect" data-effect-id="${escapeHtml(effect.id)}" ${effect.enabled ? "checked" : ""}></label>
    <div class="effect-meta"><strong>${escapeHtml(effect.name)}</strong><small>${escapeHtml(effect.category)}</small></div>
    <div class="effect-actions">
      <button data-action="reorder-effect" data-effect-id="${escapeHtml(effect.id)}" data-dir="-1" ${index === 0 ? "disabled" : ""} title="Move up">▲</button>
      <button data-action="reorder-effect" data-effect-id="${escapeHtml(effect.id)}" data-dir="1" ${index === total - 1 ? "disabled" : ""} title="Move down">▼</button>
      <button data-action="remove-effect" data-effect-id="${escapeHtml(effect.id)}" title="Remove effect">${icon("trash")}</button>
    </div>
    <div class="capability-controls">
      ${parameters.length ? capabilityParameterRows("effect", effect.id, parameters, effect.parameters) : ""}
      ${state.connectionMode === "mock" ? capability?.notes.map((note) => `<p class="capability-note">${escapeHtml(note)}</p>`).join("") ?? "" : ""}
      ${allowMask && maskKinds.length ? `<div class="mask-add-row"><span>Add Effect Mask</span>${maskKinds.map((kind) => `<button data-action="wrap-effect-mask" data-effect-id="${escapeHtml(effect.id)}" data-mask-kind="${kind.id}">${escapeHtml(kind.id[0].toUpperCase() + kind.id.slice(1))}</button>`).join("")}</div>` : ""}
    </div>
  </div>`;
}
function maskedEffectRow(group, index, total, state) {
    const maskMechanic = state.capabilities.mechanics.find((mechanic) => mechanic.id === "masks");
    const sourceKinds = maskMechanic?.sourceKinds ?? [];
    const blendModes = (maskMechanic?.blendModes ?? []);
    const outside = group.filters[1];
    return `<div class="masked-effect-group ${group.enabled ? "" : "disabled"}" data-mask-group-id="${escapeHtml(group.id)}">
    <header class="masked-effect-header">
      <label><input type="checkbox" data-action="toggle-mask-group" data-group-id="${escapeHtml(group.id)}" ${group.enabled ? "checked" : ""}><strong>Masked Effect</strong></label>
      ${maskMechanic?.invert ? `<label><input type="checkbox" data-action="toggle-mask-invert" data-group-id="${escapeHtml(group.id)}" ${group.inverted ? "checked" : ""}>Invert</label>` : ""}
      <span class="support-badge support-${maskMechanic?.support ?? "approximate"}">${supportLabel(maskMechanic?.support ?? "approximate")}</span>
      <button data-action="reorder-mask-group" data-group-id="${escapeHtml(group.id)}" data-dir="-1" ${index === 0 ? "disabled" : ""}>▲</button>
      <button data-action="reorder-mask-group" data-group-id="${escapeHtml(group.id)}" data-dir="1" ${index === total - 1 ? "disabled" : ""}>▼</button>
      <button data-action="remove-mask-group" data-group-id="${escapeHtml(group.id)}">${icon("trash")}</button>
    </header>
    <div class="masked-filter-stack">
      ${maskedFilterRow(group.id, group.filters[0], 0, "Inside", state)}
      ${outside ? maskedFilterRow(group.id, outside, 1, "Outside", state) : `<label class="outside-effect-picker"><span>Outside</span><select data-action="set-mask-outside-effect" data-group-id="${escapeHtml(group.id)}"><option value="">Add outside effect…</option>${state.capabilities.effects.filter((effect) => effect.authorable && effect.support !== "unsupported").map((effect) => `<option value="${escapeHtml(effect.id)}">${escapeHtml(effect.name)}</option>`).join("")}</select></label>`}
      ${outside ? `<div class="masked-filter-actions"><button data-action="swap-mask-filters" data-group-id="${escapeHtml(group.id)}">Swap Inside / Outside</button><button data-action="remove-mask-outside-effect" data-group-id="${escapeHtml(group.id)}">Remove Outside</button></div>` : ""}
    </div>
    <div class="mask-source-list">${group.masks.map((mask, maskIndex) => maskSourceRow(group.id, mask, maskIndex, group.masks.length, sourceKinds, blendModes, state)).join("")}</div>
    <div class="mask-add-row"><span>Add Mask</span>${sourceKinds.map((kind) => `<button data-action="add-mask-source" data-group-id="${escapeHtml(group.id)}" data-mask-kind="${kind.id}">${escapeHtml(kind.id[0].toUpperCase() + kind.id.slice(1))}</button>`).join("")}</div>
    <p class="capability-note">Tracked, Magnetic, Auto, opaque isolation, and ML masks are rejected rather than flattened.</p>
  </div>`;
}
function maskedFilterRow(groupId, effect, filterIndex, label, state) {
    const capability = state.capabilities.effects.find((candidate) => candidate.resource.uid === effect.resourceUid || candidate.handler === effect.handler);
    return `<section class="masked-filter"><header><span>${label}</span><strong>${escapeHtml(effect.name)}</strong></header>${capability?.parameters.length
        ? capabilityParameterRows("masked-filter", `${groupId}:${filterIndex}`, capability.parameters, effect.parameters)
        : `<p class="capability-note">${supportDetail(capability?.support ?? effect.support)}</p>`}</section>`;
}
function maskSourceRow(groupId, mask, index, total, sourceKinds, blendModes, state) {
    const source = sourceKinds.find((candidate) => candidate.id === mask.kind);
    const values = maskUiValues(mask);
    return `<section class="mask-source" data-mask-id="${escapeHtml(mask.id)}">
    <header>
      <label><input type="checkbox" data-action="toggle-mask-source" data-group-id="${escapeHtml(groupId)}" data-mask-id="${escapeHtml(mask.id)}" ${mask.enabled ? "checked" : ""}><strong>${escapeHtml(mask.name)}</strong></label>
      <select data-action="set-mask-blend" data-group-id="${escapeHtml(groupId)}" data-mask-id="${escapeHtml(mask.id)}">${blendModes.map((mode) => `<option value="${mode}" ${mask.blendMode === mode ? "selected" : ""}>${capitalize(mode)}</option>`).join("")}</select>
      <button data-action="reorder-mask-source" data-group-id="${escapeHtml(groupId)}" data-mask-id="${escapeHtml(mask.id)}" data-dir="-1" ${index === 0 ? "disabled" : ""}>▲</button>
      <button data-action="reorder-mask-source" data-group-id="${escapeHtml(groupId)}" data-mask-id="${escapeHtml(mask.id)}" data-dir="1" ${index === total - 1 ? "disabled" : ""}>▼</button>
      <button data-action="remove-mask-source" data-group-id="${escapeHtml(groupId)}" data-mask-id="${escapeHtml(mask.id)}">${icon("trash")}</button>
    </header>
    <div class="mask-parameters">${source?.parameters.map((parameter) => maskParameterRow(groupId, mask, parameter, values[parameter.key], state)).join("") ?? ""}</div>
    ${source?.notes ? `<p class="capability-note">${escapeHtml(source.notes)}</p>` : ""}
  </section>`;
}
function maskParameterRow(groupId, mask, parameter, value, state) {
    const attributes = `data-action="set-mask-parameter" data-group-id="${escapeHtml(groupId)}" data-mask-id="${escapeHtml(mask.id)}" data-mask-key="${escapeHtml(parameter.key)}"`;
    let control;
    if (parameter.type === "point_list") {
        const points = Array.isArray(value) ? value : [];
        control = `<div class="mask-point-list">${points.map((point, pointIndex) => `<label><small>${pointIndex + 1}</small><input type="number" step="0.1" value="${point.x}" data-point-index="${pointIndex}" data-component="x" ${attributes}><input type="number" step="0.1" value="${point.y}" data-point-index="${pointIndex}" data-component="y" ${attributes}><button data-action="remove-mask-point" data-group-id="${escapeHtml(groupId)}" data-mask-id="${escapeHtml(mask.id)}" data-point-index="${pointIndex}" ${points.length <= (parameter.minimumItems ?? 3) ? "disabled" : ""}>−</button></label>`).join("")}<button class="mask-point-add" data-action="add-mask-point" data-group-id="${escapeHtml(groupId)}" data-mask-id="${escapeHtml(mask.id)}" ${points.length >= (parameter.maximumItems ?? 64) ? "disabled" : ""}>+ Add Point</button></div>`;
    }
    else if (parameter.type === "point") {
        const point = typeof value === "object" && value !== null && "x" in value ? value : { x: 0, y: 0 };
        control = `<div class="capability-compound"><label><small>X</small><input type="number" min="${parameter.min ?? ""}" max="${parameter.max ?? ""}" step="0.1" value="${point.x}" data-component="x" ${attributes}></label><label><small>Y</small><input type="number" min="${parameter.min ?? ""}" max="${parameter.max ?? ""}" step="0.1" value="${point.y}" data-component="y" ${attributes}></label></div>`;
    }
    else if (parameter.type === "color") {
        control = `<input type="color" value="${colorHex(value)}" ${attributes}>`;
    }
    else {
        control = `<input type="number" ${parameter.min !== undefined ? `min="${parameter.min}"` : ""} ${parameter.max !== undefined ? `max="${parameter.max}"` : ""} step="0.01" value="${Number(value ?? parameter.default ?? 0)}" ${attributes}>`;
    }
    const key = maskKeyframeKey(mask, parameter.key);
    const frames = mask.parameterKeyframes[key] ?? [];
    const local = state.currentTime - (state.selectedItem?.timelineStart ?? 0);
    const active = frames.some((frame) => Math.abs(frame.time.seconds - local) < 1 / Math.max(1, state.fps));
    return `<div class="mask-parameter-row"><label><span>${escapeHtml(parameter.name)}</span><small>${escapeHtml(parameter.units)}</small></label>${control}${parameter.animatable ? `<div class="mask-keyframe-controls"><button data-action="previous-mask-keyframe" data-group-id="${escapeHtml(groupId)}" data-mask-id="${escapeHtml(mask.id)}" data-mask-key="${escapeHtml(parameter.key)}" ${frames.length ? "" : "disabled"}>‹</button><button class="keyframe-diamond ${active ? "active" : ""}" data-action="toggle-mask-keyframe" data-group-id="${escapeHtml(groupId)}" data-mask-id="${escapeHtml(mask.id)}" data-mask-key="${escapeHtml(parameter.key)}">${icon("diamond")}</button><button data-action="next-mask-keyframe" data-group-id="${escapeHtml(groupId)}" data-mask-id="${escapeHtml(mask.id)}" data-mask-key="${escapeHtml(parameter.key)}" ${frames.length ? "" : "disabled"}>›</button><button data-action="clear-mask-keyframes" data-group-id="${escapeHtml(groupId)}" data-mask-id="${escapeHtml(mask.id)}" data-mask-key="${escapeHtml(parameter.key)}" ${frames.length ? "" : "disabled"}>×</button></div>` : '<span class="mask-static">Static</span>'}</div>`;
}
function videoInspector(item, state) {
    const fullState = state;
    const localhost = fullState.connectionMode === "localhost";
    return `
    ${containerInspector(item, fullState)}
    ${generatedClipInspector(item)}
    ${effectsSection(item, fullState)}
    ${section("Compositing", "compositing", item.video.blendEnabled, "video.blendEnabled", `
      ${selectRow("Blend Mode", "video.blendMode", item.video.blendMode, fullState.capabilities.blendModes
        .filter((mode) => mode.authorable && mode.support !== "unsupported")
        .map((mode) => [mode.fcpxmlValue, mode.name]))}
      ${sliderRow("Opacity", "transform.opacity", item.transform.opacity, 0, 1, .01, "%", state, 100)}
    `)}
    ${section("Transform", "transform", item.transform.enabled, "transform.enabled", `
      ${xyRow("Position", "transform.x", item.transform.x, "transform.y", item.transform.y, "%", state, "transform.position")}
      ${knobRow("Rotation", "transform.rotation", item.transform.rotation, -180, 180, 0.1, "°", state)}
      ${sliderRow("Scale (All)", "transform.scale", item.transform.scale, .1, 4, .01, "%", state, 100, "transform.scale")}
      ${xyRow("Anchor", "transform.anchorX", item.transform.anchorX, "transform.anchorY", item.transform.anchorY, "%", state, "transform.anchor")}
    `)}
    ${section("Crop", "crop", item.video.crop.enabled, "video.crop.enabled", `
      ${localhost && item.video.crop.type === "ken-burns"
        ? `<div class="parameter-row"><label>Type</label><strong>Ken Burns</strong><span class="row-spacer"></span></div>
          <p class="capability-note">Ken Burns is preserved but not editable in the live preview.</p>`
        : `${selectRow("Type", "video.crop.type", item.video.crop.type, localhost ? [["trim", "Trim"], ["crop", "Crop"]] : [["trim", "Trim"], ["crop", "Crop"], ["ken-burns", "Ken Burns"]])}
      ${item.video.crop.type === "ken-burns" ? `
        ${segmentedRow("Edit", "video.crop.activeKenWindow", item.video.crop.activeKenWindow, [["start", "Start"], ["end", "End"]])}
        ${selectRow("Motion", "video.crop.easing", item.video.crop.easing, [["linear", "Linear (supported)"]])}
        <button class="inline-action" data-action="swap-ken-burns">Swap Start and End</button>
      ` : `
        ${sliderRow("Left", "video.crop.left", item.video.crop.left, 0, 100, 1, "%", state, 1, null)}
        ${sliderRow("Right", "video.crop.right", item.video.crop.right, 0, 100, 1, "%", state, 1, null)}
        ${sliderRow("Top", "video.crop.top", item.video.crop.top, 0, 100, 1, "%", state, 1, null)}
        ${sliderRow("Bottom", "video.crop.bottom", item.video.crop.bottom, 0, 100, 1, "%", state, 1, null)}
      `}
      `}
    `)}
    ${section("Distort", "distort", item.video.distort.enabled, "video.distort.enabled", `
      ${xyRow("Bottom Left", "video.distort.bottomLeftX", item.video.distort.bottomLeftX, "video.distort.bottomLeftY", item.video.distort.bottomLeftY, "%", state, "video.distort.bottomleft")}
      ${xyRow("Bottom Right", "video.distort.bottomRightX", item.video.distort.bottomRightX, "video.distort.bottomRightY", item.video.distort.bottomRightY, "%", state, "video.distort.bottomright")}
      ${xyRow("Top Right", "video.distort.topRightX", item.video.distort.topRightX, "video.distort.topRightY", item.video.distort.topRightY, "%", state, "video.distort.topright")}
      ${xyRow("Top Left", "video.distort.topLeftX", item.video.distort.topLeftX, "video.distort.topLeftY", item.video.distort.topLeftY, "%", state, "video.distort.topleft")}
    `)}
    ${section("Spatial Conform", "spatial", item.video.spatialConform !== "none", "", selectRow("Type", "video.spatialConform", item.video.spatialConform, [["fit", "Fit"], ["fill", "Fill"], ["none", "None"]]), true, "toggle-spatial-conform")}
    ${retimeSection(item)}
    ${localhost ? "" : `
      ${toggleSectionRow("Stabilization", "video.stabilization", item.video.stabilization)}
      ${toggleSectionRow("Rolling Shutter", "video.rollingShutter", item.video.rollingShutter)}
      ${section("Color Conform", "color-conform", item.video.colorConform, "video.colorConform", selectRow("Type", "video.colorConformType", item.video.colorConformType, [["automatic", "Automatic"], ["sdr", "SDR"], ["hdr", "HDR"]]))}
      <div class="tracker-row"><span>Trackers</span><button data-action="add-tracker">+</button></div>
    `}
  `;
}
function containerInspector(item, state) {
    const container = item.container;
    const root = state.rootProject;
    if (!container || !root)
        return "";
    const scopeName = (scopeId) => root.scopes?.[scopeId]?.name ?? scopeId;
    let body = `<button class="inline-action scope-open-inspector" data-action="enter-clip-scope" data-item-id="${escapeHtml(item.id)}">Open Timeline</button>`;
    if (container.kind === "multicam") {
        const angles = Object.entries(container.angleScopeIds);
        const options = (selected) => angles.map(([angleId, scopeId]) => `<option value="${escapeHtml(angleId)}" ${angleId === selected ? "selected" : ""}>${escapeHtml(scopeName(scopeId))}</option>`).join("");
        body += `<label class="inspector-row"><span>Video Angle</span><select data-action="set-multicam-video-angle" data-item-id="${escapeHtml(item.id)}">${options(container.videoAngleId)}</select></label><label class="inspector-row"><span>Audio Angle</span><select data-action="set-multicam-audio-angle" data-item-id="${escapeHtml(item.id)}">${options(container.audioAngleId)}</select></label>`;
    }
    else if (container.kind === "audition") {
        body += `<label class="inspector-row"><span>Active Pick</span><select data-action="set-audition-choice" data-item-id="${escapeHtml(item.id)}">${container.choiceScopeIds.map((scopeId) => `<option value="${escapeHtml(scopeId)}" ${scopeId === container.activeChoiceId ? "selected" : ""}>${escapeHtml(scopeName(scopeId))}</option>`).join("")}</select></label><p class="capability-note">All alternatives remain in the FCPXML; changing the pick only moves the active selector.</p>`;
    }
    else if (container.kind === "sync") {
        body += `<div class="sync-source-controls">${container.sources.map((source) => `<div class="sync-source-row"><strong>${escapeHtml(source.sourceId)}</strong><input type="text" value="${escapeHtml(source.role ?? "")}" data-action="set-sync-source-role" data-item-id="${escapeHtml(item.id)}" data-source-id="${escapeHtml(source.sourceId)}" ${source.role === null ? "disabled" : ""}><label><input type="checkbox" data-action="set-sync-source-enabled" data-item-id="${escapeHtml(item.id)}" data-source-id="${escapeHtml(source.sourceId)}" ${source.enabled ? "checked" : ""} ${source.role === null ? "disabled" : ""}>Enabled</label><label><input type="checkbox" data-action="set-sync-source-active" data-item-id="${escapeHtml(item.id)}" data-source-id="${escapeHtml(source.sourceId)}" ${source.active ? "checked" : ""} ${source.role === null ? "disabled" : ""}>Active</label></div>`).join("")}</div>`;
    }
    return section(`${capitalize(container.kind)} Timeline`, "container", true, "", body, false);
}
function audioInspector(item, state) {
    const fullState = state;
    const localhost = fullState.connectionMode === "localhost";
    return `
    ${section("Volume", "audio-volume", true, "", `
      ${sliderRow("Volume", "audio.gainDb", item.audio.gainDb, -96, 24, .1, "dB", state)}
      ${sliderRow("Pan", "audio.pan", item.audio.pan, -1, 1, .01, "", state)}
      ${sliderRow("Fade In", "audio.fadeIn", item.audio.fadeIn, 0, Math.min(5, item.duration), .01, "s", state, 1, null)}
      ${sliderRow("Fade Out", "audio.fadeOut", item.audio.fadeOut, 0, Math.min(5, item.duration), .01, "s", state, 1, null)}
      <div class="button-row"><button class="square-toggle ${item.audio.muted ? "active" : ""}" data-action="toggle-audio-mute">Mute</button>${localhost ? "" : `<button class="square-toggle ${item.audio.solo ? "active" : ""}" data-action="toggle-item-solo">Solo</button>`}</div>
    `)}
    ${localhost ? "" : section("Audio Configuration", "audio-config", true, "", `
      ${selectRow("Channels", "audio.channels", "stereo", [["stereo", "Stereo"], ["dual-mono", "Dual Mono"], ["mono", "Mono"]])}
      <div class="channel-strip"><span>L</span><div class="channel-wave">${waveBars(26)}</div><span>R</span></div>
    `)}
    ${localhost ? "" : section("Audio Enhancements", "audio-enhance", true, "", `
      ${sliderRow("Loudness", "audio.loudness", 0, 0, 100, 1, "%", state)}
      ${sliderRow("Noise Removal", "audio.noiseRemoval", 0, 0, 100, 1, "%", state)}
    `)}
  `;
}
function colorInspector(item, state) {
    const color = item.video.color;
    const zones = ["shadows", "midtones", "highlights"];
    if (state.connectionMode === "localhost") {
        const colorEffects = item.effects.filter((effect) => {
            const capability = state.capabilities.effects.find((candidate) => candidate.handler === effect.handler || candidate.resource.uid === effect.resourceUid);
            return /color|tint|vibran|board|wheel/i.test(`${capability?.id ?? ""} ${capability?.handler ?? ""}`);
        });
        return colorCorrectionsInspector(item, state, colorEffects);
    }
    return `
    ${section("Color Adjustments", "color", true, "", `
      ${sliderRow("Exposure", "video.color.exposure", color.exposure, -100, 100, 1, "", state)}
      ${sliderRow("Brightness", "video.color.brightness", color.brightness, -100, 100, 1, "", state)}
      ${sliderRow("Contrast", "video.color.contrast", color.contrast, -100, 100, 1, "", state)}
      ${sliderRow("Saturation", "video.color.saturation", color.saturation, -100, 100, 1, "", state)}
      ${sliderRow("Temperature", "video.color.temperature", color.temperature, -100, 100, 1, "", state)}
      ${sliderRow("Tint", "video.color.tint", color.tint, -100, 100, 1, "", state)}
      ${sliderRow("Highlights", "video.color.highlights", color.highlights, -100, 100, 1, "", state)}
      ${sliderRow("Midtones", "video.color.midtones", color.midtones, -100, 100, 1, "", state)}
      ${sliderRow("Shadows", "video.color.shadows", color.shadows, -100, 100, 1, "", state)}
      ${sliderRow("Black Point", "video.color.blackPoint", color.blackPoint, -100, 100, 1, "", state)}
      ${sliderRow("Hue", "video.color.hue", color.hue, -180, 180, 1, "°", state)}
    `)}
    ${state.connectionMode === "mock" ? `<div class="color-board" aria-label="Color tonal-range controls">
      ${zones.map((zone) => {
        const offset = color.pucks[zone];
        const px = Number(offset.x) * .14;
        const py = Number(offset.y) * .14;
        return `<button class="color-puck ${zone} ${state.activeColorZone === zone ? "active" : ""}" data-action="select-color-zone" data-zone="${zone}" data-color-wheel="${zone}" style="--wheel-x:${px}px;--wheel-y:${py}px" title="Select or drag ${capitalize(zone)} (is still a mock)"><i><b></b></i><span>${capitalize(zone)}</span><small>${Math.round(offset.x)}, ${Math.round(offset.y)}</small></button>`;
    }).join("")}
    </div>
    <div class="color-zone-context">
      <strong>${capitalize(state.activeColorZone)}</strong>
      ${sliderRow("Level", `video.color.${state.activeColorZone}`, color[state.activeColorZone], -100, 100, 1, "", state)}
    </div>` : '<p class="capability-note inspector-note">Bladeworks exposes the supported portable Color Adjustments controls above. Proprietary wheels and curves are intentionally omitted.</p>'}
  `;
}
/**
 * Project Bladeworks color effects into the same visual hierarchy as Final
 * Cut's Color Inspector while retaining the capability catalog as truth.
 *
 * Main callers:
 * - colorInspector in localhost mode.
 *
 * Why this exists:
 * Color Adjustments is one semantic correction with Light and Color groups,
 * not an undifferentiated list of effect parameters. Unsupported native-only
 * controls are omitted instead of being presented as controls that do nothing.
 */
function colorCorrectionsInspector(item, state, colorEffects) {
    const selected = colorEffects[0] ?? null;
    const selectedCapability = selected
        ? state.capabilities.effects.find((candidate) => candidate.handler === selected.handler || candidate.resource.uid === selected.resourceUid) ?? null
        : null;
    const correctionName = selected?.name ?? "No Corrections";
    return `<div class="fcp-color-inspector">
    <header class="color-correction-selector">
      <label><input type="checkbox" ${selected?.enabled ? "checked" : ""} ${selected ? `data-action="toggle-effect" data-effect-id="${escapeHtml(selected.id)}"` : "disabled"}><span>${escapeHtml(correctionName)}</span></label>
      <span class="color-selector-chevron">⌄</span>
    </header>
    ${selected && selectedCapability?.id === "effect-color-adjustments"
        ? colorAdjustmentsPanel(selected, selectedCapability)
        : selected
            ? `<div class="color-generic-correction">${effectRow(selected, 0, colorEffects.length, state, false)}</div>`
            : `<div class="color-correction-empty"><p>No color correction is applied.</p><button data-action="open-effects-browser">Add Color Adjustments</button></div>`}
    ${selected ? '<footer class="color-correction-footer"><button data-action="open-effects-browser">Add Correction</button></footer>' : ""}
  </div>`;
}
function colorAdjustmentsPanel(effect, capability) {
    const byName = new Map(capability.parameters.map((parameter) => [parameter.name, parameter]));
    const light = ["Exposure", "Contrast", "Brightness", "Highlights", "Black Point", "Shadows"];
    const color = [
        "Saturation",
        "Highlights Warmth",
        "Highlights Tint",
        "Midtones Warmth",
        "Midtones Tint",
        "Shadows Warmth",
        "Shadows Tint",
    ];
    const row = (name) => {
        const parameter = byName.get(name);
        if (!parameter)
            return "";
        const raw = effect.parameters[parameter.key] ?? parameter.default;
        const value = typeof raw === "number" ? raw : 0;
        const minimum = parameter.min ?? -100;
        const maximum = parameter.max ?? 100;
        const step = parameter.type === "integer" ? 1 : 0.01;
        const normalizedValue = value < 0
            ? (minimum < 0 ? value / Math.abs(minimum) * 100 : 0)
            : (maximum > 0 ? value / maximum * 100 : 0);
        const oneSidedNegative = maximum <= 0;
        const attributes = `data-action="set-capability-parameter" data-owner="effect" data-owner-id="${escapeHtml(effect.id)}" data-parameter-key="${escapeHtml(parameter.key)}"`;
        return `<label class="fcp-color-row">
      <span>${escapeHtml(name)}</span>
      <span class="fcp-color-slider-shell ${oneSidedNegative ? "one-sided-negative" : ""}">
        <input class="fcp-color-slider" type="range" min="-100" max="${oneSidedNegative ? 0 : 100}" step="0.1" value="${normalizedValue}" data-actual-min="${minimum}" data-actual-max="${maximum}" ${attributes} aria-label="${escapeHtml(name)} ${value}">
      </span>
      <input class="fcp-color-value" type="number" min="${minimum}" max="${maximum}" step="${step}" value="${value}" ${attributes} aria-label="${escapeHtml(name)} value">
    </label>`;
    };
    const controlRange = byName.get("Control Range");
    const controlValue = controlRange
        ? String(effect.parameters[controlRange.key] ?? controlRange.default ?? controlRange.choices?.[0] ?? "0 (SDR)")
        : "0 (SDR)";
    return `<div class="fcp-color-adjustments">
    <label class="fcp-control-range"><span>Control Range</span><select ${controlRange ? `data-action="set-capability-parameter" data-owner="effect" data-owner-id="${escapeHtml(effect.id)}" data-parameter-key="${escapeHtml(controlRange.key)}"` : "disabled"}>${(controlRange?.choices ?? [controlValue]).map((choice) => `<option value="${escapeHtml(choice)}" ${choice === controlValue ? "selected" : ""}>${escapeHtml(choice.replace(/^0 \(/, "").replace(/\)$/, ""))}</option>`).join("")}</select></label>
    <section class="fcp-color-group">
      <header><strong>LIGHT</strong></header>
      ${light.map(row).join("")}
    </section>
    <section class="fcp-color-group">
      <header><strong>COLOR</strong></header>
      ${color.map(row).join("")}
    </section>
  </div>`;
}
function retimeSection(item) {
    if (["gap", "title", "caption", "generator"].includes(item.kind))
        return "";
    const points = item.timeMap?.points ?? [];
    const sourceSpan = points.length >= 2
        ? Math.abs(points[points.length - 1].value.seconds - points[0].value.seconds)
        : item.duration;
    const rate = item.duration > 0 ? sourceSpan / item.duration : 1;
    const reverse = points.length >= 2
        && points[points.length - 1].value.seconds < points[0].value.seconds;
    const signedPercent = Number(((reverse ? -1 : 1) * rate * 100).toFixed(2));
    return section("Re-time", "retime", item.timeMap !== null, "", `
    <label class="retime-custom"><span>Speed</span><span class="retime-number"><input type="number" min="-10000" max="10000" step="0.1" value="${signedPercent}" data-action="set-retime-custom" aria-label="Speed %" title="Use a negative value to reverse playback"><em>%</em></span></label>
    <label class="retime-pitch"><input type="checkbox" data-action="toggle-retime-pitch" ${item.timeMap?.preservesPitch === false ? "" : "checked"}><span>Preserve Pitch</span></label>
  `, true, "toggle-retime-section");
}
function generatedClipInspector(item) {
    if (item.kind === "generator") {
        return section("Custom Solid", "generator", true, "", `
      <label class="inspector-row"><span>Color</span><input type="color" value="${colorHex(item.generatorColor ?? undefined)}" data-action="set-generator-color"></label>
    `, false);
    }
    if (item.kind !== "title" && item.kind !== "caption")
        return "";
    const style = item.textStyle;
    const caption = item.caption;
    return section(item.kind === "caption" ? "Caption" : "Basic Title", "title", true, "", `
    <label class="inspector-row title-text-row"><span>Text</span><input type="text" value="${escapeHtml(item.text ?? item.name)}" data-action="set-title-text"></label>
    ${style ? `
      <label class="inspector-row"><span>Font</span><input type="text" value="${escapeHtml(style.font)}" data-action="set-text-style" data-style-key="font"></label>
      <label class="inspector-row"><span>Face</span><input type="text" value="${escapeHtml(style.fontFace)}" data-action="set-text-style" data-style-key="fontFace"></label>
      <label class="inspector-row capability-number-row"><span>Size</span><input type="number" min="1" max="500" step="1" value="${style.fontSize}" data-action="set-text-style" data-style-key="fontSize"></label>
      <label class="inspector-row"><span>Color</span><input type="color" value="${colorHex(style.fontColor)}" data-action="set-text-style" data-style-key="fontColor"></label>
      ${selectActionRow("Alignment", "set-text-style", "data-style-key=\"alignment\"", style.alignment, [["left", "Left"], ["center", "Center"], ["right", "Right"]])}
    ` : '<p class="capability-note">This title does not expose the certified Basic Title style model and remains preserved.</p>'}
    ${item.kind === "caption" && caption ? `
      ${selectActionRow("Placement", "set-caption-field", "data-caption-key=\"placement\"", caption.placement, [["bottom", "Bottom"], ["top", "Top"]])}
      ${selectActionRow("Alignment", "set-caption-field", "data-caption-key=\"alignment\"", caption.alignment, [["left", "Left"], ["center", "Center"], ["right", "Right"]])}
      <label class="inspector-row"><span>Role</span><input type="text" value="${escapeHtml(caption.role)}" data-action="set-caption-field" data-caption-key="role"></label>
    ` : ""}
  `, false);
}
function selectActionRow(label, action, extraAttributes, value, options) {
    return `<label class="inspector-row"><span>${escapeHtml(label)}</span><select data-action="${action}" ${extraAttributes}>${options.map(([id, name]) => `<option value="${id}" ${id === value ? "selected" : ""}>${escapeHtml(name)}</option>`).join("")}</select></label>`;
}
function transitionInspector(transition, state) {
    const capability = state.capabilities.transitions.find((candidate) => candidate.name === transition.name) ?? state.capabilities.transitions.find((candidate) => candidate.handler === transition.handler
        || candidate.resource.uid === transition.resourceUid);
    return `
    ${section("Transition", "transition", true, "", `
      <div class="transition-heading"><strong>${escapeHtml(transition.name)}</strong></div>
      <label class="inspector-row capability-number-row"><span>Duration</span><input type="number" min="0.033" step="0.033" value="${transition.duration}" data-action="set-transition-duration" data-transition-id="${escapeHtml(transition.id)}"><em>s</em></label>
      ${capability?.parameters.length ? capabilityParameterRows("transition", transition.id, capability.parameters, transition.parameters) : ""}
      <button class="danger-inline" data-action="remove-transition" data-transition-id="${escapeHtml(transition.id)}">Remove Transition</button>
    `, false)}
  `;
}
function capabilityParameterRows(owner, ownerId, parameters, values) {
    return parameters.map((parameter) => capabilityParameterRow(owner, ownerId, parameter, values[parameter.key] ?? parameter.default)).join("");
}
function capabilityParameterRow(owner, ownerId, parameter, value) {
    const attributes = `data-action="set-capability-parameter" data-owner="${owner}" data-owner-id="${escapeHtml(ownerId)}" data-parameter-key="${escapeHtml(parameter.key)}"`;
    if (parameter.type === "boolean") {
        return `<label class="inspector-row capability-toggle"><span>${escapeHtml(parameter.name)}</span><input type="checkbox" ${value ? "checked" : ""} ${attributes}></label>`;
    }
    if (parameter.type === "enum") {
        const inherited = value === undefined ? '<option value="" selected disabled>Renderer default</option>' : "";
        return `<label class="inspector-row"><span>${escapeHtml(parameter.name)}</span><select ${attributes}>${inherited}${(parameter.choices ?? []).map((choice) => `<option value="${escapeHtml(choice)}" ${choice === value ? "selected" : ""}>${escapeHtml(choice)}</option>`).join("")}</select></label>`;
    }
    if (parameter.components?.length && (parameter.type === "number" || parameter.type === "integer")) {
        const record = typeof value === "object" && value !== null ? value : {};
        return `<div class="capability-compound"><span>${escapeHtml(parameter.name)}</span><div>${parameter.components.map((component) => `<label><small>${escapeHtml(component)}</small><input type="number" ${parameter.min !== undefined ? `min="${parameter.min}"` : ""} ${parameter.max !== undefined ? `max="${parameter.max}"` : ""} step="${parameter.type === "integer" ? 1 : 0.01}" value="${record[component] === undefined ? "" : Number(record[component])}" placeholder="Default" data-component="${escapeHtml(component)}" ${attributes}></label>`).join("")}</div></div>`;
    }
    if (parameter.type === "color") {
        return `<label class="inspector-row"><span>${escapeHtml(parameter.name)}</span><input type="color" value="${colorHex(value)}" ${attributes}></label>`;
    }
    if (parameter.type === "point" || parameter.type === "rect") {
        const components = parameter.components ?? (parameter.type === "point" ? ["x", "y"] : ["left", "top", "right", "bottom"]);
        const record = typeof value === "object" && value !== null ? value : {};
        return `<div class="capability-compound"><span>${escapeHtml(parameter.name)}</span><div>${components.map((component) => `<label><small>${escapeHtml(component)}</small><input type="number" step="0.01" value="${record[component] === undefined ? "" : Number(record[component])}" placeholder="Default" data-component="${escapeHtml(component)}" ${attributes}></label>`).join("")}</div></div>`;
    }
    return `<label class="inspector-row capability-number-row"><span>${escapeHtml(parameter.name)}</span><input type="number" ${parameter.min !== undefined ? `min="${parameter.min}"` : ""} ${parameter.max !== undefined ? `max="${parameter.max}"` : ""} step="${parameter.type === "integer" ? 1 : 0.01}" value="${value === undefined ? "" : Number(value)}" placeholder="Renderer default" ${attributes}></label>`;
}
function colorHex(value) {
    if (typeof value !== "object" || value === null || !("red" in value))
        return "#000000";
    const color = value;
    const byte = (component) => Math.round(Math.max(0, Math.min(1, component)) * 255).toString(16).padStart(2, "0");
    return `#${byte(color.red)}${byte(color.green)}${byte(color.blue)}`;
}
function infoInspector(item, fps) {
    return `
    <div class="info-table">
      ${infoRow("Name", item.name)}
      ${infoRow("Role", item.role === "storyline" ? "Dialogue" : item.role)}
      ${infoRow("Start", formatTimecode(item.timelineStart, fps))}
      ${infoRow("Duration", formatTimecode(item.duration, fps))}
      ${infoRow("Source In", formatTimecode(item.sourceStart, fps))}
      ${infoRow("Asset ID", item.assetId ?? "Generated")}
      ${infoRow("FCPXML ID", item.id)}
    </div>
  `;
}
function infoRow(label, value) {
    return `<div class="info-row"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`;
}
function sectionKeyframePath(sectionId) {
    const paths = {
        compositing: "transform.opacity",
        transform: "transform.position",
        crop: "video.crop.left",
        distort: "video.distort.topleft",
        "audio-volume": "audio.gainDb",
        "audio-enhance": "audio.loudness",
        color: "video.color.exposure",
    };
    return paths[sectionId] ?? "";
}
function section(title, sectionId, enabled, enablePath, body, resettable = true, enableAction = "") {
    return `
    <section class="inspector-section ${enabled ? "" : "disabled"}" data-section="${sectionId}">
      <div class="section-header">
        ${enablePath
        ? `<input class="section-check" type="checkbox" data-parameter-path="${enablePath}" ${enabled ? "checked" : ""}>`
        : enableAction
            ? `<input class="section-check" type="checkbox" data-action="${enableAction}" ${enabled ? "checked" : ""}>`
            : '<span class="section-check-spacer"></span>'}
        <button class="section-disclosure" data-action="toggle-inspector-section" data-section="${sectionId}" aria-label="Show or hide ${escapeHtml(title)}" aria-expanded="true">${icon("disclosure-open")}</button>
        <strong class="section-title">${escapeHtml(title)}</strong>
        ${sectionKeyframePath(sectionId) ? `<button class="section-keyframe-menu" data-action="keyframe-menu" data-path="${sectionKeyframePath(sectionId)}" title="Keyframe menu">${icon("keyframe-menu")}</button>` : '<span class="section-keyframe-spacer"></span>'}
        ${resettable ? `<button class="section-reset" data-action="reset-section" data-section="${sectionId}" title="Reset ${escapeHtml(title)}">${icon("reset")}</button>` : '<span class="section-reset-spacer"></span>'}
      </div>
      <div class="section-body">${body}</div>
    </section>
  `;
}
function sliderRow(label, path, value, min, max, step, unit, state, multiplier = 1, keyframePath = path) {
    const shown = value * multiplier;
    return `
    <div class="parameter-row slider-parameter">
      <label>${escapeHtml(label)}</label>
      <input class="fcp-slider" type="range" min="${min}" max="${max}" step="${step}" value="${value}" data-parameter-path="${path}">
      <div class="numeric-field"><input type="number" min="${min * multiplier}" max="${max * multiplier}" step="${step * multiplier}" value="${round(shown, step * multiplier)}" data-number-path="${path}" data-number-multiplier="${multiplier}"><span>${unit}</span></div>
      ${keyframePath ? keyframeButton(keyframePath, state) : '<span class="row-spacer"></span>'}
    </div>
  `;
}
function xyRow(label, pathX, valueX, pathY, valueY, unit, state, keyframePath = pathX) {
    return `
    <div class="parameter-row xy-parameter">
      <label>${escapeHtml(label)}</label>
      <div class="axis-field"><span>X</span><input type="number" step="0.1" value="${round(valueX, .1)}" data-number-path="${pathX}"><i>${unit}</i></div>
      <div class="axis-field"><span>Y</span><input type="number" step="0.1" value="${round(valueY, .1)}" data-number-path="${pathY}"><i>${unit}</i></div>
      ${keyframeButton(keyframePath, state)}
    </div>
  `;
}
function knobRow(label, path, value, min, max, step, unit, state) {
    return `
    <div class="parameter-row knob-parameter">
      <label>${escapeHtml(label)}</label>
      <div class="fcp-knob" data-knob-path="${path}" data-min="${min}" data-max="${max}" data-step="${step}" style="--knob-angle:${value}deg" title="Drag vertically or horizontally"><i></i></div>
      <div class="numeric-field"><input type="number" min="${min}" max="${max}" step="${step}" value="${round(value, step)}" data-number-path="${path}"><span>${unit}</span></div>
      ${keyframeButton(path, state)}
    </div>
  `;
}
function selectRow(label, path, value, options) {
    return `<div class="parameter-row select-parameter"><label>${escapeHtml(label)}</label><select data-parameter-path="${path}">${options.map(([id, name]) => `<option value="${id}" ${id === value ? "selected" : ""}>${escapeHtml(name)}</option>`).join("")}</select><span class="row-spacer"></span></div>`;
}
function segmentedRow(label, path, value, options) {
    return `<div class="parameter-row segmented-parameter"><label>${escapeHtml(label)}</label><div class="inspector-segmented">${options.map(([id, name]) => `<button class="${id === value ? "active" : ""}" data-action="set-parameter" data-path="${path}" data-value="${id}">${escapeHtml(name)}</button>`).join("")}</div><span class="row-spacer"></span></div>`;
}
function toggleSectionRow(label, path, value) {
    return `<div class="standalone-toggle-row"><input type="checkbox" data-parameter-path="${path}" ${value ? "checked" : ""}><strong>${escapeHtml(label)}</strong><button data-action="reset-path" data-path="${path}">${icon("reset")}</button></div>`;
}
function keyframeButton(path, state) {
    const item = state.selectedItem ?? null;
    const active = item ? itemHasKeyframeAt(item, path, state.currentTime, state.fps) : false;
    return `<button class="keyframe-diamond ${active ? "active" : ""}" data-action="toggle-keyframe" data-path="${path}" title="Add or remove keyframe">${icon("diamond")}</button>`;
}
function round(value, step) {
    const fraction = String(step).split(".")[1];
    const decimals = fraction ? fraction.length : 0;
    return Number(value.toFixed(Math.min(3, decimals)));
}
