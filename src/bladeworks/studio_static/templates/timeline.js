/**
 * Magnetic timeline chrome: toolbar, index, clips, markers, transitions.
 *
 * Architecture map:
 * EditorState + media assets
 *   -> timelineToolbar / timelineIndex / timelineTemplate HTML
 *   -> data-item-id / data-action / data-trim on clips
 *
 * Main callers: BladeworksEditorApp.renderTimeline
 */
import { projectDuration } from "../magnetic-timeline.js";
import { escapeHtml, formatTimecode, timelineRulerDuration } from "../ui.js";
import { icon } from "./icons.js";
import { capitalize, mediaVisualAttributes, waveBars } from "./shared.js";
export function timelineToolbarTemplate(state) {
    return `
    <div class="timeline-toolbar-left">
      <button class="index-button ${state.timelineIndexOpen ? "active" : ""}" data-action="toggle-index">Index</button>
      <button class="tools-popup-button ${state.activePopover === "tools" ? "active" : ""}" data-action="toggle-popover" data-popover="tools" title="Tools">${icon(toolIcon(state.tool))}${icon("chevron-down")}</button>
      ${toolsPopover(state)}
    </div>
    <div class="timeline-toolbar-center">
      <button class="timeline-nav-button" data-action="scope-back" ${state.scopeCanGoBack ? "" : "disabled"} title="Previous timeline">‹</button>
      <button class="timeline-nav-button" data-action="scope-forward" ${state.scopeCanGoForward ? "" : "disabled"} title="Next timeline">›</button>
      <nav class="scope-breadcrumbs" aria-label="Timeline location">${(state.scopeBreadcrumbs ?? [{ scopeId: "", label: state.project.name }]).map((crumb, index, all) => `${index === all.length - 1 ? `<span class="active">${escapeHtml(crumb.label)}</span>` : `<button data-action="enter-scope" data-scope-id="${escapeHtml(crumb.scopeId)}">${escapeHtml(crumb.label)}</button>`}${index < all.length - 1 ? '<span>›</span>' : ""}`).join("")}</nav>
      <span class="selection-time">${state.selectedItemIds.length > 1 ? `${state.selectedItemIds.length} selected · ` : ""}${formatTimecode(state.currentTime, state.project.fps)} / ${formatTimecode(projectDuration(state.project), state.project.fps)}</span>
      ${state.tool === "select" ? '<span class="gesture-hint">Drag background · ⇧ contiguous · ⌘ toggle</span>' : state.tool === "range" ? '<span class="gesture-hint active">Drag a clip or lane to select a range</span>' : ""}
    </div>
    <div class="timeline-toolbar-right">
      <button class="timeline-icon-button ${state.skimming ? "active" : ""}" data-action="toggle-skimming" title="Skimming (S)">${icon("skimmer")}</button>
      <button class="timeline-icon-button ${state.snapping ? "active" : ""}" data-action="toggle-snapping" title="Snapping (N)">${icon("snap")}</button>
      <button class="timeline-icon-button ${state.continuousScroll ? "active" : ""}" data-action="toggle-continuous-scroll" title="Continuous Scrolling">${icon("continuous")}</button>
      <button class="timeline-icon-button ${state.activePopover === "appearance" ? "active" : ""}" data-action="toggle-popover" data-popover="appearance" title="Clip Appearance">${icon("appearance")}</button>
      ${appearancePopover(state)}
      <button class="timeline-icon-button ${state.loopPlayback ? "active" : ""}" data-action="toggle-loop-playback" title="Loop Playback (⌘L)">${icon("loop")}</button>
      <span class="timeline-toolbar-divider" aria-hidden="true"></span>
      <button class="timeline-icon-button ${state.activeBrowser === "transitions" ? "active" : ""}" data-action="open-transitions-browser" title="Transitions Browser">${icon("transitions")}</button>
      <button class="timeline-icon-button ${state.activeBrowser === "effects" ? "active" : ""}" data-action="open-effects-browser" title="Effects Browser">${icon("effects")}</button>
      <span class="timeline-toolbar-divider" aria-hidden="true"></span>
      <button class="timeline-icon-button" data-action="undo" ${state.history.canUndo() ? "" : "disabled"} title="Undo">${icon("undo")}</button>
      <button class="timeline-icon-button" data-action="redo" ${state.history.canRedo() ? "" : "disabled"} title="Redo">${icon("redo")}</button>
    </div>
  `;
}
function toolsPopover(state) {
    if (state.activePopover !== "tools")
        return "";
    const tools = [
        ["select", "pointer", "Select", "A"],
        ["trim", "trim-tool", "Trim", "T"],
        ["position", "position-tool", "Position", "P"],
        ["range", "range-tool", "Range Selection", "R"],
        ["blade", "blade", "Blade", "B"],
        ["zoom", "zoom-in", "Zoom", "Z"],
        ["hand", "hand", "Hand", "H"],
    ];
    return `<div class="timeline-popover tools-popover">${tools.map(([id, ico, label, key]) => `<button data-action="timeline-tool" data-tool="${id}" class="${state.tool === id ? "active" : ""}">${icon(ico)}<span>${label}</span><kbd>${key}</kbd></button>`).join("")}</div>`;
}
function appearancePopover(state) {
    if (state.activePopover !== "appearance")
        return "";
    const modeLabels = ["Audio focus", "Audio emphasis", "Balanced", "Video emphasis", "Video focus", "Compact"];
    return `
    <div class="timeline-popover appearance-popover">
      <div class="appearance-modes">${[1, 2, 3, 4, 5, 6].map((mode) => `<button class="appearance-mode mode-${mode} ${state.clipAppearance.mode === mode ? "active" : ""}" data-action="clip-appearance-mode" data-mode="${mode}" aria-label="${modeLabels[mode - 1]}"><i></i><b></b></button>`).join("")}</div>
      <label class="popover-slider"><span>Clip Height</span><input type="range" min="38" max="110" value="${state.clipAppearance.height}" data-action="clip-height"></label>
      <label class="popover-slider"><span>Zoom</span><input type="range" min="18" max="150" value="${state.pixelsPerSecond}" data-action="timeline-zoom"></label>
      <label class="popover-check"><input type="checkbox" data-action="show-clip-names" ${state.clipAppearance.showNames ? "checked" : ""}> Clip Names</label>
      <label class="popover-check"><input type="checkbox" data-action="show-clip-roles" ${state.clipAppearance.showRoles ? "checked" : ""}> Clip Roles</label>
      <button class="popover-fit" data-action="fit-timeline">Zoom to Fit <kbd>⇧Z</kbd></button>
    </div>
  `;
}
function toolIcon(tool) {
    const icons = {
        select: "pointer",
        trim: "trim-tool",
        position: "position-tool",
        range: "range-tool",
        blade: "blade",
        zoom: "zoom-in",
        hand: "hand",
    };
    return icons[tool];
}
/**
 * Find the selected clip anywhere in the project (storyline or connected).
 * Effects apply to whichever clip this returns.
 */
export function selectedTimelineClip(state) {
    if (!state.selectedItemId) {
        return null;
    }
    return state.project.spine.find((clip) => clip.id === state.selectedItemId)
        ?? state.project.connected.find((clip) => clip.id === state.selectedItemId)
        ?? null;
}
/**
 * Resolve the storyline edit point implied by the current selection: the
 * selected spine clip and the clip immediately after it. Transitions attach to
 * this pair. Returns null when no spine clip is selected or it is the last clip.
 */
export function selectedStorylineEdit(state) {
    const index = state.project.spine.findIndex((clip) => clip.id === state.selectedItemId);
    if (index < 0 || index >= state.project.spine.length - 1) {
        return null;
    }
    return { index, left: state.project.spine[index], right: state.project.spine[index + 1] };
}
/**
 * Timeline Index mirrors Final Cut's searchable Clips, Tags, and Roles views.
 * It stays a projection of the same project snapshot and never owns edit state.
 */
export function timelineIndexTemplate(state) {
    if (!state.timelineIndexOpen)
        return "";
    const allItems = [...state.project.spine, ...state.project.connected]
        .filter((item) => `${item.name} ${item.role}`.toLowerCase().includes(state.timelineIndexQuery.toLowerCase()))
        .sort((a, b) => a.timelineStart - b.timelineStart || clipLane(b) - clipLane(a));
    let body = "";
    if (state.timelineIndexTab === "clips") {
        body = `<div class="index-list">${allItems.map((item) => `<button class="index-row ${(state.selectedItemIds.includes(item.id) || item.id === state.selectedItemId) ? "selected" : ""}" data-action="select-index-item" data-item-id="${escapeHtml(item.id)}"><span class="index-role-dot role-${item.role}"></span><span class="index-row-copy"><strong>${escapeHtml(item.name)}</strong><small>${escapeHtml(roleLabel(item.role))}</small></span><time>${formatTimecode(item.timelineStart, state.project.fps)}</time></button>`).join("")}</div>`;
    }
    else if (state.timelineIndexTab === "tags") {
        // Real timeline markers, not fixture data. Every marker travels with its
        // clip, so its absolute timeline time is clip.timelineStart + marker.offset.
        const query = state.timelineIndexQuery.toLowerCase();
        const tags = [...state.project.spine, ...state.project.connected]
            .flatMap((clip) => clip.markers.map((marker) => ({
            name: marker.name,
            type: marker.type,
            completed: marker.completed,
            time: clip.timelineStart + marker.offset,
        })))
            .filter((tag) => tag.name.toLowerCase().includes(query))
            .sort((a, b) => a.time - b.time);
        body = tags.length
            ? `<div class="index-list">${tags.map((tag) => `<button class="index-row" data-action="seek-index-time" data-time="${tag.time}">${icon(tag.type === "chapter" ? "sparkle" : tag.type === "todo" ? "key" : "marker")}<span class="index-row-copy"><strong>${escapeHtml(tag.name)}</strong><small>${capitalize(tag.type)}${tag.type === "todo" ? (tag.completed ? " · Done" : " · To-do") : ""}</small></span><time>${formatTimecode(tag.time, state.project.fps)}</time></button>`).join("")}</div>`
            : "";
    }
    else {
        const groups = [
            ["storyline", "Dialogue", state.project.spine.length],
            ["connected-video", "Video", state.project.connected.filter((item) => item.lane > 0 && item.kind === "video").length],
            ["title", "Titles", state.project.connected.filter((item) => item.kind === "title").length],
            ["connected-audio", "Music", state.project.connected.filter((item) => item.lane < 0).length],
        ];
        body = `<div class="index-role-list">${groups.map(([role, label, count]) => `<button class="index-role-row" data-action="filter-index-role" data-role="${role}"><span class="index-role-dot role-${role}"></span><strong>${label}</strong><span>${count}</span></button>`).join("")}</div>`;
    }
    return `<div class="timeline-index-shell">
    <div class="timeline-index-tabs">
      ${[["clips", "Clips"], ["tags", "Tags"], ["roles", "Roles"]].map(([tab, label]) => `<button class="${state.timelineIndexTab === tab ? "active" : ""}" data-action="timeline-index-tab" data-tab="${tab}">${label}</button>`).join("")}
      <button class="timeline-index-close" data-action="toggle-index" title="Close Timeline Index">×</button>
    </div>
    <label class="timeline-index-search">${icon("search")}<input id="timeline-index-search" placeholder="Search ${state.timelineIndexTab}" value="${escapeHtml(state.timelineIndexQuery)}"></label>
    ${body || '<div class="index-empty">No matching items</div>'}
  </div>`;
}
function roleLabel(role) {
    const labels = {
        storyline: "Primary Storyline",
        "connected-video": "Connected Video",
        "connected-audio": "Music",
        title: "Title",
    };
    return labels[role];
}
/** Storyline clips have no lane; connected clips do. Spine items sort as lane 0. */
function clipLane(item) {
    const maybeConnected = item;
    return typeof maybeConnected.lane === "number" ? maybeConnected.lane : 0;
}
export function timelineTemplate(state, assets = []) {
    const duration = projectDuration(state.project);
    const width = Math.max(1000, duration * state.pixelsPerSecond + 220);
    const videoLanes = state.project.connected.filter((item) => item.lane > 0).map((item) => item.lane);
    const audioLaneAbs = state.project.connected.filter((item) => item.lane < 0).map((item) => Math.abs(item.lane));
    const visualLanes = Math.max(1, ...videoLanes, 0);
    const audioLanes = Math.max(1, ...audioLaneAbs, 0);
    const clipHeight = state.clipAppearance.height;
    const connectedLaneStride = 27;
    const storylineTop = 34 + visualLanes * connectedLaneStride;
    const canvasHeight = storylineTop + clipHeight + audioLanes * 44 + 66;
    return `
    <div class="timeline-canvas tool-${state.tool} appearance-${state.clipAppearance.mode} ${state.activeScopeId ? "nested-scope" : ""} ${state.clipAppearance.showNames ? "show-names" : ""} ${state.clipAppearance.showRoles ? "show-roles" : ""} ${state.dropPreviewClipId ? "drag-previewing" : ""}" style="width:${width}px;height:${canvasHeight}px;--clip-height:${clipHeight}px" data-drop-role="storyline">
      ${rulerTemplate(duration, state.pixelsPerSecond, width)}
      <div class="timeline-lane-labels"><span style="top:${storylineTop + 6}px">Primary Storyline</span></div>
      <div class="connected-drop-zone" data-drop-role="connected-video" style="top:28px;height:${Math.max(28, visualLanes * connectedLaneStride + 7)}px"></div>
      ${state.project.connected.filter((item) => item.lane > 0).map((item) => connectedClipTemplate(item, state, storylineTop, connectedLaneStride, assets)).join("")}
      ${state.project.connected.filter((item) => item.lane > 0).map((item) => anchorTemplate(item, storylineTop, connectedLaneStride, state.pixelsPerSecond)).join("")}
      <div class="storyline-shelf" style="top:${storylineTop}px;height:${clipHeight + 7}px"></div>
      ${state.project.spine.map((item, index) => storylineClipTemplate(item, index, state, storylineTop, assets)).join("")}
      ${state.project.transitions.map((transition) => {
        const right = state.project.spine.find((item) => item.id === transition.rightItemId);
        return right ? transitionTemplate(transition, right.timelineStart, state.pixelsPerSecond, storylineTop, clipHeight, state.selectedTransitionStart) : "";
    }).join("")}
      <div class="audio-drop-zone" data-drop-role="connected-audio" style="top:${storylineTop + clipHeight + 11}px;height:${audioLanes * 44 + 22}px"></div>
      ${state.project.connected.filter((item) => item.lane < 0).map((item) => audioClipTemplate(item, state, storylineTop, clipHeight, assets)).join("")}
      ${state.rangeSelection ? `<div class="range-selection" style="left:${state.rangeSelection.start * state.pixelsPerSecond}px;width:${Math.max(1, (state.rangeSelection.end - state.rangeSelection.start) * state.pixelsPerSecond)}px;top:${state.rangeSelection.top}px;height:${Math.max(2, state.rangeSelection.bottom - state.rangeSelection.top)}px"><span></span></div>` : ""}
      ${state.marqueeSelection ? `<div class="marquee-selection" style="left:${state.marqueeSelection.left}px;top:${state.marqueeSelection.top}px;width:${state.marqueeSelection.width}px;height:${state.marqueeSelection.height}px"></div>` : ""}
      <div class="skimmer ${state.skimming ? "" : "hidden"}" id="skimmer" style="left:0px"><span></span></div>
      <div class="playhead" id="playhead" style="left:${state.currentTime * state.pixelsPerSecond}px"><span></span><i></i></div>
      <div class="timeline-end" style="left:${duration * state.pixelsPerSecond}px"></div>
      ${markerEditorOverlay(state, storylineTop)}
    </div>
  `;
}
/**
 * Inline editor for the marker named by state.markerEditorId. Anchored above the
 * marker's absolute timeline position (clip start + local offset). Lets the user
 * rename, switch type (standard / to-do / chapter), toggle completion, or delete.
 */
function markerEditorOverlay(state, storylineTop) {
    if (!state.markerEditorId) {
        return "";
    }
    let host = null;
    let marker = null;
    for (const clip of [...state.project.spine, ...state.project.connected]) {
        const found = clip.markers.find((candidate) => candidate.id === state.markerEditorId);
        if (found) {
            host = clip;
            marker = found;
            break;
        }
    }
    if (!host || !marker) {
        return "";
    }
    const absolute = host.timelineStart + marker.offset;
    const left = absolute * state.pixelsPerSecond;
    const types = [["standard", "Standard"], ["todo", "To Do"], ["chapter", "Chapter"]];
    return `
    <div class="marker-editor" style="left:${left}px;top:${Math.max(6, storylineTop - 96)}px" data-marker-id="${escapeHtml(marker.id)}" data-item-id="${escapeHtml(host.id)}">
      <input class="marker-name" type="text" value="${escapeHtml(marker.name)}" placeholder="Marker name" data-action="marker-rename" data-item-id="${escapeHtml(host.id)}" data-marker-id="${escapeHtml(marker.id)}">
      <div class="marker-types">${types.map(([id, label]) => `<button class="${marker.type === id ? "active" : ""}" data-action="marker-type" data-item-id="${escapeHtml(host.id)}" data-marker-id="${escapeHtml(marker.id)}" data-type="${id}">${label}</button>`).join("")}</div>
      <div class="marker-editor-actions">
        <button class="marker-close" data-action="marker-close">Done</button>
      </div>
    </div>
  `;
}
function rulerTemplate(duration, pps, width) {
    const interval = pps >= 100 ? 1 : pps >= 55 ? 5 : pps >= 30 ? 10 : 30;
    const ticks = [];
    // The ruler must stop at the rendered canvas boundary. Off-canvas absolute
    // ticks enlarge the scroller's scrollWidth, which makes Shift-Z appear to
    // leave several minutes of empty timeline after a short Project.
    const visibleDuration = timelineRulerDuration(duration, pps, width);
    for (let t = 0; t <= visibleDuration; t += interval) {
        const major = Math.round(t / interval) % (interval === 1 ? 5 : 3) === 0;
        ticks.push(`<span class="ruler-tick ${major ? "major" : ""}" style="left:${t * pps}px">${major ? `<b>${formatRuler(t)}</b>` : ""}</span>`);
    }
    return `<div class="timeline-ruler" data-action="seek-ruler" style="width:${width}px">${ticks.join("")}</div>`;
}
function formatRuler(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `00:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}:00`;
}
/**
 * Marker glyphs pinned inside a clip at each marker's local offset. Because the
 * offset is clip-local, the layer follows the clip after any ripple edit. Click
 * a glyph to open the inline marker editor.
 */
function markerLayer(item, state) {
    if (item.markers.length === 0) {
        return "";
    }
    return `<div class="clip-marker-layer">${item.markers.map((marker) => `<button class="clip-marker marker-${marker.type} ${marker.completed ? "done" : ""} ${marker.id === state.markerEditorId ? "editing" : ""}" data-action="edit-marker" data-item-id="${escapeHtml(item.id)}" data-marker-id="${escapeHtml(marker.id)}" style="left:${marker.offset * state.pixelsPerSecond}px" title="${escapeHtml(marker.name)}"></button>`).join("")}</div>`;
}
/** Small badge that surfaces how many effects a clip carries. */
function effectBadge(item) {
    if (item.effects.length === 0) {
        return "";
    }
    return `<span class="clip-fx-badge" title="${item.effects.length} effect${item.effects.length === 1 ? "" : "s"}">${icon("effects")}<b>${item.effects.length}</b></span>`;
}
/** FCP-style speed bar: orange for slow, blue for fast or reverse. */
function retimeBar(item) {
    const points = item.timeMap?.points ?? [];
    if (points.length < 2)
        return "";
    const first = points[0];
    const last = points[points.length - 1];
    const sourceSpan = Math.abs(last.value.seconds - first.value.seconds);
    const rate = sourceSpan / Math.max(0.001, last.time.seconds - first.time.seconds);
    const reverse = last.value.seconds < first.value.seconds;
    const tone = rate < 1 ? "slow" : "fast";
    return `<div class="clip-retime-bar ${tone}" title="${reverse ? "Reverse " : ""}${Math.round(rate * 100)}%, Fast (Floor)"><span>${reverse ? "Reverse " : ""}${Math.round(rate * 100)}%</span>${points.length > 2 ? `<i>${points.length - 1} segments</i>` : ""}</div>`;
}
function storylineClipTemplate(item, index, state, top, assets) {
    const left = item.timelineStart * state.pixelsPerSecond;
    const width = Math.max(18, item.duration * state.pixelsPerSecond);
    const selected = state.selectedItemIds.includes(item.id) || item.id === state.selectedItemId;
    const primary = item.id === state.selectedItemId;
    const asset = assets.find((candidate) => candidate.id === item.assetId);
    const visualAttributes = item.kind === "gap" || item.kind === "title" || item.timeMap
        ? ""
        : mediaVisualAttributes(asset, item.sourceStart, item.duration, Math.max(3, Math.min(12, Math.ceil(width / 70))), Math.max(24, Math.min(256, Math.round(width / 3))));
    return `
    <article class="timeline-clip storyline-clip ${selected ? "selected" : ""} ${primary ? "primary-selected" : ""} ${item.id === state.dropPreviewClipId ? "drop-preview" : ""}" draggable="${state.tool === "select"}" data-item-id="${escapeHtml(item.id)}" data-role="storyline" ${visualAttributes} style="left:${left}px;top:${top}px;width:${width}px;height:var(--clip-height);--clip-color:${item.colors.a};--clip-index:${index}">
      ${retimeBar(item)}
      <div class="clip-filmstrip"></div>
      <div class="clip-audio-wave ${item.audio.muted ? "muted-audio" : ""}" style="--volume-scale:${volumeWaveScale(item.audio.gainDb)}"></div>
      <div class="clip-volume-line ${item.audio.muted ? "muted-audio" : ""}" style="bottom:${Math.max(7, 16 + item.audio.gainDb / 3)}px"></div>
      <div class="clip-role-bar"></div>
      <div class="clip-label"><strong>${escapeHtml(item.name)}</strong><span>${formatTimecode(item.duration, state.project.fps)}</span></div>
      ${containerButton(item)}
      ${effectBadge(item)}
      ${markerLayer(item, state)}
      ${primary ? '<span class="trim-handle left" data-trim="left"></span><span class="trim-handle right" data-trim="right"></span>' : ""}
    </article>
  `;
}
function connectedClipTemplate(item, state, storylineTop, laneStride, assets) {
    const left = item.timelineStart * state.pixelsPerSecond;
    const width = Math.max(18, item.duration * state.pixelsPerSecond);
    const top = storylineTop - item.lane * laneStride;
    const selected = state.selectedItemIds.includes(item.id) || item.id === state.selectedItemId;
    const primary = item.id === state.selectedItemId;
    const visualAttributes = item.kind === "title" || item.timeMap ? "" : mediaVisualAttributes(assets.find((candidate) => candidate.id === item.assetId), item.sourceStart, item.duration, Math.max(2, Math.min(12, Math.ceil(width / 70))), 0);
    return `<article class="timeline-clip connected-clip ${item.kind === "title" ? "title-clip" : ""} ${selected ? "selected" : ""} ${primary ? "primary-selected" : ""} ${item.id === state.dropPreviewClipId ? "drop-preview" : ""}" draggable="${state.tool === "select"}" data-item-id="${escapeHtml(item.id)}" data-role="${item.role}" ${visualAttributes} style="left:${left}px;top:${top}px;width:${width}px;--clip-color:${item.colors.a}">
    ${retimeBar(item)}
    ${item.kind === "title" ? `<div class="title-glyph">T</div>` : '<div class="connected-filmstrip"></div>'}
    <div class="connected-role-dot" aria-hidden="true"></div>
    <div class="clip-label compact"><strong>${escapeHtml(item.name)}</strong></div>
    ${containerButton(item)}
    ${effectBadge(item)}
    ${markerLayer(item, state)}
    ${primary ? '<span class="trim-handle left" data-trim="left"></span><span class="trim-handle right" data-trim="right"></span>' : ""}
  </article>`;
}
function audioClipTemplate(item, state, storylineTop, clipHeight, assets) {
    const left = item.timelineStart * state.pixelsPerSecond;
    const width = Math.max(18, item.duration * state.pixelsPerSecond);
    const top = storylineTop + clipHeight + 14 + (Math.abs(item.lane) - 1) * 44;
    const selected = state.selectedItemIds.includes(item.id) || item.id === state.selectedItemId;
    const primary = item.id === state.selectedItemId;
    const visualAttributes = item.timeMap ? "" : mediaVisualAttributes(assets.find((candidate) => candidate.id === item.assetId), item.sourceStart, item.duration, 0, Math.max(24, Math.min(256, Math.round(width / 3))));
    return `<article class="timeline-clip audio-clip ${selected ? "selected" : ""} ${primary ? "primary-selected" : ""} ${item.id === state.dropPreviewClipId ? "drop-preview" : ""}" draggable="${state.tool === "select"}" data-item-id="${escapeHtml(item.id)}" data-role="connected-audio" ${visualAttributes} style="left:${left}px;top:${top}px;width:${width}px;--clip-color:${item.colors.a}">
    <div class="audio-wave ${item.audio.muted ? "muted-audio" : ""}" style="--volume-scale:${volumeWaveScale(item.audio.gainDb)}"></div><div class="audio-level-line ${item.audio.muted ? "muted-audio" : ""}" style="top:${clampCss(50 - item.audio.gainDb, 8, 84)}%"></div>
    <div class="clip-label compact"><strong>${escapeHtml(item.name)}</strong><span>${item.audio.gainDb.toFixed(1)} dB</span></div>
    ${primary ? '<span class="trim-handle left" data-trim="left"></span><span class="trim-handle right" data-trim="right"></span>' : ""}
  </article>`;
}
function containerButton(item) {
    if (!item.container)
        return "";
    const label = item.container.kind === "multicam" ? "Open angle timeline" : item.container.kind === "audition" ? "Open active audition choice" : `Open ${item.container.kind} timeline`;
    return `<button class="scope-enter-button" data-action="enter-clip-scope" data-item-id="${escapeHtml(item.id)}" title="${escapeHtml(label)}">▣</button>`;
}
function anchorTemplate(item, storylineTop, laneStride, pixelsPerSecond) {
    const left = item.timelineStart * pixelsPerSecond;
    const top = storylineTop - item.lane * laneStride + 23;
    return `<span class="anchor-line" style="left:${left}px;top:${top}px;height:${storylineTop - top + 8}px"><i></i></span>`;
}
function transitionTemplate(transition, start, pps, top, clipHeight, selectedStart) {
    const width = Math.max(34, Math.min(76, transition.duration * pps));
    const selected = selectedStart !== null && Math.abs(selectedStart - start) < .0001;
    return `<button class="transition-block ${selected ? "selected" : ""}" data-action="transition" data-transition-start="${start}" data-transition-id="${escapeHtml(transition.id)}" style="left:${start * pps - width / 2}px;top:${top + 1}px;width:${width}px;height:${Math.max(22, clipHeight - 2)}px" title="${escapeHtml(transition.name)}">
    <span class="transition-surface"><i class="transition-shade left"></i><i class="transition-shade right"></i></span>
    <span class="transition-control left" aria-hidden="true">Ⅱ</span>
    <span class="transition-control center" aria-hidden="true"><i class="transition-bowtie-mark"></i></span>
    <span class="transition-control right" aria-hidden="true">Ⅱ</span>
    <span class="transition-wave" aria-hidden="true">${waveBars(24)}</span>
    <i class="transition-edge left"></i><i class="transition-edge right"></i>
  </button>`;
}
function clampCss(value, min, max) {
    return Math.min(max, Math.max(min, value));
}
/** Convert authored decibel gain into a bounded waveform display scale. */
function volumeWaveScale(gainDb) {
    const linear = Math.pow(10, gainDb / 20);
    return Math.round(Math.min(2.2, Math.max(0.12, linear)) * 1000) / 1000;
}
