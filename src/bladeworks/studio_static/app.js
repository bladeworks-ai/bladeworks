/**
 * Bladeworks browser editor MVP entry point.
 *
 * Architecture map
 * ================
 *
 * EditorRuntime
 *   -> Library/Event/Project snapshot + mock canvas or WebRTC preview
 * BladeworksEditorApp
 *   -> transaction history
 *   -> FCP-style workspace state and resizable panels
 *   -> event delegation for controls, gestures, and shortcuts
 *   -> Studio owns right-click and editor chords; Chrome only keeps typing fields
 *   -> accepted edits via runtime.commitEdit(EditOperation)
 *   -> draft gestures via applyEdit on an immutable transactionBase
 *   -> undo/redo via runtime.restoreProject only
 * UI templates
 *   -> dense Final Cut-inspired browser surface
 *
 * Product invariant:
 * The browser is a visual client. The local process remains the owner of
 * canonical FCPXML, media paths, exact rational timing, validation, realtime
 * Bladeworks playback, and final export. Every viewer/inspector widget
 * emits the same dotted parameter path so the runtime receives one coherent
 * edit protocol.
 *
 * Commit path:
 * 1. Gesture start freezes transactionBase (immutable ProjectSnapshot).
 * 2. Pointer-move drafts with applyEdit(transactionBase, op) for live preview.
 * 3. Pointer-up / accepted UI edits call commitEdit(label, operation) once
 *    (or a short sequence for multi-clip connected moves), never restoreProject.
 * 4. Undo/redo alone call restoreProject(snapshot, baseRevision).
 */
import { colorFromHex, defaultAudio, defaultTransform, defaultVideo, getPath, resetAudioVolume, } from "./clip-state.js";
import { emptyHistory, recordHistory, redoHistory, undoHistory } from "./history.js";
import { MIN_CLIP_DURATION, applyEdit, clipAtTime, findTimelineClip, insertionIndexAtTime, normalizeProject, projectDuration, validateProject, } from "./magnetic-timeline.js";
import { MOCK_CAPABILITIES, activeMockCapabilities, mockCapabilityForAction, mockCapabilityForParameter, mockNoticeTitle, } from "./mock-capabilities.js";
import { capabilityParameterNames, defaultCapabilityParameters, effectCatalogItem, transitionCatalogItem, } from "./capability-ui.js";
import { runtimeFromLocation } from "./runtime.js";
import { MediaVisualLoader } from "./media-visuals.js";
import { createMaskedEffect } from "./fcpxml.js";
import { transformOverlayGeometry } from "./viewer-geometry.js";
import { defaultMask, maskKeyframeKey, maskUiValues, updateMaskValue, withMaskKeyframes, } from "./mask-ui.js";
import { addRetimeHold, freezeRetime, setRetimeSegmentDuration, setRetimeSegmentRate, splitRetimeAt, } from "./retime-ui.js";
import { defaultScopeTarget, projectForScope, replaceClipContainer, replaceClipMetadata, replaceScopeProject, scopeBreadcrumbs, } from "./scope-ui.js";
import { canvasControlsTemplate, catalogBrowserTemplate, escapeHtml, inspectorTemplate, libraryTemplate, mediaGridTemplate, mediaTemplate, selectedStorylineEdit, selectedTimelineClip, shellTemplate, timelineIndexTemplate, timelineTemplate, timelineToolbarTemplate, topbarTemplate, transportTemplate, viewerControlStripTemplate, viewerToolbarTemplate, } from "./templates.js";
import { clamp, fitTimelinePixelsPerSecond, formatTimecode, randomId, resultForCurrentSelection, } from "./ui.js";
function clone(value) {
    return structuredClone(value);
}
function rationalTime(seconds, _fps) {
    // An empty raw value tells the codec this was authored in Studio. The codec
    // then quantizes against the Project's exact FCPXML frameDuration, including
    // rates such as 30000/1001 that cannot be reconstructed from `fps` alone.
    return { seconds, raw: "" };
}
function supportsPortableRetime(item) {
    return !["gap", "title", "caption", "generator"].includes(item.kind);
}
function capabilityInputValue(input, current) {
    if (input instanceof HTMLInputElement && input.type === "checkbox")
        return input.checked;
    if (input instanceof HTMLInputElement && input.type === "color") {
        const red = Number.parseInt(input.value.slice(1, 3), 16) / 255;
        const green = Number.parseInt(input.value.slice(3, 5), 16) / 255;
        const blue = Number.parseInt(input.value.slice(5, 7), 16) / 255;
        return typeof current === "object" && current !== null && "alpha" in current
            ? { red, green, blue, alpha: current.alpha }
            : { red, green, blue };
    }
    const component = input.getAttribute("data-component");
    if (component) {
        const base = typeof current === "object" && current !== null ? current : {};
        const siblings = input.closest(".capability-compound")?.querySelectorAll("input[data-component]") ?? [];
        const complete = Object.fromEntries([...siblings].map((field) => [field.dataset.component, field === input ? Number(input.value) : Number(field.value || 0)]));
        return { ...base, ...complete };
    }
    if (input instanceof HTMLInputElement && input.type === "range") {
        const actualMinimum = input.getAttribute("data-actual-min");
        const actualMaximum = input.getAttribute("data-actual-max");
        if (actualMinimum !== null && actualMaximum !== null) {
            const normalized = Number(input.value);
            const minimum = Number(actualMinimum);
            const maximum = Number(actualMaximum);
            if (normalized < 0)
                return Math.abs(minimum) * normalized / 100;
            return maximum * normalized / 100;
        }
        return Number(input.value);
    }
    if (input instanceof HTMLInputElement && input.type === "number")
        return Number(input.value);
    return input.value;
}
function rgbFromHex(value) {
    const rgba = colorFromHex(value);
    return { red: rgba.red, green: rgba.green, blue: rgba.blue };
}
function isTypingTarget(target) {
    return (target instanceof HTMLElement &&
        target.matches('input, textarea, select, [contenteditable="true"]'));
}
function dropRoleFromCanvas(canvas, clientY) {
    const shelf = canvas.querySelector(".storyline-shelf");
    if (!(shelf instanceof HTMLElement)) {
        return "storyline";
    }
    const rect = shelf.getBoundingClientRect();
    if (clientY < rect.top - 8) {
        return "connected-video";
    }
    if (clientY > rect.bottom + 8) {
        return "connected-audio";
    }
    return "storyline";
}
let blankDragImage = null;
/**
 * Hide the browser's native drag badge while Studio paints its own clip ghost.
 *
 * WebKit ignores a detached element passed to `setDragImage` and falls back to
 * its generic globe/link icon. Keep this transparent canvas attached offscreen
 * for the full gesture so only `timelineDragGhost` follows the pointer.
 *
 * Main callers: handleDragStart for media cards and timeline clips.
 */
function setBlankDragImage(dataTransfer) {
    blankDragImage?.remove();
    const blank = document.createElement("canvas");
    blank.width = 1;
    blank.height = 1;
    blank.setAttribute("aria-hidden", "true");
    blank.style.position = "fixed";
    blank.style.left = "-10px";
    blank.style.top = "-10px";
    blank.style.pointerEvents = "none";
    document.body.appendChild(blank);
    blankDragImage = blank;
    dataTransfer.setDragImage(blank, 0, 0);
}
function removeBlankDragImage() {
    blankDragImage?.remove();
    blankDragImage = null;
}
function connectedLaneForClip(clip) {
    return clip.kind === "audio" || clip.role === "connected-audio" ? -1 : 1;
}
function naturalConnectRole(clip) {
    return connectedLaneForClip(clip) < 0 ? "connected-audio" : "connected-video";
}
function emptyProject() {
    return {
        revision: 0,
        id: "",
        libraryId: "",
        eventId: "",
        name: "",
        fps: 30,
        width: 1920,
        height: 1080,
        spine: [],
        connected: [],
        transitions: [],
        proposal: null,
    };
}
/** Return the last seekable frame, because the project duration is an exclusive end time. */
function finalProjectFrameTime(project) {
    const duration = projectDuration(project);
    if (duration <= 0)
        return 0;
    return Math.max(0, duration - 1 / Math.max(1, project.fps));
}
/** Draft helpers — local applyEdit only. Accepted commits go through commitEdit. */
function updateItemPath(project, clipId, path, value) {
    return applyEdit(project, { type: "updateClipPath", clipId, path, value });
}
function updateItem(project, clipId, patch) {
    return applyEdit(project, { type: "updateClip", clipId, patch });
}
function trimItem(project, clipId, edge, delta) {
    return applyEdit(project, {
        type: "trim",
        clipId,
        edge: edge === "left" ? "start" : "end",
        delta,
    });
}
function moveConnectedItem(project, clipId, timelineStart) {
    return applyEdit(project, { type: "moveConnected", clipId, timelineStart });
}
function splitOperationAtTime(project, time) {
    const clip = clipAtTime(project, time);
    if (!clip) {
        return null;
    }
    const offset = time - clip.timelineStart;
    if (offset <= 0.25 || offset >= clip.duration - 0.25) {
        return null;
    }
    return {
        type: "split",
        clipId: clip.id,
        offset,
        trailingClipId: randomId("split"),
    };
}
function insertOperationAtTime(project, asset, time) {
    const clip = timelineClipFromAsset(randomId("clip"), asset);
    return {
        type: "insert",
        clip,
        index: insertionIndexAtTime(project, time),
    };
}
function connectOperationAtTime(project, asset, time) {
    const anchor = clipAtTime(project, time);
    if (!anchor) {
        return null;
    }
    const clip = {
        ...timelineClipFromAsset(randomId("connected"), asset),
        anchorId: anchor.id,
        anchorOffset: Math.max(0, time - anchor.timelineStart),
        lane: asset.kind === "audio" ? -1 : 1,
    };
    return { type: "connect", clip };
}
/** Build an editable clip from either a fixture asset or Bladeworks inventory. */
function timelineClipFromAsset(id, asset) {
    return {
        id,
        assetId: asset.id,
        name: asset.name,
        kind: asset.kind,
        // Preserve the probed audio-stream presence so a video-only source never
        // serializes a phantom audio track. Omitted (not set to undefined, per
        // exactOptionalPropertyTypes) for fixture assets that carry no probe flag.
        ...(asset.hasAudio !== undefined ? { hasAudio: asset.hasAudio } : {}),
        ...(asset.duration > 0 ? { sourceDuration: asset.duration } : {}),
        ...(asset.width ? { sourceWidth: asset.width } : {}),
        ...(asset.height ? { sourceHeight: asset.height } : {}),
        ...(asset.frameDurationRaw ? { sourceFrameDurationRaw: asset.frameDurationRaw } : {}),
        role: "storyline",
        sourceStart: 0,
        duration: Math.max(MIN_CLIP_DURATION, Math.min(asset.duration || 5, 5)),
        timelineStart: 0,
        colors: { ...asset.colors },
        transform: defaultTransform(),
        video: defaultVideo(),
        audio: defaultAudio(),
        keyframes: {},
        timeMap: null,
        text: asset.kind === "title" ? asset.name : null,
        textStyle: null,
        caption: null,
        generatorColor: null,
        markers: [],
        effects: [],
        effectStack: [],
    };
}
const LOCALHOST_BLOCKED_MOCKS = new Set([
    "audio-enhancements",
    "color-wheels",
    "portable-video-analysis",
    "ken-burns-editor",
    "effects-browser",
    "transitions-browser",
]);
class BladeworksEditorApp {
    root;
    runtime;
    mediaVisuals;
    capabilities;
    appHistory;
    libraries;
    assets;
    preview;
    toastTimer;
    mockNoticeTimer;
    // setInterval handle for the backend health probe (null when unmonitored).
    healthMonitor;
    // Cancellation token for the in-flight preview warm-up watcher (null when
    // none is running). Setting `cancelled = true` stops its frame callbacks.
    previewWarmupToken;
    suppressNextClick;
    playbackFrame;
    lastPlaybackTime;
    playbackStopTime;
    parameterBase;
    pointerSession;
    contextMenu;
    transactionBase;
    timelineDrag;
    timelineDragGhost;
    dragLeaveTimer;
    rootProject;
    activeScopeId;
    scopePath;
    scopeNavigation;
    scopeNavigationIndex;
    // Monotonic token for concurrent Project selections. A superseded selection
    // must not adopt its (older) result into the UI after a newer one started.
    projectSelectionSeq = 0;
    exportResolution;
    exportProfile;
    exportJob;
    state;
    constructor(root) {
        this.root = root;
        this.runtime = runtimeFromLocation();
        this.mediaVisuals = new MediaVisualLoader(this.runtime);
        this.capabilities = null;
        this.appHistory = emptyHistory();
        this.libraries = [];
        this.assets = [];
        this.preview = null;
        this.toastTimer = null;
        this.mockNoticeTimer = null;
        this.healthMonitor = null;
        this.previewWarmupToken = null;
        this.suppressNextClick = false;
        this.playbackFrame = 0;
        this.lastPlaybackTime = 0;
        this.playbackStopTime = null;
        this.parameterBase = null;
        this.pointerSession = null;
        this.contextMenu = null;
        this.transactionBase = null;
        this.timelineDrag = null;
        this.timelineDragGhost = null;
        this.dragLeaveTimer = null;
        this.rootProject = emptyProject();
        this.activeScopeId = null;
        this.scopePath = [];
        this.scopeNavigation = [{ scopeId: null, path: [] }];
        this.scopeNavigationIndex = 0;
        this.exportResolution = 1080;
        this.exportProfile = "delivery";
        this.exportJob = null;
        this.state = {
            library: null,
            project: emptyProject(),
            selectedProjectId: "",
            selectedEventId: "",
            selectedItemId: null,
            selectedItemIds: [],
            selectedTransitionStart: null,
            selectedAssetId: null,
            currentTime: 0,
            playing: false,
            volume: 0.82,
            pixelsPerSecond: 34,
            tool: "select",
            snapping: true,
            skimming: true,
            audioSkimming: false,
            solo: false,
            continuousScroll: false,
            mediaQuery: "",
            mediaTab: "media",
            browserView: "list",
            librarySource: "libraries",
            browserScope: "all",
            browserSort: "date",
            inspectorTab: "video",
            viewerTool: "transform",
            guides: false,
            viewerView: {
                actionSafe: false,
                titleSafe: false,
                horizon: false,
                quality: "better-quality",
                background: "black",
            },
            overscan: false,
            viewerZoom: 50,
            panels: { library: true, browser: true, timeline: true, inspector: true },
            layout: {
                libraryWidth: 150,
                browserWidth: 485,
                inspectorWidth: window.innerWidth >= 1400 ? 404 : 330,
                timelineHeight: 330,
            },
            clipAppearance: { mode: 4, height: 58, showNames: true, showRoles: false },
            timelineIndexOpen: false,
            timelineIndexTab: "clips",
            timelineIndexQuery: "",
            loopPlayback: false,
            timelineNotes: false,
            kenBurnsLoop: false,
            activePopover: null,
            expandedEventIds: [],
            activeBrowser: null,
            browserCategory: null,
            browserQuery: "",
            markerEditorId: null,
            rangeSelection: null,
            marqueeSelection: null,
            dropPreviewClipId: null,
            activeColorZone: "midtones",
            connectionMode: this.runtime.mode,
            connectionMessage: this.runtime.mode === "localhost"
                ? "Connecting to localhost renderer…"
                : "Standalone fixture mode",
            // Optimistic until the first /healthz probe reports back, so we do not
            // flash a red "offline" light during normal startup.
            connectionHealthy: true,
            projectEditable: true,
            projectEditReasons: [],
            isSaving: false,
        };
    }
    historyAdapter() {
        return {
            canUndo: () => this.appHistory.past.length > 0,
            canRedo: () => this.appHistory.future.length > 0,
        };
    }
    /** Mirror the codec's fail-loud editability decision into visible UI state. */
    applyProjectEditability() {
        const access = this.runtime.projectEditability(this.state.project.id);
        const scope = this.activeScopeId ? this.rootProject.scopes?.[this.activeScopeId] : null;
        this.state.projectEditable = access.editable && (scope?.editable ?? true);
        this.state.projectEditReasons = [
            ...access.reasons,
            ...(scope && !scope.editable ? scope.reasons : []),
        ];
        if (access.degraded && access.editable) {
            this.state.connectionMessage = access.warnings.length
                ? `Compatibility warning: ${access.warnings.join("; ")}`
                : "Project is editable with renderer compatibility warnings";
        }
        else if (!this.state.projectEditable) {
            this.state.connectionMessage = `Read-only: ${this.state.projectEditReasons.join("; ")}`;
        }
    }
    /** Refresh Library/Event/Project navigation after a localhost source boundary. */
    adoptRuntimeCatalog() {
        const snapshot = this.runtime.snapshot();
        this.libraries = snapshot.libraries.map((library) => clone(library));
        this.assets = snapshot.assets.map((asset) => ({ ...asset, tags: [...asset.tags] }));
        this.state.library = this.libraries.find((library) => library.id === this.state.project.libraryId) ?? this.libraries[0] ?? null;
    }
    /** Adopt the latest valid runtime source after an optimistic save conflict. */
    recoverRuntimeProject(preferredProjectId) {
        const snapshot = this.runtime.snapshot();
        const project = snapshot.projects[preferredProjectId]
            ?? snapshot.projects[snapshot.activeProjectId]
            ?? Object.values(snapshot.projects)[0];
        if (!project) {
            throw new Error("The reloaded library does not contain a Project.");
        }
        this.adoptCanonicalProject(project);
        this.state.selectedProjectId = project.id;
        this.state.selectedEventId = project.eventId;
        if (!this.state.expandedEventIds.includes(project.eventId)) {
            this.state.expandedEventIds = [...this.state.expandedEventIds, project.eventId];
        }
        this.state.selectedItemId = this.state.project.spine[0]?.id ?? this.state.project.connected[0]?.id ?? null;
        this.state.selectedItemIds = this.state.selectedItemId ? [this.state.selectedItemId] : [];
        this.applyProjectEditability();
        this.adoptRuntimeCatalog();
    }
    /**
     * Freeze the project at gesture start so pointer-move drafts stay relative
     * to one immutable base revision.
     *
     * Main callers: knob / onscreen / trim / connected-move gesture starts.
     */
    currentSnapshot() {
        if (!this.transactionBase) {
            this.transactionBase = clone(this.state.project);
        }
        return this.transactionBase;
    }
    previewRealtime(payload) {
        if (payload.type === "play") {
            this.preview?.play();
        }
        else if (payload.type === "pause") {
            this.preview?.pause();
        }
        else if (payload.type === "seek") {
            this.preview?.seek(payload.time ?? this.state.currentTime);
        }
        this.syncPreview();
    }
    syncPreview() {
        this.state.currentTime = clamp(this.state.currentTime, 0, finalProjectFrameTime(this.state.project));
        this.preview?.setState({
            project: this.rootProject,
            playhead: this.state.currentTime,
            selectedClipId: this.activeScopeId ? null : this.state.selectedItemId,
        });
    }
    /** Adopt one canonical root Project while preserving a valid nested location. */
    adoptCanonicalProject(project) {
        this.rootProject = normalizeProject(project);
        if (this.activeScopeId && !this.rootProject.scopes?.[this.activeScopeId]) {
            this.activeScopeId = null;
            this.scopePath = [];
            this.scopeNavigation = [{ scopeId: null, path: [] }];
            this.scopeNavigationIndex = 0;
        }
        this.state.project = projectForScope(this.rootProject, this.activeScopeId);
    }
    enterScope(scopeId, path, record = true) {
        if (scopeId && !this.rootProject.scopes?.[scopeId]) {
            this.showToast(`Nested timeline ${scopeId} is unavailable.`, "error");
            return;
        }
        const nextPath = scopeId ? [...(path ?? [...this.scopePath, scopeId])] : [];
        this.activeScopeId = scopeId;
        this.scopePath = nextPath;
        if (record) {
            this.scopeNavigation = this.scopeNavigation.slice(0, this.scopeNavigationIndex + 1);
            this.scopeNavigation.push({ scopeId, path: nextPath });
            this.scopeNavigationIndex = this.scopeNavigation.length - 1;
        }
        this.state.project = projectForScope(this.rootProject, scopeId);
        this.state.selectedItemId = this.state.project.spine[0]?.id ?? this.state.project.connected[0]?.id ?? null;
        this.state.selectedItemIds = this.state.selectedItemId ? [this.state.selectedItemId] : [];
        this.state.selectedTransitionStart = null;
        this.state.currentTime = 0;
        this.applyProjectEditability();
        this.renderAll();
        this.syncPreview();
    }
    navigateScopeHistory(direction) {
        const index = this.scopeNavigationIndex + direction;
        const target = this.scopeNavigation[index];
        if (!target)
            return;
        this.scopeNavigationIndex = index;
        this.enterScope(target.scopeId, target.path, false);
    }
    enterSelectedContainer(itemId) {
        const item = itemId ? this.itemById(itemId) : null;
        const scopeId = item ? defaultScopeTarget(this.rootProject, item) : null;
        if (!scopeId) {
            this.showToast("This clip has no editable nested timeline.", "info");
            return;
        }
        this.enterScope(scopeId);
    }
    async start() {
        this.root.innerHTML = shellTemplate();
        this.bindEvents();
        const [bootstrap, capabilities] = await Promise.all([
            this.runtime.bootstrap(),
            this.runtime.capabilities(),
        ]);
        this.capabilities = capabilities;
        this.assets = bootstrap.assets.map((asset) => ({ ...asset, tags: [...asset.tags] }));
        const project = bootstrap.projects[bootstrap.activeProjectId];
        if (!project) {
            throw new Error(`Bootstrap selected missing Project ${bootstrap.activeProjectId}.`);
        }
        this.libraries = bootstrap.libraries.map((library) => clone(library));
        this.state.library = this.libraries.find((library) => library.id === project.libraryId) ?? this.libraries[0] ?? null;
        this.adoptCanonicalProject(project);
        this.applyProjectEditability();
        this.state.selectedProjectId = project.id;
        this.state.selectedEventId = project.eventId;
        this.state.expandedEventIds = this.libraries.flatMap((library) => library.events.map((event) => event.id));
        const initialSelection = this.state.project.spine[0]?.id ?? this.state.project.connected[0]?.id ?? null;
        this.state.selectedItemId = initialSelection;
        this.state.selectedItemIds = initialSelection ? [initialSelection] : [];
        this.state.selectedAssetId = this.assets[0]?.id ?? null;
        this.state.connectionMode = this.runtime.mode;
        this.state.connectionMessage =
            this.runtime.mode === "localhost"
                ? "Localhost runtime connected"
                : "Standalone UI mode · synthetic media";
        this.appHistory = emptyHistory();
        this.renderMockInventory();
        this.renderAll();
        this.fitTimelineAfterProjectOpen();
        this.warnMediaInventoryFailures();
        this.warnUnopenableProjects();
        await this.attachPreview();
        this.syncPreview();
        this.startHealthMonitor();
    }
    /** Find one Project's sidebar summary (carries the catalog `openError`). */
    projectSummary(projectId) {
        for (const library of this.libraries) {
            for (const event of library.events) {
                const project = event.projects.find((candidate) => candidate.id === projectId);
                if (project) {
                    return project;
                }
            }
        }
        return null;
    }
    /**
     * Tell the user at startup which Projects Bladeworks could not compile.
     *
     * Why this exists: the library opened anyway (the runtime skipped them when
     * choosing the active Project), so without this toast the only hint would be
     * a greyed row in the sidebar. The row's tooltip carries the full error.
     */
    warnUnopenableProjects() {
        const failed = this.libraries.flatMap((library) => library.events.flatMap((event) => event.projects.filter((project) => project.openError !== null)));
        if (!failed.length) {
            return;
        }
        const names = failed.map((project) => `"${project.name}"`).join(", ");
        const count = failed.length === 1 ? "1 Project" : `${failed.length} Projects`;
        this.showToast(`${count} could not be compiled and cannot be opened: ${names}. Hover the greyed entry in the Libraries sidebar for the reason.`, "error");
    }
    /**
     * Poll the backend's health so the topbar status light reflects reality
     * instead of always showing green in localhost mode. The mock runtime has no
     * `probeHealth`, so this is a no-op there (the light stays neutral). On each
     * tick we flip `connectionHealthy` and repaint only if it actually changed,
     * to avoid needless re-renders.
     */
    startHealthMonitor() {
        if (typeof this.runtime.probeHealth !== "function") {
            return;
        }
        const probe = this.runtime.probeHealth.bind(this.runtime);
        const tick = async () => {
            const healthy = await probe();
            if (healthy !== this.state.connectionHealthy) {
                this.state.connectionHealthy = healthy;
                this.state.connectionMessage = healthy
                    ? "Localhost runtime connected"
                    : "Localhost backend unreachable";
                this.renderTopbar();
            }
        };
        void tick();
        this.healthMonitor = window.setInterval(() => void tick(), 5000);
    }
    /**
     * Ask Chromium to deliver reserved chords (Cmd-N, Cmd-T, Cmd-W) to the page.
     * This succeeds in an `--app` window. A normal Chrome tab still keeps Cmd-N.
     */
    async lockEditorKeys() {
        const keyboard = navigator.keyboard;
        if (!keyboard) {
            return;
        }
        try {
            await keyboard.lock(["KeyN", "KeyT", "KeyW", "KeyR", "KeyS", "KeyO", "KeyP"]);
        }
        catch {
            return;
        }
    }
    bindEvents() {
        this.root.addEventListener("click", (event) => this.handleClick(event));
        this.root.addEventListener("dblclick", (event) => this.handleDoubleClick(event));
        this.root.addEventListener("input", (event) => this.handleInput(event));
        this.root.addEventListener("change", (event) => this.handleChange(event));
        this.root.addEventListener("keydown", (event) => this.handleControlKeyDown(event));
        this.root.addEventListener("dragstart", (event) => this.handleDragStart(event));
        this.root.addEventListener("dragover", (event) => this.handleDragOver(event));
        this.root.addEventListener("dragleave", (event) => this.handleDragLeave(event));
        this.root.addEventListener("drop", (event) => this.handleDrop(event));
        this.root.addEventListener("dragend", () => this.handleDragEnd());
        this.root.addEventListener("pointerdown", (event) => this.handlePointerDown(event));
        this.root.addEventListener("pointermove", (event) => this.handlePointerMove(event));
        this.root.addEventListener("pointerleave", (event) => this.handlePointerLeave(event));
        this.root.addEventListener("contextmenu", (event) => this.handleContextMenu(event));
        window.addEventListener("keydown", (event) => this.handleKeyDown(event), { capture: true, passive: false });
        window.addEventListener("resize", () => this.updateCanvasControls());
        void this.lockEditorKeys();
        window.addEventListener("beforeunload", () => this.preview?.destroy());
    }
    renderAll() {
        if (!this.state.library) {
            return;
        }
        this.reconcileSelection();
        this.root.classList.toggle("project-read-only", !this.state.projectEditable);
        this.applyLayout();
        this.renderTopbar();
        this.el("library-content").innerHTML = libraryTemplate(this.libraries, this.state.selectedProjectId, this.state.selectedEventId, this.state.librarySource, this.state.connectionMode, this.state.expandedEventIds);
        this.el("media-content").innerHTML = mediaTemplate(this.assets, {
            query: this.state.mediaQuery,
            activeTab: this.state.mediaTab,
            view: this.state.browserView,
            selectedAssetId: this.state.selectedAssetId,
            selectedProjectId: this.state.selectedProjectId,
            eventProjects: this.selectedEventProjects(),
            eventName: this.selectedEventName(),
            sort: this.state.browserSort,
            scope: this.state.browserScope,
            activePopover: this.state.activePopover,
        });
        this.mediaVisuals.decorate(this.el("media-content"));
        this.renderViewer();
        this.renderInspector();
        this.renderTimeline();
        this.updatePlaybackDom();
    }
    renderViewer() {
        const selectedItem = this.selectedItem();
        this.el("viewer-toolbar").innerHTML = viewerToolbarTemplate({
            project: this.state.project,
            selectedItem,
            viewerZoom: this.state.viewerZoom,
            viewerTool: this.state.viewerTool,
            activePopover: this.state.activePopover,
            viewerView: this.state.viewerView,
        });
        const controlsHtml = canvasControlsTemplate({
            selectedItem,
            viewerTool: this.state.viewerTool,
            playing: this.state.playing,
            currentTime: this.state.currentTime,
            fps: this.state.project.fps,
            connectionMode: this.state.connectionMode,
        });
        const stripHtml = viewerControlStripTemplate({
            selectedItem,
            viewerTool: this.state.viewerTool,
            playing: this.state.playing,
            currentTime: this.state.currentTime,
            fps: this.state.project.fps,
            kenBurnsLoop: this.state.kenBurnsLoop,
            connectionMode: this.state.connectionMode,
        });
        if (this.state.connectionMode === "localhost") {
            this.el("viewer-control-strip").innerHTML = "";
            this.el("canvas-controls").innerHTML = "";
            this.el("live-viewer-control-strip").innerHTML = stripHtml;
            this.el("live-canvas-controls").innerHTML = controlsHtml;
        }
        else {
            this.el("live-viewer-control-strip").innerHTML = "";
            this.el("live-canvas-controls").innerHTML = "";
            this.el("viewer-control-strip").innerHTML = stripHtml;
            this.el("canvas-controls").innerHTML = controlsHtml;
        }
        this.el("transport").innerHTML = transportTemplate({
            currentTime: this.state.currentTime,
            duration: projectDuration(this.state.project),
            playing: this.state.playing,
            volume: this.state.volume,
            connectionMode: this.state.connectionMode,
        });
        this.updateProgramFrame();
        this.updateCanvasControls();
        this.updateViewerGuides();
    }
    renderTopbar() {
        document.title = this.state.library
            ? `Bladeworks Studio: ${this.state.library.name}`
            : "Bladeworks Studio";
        this.el("topbar").innerHTML = topbarTemplate({
            ...this.state,
            project: this.rootProject,
            ...(this.capabilities ? { capabilities: this.capabilities } : {}),
            exportResolution: this.exportResolution,
            exportProfile: this.exportProfile,
            exportProgress: this.exportJob ? {
                status: this.exportJob.progress.status,
                completedFrames: this.exportJob.progress.completedFrames,
                totalFrames: this.exportJob.progress.totalFrames,
            } : null,
        });
    }
    renderInspector() {
        this.el("inspector-content").innerHTML = inspectorTemplate({
            selectedItem: this.selectedItem(),
            inspectorTab: this.state.inspectorTab,
            currentTime: this.state.currentTime,
            fps: this.state.project.fps,
            activeColorZone: this.state.activeColorZone,
            selectionCount: this.state.selectedItemIds.length,
            connectionMode: this.state.connectionMode,
            capabilities: this.requireCapabilities(),
            selectedTransition: this.selectedTransition(),
            rootProject: this.rootProject,
        });
    }
    renderTimeline() {
        this.el("timeline-toolbar").innerHTML = timelineToolbarTemplate({
            ...this.state,
            history: this.historyAdapter(),
            scopeBreadcrumbs: scopeBreadcrumbs(this.rootProject, this.activeScopeId, this.scopePath),
            scopeCanGoBack: this.scopeNavigationIndex > 0,
            scopeCanGoForward: this.scopeNavigationIndex < this.scopeNavigation.length - 1,
        });
        this.el("timeline-index").innerHTML = timelineIndexTemplate(this.state);
        this.el("timeline-body").classList.toggle("index-open", this.state.timelineIndexOpen);
        this.el("timeline-content").innerHTML = timelineTemplate({ ...this.state, activeScopeId: this.activeScopeId }, this.assets);
        this.mediaVisuals.decorate(this.el("timeline-content"));
        this.el("catalog-browser").innerHTML = catalogBrowserTemplate({
            ...this.state,
            capabilities: this.requireCapabilities(),
        });
        this.el("timeline-body").classList.toggle("browser-open", this.state.activeBrowser !== null);
        this.updatePlaybackDom();
    }
    applyLayout() {
        const app = this.el("editor-app");
        app.style.setProperty("--library-width", this.state.panels.library ? `${this.state.layout.libraryWidth}px` : "0px");
        app.style.setProperty("--browser-width", this.state.panels.browser ? `${this.state.layout.browserWidth}px` : "0px");
        app.style.setProperty("--inspector-width", this.state.panels.inspector ? `${this.state.layout.inspectorWidth}px` : "0px");
        app.style.setProperty("--timeline-height", this.state.panels.timeline ? `${this.state.layout.timelineHeight}px` : "0px");
        const panels = ["library", "browser", "inspector", "timeline"];
        for (const panel of panels) {
            app.classList.toggle(`hide-${panel}`, !this.state.panels[panel]);
        }
        this.updateCanvasControls();
    }
    el(id) {
        const element = document.getElementById(id);
        if (!element) {
            throw new Error(`Editor shell is missing #${id}`);
        }
        return element;
    }
    requireCapabilities() {
        if (!this.capabilities) {
            throw new Error("Bladeworks capabilities have not loaded.");
        }
        return this.capabilities;
    }
    selectedTransition() {
        if (this.state.selectedTransitionStart === null)
            return null;
        return this.state.project.transitions.find((transition) => {
            const right = this.state.project.spine.find((clip) => clip.id === transition.rightItemId);
            return right && Math.abs(right.timelineStart - this.state.selectedTransitionStart) < 1e-6;
        }) ?? null;
    }
    selectedItem() {
        const id = this.state.selectedItemId;
        if (!id) {
            return null;
        }
        return ([...this.state.project.spine, ...this.state.project.connected].find((item) => item.id === id) ??
            null);
    }
    selectedItems() {
        const wanted = new Set(this.state.selectedItemIds);
        return [...this.state.project.spine, ...this.state.project.connected].filter((item) => wanted.has(item.id));
    }
    /**
     * Replace or extend the timeline selection while retaining one primary item
     * for the viewer and inspector.
     *
     * Main callers: marquee selection, click modifiers, delete cleanup.
     */
    setTimelineSelection(itemIds, primaryId = null) {
        const valid = [...new Set(itemIds)].filter((id) => Boolean(this.itemById(id)));
        this.state.selectedItemIds = valid;
        this.state.selectedItemId =
            primaryId && valid.includes(primaryId) ? primaryId : (valid.at(-1) ?? null);
        this.state.selectedTransitionStart = null;
    }
    toggleTimelineSelection(itemId) {
        const next = new Set(this.state.selectedItemIds);
        if (next.has(itemId)) {
            next.delete(itemId);
        }
        else {
            next.add(itemId);
        }
        this.setTimelineSelection([...next], next.has(itemId) ? itemId : ([...next].at(-1) ?? null));
    }
    clearTimelineSelection() {
        this.state.selectedItemIds = [];
        this.state.selectedItemId = null;
        this.state.selectedTransitionStart = null;
    }
    selectedEventName() {
        return (this.state.library?.events.find((event) => event.id === this.state.selectedEventId)?.name ??
            "All Clips");
    }
    selectedEventProjects() {
        return this.state.library?.events.find((event) => event.id === this.state.selectedEventId)?.projects ?? [];
    }
    reconcileSelection() {
        const live = new Set([...this.state.project.spine, ...this.state.project.connected].map((item) => item.id));
        this.state.selectedItemIds = this.state.selectedItemIds.filter((id) => live.has(id));
        if (this.state.selectedItemId && !live.has(this.state.selectedItemId)) {
            this.state.selectedItemId = null;
        }
        if (this.state.selectedItemId &&
            !this.state.selectedItemIds.includes(this.state.selectedItemId)) {
            this.state.selectedItemIds.push(this.state.selectedItemId);
        }
        if (!this.state.selectedItemId && this.state.selectedItemIds.length) {
            this.state.selectedItemId = this.state.selectedItemIds.at(-1) ?? null;
        }
    }
    suppressGeneratedClick() {
        this.suppressNextClick = true;
        setTimeout(() => {
            this.suppressNextClick = false;
        }, 80);
    }
    orderedTimelineItems() {
        const rows = [
            ...this.state.project.spine.map((item) => ({ item, lane: 0 })),
            ...this.state.project.connected.map((item) => ({ item, lane: item.lane })),
        ];
        rows.sort((a, b) => {
            if (a.item.timelineStart !== b.item.timelineStart) {
                return a.item.timelineStart - b.item.timelineStart;
            }
            if (a.lane !== b.lane) {
                return b.lane - a.lane;
            }
            return a.item.id.localeCompare(b.item.id);
        });
        return rows.map((row) => row.item);
    }
    selectItemWithModifiers(itemId, event) {
        const current = new Set(this.state.selectedItemIds);
        if (event.metaKey || event.ctrlKey) {
            if (current.has(itemId)) {
                current.delete(itemId);
            }
            else {
                current.add(itemId);
            }
            this.state.selectedItemIds = [...current];
            this.state.selectedItemId = current.has(itemId)
                ? itemId
                : (this.state.selectedItemIds.at(-1) ?? null);
            return;
        }
        if (event.shiftKey && this.state.selectedItemId) {
            const ordered = this.orderedTimelineItems();
            const from = ordered.findIndex((item) => item.id === this.state.selectedItemId);
            const to = ordered.findIndex((item) => item.id === itemId);
            if (from !== -1 && to !== -1) {
                const start = Math.min(from, to);
                const end = Math.max(from, to);
                this.state.selectedItemIds = ordered.slice(start, end + 1).map((item) => item.id);
                this.state.selectedItemId = itemId;
                return;
            }
        }
        this.state.selectedItemIds = [itemId];
        this.state.selectedItemId = itemId;
    }
    handleClick(event) {
        if (this.suppressNextClick) {
            this.suppressNextClick = false;
            return;
        }
        const target = event.target;
        if (!(target instanceof Element)) {
            return;
        }
        const menuAction = this.contextMenu?.contains(target)
            ? target.closest("[data-action]")
            : null;
        if (this.contextMenu && !menuAction) {
            this.closeStudioMenu();
            return;
        }
        this.closeStudioMenu();
        if (target.id === "mock-inventory") {
            this.closeMockInventory();
            return;
        }
        const actionElement = target.closest("[data-action]");
        if (actionElement) {
            this.runAction(actionElement.getAttribute("data-action"), actionElement, event);
            return;
        }
        const assetCard = target.closest(".asset-card");
        if (assetCard) {
            if (this.state.connectionMode !== "localhost") {
                this.showMockNotice("fixture-projects");
            }
            this.state.selectedAssetId = assetCard.getAttribute("data-asset-id");
            this.refreshBrowserResults();
            return;
        }
        const kenWindow = target.closest("[data-ken-window]");
        if (kenWindow && this.selectedItem()) {
            void this.commitParameter("video.crop.activeKenWindow", kenWindow.getAttribute("data-ken-window"), "Select Ken Burns window");
            return;
        }
        const clip = target.closest(".timeline-clip");
        if (clip) {
            const id = clip.getAttribute("data-item-id");
            if (!id) {
                return;
            }
            const mouse = event;
            if (this.state.tool === "blade" && clip.getAttribute("data-role") === "storyline") {
                const rect = clip.getBoundingClientRect();
                const item = this.state.project.spine.find((candidate) => candidate.id === id);
                if (!item) {
                    return;
                }
                const splitTime = item.timelineStart +
                    clamp((mouse.clientX - rect.left) / this.state.pixelsPerSecond, 0, item.duration);
                const operation = splitOperationAtTime(this.state.project, splitTime);
                if (operation) {
                    void this.commitEdit("Split clip", operation);
                }
                return;
            }
            if (this.state.tool === "zoom") {
                this.zoomTimelineAt(mouse.clientX, 1.35);
                return;
            }
            this.selectItemWithModifiers(id, mouse);
            this.state.selectedTransitionStart = null;
            this.renderViewer();
            this.renderTimeline();
            // Selection is intentionally independent from transport. Clicking a clip
            // changes the Inspector target while the playhead remains parked.
            this.renderInspector();
            return;
        }
        const canvas = target.closest(".timeline-canvas");
        if (canvas && !target.closest(".timeline-ruler") && canvas instanceof HTMLElement) {
            const mouse = event;
            const time = this.timeFromCanvasPointer(mouse.clientX, canvas);
            if (this.state.tool === "zoom") {
                this.zoomTimelineAt(mouse.clientX, mouse.altKey ? 0.72 : 1.35);
            }
            else {
                this.state.selectedItemIds = [];
                this.state.selectedItemId = null;
                this.state.selectedTransitionStart = null;
                this.renderInspector();
                this.seek(time);
                this.renderTimeline();
            }
        }
    }
    runAction(action, element, event) {
        if (!["show-mock-inventory", "close-mock-inventory"].includes(action ?? "")) {
            const mockId = mockCapabilityForAction(action, this.state.connectionMode);
            if (mockId) {
                this.showMockNotice(mockId);
                if (this.state.connectionMode === "localhost" && LOCALHOST_BLOCKED_MOCKS.has(mockId)) {
                    return;
                }
            }
        }
        switch (action) {
            case "scope-back":
                this.navigateScopeHistory(-1);
                break;
            case "scope-forward":
                this.navigateScopeHistory(1);
                break;
            case "enter-scope": {
                const scopeId = element.getAttribute("data-scope-id") || null;
                const index = scopeId ? this.scopePath.indexOf(scopeId) : -1;
                this.enterScope(scopeId, scopeId ? this.scopePath.slice(0, index + 1) : []);
                break;
            }
            case "enter-clip-scope":
                this.enterSelectedContainer(element.getAttribute("data-item-id"));
                break;
            case "toggle-panel":
                this.togglePanel(element.getAttribute("data-panel"));
                break;
            case "toggle-play":
                this.togglePlayback();
                break;
            case "step-back":
                this.seek(this.state.currentTime - 1 / this.state.project.fps);
                break;
            case "step-forward":
                this.seek(this.state.currentTime + 1 / this.state.project.fps);
                break;
            case "undo":
                this.undo();
                break;
            case "redo":
                this.redo();
                break;
            case "timeline-tool":
                this.setTimelineTool(element.getAttribute("data-tool"));
                break;
            case "toggle-popover":
                this.togglePopover(element.getAttribute("data-popover"));
                break;
            case "toggle-create-menu":
                this.state.activePopover = this.state.activePopover === "create" ? null : "create";
                this.renderTopbar();
                break;
            case "toggle-index":
                this.state.timelineIndexOpen = !this.state.timelineIndexOpen;
                this.state.activePopover = null;
                this.renderTimeline();
                break;
            case "open-index-tab":
                this.state.timelineIndexOpen = true;
                this.state.timelineIndexTab = (element.getAttribute("data-tab") ??
                    "clips");
                this.renderTimeline();
                break;
            case "timeline-index-tab":
                this.state.timelineIndexTab = (element.getAttribute("data-tab") ??
                    "clips");
                this.renderTimeline();
                break;
            case "select-index-item":
                this.selectTimelineItem(element.getAttribute("data-item-id"));
                break;
            case "seek-index-time":
                this.seek(Number(element.getAttribute("data-time") ?? 0));
                break;
            case "filter-index-role":
                this.state.timelineIndexTab = "clips";
                this.state.timelineIndexQuery = roleQuery(element.getAttribute("data-role"));
                this.renderTimeline();
                break;
            case "toggle-snapping":
                this.state.snapping = !this.state.snapping;
                this.renderTimeline();
                break;
            case "toggle-skimming":
                this.state.skimming = !this.state.skimming;
                this.renderTimeline();
                break;
            case "toggle-continuous-scroll":
                this.state.continuousScroll = !this.state.continuousScroll;
                this.renderTimeline();
                break;
            case "clip-appearance-mode":
                this.state.clipAppearance = {
                    ...this.state.clipAppearance,
                    mode: Number(element.getAttribute("data-mode")),
                };
                this.state.activePopover = null;
                this.renderTimeline();
                break;
            case "fit-timeline":
                this.fitTimeline();
                this.state.activePopover = null;
                break;
            case "clear-browser-search":
                this.state.mediaQuery = "";
                this.refreshBrowser();
                break;
            case "browser-scope":
                this.togglePopover("browser-scope");
                this.refreshBrowser();
                break;
            case "set-browser-scope":
                this.state.browserScope = (element.getAttribute("data-value") ?? "all");
                this.state.activePopover = null;
                this.refreshBrowser();
                break;
            case "browser-sort":
                this.togglePopover("browser-sort");
                this.refreshBrowser();
                break;
            case "set-browser-sort":
                this.state.browserSort = (element.getAttribute("data-value") ?? "date");
                this.state.activePopover = null;
                this.refreshBrowser();
                break;
            case "browser-view":
                this.state.browserView = (element.getAttribute("data-view") ?? "list");
                this.state.activePopover = null;
                this.refreshBrowser();
                break;
            case "media-tab":
                this.state.mediaTab = (element.getAttribute("data-tab") ?? "media");
                this.state.activePopover = null;
                this.refreshBrowser();
                break;
            case "library-source":
                this.selectLibrarySource(element.getAttribute("data-source"));
                break;
            case "show-generators":
                this.state.mediaTab = "titles";
                this.state.panels = { ...this.state.panels, browser: true };
                this.applyLayout();
                this.refreshBrowser();
                break;
            case "toggle-event": {
                const eventId = element.getAttribute("data-event-id");
                if (!eventId)
                    break;
                this.state.selectedEventId = eventId;
                this.state.expandedEventIds = this.state.expandedEventIds.includes(eventId)
                    ? this.state.expandedEventIds.filter((candidate) => candidate !== eventId)
                    : [...this.state.expandedEventIds, eventId];
                this.state.library = this.libraries.find((library) => library.events.some((candidate) => candidate.id === this.state.selectedEventId)) ?? this.state.library;
                if (this.runtime.mode === "localhost" && this.runtime.mediaForEvent) {
                    this.assets = this.runtime.mediaForEvent(this.state.selectedEventId)
                        .map((asset) => ({ ...asset, tags: [...asset.tags] }));
                }
                this.renderAll();
                break;
            }
            case "select-project":
                void this.selectProject(element.getAttribute("data-project-id"));
                break;
            case "inspector-tab":
                this.state.inspectorTab = (element.getAttribute("data-tab") ?? "video");
                this.renderInspector();
                break;
            case "toggle-viewer-tools":
                this.toggleViewerToolPopover();
                break;
            case "viewer-zoom-menu":
                this.togglePopover("viewer-zoom");
                this.renderViewer();
                break;
            case "viewer-view-menu":
                this.togglePopover("viewer-view");
                this.renderViewer();
                break;
            case "set-viewer-zoom":
                this.setViewerZoom(element.getAttribute("data-value"));
                break;
            case "set-view-option":
                this.setViewerOption(element.getAttribute("data-option"), element.getAttribute("data-value"));
                break;
            case "viewer-tool":
                void this.activateViewerTool(element.getAttribute("data-tool"));
                break;
            case "viewer-done":
                this.state.viewerTool = "none";
                this.renderViewer();
                break;
            case "crop-mode":
                void this.setCropMode(element.getAttribute("data-mode"));
                break;
            case "loop-ken-burns":
                this.state.kenBurnsLoop = !this.state.kenBurnsLoop;
                this.showToast(this.state.kenBurnsLoop
                    ? "Ken Burns loop preview enabled."
                    : "Ken Burns loop preview disabled.", "info");
                this.renderViewer();
                break;
            case "toggle-overscan":
                this.state.overscan = !this.state.overscan;
                this.updateProgramFrame();
                break;
            case "toggle-inspector-section": {
                const section = element.closest(".inspector-section");
                this.toggleInspectorSection(section);
                break;
            }
            case "keyframe-menu":
                this.showKeyframeMenu(element);
                break;
            case "previous-keyframe":
                this.seekParameterKeyframe(element.getAttribute("data-path"), -1);
                break;
            case "next-keyframe":
                this.seekParameterKeyframe(element.getAttribute("data-path"), 1);
                break;
            case "clear-keyframes":
                void this.clearParameterKeyframes(element.getAttribute("data-path"));
                break;
            case "toggle-keyframe":
                void this.toggleKeyframe(element.getAttribute("data-path"));
                break;
            case "set-parameter":
                void this.commitParameter(element.getAttribute("data-path"), element.getAttribute("data-value"), "Set parameter");
                break;
            case "reset-section":
                void this.resetSection(element.getAttribute("data-section"));
                break;
            case "reset-path":
                void this.commitParameter(element.getAttribute("data-path"), false, "Reset control");
                break;
            case "toggle-audio-mute":
                void this.toggleBooleanPath("audio.muted");
                break;
            case "toggle-item-solo":
                void this.toggleBooleanPath("audio.solo");
                break;
            case "transition":
                this.state.selectedTransitionStart = Number(element.getAttribute("data-transition-start"));
                this.state.selectedItemIds = [];
                this.state.selectedItemId = null;
                this.renderTimeline();
                this.renderInspector();
                break;
            case "fullscreen":
                void this.el("viewer-wrap").requestFullscreen?.();
                break;
            case "seek-ruler":
                this.seekFromPointer(event);
                break;
            case "previous-edit":
                this.jumpEdit(-1);
                break;
            case "next-edit":
                this.jumpEdit(1);
                break;
            case "toggle-favorite":
                void this.toggleFavorite(element.getAttribute("data-asset-id"));
                break;
            case "export":
                if (this.exportJob) {
                    this.state.activePopover = "export";
                    this.renderTopbar();
                    break;
                }
                this.state.activePopover = this.state.activePopover === "export" ? null : "export";
                this.renderTopbar();
                break;
            case "export-profile":
            case "export-start":
                this.exportProject(this.exportProfile);
                break;
            case "export-cancel":
                this.cancelExport();
                break;
            case "import-media":
                this.mockImportMedia();
                break;
            case "refresh-media":
                void this.refreshMediaInventory();
                break;
            case "keyword-editor":
                this.state.timelineIndexOpen = true;
                this.state.timelineIndexTab = "tags";
                this.renderTimeline();
                break;
            case "save-effects-preset":
                break;
            case "add-tracker":
                break;
            case "new-event":
                void this.createNewEvent();
                break;
            case "dismiss-media-warning":
                this.dismissMediaWarning();
                break;
            case "new-project":
                void this.createNewProject();
                break;
            case "select-all":
                this.selectAllClips();
                break;
            case "blade-at-playhead": {
                const operation = splitOperationAtTime(this.state.project, this.state.currentTime);
                if (operation) {
                    void this.commitEdit("Blade at playhead", operation);
                }
                break;
            }
            case "ripple-delete":
                void this.deleteSelection();
                break;
            case "add-marker":
                void this.addMarkerAtPlayhead();
                break;
            case "toggle-loop-playback":
                this.state.loopPlayback = !this.state.loopPlayback;
                this.renderTimeline();
                this.showToast(this.state.loopPlayback ? "Loop Playback enabled." : "Loop Playback disabled.", "info");
                break;
            case "edit-timecode":
                this.editTimecode();
                break;
            case "viewer-color":
                this.state.inspectorTab = "color";
                this.state.panels = { ...this.state.panels, inspector: true };
                this.applyLayout();
                this.renderInspector();
                this.renderTopbar();
                break;
            case "select-color-zone":
                this.selectColorZone(element.getAttribute("data-zone"));
                break;
            case "show-mock-inventory":
                this.openMockInventory();
                break;
            case "close-mock-inventory":
                this.closeMockInventory();
                break;
            case "open-effects-browser":
                this.openBrowser("effects");
                break;
            case "open-transitions-browser":
                this.openBrowser("transitions");
                break;
            case "close-browser":
                this.state.activeBrowser = null;
                this.renderTimeline();
                break;
            case "browser-category":
                this.state.browserCategory = element.getAttribute("data-category") || null;
                this.renderTimeline();
                break;
            case "apply-effect":
                void this.applyEffectFromCatalog(element.getAttribute("data-capability-id"));
                break;
            case "apply-transition":
                void this.applyTransitionFromCatalog(element.getAttribute("data-capability-id"));
                break;
            case "toggle-effect":
                void this.toggleEffect(element.getAttribute("data-effect-id"));
                break;
            case "toggle-effects-section":
                void this.toggleEffectsSection(element instanceof HTMLInputElement && element.checked);
                break;
            case "toggle-spatial-conform":
                void this.commitParameter("video.spatialConform", element instanceof HTMLInputElement && element.checked ? "fit" : "none", "Toggle Spatial Conform");
                break;
            case "toggle-retime-section":
                void this.toggleRetimeSection(element instanceof HTMLInputElement && element.checked);
                break;
            case "reorder-effect":
                void this.reorderEffect(element.getAttribute("data-effect-id"), Number(element.getAttribute("data-dir")));
                break;
            case "remove-effect":
                void this.removeEffect(element.getAttribute("data-effect-id"));
                break;
            case "wrap-effect-mask":
                void this.wrapEffectInMask(element.getAttribute("data-effect-id"), element.getAttribute("data-mask-kind"));
                break;
            case "add-mask-source":
                void this.addMaskSource(element.getAttribute("data-group-id"), element.getAttribute("data-mask-kind"));
                break;
            case "toggle-mask-group":
                void this.patchMaskedGroup(element.getAttribute("data-group-id"), "Toggle mask group", (group) => ({ ...group, enabled: !group.enabled }));
                break;
            case "toggle-mask-invert":
                void this.patchMaskedGroup(element.getAttribute("data-group-id"), "Invert mask group", (group) => ({ ...group, inverted: !group.inverted }));
                break;
            case "remove-mask-group":
                void this.removeMaskedGroup(element.getAttribute("data-group-id"));
                break;
            case "reorder-mask-group":
                void this.reorderMaskedGroup(element.getAttribute("data-group-id"), Number(element.getAttribute("data-dir")));
                break;
            case "toggle-mask-source":
                void this.patchMaskSource(element, "Toggle mask", (mask) => ({ ...mask, enabled: !mask.enabled }));
                break;
            case "remove-mask-source":
                void this.removeMaskSource(element);
                break;
            case "reorder-mask-source":
                void this.reorderMaskSource(element, Number(element.getAttribute("data-dir")));
                break;
            case "swap-mask-filters":
                void this.patchMaskedGroup(element.getAttribute("data-group-id"), "Swap mask filters", (group) => ({ ...group, filters: [...group.filters].reverse() }));
                break;
            case "remove-mask-outside-effect":
                void this.patchMaskedGroup(element.getAttribute("data-group-id"), "Remove outside effect", (group) => ({ ...group, filters: group.filters.slice(0, 1) }));
                break;
            case "toggle-mask-keyframe":
                void this.toggleMaskKeyframe(element);
                break;
            case "previous-mask-keyframe":
                this.seekMaskKeyframe(element, -1);
                break;
            case "next-mask-keyframe":
                this.seekMaskKeyframe(element, 1);
                break;
            case "clear-mask-keyframes":
                void this.clearMaskKeyframes(element);
                break;
            case "add-mask-point":
                void this.adjustDrawMaskPoint(element, true);
                break;
            case "remove-mask-point":
                void this.adjustDrawMaskPoint(element, false);
                break;
            case "remove-transition":
                void this.removeTransition(element.getAttribute("data-transition-id"));
                break;
            case "set-retime-rate":
                void this.setRetimeRate(Number(element.getAttribute("data-rate")), element.getAttribute("data-reverse") === "true");
                break;
            case "split-retime":
                void this.splitRetimeAtPlayhead();
                break;
            case "add-retime-hold":
                void this.addRetimeHoldAtPlayhead(Number(element.closest(".retime-actions")?.querySelector("[data-retime-hold-duration]")?.value ?? 2));
                break;
            case "freeze-retime":
                void this.freezeRetimeAtPlayhead();
                break;
            case "insert-basic-title":
                void this.insertConstructedClip("title");
                break;
            case "insert-caption":
                void this.insertConstructedClip("caption");
                break;
            case "insert-custom-solid":
                void this.insertConstructedClip("generator");
                break;
            case "swap-ken-burns":
                void this.swapKenBurns();
                break;
            case "edit-marker":
                this.openMarkerEditor(element.getAttribute("data-marker-id"));
                break;
            case "marker-type":
                void this.updateMarkerFromElement(element, {
                    type: element.getAttribute("data-type"),
                });
                break;
            case "marker-toggle-done":
                this.toggleMarkerDone(element);
                break;
            case "marker-delete":
                void this.deleteMarkerFromElement(element);
                break;
            case "marker-close":
                this.state.markerEditorId = null;
                this.renderTimeline();
                break;
            default:
                break;
        }
    }
    handleDoubleClick(event) {
        const target = event.target;
        if (!(target instanceof Element)) {
            return;
        }
        const sectionTitle = target.closest(".section-title");
        if (sectionTitle) {
            this.toggleInspectorSection(sectionTitle.closest(".inspector-section"));
            event.preventDefault();
            return;
        }
        const timelineClip = target.closest(".timeline-clip");
        if (timelineClip instanceof HTMLElement && timelineClip.dataset.itemId) {
            const item = this.itemById(timelineClip.dataset.itemId);
            if (item?.container)
                this.enterSelectedContainer(item.id);
            return;
        }
        const card = target.closest(".asset-card");
        if (!(card instanceof HTMLElement)) {
            return;
        }
        const asset = this.assetById(card.dataset.assetId ?? null);
        if (!asset) {
            return;
        }
        if (this.state.connectionMode !== "localhost") {
            this.showMockNotice("fixture-projects");
        }
        if (asset.kind === "video" || asset.kind === "image") {
            void this.commitEdit("Append media", insertOperationAtTime(this.state.project, asset, projectDuration(this.state.project)));
            return;
        }
        const operation = connectOperationAtTime(this.state.project, asset, this.state.currentTime);
        if (operation) {
            void this.commitEdit("Connect asset", operation);
        }
    }
    /**
     * Collapse or reveal one Inspector section and keep its disclosure control
     * truthful for keyboard and screen-reader users.
     *
     * Main callers: the disclosure-button click and the Final Cut-style
     * double-click on a section title.
     */
    toggleInspectorSection(section) {
        if (!section) {
            return;
        }
        const collapsed = section.classList.toggle("collapsed");
        section.querySelector(".section-disclosure")?.setAttribute("aria-expanded", String(!collapsed));
    }
    handleInput(event) {
        const input = event.target;
        if (!(input instanceof HTMLInputElement || input instanceof HTMLSelectElement)) {
            return;
        }
        if (input.id === "asset-search") {
            this.state.mediaQuery = input.value;
            this.refreshBrowserResults();
            return;
        }
        if (input.id === "timeline-index-search") {
            this.state.timelineIndexQuery = input.value;
            this.renderTimeline();
            const search = document.getElementById("timeline-index-search");
            if (search instanceof HTMLInputElement) {
                search.focus();
                search.setSelectionRange(this.state.timelineIndexQuery.length, this.state.timelineIndexQuery.length);
            }
            return;
        }
        const action = input.getAttribute("data-action");
        if (action === "browser-search") {
            // Live-filter the catalog while typing; re-render only the browser panel
            // and restore the caret so focus is not lost mid-search.
            this.state.browserQuery = input.value;
            this.el("catalog-browser").innerHTML = catalogBrowserTemplate({
                ...this.state,
                capabilities: this.requireCapabilities(),
            });
            const search = this.el("catalog-browser").querySelector('[data-action="browser-search"]');
            if (search) {
                search.focus();
                search.setSelectionRange(input.value.length, input.value.length);
            }
            return;
        }
        if (action === "timeline-zoom") {
            this.state.pixelsPerSecond = Number(input.value);
            return;
        }
        if (action === "clip-height") {
            this.state.clipAppearance = {
                ...this.state.clipAppearance,
                height: Number(input.value),
            };
            return;
        }
        if (action === "show-clip-names" && input instanceof HTMLInputElement) {
            this.state.clipAppearance = {
                ...this.state.clipAppearance,
                showNames: input.checked,
            };
            this.renderTimeline();
            return;
        }
        if (action === "show-clip-roles" && input instanceof HTMLInputElement) {
            this.state.clipAppearance = {
                ...this.state.clipAppearance,
                showRoles: input.checked,
            };
            this.renderTimeline();
            return;
        }
        const parameterPath = input.getAttribute("data-parameter-path");
        if (parameterPath) {
            const value = input instanceof HTMLInputElement && input.type === "checkbox"
                ? input.checked
                : input instanceof HTMLInputElement && ["range", "number"].includes(input.type)
                    ? Number(input.value)
                    : input.value;
            this.previewParameter(parameterPath, value, input);
            return;
        }
        const numberPath = input.getAttribute("data-number-path");
        if (numberPath) {
            const multiplier = Number(input.getAttribute("data-number-multiplier") ?? 1);
            this.previewParameter(numberPath, Number(input.value) / multiplier, input);
        }
    }
    /** Commit an Inspector value immediately when the user presses Enter. */
    handleControlKeyDown(event) {
        if (!(event.target instanceof HTMLInputElement)) {
            return;
        }
        if (event.target.type === "range") {
            const input = event.target;
            const minimum = Number(input.min || 0);
            const maximum = Number(input.max || 100);
            const step = Number(input.step || 1);
            const current = Number(input.value);
            const next = event.key === "Home" ? minimum
                : event.key === "End" ? maximum
                    : event.key === "ArrowLeft" || event.key === "ArrowDown" ? current - step
                        : event.key === "ArrowRight" || event.key === "ArrowUp" ? current + step
                            : null;
            if (next !== null) {
                event.preventDefault();
                input.value = String(clamp(next, minimum, maximum));
                input.dispatchEvent(new Event("input", { bubbles: true }));
                input.dispatchEvent(new Event("change", { bubbles: true }));
            }
            return;
        }
        if (event.key !== "Enter")
            return;
        const action = event.target.getAttribute("data-action");
        const commitsText = event.target.type === "text" && [
            "set-title-text",
            "set-text-style",
            "set-caption-field",
            "set-sync-source-role",
            "marker-rename",
        ].includes(action ?? "");
        if (event.target.type !== "number" && !commitsText)
            return;
        event.preventDefault();
        event.target.dispatchEvent(new Event("change", { bubbles: true }));
    }
    handleChange(event) {
        const input = event.target;
        if (!(input instanceof HTMLInputElement || input instanceof HTMLSelectElement)) {
            return;
        }
        const changeAction = input.getAttribute("data-action");
        if (changeAction === "timeline-zoom" || changeAction === "clip-height") {
            // The input handler already adopted the value. Rendering here, after the
            // native drag ends, avoids replacing the active range input mid-gesture.
            this.renderTimeline();
            return;
        }
        if (changeAction === "viewer-background") {
            this.setViewerOption("background", input.value);
            return;
        }
        if (changeAction === "set-capability-parameter") {
            void this.setCapabilityParameter(input);
            return;
        }
        if (changeAction === "set-mask-parameter") {
            void this.setMaskParameter(input);
            return;
        }
        if (["set-multicam-video-angle", "set-multicam-audio-angle", "set-audition-choice", "set-sync-source-role", "set-sync-source-enabled", "set-sync-source-active"].includes(changeAction ?? "")) {
            void this.setContainerControl(input, changeAction);
            return;
        }
        if (["set-clip-role-name", "set-clip-audio-role", "set-clip-audio-start", "set-clip-audio-duration"].includes(changeAction ?? "")) {
            void this.setClipAudioMetadata(input, changeAction);
            return;
        }
        if (changeAction === "set-project-audio-layout") {
            void this.setProjectAudioLayout(input.value);
            return;
        }
        if (changeAction === "export-resolution") {
            const resolution = Number(input.value);
            if (!this.requireCapabilities().export.supportedResolutions.includes(resolution)) {
                this.showToast(`Export resolution ${resolution}p is not supported.`, "error");
                return;
            }
            this.exportResolution = resolution;
            this.renderTopbar();
            return;
        }
        if (changeAction === "export-format") {
            const profile = input.value;
            if (profile !== "delivery" && profile !== "delivery_alpha") {
                this.showToast(`Export format ${input.value} is not supported.`, "error");
                return;
            }
            this.exportProfile = profile;
            return;
        }
        if (changeAction === "set-mask-blend") {
            const allowed = this.requireCapabilities().mechanics
                .find((mechanic) => mechanic.id === "masks")?.blendModes ?? [];
            if (!allowed.includes(input.value)) {
                this.showToast(`Mask blend mode ${input.value} is not advertised by Bladeworks.`, "error");
                return;
            }
            void this.patchMaskSource(input, "Set mask blend", (mask) => ({
                ...mask,
                blendMode: input.value,
            }));
            return;
        }
        if (changeAction === "set-mask-outside-effect") {
            void this.setMaskOutsideEffect(input);
            return;
        }
        if (changeAction === "set-transition-duration") {
            void this.setTransitionDuration(input.getAttribute("data-transition-id"), Number(input.value));
            return;
        }
        if (changeAction === "toggle-retime-pitch") {
            void this.toggleRetimePitch(input instanceof HTMLInputElement && input.checked);
            return;
        }
        if (changeAction === "set-retime-custom") {
            const percent = Number(input.value);
            if (!Number.isFinite(percent) || percent === 0) {
                this.showToast("Speed cannot be 0%.", "error");
                this.renderInspector();
                return;
            }
            void this.setRetimeRate(Math.abs(percent) / 100, percent < 0);
            return;
        }
        if (changeAction === "set-retime-segment-rate") {
            void this.setRetimeSegmentRate(Number(input.getAttribute("data-segment-index")), Number(input.value) / 100);
            return;
        }
        if (changeAction === "set-retime-segment-duration") {
            void this.setRetimeSegmentDuration(Number(input.getAttribute("data-segment-index")), Number(input.value));
            return;
        }
        if (changeAction === "set-title-text") {
            void this.setTitleText(input.value);
            return;
        }
        if (changeAction === "set-text-style") {
            void this.setTextStyle(input);
            return;
        }
        if (changeAction === "set-caption-field") {
            void this.setCaptionField(input);
            return;
        }
        if (changeAction === "set-generator-color") {
            void this.setGeneratorColor(input.value);
            return;
        }
        if (changeAction === "marker-rename") {
            void this.updateMarkerFromElement(input, { name: input.value });
            return;
        }
        const path = input.getAttribute("data-parameter-path") ?? input.getAttribute("data-number-path");
        if (!path || !this.parameterBase) {
            return;
        }
        const mockId = mockCapabilityForParameter(path, this.state.connectionMode);
        if (mockId) {
            this.showMockNotice(mockId);
        }
        const item = this.selectedItem();
        if (!item) {
            return;
        }
        const multiplier = Number(input.getAttribute("data-number-multiplier") ?? 1);
        const value = input instanceof HTMLInputElement && input.type === "checkbox"
            ? input.checked
            : input instanceof HTMLInputElement && ["range", "number"].includes(input.type)
                ? Number(input.value) / (input.hasAttribute("data-number-multiplier") ? multiplier : 1)
                : input.hasAttribute("data-number-path")
                    ? Number(input.value) / multiplier
                    : input.value;
        // Commit the same updateClipPath against the gesture base, not a snapshot restore.
        this.transactionBase = this.parameterBase;
        void this.commitEdit(`Adjust ${path}`, {
            type: "updateClipPath",
            clipId: item.id,
            path,
            value,
        });
    }
    previewParameter(path, value, input) {
        const mockId = mockCapabilityForParameter(path, this.state.connectionMode);
        if (this.state.connectionMode === "localhost" && mockId && LOCALHOST_BLOCKED_MOCKS.has(mockId)) {
            this.showMockNotice(mockId);
            return;
        }
        const item = this.selectedItem();
        if (!item) {
            return;
        }
        if (!this.parameterBase) {
            this.parameterBase = this.currentSnapshot();
        }
        this.state.project = updateItemPath(this.parameterBase, item.id, path, value);
        this.syncLinkedParameterControls(path, value, input);
        this.updateProgramFrame();
        this.updateCanvasControls();
        this.renderTimeline();
        this.previewRealtime({
            type: "set-item-parameter",
            itemId: item.id,
            path,
            value,
        });
    }
    syncLinkedParameterControls(path, value, source) {
        for (const element of this.root.querySelectorAll(`[data-parameter-path="${CSS.escape(path)}"], [data-number-path="${CSS.escape(path)}"]`)) {
            if (element === source ||
                (!(element instanceof HTMLInputElement) && !(element instanceof HTMLSelectElement))) {
                continue;
            }
            if (element instanceof HTMLInputElement && element.type === "checkbox") {
                element.checked = Boolean(value);
            }
            else if (element.hasAttribute("data-number-multiplier")) {
                element.value = String(Number(value) * Number(element.getAttribute("data-number-multiplier") ?? 1));
            }
            else {
                element.value = String(value);
            }
        }
    }
    handleDragStart(event) {
        const target = event.target;
        if (!(target instanceof Element) || !event.dataTransfer) {
            return;
        }
        const assetCard = target.closest(".asset-card");
        if (assetCard instanceof HTMLElement) {
            const assetId = assetCard.getAttribute("data-asset-id") ?? "";
            const rect = assetCard.getBoundingClientRect();
            event.dataTransfer.effectAllowed = "copy";
            event.dataTransfer.setData("application/x-bladeworks-asset", assetId);
            setBlankDragImage(event.dataTransfer);
            assetCard.classList.add("dragging");
            this.beginTimelineDrag({
                kind: "insert",
                assetId,
                clipId: null,
                connectedClipId: null,
                previewClipId: randomId("clip"),
                grabX: event.clientX - rect.left,
                grabY: event.clientY - rect.top,
            });
            this.attachTimelineDragGhost(assetCard, event.clientX, event.clientY);
            return;
        }
        const clip = target.closest(".storyline-clip");
        if (clip instanceof HTMLElement && ["select", "position"].includes(this.state.tool)) {
            const clipId = clip.getAttribute("data-item-id") ?? "";
            const rect = clip.getBoundingClientRect();
            event.dataTransfer.effectAllowed = "move";
            event.dataTransfer.setData("application/x-bladeworks-storyline", clipId);
            setBlankDragImage(event.dataTransfer);
            clip.classList.add("dragging");
            this.beginTimelineDrag({
                kind: "reorder",
                assetId: null,
                clipId,
                connectedClipId: null,
                previewClipId: clipId,
                grabX: event.clientX - rect.left,
                grabY: event.clientY - rect.top,
            });
            this.attachTimelineDragGhost(clip, event.clientX, event.clientY);
            const canvas = clip.closest(".timeline-canvas");
            if (canvas instanceof HTMLElement) {
                this.previewTimelineDrag(event.clientX, event.clientY, canvas);
            }
            return;
        }
        const connected = target.closest(".connected-clip, .audio-clip");
        if (connected instanceof HTMLElement && this.state.tool === "select") {
            const clipId = connected.getAttribute("data-item-id") ?? "";
            const rect = connected.getBoundingClientRect();
            event.dataTransfer.effectAllowed = "move";
            event.dataTransfer.setData("application/x-bladeworks-connected", clipId);
            setBlankDragImage(event.dataTransfer);
            connected.classList.add("dragging");
            this.beginTimelineDrag({
                kind: "connect",
                assetId: null,
                clipId: null,
                connectedClipId: clipId,
                previewClipId: clipId,
                grabX: event.clientX - rect.left,
                grabY: event.clientY - rect.top,
            });
            this.attachTimelineDragGhost(connected, event.clientX, event.clientY);
            const canvas = connected.closest(".timeline-canvas");
            if (canvas instanceof HTMLElement) {
                this.previewTimelineDrag(event.clientX, event.clientY, canvas);
            }
        }
    }
    /**
     * Freeze the storyline at gesture start. Chrome does not expose custom
     * drag payload during dragover, so the ids live on `timelineDrag`.
     */
    beginTimelineDrag(partial) {
        this.transactionBase = clone(this.state.project);
        this.timelineDrag = {
            ...partial,
            signature: "",
            operation: null,
            label: "",
            committed: false,
            connectWarning: null,
        };
        this.state.dropPreviewClipId = partial.previewClipId;
        window.addEventListener("dragover", this.onWindowTimelineDragOver);
    }
    onWindowTimelineDragOver = (event) => {
        this.positionTimelineDragGhost(event.clientX, event.clientY);
    };
    /**
     * Clone the actual dragged clip (filmstrip, waveform, name) onto the page
     * so it follows the pointer. Native HTML5 drag images are a 1x1 blank for
     * storyline clips, because that ghost cannot move vertically or keep live
     * thumbnails.
     *
     * Main callers: handleDragStart.
     */
    attachTimelineDragGhost(source, clientX, clientY) {
        this.removeTimelineDragGhost();
        const ghost = source.cloneNode(true);
        ghost.removeAttribute("draggable");
        ghost.removeAttribute("data-item-id");
        ghost.classList.add("timeline-drag-ghost");
        ghost.classList.remove("dragging", "drop-preview", "selected", "primary-selected");
        ghost.querySelectorAll("[data-trim]").forEach((node) => node.remove());
        const rect = source.getBoundingClientRect();
        const canvas = source.closest(".timeline-canvas");
        if (canvas instanceof HTMLElement) {
            ghost.style.setProperty("--clip-height", getComputedStyle(canvas).getPropertyValue("--clip-height"));
        }
        ghost.style.position = "fixed";
        ghost.style.left = `${rect.left}px`;
        ghost.style.top = `${rect.top}px`;
        ghost.style.width = `${rect.width}px`;
        ghost.style.height = `${rect.height}px`;
        ghost.style.margin = "0";
        ghost.style.zIndex = "10050";
        ghost.style.pointerEvents = "none";
        document.body.appendChild(ghost);
        this.timelineDragGhost = ghost;
        this.positionTimelineDragGhost(clientX, clientY);
    }
    positionTimelineDragGhost(clientX, clientY) {
        const ghost = this.timelineDragGhost;
        const drag = this.timelineDrag;
        if (!ghost || !drag) {
            return;
        }
        ghost.style.left = `${clientX - drag.grabX}px`;
        ghost.style.top = `${clientY - drag.grabY}px`;
    }
    removeTimelineDragGhost() {
        this.timelineDragGhost?.remove();
        this.timelineDragGhost = null;
    }
    handleDragOver(event) {
        const target = event.target;
        if (!(target instanceof Element) || !this.timelineDrag) {
            return;
        }
        this.positionTimelineDragGhost(event.clientX, event.clientY);
        const canvas = target.closest(".timeline-canvas");
        if (!(canvas instanceof HTMLElement)) {
            return;
        }
        event.preventDefault();
        if (this.dragLeaveTimer !== null) {
            window.clearTimeout(this.dragLeaveTimer);
            this.dragLeaveTimer = null;
        }
        if (event.dataTransfer) {
            event.dataTransfer.dropEffect = this.timelineDrag.clipId || this.timelineDrag.connectedClipId
                ? "move"
                : "copy";
        }
        this.previewTimelineDrag(event.clientX, event.clientY, canvas);
    }
    handleDragLeave(event) {
        const canvas = this.root.querySelector(".timeline-canvas");
        if (!(canvas instanceof HTMLElement) || !this.timelineDrag) {
            return;
        }
        const related = event.relatedTarget;
        if (related instanceof Node && canvas.contains(related)) {
            return;
        }
        if (this.dragLeaveTimer !== null) {
            window.clearTimeout(this.dragLeaveTimer);
        }
        this.dragLeaveTimer = window.setTimeout(() => {
            this.dragLeaveTimer = null;
            this.restoreTimelineDragPreview();
        }, 40);
    }
    handleDrop(event) {
        const target = event.target;
        if (!(target instanceof Element) || !event.dataTransfer) {
            return;
        }
        const canvas = target.closest(".timeline-canvas");
        if (!(canvas instanceof HTMLElement)) {
            return;
        }
        event.preventDefault();
        if (this.dragLeaveTimer !== null) {
            window.clearTimeout(this.dragLeaveTimer);
            this.dragLeaveTimer = null;
        }
        this.previewTimelineDrag(event.clientX, event.clientY, canvas);
        const drag = this.timelineDrag;
        if (!drag?.operation) {
            return;
        }
        drag.committed = true;
        const label = drag.label;
        const operation = drag.operation;
        this.removeTimelineDragGhost();
        this.timelineDrag = null;
        this.state.dropPreviewClipId = null;
        void this.commitEdit(label, operation);
    }
    handleDragEnd() {
        removeBlankDragImage();
        this.root.querySelectorAll(".dragging").forEach((node) => node.classList.remove("dragging"));
        if (!this.timelineDrag) {
            this.removeTimelineDragGhost();
            window.removeEventListener("dragover", this.onWindowTimelineDragOver);
            return;
        }
        if (!this.timelineDrag.committed) {
            this.restoreTimelineDragPreview();
        }
        this.finishTimelineDrag();
    }
    /**
     * Live magnetic preview: snap the pointer to nearby edit points, then
     * insert/reorder/connect against the frozen storyline. Neighbor clips
     * slide in place when the snapped slot changes. The dragged clip itself
     * follows the pointer as `timelineDragGhost`; the in-timeline copy is only
     * a faint landing slot.
     *
     * Main callers: handleDragOver, handleDrop, handleDragStart.
     */
    previewTimelineDrag(clientX, clientY, canvas) {
        const drag = this.timelineDrag;
        const base = this.transactionBase;
        if (!drag || !base) {
            return;
        }
        this.positionTimelineDragGhost(clientX, clientY);
        const dropRole = dropRoleFromCanvas(canvas, clientY);
        const pointerTime = this.timeFromCanvasPointer(clientX, canvas, base);
        const startTime = this.timeFromCanvasPointer(clientX - drag.grabX, canvas, base);
        const raw = dropRole === "storyline" ? pointerTime : startTime;
        const snapSource = (drag.clipId || drag.connectedClipId) && dropRole !== "storyline"
            ? { ...base, spine: base.spine.filter((clip) => clip.id !== drag.clipId) }
            : base;
        const time = this.snapTime(raw, snapSource);
        this.updateDragSkimmer(time, raw);
        canvas.classList.toggle("connecting-video", dropRole === "connected-video");
        canvas.classList.toggle("connecting-audio", dropRole === "connected-audio");
        canvas.classList.toggle("connecting-storyline", dropRole === "storyline" && Boolean(drag.connectedClipId));
        const next = this.timelineDragOperation(drag, base, time, dropRole);
        if (!next) {
            return;
        }
        if (next.signature === drag.signature) {
            return;
        }
        try {
            const drafted = applyEdit(base, next.operation);
            const kindChanged = drag.signature !== "" && !drag.signature.startsWith(`${next.kind}:`);
            drag.kind = next.kind;
            drag.signature = next.signature;
            drag.operation = next.operation;
            drag.label = next.label;
            this.state.dropPreviewClipId = drag.previewClipId;
            this.state.project = drafted;
            if (kindChanged) {
                this.renderTimeline();
                const nextCanvas = this.root.querySelector(".timeline-canvas");
                if (nextCanvas instanceof HTMLElement) {
                    nextCanvas.classList.toggle("connecting-video", dropRole === "connected-video");
                    nextCanvas.classList.toggle("connecting-audio", dropRole === "connected-audio");
                    nextCanvas.classList.toggle("connecting-storyline", dropRole === "storyline" && Boolean(drag.connectedClipId));
                }
                return;
            }
            this.paintTimelineDraft(drafted, drag.previewClipId);
        }
        catch {
            return;
        }
    }
    timelineDragOperation(drag, base, time, dropRole) {
        if (drag.connectedClipId) {
            return this.connectedClipDragOperation(drag, base, time, dropRole);
        }
        if (drag.clipId) {
            return this.spineClipDragOperation(drag, base, time, dropRole);
        }
        const asset = this.assetById(drag.assetId);
        if (!asset) {
            return null;
        }
        const shouldConnect = dropRole !== "storyline" || asset.kind === "audio" || asset.kind === "title";
        if (shouldConnect) {
            const connected = connectOperationAtTime(base, asset, time);
            if (!connected || connected.type !== "connect") {
                return null;
            }
            const clip = { ...connected.clip, id: drag.previewClipId };
            const frame = 1 / Math.max(1, base.fps);
            const quantized = Math.round(time / frame) * frame;
            return {
                kind: "connect",
                signature: `connect:${clip.anchorId}:${quantized.toFixed(4)}`,
                operation: { type: "connect", clip },
                label: "Connect asset",
            };
        }
        const index = insertionIndexAtTime(base, time);
        const clip = { ...timelineClipFromAsset(drag.previewClipId, asset), timelineStart: time };
        return {
            kind: "insert",
            signature: `insert:${index}`,
            operation: { type: "insert", clip, index },
            label: "Insert media",
        };
    }
    /**
     * Spine-clip drag: stay on the storyline shelf to reorder, or leave it
     * vertically to convert the clip into a connected clip (FCP Select-tool
     * ripple). Blocked if this clip still has dependents, or if it is the last
     * remaining primary-storyline clip.
     */
    spineClipDragOperation(drag, base, time, dropRole) {
        const clipId = drag.clipId;
        if (!clipId) {
            return null;
        }
        const moving = base.spine.find((clip) => clip.id === clipId);
        if (!moving) {
            return null;
        }
        const remaining = { ...base, spine: base.spine.filter((clip) => clip.id !== clipId) };
        if (dropRole !== "storyline" && dropRole === naturalConnectRole(moving)) {
            const blocked = this.spineConnectBlockReason(base, clipId);
            if (blocked) {
                this.warnTimelineDrag(drag, blocked);
            }
            else {
                const frame = 1 / Math.max(1, base.fps);
                const quantized = Math.round(time / frame) * frame;
                return {
                    kind: "connect",
                    signature: `connect:${clipId}:${quantized.toFixed(4)}:${dropRole}`,
                    operation: {
                        type: "spineToConnected",
                        clipId,
                        timelineStart: Math.max(0, quantized),
                        lane: connectedLaneForClip(moving),
                    },
                    label: "Connect from storyline",
                };
            }
        }
        const toIndex = insertionIndexAtTime(remaining, time);
        return {
            kind: "reorder",
            signature: `reorder:${toIndex}`,
            operation: { type: "reorder", clipId, toIndex },
            label: "Reorder storyline",
        };
    }
    /**
     * Connected-clip drag: stay in a connected lane to slide along time, or drop
     * onto the storyline shelf to insert (FCP Select-tool reverse of lift).
     */
    connectedClipDragOperation(drag, base, time, dropRole) {
        const clipId = drag.connectedClipId;
        if (!clipId) {
            return null;
        }
        if (!base.connected.some((clip) => clip.id === clipId)) {
            return null;
        }
        if (dropRole === "storyline") {
            const toIndex = insertionIndexAtTime(base, time);
            return {
                kind: "insert",
                signature: `insert:connected:${toIndex}`,
                operation: { type: "connectedToSpine", clipId, toIndex },
                label: "Insert from connected",
            };
        }
        const frame = 1 / Math.max(1, base.fps);
        const quantized = Math.round(Math.max(0, time) / frame) * frame;
        return {
            kind: "connect",
            signature: `connect:move:${clipId}:${quantized.toFixed(4)}`,
            operation: { type: "moveConnected", clipId, timelineStart: quantized },
            label: "Move connected clip",
        };
    }
    spineConnectBlockReason(base, clipId) {
        if (base.connected.some((clip) => clip.anchorId === clipId)) {
            return "Move or remove clips connected to this storyline clip first.";
        }
        if (base.spine.length <= 1) {
            return "Connect needs another clip on the primary storyline.";
        }
        return null;
    }
    warnTimelineDrag(drag, message) {
        if (drag.connectWarning === message) {
            return;
        }
        drag.connectWarning = message;
        this.showToast(message, "error");
    }
    /**
     * Slide existing clip nodes to the drafted times. Full render only when the
     * landing-slot clip is not in the DOM yet, so neighbors can CSS-transition.
     */
    paintTimelineDraft(project, previewClipId) {
        const canvas = this.root.querySelector(".timeline-canvas");
        const ghost = canvas instanceof HTMLElement
            ? canvas.querySelector(`[data-item-id="${previewClipId}"]`)
            : null;
        if (!(canvas instanceof HTMLElement) || !ghost) {
            this.renderTimeline();
            return;
        }
        canvas.classList.add("drag-previewing");
        const pps = this.state.pixelsPerSecond;
        for (const clip of [...project.spine, ...project.connected]) {
            const node = canvas.querySelector(`[data-item-id="${clip.id}"]`);
            if (!(node instanceof HTMLElement)) {
                this.renderTimeline();
                return;
            }
            node.classList.toggle("drop-preview", clip.id === previewClipId);
            node.style.left = `${clip.timelineStart * pps}px`;
            node.style.width = `${Math.max(18, clip.duration * pps)}px`;
            const shelf = canvas.querySelector(".storyline-shelf");
            if (shelf instanceof HTMLElement) {
                const connected = "lane" in clip ? clip : null;
                const clipHeight = Number.parseFloat(getComputedStyle(canvas).getPropertyValue("--clip-height")) || 48;
                if (connected && connected.lane > 0) {
                    node.style.top = `${shelf.offsetTop - connected.lane * 27}px`;
                }
                else if (connected && connected.lane < 0) {
                    node.style.top = `${shelf.offsetTop + clipHeight + 14 + (Math.abs(connected.lane) - 1) * 44}px`;
                }
                else {
                    node.style.top = `${shelf.offsetTop}px`;
                }
            }
        }
        const duration = projectDuration(project);
        canvas.style.width = `${Math.max(1000, duration * pps + 220)}px`;
        const end = canvas.querySelector(".timeline-end");
        if (end instanceof HTMLElement) {
            end.style.left = `${duration * pps}px`;
        }
    }
    updateDragSkimmer(time, raw) {
        const skimmer = document.getElementById("skimmer");
        if (!skimmer) {
            return;
        }
        skimmer.classList.remove("hidden", "out");
        skimmer.style.left = `${time * this.state.pixelsPerSecond}px`;
        skimmer.classList.toggle("snapped", Math.abs(time - raw) > 1e-6);
    }
    restoreTimelineDragPreview() {
        if (!this.timelineDrag || this.timelineDrag.committed) {
            return;
        }
        if (this.transactionBase) {
            this.state.project = clone(this.transactionBase);
        }
        this.timelineDrag.signature = "";
        this.timelineDrag.operation = null;
        this.state.dropPreviewClipId = null;
        const canvas = this.root.querySelector(".timeline-canvas");
        if (canvas instanceof HTMLElement) {
            canvas.classList.remove("connecting-video", "connecting-audio", "connecting-storyline");
        }
        this.renderTimeline();
    }
    finishTimelineDrag() {
        if (this.dragLeaveTimer !== null) {
            window.clearTimeout(this.dragLeaveTimer);
            this.dragLeaveTimer = null;
        }
        window.removeEventListener("dragover", this.onWindowTimelineDragOver);
        removeBlankDragImage();
        this.removeTimelineDragGhost();
        const canvas = this.root.querySelector(".timeline-canvas");
        if (canvas instanceof HTMLElement) {
            canvas.classList.remove("connecting-video", "connecting-audio", "connecting-storyline");
        }
        this.timelineDrag = null;
        this.state.dropPreviewClipId = null;
        this.transactionBase = null;
    }
    handlePointerDown(event) {
        const target = event.target;
        if (!(target instanceof Element)) {
            return;
        }
        if (this.contextMenu && !this.contextMenu.contains(target)) {
            this.closeStudioMenu();
            this.suppressNextClick = true;
            return;
        }
        if (event.button !== 0) {
            return;
        }
        if (target instanceof HTMLInputElement && target.type === "range") {
            this.startRangeInputDrag(event, target);
            return;
        }
        const markerEditor = target.closest(".marker-editor");
        if (markerEditor) {
            // Marker controls live over the timeline and the operation may rerender
            // this entire popover. Execute buttons on pointer-down while the original
            // element still exists, then swallow the detached follow-up click.
            const markerAction = target.closest("[data-action]");
            if (markerAction) {
                event.preventDefault();
                event.stopPropagation();
                this.runAction(markerAction.getAttribute("data-action"), markerAction, event);
                this.suppressGeneratedClick();
            }
            else {
                this.suppressNextClick = false;
            }
            return;
        }
        const markerButton = target.closest(".clip-marker");
        if (markerButton) {
            event.preventDefault();
            event.stopPropagation();
            this.suppressGeneratedClick();
            this.openMarkerEditor(markerButton.getAttribute("data-marker-id"));
            return;
        }
        const resize = target.closest("[data-resize]");
        if (resize) {
            this.startPanelResize(event, resize.getAttribute("data-resize"));
            return;
        }
        const knob = target.closest("[data-knob-path]");
        if (knob instanceof HTMLElement) {
            this.startKnobDrag(event, knob);
            return;
        }
        const colorWheel = target.closest("[data-color-wheel]");
        if (colorWheel) {
            this.startColorWheelDrag(event, colorWheel.getAttribute("data-color-wheel"));
            return;
        }
        const canvasHandle = target.closest("[data-canvas-handle]");
        if (canvasHandle) {
            this.startTransformControl(event, canvasHandle.getAttribute("data-canvas-handle"));
            return;
        }
        const cropHandle = target.closest("[data-crop-handle]");
        if (cropHandle) {
            this.startCropControl(event, cropHandle.getAttribute("data-crop-handle"));
            return;
        }
        const kenHandle = target.closest("[data-ken-handle]");
        if (kenHandle) {
            this.startKenBurnsControl(event, kenHandle.getAttribute("data-ken-handle"));
            return;
        }
        const distortHandle = target.closest("[data-distort-handle]");
        if (distortHandle) {
            this.startDistortControl(event, distortHandle.getAttribute("data-distort-handle"));
            return;
        }
        const trimHandle = target.closest("[data-trim]");
        if (trimHandle && ["select", "trim"].includes(this.state.tool)) {
            event.preventDefault();
            event.stopPropagation();
            const clip = trimHandle.closest(".timeline-clip");
            const itemId = clip?.getAttribute("data-item-id");
            const edge = trimHandle.getAttribute("data-trim");
            if (itemId && (edge === "left" || edge === "right")) {
                // Trim tool edge-drag rolls the shared edit point; Select tool edge-drag
                // ripple-trims the single clip.
                if (this.state.tool === "trim") {
                    this.startRoll(event, itemId, edge);
                }
                else {
                    this.startTrim(event, itemId, edge);
                }
            }
            return;
        }
        // Trim tool body-drag on a storyline clip: Slip (change source in/out) by
        // default, Slide (shift the clip, absorbing into neighbors) with Option held.
        const trimBodyClip = target.closest(".storyline-clip");
        if (trimBodyClip &&
            this.state.tool === "trim" &&
            !target.closest("button, [data-trim], .clip-marker")) {
            const itemId = trimBodyClip.getAttribute("data-item-id");
            if (itemId) {
                event.preventDefault();
                event.stopPropagation();
                if (event.altKey) {
                    this.startSlide(event, itemId);
                }
                else {
                    this.startSlip(event, itemId);
                }
            }
            return;
        }
        const playhead = target.closest("#playhead");
        if (playhead) {
            this.startPlayheadDrag(event);
            return;
        }
        const canvas = target.closest(".timeline-canvas");
        if (canvas && this.state.tool === "hand") {
            this.startHandDrag(event);
            return;
        }
        // Range Selection intentionally wins over clip movement. In Final Cut the
        // Range tool can begin directly on a clip, including a connected clip.
        if (canvas && this.state.tool === "range") {
            this.startRangeSelection(event);
            return;
        }
        const connectedClip = target.closest(".connected-clip, .audio-clip, .title-clip");
        if (connectedClip && !target.closest("button")) {
            if (event.metaKey || event.ctrlKey || event.shiftKey) {
                return;
            }
            const id = connectedClip.getAttribute("data-item-id");
            if (!id) {
                return;
            }
            const selectedConnected = this.state.project.connected.filter((candidate) => this.state.selectedItemIds.includes(candidate.id));
            const multi = selectedConnected.length > 1 && selectedConnected.some((clip) => clip.id === id);
            if (this.state.tool !== "select" || multi) {
                this.startConnectedMove(event, id, connectedClip);
            }
            return;
        }
        if (canvas instanceof HTMLElement &&
            this.state.tool === "select" &&
            !target.closest(".timeline-clip, .transition-block, .timeline-ruler, #playhead, button, input, select, textarea, [contenteditable='true']")) {
            this.startMarqueeSelection(event, canvas);
        }
    }
    /**
     * Give every visible range control deterministic editor-owned mechanics.
     *
     * Chromium's native range default is unreliable inside the packaged app
     * surface, where pointer capture and keyboard locking coexist with delegated
     * editor events. Convert the pointer position to the declared min/max range,
     * emit ordinary input events during the gesture, and commit once on release.
     *
     * Main callers: handlePointerDown for Timeline, Inspector, and Color sliders.
     */
    startRangeInputDrag(event, input) {
        event.preventDefault();
        event.stopPropagation();
        const rectangle = input.getBoundingClientRect();
        const minimum = Number(input.min || 0);
        const maximum = Number(input.max || 100);
        const step = Number(input.step || 1);
        const quantize = (raw) => {
            const stepped = minimum + Math.round((raw - minimum) / step) * step;
            return clamp(stepped, minimum, maximum);
        };
        const applyPointer = (pointer) => {
            const ratio = clamp((pointer.clientX - rectangle.left) / Math.max(1, rectangle.width), 0, 1);
            input.value = String(quantize(minimum + ratio * (maximum - minimum)));
            input.dispatchEvent(new Event("input", { bubbles: true }));
        };
        const onUp = (pointer) => {
            applyPointer(pointer);
            window.removeEventListener("pointermove", applyPointer);
            window.removeEventListener("pointerup", onUp);
            input.dispatchEvent(new Event("change", { bubbles: true }));
        };
        applyPointer(event);
        input.focus();
        window.addEventListener("pointermove", applyPointer);
        window.addEventListener("pointerup", onUp, { once: true });
    }
    handlePointerMove(event) {
        if (!this.state.skimming || this.pointerSession) {
            return;
        }
        const target = event.target;
        if (!(target instanceof Element)) {
            return;
        }
        const canvas = target.closest(".timeline-canvas");
        const skimmer = document.getElementById("skimmer");
        if (!(canvas instanceof HTMLElement) || !skimmer) {
            return;
        }
        const raw = this.timeFromCanvasPointer(event.clientX, canvas);
        const time = this.snapTime(raw);
        skimmer.style.left = `${time * this.state.pixelsPerSecond}px`;
        skimmer.classList.toggle("snapped", Math.abs(time - raw) > 1e-6);
    }
    handlePointerLeave(event) {
        const target = event.target;
        if (target instanceof Element && target.classList.contains("timeline-scroller")) {
            document.getElementById("skimmer")?.classList.add("out");
        }
    }
    startPanelResize(event, side) {
        if (!side) {
            return;
        }
        event.preventDefault();
        const startX = event.clientX;
        const startY = event.clientY;
        const original = { ...this.state.layout };
        // The workspace grid has exactly one flexible column (the viewer). If the
        // fixed side panels are allowed to grow past `viewport - viewerFloor`, the
        // grid overflows and the rightmost column is pushed off-screen. So every
        // horizontal resize is capped by what the other visible panels already use,
        // keeping the viewer at or above its floor.
        const splitterAllowance = 16;
        const viewerFloor = 340;
        const widthOf = (panel) => this.state.panels[panel] ? original[`${panel}Width`] : 0;
        const onMove = (move) => {
            let next = { ...this.state.layout };
            if (side === "library") {
                const ceiling = Math.min(280, Math.max(110, window.innerWidth - widthOf("browser") - widthOf("inspector") - splitterAllowance - viewerFloor));
                next = {
                    ...next,
                    libraryWidth: clamp(original.libraryWidth + move.clientX - startX, 110, ceiling),
                };
            }
            if (side === "browser") {
                const ceiling = Math.min(680, Math.max(260, window.innerWidth - widthOf("library") - widthOf("inspector") - splitterAllowance - viewerFloor));
                next = {
                    ...next,
                    browserWidth: clamp(original.browserWidth + move.clientX - startX, 260, ceiling),
                };
            }
            if (side === "inspector") {
                const ceiling = Math.min(520, Math.max(250, window.innerWidth - widthOf("library") - widthOf("browser") - splitterAllowance - viewerFloor));
                next = {
                    ...next,
                    inspectorWidth: clamp(original.inspectorWidth - (move.clientX - startX), 250, ceiling),
                };
            }
            if (side === "timeline") {
                next = {
                    ...next,
                    timelineHeight: clamp(original.timelineHeight - (move.clientY - startY), 180, window.innerHeight - 250),
                };
            }
            this.state.layout = next;
            this.applyLayout();
        };
        const onUp = () => {
            window.removeEventListener("pointermove", onMove);
            window.removeEventListener("pointerup", onUp);
        };
        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", onUp, { once: true });
    }
    startKnobDrag(event, knob) {
        const path = knob.dataset.knobPath;
        const item = this.selectedItem();
        if (!path || !item) {
            return;
        }
        event.preventDefault();
        const base = this.currentSnapshot();
        const startValue = Number(getPath(item, path) ?? 0);
        const startX = event.clientX;
        const startY = event.clientY;
        const min = Number(knob.dataset.min ?? -180);
        const max = Number(knob.dataset.max ?? 180);
        const step = Number(knob.dataset.step ?? 0.1);
        let lastValue = startValue;
        const onMove = (move) => {
            const raw = startValue + (move.clientX - startX) - (move.clientY - startY);
            const value = clamp(Math.round(raw / step) * step, min, max);
            lastValue = value;
            this.state.project = updateItemPath(base, item.id, path, value);
            knob.style.setProperty("--knob-angle", `${value}deg`);
            this.syncLinkedParameterControls(path, value, knob);
            this.updateProgramFrame();
            this.updateCanvasControls();
            this.previewRealtime({
                type: "set-item-parameter",
                itemId: item.id,
                path,
                value,
            });
        };
        const onUp = () => {
            window.removeEventListener("pointermove", onMove);
            window.removeEventListener("pointerup", onUp);
            void this.commitEdit(`Adjust ${path}`, {
                type: "updateClipPath",
                clipId: item.id,
                path,
                value: lastValue,
            });
        };
        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", onUp, { once: true });
    }
    /**
     * Select or drag one tonal-range puck.
     *
     * The puck writes one coherent X/Y/level triple into a single updateClip
     * commit. This remains a mock capability until the localhost renderer defines
     * the exact FCPXML color-wheel mapping.
     *
     * Main callers: pointerdown on [data-color-wheel].
     */
    startColorWheelDrag(event, zone) {
        if (!zone || !["shadows", "midtones", "highlights"].includes(zone)) {
            return;
        }
        const selected = this.selectedItem();
        if (!selected) {
            return;
        }
        const typedZone = zone;
        event.preventDefault();
        event.stopPropagation();
        this.state.activeColorZone = typedZone;
        this.showMockNotice("color-wheels");
        const target = event.target;
        const button = target instanceof Element ? target.closest("[data-color-wheel]") : null;
        const buttonElement = button instanceof HTMLElement ? button : null;
        for (const puck of this.root.querySelectorAll(".color-puck")) {
            puck.classList.toggle("active", puck === buttonElement);
        }
        const base = this.currentSnapshot();
        const startPuck = { ...selected.video.color.pucks[typedZone] };
        const startX = event.clientX;
        const startY = event.clientY;
        let moved = false;
        let lastX = startPuck.x;
        let lastY = startPuck.y;
        this.pointerSession = { active: true };
        const onMove = (move) => {
            const dx = move.clientX - startX;
            const dy = move.clientY - startY;
            moved = moved || Math.abs(dx) > 2 || Math.abs(dy) > 2;
            const x = clamp(startPuck.x + dx * 3.2, -100, 100);
            const y = clamp(startPuck.y - dy * 3.2, -100, 100);
            lastX = x;
            lastY = y;
            const baseItem = findTimelineClip(base, selected.id);
            if (!baseItem) {
                return;
            }
            this.state.project = updateItem(base, selected.id, {
                video: {
                    ...baseItem.video,
                    color: {
                        ...baseItem.video.color,
                        [typedZone]: y,
                        pucks: {
                            ...baseItem.video.color.pucks,
                            [typedZone]: { x, y },
                        },
                    },
                },
            });
            if (buttonElement) {
                buttonElement.style.setProperty("--wheel-x", `${x * 0.14}px`);
                buttonElement.style.setProperty("--wheel-y", `${y * 0.14}px`);
                const coordinates = buttonElement.querySelector("small");
                if (coordinates) {
                    coordinates.textContent = `${Math.round(x)}, ${Math.round(y)}`;
                }
            }
            this.syncLinkedParameterControls(`video.color.${typedZone}`, y, buttonElement ?? this.root);
            this.updateProgramFrame();
        };
        const onUp = () => {
            window.removeEventListener("pointermove", onMove);
            window.removeEventListener("pointerup", onUp);
            this.pointerSession = null;
            this.suppressGeneratedClick();
            if (moved) {
                const baseItem = findTimelineClip(base, selected.id);
                if (!baseItem) {
                    this.transactionBase = null;
                    return;
                }
                void this.commitEdit(`Adjust ${typedZone} wheel`, {
                    type: "updateClip",
                    clipId: selected.id,
                    patch: {
                        video: {
                            ...baseItem.video,
                            color: {
                                ...baseItem.video.color,
                                [typedZone]: lastY,
                                pucks: {
                                    ...baseItem.video.color.pucks,
                                    [typedZone]: { x: lastX, y: lastY },
                                },
                            },
                        },
                    },
                });
            }
            else {
                this.transactionBase = null;
                this.renderInspector();
                this.updateProgramFrame();
            }
        };
        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", onUp, { once: true });
    }
    startTransformControl(event, handle) {
        const selected = this.selectedItem();
        if (!selected || selected.kind === "audio" || !handle) {
            return;
        }
        event.preventDefault();
        const base = this.currentSnapshot();
        const original = clone(selected.transform);
        let lastTransform = original;
        const frame = this.canvasControlFrame();
        const startX = event.clientX;
        const startY = event.clientY;
        const heightToWidth = this.state.project.height / this.state.project.width;
        const center = {
            x: frame.left + frame.width * (0.5 + original.x * heightToWidth / 100),
            y: frame.top + frame.height * (0.5 - original.y / 100),
        };
        const startAngle = (Math.atan2(startY - center.y, startX - center.x) * 180) / Math.PI;
        const onMove = (move) => {
            const dx = ((move.clientX - startX) / frame.height) * 100;
            const dy = -((move.clientY - startY) / frame.height) * 100;
            if (handle === "move") {
                lastTransform = { ...original, x: original.x + dx, y: original.y + dy };
            }
            else if (handle === "rotate") {
                const angle = (Math.atan2(move.clientY - center.y, move.clientX - center.x) * 180) / Math.PI;
                lastTransform = {
                    ...original,
                    // Screen angles increase clockwise; Final Cut angles increase in the
                    // opposite direction. Keep the dragged handle under the pointer.
                    rotation: original.rotation - (angle - startAngle),
                };
            }
            else {
                const screenX = move.clientX - startX;
                const screenY = move.clientY - startY;
                const cssAngle = (-original.rotation * Math.PI) / 180;
                const localX = screenX * Math.cos(cssAngle) + screenY * Math.sin(cssAngle);
                const localY = -screenX * Math.sin(cssAngle) + screenY * Math.cos(cssAngle);
                const displayedWidth = Math.max(1, frame.width * Math.abs(original.scale * original.scaleX));
                const displayedHeight = Math.max(1, frame.height * Math.abs(original.scale * original.scaleY));
                const horizontalSign = handle.includes("w") ? -1 : 1;
                const verticalSign = handle.includes("n") ? -1 : 1;
                const sx = Math.max(0.1, 1 + horizontalSign * 2 * localX / displayedWidth);
                const sy = Math.max(0.1, 1 + verticalSign * 2 * localY / displayedHeight);
                if (handle === "e" || handle === "w") {
                    lastTransform = {
                        ...original,
                        scaleX: original.scaleX * sx,
                    };
                }
                else if (handle === "n" || handle === "s") {
                    lastTransform = {
                        ...original,
                        scaleY: original.scaleY * sy,
                    };
                }
                else {
                    lastTransform = {
                        ...original,
                        scale: original.scale * Math.max(0.1, (sx + sy) / 2),
                    };
                }
            }
            this.state.project = updateItem(base, selected.id, { transform: lastTransform });
            this.updateProgramFrame();
            this.updateCanvasControls();
            this.renderInspector();
            this.previewRealtime({
                type: "set-item-transform",
                itemId: selected.id,
                transform: this.selectedItem()?.transform,
            });
        };
        this.finishPointerTransaction(onMove, () => ({
            label: handle === "move"
                ? "Move clip in viewer"
                : handle === "rotate"
                    ? "Rotate clip"
                    : "Scale clip",
            operation: {
                type: "updateClip",
                clipId: selected.id,
                patch: { transform: lastTransform },
            },
        }));
    }
    startCropControl(event, handle) {
        const selected = this.selectedItem();
        if (!selected || !handle) {
            return;
        }
        event.preventDefault();
        const base = this.currentSnapshot();
        const original = clone(selected.video.crop);
        let lastCrop = original;
        const frame = this.canvasControlFrame();
        const startX = event.clientX;
        const startY = event.clientY;
        const onMove = (move) => {
            const dx = ((move.clientX - startX) / frame.width) * 100;
            const dy = ((move.clientY - startY) / frame.height) * 100;
            if (handle === "move") {
                lastCrop = {
                    ...original,
                    left: clamp(original.left + dx, 0, 100 - original.right),
                    right: clamp(original.right - dx, 0, 100 - original.left),
                    top: clamp(original.top + dy, 0, 100 - original.bottom),
                    bottom: clamp(original.bottom - dy, 0, 100 - original.top),
                };
            }
            else {
                lastCrop = {
                    ...original,
                    left: handle.includes("w") ? clamp(original.left + dx, 0, 95) : original.left,
                    right: handle.includes("e") ? clamp(original.right - dx, 0, 95) : original.right,
                    top: handle.includes("n") ? clamp(original.top + dy, 0, 95) : original.top,
                    bottom: handle.includes("s") ? clamp(original.bottom - dy, 0, 95) : original.bottom,
                };
            }
            this.state.project = updateItem(base, selected.id, {
                video: { ...selected.video, crop: lastCrop },
            });
            this.updateProgramFrame();
            this.updateCanvasControls();
            this.renderInspector();
        };
        this.finishPointerTransaction(onMove, () => ({
            label: "Adjust crop",
            operation: {
                type: "updateClip",
                clipId: selected.id,
                patch: { video: { ...selected.video, crop: lastCrop } },
            },
        }));
    }
    startKenBurnsControl(event, token) {
        const selected = this.selectedItem();
        if (!selected || !token) {
            return;
        }
        event.preventDefault();
        const parts = token.split(/-(?=[^-]+$)/);
        const prefix = parts[0] ?? "";
        const corner = parts[1] ?? "";
        const which = prefix.includes("start") ? "kenStart" : "kenEnd";
        const base = this.currentSnapshot();
        const original = clone(selected.video.crop[which]);
        let lastCrop = selected.video.crop;
        const frame = this.canvasControlFrame();
        const startX = event.clientX;
        const startY = event.clientY;
        const onMove = (move) => {
            const dx = ((move.clientX - startX) / frame.width) * 100;
            const dy = ((move.clientY - startY) / frame.height) * 100;
            const width = clamp(original.width + (corner.includes("e") ? dx : -dx), 5, 100);
            const height = clamp(original.height + (corner.includes("s") ? dy : -dy), 5, 100);
            lastCrop = {
                ...selected.video.crop,
                [which]: { ...original, width, height },
            };
            this.state.project = updateItem(base, selected.id, {
                video: { ...selected.video, crop: lastCrop },
            });
            this.updateProgramFrame();
            this.updateCanvasControls();
            this.renderInspector();
        };
        this.finishPointerTransaction(onMove, () => ({
            label: `Adjust Ken Burns ${which}`,
            operation: {
                type: "updateClip",
                clipId: selected.id,
                patch: { video: { ...selected.video, crop: lastCrop } },
            },
        }));
    }
    startDistortControl(event, corner) {
        const selected = this.selectedItem();
        if (!selected || !corner) {
            return;
        }
        event.preventDefault();
        const base = this.currentSnapshot();
        const original = clone(selected.video.distort);
        let lastDistort = original;
        const frame = this.canvasControlFrame();
        const startX = event.clientX;
        const startY = event.clientY;
        const mapping = {
            "top-left": ["topLeftX", "topLeftY"],
            "top-right": ["topRightX", "topRightY"],
            "bottom-left": ["bottomLeftX", "bottomLeftY"],
            "bottom-right": ["bottomRightX", "bottomRightY"],
        };
        const xKeys = [
            "topLeftX",
            "topRightX",
            "bottomLeftX",
            "bottomRightX",
        ];
        const yKeys = [
            "topLeftY",
            "topRightY",
            "bottomLeftY",
            "bottomRightY",
        ];
        const onMove = (move) => {
            const dx = ((move.clientX - startX) / frame.width) * 100;
            const dy = ((move.clientY - startY) / frame.height) * 100;
            let next = { ...original };
            if (corner === "move") {
                for (const key of xKeys) {
                    const current = original[key];
                    if (typeof current === "number") {
                        next = { ...next, [key]: current + dx };
                    }
                }
                for (const key of yKeys) {
                    const current = original[key];
                    if (typeof current === "number") {
                        next = { ...next, [key]: current + dy };
                    }
                }
            }
            else {
                const pair = mapping[corner];
                if (!pair) {
                    return;
                }
                const [xKey, yKey] = pair;
                const xValue = original[xKey];
                const yValue = original[yKey];
                if (typeof xValue === "number" && typeof yValue === "number") {
                    next = { ...next, [xKey]: xValue + dx, [yKey]: yValue + dy };
                }
            }
            lastDistort = next;
            this.state.project = updateItem(base, selected.id, {
                video: { ...selected.video, distort: lastDistort },
            });
            this.updateProgramFrame();
            this.updateCanvasControls();
            this.renderInspector();
        };
        this.finishPointerTransaction(onMove, () => ({
            label: "Adjust distort",
            operation: {
                type: "updateClip",
                clipId: selected.id,
                patch: { video: { ...selected.video, distort: lastDistort } },
            },
        }));
    }
    /**
     * Attach pointer-move listeners and commit one EditOperation on pointer-up.
     *
     * Why this exists: live preview mutates a draft from transactionBase; the
     * accepted commit must send the same logical op through runtime.commitEdit,
     * not restore a full snapshot.
     */
    finishPointerTransaction(onMove, result) {
        this.pointerSession = { active: true };
        const onUp = () => {
            window.removeEventListener("pointermove", onMove);
            window.removeEventListener("pointerup", onUp);
            this.pointerSession = null;
            const transaction = result();
            if (!transaction) {
                this.transactionBase = null;
                return;
            }
            void this.commitEdit(transaction.label, transaction.operation);
        };
        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", onUp, { once: true });
    }
    startTrim(event, itemId, edge) {
        const base = this.currentSnapshot();
        const spineIndex = base.spine.findIndex((clip) => clip.id === itemId);
        const isStorylineStartTrim = edge === "left" && spineIndex >= 0;
        const startX = event.clientX;
        let lastDelta = 0;
        const onMove = (move) => {
            const delta = (move.clientX - startX) / this.state.pixelsPerSecond;
            lastDelta = delta;
            // Final Cut affordance: ripple-trimming a storyline clip's HEAD drags the
            // grabbed clip and everything downstream to the right (so the edit point
            // follows the cursor) during the gesture, then the magnetic timeline snaps
            // the storyline back to its anchor on release. Re-anchoring every frame
            // (the default snapped draft) would instead pin the left edge in place.
            this.state.project = isStorylineStartTrim
                ? this.draftStorylineHeadTrim(base, spineIndex, delta)
                : trimItem(base, itemId, edge, delta);
            this.renderTimeline();
            this.renderViewer();
            this.renderInspector();
        };
        this.finishPointerTransaction(onMove, () => ({
            label: "Ripple trim",
            operation: {
                type: "trim",
                clipId: itemId,
                edge: edge === "left" ? "start" : "end",
                delta: lastDelta,
            },
        }));
    }
    /**
     * Build the non-snapped preview for a storyline head trim: shift the grabbed
     * clip and every clip after it (plus any connected clips anchored to them)
     * right by the trim amount, keeping durations intact so the storyline appears
     * to slide under the cursor. The amount is clamped to the valid trim range so
     * the preview cannot exceed what the committed trim will actually do; the real
     * trim + normalize on pointer-up produces the snapped result.
     */
    draftStorylineHeadTrim(base, spineIndex, delta) {
        const clip = base.spine[spineIndex];
        const shift = clamp(delta, -clip.sourceStart, clip.duration - MIN_CLIP_DURATION);
        const shiftedIds = new Set(base.spine.slice(spineIndex).map((candidate) => candidate.id));
        const spine = base.spine.map((candidate, index) => index >= spineIndex ? { ...candidate, timelineStart: candidate.timelineStart + shift } : candidate);
        const connected = base.connected.map((candidate) => shiftedIds.has(candidate.anchorId)
            ? { ...candidate, timelineStart: candidate.timelineStart + shift }
            : candidate);
        return { ...base, spine, connected };
    }
    /**
     * Roll the shared edit point next to a storyline clip (Trim tool edge-drag).
     *
     * The rollTrim reducer op works on the LEFT clip of an edit point, so a
     * right-edge drag rolls this clip, and a left-edge drag rolls the previous
     * clip. When the edge has no neighbor to roll against (start of the first clip
     * / end of the last clip) we fall back to a ripple trim so the drag still does
     * something sensible.
     */
    startRoll(event, itemId, edge) {
        const spineIds = this.state.project.spine.map((clip) => clip.id);
        const index = spineIds.indexOf(itemId);
        const rollClipId = edge === "right" ? itemId : spineIds[index - 1];
        const rollIndex = rollClipId ? spineIds.indexOf(rollClipId) : -1;
        if (rollIndex < 0 || rollIndex >= spineIds.length - 1) {
            this.startTrim(event, itemId, edge);
            return;
        }
        const rollId = rollClipId;
        const base = this.currentSnapshot();
        const startX = event.clientX;
        let lastDelta = 0;
        const onMove = (move) => {
            lastDelta = (move.clientX - startX) / this.state.pixelsPerSecond;
            this.state.project = applyEdit(base, { type: "rollTrim", clipId: rollId, delta: lastDelta });
            this.renderTimeline();
            this.renderViewer();
            this.renderInspector();
        };
        this.finishPointerTransaction(onMove, () => ({
            label: "Roll edit",
            operation: { type: "rollTrim", clipId: rollId, delta: lastDelta },
        }));
    }
    /**
     * Slip a storyline clip's source in/out (Trim tool body-drag). Timeline
     * position and duration stay put; only the visible footage window moves.
     * Dragging right reveals earlier footage, matching Final Cut's feel.
     */
    startSlip(event, itemId) {
        this.selectStorylineItem(itemId);
        const base = this.currentSnapshot();
        const startX = event.clientX;
        let lastDelta = 0;
        const onMove = (move) => {
            lastDelta = -(move.clientX - startX) / this.state.pixelsPerSecond;
            this.state.project = applyEdit(base, { type: "slip", clipId: itemId, delta: lastDelta });
            this.renderTimeline();
            this.renderViewer();
            this.renderInspector();
        };
        this.finishPointerTransaction(onMove, () => ({
            label: "Slip clip",
            operation: { type: "slip", clipId: itemId, delta: lastDelta },
        }));
    }
    /**
     * Slide a storyline clip along the timeline (Trim tool Option-body-drag),
     * absorbing the shift into its two neighbors so the storyline stays contiguous
     * and equal length.
     */
    startSlide(event, itemId) {
        this.selectStorylineItem(itemId);
        const base = this.currentSnapshot();
        const startX = event.clientX;
        let lastDelta = 0;
        const onMove = (move) => {
            lastDelta = (move.clientX - startX) / this.state.pixelsPerSecond;
            this.state.project = applyEdit(base, { type: "slide", clipId: itemId, delta: lastDelta });
            this.renderTimeline();
            this.renderViewer();
            this.renderInspector();
        };
        this.finishPointerTransaction(onMove, () => ({
            label: "Slide clip",
            operation: { type: "slide", clipId: itemId, delta: lastDelta },
        }));
    }
    /** Select a single storyline clip (used when a Trim gesture begins on its body). */
    selectStorylineItem(itemId) {
        this.state.selectedItemId = itemId;
        this.state.selectedItemIds = [itemId];
        this.state.selectedTransitionStart = null;
        this.renderInspector();
    }
    startConnectedMove(event, itemId, element) {
        if (!["select", "position"].includes(this.state.tool)) {
            return;
        }
        event.preventDefault();
        if (!this.state.selectedItemIds.includes(itemId)) {
            this.setTimelineSelection([itemId], itemId);
        }
        else {
            this.state.selectedItemId = itemId;
        }
        const base = this.currentSnapshot();
        const selectedConnected = base.connected.filter((candidate) => this.state.selectedItemIds.includes(candidate.id));
        const movingItems = selectedConnected.length > 0
            ? selectedConnected
            : base.connected.filter((candidate) => candidate.id === itemId);
        if (!movingItems.length) {
            return;
        }
        const startX = event.clientX;
        const origins = new Map(movingItems.map((item) => [item.id, item.timelineStart]));
        let lastStarts = new Map(origins);
        element.classList.add("moving");
        const onMove = (move) => {
            const rawDelta = (move.clientX - startX) / this.state.pixelsPerSecond;
            const primaryOrigin = origins.get(itemId) ?? 0;
            const snappedPrimary = this.snapTime(primaryOrigin + rawDelta);
            const delta = snappedPrimary - primaryOrigin;
            let next = base;
            const starts = new Map();
            for (const moving of movingItems) {
                const timelineStart = Math.max(0, (origins.get(moving.id) ?? moving.timelineStart) + delta);
                starts.set(moving.id, timelineStart);
                next = moveConnectedItem(next, moving.id, timelineStart);
            }
            lastStarts = starts;
            this.state.project = next;
            this.renderTimeline();
            this.renderViewer();
            this.renderInspector();
        };
        this.finishPointerTransaction(onMove, () => {
            const operations = movingItems.map((moving) => ({
                type: "moveConnected",
                clipId: moving.id,
                timelineStart: lastStarts.get(moving.id) ?? moving.timelineStart,
            }));
            // Multi-select connected moves are one logical gesture; commitEditSequence
            // records a single history entry then applies each moveConnected op.
            void this.commitEditSequence(movingItems.length > 1
                ? `Move ${movingItems.length} connected clips`
                : "Move connected clip", operations);
            return null;
        });
    }
    startPlayheadDrag(event) {
        event.preventDefault();
        const canvas = this.el("timeline-content").querySelector(".timeline-canvas");
        if (!(canvas instanceof HTMLElement)) {
            return;
        }
        const onMove = (move) => {
            this.seek(this.timeFromCanvasPointer(move.clientX, canvas), false);
        };
        const onUp = () => {
            window.removeEventListener("pointermove", onMove);
            window.removeEventListener("pointerup", onUp);
            this.previewRealtime({ type: "seek", time: this.state.currentTime });
        };
        onMove(event);
        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", onUp, { once: true });
    }
    startHandDrag(event) {
        event.preventDefault();
        const scroller = this.el("timeline-scroller");
        const startX = event.clientX;
        const startY = event.clientY;
        const left = scroller.scrollLeft;
        const top = scroller.scrollTop;
        const onMove = (move) => {
            scroller.scrollLeft = left - (move.clientX - startX);
            scroller.scrollTop = top - (move.clientY - startY);
        };
        const onUp = () => {
            window.removeEventListener("pointermove", onMove);
            window.removeEventListener("pointerup", onUp);
        };
        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", onUp, { once: true });
    }
    /**
     * Drag a marquee with the Select tool and select every intersecting clip.
     * Shift or Command/Ctrl adds the hit clips to the existing selection.
     */
    startMarqueeSelection(event, canvas) {
        event.preventDefault();
        const rect = canvas.getBoundingClientRect();
        const start = { x: event.clientX - rect.left, y: event.clientY - rect.top };
        const additive = event.shiftKey || event.metaKey || event.ctrlKey;
        const priorSelection = new Set(this.state.selectedItemIds);
        const overlay = document.createElement("div");
        overlay.className = "marquee-selection";
        canvas.append(overlay);
        this.pointerSession = { active: true };
        let moved = false;
        let liveHits = [];
        const onMove = (move) => {
            const current = { x: move.clientX - rect.left, y: move.clientY - rect.top };
            const left = Math.min(start.x, current.x);
            const top = Math.min(start.y, current.y);
            const width = Math.abs(current.x - start.x);
            const height = Math.abs(current.y - start.y);
            moved = moved || width > 4 || height > 4;
            this.state.marqueeSelection = { left, top, width, height };
            Object.assign(overlay.style, {
                left: `${left}px`,
                top: `${top}px`,
                width: `${width}px`,
                height: `${height}px`,
            });
            // Compare in canvas-local coordinates. Screen-space DOMRects become
            // unreliable as the timeline scrolls or when the drag reaches outside
            // the canvas viewport; local geometry remains stable in both cases.
            const selectionBox = {
                left,
                top,
                right: left + width,
                bottom: top + height,
            };
            liveHits = [...canvas.querySelectorAll(".timeline-clip")]
                .filter((element) => {
                const clipRect = element.getBoundingClientRect();
                const localClip = {
                    left: clipRect.left - rect.left,
                    top: clipRect.top - rect.top,
                    right: clipRect.right - rect.left,
                    bottom: clipRect.bottom - rect.top,
                };
                return boxesIntersect(selectionBox, localClip);
            })
                .flatMap((element) => {
                const id = element.getAttribute("data-item-id");
                return id ? [id] : [];
            });
            const hitSet = new Set(liveHits);
            for (const element of canvas.querySelectorAll(".timeline-clip")) {
                const id = element.getAttribute("data-item-id");
                element.classList.toggle("marquee-hit", Boolean(id && hitSet.has(id)));
            }
        };
        const onUp = () => {
            window.removeEventListener("pointermove", onMove);
            window.removeEventListener("pointerup", onUp);
            this.pointerSession = null;
            this.suppressGeneratedClick();
            for (const element of canvas.querySelectorAll(".timeline-clip.marquee-hit")) {
                element.classList.remove("marquee-hit");
            }
            if (moved) {
                const selected = additive
                    ? new Set([...priorSelection, ...liveHits])
                    : new Set(liveHits);
                this.setTimelineSelection([...selected], liveHits.at(-1) ?? (additive ? this.state.selectedItemId : null));
                this.showToast(`${this.state.selectedItemIds.length} clip${this.state.selectedItemIds.length === 1 ? "" : "s"} selected.`, "info");
            }
            else {
                this.clearTimelineSelection();
                this.seek(this.timeFromCanvasPointer(event.clientX, canvas));
            }
            overlay.remove();
            this.state.marqueeSelection = null;
            this.renderViewer();
            this.renderInspector();
            this.renderTimeline();
        };
        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", onUp, { once: true });
    }
    startRangeSelection(event) {
        const target = event.target;
        if (!(target instanceof Element)) {
            return;
        }
        const canvas = target.closest(".timeline-canvas");
        if (!(canvas instanceof HTMLElement)) {
            return;
        }
        event.preventDefault();
        const rect = canvas.getBoundingClientRect();
        const clip = target.closest(".timeline-clip");
        const clipRect = clip instanceof HTMLElement ? clip.getBoundingClientRect() : null;
        const fixedBand = clipRect
            ? {
                top: clamp(clipRect.top - rect.top, 27, rect.height),
                bottom: clamp(clipRect.bottom - rect.top, 27, rect.height),
            }
            : null;
        const startTime = this.timeFromCanvasPointer(event.clientX, canvas);
        const startY = clamp(event.clientY - rect.top, 27, rect.height);
        const overlay = document.createElement("div");
        overlay.className = "range-selection";
        overlay.innerHTML = "<span></span>";
        canvas.append(overlay);
        this.pointerSession = { active: true };
        const onMove = (move) => {
            const time = this.timeFromCanvasPointer(move.clientX, canvas);
            const y = clamp(move.clientY - rect.top, 27, rect.height);
            this.state.rangeSelection = {
                start: Math.min(startTime, time),
                end: Math.max(startTime, time),
                top: fixedBand?.top ?? Math.min(startY, y),
                bottom: fixedBand?.bottom ?? Math.max(startY, y),
            };
            const range = this.state.rangeSelection;
            Object.assign(overlay.style, {
                left: `${range.start * this.state.pixelsPerSecond}px`,
                width: `${Math.max(1, (range.end - range.start) * this.state.pixelsPerSecond)}px`,
                top: `${range.top}px`,
                height: `${Math.max(5, range.bottom - range.top)}px`,
            });
        };
        const onUp = () => {
            window.removeEventListener("pointermove", onMove);
            window.removeEventListener("pointerup", onUp);
            this.pointerSession = null;
            this.suppressGeneratedClick();
            overlay.remove();
            this.renderTimeline();
            this.showToast(`Range ${formatTimecode(this.state.rangeSelection?.start ?? 0, this.state.project.fps)} – ${formatTimecode(this.state.rangeSelection?.end ?? 0, this.state.project.fps)}`, "info");
        };
        onMove(event);
        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", onUp, { once: true });
    }
    /**
     * Keep Chrome from owning the editor surface.
     *
     * Right-click shows a Studio menu instead of Inspect/Reload. Editor chords
     * call preventDefault so Cmd+R does not reload, Cmd+S does not save HTML,
     * Backspace does not leave the page, and tool keys do not leak.
     *
     * Main callers: bindEvents on contextmenu and capture-phase keydown.
     *
     * Why this exists: Studio is a full editor hosted in a browser tab. Native
     * page chrome is the wrong owner for those events except while typing.
     */
    handleContextMenu(event) {
        if (isTypingTarget(event.target)) {
            return;
        }
        event.preventDefault();
        const target = event.target;
        if (!(target instanceof Element)) {
            return;
        }
        if (this.contextMenu && this.contextMenu.contains(target)) {
            return;
        }
        this.prepareContextSelection(target);
        this.openStudioMenu(event.clientX, event.clientY, this.studioMenuEntries(target));
    }
    closeStudioMenu() {
        this.contextMenu?.remove();
        this.contextMenu = null;
    }
    prepareContextSelection(target) {
        const clip = target.closest(".timeline-clip");
        const clipId = clip?.getAttribute("data-item-id");
        if (clipId && !this.state.selectedItemIds.includes(clipId)) {
            this.setTimelineSelection([clipId], clipId);
            this.renderViewer();
            this.renderInspector();
            this.renderTimeline();
        }
        const assetCard = target.closest(".asset-card");
        const assetId = assetCard?.getAttribute("data-asset-id");
        if (assetId && this.state.selectedAssetId !== assetId) {
            this.state.selectedAssetId = assetId;
            this.refreshBrowserResults();
        }
        const projectId = target.closest("[data-action='select-project']")?.getAttribute("data-project-id");
        if (projectId && projectId !== this.state.selectedProjectId) {
            void this.selectProject(projectId);
        }
        const eventId = target.closest("[data-action='toggle-event']")?.getAttribute("data-event-id");
        if (eventId && eventId !== this.state.selectedEventId) {
            this.state.selectedEventId = eventId;
            this.renderAll();
        }
    }
    studioMenuEntries(target) {
        const history = this.historyAdapter();
        const historyItems = [
            { action: "undo", label: "Undo", shortcut: "⌘Z", disabled: !history.canUndo() },
            { action: "redo", label: "Redo", shortcut: "⇧⌘Z", disabled: !history.canRedo() },
        ];
        if (target.closest(".timeline-clip")) {
            return [
                { action: "blade-at-playhead", label: "Blade at Playhead", shortcut: "⌘B" },
                { action: "add-marker", label: "Add Marker", shortcut: "M" },
                "separator",
                { action: "ripple-delete", label: "Ripple Delete", shortcut: "⌫" },
                "separator",
                ...historyItems,
            ];
        }
        if (target.closest("#timeline-panel")) {
            return [
                { action: "select-all", label: "Select All", shortcut: "⌘A" },
                { action: "fit-timeline", label: "Zoom to Fit", shortcut: "⇧Z" },
                "separator",
                ...historyItems,
            ];
        }
        if (target.closest("#library-panel")) {
            return [
                { action: "new-project", label: "New Project", shortcut: "⌘N" },
                { action: "new-event", label: "New Event", shortcut: "⌥N" },
            ];
        }
        if (target.closest("#browser-panel")) {
            return [{ action: "refresh-media", label: "Refresh" }];
        }
        if (target.closest("#viewer-panel")) {
            return [
                { action: "toggle-play", label: this.state.playing ? "Pause" : "Play", shortcut: "Space" },
                { action: "toggle-loop-playback", label: this.state.loopPlayback ? "Loop Off" : "Loop Playback", shortcut: "⌘L" },
                "separator",
                ...historyItems,
            ];
        }
        return [
            { action: "select-all", label: "Select All", shortcut: "⌘A" },
            "separator",
            ...historyItems,
        ];
    }
    openStudioMenu(x, y, entries) {
        this.closeStudioMenu();
        const menu = document.createElement("div");
        menu.className = "studio-context-menu";
        menu.setAttribute("role", "menu");
        menu.innerHTML = entries
            .map((entry) => {
            if (entry === "separator") {
                return '<div class="studio-context-separator"></div>';
            }
            const attrs = Object.entries(entry.attrs ?? {})
                .map(([name, value]) => ` ${name}="${escapeHtml(value)}"`)
                .join("");
            const shortcut = entry.shortcut ? `<kbd>${escapeHtml(entry.shortcut)}</kbd>` : "";
            const disabled = entry.disabled ? " disabled" : "";
            return `<button data-action="${escapeHtml(entry.action)}" role="menuitem"${disabled}${attrs}><span>${escapeHtml(entry.label)}</span>${shortcut}</button>`;
        })
            .join("");
        this.root.append(menu);
        const width = menu.offsetWidth;
        const height = menu.offsetHeight;
        menu.style.left = `${Math.max(8, Math.min(x, window.innerWidth - width - 8))}px`;
        menu.style.top = `${Math.max(8, Math.min(y, window.innerHeight - height - 8))}px`;
        this.contextMenu = menu;
    }
    selectAllClips() {
        const allIds = this.orderedTimelineItems().map((item) => item.id);
        this.setTimelineSelection(allIds, allIds.at(-1) ?? null);
        this.renderViewer();
        this.renderInspector();
        this.renderTimeline();
        this.showToast(`${allIds.length} clips selected.`, "info");
    }
    requireBrowserAsset() {
        const asset = this.assetById(this.state.selectedAssetId);
        if (!asset) {
            this.showToast("Select a clip in the browser first.", "info");
        }
        return asset;
    }
    editFromBrowserAsset(kind) {
        const asset = this.requireBrowserAsset();
        if (!asset) {
            return;
        }
        if (kind === "append") {
            if (asset.kind === "video" || asset.kind === "image") {
                void this.commitEdit("Append media", insertOperationAtTime(this.state.project, asset, projectDuration(this.state.project)));
                return;
            }
            const connected = connectOperationAtTime(this.state.project, asset, projectDuration(this.state.project));
            if (connected) {
                void this.commitEdit("Connect asset", connected);
            }
            return;
        }
        if (kind === "insert") {
            void this.commitEdit("Insert media", insertOperationAtTime(this.state.project, asset, this.state.currentTime));
            return;
        }
        if (kind === "connect") {
            const connected = connectOperationAtTime(this.state.project, asset, this.state.currentTime);
            if (!connected) {
                this.showToast("Connect needs a storyline clip under the playhead.", "info");
                return;
            }
            void this.commitEdit("Connect asset", connected);
            return;
        }
        void this.commitEdit("Overwrite media", {
            type: "overwrite",
            clip: timelineClipFromAsset(randomId("clip"), asset),
            timelineStart: this.state.currentTime,
        });
    }
    setRangePoint(edge) {
        const time = this.state.currentTime;
        const current = this.state.rangeSelection;
        const frame = 1 / Math.max(1, this.state.project.fps);
        if (edge === "start") {
            const end = current && current.end > time ? current.end : time + frame;
            this.state.rangeSelection = {
                start: time,
                end,
                top: current?.top ?? 0,
                bottom: current?.bottom ?? 24,
            };
        }
        else {
            const start = current && current.start < time ? current.start : Math.max(0, time - frame);
            this.state.rangeSelection = {
                start,
                end: time,
                top: current?.top ?? 0,
                bottom: current?.bottom ?? 24,
            };
        }
        this.setTimelineTool("range");
        this.renderTimeline();
    }
    selectClipUnderPlayhead() {
        const clip = clipAtTime(this.state.project, this.state.currentTime);
        if (!clip) {
            return;
        }
        this.setTimelineSelection([clip.id], clip.id);
        this.renderViewer();
        this.renderInspector();
        this.renderTimeline();
    }
    selectClipRange() {
        const clip = this.selectedItem() ?? clipAtTime(this.state.project, this.state.currentTime);
        if (!clip) {
            return;
        }
        this.state.rangeSelection = {
            start: clip.timelineStart,
            end: clip.timelineStart + clip.duration,
            top: 0,
            bottom: 24,
        };
        this.setTimelineTool("range");
        this.renderTimeline();
    }
    playSelection() {
        const clip = this.selectedItem();
        const range = this.state.rangeSelection;
        const start = range ? range.start : clip?.timelineStart;
        const end = range ? range.end : clip ? clip.timelineStart + clip.duration : null;
        if (start === undefined || end === null) {
            this.showToast("Select a clip or range to play.", "info");
            return;
        }
        this.seek(start);
        if (!this.state.playing) {
            this.togglePlayback();
        }
        this.playbackStopTime = end;
    }
    showRetimeEditor() {
        const item = this.selectedItem();
        if (!item || !supportsPortableRetime(item)) {
            this.showToast("Select a media clip to show the retime editor.", "info");
            return;
        }
        this.state.panels = { ...this.state.panels, inspector: true };
        this.state.inspectorTab = "video";
        this.renderAll();
    }
    addDefaultTransition() {
        const capability = this.requireCapabilities().transitions.find((candidate) => candidate.support !== "unsupported");
        if (!capability) {
            this.showToast("No supported default transition is available.", "info");
            return;
        }
        void this.applyTransitionFromCatalog(capability.id);
    }
    skipFrames(frames) {
        this.seek(this.state.currentTime + frames / Math.max(1, this.state.project.fps));
    }
    handleKeyDown(event) {
        const modifier = event.metaKey || event.ctrlKey;
        const key = event.key.toLowerCase();
        const claim = () => {
            event.preventDefault();
            event.stopImmediatePropagation();
        };
        if (isTypingTarget(event.target)) {
            if (modifier && event.code === "KeyN") {
                claim();
                void this.createNewProject();
                return;
            }
            if (modifier && (key === "s" || key === "p" || key === "r" || key === "o")) {
                claim();
            }
            return;
        }
        if (event.key !== "Meta" &&
            event.key !== "Control" &&
            event.key !== "Alt" &&
            event.key !== "Shift") {
            this.closeStudioMenu();
        }
        if (modifier && event.shiftKey && key === "f") {
            claim();
            void this.el("viewer-wrap").requestFullscreen?.();
            return;
        }
        if (modifier && event.shiftKey && event.code === "Digit2") {
            claim();
            this.state.timelineIndexOpen = !this.state.timelineIndexOpen;
            this.renderTimeline();
            return;
        }
        if (modifier && event.ctrlKey && event.code === "Digit5") {
            claim();
            this.openBrowser("transitions");
            return;
        }
        if (modifier && event.code === "KeyN") {
            claim();
            void this.createNewProject();
            return;
        }
        if (modifier && key === "i") {
            event.preventDefault();
            this.mockImportMedia();
            return;
        }
        if (modifier && key === "o") {
            event.preventDefault();
            return;
        }
        if (modifier && key === "e") {
            event.preventDefault();
            this.exportProject(this.exportProfile);
            return;
        }
        if (modifier && event.code === "Digit5") {
            event.preventDefault();
            this.openBrowser("effects");
            return;
        }
        if (modifier && key === "f") {
            event.preventDefault();
            this.state.timelineIndexOpen = !this.state.timelineIndexOpen;
            this.renderTimeline();
            return;
        }
        if (modifier && key === "t") {
            event.preventDefault();
            this.addDefaultTransition();
            return;
        }
        if (modifier && key === "r") {
            claim();
            this.showRetimeEditor();
            return;
        }
        if (modifier && (key === "c" || key === "x" || key === "v" || key === "s" || key === "p" || key === "g")) {
            event.preventDefault();
            return;
        }
        if (modifier && key === "a") {
            event.preventDefault();
            this.selectAllClips();
            return;
        }
        if (modifier && key === "z") {
            event.preventDefault();
            if (event.shiftKey) {
                this.redo();
            }
            else {
                this.undo();
            }
            return;
        }
        if (modifier && key === "b") {
            event.preventDefault();
            const operation = splitOperationAtTime(this.state.project, this.state.currentTime);
            if (operation) {
                void this.commitEdit("Blade at playhead", operation);
            }
            return;
        }
        if (event.metaKey && key === "l") {
            event.preventDefault();
            this.state.loopPlayback = !this.state.loopPlayback;
            this.renderTimeline();
            return;
        }
        if (event.metaKey && !event.ctrlKey && event.code === "Digit4") {
            event.preventDefault();
            this.togglePanel("inspector");
            return;
        }
        if (event.metaKey && event.ctrlKey && event.code === "Digit1") {
            event.preventDefault();
            this.togglePanel("browser");
            return;
        }
        if (event.metaKey && event.ctrlKey && event.code === "Digit2") {
            event.preventDefault();
            this.togglePanel("timeline");
            return;
        }
        if (modifier && (key === "=" || key === "+" || key === "-")) {
            event.preventDefault();
            const delta = key === "-" ? -8 : 8;
            this.state.pixelsPerSecond = clamp(this.state.pixelsPerSecond + delta, 18, 150);
            this.renderTimeline();
            return;
        }
        if (event.altKey && key === "n") {
            event.preventDefault();
            void this.createNewEvent();
            return;
        }
        if (event.altKey && key === "d") {
            event.preventDefault();
            void this.activateViewerTool("distort");
            return;
        }
        if (event.altKey && (key === "v" || key === "w" || key === "[" || key === "]" || key === "\\")) {
            event.preventDefault();
            return;
        }
        if (event.shiftKey && key === "z") {
            event.preventDefault();
            this.fitTimeline();
            return;
        }
        if (event.shiftKey && key === "t") {
            event.preventDefault();
            void this.activateViewerTool("transform");
            return;
        }
        if (event.shiftKey && key === "c") {
            event.preventDefault();
            void this.activateViewerTool("crop");
            return;
        }
        if (event.shiftKey && (key === "arrowleft" || key === ",")) {
            event.preventDefault();
            this.skipFrames(-10);
            return;
        }
        if (event.shiftKey && (key === "arrowright" || key === ".")) {
            event.preventDefault();
            this.skipFrames(10);
            return;
        }
        if (event.shiftKey && (key === "r" || key === "b" || key === "delete" || key === "backspace")) {
            event.preventDefault();
            return;
        }
        if (modifier) {
            return;
        }
        switch (key) {
            case " ":
                event.preventDefault();
                this.playbackStopTime = null;
                this.togglePlayback();
                break;
            case "e":
                event.preventDefault();
                this.editFromBrowserAsset("append");
                break;
            case "q":
                event.preventDefault();
                this.editFromBrowserAsset("connect");
                break;
            case "w":
                event.preventDefault();
                this.editFromBrowserAsset("insert");
                break;
            case "d":
                event.preventDefault();
                this.editFromBrowserAsset("overwrite");
                break;
            case "i":
                event.preventDefault();
                this.setRangePoint("start");
                break;
            case "o":
                event.preventDefault();
                this.setRangePoint("end");
                break;
            case "x":
                event.preventDefault();
                this.selectClipRange();
                break;
            case "c":
                event.preventDefault();
                this.selectClipUnderPlayhead();
                break;
            case "f":
                event.preventDefault();
                this.toggleFavorite(this.state.selectedAssetId);
                break;
            case "a":
                event.preventDefault();
                this.setTimelineTool("select");
                break;
            case "t":
                event.preventDefault();
                this.setTimelineTool("trim");
                break;
            case "p":
                event.preventDefault();
                this.setTimelineTool("position");
                break;
            case "r":
                event.preventDefault();
                this.setTimelineTool("range");
                break;
            case "b":
                event.preventDefault();
                this.setTimelineTool("blade");
                break;
            case "z":
                event.preventDefault();
                this.setTimelineTool("zoom");
                break;
            case "h":
                event.preventDefault();
                this.setTimelineTool("hand");
                break;
            case "n":
                event.preventDefault();
                this.state.snapping = !this.state.snapping;
                this.renderTimeline();
                break;
            case "s":
                event.preventDefault();
                this.state.skimming = !this.state.skimming;
                this.renderTimeline();
                break;
            case "m":
                event.preventDefault();
                void this.addMarkerAtPlayhead();
                break;
            case "j":
            case "v":
            case "[":
            case "]":
            case "\\":
                event.preventDefault();
                break;
            case "k":
                event.preventDefault();
                if (this.state.playing) {
                    this.togglePlayback();
                }
                break;
            case "l":
                event.preventDefault();
                if (!this.state.playing) {
                    this.togglePlayback();
                }
                break;
            case "delete":
            case "backspace":
                event.preventDefault();
                void this.deleteSelection();
                break;
            case "arrowleft":
            case ",":
                event.preventDefault();
                this.skipFrames(-1);
                break;
            case "arrowright":
            case ".":
                event.preventDefault();
                this.skipFrames(1);
                break;
            case "arrowup":
            case ";":
                event.preventDefault();
                this.jumpEdit(-1);
                break;
            case "arrowdown":
            case "'":
                event.preventDefault();
                this.jumpEdit(1);
                break;
            case "home":
                event.preventDefault();
                this.seek(0);
                break;
            case "end":
                event.preventDefault();
                this.seek(projectDuration(this.state.project));
                break;
            case "/":
                event.preventDefault();
                this.playSelection();
                break;
            case "escape":
                event.preventDefault();
                this.state.activePopover = null;
                this.state.rangeSelection = null;
                this.state.marqueeSelection = null;
                this.clearTimelineSelection();
                this.closeMockInventory();
                document.getElementById("viewer-tools-popover")?.setAttribute("hidden", "");
                for (const popover of this.root.querySelectorAll(".section-keyframe-popover")) {
                    popover.remove();
                }
                this.renderTimeline();
                this.renderViewer();
                this.renderInspector();
                break;
            default:
                break;
        }
    }
    async deleteSelection() {
        const itemIds = this.state.selectedItemIds.length
            ? [...this.state.selectedItemIds]
            : this.state.selectedItemId
                ? [this.state.selectedItemId]
                : [];
        if (!itemIds.length) {
            return;
        }
        this.state.selectedItemIds = [];
        this.state.selectedItemId = null;
        await this.commitEdit(itemIds.length === 1 ? "Ripple delete" : `Ripple delete ${itemIds.length} clips`, { type: "delete", clipIds: itemIds });
    }
    setTimelineTool(tool) {
        if (!tool || !["select", "trim", "position", "range", "blade", "zoom", "hand"].includes(tool)) {
            return;
        }
        this.state.tool = tool;
        this.state.activePopover = null;
        this.renderTimeline();
    }
    async activateViewerTool(tool) {
        if (!tool || !["transform", "crop", "distort", "none"].includes(tool)) {
            return;
        }
        const selected = this.selectedItem();
        this.state.viewerTool = tool;
        this.state.activePopover = null;
        if (!selected || tool === "none") {
            this.renderViewer();
            return;
        }
        const path = tool === "transform"
            ? "transform.enabled"
            : tool === "crop"
                ? "video.crop.enabled"
                : "video.distort.enabled";
        if (!Boolean(getPath(selected, path))) {
            await this.commitEdit(`Enable ${tool}`, {
                type: "updateClipPath",
                clipId: selected.id,
                path,
                value: true,
            });
            return;
        }
        this.renderViewer();
        this.renderInspector();
    }
    async setCropMode(mode) {
        if (!mode || !["trim", "crop", "ken-burns"].includes(mode)) {
            return;
        }
        if (mode === "ken-burns" && this.runtime.mode === "localhost") {
            this.showMockNotice("ken-burns-editor");
            return;
        }
        const item = this.selectedItem();
        if (!item) {
            return;
        }
        await this.commitEdit(`Set Crop to ${mode}`, {
            type: "updateClip",
            clipId: item.id,
            patch: {
                video: {
                    ...item.video,
                    crop: {
                        ...item.video.crop,
                        enabled: true,
                        type: mode,
                    },
                },
            },
        });
    }
    selectLibrarySource(source) {
        if (!source || !["libraries", "photos-audio", "titles-generators"].includes(source)) {
            return;
        }
        this.state.librarySource = source;
        this.state.panels = { ...this.state.panels, browser: true };
        if (source === "photos-audio") {
            this.state.mediaTab = "audio";
        }
        if (source === "titles-generators") {
            this.state.mediaTab = "titles";
        }
        this.state.activePopover = null;
        this.renderAll();
    }
    showKeyframeMenu(element) {
        const section = element.closest(".inspector-section");
        const header = element.closest(".section-header");
        const path = element.getAttribute("data-path");
        if (!section || !header || !path) {
            return;
        }
        const existing = section.querySelector(".section-keyframe-popover");
        for (const open of this.root.querySelectorAll(".section-keyframe-popover")) {
            open.remove();
        }
        if (existing) {
            return;
        }
        const item = this.selectedItem();
        const count = item?.keyframes[path]?.length ?? 0;
        const popover = document.createElement("div");
        popover.className = "section-keyframe-popover";
        popover.innerHTML = `
      <button data-action="toggle-keyframe" data-path="${path}">${count ? "Add / Remove at Playhead" : "Add Keyframe at Playhead"}</button>
      <button data-action="previous-keyframe" data-path="${path}" ${count ? "" : "disabled"}>Previous Keyframe</button>
      <button data-action="next-keyframe" data-path="${path}" ${count ? "" : "disabled"}>Next Keyframe</button>
      <button data-action="clear-keyframes" data-path="${path}" ${count ? "" : "disabled"}>Delete All Keyframes</button>`;
        header.append(popover);
    }
    seekParameterKeyframe(path, direction) {
        const item = this.selectedItem();
        if (!item || !path) {
            return;
        }
        const values = [...(item.keyframes[path] ?? [])].sort((left, right) => left.time.seconds - right.time.seconds);
        if (!values.length) {
            return;
        }
        const local = this.state.currentTime - item.timelineStart;
        const candidate = direction < 0
            ? [...values].reverse().find((frame) => frame.time.seconds < local - 1 / this.state.project.fps) ??
                values.at(-1)
            : (values.find((frame) => frame.time.seconds > local + 1 / this.state.project.fps) ?? values[0]);
        if (candidate !== undefined) {
            this.seek(item.timelineStart + candidate.time.seconds);
        }
        this.renderInspector();
    }
    async clearParameterKeyframes(path) {
        const item = this.selectedItem();
        if (!item || !path) {
            return;
        }
        await this.commitEdit(`Delete ${path} keyframes`, {
            type: "clearKeyframes",
            clipId: item.id,
            path,
        });
    }
    togglePlayback() {
        const duration = projectDuration(this.state.project);
        if (!this.state.playing && this.state.currentTime >= duration - 1e-6) {
            this.state.currentTime = 0;
        }
        this.state.playing = !this.state.playing;
        this.renderViewer();
        this.previewRealtime({ type: this.state.playing ? "play" : "pause" });
        if (this.state.playing) {
            this.lastPlaybackTime = performance.now();
            this.playbackFrame = requestAnimationFrame((time) => this.tickPlayback(time));
            this.watchPreviewWarmup();
        }
        else {
            cancelAnimationFrame(this.playbackFrame);
            this.clearPreviewWarmup();
        }
    }
    /**
     * Cover the viewer with a brief "Preparing preview…" spinner while the
     * localhost render pipeline cold-starts.
     *
     * Why this exists:
     * The backend recreates its decoders on the first play of a session and on
     * every quality change, so the first ~1s of a fresh producer arrives at a
     * choppy ~7fps before settling to realtime 30fps. Rather than let that read
     * as lag, we show an explicit buffering state until real frames flow.
     *
     * How it decides "warm": it polls the <video>'s decoded-frame counter
     * (getVideoPlaybackQuality) every 150ms and computes the delivered fps for
     * that window. Two consecutive windows at >=20fps means the stream reached
     * realtime, so we hide the spinner. (We poll decoded frames rather than use
     * requestVideoFrameCallback because rVFC only fires on actual compositing,
     * which is unreliable in background/automation tabs, whereas the decode
     * counter always advances.) To avoid flashing on an already-warm play we only
     * reveal the spinner if still cold after a 200ms grace period, and a 3s
     * safety timeout guarantees it can never stick.
     *
     * Only the localhost runtime streams a real <video> that cold-starts; the
     * mock canvas has no warm-up, so this is a no-op there.
     */
    watchPreviewWarmup() {
        // Only the WebRTC <video> path warms via the decode counter. The raw-frame
        // transport owns its own spinner (revealed on Play, hidden on first painted
        // frame), so this heuristic must not run there or it would spin for its
        // full safety timeout against a video element that never decodes.
        if (this.state.connectionMode !== "localhost" || this.preview?.mode !== "webrtc") {
            return;
        }
        const video = document.getElementById("preview-video");
        const overlay = document.getElementById("viewer-warmup");
        if (!video || !overlay || typeof video.getVideoPlaybackQuality !== "function") {
            return;
        }
        // Replace any prior watcher so only one runs at a time.
        this.clearPreviewWarmup();
        const token = { cancelled: false };
        this.previewWarmupToken = token;
        const SAMPLE_MS = 150;
        const WARM_FPS = 20; // delivered fps that counts as "realtime enough"
        const WARM_SAMPLES = 2; // consecutive good windows before we reveal the video
        let lastFrames = video.getVideoPlaybackQuality().totalVideoFrames;
        let lastTs = performance.now();
        let goodRun = 0;
        const showTimer = window.setTimeout(() => {
            if (!token.cancelled) {
                overlay.hidden = false;
            }
        }, 200);
        const finish = () => {
            if (token.cancelled) {
                return;
            }
            token.cancelled = true;
            window.clearTimeout(showTimer);
            window.clearTimeout(safetyTimer);
            window.clearInterval(poll);
            overlay.hidden = true;
        };
        // Backstop so the spinner can never stick. A cold start warms in ~1s, but a
        // quality change mid-playback (full producer swap + renegotiation) can take
        // ~3s, so this is generous enough to let detection win on that slower path.
        const safetyTimer = window.setTimeout(finish, 5000);
        const poll = window.setInterval(() => {
            if (token.cancelled || !this.state.playing) {
                finish();
                return;
            }
            const now = performance.now();
            const frames = video.getVideoPlaybackQuality().totalVideoFrames;
            if (frames < lastFrames) {
                // A quality change renegotiates the track and resets the decode
                // counter; re-baseline instead of reading a bogus negative rate.
                lastFrames = frames;
                lastTs = now;
                goodRun = 0;
                return;
            }
            const fps = (frames - lastFrames) / ((now - lastTs) / 1000);
            lastFrames = frames;
            lastTs = now;
            goodRun = fps >= WARM_FPS ? goodRun + 1 : 0;
            if (goodRun >= WARM_SAMPLES) {
                finish();
            }
        }, SAMPLE_MS);
    }
    /** True while the localhost producer has not yet delivered realtime frames. */
    previewIsWarming() {
        if (this.previewWarmupToken) {
            return true;
        }
        const overlay = document.getElementById("viewer-warmup");
        return overlay !== null && overlay.hidden === false;
    }
    /** Cancel the warm-up watcher and hide the spinner immediately. */
    clearPreviewWarmup() {
        if (this.previewWarmupToken) {
            this.previewWarmupToken.cancelled = true;
            this.previewWarmupToken = null;
        }
        const overlay = document.getElementById("viewer-warmup");
        if (overlay) {
            overlay.hidden = true;
        }
    }
    playReverse() {
        if (this.state.playing) {
            this.togglePlayback();
        }
        this.seek(this.state.currentTime - 1);
        this.showToast("J · reverse shuttle preview", "info");
    }
    tickPlayback(now) {
        if (!this.state.playing) {
            return;
        }
        // Keep the playhead parked until the preview has a real frame. The local
        // rAF clock otherwise walks ahead of a cold producer, then the first SSE
        // time event snaps it back.
        if (this.previewIsWarming()) {
            this.lastPlaybackTime = now;
            this.playbackFrame = requestAnimationFrame((time) => this.tickPlayback(time));
            return;
        }
        const delta = Math.min(0.1, (now - this.lastPlaybackTime) / 1000);
        this.lastPlaybackTime = now;
        const duration = projectDuration(this.state.project);
        this.state.currentTime += delta;
        const stopAt = this.playbackStopTime ?? duration;
        if (this.state.currentTime >= stopAt) {
            if (this.playbackStopTime !== null || !this.state.loopPlayback) {
                this.state.currentTime = this.playbackStopTime !== null ? this.playbackStopTime : duration;
                this.playbackStopTime = null;
                this.state.playing = false;
                cancelAnimationFrame(this.playbackFrame);
                this.renderViewer();
                this.previewRealtime({ type: "pause" });
                return;
            }
            this.state.currentTime = 0;
            this.previewRealtime({ type: "seek", time: 0 });
        }
        this.updatePlaybackDom();
        if (this.state.continuousScroll) {
            const scroller = this.el("timeline-scroller");
            const x = this.state.currentTime * this.state.pixelsPerSecond;
            scroller.scrollLeft = Math.max(0, x - scroller.clientWidth / 2);
        }
        this.playbackFrame = requestAnimationFrame((time) => this.tickPlayback(time));
    }
    /**
     * Move the playhead to `time`.
     *
     * `notify` echoes the seek to the preview runtime (true for user scrubs,
     * false for playback time-updates streamed back FROM the preview).
     *
     * `rebuildInspector` controls whether the whole Inspector panel is
     * regenerated. A passive time-update fires on every preview frame; a full
     * `innerHTML` teardown dozens of times per second saturates the main thread
     * and makes the preview drift behind realtime. Those passive frames instead
     * do a cheap text refresh of the source-time readout via
     * `updateInspectorPlayhead()`. User-initiated seeks keep the full rebuild so
     * parameter values stay in sync.
     */
    seek(time, notify = true, rebuildInspector = true) {
        this.state.currentTime = clamp(time, 0, projectDuration(this.state.project));
        this.updatePlaybackDom();
        if (rebuildInspector) {
            this.renderInspector();
        }
        else {
            this.updateInspectorPlayhead();
        }
        if (notify) {
            const duration = projectDuration(this.state.project);
            const previewTime = this.state.currentTime < duration || duration <= 0
                ? this.state.currentTime
                : Math.max(0, duration - 1 / Math.max(1, this.state.project.fps));
            this.previewRealtime({ type: "seek", time: previewTime });
        }
    }
    /**
     * Cheaply refresh only the Inspector's playhead-dependent readout (the
     * source-time span) without regenerating the panel. Used on passive playback
     * frames so the inspector still tracks the playhead but costs a single text
     * assignment instead of a full innerHTML rebuild.
     */
    updateInspectorPlayhead() {
        const readout = document.getElementById("inspector-source-time");
        if (!readout) {
            return;
        }
        const item = this.selectedItem();
        readout.textContent = item
            ? formatTimecode(item.sourceStart + Math.max(0, this.state.currentTime - item.timelineStart), this.state.project.fps)
            : "";
    }
    updatePlaybackDom() {
        const left = this.state.currentTime * this.state.pixelsPerSecond;
        const playhead = document.getElementById("playhead");
        if (playhead) {
            playhead.style.left = `${left}px`;
        }
        const current = document.getElementById("transport-current");
        if (current) {
            current.textContent = formatTimecode(this.state.currentTime, this.state.project.fps);
        }
        const active = this.activeStorylineItem();
        const art = document.getElementById("program-art");
        if (art && active) {
            art.style.setProperty("--active-color", active.colors.a);
            art.style.setProperty("--active-index", String(this.state.project.spine.indexOf(active)));
        }
        if (this.state.playing) {
            this.updateProgramFrame();
        }
    }
    updateProgramFrame() {
        const frame = this.el("program-frame");
        const art = this.el("program-art");
        const item = this.selectedItem() ?? this.activeStorylineItem();
        frame.classList.toggle("overscan", this.state.overscan);
        frame.classList.toggle("guides-visible", this.state.guides);
        const fitMode = this.state.viewerZoom === 0;
        frame.classList.toggle("fit-view", fitMode);
        frame.style.aspectRatio = `${this.state.project.width} / ${this.state.project.height}`;
        frame.style.setProperty("--viewer-zoom", String((this.state.viewerZoom || 50) / 100));
        if (fitMode) {
            this.applyFitScale(frame);
        }
        const matte = this.el("viewer-wrap").querySelector(".viewer-black-matte");
        if (matte instanceof HTMLElement) {
            matte.dataset.background = this.state.viewerView.background;
        }
        if (!item || item.kind === "audio") {
            art.style.inset = "0";
            art.style.width = "auto";
            art.style.height = "auto";
            art.style.transform = "none";
            art.style.filter = "none";
            art.style.clipPath = "none";
            return;
        }
        const t = item.transform;
        const displayTransform = t.enabled ? t : defaultTransform();
        const c = item.video.color;
        const geometry = transformOverlayGeometry(this.state.project.width, this.state.project.height, displayTransform);
        art.style.inset = "auto";
        art.style.width = `${geometry.widthPercent}%`;
        art.style.height = `${geometry.heightPercent}%`;
        art.style.left = `${geometry.centerXPercent - geometry.widthPercent / 2}%`;
        art.style.top = `${geometry.centerYPercent - geometry.heightPercent / 2}%`;
        art.style.transformOrigin = "center";
        art.style.transform = `rotate(${geometry.rotationDegrees}deg)`;
        art.style.opacity = String(displayTransform.opacity);
        const tonalBrightness = c.midtones / 420 + c.highlights / 900 + c.shadows / 900;
        art.style.filter = `brightness(${Math.max(0.15, 1 + c.exposure / 150 + tonalBrightness)}) contrast(${Math.max(0.1, 1 + c.contrast / 100)}) saturate(${Math.max(0, 1 + c.saturation / 100)}) sepia(${Math.max(0, c.temperature) / 400}) hue-rotate(${c.tint * 0.45}deg)`;
        const crop = item.video.crop;
        art.style.clipPath =
            crop.enabled && crop.type !== "ken-burns"
                ? `inset(${crop.top}% ${crop.right}% ${crop.bottom}% ${crop.left}%)`
                : "none";
        const title = this.el("title-preview");
        const connectedTitle = this.state.project.connected.find((candidate) => candidate.kind === "title" &&
            this.state.currentTime >= candidate.timelineStart &&
            this.state.currentTime < candidate.timelineStart + candidate.duration);
        title.textContent = item.kind === "title"
            ? (item.text || item.name)
            : (connectedTitle?.text || connectedTitle?.name || "");
        title.hidden = item.kind !== "title" && !connectedTitle;
    }
    /**
     * Compute a real "Fit" scale so the program frame fills the viewer area,
     * instead of the old fixed 0.96. `offsetWidth/Height` report the frame's
     * untransformed layout box (CSS transforms do not affect them), so we can
     * measure it against the available viewer-wrap area and pick the largest
     * uniform scale that still fits both dimensions, with a small margin. Written
     * to `--fit-scale`, which `.program-frame.fit-view` consumes.
     */
    applyFitScale(frame) {
        const wrap = this.el("viewer-wrap");
        const baseWidth = frame.offsetWidth;
        const baseHeight = frame.offsetHeight;
        if (baseWidth === 0 || baseHeight === 0) {
            frame.style.setProperty("--fit-scale", "0.96");
            return;
        }
        const availableWidth = Math.max(0, wrap.clientWidth - 24);
        const availableHeight = Math.max(0, wrap.clientHeight - 24);
        const fit = Math.min(availableWidth / baseWidth, availableHeight / baseHeight);
        frame.style.setProperty("--fit-scale", String(Math.max(0.05, Math.min(4, fit))));
    }
    updateCanvasControls() {
        const item = this.selectedItem();
        this.updateLiveCanvasOverlayGeometry();
        const controls = this.state.connectionMode === "localhost"
            ? this.el("live-canvas-controls")
            : this.el("canvas-controls");
        if (!item || item.kind === "audio" || this.state.playing || this.state.viewerTool === "none") {
            return;
        }
        if (this.state.viewerTool === "transform") {
            const box = controls.querySelector(".transform-box");
            if (!(box instanceof HTMLElement)) {
                return;
            }
            const geometry = transformOverlayGeometry(this.state.project.width, this.state.project.height, item.transform);
            box.style.width = `${geometry.widthPercent}%`;
            box.style.height = `${geometry.heightPercent}%`;
            box.style.left = `${geometry.centerXPercent - geometry.widthPercent / 2}%`;
            box.style.top = `${geometry.centerYPercent - geometry.heightPercent / 2}%`;
            box.style.transform = `rotate(${geometry.rotationDegrees}deg)`;
            const anchor = box.querySelector(".anchor-center");
            if (anchor instanceof HTMLElement) {
                anchor.style.left = `${geometry.anchorXPercent}%`;
                anchor.style.top = `${geometry.anchorYPercent}%`;
            }
            const rotationArm = box.querySelector(".rotation-arm");
            if (rotationArm instanceof HTMLElement) {
                rotationArm.style.left = `${geometry.anchorXPercent}%`;
                rotationArm.style.top = `${geometry.anchorYPercent}%`;
            }
            const rotationHandle = box.querySelector(".rotation-handle");
            if (rotationHandle instanceof HTMLElement) {
                rotationHandle.style.left = `${geometry.anchorXPercent}%`;
                rotationHandle.style.top = `calc(${geometry.anchorYPercent}% - 48px)`;
            }
            return;
        }
        if (this.state.viewerTool === "crop") {
            const crop = item.video.crop;
            if (crop.type === "ken-burns") {
                for (const which of ["start", "end"]) {
                    const data = which === "start" ? crop.kenStart : crop.kenEnd;
                    const win = controls.querySelector(`.ken-window.${which}`);
                    if (win instanceof HTMLElement) {
                        win.style.width = `${data.width}%`;
                        win.style.height = `${data.height}%`;
                        win.style.left = `${data.x - data.width / 2}%`;
                        win.style.top = `${data.y - data.height / 2}%`;
                    }
                }
            }
            else {
                const win = controls.querySelector(".crop-window");
                if (win instanceof HTMLElement) {
                    win.style.left = `${crop.left}%`;
                    win.style.right = `${crop.right}%`;
                    win.style.top = `${crop.top}%`;
                    win.style.bottom = `${crop.bottom}%`;
                }
                const shades = {
                    top: ["top", "0", "0", `${crop.top}%`],
                    right: ["0", "0", `${crop.right}%`, "0"],
                    bottom: ["0", "0", "0", `${crop.bottom}%`],
                    left: ["0", `${crop.left}%`, "0", "0"],
                };
                for (const [name, values] of Object.entries(shades)) {
                    const shade = controls.querySelector(`.crop-shade.${name}`);
                    if (shade instanceof HTMLElement) {
                        shade.style.top = values[0];
                        shade.style.right = values[1];
                        shade.style.bottom = values[2];
                        shade.style.left = values[3];
                    }
                }
            }
            return;
        }
        if (this.state.viewerTool === "distort") {
            const d = item.video.distort;
            const points = {
                "top-left": [d.topLeftX, d.topLeftY],
                "top-right": [100 + d.topRightX, d.topRightY],
                "bottom-right": [100 + d.bottomRightX, 100 + d.bottomRightY],
                "bottom-left": [d.bottomLeftX, 100 + d.bottomLeftY],
            };
            for (const [name, pair] of Object.entries(points)) {
                const handle = controls.querySelector(`.distort-handle.${name}`);
                if (handle instanceof HTMLElement) {
                    handle.style.left = `${pair[0]}%`;
                    handle.style.top = `${pair[1]}%`;
                }
            }
            const path = controls.querySelector("#distort-path");
            const topLeft = points["top-left"];
            const topRight = points["top-right"];
            const bottomRight = points["bottom-right"];
            const bottomLeft = points["bottom-left"];
            if (path && topLeft && topRight && bottomRight && bottomLeft) {
                path.setAttribute("d", `M${topLeft[0]} ${topLeft[1]} L${topRight[0]} ${topRight[1]} L${bottomRight[0]} ${bottomRight[1]} L${bottomLeft[0]} ${bottomLeft[1]} Z`);
            }
        }
    }
    /**
     * Return the exact displayed program-image rectangle used by pointer drags.
     *
     * Main callers: Transform, Crop, Ken Burns, and Distort gesture starts.
     * Why this exists: the live preview canvas fills the viewer but uses
     * object-fit: contain, so its editable image can be letterboxed inside that
     * element. Pointer deltas must be normalized against the visible image, not
     * against the entire viewer or the hidden fixture frame.
     */
    canvasControlFrame() {
        if (this.state.connectionMode === "localhost") {
            this.updateLiveCanvasOverlayGeometry();
            return this.el("live-canvas-overlay").getBoundingClientRect();
        }
        return this.el("program-frame").getBoundingClientRect();
    }
    /** Align live onscreen controls with the object-fit:contain preview pixels. */
    updateLiveCanvasOverlayGeometry() {
        if (this.state.connectionMode !== "localhost") {
            return;
        }
        const wrap = this.el("viewer-wrap").getBoundingClientRect();
        const canvas = document.getElementById("preview-canvas");
        const video = document.getElementById("preview-video");
        const surface = canvas instanceof HTMLCanvasElement && !canvas.hidden
            ? canvas
            : video instanceof HTMLVideoElement && !video.hidden
                ? video
                : null;
        if (!surface) {
            return;
        }
        const surfaceRect = surface.getBoundingClientRect();
        const aspect = this.state.project.width / this.state.project.height;
        let width = surfaceRect.width;
        let height = width / aspect;
        if (height > surfaceRect.height) {
            height = surfaceRect.height;
            width = height * aspect;
        }
        const overlay = this.el("live-canvas-overlay");
        overlay.style.left = `${surfaceRect.left - wrap.left + (surfaceRect.width - width) / 2}px`;
        overlay.style.top = `${surfaceRect.top - wrap.top + (surfaceRect.height - height) / 2}px`;
        overlay.style.width = `${width}px`;
        overlay.style.height = `${height}px`;
    }
    updateViewerGuides() {
        const guides = this.el("viewer-guides");
        guides.classList.toggle("visible", this.state.viewerView.actionSafe ||
            this.state.viewerView.titleSafe ||
            this.state.viewerView.horizon);
        guides.classList.toggle("show-action", this.state.viewerView.actionSafe);
        guides.classList.toggle("show-title", this.state.viewerView.titleSafe);
        guides.classList.toggle("show-horizon", this.state.viewerView.horizon);
    }
    /**
     * Accept one user edit through the runtime EditOperation path.
     *
     * Main callers: clicks, shortcuts, inspector commits, gesture pointer-up.
     * Why this exists: pointer-move may draft locally, but only commitEdit
     * advances the canonical revision. Undo/redo must not use this path.
     */
    async commitEdit(label, operation) {
        if (!this.state.projectEditable) {
            this.showToast(`Project is read-only: ${this.state.projectEditReasons.join("; ")}`, "error");
            return;
        }
        const base = this.transactionBase ?? this.parameterBase ?? this.state.project;
        const rootBase = this.rootProject;
        const selectionSeq = this.projectSelectionSeq;
        const baseRevision = base.revision;
        let preview;
        try {
            preview = applyEdit(base, operation);
        }
        catch (error) {
            this.showToast(`Edit rejected: ${error instanceof Error ? error.message : String(error)}`, "error");
            this.state.project = base;
            this.transactionBase = null;
            this.parameterBase = null;
            this.renderAll();
            this.fitTimelineAfterProjectOpen();
            this.syncPreview();
            return;
        }
        const errors = validateProject(preview);
        if (errors.length) {
            this.showToast(`Edit rejected: ${errors[0]}`, "error");
            this.state.project = base;
            this.transactionBase = null;
            this.parameterBase = null;
            this.renderAll();
            this.syncPreview();
            return;
        }
        this.state.isSaving = true;
        this.renderTopbar();
        try {
            const canonical = this.activeScopeId
                ? await this.runtime.restoreProject(replaceScopeProject(rootBase, this.activeScopeId, preview), rootBase.revision)
                : await this.runtime.commitEdit({
                    projectId: base.id,
                    baseRevision,
                    label,
                    operation,
                });
            if (selectionSeq !== this.projectSelectionSeq || rootBase.id !== this.state.selectedProjectId) {
                return;
            }
            this.appHistory = recordHistory(this.appHistory, rootBase);
            this.adoptCanonicalProject(canonical);
            this.applyProjectEditability();
            this.state.currentTime = clamp(this.state.currentTime, 0, projectDuration(this.state.project));
            this.state.connectionMessage =
                this.runtime.mode === "localhost"
                    ? `Saved ${this.state.project.name}`
                    : `Fixture transaction · ${label}`;
        }
        catch (error) {
            if (selectionSeq !== this.projectSelectionSeq || rootBase.id !== this.state.selectedProjectId) {
                return;
            }
            if (this.runtime.mode === "localhost") {
                this.recoverRuntimeProject(base.id);
            }
            else {
                this.rootProject = rootBase;
                this.state.project = projectForScope(rootBase, this.activeScopeId);
            }
            this.state.connectionMessage =
                error instanceof Error ? error.message : "Unknown save failure";
            this.showToast(`Not saved: ${this.state.connectionMessage}`, "error");
        }
        finally {
            this.transactionBase = null;
            this.parameterBase = null;
            this.state.isSaving = false;
            this.renderAll();
            this.syncPreview();
        }
    }
    /**
     * Commit several EditOperations as one logical history entry.
     *
     * Why this exists: multi-select connected-clip moves need one undo step but
     * the reducer only has single-clip moveConnected.
     */
    async commitEditSequence(label, operations) {
        if (!this.state.projectEditable) {
            this.showToast(`Project is read-only: ${this.state.projectEditReasons.join("; ")}`, "error");
            return;
        }
        if (operations.length === 0) {
            this.transactionBase = null;
            this.parameterBase = null;
            return;
        }
        const first = operations[0];
        if (operations.length === 1 && first) {
            await this.commitEdit(label, first);
            return;
        }
        const base = this.transactionBase ?? this.parameterBase ?? this.state.project;
        const rootBase = this.rootProject;
        const selectionSeq = this.projectSelectionSeq;
        let cursor = base;
        for (const operation of operations) {
            try {
                cursor = applyEdit(cursor, operation);
            }
            catch (error) {
                this.showToast(`Edit rejected: ${error instanceof Error ? error.message : String(error)}`, "error");
                this.state.project = base;
                this.transactionBase = null;
                this.parameterBase = null;
                this.renderAll();
                this.syncPreview();
                return;
            }
        }
        const errors = validateProject(cursor);
        if (errors.length) {
            this.showToast(`Edit rejected: ${errors[0]}`, "error");
            this.state.project = base;
            this.transactionBase = null;
            this.parameterBase = null;
            this.renderAll();
            this.syncPreview();
            return;
        }
        this.state.isSaving = true;
        this.renderTopbar();
        try {
            const canonical = this.activeScopeId
                ? await this.runtime.restoreProject(replaceScopeProject(rootBase, this.activeScopeId, cursor), rootBase.revision)
                : await this.runtime.commitEditSequence({
                    projectId: base.id,
                    baseRevision: base.revision,
                    label,
                    operations,
                });
            if (selectionSeq !== this.projectSelectionSeq || rootBase.id !== this.state.selectedProjectId) {
                return;
            }
            this.appHistory = recordHistory(this.appHistory, rootBase);
            this.adoptCanonicalProject(canonical);
            this.applyProjectEditability();
            this.state.currentTime = clamp(this.state.currentTime, 0, projectDuration(this.state.project));
            this.state.connectionMessage =
                this.runtime.mode === "localhost"
                    ? `Saved ${this.state.project.name}`
                    : `Fixture transaction · ${label}`;
        }
        catch (error) {
            if (selectionSeq !== this.projectSelectionSeq || rootBase.id !== this.state.selectedProjectId) {
                return;
            }
            if (this.runtime.mode === "localhost") {
                this.recoverRuntimeProject(base.id);
            }
            else {
                this.rootProject = rootBase;
                this.state.project = projectForScope(rootBase, this.activeScopeId);
            }
            this.state.connectionMessage =
                error instanceof Error ? error.message : "Unknown save failure";
            this.showToast(`Not saved: ${this.state.connectionMessage}`, "error");
        }
        finally {
            this.transactionBase = null;
            this.parameterBase = null;
            this.state.isSaving = false;
            this.renderAll();
            this.syncPreview();
        }
    }
    /** Save a direct Project model mutation as one complete-library history entry. */
    async commitDirectProject(label, editedDisplay) {
        if (!this.state.projectEditable) {
            this.showToast(`Project is read-only: ${this.state.projectEditReasons.join("; ")}`, "error");
            return;
        }
        const rootBase = this.rootProject;
        const candidate = this.activeScopeId
            ? replaceScopeProject(rootBase, this.activeScopeId, editedDisplay)
            : editedDisplay;
        this.state.isSaving = true;
        this.renderTopbar();
        try {
            const canonical = await this.runtime.restoreProject(candidate, rootBase.revision);
            this.appHistory = recordHistory(this.appHistory, rootBase);
            this.adoptCanonicalProject(canonical);
            this.applyProjectEditability();
            this.state.connectionMessage = this.runtime.mode === "localhost" ? `Saved ${rootBase.name}` : `Fixture transaction · ${label}`;
        }
        catch (error) {
            if (this.runtime.mode === "localhost")
                this.recoverRuntimeProject(rootBase.id);
            else
                this.adoptCanonicalProject(rootBase);
            this.showToast(`Not saved: ${error instanceof Error ? error.message : String(error)}`, "error");
        }
        finally {
            this.state.isSaving = false;
            this.renderAll();
            this.syncPreview();
        }
    }
    async setContainerControl(input, action) {
        const itemId = input.getAttribute("data-item-id");
        const item = itemId ? this.itemById(itemId) : null;
        if (!item?.container)
            return;
        let container = structuredClone(item.container);
        if (container.kind === "multicam" && action === "set-multicam-video-angle") {
            container = { ...container, videoAngleId: input.value };
        }
        else if (container.kind === "multicam" && action === "set-multicam-audio-angle") {
            container = { ...container, audioAngleId: input.value };
        }
        else if (container.kind === "audition" && action === "set-audition-choice") {
            container = { ...container, activeChoiceId: input.value };
        }
        else if (container.kind === "sync" && action.startsWith("set-sync-source-")) {
            const sourceId = input.getAttribute("data-source-id");
            if (!sourceId)
                return;
            container = {
                ...container,
                sources: container.sources.map((source) => source.sourceId !== sourceId ? source : {
                    ...source,
                    role: action === "set-sync-source-role" ? input.value : source.role,
                    enabled: action === "set-sync-source-enabled" && input instanceof HTMLInputElement ? input.checked : source.enabled,
                    active: action === "set-sync-source-active" && input instanceof HTMLInputElement ? input.checked : source.active,
                }),
            };
        }
        else
            return;
        await this.commitDirectProject("Update nested clip selection", replaceClipContainer(this.state.project, item.id, container));
    }
    async setClipAudioMetadata(input, action) {
        const itemId = input.getAttribute("data-item-id");
        if (!itemId)
            return;
        const seconds = input.value === "" ? null : { seconds: Number(input.value), raw: "" };
        const patch = action === "set-clip-role-name"
            ? { roleName: input.value || null }
            : action === "set-clip-audio-role"
                ? { audioRole: input.value || null }
                : action === "set-clip-audio-start"
                    ? { audioStart: seconds }
                    : { audioDuration: seconds };
        if (seconds && (!Number.isFinite(seconds.seconds) || (action === "set-clip-audio-duration" && seconds.seconds < 0))) {
            this.showToast("Audio component timing must be a finite non-negative duration.", "error");
            return;
        }
        await this.commitDirectProject("Update semantic audio metadata", replaceClipMetadata(this.state.project, itemId, patch));
    }
    async setProjectAudioLayout(audioLayout) {
        const rootBase = this.rootProject;
        this.state.isSaving = true;
        try {
            const canonical = await this.runtime.restoreProject({ ...rootBase, audioLayout }, rootBase.revision);
            this.appHistory = recordHistory(this.appHistory, rootBase);
            this.adoptCanonicalProject(canonical);
            this.showToast(`Project output set to ${audioLayout}.`, "success");
        }
        catch (error) {
            this.showToast(`Audio layout not saved: ${error instanceof Error ? error.message : String(error)}`, "error");
        }
        finally {
            this.state.isSaving = false;
            this.renderAll();
            this.syncPreview();
        }
    }
    // ---- Effects / Transitions browser + effect stack --------------------
    //
    // The two catalog browsers project fixtures.EFFECTS_CATALOG /
    // TRANSITIONS_CATALOG. Choosing an item drives the real reducer: an effect is
    // pushed onto the selected clip's stack; a transition is written onto the
    // selected storyline edit point. The rendered pixel result is a labeled mock.
    /** Open (or re-focus) one of the FCP-style catalog browsers docked at the timeline's right edge. */
    openBrowser(kind) {
        this.state.activeBrowser = this.state.activeBrowser === kind ? null : kind;
        this.state.browserCategory = null;
        this.renderTimeline();
    }
    /** Add a catalog effect to the selected clip's effect stack. */
    async applyEffectFromCatalog(capabilityId) {
        const item = selectedTimelineClip(this.state);
        const capability = this.requireCapabilities().effects.find((candidate) => candidate.id === capabilityId);
        if (!capability || capability.support === "unsupported" || !item) {
            this.showToast("Select a clip first, then choose an effect.", "info");
            return;
        }
        const requiredValues = {};
        for (const parameter of capability.parameters.filter((candidate) => candidate.required && candidate.default === undefined)) {
            if (parameter.type !== "color") {
                this.showToast(`${capability.name} requires ${parameter.name} before it can be applied.`, "error");
                return;
            }
            const value = globalThis.prompt?.(`Choose ${parameter.name} as an RGB hex color`, "#00ff00");
            if (value === null || value === undefined)
                return;
            try {
                requiredValues[parameter.key] = rgbFromHex(value);
            }
            catch (error) {
                this.showToast(error instanceof Error ? error.message : String(error), "error");
                return;
            }
        }
        const effect = this.clipEffectFromCapability(capability, requiredValues);
        await this.commitEdit(`Add effect · ${capability.name}`, {
            type: "addEffect",
            clipId: item.id,
            effect,
        });
        this.showToast(`Added “${capability.name}” to ${item.name}.`, "success");
    }
    clipEffectFromCapability(capability, requiredValues = {}) {
        if (capability.support === "unsupported") {
            throw new Error(`${capability.name} is not authorable.`);
        }
        return {
            id: randomId("fx"),
            name: capability.name,
            category: effectCatalogItem(capability).category,
            enabled: true,
            resourceId: randomId("resource"),
            resourceUid: capability.resource.uid,
            handler: capability.handler,
            support: capability.support,
            parameters: { ...defaultCapabilityParameters(capability), ...requiredValues },
            parameterNames: capabilityParameterNames(capability),
            parameterKeyframes: {},
        };
    }
    /** Drop a catalog transition on the selected storyline edit point. */
    async applyTransitionFromCatalog(capabilityId) {
        const edit = selectedStorylineEdit(this.state);
        const capability = this.requireCapabilities().transitions.find((candidate) => candidate.id === capabilityId);
        if (!capability || capability.support === "unsupported" || !edit) {
            this.showToast("Select a storyline clip that has a clip after it, then choose a transition.", "info");
            return;
        }
        const category = transitionCatalogItem(capability).category;
        await this.commitEdit(`Add transition · ${capability.name}`, {
            type: "addTransition",
            transition: {
                id: randomId("transition"),
                name: capability.name,
                category,
                leftItemId: edit.left.id,
                rightItemId: edit.right.id,
                duration: 1,
                resourceId: randomId("resource"),
                resourceUid: capability.resource.uid,
                handler: capability.handler,
                support: capability.support,
                parameters: defaultCapabilityParameters(capability),
                parameterNames: capabilityParameterNames(capability),
                parameterKeyframes: {},
            },
        });
        this.showToast(`Added “${capability.name}” between ${edit.left.name} and ${edit.right.name}.`, "success");
    }
    async toggleEffect(effectId) {
        const item = selectedTimelineClip(this.state);
        if (!item || !effectId) {
            return;
        }
        const effectStack = item.effectStack.map((entry) => entry.kind === "effect" && entry.effect.id === effectId
            ? { kind: "effect", effect: { ...entry.effect, enabled: !entry.effect.enabled } }
            : entry);
        await this.commitEffectStack("Toggle effect", item, effectStack);
    }
    async toggleEffectsSection(enabled) {
        const item = selectedTimelineClip(this.state);
        if (!item || item.effectStack.length === 0) {
            this.renderInspector();
            return;
        }
        const effectStack = item.effectStack.map((entry) => entry.kind === "effect"
            ? { kind: "effect", effect: { ...entry.effect, enabled } }
            : { kind: "masked-effect", maskedEffect: { ...entry.maskedEffect, enabled } });
        await this.commitEffectStack(enabled ? "Enable effects" : "Disable effects", item, effectStack);
    }
    async reorderEffect(effectId, direction) {
        const item = selectedTimelineClip(this.state);
        if (!item || !effectId || Number.isNaN(direction)) {
            return;
        }
        const effectStack = [...item.effectStack];
        const index = effectStack.findIndex((entry) => entry.kind === "effect" && entry.effect.id === effectId);
        if (index < 0) {
            return;
        }
        const toIndex = clamp(index + direction, 0, effectStack.length - 1);
        if (toIndex === index) {
            return;
        }
        const [entry] = effectStack.splice(index, 1);
        effectStack.splice(toIndex, 0, entry);
        await this.commitEffectStack("Reorder effect", item, effectStack);
    }
    async removeEffect(effectId) {
        const item = selectedTimelineClip(this.state);
        if (!item || !effectId) {
            return;
        }
        await this.commitEffectStack("Remove effect", item, item.effectStack.filter((entry) => entry.kind !== "effect" || entry.effect.id !== effectId));
    }
    /** Keep the compatibility plain-effect projection synchronized with the canonical ordered stack. */
    async commitEffectStack(label, item, effectStack) {
        await this.commitEdit(label, {
            type: "updateClip",
            clipId: item.id,
            patch: { effectStack },
        });
    }
    async wrapEffectInMask(effectId, kind) {
        const item = this.selectedItem();
        if (!item || !effectId)
            return;
        const index = item.effectStack.findIndex((entry) => entry.kind === "effect" && entry.effect.id === effectId);
        const entry = item.effectStack[index];
        if (index < 0 || !entry || entry.kind !== "effect")
            return;
        const next = [...item.effectStack];
        next[index] = createMaskedEffect(randomId("masked"), [defaultMask(kind, randomId("mask"))], entry.effect);
        await this.commitEffectStack("Add effect mask", item, next);
    }
    async patchMaskedGroup(groupId, label, mutate) {
        const item = this.selectedItem();
        if (!item || !groupId)
            return;
        let touched = false;
        let effectStack;
        try {
            effectStack = item.effectStack.map((entry) => {
                if (entry.kind !== "masked-effect" || entry.maskedEffect.id !== groupId)
                    return entry;
                touched = true;
                return { kind: "masked-effect", maskedEffect: mutate(structuredClone(entry.maskedEffect)) };
            });
        }
        catch (error) {
            this.showToast(error instanceof Error ? error.message : String(error), "error");
            return;
        }
        if (!touched)
            return;
        await this.commitEffectStack(label, item, effectStack);
    }
    maskSourceFromElement(element) {
        const groupId = element.getAttribute("data-group-id");
        const maskId = element.getAttribute("data-mask-id");
        return groupId && maskId ? { groupId, maskId } : null;
    }
    async patchMaskSource(element, label, mutate) {
        const identity = this.maskSourceFromElement(element);
        if (!identity)
            return;
        await this.patchMaskedGroup(identity.groupId, label, (group) => ({
            ...group,
            masks: group.masks.map((mask) => mask.id === identity.maskId ? mutate(mask) : mask),
        }));
    }
    async addMaskSource(groupId, kind) {
        if (!groupId)
            return;
        const maximum = this.requireCapabilities().mechanics.find((mechanic) => mechanic.id === "masks")?.maximumMasks ?? 32;
        await this.patchMaskedGroup(groupId, `Add ${kind} mask`, (group) => {
            if (group.masks.length >= maximum)
                throw new Error(`A masked effect supports at most ${maximum} masks.`);
            return { ...group, masks: [...group.masks, defaultMask(kind, randomId("mask"))] };
        });
    }
    async removeMaskSource(element) {
        const identity = this.maskSourceFromElement(element);
        if (!identity)
            return;
        await this.patchMaskedGroup(identity.groupId, "Remove mask", (group) => {
            if (group.masks.length <= 1)
                throw new Error("A masked effect requires at least one mask source.");
            return { ...group, masks: group.masks.filter((mask) => mask.id !== identity.maskId) };
        });
    }
    async reorderMaskSource(element, direction) {
        const identity = this.maskSourceFromElement(element);
        if (!identity)
            return;
        await this.patchMaskedGroup(identity.groupId, "Reorder mask", (group) => {
            const masks = [...group.masks];
            const index = masks.findIndex((mask) => mask.id === identity.maskId);
            if (index < 0)
                return group;
            const target = clamp(index + direction, 0, masks.length - 1);
            const [mask] = masks.splice(index, 1);
            masks.splice(target, 0, mask);
            return { ...group, masks };
        });
    }
    async removeMaskedGroup(groupId) {
        const item = this.selectedItem();
        if (!item || !groupId)
            return;
        const effectStack = item.effectStack.flatMap((entry) => entry.kind === "masked-effect" && entry.maskedEffect.id === groupId
            ? entry.maskedEffect.filters.map((effect) => ({ kind: "effect", effect }))
            : [entry]);
        await this.commitEffectStack("Remove effect mask", item, effectStack);
    }
    async reorderMaskedGroup(groupId, direction) {
        const item = this.selectedItem();
        if (!item || !groupId)
            return;
        const effectStack = [...item.effectStack];
        const index = effectStack.findIndex((entry) => entry.kind === "masked-effect" && entry.maskedEffect.id === groupId);
        if (index < 0)
            return;
        const target = clamp(index + direction, 0, effectStack.length - 1);
        const [entry] = effectStack.splice(index, 1);
        effectStack.splice(target, 0, entry);
        await this.commitEffectStack("Reorder masked effect", item, effectStack);
    }
    async setMaskOutsideEffect(input) {
        const groupId = input.getAttribute("data-group-id");
        const capability = this.requireCapabilities().effects.find((effect) => effect.id === input.value);
        if (!groupId || !capability || capability.support === "unsupported")
            return;
        const outside = this.clipEffectFromCapability(capability);
        await this.patchMaskedGroup(groupId, "Add outside mask effect", (group) => ({
            ...group, filters: [group.filters[0], outside],
        }));
    }
    async setMaskParameter(input) {
        const identity = this.maskSourceFromElement(input);
        const key = input.getAttribute("data-mask-key");
        if (!identity || !key)
            return;
        await this.patchMaskSource(input, "Edit mask", (mask) => {
            const current = maskUiValues(mask)[key];
            const component = input.getAttribute("data-component");
            const pointIndex = input.getAttribute("data-point-index");
            let value;
            if (pointIndex !== null && component && Array.isArray(current)) {
                const points = current.map((point) => ({ ...point }));
                const index = Number(pointIndex);
                points[index] = { ...points[index], [component]: Number(input.value) };
                value = points;
            }
            else if (component && current && typeof current === "object" && !Array.isArray(current)) {
                value = { ...current, [component]: Number(input.value) };
            }
            else if (input instanceof HTMLInputElement && input.type === "color") {
                const alpha = typeof current === "object" && current !== null && "alpha" in current
                    ? Number(current.alpha)
                    : 1;
                value = colorFromHex(input.value, alpha);
            }
            else {
                value = Number(input.value);
            }
            let updated = updateMaskValue(mask, key, value);
            const capability = this.requireCapabilities().mechanics.find((mechanic) => mechanic.id === "masks")?.sourceKinds?.find((source) => source.id === mask.kind)?.parameters.find((parameter) => parameter.key === key);
            if (capability?.animatable) {
                const storedKey = maskKeyframeKey(updated, key);
                const local = this.state.currentTime - (this.selectedItem()?.timelineStart ?? 0);
                const threshold = 1 / this.state.project.fps;
                const frames = (updated.parameterKeyframes[storedKey] ?? []).map((frame) => Math.abs(frame.time.seconds - local) < threshold
                    ? { ...frame, value: updated.parameters[storedKey] ?? value }
                    : frame);
                updated = withMaskKeyframes(updated, storedKey, frames);
            }
            return updated;
        });
    }
    async toggleMaskKeyframe(element) {
        const identity = this.maskSourceFromElement(element);
        const capabilityKey = element.getAttribute("data-mask-key");
        const item = this.selectedItem();
        if (!identity || !capabilityKey || !item)
            return;
        const stackEntry = item.effectStack.find((entry) => entry.kind === "masked-effect" && entry.maskedEffect.id === identity.groupId);
        const maskKind = stackEntry?.kind === "masked-effect"
            ? stackEntry.maskedEffect.masks.find((mask) => mask.id === identity.maskId)?.kind
            : undefined;
        const sourceCapability = this.requireCapabilities().mechanics
            .find((mechanic) => mechanic.id === "masks")?.sourceKinds
            ?.find((source) => source.id === maskKind);
        if (!sourceCapability?.parameters.find((parameter) => parameter.key === capabilityKey)?.animatable) {
            this.showToast("This mask parameter is static in Tensor.", "error");
            return;
        }
        await this.patchMaskSource(element, "Toggle mask keyframe", (mask) => {
            const key = maskKeyframeKey(mask, capabilityKey);
            const local = clamp(this.state.currentTime - item.timelineStart, 0, item.duration);
            const threshold = 1 / this.state.project.fps;
            const frames = [...(mask.parameterKeyframes[key] ?? [])];
            const index = frames.findIndex((frame) => Math.abs(frame.time.seconds - local) < threshold);
            if (index >= 0)
                frames.splice(index, 1);
            else {
                const value = maskUiValues(mask)[capabilityKey];
                if (Array.isArray(value) && value.some((entry) => typeof entry === "object")) {
                    throw new Error(`${mask.name} ${capabilityKey} is not animatable.`);
                }
                frames.push({
                    time: { seconds: local, raw: "" },
                    value: structuredClone(value),
                    interpolation: "linear",
                });
            }
            frames.sort((left, right) => left.time.seconds - right.time.seconds);
            return withMaskKeyframes(mask, key, frames);
        });
    }
    seekMaskKeyframe(element, direction) {
        const identity = this.maskSourceFromElement(element);
        const capabilityKey = element.getAttribute("data-mask-key");
        const item = this.selectedItem();
        if (!identity || !capabilityKey || !item)
            return;
        const group = item.effectStack.find((entry) => entry.kind === "masked-effect" && entry.maskedEffect.id === identity.groupId);
        if (!group || group.kind !== "masked-effect")
            return;
        const mask = group.maskedEffect.masks.find((candidate) => candidate.id === identity.maskId);
        if (!mask)
            return;
        const frames = [...(mask.parameterKeyframes[maskKeyframeKey(mask, capabilityKey)] ?? [])]
            .sort((left, right) => left.time.seconds - right.time.seconds);
        const local = this.state.currentTime - item.timelineStart;
        const candidate = direction < 0
            ? [...frames].reverse().find((frame) => frame.time.seconds < local - 1 / this.state.project.fps) ?? frames.at(-1)
            : frames.find((frame) => frame.time.seconds > local + 1 / this.state.project.fps) ?? frames[0];
        if (candidate)
            this.seek(item.timelineStart + candidate.time.seconds);
    }
    async clearMaskKeyframes(element) {
        const capabilityKey = element.getAttribute("data-mask-key");
        if (!capabilityKey)
            return;
        await this.patchMaskSource(element, "Delete mask keyframes", (mask) => withMaskKeyframes(mask, maskKeyframeKey(mask, capabilityKey), []));
    }
    async adjustDrawMaskPoint(element, add) {
        const rawIndex = element.getAttribute("data-point-index");
        await this.patchMaskSource(element, add ? "Add Draw Mask point" : "Remove Draw Mask point", (mask) => {
            if (mask.kind !== "draw")
                throw new Error("Only Draw Masks contain editable polygon points.");
            const values = maskUiValues(mask);
            const points = structuredClone(values.points);
            if (add) {
                if (points.length >= 64)
                    throw new Error("Draw Mask supports at most 64 points.");
                const left = points.at(-1);
                const right = points[0];
                points.push({ x: (left.x + right.x) / 2, y: (left.y + right.y) / 2 });
            }
            else {
                if (points.length <= 3)
                    throw new Error("Draw Mask requires at least 3 points.");
                const index = Number(rawIndex);
                if (!Number.isInteger(index) || index < 0 || index >= points.length)
                    throw new Error("Draw Mask point does not exist.");
                points.splice(index, 1);
            }
            return updateMaskValue(mask, "points", points);
        });
    }
    async removeTransition(transitionId) {
        if (!transitionId)
            return;
        await this.commitEdit("Remove transition", { type: "removeTransition", transitionId });
        this.state.selectedTransitionStart = null;
    }
    async setTransitionDuration(transitionId, duration) {
        if (!transitionId || !Number.isFinite(duration))
            return;
        await this.commitEdit("Adjust transition duration", {
            type: "updateTransition",
            transitionId,
            patch: { duration: Math.max(1 / this.state.project.fps, duration) },
        });
    }
    async setCapabilityParameter(input) {
        const owner = input.getAttribute("data-owner");
        const ownerId = input.getAttribute("data-owner-id");
        const key = input.getAttribute("data-parameter-key");
        if (!ownerId || !key || !["effect", "transition", "masked-filter"].includes(owner ?? ""))
            return;
        if (owner === "effect") {
            const item = this.selectedItem();
            const effect = item?.effects.find((candidate) => candidate.id === ownerId);
            if (!item || !effect)
                return;
            const value = capabilityInputValue(input, effect.parameters[key]);
            await this.commitEdit(`Adjust ${effect.parameterNames[key] ?? effect.name}`, {
                type: "updateEffect",
                clipId: item.id,
                effectId: effect.id,
                patch: { parameters: { ...effect.parameters, [key]: value } },
            });
            return;
        }
        if (owner === "masked-filter") {
            const [groupId, rawIndex] = ownerId.split(":");
            const filterIndex = Number(rawIndex);
            if (!groupId || !Number.isInteger(filterIndex))
                return;
            await this.patchMaskedGroup(groupId, "Adjust masked effect", (group) => {
                const effect = group.filters[filterIndex];
                if (!effect)
                    return group;
                const value = capabilityInputValue(input, effect.parameters[key]);
                const filters = [...group.filters];
                filters[filterIndex] = { ...effect, parameters: { ...effect.parameters, [key]: value } };
                return { ...group, filters };
            });
            return;
        }
        const transition = this.state.project.transitions.find((candidate) => candidate.id === ownerId);
        if (!transition)
            return;
        const value = capabilityInputValue(input, transition.parameters[key]);
        await this.commitEdit(`Adjust ${transition.parameterNames[key] ?? transition.name}`, {
            type: "updateTransition",
            transitionId: transition.id,
            patch: { parameters: { ...transition.parameters, [key]: value } },
        });
    }
    async setTitleText(text) {
        const item = this.selectedItem();
        if (!item || (item.kind !== "title" && item.kind !== "caption"))
            return;
        await this.commitEdit("Edit title text", {
            type: "updateClip",
            clipId: item.id,
            patch: { text },
        });
    }
    async setTextStyle(input) {
        const item = this.selectedItem();
        const key = input.getAttribute("data-style-key");
        if (!item?.textStyle || !key)
            return;
        const value = key === "fontSize"
            ? Math.max(1, Math.min(500, Number(input.value)))
            : key === "fontColor"
                ? colorFromHex(input.value, item.textStyle.fontColor.alpha)
                : input.value;
        await this.commitEdit("Edit text style", {
            type: "updateClip",
            clipId: item.id,
            patch: { textStyle: { ...item.textStyle, [key]: value } },
        });
    }
    async setCaptionField(input) {
        const item = this.selectedItem();
        const key = input.getAttribute("data-caption-key");
        if (!item?.caption || item.kind !== "caption" || !key)
            return;
        await this.commitEdit("Edit caption", {
            type: "updateClip",
            clipId: item.id,
            patch: { caption: { ...item.caption, [key]: input.value } },
        });
    }
    async setGeneratorColor(value) {
        const item = this.selectedItem();
        if (!item || item.kind !== "generator")
            return;
        await this.commitEdit("Edit Custom Solid color", {
            type: "updateClip",
            clipId: item.id,
            patch: { generatorColor: colorFromHex(value, item.generatorColor?.alpha ?? 1) },
        });
    }
    async insertConstructedClip(kind) {
        const previousIds = new Set([...this.state.project.spine, ...this.state.project.connected].map((item) => item.id));
        const id = randomId(kind);
        const duration = kind === "caption" ? 3 : 5;
        const clip = kind === "title"
            ? this.runtime.createBasicTitleClip({ id, text: "Basic Title", duration })
            : kind === "caption"
                ? this.runtime.createCaptionClip({ id, text: "Caption", duration })
                : this.runtime.createCustomSolidClip({
                    id,
                    duration,
                    color: { red: 0.18, green: 0.18, blue: 0.18, alpha: 1 },
                });
        if (kind === "generator") {
            this.state.selectedItemId = id;
            this.state.selectedItemIds = [id];
            await this.commitEdit("Insert Custom Solid", {
                type: "insert",
                clip,
                index: insertionIndexAtTime(this.state.project, this.state.currentTime),
            });
            this.selectInsertedClip(previousIds, kind);
            return;
        }
        const anchor = clipAtTime(this.state.project, this.state.currentTime);
        if (!anchor) {
            this.showToast(`Cannot connect ${kind} without a primary-storyline clip at the playhead.`, "error");
            return;
        }
        this.state.selectedItemId = id;
        this.state.selectedItemIds = [id];
        await this.commitEdit(kind === "title" ? "Connect Basic Title" : "Connect Caption", {
            type: "connect",
            clip: {
                ...clip,
                role: "title",
                timelineStart: this.state.currentTime,
                anchorId: anchor.id,
                anchorOffset: Math.max(0, this.state.currentTime - anchor.timelineStart),
                lane: 1,
            },
        });
        this.selectInsertedClip(previousIds, kind);
    }
    /**
     * Select the generated item that the localhost round trip accepted.
     *
     * Main callers: insertConstructedClip after inserting a title, caption, or
     * Custom Solid.
     *
     * Why this exists: FCPXML assigns a canonical structural ID when a newly
     * authored element is first serialized. The temporary client ID can change,
     * so selection must follow the new item instead of the discarded ID.
     */
    selectInsertedClip(previousIds, kind) {
        const inserted = [...this.state.project.spine, ...this.state.project.connected]
            .find((item) => item.kind === kind && !previousIds.has(item.id));
        if (!inserted) {
            return;
        }
        this.state.selectedItemId = inserted.id;
        this.state.selectedItemIds = [inserted.id];
        this.state.selectedTransitionStart = null;
        this.renderAll();
        this.syncPreview();
    }
    async setRetimeRate(rate, reverse = false) {
        const item = this.selectedItem();
        if (!item || !supportsPortableRetime(item) || !Number.isFinite(rate) || rate <= 0)
            return;
        const currentPoints = item.timeMap?.points ?? [];
        const sourceSpan = currentPoints.length >= 2
            ? Math.abs(currentPoints[currentPoints.length - 1].value.seconds - currentPoints[0].value.seconds)
            : item.duration;
        const sourceStart = currentPoints.length >= 2
            ? Math.min(currentPoints[0].value.seconds, currentPoints[currentPoints.length - 1].value.seconds)
            : item.sourceStart;
        const duration = Math.max(1 / this.state.project.fps, sourceSpan / rate);
        const timeMap = {
            frameSampling: "floor",
            preservesPitch: item.timeMap?.preservesPitch ?? true,
            points: [
                {
                    time: rationalTime(item.sourceStart, this.state.project.fps),
                    value: rationalTime(reverse ? sourceStart + sourceSpan : sourceStart, this.state.project.fps),
                    interpolation: "linear",
                },
                {
                    time: rationalTime(item.sourceStart + duration, this.state.project.fps),
                    value: rationalTime(reverse ? sourceStart : sourceStart + sourceSpan, this.state.project.fps),
                    interpolation: "linear",
                },
            ],
        };
        await this.commitEdit(`${reverse ? "Reverse" : "Retime"} clip to ${Math.round(rate * 100)}%`, {
            type: "updateClip",
            clipId: item.id,
            patch: { duration, timeMap },
        });
    }
    async toggleRetimeSection(enabled) {
        const item = this.selectedItem();
        if (!item || !supportsPortableRetime(item))
            return;
        if (enabled) {
            await this.toggleRetimePitch(item.timeMap?.preservesPitch ?? true);
            return;
        }
        const points = item.timeMap?.points ?? [];
        const sourceSpan = points.length >= 2
            ? Math.abs(points[points.length - 1].value.seconds - points[0].value.seconds)
            : item.duration;
        await this.commitEdit("Disable Re-time", {
            type: "updateClip",
            clipId: item.id,
            patch: { duration: Math.max(1 / this.state.project.fps, sourceSpan), timeMap: null },
        });
    }
    async toggleRetimePitch(preservesPitch) {
        const item = this.selectedItem();
        if (!item || !supportsPortableRetime(item))
            return;
        const timeMap = item.timeMap ?? {
            frameSampling: "floor",
            preservesPitch,
            points: [
                { time: rationalTime(item.sourceStart, this.state.project.fps), value: rationalTime(item.sourceStart, this.state.project.fps), interpolation: "linear" },
                { time: rationalTime(item.sourceStart + item.duration, this.state.project.fps), value: rationalTime(item.sourceStart + item.duration, this.state.project.fps), interpolation: "linear" },
            ],
        };
        await this.commitEdit("Toggle retime pitch preservation", {
            type: "updateClip",
            clipId: item.id,
            patch: { timeMap: { ...timeMap, preservesPitch } },
        });
    }
    async splitRetimeAtPlayhead() {
        const item = this.selectedItem();
        if (!item || !supportsPortableRetime(item))
            return;
        const edit = splitRetimeAt(item, this.state.currentTime - item.timelineStart);
        await this.commitEdit("Split speed segment", {
            type: "updateClip", clipId: item.id, patch: edit,
        });
    }
    async addRetimeHoldAtPlayhead(duration) {
        const item = this.selectedItem();
        if (!item || !supportsPortableRetime(item))
            return;
        try {
            const edit = addRetimeHold(item, this.state.currentTime - item.timelineStart, duration);
            await this.commitEdit("Add retime hold", {
                type: "updateClip", clipId: item.id, patch: edit,
            });
        }
        catch (error) {
            this.showToast(error instanceof Error ? error.message : String(error), "error");
        }
    }
    async freezeRetimeAtPlayhead() {
        const item = this.selectedItem();
        if (!item || !supportsPortableRetime(item))
            return;
        const edit = freezeRetime(item, this.state.currentTime - item.timelineStart);
        await this.commitEdit("Freeze clip", {
            type: "updateClip", clipId: item.id, patch: edit,
        });
    }
    async setRetimeSegmentRate(segmentIndex, rate) {
        const item = this.selectedItem();
        if (!item || !supportsPortableRetime(item))
            return;
        try {
            const edit = setRetimeSegmentRate(item, segmentIndex, rate);
            await this.commitEdit(`Set speed segment ${segmentIndex + 1}`, {
                type: "updateClip", clipId: item.id, patch: edit,
            });
        }
        catch (error) {
            this.showToast(error instanceof Error ? error.message : String(error), "error");
        }
    }
    async setRetimeSegmentDuration(segmentIndex, duration) {
        const item = this.selectedItem();
        if (!item || !supportsPortableRetime(item))
            return;
        try {
            const edit = setRetimeSegmentDuration(item, segmentIndex, duration);
            await this.commitEdit(`Set speed segment ${segmentIndex + 1} duration`, {
                type: "updateClip", clipId: item.id, patch: edit,
            });
        }
        catch (error) {
            this.showToast(error instanceof Error ? error.message : String(error), "error");
        }
    }
    // ---- Markers ---------------------------------------------------------
    //
    // Markers attach to a clip by a local offset, so they ripple with the clip.
    // Add lands one at the playhead on the selected clip and opens its editor.
    async addMarkerAtPlayhead() {
        const item = this.selectedItem() ?? clipAtTime(this.state.project, this.state.currentTime);
        if (!item) {
            this.showToast("Select a clip to add a marker.", "info");
            return;
        }
        const offset = clamp(this.state.currentTime - item.timelineStart, 0, item.duration);
        const previousMarkerIds = new Set([...this.state.project.spine, ...this.state.project.connected]
            .flatMap((clip) => clip.markers.map((marker) => marker.id)));
        const markerId = randomId("marker");
        await this.commitEdit("Add marker", {
            type: "addMarker",
            clipId: item.id,
            marker: { id: markerId, offset, name: "Marker", type: "standard", completed: false },
        });
        const inserted = [...this.state.project.spine, ...this.state.project.connected]
            .flatMap((clip) => clip.markers)
            .find((marker) => !previousMarkerIds.has(marker.id));
        this.state.markerEditorId = inserted?.id ?? null;
        this.renderTimeline();
    }
    openMarkerEditor(markerId) {
        this.state.markerEditorId = markerId;
        this.renderTimeline();
    }
    async updateMarkerFromElement(element, patch) {
        const clipId = element.getAttribute("data-item-id");
        const markerId = element.getAttribute("data-marker-id");
        if (!clipId || !markerId) {
            return;
        }
        this.state.markerEditorId = markerId;
        await this.commitEdit("Edit marker", { type: "updateMarker", clipId, markerId, patch });
        this.renderTimeline();
    }
    toggleMarkerDone(element) {
        const clipId = element.getAttribute("data-item-id");
        const markerId = element.getAttribute("data-marker-id");
        if (!clipId || !markerId) {
            return;
        }
        const marker = this.itemById(clipId)?.markers.find((candidate) => candidate.id === markerId);
        if (!marker) {
            return;
        }
        void this.updateMarkerFromElement(element, { completed: !marker.completed });
    }
    async deleteMarkerFromElement(element) {
        const clipId = element.getAttribute("data-item-id");
        const markerId = element.getAttribute("data-marker-id");
        if (!clipId || !markerId) {
            return;
        }
        this.state.markerEditorId = null;
        await this.commitEdit("Delete marker", { type: "deleteMarker", clipId, markerId });
    }
    undo() {
        const current = this.rootProject;
        const selectionSeq = this.projectSelectionSeq;
        const selectedProjectId = this.state.selectedProjectId;
        const result = undoHistory(this.appHistory, current);
        if (!result) {
            return;
        }
        void this.runtime
            .restoreProject(result.project, current.revision)
            .then((restored) => {
            if (selectionSeq !== this.projectSelectionSeq || selectedProjectId !== this.state.selectedProjectId) {
                return;
            }
            this.appHistory = result.history;
            this.adoptCanonicalProject(restored);
            this.renderAll();
            this.syncPreview();
        })
            .catch((error) => {
            this.showToast(`Undo rejected: ${error instanceof Error ? error.message : String(error)}`, "error");
        });
    }
    redo() {
        const current = this.rootProject;
        const selectionSeq = this.projectSelectionSeq;
        const selectedProjectId = this.state.selectedProjectId;
        const result = redoHistory(this.appHistory, current);
        if (!result) {
            return;
        }
        void this.runtime
            .restoreProject(result.project, current.revision)
            .then((restored) => {
            if (selectionSeq !== this.projectSelectionSeq || selectedProjectId !== this.state.selectedProjectId) {
                return;
            }
            this.appHistory = result.history;
            this.adoptCanonicalProject(restored);
            this.renderAll();
            this.syncPreview();
        })
            .catch((error) => {
            this.showToast(`Redo rejected: ${error instanceof Error ? error.message : String(error)}`, "error");
        });
    }
    async resetSection(section) {
        const item = this.selectedItem();
        if (!item || !section) {
            return;
        }
        if (section === "transform") {
            await this.commitEdit(`Reset ${section}`, {
                type: "updateClip",
                clipId: item.id,
                patch: { transform: defaultTransform() },
            });
            return;
        }
        if (section === "crop") {
            await this.commitEdit(`Reset ${section}`, {
                type: "updateClip",
                clipId: item.id,
                patch: { video: { ...item.video, crop: defaultVideo().crop } },
            });
            return;
        }
        if (section === "distort") {
            await this.commitEdit(`Reset ${section}`, {
                type: "updateClip",
                clipId: item.id,
                patch: { video: { ...item.video, distort: defaultVideo().distort } },
            });
            return;
        }
        if (section === "compositing") {
            await this.commitEdit(`Reset ${section}`, {
                type: "updateClip",
                clipId: item.id,
                patch: {
                    video: { ...item.video, blendMode: "normal" },
                    transform: { ...item.transform, opacity: 1 },
                },
            });
            return;
        }
        if (section === "effects-stack") {
            await this.commitEdit(`Reset ${section}`, {
                type: "updateClip",
                clipId: item.id,
                patch: { effectStack: [] },
            });
            return;
        }
        if (section === "spatial") {
            await this.commitEdit(`Reset ${section}`, {
                type: "updateClip",
                clipId: item.id,
                patch: { video: { ...item.video, spatialConform: "fit" } },
            });
            return;
        }
        if (section === "retime") {
            await this.toggleRetimeSection(false);
            return;
        }
        if (section === "color") {
            await this.commitEdit(`Reset ${section}`, {
                type: "updateClip",
                clipId: item.id,
                patch: { video: { ...item.video, color: defaultVideo().color } },
            });
            return;
        }
        if (section === "audio-volume") {
            await this.commitEdit(`Reset ${section}`, {
                type: "updateClip",
                clipId: item.id,
                patch: { audio: resetAudioVolume(item.audio) },
            });
        }
    }
    async toggleKeyframe(path) {
        const item = this.selectedItem();
        if (!item || !path) {
            return;
        }
        await this.commitEdit(`Toggle ${path} keyframe`, {
            type: "toggleKeyframe",
            clipId: item.id,
            path,
            time: this.state.currentTime,
        });
    }
    async toggleBooleanPath(path) {
        const item = this.selectedItem();
        if (!item) {
            return;
        }
        await this.commitParameter(path, !Boolean(getPath(item, path)), `Toggle ${path}`);
    }
    async commitParameter(path, value, label) {
        const item = this.selectedItem();
        if (!item || !path) {
            return;
        }
        await this.commitEdit(label, {
            type: "updateClipPath",
            clipId: item.id,
            path,
            value,
        });
    }
    async swapKenBurns() {
        const item = this.selectedItem();
        if (!item) {
            return;
        }
        await this.commitEdit("Swap Ken Burns start and end", {
            type: "updateClip",
            clipId: item.id,
            patch: {
                video: {
                    ...item.video,
                    crop: {
                        ...item.video.crop,
                        kenStart: clone(item.video.crop.kenEnd),
                        kenEnd: clone(item.video.crop.kenStart),
                    },
                },
            },
        });
    }
    setViewerZoom(value) {
        if (!value) {
            return;
        }
        if (value === "fit") {
            this.state.viewerZoom = 0;
        }
        else {
            this.state.viewerZoom = clamp(Number(value), 25, 200);
        }
        this.state.activePopover = null;
        this.renderViewer();
    }
    setViewerOption(option, value) {
        if (!option || !value) {
            return;
        }
        if (option === "quality" &&
            (value === "better-quality" || value === "better-performance" || value === "best-performance")) {
            this.state.viewerView = { ...this.state.viewerView, quality: value };
            this.preview?.setQuality(value);
            // A quality change recreates the backend producer and its decoders, so
            // the stream cold-starts again. Re-arm the warm-up spinner if we are
            // mid-playback so the re-warm reads as buffering, not lag.
            if (this.state.playing) {
                this.watchPreviewWarmup();
            }
        }
        if (option === "background" && (value === "black" || value === "checker" || value === "white")) {
            this.state.viewerView = { ...this.state.viewerView, background: value };
        }
        this.state.activePopover = null;
        this.updateViewerGuides();
        this.renderViewer();
        this.previewRealtime({ type: "set-viewer-option", option, value });
    }
    selectTimelineItem(itemId) {
        if (!itemId || !this.itemById(itemId)) {
            return;
        }
        this.state.selectedItemId = itemId;
        this.state.selectedItemIds = [itemId];
        this.state.selectedTransitionStart = null;
        this.state.currentTime = this.itemById(itemId)?.timelineStart ?? this.state.currentTime;
        this.renderViewer();
        this.renderInspector();
        this.renderTimeline();
    }
    selectColorZone(zone) {
        if (!zone || !["shadows", "midtones", "highlights"].includes(zone)) {
            return;
        }
        this.state.activeColorZone = zone;
        this.renderInspector();
    }
    renderMockInventory() {
        const container = document.getElementById("mock-inventory-content");
        if (!container) {
            return;
        }
        const groups = new Map();
        for (const entry of activeMockCapabilities(this.state.connectionMode)) {
            const category = entry.capability.category;
            const rows = groups.get(category) ?? [];
            rows.push(entry);
            groups.set(category, rows);
        }
        container.innerHTML = [...groups.entries()]
            .map(([category, rows]) => `
      <section class="mock-category"><h3>${category}</h3>${rows
            .map((entry) => `
        <div class="mock-row"><strong>${entry.capability.label} (is still a mock)</strong><p>${entry.capability.detail}</p></div>
      `)
            .join("")}</section>
    `)
            .join("");
    }
    openMockInventory() {
        this.renderMockInventory();
        const modal = this.el("mock-inventory");
        modal.hidden = false;
    }
    closeMockInventory() {
        const modal = document.getElementById("mock-inventory");
        if (modal) {
            modal.hidden = true;
        }
    }
    showMockNotice(capabilityId) {
        const capability = MOCK_CAPABILITIES[capabilityId];
        if (!capability) {
            return;
        }
        const notice = this.el("mock-notice");
        const title = this.el("mock-notice-title");
        const detail = this.el("mock-notice-detail");
        title.textContent = mockNoticeTitle(capabilityId);
        detail.textContent = capability.detail;
        notice.classList.add("visible");
        if (this.mockNoticeTimer !== null) {
            clearTimeout(this.mockNoticeTimer);
        }
        this.mockNoticeTimer = setTimeout(() => notice.classList.remove("visible"), 4200);
    }
    mockImportMedia() {
        if (this.state.connectionMode === "localhost") {
            const picker = document.createElement("input");
            picker.type = "file";
            picker.multiple = true;
            picker.accept = "video/*,audio/*,image/*";
            picker.className = "media-import-picker";
            picker.hidden = true;
            document.body.append(picker);
            picker.addEventListener("cancel", () => picker.remove(), { once: true });
            picker.addEventListener("change", () => {
                const files = [...(picker.files ?? [])];
                picker.remove();
                void (async () => {
                    try {
                        let assets = this.assets;
                        let imported = 0;
                        const errors = [];
                        for (const file of files) {
                            try {
                                assets = await this.runtime.importMedia(file);
                                imported += 1;
                                this.assets = assets.map((asset) => ({ ...asset, tags: [...asset.tags] }));
                                this.state.selectedAssetId = this.assets.at(-1)?.id ?? this.state.selectedAssetId;
                                this.refreshBrowser();
                            }
                            catch (error) {
                                errors.push(`${file.name}: ${error instanceof Error ? error.message : String(error)}`);
                            }
                        }
                        this.paintMediaInventoryWarning();
                        if (imported > 0 && errors.length === 0) {
                            this.showToast(`${imported} media file${imported === 1 ? "" : "s"} imported.`, "success");
                        }
                        else if (imported > 0) {
                            this.showToast(`Imported ${imported} of ${files.length}. Failed: ${errors.join("; ")}`, "error");
                        }
                        else if (errors.length > 0) {
                            this.showToast(`Import failed: ${errors.join("; ")}`, "error");
                        }
                    }
                    catch (error) {
                        this.showToast(`Import failed: ${error instanceof Error ? error.message : String(error)}`, "error");
                    }
                })();
            }, { once: true });
            picker.click();
            return;
        }
        const nextIndex = this.assets.length + 1;
        const asset = {
            id: `asset_import_${nextIndex}`,
            name: `Imported Camera Clip ${String(nextIndex).padStart(2, "0")}`,
            kind: "video",
            duration: 42.4,
            colors: { a: "#324e6a", b: "#80b9e9" },
            glyph: "◉",
            tags: ["imported", "camera", "fixture"],
            createdAt: "Just now",
            favorite: false,
        };
        this.assets.push(asset);
        this.state.selectedAssetId = asset.id;
        this.refreshBrowser();
        this.showToast(`${asset.name} added to the Event.`, "success");
    }
    /**
     * Create a new FCPXML Project in the selected Event and open it.
     *
     * Main callers: Cmd-N, library context menu "New Project".
     */
    async createNewProject() {
        this.state.activePopover = null;
        const eventId = this.state.selectedEventId;
        if (!eventId) {
            this.showToast("Select an Event before creating a Project.", "info");
            return;
        }
        try {
            const project = await this.runtime.createProject(eventId);
            await this.selectProject(project.id, `${project.name} created`);
        }
        catch (error) {
            this.showToast(`Could not create Project: ${error instanceof Error ? error.message : String(error)}`, "error");
        }
    }
    async createNewEvent() {
        if (!this.state.library) {
            return;
        }
        this.state.activePopover = null;
        try {
            const event = await this.runtime.createEvent(this.state.library.id);
            const snapshot = this.runtime.snapshot();
            this.libraries = snapshot.libraries.map((library) => clone(library));
            this.state.library = this.libraries.find((library) => library.id === event.libraryId) ?? null;
            this.state.selectedEventId = event.id;
            if (this.runtime.mediaForEvent) {
                this.assets = this.runtime.mediaForEvent(event.id)
                    .map((asset) => ({ ...asset, tags: [...asset.tags] }));
            }
            else {
                this.assets = this.assets.map((asset) => ({ ...asset, favorite: false }));
            }
            this.state.expandedEventIds = [...new Set([...this.state.expandedEventIds, event.id])];
            this.renderAll();
            this.showToast(`${event.name} created.`, "success");
        }
        catch (error) {
            this.showToast(`Could not create Event: ${error instanceof Error ? error.message : String(error)}`, "error");
        }
    }
    editTimecode() {
        const value = globalThis.prompt?.("Go to project timecode", formatTimecode(this.state.currentTime, this.state.project.fps));
        if (!value) {
            return;
        }
        const digits = value.replace(/\D/g, "").padStart(8, "0").slice(-8);
        const hours = Number(digits.slice(0, 2));
        const minutes = Number(digits.slice(2, 4));
        const seconds = Number(digits.slice(4, 6));
        const frames = Number(digits.slice(6, 8));
        this.seek(hours * 3600 + minutes * 60 + seconds + frames / this.state.project.fps);
    }
    togglePanel(panel) {
        if (!panel || !["library", "browser", "timeline", "inspector"].includes(panel)) {
            return;
        }
        const key = panel;
        this.state.panels = { ...this.state.panels, [key]: !this.state.panels[key] };
        this.applyLayout();
        this.renderTopbar();
        if (key === "timeline" && this.state.panels.timeline) {
            setTimeout(() => this.fitTimeline(), 0);
        }
    }
    togglePopover(popover) {
        this.state.activePopover = this.state.activePopover === popover ? null : popover;
        this.renderTimeline();
    }
    toggleViewerToolPopover() {
        const pop = this.el("viewer-tools-popover");
        pop.hidden = !pop.hidden;
    }
    async selectProject(projectId, openedMessage = "Project opened") {
        if (!projectId || (projectId === this.state.selectedProjectId && this.runtime.mode === "mock")) {
            return;
        }
        // A greyed (unopenable) row keeps its click action so the user gets the
        // compile error instead of a silent no-op or a backend 404. The refusal
        // is left to `runtime.selectProject`, which first refreshes the library
        // (and its Project catalog) at the boundary and THEN checks openability:
        // consulting the sidebar's own `openError` here would answer from a
        // catalog that can be stale -- a Project fixed on disk since the last
        // boundary (a Final Cut re-export) would stay refused until some other
        // action refreshed it. The failure surfaces as the "Project load failed"
        // toast below.
        const seq = ++this.projectSelectionSeq;
        try {
            this.preview?.pause();
            cancelAnimationFrame(this.playbackFrame);
            this.clearPreviewWarmup();
            this.state.playing = false;
            const project = await this.runtime.selectProject(projectId);
            // A newer selection started while this one was awaiting: drop this stale
            // result so the UI (and the runtime's active Project, guarded in parallel
            // inside runtime.selectProject) settle on the most recent click only.
            if (seq !== this.projectSelectionSeq) {
                return;
            }
            this.activeScopeId = null;
            this.scopePath = [];
            this.scopeNavigation = [{ scopeId: null, path: [] }];
            this.scopeNavigationIndex = 0;
            this.adoptCanonicalProject(project);
            this.applyProjectEditability();
            this.adoptRuntimeCatalog();
            this.state.selectedProjectId = project.id;
            this.state.selectedEventId = project.eventId;
            if (!this.state.expandedEventIds.includes(project.eventId)) {
                this.state.expandedEventIds = [...this.state.expandedEventIds, project.eventId];
            }
            this.state.selectedItemId = this.state.project.spine[0]?.id ?? null;
            this.state.selectedItemIds = this.state.selectedItemId ? [this.state.selectedItemId] : [];
            this.state.selectedTransitionStart = null;
            this.state.currentTime = 0;
            this.appHistory = emptyHistory();
            this.transactionBase = null;
            this.parameterBase = null;
            this.renderAll();
            this.syncPreview();
            this.showToast(openedMessage, "success");
        }
        catch (error) {
            this.showToast(`Project load failed: ${error instanceof Error ? error.message : String(error)}`, "error");
        }
    }
    refreshBrowser() {
        this.el("media-content").innerHTML = mediaTemplate(this.assets, {
            query: this.state.mediaQuery,
            activeTab: this.state.mediaTab,
            view: this.state.browserView,
            selectedAssetId: this.state.selectedAssetId,
            selectedProjectId: this.state.selectedProjectId,
            eventProjects: this.selectedEventProjects(),
            eventName: this.selectedEventName(),
            sort: this.state.browserSort,
            scope: this.state.browserScope,
            activePopover: this.state.activePopover,
        });
        this.mediaVisuals.decorate(this.el("media-content"));
    }
    async refreshMediaInventory() {
        if (this.runtime.mode === "mock") {
            this.refreshBrowser();
            this.showToast("Fixture media refreshed.", "success");
            return;
        }
        try {
            const refreshed = await this.runtime.refreshMedia();
            const assets = this.runtime.mediaForEvent?.(this.state.selectedEventId) ?? refreshed;
            this.assets = assets.map((asset) => ({ ...asset, tags: [...asset.tags] }));
            this.refreshBrowser();
            this.warnMediaInventoryFailures({ refreshed: true });
        }
        catch (error) {
            this.showToast(`Media refresh failed: ${error instanceof Error ? error.message : String(error)}`, "error");
        }
    }
    refreshBrowserResults() {
        const container = this.el("asset-grid");
        container.className = `asset-browser ${this.state.browserView}`;
        container.innerHTML = mediaGridTemplate(this.assets, {
            query: this.state.mediaQuery,
            activeTab: this.state.mediaTab,
            view: this.state.browserView,
            selectedAssetId: this.state.selectedAssetId,
            selectedProjectId: this.state.selectedProjectId,
            eventProjects: this.selectedEventProjects(),
            sort: this.state.browserSort,
            scope: this.state.browserScope,
        });
        this.mediaVisuals.decorate(container);
        const status = document.getElementById("browser-status");
        if (status) {
            const visibleClips = container.querySelectorAll(".asset-card:not(.project-browser-card)").length;
            const visibleProjects = container.querySelectorAll(".project-browser-card").length;
            const totalProjects = this.state.mediaTab === "media" && this.state.browserScope === "all"
                ? this.selectedEventProjects().length
                : 0;
            status.textContent = `${visibleClips + visibleProjects} of ${this.assets.length + totalProjects} items; ${visibleClips} ${visibleClips === 1 ? "clip" : "clips"}, ${visibleProjects} ${visibleProjects === 1 ? "Project" : "Projects"}`;
        }
    }
    fitTimeline() {
        const scroller = this.el("timeline-scroller");
        this.state.pixelsPerSecond = fitTimelinePixelsPerSecond(scroller.clientWidth, projectDuration(this.state.project));
        this.renderTimeline();
        scroller.scrollLeft = 0;
    }
    /**
     * Apply the same framing as Shift-Z after the browser adopts a Project.
     *
     * The first render creates and sizes the timeline scroller. Defer one browser
     * turn, then calculate pixels-per-second from that real width and render the
     * complete Project into the available space.
     *
     * Main callers: initial Studio startup and selectProject.
     */
    fitTimelineAfterProjectOpen() {
        window.setTimeout(() => {
            if (!this.state.panels.timeline) {
                return;
            }
            this.fitTimeline();
        }, 0);
    }
    zoomTimelineAt(clientX, factor) {
        const scroller = this.el("timeline-scroller");
        const rect = scroller.getBoundingClientRect();
        const before = (scroller.scrollLeft + clientX - rect.left) / this.state.pixelsPerSecond;
        this.state.pixelsPerSecond = clamp(this.state.pixelsPerSecond * factor, 18, 150);
        this.renderTimeline();
        scroller.scrollLeft = Math.max(0, before * this.state.pixelsPerSecond - (clientX - rect.left));
    }
    jumpEdit(direction) {
        const edits = [0, ...this.state.project.spine.map((item) => item.timelineStart + item.duration)];
        const next = direction > 0
            ? edits.find((time) => time > this.state.currentTime + 0.001)
            : [...edits].reverse().find((time) => time < this.state.currentTime - 0.001);
        if (next !== undefined) {
            this.seek(next);
        }
    }
    async toggleFavorite(assetId) {
        const index = this.assets.findIndex((candidate) => candidate.id === assetId);
        if (index < 0) {
            return;
        }
        const asset = this.assets[index];
        if (!asset) {
            return;
        }
        const favorite = !asset.favorite;
        if (this.runtime.mode === "localhost") {
            if (!this.runtime.setMediaFavorite) {
                throw new Error("Localhost runtime does not support FCPXML Favorites.");
            }
            try {
                const eventId = this.state.selectedEventId;
                const assets = await resultForCurrentSelection(eventId, () => this.runtime.setMediaFavorite(eventId, asset.id, favorite), () => this.state.selectedEventId);
                if (!assets) {
                    return;
                }
                this.assets = assets.map((candidate) => ({ ...candidate, tags: [...candidate.tags] }));
                this.refreshBrowserResults();
                this.showToast(favorite ? "Marked Favorite" : "Removed Favorite", "success");
            }
            catch (error) {
                this.showToast(`Favorite update failed: ${error instanceof Error ? error.message : String(error)}`, "error");
            }
            return;
        }
        this.assets[index] = { ...asset, favorite };
        this.refreshBrowserResults();
    }
    exportProject(profile = "delivery") {
        if (this.exportJob) {
            this.state.activePopover = "export";
            this.renderTopbar();
            this.showToast("An export is already in progress.", "info");
            return;
        }
        const controller = new AbortController();
        this.exportJob = {
            controller,
            progress: { status: "queued", completedFrames: 0, totalFrames: 0 },
        };
        this.state.activePopover = "export";
        this.renderTopbar();
        void this.runtime
            .renderProject(this.rootProject.id, { profile, resolution: this.exportResolution }, {
            signal: controller.signal,
            onProgress: (progress) => {
                if (this.exportJob?.controller !== controller)
                    return;
                this.exportJob.progress = progress;
                this.renderTopbar();
            },
        })
            .then((result) => {
            if (this.runtime.mode === "localhost") {
                this.recoverRuntimeProject(this.state.project.id);
                this.renderAll();
                this.syncPreview();
            }
            this.exportJob = null;
            this.state.activePopover = null;
            this.renderTopbar();
            this.showToast(result.message, "success");
        })
            .catch((error) => {
            this.exportJob = null;
            this.state.activePopover = null;
            this.renderTopbar();
            if (error instanceof DOMException && error.name === "AbortError") {
                this.showToast("Export cancelled.", "info");
                return;
            }
            this.showToast(`Export failed: ${error instanceof Error ? error.message : String(error)}`, "error");
        });
    }
    cancelExport() {
        if (!this.exportJob || this.exportJob.progress.status === "cancelling")
            return;
        this.exportJob.progress = { ...this.exportJob.progress, status: "cancelling" };
        this.exportJob.controller.abort();
        this.renderTopbar();
    }
    assetById(assetId) {
        if (!assetId) {
            return null;
        }
        return this.assets.find((asset) => asset.id === assetId) ?? null;
    }
    itemById(itemId) {
        return ([...this.state.project.spine, ...this.state.project.connected].find((item) => item.id === itemId) ?? null);
    }
    activeStorylineItem() {
        return (this.state.project.spine.find((item) => this.state.currentTime >= item.timelineStart &&
            this.state.currentTime < item.timelineStart + item.duration) ??
            this.state.project.spine.at(-1) ??
            null);
    }
    storylineIndexAtTime(time) {
        const index = this.state.project.spine.findIndex((item) => time < item.timelineStart + item.duration / 2);
        return index === -1 ? this.state.project.spine.length : index;
    }
    timeFromCanvasPointer(clientX, canvas, project = this.state.project) {
        const rect = canvas.getBoundingClientRect();
        const raw = (clientX - rect.left) / this.state.pixelsPerSecond;
        const duration = projectDuration(project);
        return clamp(raw, 0, Math.max(duration, raw));
    }
    snapTime(time, project = this.state.project) {
        if (!this.state.snapping) {
            return time;
        }
        const candidates = [
            0,
            projectDuration(project),
            this.state.currentTime,
            ...project.spine.flatMap((item) => [
                item.timelineStart,
                item.timelineStart + item.duration,
            ]),
        ];
        const threshold = 8 / this.state.pixelsPerSecond;
        const nearest = candidates.reduce((best, value) => (Math.abs(value - time) < Math.abs(best - time) ? value : best), candidates[0] ?? time);
        return Math.abs(nearest - time) <= threshold ? nearest : time;
    }
    seekFromPointer(event) {
        const target = event.target;
        if (!(target instanceof Element)) {
            return;
        }
        const canvas = target.closest(".timeline-canvas");
        if (canvas instanceof HTMLElement) {
            this.seek(this.timeFromCanvasPointer(event.clientX, canvas));
        }
    }
    /**
     * Attach mock canvas or WebRTC preview.
     *
     * In mock mode the canvas stays visible inside mock-stage chrome. Do not hide
     * the canvas — MockEditorRuntime draws into it.
     */
    async attachPreview() {
        const video = this.el("preview-video");
        if (!(video instanceof HTMLVideoElement)) {
            throw new Error("Editor shell is missing #preview-video");
        }
        const mock = this.el("mock-stage");
        const existingCanvas = document.getElementById("preview-canvas");
        let canvas;
        if (existingCanvas instanceof HTMLCanvasElement) {
            canvas = existingCanvas;
        }
        else {
            canvas = document.createElement("canvas");
            canvas.id = "preview-canvas";
            this.el("viewer-wrap").prepend(canvas);
        }
        this.preview = await this.runtime.attachPreview(canvas, video, (time) => this.seek(time, false, false), (playing) => {
            if (!playing && this.state.loopPlayback && this.state.playing) {
                this.state.currentTime = 0;
                this.previewRealtime({ type: "seek", time: 0 });
                this.previewRealtime({ type: "play" });
                this.updatePlaybackDom();
                return;
            }
            this.state.playing = playing;
            this.renderViewer();
        }, this.state.viewerView.quality, (event) => {
            this.state.connectionMessage = event.message;
            if (event.type === "source_changed") {
                this.recoverRuntimeProject(this.state.project.id);
                this.renderAll();
                this.syncPreview();
            }
            if (event.type === "error" || event.type === "missing_media") {
                this.showToast(event.message, event.type === "error" ? "error" : "info");
            }
            if (event.type === "quality" || event.type === "buffering") {
                this.renderTopbar();
            }
        });
        // Pick the visible render surface by the transport the controller chose.
        // The mock and raw-frame paths paint into the canvas; only the quarantined
        // WebRTC path uses the <video> element.
        if (this.preview.mode === "webrtc") {
            mock.hidden = true;
            canvas.hidden = true;
            video.hidden = false;
        }
        else if (this.preview.mode === "rawframe") {
            mock.hidden = true;
            video.hidden = true;
            canvas.hidden = false;
        }
        else {
            video.hidden = true;
            canvas.hidden = false;
            mock.hidden = false;
        }
        this.updateCanvasControls();
    }
    /**
     * Show a persistent banner when Media files failed to probe on open/refresh.
     *
     * Main callers: start, refreshMediaInventory.
     * Why this exists: a damaged sidecar must not block Studio, but skipping it
     * silently would look like the file was never in the bundle.
     */
    warnMediaInventoryFailures(options = {}) {
        const failures = this.paintMediaInventoryWarning();
        if (failures.length === 0) {
            if (options.refreshed) {
                this.showToast("Media inventory refreshed.", "success");
            }
            return;
        }
        const count = failures.length;
        this.showToast(count === 1
            ? "A Media file could not be read. See the warning above."
            : `${count} Media files could not be read. See the warning above.`, "error");
    }
    paintMediaInventoryWarning() {
        const failures = this.runtime.mediaInventoryFailures();
        const banner = this.el("media-warning");
        const detail = this.el("media-warning-detail");
        if (failures.length === 0) {
            banner.hidden = true;
            detail.textContent = "";
            return failures;
        }
        const count = failures.length;
        const noun = count === 1 ? "file" : "files";
        const verb = count === 1 ? "was" : "were";
        detail.textContent = `${count} ${noun} in Media could not be probed and ${verb} skipped:\n${failures.join("\n")}\nStudio opened the Project anyway. Those files are not in the browser.`;
        banner.hidden = false;
        return failures;
    }
    dismissMediaWarning() {
        this.el("media-warning").hidden = true;
    }
    showToast(message, kind = "info") {
        const toast = this.el("toast");
        toast.textContent = message;
        toast.className = `toast visible ${kind}`;
        if (this.toastTimer !== null) {
            clearTimeout(this.toastTimer);
        }
        this.toastTimer = setTimeout(() => {
            toast.className = "toast";
        }, 3000);
    }
}
function boxesIntersect(a, b) {
    return a.left < b.right && a.right > b.left && a.top < b.bottom && a.bottom > b.top;
}
function roleQuery(role) {
    const map = {
        storyline: "storyline",
        "connected-video": "connected-video",
        "connected-audio": "connected-audio",
        title: "title",
    };
    if (!role || !(role in map)) {
        return "";
    }
    return map[role];
}
const root = document.getElementById("app");
if (!root) {
    throw new Error("Missing #app root");
}
const appRoot = root;
function showStartupFailure(error) {
    console.error(error);
    appRoot.innerHTML = `<div class="fatal-error"><strong>Browser editor failed to start.</strong><pre>${String(error)}</pre></div>`;
}
try {
    const app = new BladeworksEditorApp(appRoot);
    globalThis.__bladeworksEditor = app;
    app.start().catch(showStartupFailure);
}
catch (error) {
    showStartupFailure(error);
}
