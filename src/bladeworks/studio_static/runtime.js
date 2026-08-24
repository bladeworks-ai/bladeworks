/**
 * Browser-to-editor-runtime boundary.
 *
 * Architecture map:
 * EditorRuntime interface
 *   ├─ MockEditorRuntime: in-memory reducer + neutral preview canvas
 *   └─ LocalhostEditorRuntime: complete-library HTTP + SSE + WebRTC media
 *
 * Main callers:
 * - BladeworksEditorApp during boot, project selection, edit commits, playback,
 *   and render handoff.
 *
 * Why this exists:
 * The browser UI must not become coupled to synthetic fixtures or to a specific
 * native process implementation. The same UI should run in standalone mock
 * mode today and against canonical FCPXML tomorrow.
 */
import { createFixtureBootstrap } from "./fixtures.js";
import { MOCK_EDITOR_CAPABILITIES } from "./mock-capabilities.js";
import { addEventToFCPXML, addProjectToFCPXML, createBasicTitleClip, createCaptionClip, createCustomSolidClip, favoriteAssetIdsForEvent, parseFCPXMLLibrary, replaceProjectInFCPXML, setEventAssetFavorite, } from "./fcpxml.js";
import { applyEdit, clipAtTime, projectDuration } from "./magnetic-timeline.js";
function cloneValue(value) {
    return structuredClone(value);
}
function requireOk(response, context) {
    if (!response.ok) {
        return response.text().then((body) => {
            throw new Error(`${context} failed (${response.status}): ${body || response.statusText}`);
        });
    }
    return Promise.resolve(response);
}
class MockCanvasPreview {
    mode = "mock-canvas";
    canvas;
    context;
    onTimeUpdate;
    onPlayingChange;
    state = null;
    playing = false;
    animationFrame = null;
    lastFrameTime = null;
    resizeObserver;
    constructor(canvas, onTimeUpdate, onPlayingChange) {
        this.canvas = canvas;
        const context = canvas.getContext("2d");
        if (!context) {
            throw new Error("The browser could not create a 2D preview canvas.");
        }
        this.context = context;
        this.onTimeUpdate = onTimeUpdate;
        this.onPlayingChange = onPlayingChange;
        this.resizeObserver = new ResizeObserver(() => this.draw());
        this.resizeObserver.observe(canvas);
    }
    setState(state) {
        this.state = cloneValue(state);
        if (!this.playing) {
            this.draw();
        }
    }
    setQuality(_quality) { }
    play() {
        if (this.playing || !this.state) {
            return;
        }
        this.playing = true;
        this.lastFrameTime = null;
        this.onPlayingChange(true);
        this.animationFrame = requestAnimationFrame((timestamp) => this.tick(timestamp));
    }
    pause() {
        if (!this.playing) {
            return;
        }
        this.playing = false;
        this.lastFrameTime = null;
        if (this.animationFrame !== null) {
            cancelAnimationFrame(this.animationFrame);
            this.animationFrame = null;
        }
        this.onPlayingChange(false);
        this.draw();
    }
    seek(time) {
        if (!this.state) {
            return;
        }
        const duration = projectDuration(this.state.project);
        const nextTime = Math.min(duration, Math.max(0, time));
        this.state = { ...this.state, playhead: nextTime };
        this.onTimeUpdate(nextTime);
        this.draw();
    }
    destroy() {
        this.pause();
        this.resizeObserver.disconnect();
    }
    tick(timestamp) {
        if (!this.playing || !this.state) {
            return;
        }
        if (this.lastFrameTime === null) {
            this.lastFrameTime = timestamp;
        }
        const elapsed = Math.min(0.1, (timestamp - this.lastFrameTime) / 1_000);
        this.lastFrameTime = timestamp;
        const duration = projectDuration(this.state.project);
        let nextTime = this.state.playhead + elapsed;
        if (nextTime >= duration) {
            nextTime = 0;
        }
        this.state = { ...this.state, playhead: nextTime };
        this.onTimeUpdate(nextTime);
        this.draw();
        this.animationFrame = requestAnimationFrame((nextTimestamp) => this.tick(nextTimestamp));
    }
    prepareCanvas() {
        const bounds = this.canvas.getBoundingClientRect();
        const ratio = Math.min(window.devicePixelRatio || 1, 2);
        const width = Math.max(1, Math.round(bounds.width * ratio));
        const height = Math.max(1, Math.round(bounds.height * ratio));
        if (this.canvas.width !== width || this.canvas.height !== height) {
            this.canvas.width = width;
            this.canvas.height = height;
        }
        this.context.setTransform(ratio, 0, 0, ratio, 0, 0);
        return { width: bounds.width, height: bounds.height };
    }
    draw() {
        const { width, height } = this.prepareCanvas();
        const context = this.context;
        context.clearRect(0, 0, width, height);
        context.fillStyle = "#0b0d13";
        context.fillRect(0, 0, width, height);
        if (!this.state) {
            context.fillStyle = "rgba(255,255,255,0.35)";
            context.font = "12px -apple-system, sans-serif";
            context.textAlign = "center";
            context.fillText("No Project selected", width / 2, height / 2);
            return;
        }
        const active = clipAtTime(this.state.project, this.state.playhead);
        if (active && active.kind !== "gap") {
            context.fillStyle = "rgba(255,255,255,0.42)";
            context.font = "11px -apple-system, sans-serif";
            context.textAlign = "left";
            context.fillText(active.name, 14, height - 20);
        }
        const duration = projectDuration(this.state.project);
        const progress = duration > 0 ? Math.min(1, this.state.playhead / duration) : 0;
        context.fillStyle = "#252525";
        context.fillRect(14, height - 10, Math.max(0, width - 28), 2);
        context.fillStyle = "#777";
        context.fillRect(14, height - 10, Math.max(0, width - 28) * progress, 2);
    }
}
export class MockEditorRuntime {
    mode = "mock";
    bootstrapData;
    projects = new Map();
    libraries;
    activeProjectId;
    constructorWorkspace = null;
    constructor() {
        this.bootstrapData = createFixtureBootstrap();
        this.libraries = cloneValue(this.bootstrapData.libraries);
        this.activeProjectId = this.bootstrapData.activeProjectId;
        for (const [projectId, project] of Object.entries(this.bootstrapData.projects)) {
            this.projects.set(projectId, cloneValue(project));
        }
    }
    async bootstrap() {
        const projects = Object.fromEntries([...this.projects.entries()].map(([projectId, project]) => [projectId, cloneValue(project)]));
        return cloneValue({
            ...this.bootstrapData,
            libraries: this.libraries,
            projects,
            activeProjectId: this.activeProjectId,
        });
    }
    async capabilities() {
        return cloneValue(MOCK_EDITOR_CAPABILITIES);
    }
    createBasicTitleClip(options) {
        return createBasicTitleClip(this.requireConstructorWorkspace(), options);
    }
    createCaptionClip(options) {
        return createCaptionClip(this.requireConstructorWorkspace(), options);
    }
    createCustomSolidClip(options) {
        return createCustomSolidClip(this.requireConstructorWorkspace(), options);
    }
    snapshot() {
        const projects = Object.fromEntries([...this.projects.entries()].map(([projectId, project]) => [projectId, cloneValue(project)]));
        return cloneValue({
            ...this.bootstrapData,
            libraries: this.libraries,
            projects,
            activeProjectId: this.activeProjectId,
        });
    }
    async selectProject(projectId) {
        const project = this.projects.get(projectId);
        if (!project) {
            throw new Error(`Mock Project ${projectId} does not exist.`);
        }
        this.activeProjectId = projectId;
        return cloneValue(project);
    }
    async createProject(eventId) {
        const event = this.libraries
            .flatMap((library) => library.events)
            .find((candidate) => candidate.id === eventId);
        if (!event) {
            throw new Error(`Event ${eventId} does not exist.`);
        }
        const current = this.projects.get(this.activeProjectId);
        const existingNames = event.projects.map((project) => project.name);
        const name = existingNames.includes("Untitled Project")
            ? `Untitled Project ${existingNames.filter((candidate) => candidate.startsWith("Untitled Project")).length + 1}`
            : "Untitled Project";
        const id = `project_${Date.now()}`;
        const snapshot = {
            revision: 0,
            id,
            libraryId: event.libraryId,
            eventId: event.id,
            name,
            fps: current?.fps ?? 30,
            width: current?.width ?? 1920,
            height: current?.height ?? 1080,
            audioLayout: current?.audioLayout ?? "stereo",
            spine: [],
            connected: [],
            transitions: [],
            proposal: null,
        };
        this.projects.set(id, snapshot);
        this.libraries = this.libraries.map((library) => ({
            ...library,
            events: library.events.map((candidate) => candidate.id === eventId
                ? {
                    ...candidate,
                    projects: [
                        ...candidate.projects,
                        { id, eventId, name, duration: 0, proposal: null },
                    ],
                }
                : candidate),
        }));
        this.activeProjectId = id;
        return cloneValue(snapshot);
    }
    async createEvent(libraryId) {
        const library = this.libraries.find((candidate) => candidate.id === libraryId);
        if (!library)
            throw new Error(`Library ${libraryId} does not exist.`);
        const existingNames = library.events.map((event) => event.name);
        let name = "New Event";
        let suffix = 2;
        while (existingNames.includes(name))
            name = `New Event ${suffix++}`;
        const id = `${libraryId}/event[${library.events.length + 1}]`;
        const event = { id, libraryId, name, projects: [] };
        this.libraries = this.libraries.map((candidate) => candidate.id === libraryId
            ? { ...candidate, events: [...candidate.events, event] }
            : candidate);
        return cloneValue(event);
    }
    async commitEdit(request) {
        const current = this.projects.get(request.projectId);
        if (!current) {
            throw new Error(`Mock Project ${request.projectId} does not exist.`);
        }
        if (current.revision !== request.baseRevision) {
            throw new Error(`Project revision conflict: browser has ${request.baseRevision}, runtime has ${current.revision}.`);
        }
        const next = applyEdit(current, request.operation);
        this.projects.set(request.projectId, next);
        return cloneValue(next);
    }
    async commitEditSequence(request) {
        let current = this.projects.get(request.projectId);
        if (!current) {
            throw new Error(`Mock Project ${request.projectId} does not exist.`);
        }
        if (current.revision !== request.baseRevision) {
            throw new Error(`Project revision conflict: browser has ${request.baseRevision}, runtime has ${current.revision}.`);
        }
        for (const operation of request.operations) {
            current = applyEdit(current, operation);
        }
        this.projects.set(request.projectId, current);
        return cloneValue(current);
    }
    async restoreProject(project, baseRevision) {
        const current = this.projects.get(project.id);
        if (!current) {
            throw new Error(`Mock Project ${project.id} does not exist.`);
        }
        if (current.revision !== baseRevision) {
            throw new Error(`Project revision conflict: browser has ${baseRevision}, runtime has ${current.revision}.`);
        }
        const restored = { ...cloneValue(project), revision: current.revision + 1 };
        this.projects.set(project.id, restored);
        return cloneValue(restored);
    }
    async undoProject() {
        throw new Error("Mock undo is owned by the browser history adapter.");
    }
    async redoProject() {
        throw new Error("Mock redo is owned by the browser history adapter.");
    }
    historyState() {
        return { canUndo: false, canRedo: false, index: 0, length: 1 };
    }
    projectEditability() {
        return { editable: true, reasons: [], degraded: false, warnings: [] };
    }
    mediaInventoryFailures() {
        return [];
    }
    async refreshMedia() {
        return cloneValue(this.bootstrapData.assets);
    }
    async importMedia() {
        throw new Error("Mock media import is owned by the fixture UI.");
    }
    async loadMediaVisual(_request) {
        throw new Error("Real media visuals are unavailable in standalone fixture mode.");
    }
    async attachPreview(canvas, video, onTimeUpdate, onPlayingChange, _quality, _onRuntimeEvent) {
        video.style.display = "none";
        canvas.style.display = "block";
        return new MockCanvasPreview(canvas, onTimeUpdate, onPlayingChange);
    }
    async renderProject(projectId, options = { profile: "delivery", resolution: 1080 }, lifecycle = {}) {
        const project = this.projects.get(projectId);
        if (!project) {
            throw new Error(`Mock Project ${projectId} does not exist.`);
        }
        if (lifecycle.signal?.aborted)
            throw new DOMException("Export cancelled.", "AbortError");
        lifecycle.onProgress?.({ status: "queued", completedFrames: 0, totalFrames: 1 });
        lifecycle.onProgress?.({ status: "completed", completedFrames: 1, totalFrames: 1 });
        return {
            message: `Mock mode prepared “${project.name}” as ${options.profile} at ${options.resolution}p. No movie was written.`,
        };
    }
    requireConstructorWorkspace() {
        if (!this.constructorWorkspace) {
            this.constructorWorkspace = parseFCPXMLLibrary('<?xml version="1.0"?><!DOCTYPE fcpxml><fcpxml version="1.14"><resources><format id="r1" frameDuration="1/30s" width="1920" height="1080"/></resources><library><event><project name="Fixture"><sequence format="r1" duration="1s"><spine><gap name="Gap" offset="0s" duration="1s"/></spine></sequence></project></event></library></fcpxml>');
        }
        return this.constructorWorkspace;
    }
}
/**
 * Which live-media transport the localhost editor uses.
 *
 * "rawframe" is the default: uncompressed frames over a WebSocket, painted to a
 * canvas (~50 ms glass-to-glass on loopback). "webrtc" is the quarantined
 * legacy path (aiortc encode + browser jitter buffer, ~1 s latency) and also
 * needs the server started with BLADEFRAME_PREVIEW_WEBRTC=1.
 */
const PREVIEW_TRANSPORT = "rawframe";
export function previewResolutionForQuality(quality) {
    const resolutions = {
        "better-quality": "720p",
        "better-performance": "540p",
        "best-performance": "480p",
    };
    return resolutions[quality];
}
function previewQualityPayload(quality) {
    return { resolution: previewResolutionForQuality(quality) };
}
function etagVersion(response) {
    const raw = response.headers.get("ETag") ?? response.headers.get("X-Bladeworks-Disk-Version");
    if (!raw) {
        throw new Error("Bladeworks source response did not include a source version.");
    }
    return raw.startsWith('"') && raw.endsWith('"') ? raw.slice(1, -1) : raw;
}
async function sha256Version(value) {
    // WebCrypto is unavailable on non-secure LAN origins. The source server is
    // still authoritative and returns its own SHA-256 after accepting the exact
    // request body, so the client-side duplicate check is optional there.
    if (!globalThis.crypto?.subtle)
        return null;
    const digest = await globalThis.crypto.subtle.digest("SHA-256", new TextEncoder().encode(value));
    const hex = [...new Uint8Array(digest)]
        .map((byte) => byte.toString(16).padStart(2, "0"))
        .join("");
    return `sha256:${hex}`;
}
function mediaColors(value) {
    let hash = 2166136261;
    for (const character of value) {
        hash = Math.imul(hash ^ character.charCodeAt(0), 16777619);
    }
    const hue = Math.abs(hash) % 360;
    return { a: `hsl(${hue} 38% 25%)`, b: `hsl(${(hue + 42) % 360} 55% 54%)` };
}
// Still-image container extensions. PyAV exposes an ordinary PNG or JPEG as a
// single-frame *video* stream, so the backend probe reports ``hasVideo: true``
// for stills exactly as it does for motion video. The stream flags therefore
// cannot tell the two apart, and the filename extension is the explicit signal
// we classify on. Anything not listed here that carries a video stream is
// treated as motion video.
const STILL_IMAGE_EXTENSIONS = new Set([
    "png", "jpg", "jpeg", "gif", "bmp", "tif", "tiff", "webp", "heic", "heif",
]);
function isStillImagePath(relativePath) {
    const dot = relativePath.lastIndexOf(".");
    if (dot < 0) {
        return false;
    }
    return STILL_IMAGE_EXTENSIONS.has(relativePath.slice(dot + 1).toLowerCase());
}
function mediaAsset(record) {
    // A still image probes as a video stream, so it must be classified by
    // extension *before* the ``hasVideo`` check; only then does a real motion
    // video fall through to ``video``. ``hasAudio`` is carried verbatim so the
    // FCPXML serializer can declare the true audio presence of the source
    // instead of assuming every video clip has an audio stream.
    const kind = isStillImagePath(record.relativePath)
        ? "image"
        : record.hasVideo
            ? "video"
            : record.hasAudio
                ? "audio"
                : "image";
    return {
        id: `media:${record.relativePath}`,
        sourcePath: record.relativePath,
        name: record.filename,
        kind,
        hasAudio: record.hasAudio,
        duration: record.duration ?? (kind === "image" ? 5 : 0),
        ...(record.width ? { width: record.width } : {}),
        ...(record.height ? { height: record.height } : {}),
        ...(record.frameDuration ? { frameDurationRaw: record.frameDuration } : {}),
        colors: mediaColors(record.relativePath),
        tags: ["Media", ...record.relativePath.split("/").slice(0, -1)],
        glyph: kind === "audio" ? "♫" : kind === "image" ? "▣" : "◉",
    };
}
function previewEventMessage(event, data) {
    if (event === "quality") {
        return { type: "quality", message: `Preview quality: ${String(data.resolution ?? "updated")}` };
    }
    if (event === "buffering") {
        return { type: "buffering", message: "Preview is buffering." };
    }
    if (event === "missing_media") {
        const paths = Array.isArray(data.paths) ? data.paths.map((item) => {
            if (item && typeof item === "object") {
                const record = item;
                return String(record.basename ?? record.locator ?? "unknown media");
            }
            return String(item);
        }) : [];
        return { type: "missing_media", message: paths.length ? `Missing media: ${paths.join(", ")}` : "Missing media." };
    }
    if (event === "ready" || event === "ended" || event === "error") {
        const message = typeof data.message === "string" ? data.message : `Preview ${event}.`;
        return { type: event, message };
    }
    return null;
}
/** Marker subprotocol that carries the bearer token on the stream WebSocket. */
const RAW_STREAM_SUBPROTOCOL = "bladeworks-preview";
/** Quarantined WebRTC transport: the negotiated peer feeding a <video>. */
class WebRTCTransport {
    mode = "webrtc";
    peer;
    video;
    constructor(peer, video) {
        this.peer = peer;
        this.video = video;
    }
    beginFrameUpdate() {
        // The legacy WebRTC stream retains its last decoded frame while syncing.
    }
    startMediaPlayback() {
        this.video.muted = false;
        void this.video.play().catch(() => undefined);
    }
    stopMediaPlayback() {
        // WebRTC schedules nothing on the client, so pausing the <video> element is
        // the whole of the pause/seek cut here.
        this.video.pause();
    }
    close() {
        this.peer.close();
    }
}
/**
 * The main preview transport: uncompressed frames over a WebSocket.
 *
 * Architecture:
 * - A binary WebSocket delivers struct-framed messages. Video frames are RGBA
 *   painted straight onto the canvas with putImageData (~50 ms glass-to-glass
 *   on loopback, versus ~1 s through WebRTC's encoder + jitter buffer). Audio
 *   frames are PCM scheduled sequentially into a WebAudio graph.
 * - The backend scan loop is the master clock: it already paces both video and
 *   audio to wall-clock realtime, so painting video on arrival and scheduling
 *   audio on arrival keeps them in sync at the source — no client-side clock.
 *
 * Wire format matches backend/preview/rawframe.py.
 */
export class RawFrameTransport {
    mode = "rawframe";
    socket;
    canvas;
    context;
    warmup;
    onError;
    audioContext = null;
    nextAudioTime = 0;
    audioGeneration = 0;
    // Every scheduled-but-not-yet-finished PCM chunk. A fire-and-forget
    // AudioBufferSourceNode cannot be stopped once discarded, so we retain each
    // one here and drop it when it ends, letting pause/seek cancel whatever audio
    // is still queued into the future.
    activeSources = new Set();
    frameWidth = 0;
    frameHeight = 0;
    awaitingFirstFrame = false;
    warmupTimer = null;
    closed = false;
    constructor(options) {
        this.canvas = options.canvas;
        const context = options.canvas.getContext("2d");
        if (!context) {
            throw new Error("The browser could not create a 2D preview canvas.");
        }
        this.context = context;
        this.warmup = options.warmup;
        this.onError = options.onError;
        // The bearer token rides as a subprotocol because a browser cannot set an
        // Authorization header on a WebSocket handshake.
        this.socket = new WebSocket(options.url, [RAW_STREAM_SUBPROTOCOL, options.token]);
        this.socket.binaryType = "arraybuffer";
        this.socket.onmessage = (event) => this.onMessage(event);
        this.socket.onerror = () => {
            if (!this.closed) {
                this.onError("Preview frame stream failed.");
            }
        };
    }
    startMediaPlayback() {
        // Resume the audio graph inside the user gesture so scheduled PCM is
        // audible, and reveal the warm-up spinner until the first frame lands.
        this.ensureAudioContext();
        void this.audioContext?.resume().catch(() => undefined);
        this.awaitingFirstFrame = true;
        this.showWarmup();
    }
    beginFrameUpdate() {
        this.awaitingFirstFrame = true;
        this.showWarmup();
    }
    close() {
        this.closed = true;
        this.hideWarmup();
        try {
            this.socket.close();
        }
        catch {
            // The socket may already be closing; nothing more to do.
        }
        if (this.audioContext) {
            void this.audioContext.close().catch(() => undefined);
            this.audioContext = null;
        }
    }
    onMessage(event) {
        if (typeof event.data === "string") {
            // The only text message is the connection "meta" acknowledgement.
            return;
        }
        const buffer = event.data;
        const view = new DataView(buffer);
        const kind = view.getUint8(0);
        if (kind === 0) {
            this.paintVideo(buffer, view);
        }
        else if (kind === 1) {
            this.playAudio(buffer, view);
        }
    }
    paintVideo(buffer, view) {
        const width = view.getUint16(5, true);
        const height = view.getUint16(7, true);
        if (width !== this.frameWidth || height !== this.frameHeight) {
            this.canvas.width = width;
            this.canvas.height = height;
            this.frameWidth = width;
            this.frameHeight = height;
        }
        const pixels = new Uint8ClampedArray(buffer, 9);
        this.context.putImageData(new ImageData(pixels, width, height), 0, 0);
        if (this.awaitingFirstFrame) {
            this.awaitingFirstFrame = false;
            this.hideWarmup();
        }
    }
    playAudio(buffer, view) {
        const audioContext = this.ensureAudioContext();
        if (!audioContext) {
            return;
        }
        if (audioContext.state !== "running") {
            const generation = this.audioGeneration;
            void audioContext.resume().then(() => {
                if (!this.closed && generation === this.audioGeneration) {
                    this.playAudio(buffer, view);
                }
            }).catch(() => undefined);
            return;
        }
        const sampleRate = view.getUint32(9, true);
        const channels = view.getUint8(13);
        if (channels < 1) {
            return;
        }
        const interleaved = new Int16Array(buffer, 14);
        const frames = Math.floor(interleaved.length / channels);
        if (frames < 1) {
            return;
        }
        const audioBuffer = audioContext.createBuffer(channels, frames, sampleRate);
        for (let channel = 0; channel < channels; channel += 1) {
            const channelData = audioBuffer.getChannelData(channel);
            for (let frame = 0; frame < frames; frame += 1) {
                channelData[frame] = (interleaved[frame * channels + channel] ?? 0) / 32768;
            }
        }
        const source = audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(audioContext.destination);
        // Retain the node so pause/seek can stop it, and release it when it ends.
        this.activeSources.add(source);
        source.onended = () => {
            this.activeSources.delete(source);
        };
        // Schedule contiguously. If we have fallen behind (a stall), restart from
        // the current time rather than piling up ever-growing latency.
        const startAt = Math.max(audioContext.currentTime, this.nextAudioTime);
        source.start(startAt);
        this.nextAudioTime = startAt + audioBuffer.duration;
    }
    stopMediaPlayback() {
        this.audioGeneration += 1;
        // Stop every chunk scheduled into the future and forget the schedule so the
        // next play/seek starts contiguously from the audio clock again. `stop()`
        // fires `onended`, which removes each node from the set.
        for (const source of [...this.activeSources]) {
            try {
                source.stop();
            }
            catch {
                // Already stopped or never started; nothing to cancel.
            }
            source.disconnect();
        }
        this.activeSources.clear();
        this.nextAudioTime = 0;
    }
    ensureAudioContext() {
        if (this.audioContext || this.closed) {
            return this.audioContext;
        }
        const AudioContextCtor = window.AudioContext ?? window.webkitAudioContext;
        if (!AudioContextCtor) {
            return null;
        }
        this.audioContext = new AudioContextCtor();
        this.nextAudioTime = 0;
        return this.audioContext;
    }
    showWarmup() {
        if (this.warmup) {
            this.warmup.hidden = false;
        }
        if (this.warmupTimer !== null) {
            window.clearTimeout(this.warmupTimer);
        }
        // Safety net: the spinner can never stick even if a frame never arrives.
        this.warmupTimer = window.setTimeout(() => this.hideWarmup(), 4000);
    }
    hideWarmup() {
        if (this.warmup) {
            this.warmup.hidden = true;
        }
        if (this.warmupTimer !== null) {
            window.clearTimeout(this.warmupTimer);
            this.warmupTimer = null;
        }
    }
}
/** HTTP/SSE control plane for one long-lived preview session. */
export class LocalhostPreviewController {
    get mode() {
        return this.transport.mode;
    }
    transport;
    sessionId;
    eventsUrl;
    request;
    identity;
    beforePlay;
    onTimeUpdate;
    onPlayingChange;
    onRuntimeEvent;
    eventsAbort = new AbortController();
    commandQueue = Promise.resolve();
    synced;
    quality;
    syncedQuality;
    playhead;
    requestNumber = 0;
    destroyed = false;
    // Whether the client wants playback running. A quality change stops the
    // backend scan (it syncs a new producer); this flag lets us auto-resume.
    playing = false;
    constructor(options) {
        this.transport = options.transport;
        this.sessionId = options.sessionId;
        this.eventsUrl = options.eventsUrl;
        this.request = options.request;
        this.identity = options.identity;
        this.beforePlay = options.beforePlay;
        this.onTimeUpdate = options.onTimeUpdate;
        this.onPlayingChange = options.onPlayingChange;
        this.onRuntimeEvent = options.onRuntimeEvent;
        this.synced = options.initialIdentity;
        this.quality = options.initialQuality;
        this.syncedQuality = options.initialQuality;
        this.playhead = options.initialPlayhead;
        void this.consumeEvents();
    }
    setState(state) {
        this.playhead = state.playhead;
        const current = this.identity();
        if (current.sourceVersion !== this.synced.sourceVersion || current.projectRef !== this.synced.projectRef) {
            this.transport.stopMediaPlayback();
            // Keep the last good canvas visible, but make it clear that it belongs
            // to the previous revision until the first replacement frame arrives.
            this.transport.beginFrameUpdate();
            this.enqueue(async () => {
                await this.sync();
                await this.refreshPreviewAfterSync();
            });
        }
    }
    setQuality(quality) {
        if (quality === this.quality)
            return;
        this.quality = quality;
        this.transport.stopMediaPlayback();
        this.transport.beginFrameUpdate();
        this.enqueue(async () => {
            await this.sync();
            await this.refreshPreviewAfterSync();
        });
    }
    play() {
        this.playing = true;
        // This method is called directly from a user click. Start the media
        // transport before awaiting HTTP so browsers preserve that user gesture
        // and permit audio (the WebAudio graph / negotiated track) to be audible.
        this.transport.startMediaPlayback();
        this.enqueue(async () => {
            await this.beforePlay();
            await this.syncIfNeeded();
            await requireOk(await this.request(`/api/editor/preview/sessions/${this.sessionId}/play`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ time: this.playhead }),
            }), "Preview play");
        });
    }
    pause() {
        this.playing = false;
        // Cut client-scheduled audio immediately so it does not play past the pause
        // while the HTTP pause command is still in flight.
        this.transport.stopMediaPlayback();
        this.enqueue(async () => {
            await requireOk(await this.request(`/api/editor/preview/sessions/${this.sessionId}/pause`, { method: "POST" }), "Preview pause");
        });
    }
    seek(time) {
        this.playhead = time;
        // Drop audio scheduled for the old position so the seek target is not
        // delayed behind stale, now-wrong samples.
        this.transport.stopMediaPlayback();
        const sequence = ++this.requestNumber;
        this.enqueue(async () => {
            await this.syncIfNeeded();
            const response = await requireOk(await this.request(`/api/editor/preview/sessions/${this.sessionId}/seek`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ time, requestId: `seek-${sequence}` }),
            }), "Preview seek");
            const result = await response.json();
            if (sequence !== this.requestNumber) {
                return;
            }
            if (typeof result.actualTime === "number") {
                this.playhead = result.actualTime;
                this.onTimeUpdate(result.actualTime);
            }
            // Backend seek stops the scan. If the user was playing, restart from the
            // new playhead; otherwise the canvas stays on the still while the app
            // clock keeps walking.
            await this.resumePlaybackIfNeeded();
        });
    }
    destroy() {
        if (this.destroyed) {
            return;
        }
        this.destroyed = true;
        this.eventsAbort.abort();
        this.transport.close();
        // keepalive lets the browser finish the authenticated close while a Studio
        // tab itself is unloading. Server shutdown remains the final backstop.
        void this.request(`/api/editor/preview/sessions/${this.sessionId}`, {
            method: "DELETE",
            keepalive: true,
        });
    }
    enqueue(command) {
        this.commandQueue = this.commandQueue.then(command).catch((error) => {
            this.onRuntimeEvent?.({
                type: "error",
                message: error instanceof Error ? error.message : String(error),
            });
        });
    }
    /**
     * Paint a current frame after sync replaces the producer and stops the scan.
     *
     * Why this exists: `/sync` only swaps the compiled source. Without a still
     * seek (paused) or play restart (playing), the canvas keeps the pre-edit
     * frame while the UI clock can keep walking.
     *
     * Main callers: setState after an accepted edit changes sourceVersion;
     * setQuality after a resolution change.
     */
    async refreshPreviewAfterSync() {
        if (this.destroyed) {
            return;
        }
        if (this.playing) {
            await this.resumePlaybackIfNeeded();
            return;
        }
        const sequence = ++this.requestNumber;
        const response = await requireOk(await this.request(`/api/editor/preview/sessions/${this.sessionId}/seek`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ time: this.playhead, requestId: `sync-still-${sequence}` }),
        }), "Preview still");
        if (this.destroyed || sequence !== this.requestNumber) {
            return;
        }
        const result = await response.json();
        if (typeof result.actualTime === "number") {
            this.playhead = result.actualTime;
            this.onTimeUpdate(result.actualTime);
        }
    }
    async resumePlaybackIfNeeded() {
        if (!this.playing) {
            return;
        }
        this.transport.startMediaPlayback();
        await requireOk(await this.request(`/api/editor/preview/sessions/${this.sessionId}/play`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ time: this.playhead }),
        }), "Preview resume");
    }
    async syncIfNeeded() {
        const current = this.identity();
        if (current.sourceVersion !== this.synced.sourceVersion || current.projectRef !== this.synced.projectRef || this.quality !== this.syncedQuality) {
            await this.sync();
        }
    }
    async sync() {
        const current = this.identity();
        await requireOk(await this.request(`/api/editor/preview/sessions/${this.sessionId}/sync`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ...current, playhead: this.playhead, quality: previewQualityPayload(this.quality) }),
        }), "Preview sync");
        this.synced = current;
        this.syncedQuality = this.quality;
    }
    async consumeEvents() {
        let lastEventId = "";
        while (!this.destroyed) {
            try {
                const headers = new Headers();
                if (lastEventId) {
                    headers.set("Last-Event-ID", lastEventId);
                }
                const response = await requireOk(await this.request(this.eventsUrl, { headers, signal: this.eventsAbort.signal }), "Preview events");
                const reader = response.body?.getReader();
                if (!reader) {
                    throw new Error("Preview event response did not expose a stream.");
                }
                const decoder = new TextDecoder();
                let buffer = "";
                while (!this.destroyed) {
                    const chunk = await reader.read();
                    if (chunk.done) {
                        break;
                    }
                    buffer += decoder.decode(chunk.value, { stream: true }).replace(/\r\n/g, "\n");
                    let boundary = buffer.indexOf("\n\n");
                    while (boundary >= 0) {
                        const block = buffer.slice(0, boundary);
                        buffer = buffer.slice(boundary + 2);
                        const parsed = this.parseEvent(block);
                        if (parsed) {
                            lastEventId = parsed.id || lastEventId;
                            this.handleEvent(parsed.event, parsed.data);
                        }
                        boundary = buffer.indexOf("\n\n");
                    }
                }
            }
            catch (error) {
                if (this.destroyed || this.eventsAbort.signal.aborted) {
                    return;
                }
                this.onRuntimeEvent?.({
                    type: "error",
                    message: `Preview event stream disconnected: ${error instanceof Error ? error.message : String(error)}`,
                });
                await new Promise((resolve) => setTimeout(resolve, 500));
            }
        }
    }
    parseEvent(block) {
        let id = "";
        let event = "message";
        const data = [];
        for (const line of block.split("\n")) {
            if (line.startsWith("id:")) {
                id = line.slice(3).trim();
            }
            else if (line.startsWith("event:")) {
                event = line.slice(6).trim();
            }
            else if (line.startsWith("data:")) {
                data.push(line.slice(5).trimStart());
            }
        }
        if (!data.length) {
            return null;
        }
        return { id, event, data: JSON.parse(data.join("\n")) };
    }
    handleEvent(event, data) {
        const current = this.identity();
        if (data.sourceVersion !== current.sourceVersion || data.projectRef !== current.projectRef) {
            return;
        }
        if (event === "time" && typeof data.time === "number") {
            this.playhead = data.time;
            this.onTimeUpdate(data.time);
        }
        else if (event === "playing" && typeof data.playing === "boolean") {
            this.playing = data.playing;
            this.onPlayingChange(data.playing);
        }
        const visible = previewEventMessage(event, data);
        if (visible) {
            this.onRuntimeEvent?.(visible);
        }
    }
}
/**
 * Canonical localhost adapter.
 *
 * The complete Info.fcpxml document and its SHA-256 hash are the concurrency
 * boundary. Projects are version-scoped views inside that document. Every
 * accepted browser gesture serializes one complete XML replacement.
 */
export class LocalhostEditorRuntime {
    mode = "localhost";
    baseUrl;
    token;
    fetcher;
    workspace = null;
    sourceVersion = null;
    activeProjectId = null;
    // Monotonic token for concurrent selectProject calls: only the latest may
    // adopt `activeProjectId`, which preview identity reads.
    selectionSeq = 0;
    mediaAssets = [];
    // Paths/messages for Media files PyAV could not probe. Empty means the
    // inventory scan completed without skipped files. Bootstrap must still
    // succeed when this list is non-empty; the UI warns loudly instead.
    mediaFailures = [];
    access = new Map();
    history = { canUndo: false, canRedo: false, index: 0, length: 1 };
    mutationTail = Promise.resolve();
    capabilityData = null;
    constructor(baseUrl, token, fetcher = fetch) {
        this.baseUrl = new URL(baseUrl);
        this.token = token;
        // Browser fetch is a Web API method, not a receiver-free ordinary
        // function. Calling a stored `window.fetch` as `this.fetcher(...)` binds
        // `this` to the runtime instance and Chromium rejects it as an illegal
        // invocation. Bind the injected/default function once to the global Web
        // API receiver. Test fetch functions do not depend on the receiver.
        this.fetcher = fetcher.bind(globalThis);
    }
    async bootstrap() {
        await this.loadSource();
        await this.loadMedia(false);
        const bootstrap = this.requireWorkspace().bootstrap;
        const selected = bootstrap.projects[bootstrap.activeProjectId]
            ? bootstrap.activeProjectId
            : Object.keys(bootstrap.projects)[0];
        if (!selected) {
            throw new Error("The opened FCPXML library does not contain a Project.");
        }
        this.activeProjectId = selected;
        await this.loadCompatibility(selected);
        return this.assembledBootstrap(selected);
    }
    /** Load the renderer-owned authoring surface once for this server process. */
    async capabilities() {
        if (this.capabilityData) {
            return cloneValue(this.capabilityData);
        }
        const response = await requireOk(await this.request("/api/editor/capabilities"), "Bladeworks capability catalog");
        const payload = await response.json();
        if (payload.schemaVersion !== 1 || payload.renderer !== "tensor") {
            throw new Error(`Bladeworks capability catalog has unsupported schema ${String(payload.schemaVersion)} or renderer ${String(payload.renderer)}.`);
        }
        this.capabilityData = cloneValue(payload);
        return cloneValue(payload);
    }
    createBasicTitleClip(options) {
        return createBasicTitleClip(this.requireWorkspace(), options);
    }
    createCaptionClip(options) {
        return createCaptionClip(this.requireWorkspace(), options);
    }
    createCustomSolidClip(options) {
        return createCustomSolidClip(this.requireWorkspace(), options);
    }
    snapshot() {
        if (!this.activeProjectId) {
            throw new Error("The FCPXML library does not have an active Project.");
        }
        return this.assembledBootstrap(this.activeProjectId);
    }
    async selectProject(projectId) {
        // Two rapid selections run concurrently through the awaits below. Only the
        // most recent one may own `activeProjectId`: preview identity reads it, so a
        // superseded selection winning the race would bind the preview to a
        // different Project than the caller ultimately adopts into the timeline.
        const seq = ++this.selectionSeq;
        await this.refreshSourceAtBoundary();
        const project = this.requireWorkspace().bootstrap.projects[projectId];
        if (!project) {
            throw new Error(`Project ${projectId} does not exist in the current library version.`);
        }
        if (seq === this.selectionSeq) {
            this.activeProjectId = projectId;
        }
        await this.loadCompatibility(projectId);
        return cloneValue(project);
    }
    async createProject(eventId) {
        return this.enqueueMutation(() => this.createProjectNow(eventId));
    }
    async createEvent(libraryId) {
        return this.enqueueMutation(() => this.createEventNow(libraryId));
    }
    async createEventNow(libraryId) {
        const versionBeforeCheck = this.requireSourceVersion();
        await this.loadSource();
        if (this.requireSourceVersion() !== versionBeforeCheck) {
            throw new Error("Source changed on disk. The library was reloaded; review it before creating an Event.");
        }
        const created = addEventToFCPXML(this.requireWorkspace(), libraryId);
        await this.putWorkspace(created.workspace, versionBeforeCheck);
        const event = this.requireWorkspace().bootstrap.libraries
            .flatMap((library) => library.events)
            .find((candidate) => candidate.id === created.eventId);
        if (!event)
            throw new Error(`Created Event ${created.eventId} is missing after source save.`);
        return cloneValue(event);
    }
    /**
     * Append a new FCPXML Project to the selected Event and compile the
     * complete library. The editor then selects the new structural ref.
     *
     * Main callers: BladeworksEditorApp.createNewProject (Cmd-N / library menu).
     */
    async createProjectNow(eventId) {
        const versionBeforeCheck = this.requireSourceVersion();
        await this.loadSource();
        if (this.requireSourceVersion() !== versionBeforeCheck) {
            throw new Error("Source changed on disk. The library was reloaded; review it before creating a Project.");
        }
        const workspace = this.requireWorkspace();
        const templateId = this.activeProjectId ?? workspace.bootstrap.activeProjectId;
        const created = addProjectToFCPXML(workspace, eventId, templateId);
        await this.putWorkspace(created.workspace, versionBeforeCheck);
        const project = this.requireWorkspace().bootstrap.projects[created.projectId];
        if (!project) {
            throw new Error(`Created Project ${created.projectId} is missing after source save.`);
        }
        this.activeProjectId = created.projectId;
        await this.loadCompatibility(created.projectId);
        return cloneValue(project);
    }
    async commitEdit(request) {
        return this.commitEditSequence({
            projectId: request.projectId,
            baseRevision: request.baseRevision,
            label: request.label,
            operations: [request.operation],
        });
    }
    async commitEditSequence(request) {
        return this.enqueueMutation(() => this.commitEditSequenceNow(request));
    }
    async commitEditSequenceNow(request) {
        const versionBeforeCheck = this.requireSourceVersion();
        await this.loadSource();
        if (this.requireSourceVersion() !== versionBeforeCheck) {
            throw new Error("Source changed on disk. The library was reloaded; review it before retrying the edit.");
        }
        const workspace = this.requireWorkspace();
        let project = workspace.bootstrap.projects[request.projectId];
        if (!project) {
            throw new Error(`Project ${request.projectId} no longer exists in the current library.`);
        }
        const editability = this.projectEditability(request.projectId);
        if (!editability.editable) {
            throw new Error(`Project is read-only: ${editability.reasons.join("; ")}`);
        }
        for (const operation of request.operations) {
            project = applyEdit(project, operation);
        }
        const updated = replaceProjectInFCPXML(workspace, project);
        await this.putWorkspace(updated, versionBeforeCheck);
        const canonical = this.requireWorkspace().bootstrap.projects[request.projectId];
        if (!canonical) {
            throw new Error("Accepted source replacement did not contain the edited Project.");
        }
        this.activeProjectId = request.projectId;
        await this.loadCompatibility(request.projectId);
        return { ...cloneValue(canonical), revision: request.baseRevision + 1 };
    }
    /**
     * Replace one Project wholesale from a caller-supplied full snapshot (undo,
     * redo, and the complex direct edits that cannot be expressed as operations:
     * nested-scope, container, semantic-audio, output-layout).
     *
     * Unlike `commitEditSequence`, this does NOT re-derive from current state — it
     * applies exactly the snapshot handed in. That makes it vulnerable to a
     * lost-update: if two direct edits are started from the same base before the
     * first save lands, both snapshots are built on the old base, and the second
     * would overwrite the first accepted edit. To prevent that we capture the
     * source version *synchronously here*, at the instant the caller derived the
     * snapshot, so it names the snapshot's true base. `restoreProjectNow` then
     * rejects the write if a concurrent edit queued ahead of it has advanced the
     * source version in the meantime. Capturing inside the serialized mutation
     * (as the old code did) is too late: the prior edit has already run and moved
     * the version, so the stale snapshot passes the check and clobbers it.
     */
    async restoreProject(project, baseRevision) {
        const baseVersion = this.requireSourceVersion();
        return this.enqueueMutation(() => this.restoreProjectNow(project, baseVersion, baseRevision));
    }
    async restoreProjectNow(project, baseVersion, baseRevision) {
        if (this.requireSourceVersion() !== baseVersion) {
            throw new Error("This Project changed since the edit began — a newer change was accepted first. " +
                "The edit was not applied so that change is not discarded; reopen the Project and redo it.");
        }
        const versionBeforeCheck = this.requireSourceVersion();
        await this.loadSource();
        if (this.requireSourceVersion() !== versionBeforeCheck) {
            throw new Error("Source changed on disk. The library was reloaded before restore.");
        }
        const updated = replaceProjectInFCPXML(this.requireWorkspace(), project);
        await this.putWorkspace(updated, versionBeforeCheck);
        const canonical = this.requireWorkspace().bootstrap.projects[project.id];
        if (!canonical) {
            throw new Error("Accepted source replacement did not contain the restored Project.");
        }
        // Refresh renderer compatibility (degraded flag, warnings) for the restored
        // Project, matching commitEditSequence, undo, redo, and selection. Without
        // this the restored Project keeps stale compatibility state until the next
        // operation happens to reload it.
        await this.loadCompatibility(project.id);
        return { ...cloneValue(canonical), revision: baseRevision + 1 };
    }
    async undoProject(projectId) {
        return this.enqueueMutation(() => this.moveHistory("undo", projectId));
    }
    async redoProject(projectId) {
        return this.enqueueMutation(() => this.moveHistory("redo", projectId));
    }
    historyState() {
        return this.history;
    }
    projectEditability(projectId) {
        return this.access.get(projectId) ?? { editable: false, reasons: ["Project has not been loaded."], degraded: false, warnings: [] };
    }
    async refreshMedia() {
        return this.loadMedia(true);
    }
    mediaForEvent(eventId) {
        return cloneValue(this.assembledAssets(eventId));
    }
    async setMediaFavorite(eventId, assetId, favorite) {
        return this.enqueueMutation(async () => {
            const versionBeforeCheck = this.requireSourceVersion();
            await this.loadSource();
            if (this.requireSourceVersion() !== versionBeforeCheck) {
                throw new Error("Source changed on disk. The library was reloaded before updating the Favorite.");
            }
            await this.putWorkspace(setEventAssetFavorite(this.requireWorkspace(), eventId, assetId, favorite, this.assembledBootstrap(this.activeProjectId ?? this.requireWorkspace().bootstrap.activeProjectId).assets
                .find((asset) => asset.id === assetId)), versionBeforeCheck);
            return this.mediaForEvent(eventId);
        });
    }
    mediaInventoryFailures() {
        return [...this.mediaFailures];
    }
    async loadMedia(refresh) {
        const response = await requireOk(await this.request(refresh ? "/api/editor/media/refresh" : "/api/editor/media", refresh ? { method: "POST" } : {}), refresh ? "Media refresh" : "Media inventory");
        const payload = await response.json();
        this.mediaAssets = payload.items.map(mediaAsset);
        this.mediaFailures = (payload.failures ?? []).map((failure) => {
            const path = typeof failure.relativePath === "string" && failure.relativePath
                ? failure.relativePath
                : "unknown media file";
            const message = typeof failure.message === "string" && failure.message
                ? failure.message
                : "could not be probed";
            return `${path}: ${message}`;
        });
        if (this.workspace) {
            const active = this.activeProjectId ?? this.workspace.bootstrap.activeProjectId;
            return this.assembledBootstrap(active).assets;
        }
        return cloneValue(this.mediaAssets);
    }
    async importMedia(file) {
        await requireOk(await this.request("/api/editor/media/upload", {
            method: "POST",
            headers: {
                "Content-Type": "application/octet-stream",
                "X-Bladeworks-Filename": encodeURIComponent(file.name),
            },
            body: file,
        }), "Media upload");
        return this.loadMedia(false);
    }
    async loadMediaVisual(request) {
        const response = await requireOk(await this.request("/api/editor/media/visuals", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(request),
        }), "Timeline media sample");
        const payload = await response.json();
        return {
            thumbnails: [...payload.thumbnails],
            audioBands: [...payload.audioBands],
            hasVideo: payload.hasVideo,
            hasAudio: payload.hasAudio,
        };
    }
    async attachPreview(canvas, video, onTimeUpdate, onPlayingChange, quality, onRuntimeEvent) {
        if (PREVIEW_TRANSPORT === "rawframe") {
            return this.attachRawPreview(canvas, video, onTimeUpdate, onPlayingChange, quality, onRuntimeEvent);
        }
        return this.attachWebRTCPreview(canvas, video, onTimeUpdate, onPlayingChange, quality, onRuntimeEvent);
    }
    /**
     * Main preview path: create a raw-frame session and stream uncompressed
     * frames over a WebSocket, painting them into the canvas. See
     * RawFrameTransport and backend/preview/rawframe.py.
     */
    async attachRawPreview(canvas, video, onTimeUpdate, onPlayingChange, quality, onRuntimeEvent) {
        const identity = this.previewIdentity();
        const response = await requireOk(await this.request("/api/editor/preview/sessions/raw", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ...identity, playhead: 0, quality: previewQualityPayload(quality) }),
        }), "Raw preview session");
        const result = await response.json();
        // The canvas is the render surface for this path. Clear any stale inline
        // display so app.ts can drive visibility through the `hidden` attribute.
        video.style.display = "";
        canvas.style.display = "";
        canvas.classList.add("preview-frame-canvas");
        const transport = new RawFrameTransport({
            url: this.streamUrl(result.streamUrl),
            token: this.token,
            canvas,
            warmup: typeof document !== "undefined" ? document.getElementById("viewer-warmup") : null,
            onError: (message) => onRuntimeEvent?.({ type: "error", message }),
        });
        const controller = new LocalhostPreviewController({
            transport,
            sessionId: result.sessionId,
            eventsUrl: result.eventsUrl,
            request: (path, init) => this.request(path, init),
            identity: () => this.previewIdentity(),
            beforePlay: async () => {
                const changed = await this.refreshSourceAtBoundary();
                if (changed) {
                    onRuntimeEvent?.({ type: "source_changed", message: "Source reloaded from disk before playback." });
                }
            },
            onTimeUpdate,
            onPlayingChange,
            ...(onRuntimeEvent ? { onRuntimeEvent } : {}),
            initialIdentity: identity,
            initialQuality: quality,
            initialPlayhead: 0,
        });
        // Paint the paused first frame immediately, before any Play gesture. The
        // still is queued by the backend and released as soon as the WebSocket
        // finishes connecting.
        controller.seek(0);
        return controller;
    }
    /** Convert a preview stream API path into a same-origin ws:// / wss:// URL. */
    streamUrl(streamPath) {
        const url = new URL(streamPath, this.baseUrl);
        url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
        return url.toString();
    }
    /** Quarantined legacy transport. Requires BLADEFRAME_PREVIEW_WEBRTC=1. */
    async attachWebRTCPreview(canvas, video, onTimeUpdate, onPlayingChange, quality, onRuntimeEvent) {
        const identity = this.previewIdentity();
        const peer = new RTCPeerConnection();
        peer.addTransceiver("video", { direction: "recvonly" });
        peer.addTransceiver("audio", { direction: "recvonly" });
        const stream = new MediaStream();
        // Muted autoplay is permitted during boot and is necessary for paused seek
        // frames to become visible before the user has clicked Play. The controller
        // unmutes synchronously from the later Play gesture.
        video.autoplay = true;
        video.muted = true;
        peer.ontrack = (event) => {
            stream.addTrack(event.track);
            video.srcObject = stream;
            void video.play().catch(() => undefined);
        };
        const offer = await peer.createOffer();
        await peer.setLocalDescription(offer);
        await this.waitForIce(peer);
        const response = await requireOk(await this.request("/api/editor/preview/sessions", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                ...identity,
                playhead: 0,
                offer: peer.localDescription,
                quality: previewQualityPayload(quality),
            }),
        }), "WebRTC preview negotiation");
        const result = await response.json();
        await peer.setRemoteDescription(result.answer);
        canvas.style.display = "none";
        video.style.display = "block";
        return new LocalhostPreviewController({
            transport: new WebRTCTransport(peer, video),
            sessionId: result.sessionId,
            eventsUrl: result.eventsUrl,
            request: (path, init) => this.request(path, init),
            identity: () => this.previewIdentity(),
            beforePlay: async () => {
                const changed = await this.refreshSourceAtBoundary();
                if (changed) {
                    onRuntimeEvent?.({ type: "source_changed", message: "Source reloaded from disk before playback." });
                }
            },
            onTimeUpdate,
            onPlayingChange,
            ...(onRuntimeEvent ? { onRuntimeEvent } : {}),
            initialIdentity: identity,
            initialQuality: quality,
            initialPlayhead: 0,
        });
    }
    /**
     * Liveness probe for the topbar status light. Hits the backend's `/healthz`
     * and reports whether it answered OK. Any transport failure (backend down,
     * connection refused) resolves to false rather than throwing, so the caller
     * can simply flip the light red.
     */
    async probeHealth() {
        try {
            const response = await this.request("/healthz");
            return response.ok;
        }
        catch {
            return false;
        }
    }
    async renderProject(projectId, options = { profile: "delivery", resolution: 1080 }, lifecycle = {}) {
        await this.refreshSourceAtBoundary();
        if (!this.requireWorkspace().bootstrap.projects[projectId]) {
            throw new Error(`Project ${projectId} no longer exists in the current library.`);
        }
        this.activeProjectId = projectId;
        const response = await requireOk(await this.request("/api/editor/render", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                ...this.previewIdentity(),
                resolution: `${options.resolution}p`,
                profile: options.profile,
            }),
        }), "Project render");
        let job = await response.json();
        lifecycle.onProgress?.(job);
        let cancellationSent = false;
        while (["queued", "running", "cancelling"].includes(job.status)) {
            if (lifecycle.signal?.aborted && !cancellationSent) {
                cancellationSent = true;
                const cancellation = await requireOk(await this.request(`/api/editor/renders/${job.jobId}`, { method: "DELETE" }), "Cancel export");
                job = await cancellation.json();
                lifecycle.onProgress?.(job);
            }
            await new Promise((resolve) => setTimeout(resolve, 250));
            const poll = await requireOk(await this.request(`/api/editor/renders/${job.jobId}`), "Render status");
            job = await poll.json();
            lifecycle.onProgress?.(job);
        }
        if (job.status === "cancelled")
            throw new DOMException("Export cancelled.", "AbortError");
        if (job.status !== "completed" || !job.artifact) {
            throw new Error(job.error?.message ?? `Render ended with status ${job.status}.`);
        }
        const link = document.createElement("a");
        link.href = new URL(job.artifact.url, this.baseUrl).toString();
        link.download = job.artifact.fileName;
        link.click();
        return { message: `${options.resolution}p ${options.profile} render completed and downloaded.` };
    }
    async moveHistory(direction, projectId) {
        const response = await requireOk(await this.request(`/api/editor/source/${direction}`, { method: "POST" }), direction === "undo" ? "Undo" : "Redo");
        this.adoptStatus(await response.json());
        await this.loadSource();
        const workspace = this.requireWorkspace();
        const selected = workspace.bootstrap.projects[projectId]
            ? projectId
            : Object.keys(workspace.bootstrap.projects)[0];
        if (!selected) {
            throw new Error("Restored library does not contain a Project.");
        }
        this.activeProjectId = selected;
        await this.loadCompatibility(selected);
        return cloneValue(workspace.bootstrap.projects[selected]);
    }
    async putWorkspace(workspace, expectedVersion) {
        const expectedContentVersion = await sha256Version(workspace.xml);
        const response = await this.request("/api/editor/source", {
            method: "PUT",
            headers: {
                "Content-Type": "application/xml",
                "If-Match": `"${expectedVersion}"`,
            },
            body: workspace.xml,
        });
        if (response.status === 409) {
            await this.loadSource();
        }
        await requireOk(response, "Source save");
        const status = await response.json();
        const returnedVersion = status.version ?? status.diskVersion ?? etagVersion(response);
        if (expectedContentVersion && returnedVersion !== expectedContentVersion) {
            throw new Error(`Bladeworks accepted source as ${returnedVersion}, but submitted XML hashes to ${expectedContentVersion}.`);
        }
        this.workspace = workspace;
        this.sourceVersion = returnedVersion;
        this.adoptWorkspaceAccess(workspace);
        this.adoptStatus(status);
    }
    async loadSource() {
        const response = await requireOk(await this.request("/api/editor/source"), "Source load");
        const xml = await response.text();
        const workspace = parseFCPXMLLibrary(xml);
        this.workspace = workspace;
        this.sourceVersion = etagVersion(response);
        this.adoptWorkspaceAccess(workspace);
        const historyIndex = Number(response.headers.get("X-Bladeworks-History-Index"));
        const historyLength = Number(response.headers.get("X-Bladeworks-History-Length"));
        if (Number.isInteger(historyIndex) && Number.isInteger(historyLength) && historyLength > 0) {
            this.adoptStatus({
                diskVersion: this.sourceVersion,
                loadedVersion: this.sourceVersion,
                compileStatus: response.headers.get("X-Bladeworks-Compile-Status") ?? "ready",
                degraded: false,
                historyIndex,
                historyLength,
            });
        }
    }
    async refreshSourceAtBoundary() {
        await this.mutationTail;
        const previous = this.sourceVersion;
        await this.loadSource();
        if (previous && previous !== this.sourceVersion && this.activeProjectId && !this.requireWorkspace().bootstrap.projects[this.activeProjectId]) {
            this.activeProjectId = Object.keys(this.requireWorkspace().bootstrap.projects)[0] ?? null;
        }
        if (this.activeProjectId) {
            await this.loadCompatibility(this.activeProjectId);
        }
        return previous !== null && previous !== this.sourceVersion;
    }
    /**
     * Run source mutations in browser gesture order.
     *
     * A blur commit and the next control change can arrive in the same event
     * turn. Both must apply to the latest complete-library XML; otherwise the
     * second PUT can replace the first edit with an older workspace snapshot.
     * Rejections are isolated so one failed save does not poison later work.
     */
    enqueueMutation(mutation) {
        const result = this.mutationTail.then(mutation);
        this.mutationTail = result.then(() => undefined, () => undefined);
        return result;
    }
    async loadCompatibility(projectId) {
        const parameters = new URLSearchParams({
            sourceVersion: this.requireSourceVersion(),
            projectRef: projectId,
        });
        const response = await requireOk(await this.request(`/api/editor/compatibility?${parameters.toString()}`), "Project compatibility");
        const payload = await response.json();
        const existing = this.projectEditability(projectId);
        const warnings = (payload.compatibility?.findings ?? [])
            .filter((finding) => finding.outcome !== "exact" && finding.outcome !== "info")
            .map((finding) => `${finding.construct ?? "FCPXML construct"}: ${finding.disposition ?? finding.outcome ?? "degraded"}`);
        this.access.set(projectId, { ...existing, degraded: payload.degraded === true, warnings });
    }
    adoptWorkspaceAccess(workspace) {
        this.access.clear();
        for (const projectId of Object.keys(workspace.bootstrap.projects)) {
            const result = workspace.editableProjects[projectId];
            this.access.set(projectId, result
                ? { editable: result.editable, reasons: [...result.reasons], degraded: false, warnings: [] }
                : { editable: false, reasons: ["The FCPXML codec did not report this Project's editability."], degraded: false, warnings: [] });
        }
    }
    adoptStatus(status) {
        this.history = {
            canUndo: status.historyIndex > 0,
            canRedo: status.historyIndex < status.historyLength - 1,
            index: status.historyIndex,
            length: status.historyLength,
        };
    }
    assembledBootstrap(activeProjectId) {
        const bootstrap = this.requireWorkspace().bootstrap;
        const eventId = bootstrap.projects[activeProjectId]?.eventId;
        return cloneValue({
            ...bootstrap,
            assets: this.assembledAssets(eventId),
            activeProjectId,
        });
    }
    assembledAssets(eventId) {
        const bootstrap = this.requireWorkspace().bootstrap;
        const known = new Set(bootstrap.assets.map((asset) => asset.id));
        const knownPaths = new Set(bootstrap.assets.flatMap((asset) => asset.sourcePath ? [asset.sourcePath] : []));
        const inventory = this.mediaAssets.filter((asset) => !known.has(asset.id) && (!asset.sourcePath || !knownPaths.has(asset.sourcePath)));
        const favorites = eventId ? favoriteAssetIdsForEvent(this.requireWorkspace(), eventId) : new Set();
        return [...bootstrap.assets, ...inventory].map((asset) => ({
            ...asset,
            favorite: favorites.has(asset.id),
        }));
    }
    previewIdentity() {
        if (!this.activeProjectId) {
            throw new Error("Cannot preview before a Project is selected.");
        }
        return { sourceVersion: this.requireSourceVersion(), projectRef: this.activeProjectId };
    }
    requireWorkspace() {
        if (!this.workspace) {
            throw new Error("The FCPXML library has not been loaded.");
        }
        return this.workspace;
    }
    requireSourceVersion() {
        if (!this.sourceVersion) {
            throw new Error("The FCPXML source version is unavailable.");
        }
        return this.sourceVersion;
    }
    async request(path, init = {}) {
        const headers = new Headers(init.headers);
        headers.set("Authorization", `Bearer ${this.token}`);
        return this.fetcher(new URL(path, this.baseUrl), { ...init, headers });
    }
    async waitForIce(peer) {
        if (peer.iceGatheringState === "complete") {
            return;
        }
        await new Promise((resolve) => {
            const listener = () => {
                if (peer.iceGatheringState === "complete") {
                    peer.removeEventListener("icegatheringstatechange", listener);
                    resolve();
                }
            };
            peer.addEventListener("icegatheringstatechange", listener);
        });
    }
}
export function runtimeFromLocation(location = window.location) {
    const parameters = new URLSearchParams(location.search);
    if (parameters.get("runtime") === "localhost") {
        const fragment = new URLSearchParams(location.hash.replace(/^#/, ""));
        const inBrowser = typeof window !== "undefined" && location === window.location;
        const token = fragment.get("runtimeToken")
            ?? (inBrowser ? sessionStorage.getItem("bladeworks.runtimeToken") : null);
        if (!token) {
            throw new Error("Bladeworks Studio URL is missing its runtimeToken fragment.");
        }
        if (inBrowser) {
            sessionStorage.setItem("bladeworks.runtimeToken", token);
            window.history?.replaceState(null, "", `${location.pathname}${location.search}`);
        }
        const baseUrl = parameters.get("runtimeBase") ?? location.origin;
        return new LocalhostEditorRuntime(baseUrl, token);
    }
    return new MockEditorRuntime();
}
