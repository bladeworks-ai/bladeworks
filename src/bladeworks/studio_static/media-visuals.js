/**
 * Memory-only Studio filmstrip and waveform decoration.
 *
 * Architecture map:
 * visible DOM placeholder -> two-request worker queue -> authenticated runtime
 * request -> bounded LRU result -> real JPEG frames and normalized audio bars.
 *
 * Product rules:
 * - At most two media decodes run concurrently, keeping interaction responsive.
 * - At most 32 clip-range results remain in browser memory.
 * - Re-rendered DOM reuses results but no result survives a page reload.
 * - Missing or undecodable files get a visible warning instead of synthetic art.
 */
const MAX_CACHE_ENTRIES = 32;
const MAX_CONCURRENT = 2;
const MAX_PENDING = 32;
const MAX_DEFERRED = 64;
export class MediaVisualLoader {
    runtime;
    cache = new Map();
    inFlight = new Map();
    queue = [];
    deferred = new Set();
    observer;
    observed = new Set();
    active = 0;
    constructor(runtime) {
        this.runtime = runtime;
        this.observer = typeof IntersectionObserver === "undefined"
            ? null
            : new IntersectionObserver((entries) => {
                for (const entry of entries) {
                    if (!entry.isIntersecting)
                        continue;
                    const element = entry.target;
                    this.observer?.unobserve(element);
                    this.observed.delete(element);
                    this.schedule(element);
                }
            }, { rootMargin: "120px" });
    }
    /** Decorate every newly rendered placeholder without blocking UI rendering. */
    decorate(root) {
        if (this.runtime.mode !== "localhost")
            return;
        for (const previous of this.observed) {
            if (previous.isConnected)
                continue;
            this.observer?.unobserve(previous);
            this.observed.delete(previous);
        }
        for (const element of root.querySelectorAll("[data-media-visual]")) {
            if (element.dataset.mediaVisualState)
                continue;
            const request = requestFromElement(element);
            if (!request) {
                markUnavailable(element, "Media path unavailable");
                continue;
            }
            element.dataset.mediaVisualState = "waiting";
            if (this.observer) {
                this.observer.observe(element);
                this.observed.add(element);
            }
            else {
                this.schedule(element);
            }
        }
    }
    schedule(element) {
        if (!element.isConnected || element.dataset.mediaVisualState === "loading")
            return;
        const request = requestFromElement(element);
        if (!request) {
            markUnavailable(element, "Media path unavailable");
            return;
        }
        const key = requestKey(request);
        if (!this.cache.has(key) && !this.inFlight.has(key) && this.inFlight.size >= MAX_PENDING) {
            if (this.deferred.size >= MAX_DEFERRED) {
                markUnavailable(element, "Too many visible media samples; scroll and try again.");
            }
            else {
                element.dataset.mediaVisualState = "waiting";
                this.deferred.add(element);
            }
            return;
        }
        element.dataset.mediaVisualState = "loading";
        void this.load(request, element).then((result) => applyResult(element, result), (error) => markUnavailable(element, error instanceof Error ? error.message : "Media could not be decoded"));
    }
    load(request, element) {
        const key = requestKey(request);
        const cached = this.cache.get(key);
        if (cached) {
            this.cache.delete(key);
            this.cache.set(key, cached);
            return Promise.resolve(cached);
        }
        const existing = this.inFlight.get(key);
        if (existing) {
            existing.consumers.add(element);
            return existing.promise;
        }
        let resolve;
        let reject;
        const pending = new Promise((accept, decline) => {
            resolve = accept;
            reject = decline;
        });
        const consumers = new Set([element]);
        this.inFlight.set(key, { promise: pending, consumers });
        void pending.then((result) => {
            this.cache.set(key, result);
            while (this.cache.size > MAX_CACHE_ENTRIES) {
                const oldest = this.cache.keys().next().value;
                if (oldest === undefined)
                    break;
                this.cache.delete(oldest);
            }
        }).finally(() => {
            this.inFlight.delete(key);
            consumers.clear();
            this.drainDeferred();
        }).catch(() => undefined);
        this.queue.push({ request, consumers, resolve, reject });
        this.pump();
        return pending;
    }
    pump() {
        while (this.active < MAX_CONCURRENT && this.queue.length) {
            const entry = this.queue.shift();
            if (![...entry.consumers].some((element) => element.isConnected)) {
                entry.reject(new Error("Media visual is no longer visible."));
                continue;
            }
            this.active += 1;
            void this.runtime.loadMediaVisual(entry.request).then(entry.resolve, entry.reject).finally(() => {
                this.active -= 1;
                this.pump();
            });
        }
    }
    drainDeferred() {
        for (const element of this.deferred) {
            this.deferred.delete(element);
            if (!element.isConnected)
                continue;
            this.schedule(element);
            if (this.inFlight.size >= MAX_PENDING)
                break;
        }
    }
}
function requestKey(request) {
    return JSON.stringify(request);
}
function requestFromElement(element) {
    const relativePath = element.dataset.mediaPath;
    if (!relativePath)
        return null;
    const numeric = (name) => Number(element.dataset[name]);
    const request = {
        relativePath,
        start: numeric("mediaStart"),
        duration: numeric("mediaDuration"),
        thumbnailCount: numeric("mediaThumbnails"),
        thumbnailWidth: 96,
        audioBands: numeric("mediaAudioBands"),
    };
    return Object.values(request).some((value) => typeof value === "number" && !Number.isFinite(value)) ? null : request;
}
function applyResult(element, result) {
    if (!element.isConnected)
        return;
    const filmstrip = element.querySelector(".clip-filmstrip, .connected-filmstrip, .asset-filmstrip");
    if (filmstrip && result.thumbnails.length) {
        filmstrip.replaceChildren(...result.thumbnails.map((source) => {
            const image = document.createElement("img");
            image.src = source;
            image.alt = "";
            image.draggable = false;
            return image;
        }));
    }
    else if (filmstrip) {
        filmstrip.replaceChildren();
    }
    const waveform = element.querySelector(".clip-audio-wave, .audio-wave, .browser-waveform");
    if (waveform && result.audioBands.length) {
        waveform.replaceChildren(...result.audioBands.map((peak) => {
            const bar = document.createElement("i");
            bar.style.height = `${audioBandHeightPercent(peak)}%`;
            return bar;
        }));
    }
    else if (waveform) {
        waveform.replaceChildren();
    }
    element.dataset.mediaVisualState = "ready";
}
export function audioBandHeightPercent(peak) {
    const fullScalePeak = Math.min(1, Math.max(0, peak));
    if (fullScalePeak === 0)
        return 2;
    // Audio level is perceived logarithmically. Mapping the lower 60 dB into
    // the lane keeps ordinary dialogue visibly detailed without normalizing
    // every clip to its own peak, so real level differences remain truthful.
    const decibels = 20 * Math.log10(fullScalePeak);
    const visibleRange = Math.min(1, Math.max(0, (decibels + 60) / 60));
    return Math.max(2, Math.round(visibleRange * 1000) / 10);
}
function markUnavailable(element, message) {
    if (!element.isConnected)
        return;
    element.dataset.mediaVisualState = "unavailable";
    element.title = message;
    element.querySelector(".clip-filmstrip, .connected-filmstrip, .asset-filmstrip")?.replaceChildren();
    element.querySelector(".clip-audio-wave, .audio-wave, .browser-waveform")?.replaceChildren();
    const warning = document.createElement("span");
    warning.className = "media-visual-warning";
    warning.textContent = "⚠ Media unavailable";
    element.append(warning);
}
