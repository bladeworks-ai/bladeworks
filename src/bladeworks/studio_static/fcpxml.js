/**
 * Complete-library FCPXML projection and mutation boundary.
 *
 * Architecture map:
 * complete Info.fcpxml text
 *   -> hardened browser XML parse
 *   -> Library/Event/Project catalog with structural Project refs
 *   -> editable ProjectSnapshot projections
 *   -> clone the original XML DOM and rewrite one selected Project
 *   -> append a new empty Project to a selected Event
 *   -> complete serialized Info.fcpxml text
 *
 * Product invariants:
 * - Project refs are one-based, tag-specific structural addresses.
 * - A ProjectSnapshot ID is exactly its Project ref.
 * - The complete library is always returned after an edit.
 * - Unselected Projects and shared resources are never reconstructed.
 * - A Project is read-only when its active timeline cannot be represented
 *   without losing editorial meaning.
 * - Changed times are quantized to the Project frame duration. Unchanged
 *   rational strings are retained exactly.
 *
 * Why this exists:
 * The browser reducer works with a small, friendly object model, while Final
 * Cut stores a complete library in XML. Keeping that translation in one module
 * prevents UI state from becoming a second source of truth.
 */
import { defaultAudio, defaultDistort, defaultTransform, defaultVideo, } from "./clip-state.js";
import { projectDuration } from "./magnetic-timeline.js";
const workspaceOrigins = new WeakMap();
const TIMELINE_TAGS = new Set([
    "asset-clip", "video", "audio", "title", "caption", "gap",
    "ref-clip", "mc-clip", "sync-clip", "audition",
]);
const MARKER_TAGS = new Set(["marker", "chapter-marker", "todo-marker"]);
const UNSAFE_ACTIVE_TAGS = new Set([
    "clip",
]);
const PASSIVE_CLIP_TAGS = new Set([
    "adjust-blend",
    "adjust-conform",
    "adjust-corners",
    "adjust-crop",
    "adjust-colorConform",
    "adjust-loudness",
    "adjust-noiseReduction",
    "adjust-panner",
    "adjust-rollingShutter",
    "adjust-stabilization",
    "adjust-transform",
    "adjust-volume",
    "adjust-voiceIsolation",
    "analysis-marker",
    "audio-channel-source",
    "audio-role-source",
    "chapter-marker",
    "conform-rate",
    "fadeIn",
    "fadeOut",
    "filter-audio",
    "filter-video",
    "filter-video-mask",
    "keyword",
    "marker",
    "metadata",
    "mc-source",
    "mute",
    "note",
    "param",
    "rating",
    "text",
    "text-style-def",
    "timeMap",
    "todo-marker",
    "sync-source",
]);
const EPSILON = 1e-8;
const BASIC_TITLE_UID = ".../Titles.localized/Bumper:Opener.localized/Basic Title.localized/Basic Title.moti";
const CUSTOM_SOLID_UID = ".../Generators.localized/Solids.localized/Custom.localized/Custom.motn";
const CUSTOM_SOLID_COLOR_KEY = "9999/10008/10006/2/1/1";
const BLEND_MODES = new Map([
    "normal", "behind", "add", "subtract", "darken", "lighten", "multiply", "screen",
    "overlay", "soft-light", "hard-light", "difference", "exclusion", "color-burn",
    "color-dodge", "divide", "linear-light", "pin-light", "hard-mix", "stencil-alpha",
    "silhouette-alpha", "stencil-luma", "silhouette-luma",
].map((mode) => [mode.replace(/[^a-z0-9]/g, ""), mode]));
function supportedBlendMode(raw) {
    if (!raw) {
        return "normal";
    }
    const mode = BLEND_MODES.get(raw.toLowerCase().replace(/[^a-z0-9]/g, ""));
    if (!mode) {
        throw new Error(`Unsupported Bladeworks blend mode ${JSON.stringify(raw)}.`);
    }
    return mode;
}
function tagName(node) {
    return node.localName || node.tagName.split(":").at(-1) || node.tagName;
}
function childElements(parent) {
    const result = [];
    for (let index = 0; index < parent.childNodes.length; index += 1) {
        const child = parent.childNodes.item(index);
        if (child?.nodeType === 1) {
            result.push(child);
        }
    }
    return result;
}
function childrenNamed(parent, name) {
    return childElements(parent).filter((child) => tagName(child) === name);
}
function firstChild(parent, name) {
    return childrenNamed(parent, name)[0] ?? null;
}
/** Find a Final Cut audio fade in either supported volume-adjustment location. */
function findAudioFade(volume, name) {
    if (!volume)
        return null;
    const amountParameters = childrenNamed(volume, "param").filter((parameter) => parameterIdentity(parameter).toLowerCase() === "amount");
    for (const container of [volume, ...amountParameters]) {
        const fade = firstChild(container, name);
        if (fade)
            return fade;
    }
    return null;
}
function descendantsNamed(parent, name) {
    const result = [];
    const visit = (node) => {
        for (const child of childElements(node)) {
            if (tagName(child) === name) {
                result.push(child);
            }
            visit(child);
        }
    };
    visit(parent);
    return result;
}
function requireAttribute(element, name, context) {
    const value = element.getAttribute(name);
    if (value === null || value.length === 0) {
        throw new Error(`${context} is missing required ${name}.`);
    }
    return value;
}
function gcd(left, right) {
    let a = left < 0n ? -left : left;
    let b = right < 0n ? -right : right;
    while (b !== 0n) {
        const remainder = a % b;
        a = b;
        b = remainder;
    }
    return a || 1n;
}
function reduceRational(value) {
    if (value.denominator === 0n) {
        throw new Error("FCPXML time denominator cannot be zero.");
    }
    const sign = value.denominator < 0n ? -1n : 1n;
    const numerator = value.numerator * sign;
    const denominator = value.denominator * sign;
    const divisor = gcd(numerator, denominator);
    return { numerator: numerator / divisor, denominator: denominator / divisor };
}
function parseRationalTime(raw, context) {
    const match = /^([+-]?\d+)(?:\/(\d+))?s$/.exec(raw.trim());
    if (!match?.[1]) {
        throw new Error(`${context} has invalid FCPXML time ${JSON.stringify(raw)}.`);
    }
    return reduceRational({
        numerator: BigInt(match[1]),
        denominator: BigInt(match[2] ?? "1"),
    });
}
function rationalSeconds(value) {
    return Number(value.numerator) / Number(value.denominator);
}
function addRational(left, right) {
    return reduceRational({
        numerator: (left.numerator * right.denominator) + (right.numerator * left.denominator),
        denominator: left.denominator * right.denominator,
    });
}
function subtractRational(left, right) {
    return reduceRational({
        numerator: (left.numerator * right.denominator) - (right.numerator * left.denominator),
        denominator: left.denominator * right.denominator,
    });
}
function sameRationalTime(left, right, context) {
    const a = parseRationalTime(left, context);
    const b = parseRationalTime(right, context);
    return a.numerator === b.numerator && a.denominator === b.denominator;
}
function seconds(raw, fallback, context) {
    return raw === null ? fallback : rationalSeconds(parseRationalTime(raw, context));
}
function rationalString(value) {
    const reduced = reduceRational(value);
    if (reduced.numerator === 0n) {
        return "0s";
    }
    if (reduced.denominator === 1n) {
        return `${reduced.numerator}s`;
    }
    return `${reduced.numerator}/${reduced.denominator}s`;
}
function quantizedTime(value, frameDuration) {
    if (!Number.isFinite(value)) {
        throw new Error(`Cannot serialize non-finite timeline time ${value}.`);
    }
    const frameSeconds = rationalSeconds(frameDuration);
    const frames = BigInt(Math.round(value / frameSeconds));
    return rationalString({
        numerator: frames * frameDuration.numerator,
        denominator: frameDuration.denominator,
    });
}
function preservedOrQuantized(value, originalValue, originalRaw, frameDuration) {
    return Math.abs(value - originalValue) <= EPSILON
        ? originalRaw
        : quantizedTime(value, frameDuration);
}
function finiteNumber(raw, fallback) {
    if (raw === null || raw.trim() === "") {
        return fallback;
    }
    const value = Number(raw.replace(/dB$/, ""));
    return Number.isFinite(value) ? value : fallback;
}
function rationalTime(raw, context) {
    const exact = raw ?? "0s";
    return { seconds: seconds(exact, 0, context), raw: exact };
}
function authoredTime(value, frameDuration) {
    try {
        if (Math.abs(seconds(value.raw, Number.NaN, "authored rational time") - value.seconds) <= EPSILON) {
            return value.raw;
        }
    }
    catch {
        // A UI edit may intentionally replace the exact raw value. Quantize it below.
    }
    return quantizedTime(value.seconds, frameDuration);
}
function parseParameterValue(raw) {
    const value = raw ?? "";
    const numeric = Number(value.replace(/dB$/, ""));
    if (value.trim() !== "" && Number.isFinite(numeric)) {
        return numeric;
    }
    const coordinates = value.trim().split(/\s+/).map(Number);
    if (coordinates.length === 2 && coordinates.every(Number.isFinite)) {
        return { x: coordinates[0], y: coordinates[1] };
    }
    if (coordinates.length === 3 && coordinates.every(Number.isFinite)) {
        return { red: coordinates[0], green: coordinates[1], blue: coordinates[2] };
    }
    if (coordinates.length === 4 && coordinates.every(Number.isFinite)) {
        return { red: coordinates[0], green: coordinates[1], blue: coordinates[2], alpha: coordinates[3] };
    }
    return value;
}
function parameterValueString(value) {
    if (typeof value === "string" || typeof value === "number") {
        return String(value);
    }
    if (typeof value === "boolean") {
        return value ? "1" : "0";
    }
    if (Array.isArray(value)) {
        return value.join(" ");
    }
    const object = value;
    if ("x" in object) {
        return `${object.x} ${object.y}`;
    }
    if ("left" in object) {
        return `${object.left} ${object.top} ${object.right} ${object.bottom}`;
    }
    if ("red" in object) {
        return "alpha" in object
            ? `${object.red} ${object.green} ${object.blue} ${object.alpha}`
            : `${object.red} ${object.green} ${object.blue}`;
    }
    throw new Error("Effect parameter value has an unsupported shape.");
}
/**
 * Return the multiplier from Studio's normalized pan model to FCPXML units.
 *
 * Final Cut's canonical Stereo Left/Right modes store percentages, while the
 * Inspector and renderer use a normalized [-1, 1] value. Keeping this
 * conversion at the XML boundary prevents unrelated edits from clamping an
 * imported value such as -50 to -1 and then serializing the wrong amount.
 *
 * Main callers:
 * - ``parseTimelineElement`` and ``parseKeyframes`` while importing FCPXML.
 * - ``updateAudio`` and ``updateKeyframes`` while serializing a Project.
 */
function pannerXmlScale(panner) {
    const mode = panner?.getAttribute("mode")?.trim();
    return mode === "1" || mode === "1 (Stereo Left/Right)" ? 100 : 1;
}
function normalizedPannerValue(value, panner, context) {
    if (typeof value !== "number" || !Number.isFinite(value)) {
        throw new Error(`${context} must be a finite number.`);
    }
    return value / pannerXmlScale(panner);
}
function pannerValueForXml(value, panner, context) {
    if (typeof value !== "number" || !Number.isFinite(value)) {
        throw new Error(`${context} must be a finite number.`);
    }
    return value * pannerXmlScale(panner);
}
function parseParameterKeyframes(parameter, context) {
    const animation = firstChild(parameter, "keyframeAnimation");
    if (!animation) {
        return [];
    }
    return childrenNamed(animation, "keyframe").map((keyframe, index) => ({
        time: rationalTime(keyframe.getAttribute("time"), `${context} keyframe ${index + 1}`),
        value: parseParameterValue(keyframe.getAttribute("value")),
        interpolation: keyframe.getAttribute("interp") || "linear",
    }));
}
function parameterIdentity(parameter) {
    return parameter.getAttribute("key") || parameter.getAttribute("name") || "value";
}
function parseParameters(container) {
    const values = {};
    const names = {};
    const keyframes = {};
    for (const parameter of childrenNamed(container, "param")) {
        const key = parameterIdentity(parameter);
        if (Object.hasOwn(values, key)) {
            throw new Error(`FCPXML parameter key ${JSON.stringify(key)} is duplicated.`);
        }
        values[key] = parseParameterValue(parameter.getAttribute("value"));
        names[key] = parameter.getAttribute("name") || key;
        const frames = parseParameterKeyframes(parameter, key);
        if (frames.length > 0) {
            keyframes[key] = frames;
        }
    }
    return { values, names, keyframes };
}
function pair(raw, fallback) {
    if (raw === null) {
        return [...fallback];
    }
    // Same token rule as backend core.model.pair: commas are whitespace.
    const values = raw.replace(/,/g, " ").trim().split(/\s+/).map(Number);
    if (values.length < 2 || !Number.isFinite(values[0]) || !Number.isFinite(values[1])) {
        return [...fallback];
    }
    return [values[0], values[1]];
}
function parseDocument(xml) {
    if (xml.length > 128 * 1024 * 1024) {
        throw new Error("FCPXML exceeds the 128 MiB browser parser limit.");
    }
    if (/<!ENTITY\b/i.test(xml)) {
        throw new Error("FCPXML entity declarations are not supported.");
    }
    const doctypes = [...xml.matchAll(/<!DOCTYPE\s+([^>]+)>/gi)];
    if (doctypes.some((match) => match[1]?.trim().toLowerCase() !== "fcpxml")) {
        throw new Error("Only the standard <!DOCTYPE fcpxml> declaration is supported.");
    }
    assertBalancedMarkup(xml);
    if (typeof DOMParser === "undefined") {
        throw new Error("This environment does not provide the browser DOMParser API.");
    }
    const parseErrors = [];
    const Parser = DOMParser;
    const parser = new Parser({
        onError: (level, message) => {
            if (level === "error" || level === "fatalError") {
                parseErrors.push(message);
            }
        },
        errorHandler: {
            warning: () => undefined,
            error: (message) => parseErrors.push(message),
            fatalError: (message) => parseErrors.push(message),
        },
    });
    const document = parser.parseFromString(xml, "application/xml");
    const parserErrors = document.getElementsByTagName("parsererror");
    if (parseErrors.length > 0 || parserErrors.length > 0 || tagName(document.documentElement) === "parsererror") {
        const detail = parseErrors[0] || parserErrors.item(0)?.textContent?.trim() || "unknown XML parser error";
        throw new Error(`Invalid FCPXML: ${detail}`);
    }
    if (tagName(document.documentElement) !== "fcpxml") {
        throw new Error("FCPXML root element must be <fcpxml>.");
    }
    return document;
}
/**
 * Check element nesting before DOMParser so the Node test implementation and
 * browsers reject the same malformed documents. This is not a second XML
 * parser: attributes and entity decoding remain DOMParser's responsibility.
 */
function assertBalancedMarkup(xml) {
    const stack = [];
    let cursor = 0;
    while (cursor < xml.length) {
        const opening = xml.indexOf("<", cursor);
        if (opening < 0) {
            break;
        }
        if (xml.startsWith("<!--", opening)) {
            const end = xml.indexOf("-->", opening + 4);
            if (end < 0) {
                throw new Error("Invalid FCPXML: unterminated XML comment.");
            }
            cursor = end + 3;
            continue;
        }
        if (xml.startsWith("<![CDATA[", opening)) {
            const end = xml.indexOf("]]>", opening + 9);
            if (end < 0) {
                throw new Error("Invalid FCPXML: unterminated CDATA section.");
            }
            cursor = end + 3;
            continue;
        }
        if (xml.startsWith("<?", opening)) {
            const end = xml.indexOf("?>", opening + 2);
            if (end < 0) {
                throw new Error("Invalid FCPXML: unterminated processing instruction.");
            }
            cursor = end + 2;
            continue;
        }
        let quote = null;
        let end = opening + 1;
        for (; end < xml.length; end += 1) {
            const character = xml[end];
            if (quote) {
                if (character === quote) {
                    quote = null;
                }
            }
            else if (character === '"' || character === "'") {
                quote = character;
            }
            else if (character === ">") {
                break;
            }
        }
        if (end >= xml.length) {
            throw new Error("Invalid FCPXML: unterminated element tag.");
        }
        const token = xml.slice(opening + 1, end).trim();
        cursor = end + 1;
        if (token.startsWith("!")) {
            continue;
        }
        const closing = token.startsWith("/");
        const selfClosing = token.endsWith("/");
        const nameMatch = /^\/?\s*([^\s/>]+)/.exec(token);
        const name = nameMatch?.[1];
        if (!name) {
            throw new Error("Invalid FCPXML: element tag has no name.");
        }
        if (closing) {
            const expected = stack.pop();
            if (expected !== name) {
                throw new Error(`Invalid FCPXML: closing <${name}> does not match <${expected ?? "none"}>.`);
            }
        }
        else if (!selfClosing) {
            stack.push(name);
        }
    }
    if (stack.length > 0) {
        throw new Error(`Invalid FCPXML: unclosed <${stack.at(-1)}> element.`);
    }
}
function serializeDocument(document, original) {
    if (typeof XMLSerializer === "undefined") {
        throw new Error("This environment does not provide the browser XMLSerializer API.");
    }
    let serialized = new XMLSerializer().serializeToString(document);
    if (/^\s*<\?xml\b/.test(original) && !/^\s*<\?xml\b/.test(serialized)) {
        serialized = `<?xml version="1.0"?>${serialized}`;
    }
    return serialized;
}
function parseResources(document) {
    const resources = firstChild(document.documentElement, "resources");
    const formats = new Map();
    const assets = new Map();
    const effects = new Map();
    const media = new Map();
    if (!resources) {
        return { formats, assets, effects, media, mediaAssets: [] };
    }
    for (const element of childElements(resources)) {
        const tag = tagName(element);
        const id = element.getAttribute("id");
        if (!id) {
            continue;
        }
        if (tag === "media") {
            media.set(id, element);
            continue;
        }
        if (tag === "format") {
            const frameRaw = element.getAttribute("frameDuration") ?? "1/30s";
            formats.set(id, {
                id,
                frameDuration: parseRationalTime(frameRaw, `format ${id}`),
                width: Math.max(1, Math.trunc(finiteNumber(element.getAttribute("width"), 1920))),
                height: Math.max(1, Math.trunc(finiteNumber(element.getAttribute("height"), 1080))),
            });
            continue;
        }
        if (tag === "effect") {
            effects.set(id, {
                id,
                name: element.getAttribute("name") || id,
                uid: element.getAttribute("uid"),
            });
            continue;
        }
        if (tag !== "asset") {
            continue;
        }
        const mediaRep = childrenNamed(element, "media-rep").find((candidate) => candidate.getAttribute("kind") === "original-media") ?? firstChild(element, "media-rep");
        const source = mediaRep?.getAttribute("src") ?? null;
        const hasVideo = element.getAttribute("hasVideo") !== "0";
        const hasAudio = element.getAttribute("hasAudio") === "1";
        const extension = source?.split(/[?#]/, 1)[0]?.split(".").at(-1)?.toLowerCase() ?? "";
        const kind = !hasVideo && hasAudio
            ? "audio"
            : new Set(["avif", "bmp", "gif", "heic", "jpeg", "jpg", "png", "tif", "tiff", "webp"]).has(extension)
                ? "image"
                : "video";
        const asset = {
            id,
            name: element.getAttribute("name") || decodedBasename(source) || id,
            kind,
            start: seconds(element.getAttribute("start"), 0, `asset ${id} start`),
            duration: Math.max(0, seconds(element.getAttribute("duration"), 0, `asset ${id} duration`)),
            source,
            formatId: element.getAttribute("format"),
        };
        assets.set(id, asset);
    }
    return {
        formats,
        assets,
        effects,
        media,
        mediaAssets: [...assets.values()].map(mediaAssetFromResource),
    };
}
function decodedBasename(source) {
    if (!source) {
        return null;
    }
    const value = source.split(/[?#]/, 1)[0]?.replace(/\\/g, "/").split("/").at(-1);
    if (!value) {
        return null;
    }
    try {
        return decodeURIComponent(value);
    }
    catch {
        return value;
    }
}
function mediaAssetFromResource(asset) {
    const hash = [...asset.id].reduce((value, character) => ((value * 31) + character.charCodeAt(0)) >>> 0, 0);
    const hueA = hash % 360;
    const hueB = (hueA + 47 + (hash % 83)) % 360;
    return {
        id: asset.id,
        ...(asset.source ? { sourcePath: decodedBundleMediaPath(asset.source) } : {}),
        name: asset.name,
        kind: asset.kind,
        duration: asset.duration,
        colors: { a: `hsl(${hueA} 32% 28%)`, b: `hsl(${hueB} 45% 48%)` },
        tags: [],
        glyph: asset.kind === "audio" ? "A" : asset.kind === "image" ? "I" : "V",
    };
}
/** Decode the segment-wise URI encoding used by relative FCPXML media locators. */
function decodedBundleMediaPath(source) {
    const relative = source.replace(/^\.\//, "");
    try {
        return relative.split("/").map((segment) => decodeURIComponent(segment)).join("/");
    }
    catch (error) {
        throw new Error(`Invalid percent-encoding in media locator ${source}.`, { cause: error });
    }
}
function projectFormat(sequence, formats) {
    const requested = sequence.getAttribute("format");
    const found = requested ? formats.get(requested) : undefined;
    return found ?? {
        id: requested ?? "",
        frameDuration: { numerator: 1n, denominator: 30n },
        width: 1920,
        height: 1080,
    };
}
function activeTimelineReasons(project, spine) {
    const reasons = new Set();
    if (descendantsNamed(spine, "asset-clip").some((clip) => clip.getAttribute("enabled") === "0")) {
        reasons.add("Disabled timeline clips are preserved read-only because Studio does not expose clip activation.");
    }
    if (descendantsNamed(project, "text").some((text) => childrenNamed(text, "text-style").length > 1)) {
        reasons.add("Titles with multiple text styles are preserved read-only because Studio cannot edit individual style runs safely.");
    }
    if (descendantsNamed(project, "keyframe").some((keyframe) => keyframe.hasAttribute("curve") || keyframe.hasAttribute("auxValue"))) {
        reasons.add("Keyframe curve and auxiliary metadata are preserved read-only because Studio cannot edit them safely.");
    }
    for (const element of descendantsNamed(project, "adjust-blend")) {
        try {
            supportedBlendMode(element.getAttribute("mode"));
        }
        catch (error) {
            reasons.add(error instanceof Error ? error.message : "Unsupported blend mode.");
        }
    }
    for (const element of descendantsNamed(project, "timeMap")) {
        if (childrenNamed(element, "timept").some((point) => (point.getAttribute("interp") || "smooth2").toLowerCase() !== "linear")) {
            reasons.add("Non-linear retime interpolation is not editable because Bladeworks rejects it.");
        }
    }
    for (const element of descendantsNamed(project, "adjust-transform")) {
        const allowed = new Set(["position", "scale", "rotation", "anchor"]);
        for (const parameter of childrenNamed(element, "param")) {
            const name = (parameter.getAttribute("name") || "").toLowerCase();
            if (!allowed.has(name))
                reasons.add(`Transform parameter ${name || "<unnamed>"} is unsupported.`);
        }
    }
    for (const group of descendantsNamed(project, "filter-video-mask")) {
        for (const child of childElements(group)) {
            if (!new Set(["mask-shape", "mask-isolation", "filter-video"]).has(tagName(child))) {
                reasons.add(`Masked effect contains unsupported <${tagName(child)}> data.`);
            }
        }
        const masks = childElements(group).filter((child) => new Set(["mask-shape", "mask-isolation"]).has(tagName(child)));
        if (masks.length === 0)
            reasons.add("Masked effect has no mask sources.");
        if (childrenNamed(group, "filter-video").length < 1 || childrenNamed(group, "filter-video").length > 2) {
            reasons.add("Masked effect must contain one inside filter and at most one outside filter.");
        }
        for (const mask of masks) {
            const name = mask.getAttribute("name") || tagName(mask);
            const normalized = name.toLowerCase();
            if (mask.hasAttribute("tracking") || normalized.includes("magnetic mask") || normalized.includes("auto mask")) {
                reasons.add(`${name} uses unsupported tracking or ML mask data.`);
            }
            const blend = mask.getAttribute("blendMode") || "add";
            if (!new Set(["add", "subtract", "multiply"]).has(blend)) {
                reasons.add(`${name} uses unsupported mask blend mode ${blend}.`);
            }
            if (tagName(mask) === "mask-isolation") {
                const data = firstChild(mask, "data")?.textContent?.trim();
                if (data) {
                    try {
                        const decoded = JSON.parse(data);
                        if (decoded.abi !== "spell-mask-isolation-v1")
                            reasons.add(`${name} uses opaque isolation data.`);
                    }
                    catch {
                        reasons.add(`${name} uses opaque isolation data.`);
                    }
                }
            }
        }
    }
    for (const element of descendantsNamed(project, "object-tracker")) {
        void element;
        reasons.add("Object trackers are not editable yet.");
    }
    for (const mute of descendantsNamed(project, "mute")) {
        if (mute.hasAttribute("start") || mute.hasAttribute("duration") || childElements(mute).length > 0) {
            reasons.add("Partial or animated audio mute ranges are preserved read-only; Studio authors full-clip mute only.");
        }
    }
    const visitClip = (clip, depth) => {
        for (const child of childElements(clip)) {
            const tag = tagName(child);
            if (TIMELINE_TAGS.has(tag)) {
                if (depth >= 1 && tagName(clip) !== "sync-clip" && tagName(clip) !== "audition") {
                    reasons.add("Attachments nested beneath connected clips are preserved read-only.");
                }
                visitClip(child, depth + 1);
            }
            else if (UNSAFE_ACTIVE_TAGS.has(tag)) {
                reasons.add(`<${tag}> timeline items are not editable yet.`);
            }
            else if (!PASSIVE_CLIP_TAGS.has(tag)) {
                reasons.add(`<${tag}> clip children cannot be preserved safely during edits.`);
            }
        }
    };
    let hasPreviousStorylineClip = false;
    let hasPendingTransition = false;
    for (const child of childElements(spine)) {
        const tag = tagName(child);
        if (tag === "transition") {
            if (!hasPreviousStorylineClip)
                reasons.add("Primary storyline contains a transition without a preceding clip.");
            if (hasPendingTransition)
                reasons.add("Primary storyline contains consecutive transitions.");
            hasPendingTransition = true;
        }
        else if (TIMELINE_TAGS.has(tag)) {
            visitClip(child, 0);
            hasPreviousStorylineClip = true;
            hasPendingTransition = false;
        }
        else {
            reasons.add(`<${tag}> primary-storyline items are not editable yet.`);
        }
    }
    if (hasPendingTransition)
        reasons.add("Primary storyline contains a transition without a following clip.");
    return [...reasons];
}
function parseTransition(id, element, duration, leftClip, rightClip, effects) {
    const filter = firstChild(element, "filter-video");
    const effectRef = filter?.getAttribute("ref") ?? "";
    const effectResource = effects.get(effectRef);
    const parameters = filter ? parseParameters(filter) : { values: {}, names: {}, keyframes: {} };
    return {
        id,
        name: element.getAttribute("name") || "Transition",
        category: "Video",
        leftItemId: leftClip.id,
        rightItemId: rightClip.id,
        duration,
        resourceId: effectRef,
        resourceUid: effectResource?.uid ?? null,
        handler: null,
        support: "partial",
        parameters: parameters.values,
        parameterNames: parameters.names,
        parameterKeyframes: parameters.keyframes,
    };
}
function transitionOrigin(element, model) {
    return {
        element,
        model,
        offset: element.getAttribute("offset") ?? "0s",
        offsetSeconds: seconds(element.getAttribute("offset"), 0, `${model.id} offset`),
        duration: requireAttribute(element, "duration", model.id),
    };
}
function parseProject(ref, project, libraryId, eventId, formats, assets, effects, media) {
    const sequence = firstChild(project, "sequence");
    if (!sequence) {
        throw new Error(`Project ${ref} does not contain a <sequence>.`);
    }
    const spine = firstChild(sequence, "spine");
    if (!spine) {
        throw new Error(`Project ${ref} does not contain a <spine>.`);
    }
    const format = projectFormat(sequence, formats);
    const tcStart = rationalTime(sequence.getAttribute("tcStart"), `${ref} sequence tcStart`);
    const scopeCollector = {
        scopes: {}, order: [], origins: new Map(), assets, effects, media,
        frameDuration: format.frameDuration, visiting: new Set(),
    };
    const clipOrigins = new Map();
    const transitionOrigins = new Map();
    const spineClips = [];
    const connected = [];
    const transitions = [];
    const tagCounts = new Map();
    const identityCounts = new Map();
    let previousClip = null;
    let pendingTransition = null;
    for (const element of childElements(spine)) {
        const tag = tagName(element);
        const index = (tagCounts.get(tag) ?? 0) + 1;
        tagCounts.set(tag, index);
        if (tag === "transition") {
            const id = `${ref}/spine/transition[${index}]`;
            pendingTransition = {
                id,
                element,
                duration: seconds(element.getAttribute("duration"), 0, `${id} duration`),
            };
            continue;
        }
        if (!TIMELINE_TAGS.has(tag)) {
            continue;
        }
        const id = timelineItemId(ref, element, identityCounts);
        let clip = parseTimelineClip(element, id, "storyline", assets, effects);
        clip = { ...clip, timelineStart: clip.timelineStart - tcStart.seconds };
        clip = { ...clip, container: collectContainerScopes(scopeCollector, element, clip, null) };
        spineClips.push(clip);
        clipOrigins.set(id, {
            element,
            model: clip,
            offset: element.getAttribute("offset") ?? quantizedTime(clip.timelineStart + tcStart.seconds, format.frameDuration),
            start: element.getAttribute("start") ?? "0s",
            duration: requireAttribute(element, "duration", id),
        });
        if (clip.container?.kind !== "sync" && clip.container?.kind !== "audition") {
            parseConnectedChildren(element, clip, assets, effects, connected, clipOrigins, ref, identityCounts, scopeCollector, null);
        }
        if (pendingTransition && previousClip) {
            const transition = parseTransition(pendingTransition.id, pendingTransition.element, pendingTransition.duration, previousClip, clip, effects);
            transitions.push(transition);
            transitionOrigins.set(transition.id, transitionOrigin(pendingTransition.element, transition));
        }
        pendingTransition = null;
        previousClip = clip;
    }
    const snapshot = {
        revision: 0,
        id: ref,
        libraryId,
        eventId,
        name: project.getAttribute("name") || "Untitled Project",
        fps: 1 / rationalSeconds(format.frameDuration),
        width: format.width,
        height: format.height,
        audioLayout: sequence.getAttribute("audioLayout") === "mono" ? "mono" : "stereo",
        spine: spineClips,
        connected,
        transitions,
        scopes: scopeCollector.scopes,
        scopeOrder: scopeCollector.order,
        proposal: null,
    };
    return {
        snapshot,
        origin: {
            ref,
            element: project,
            sequence,
            spine,
            frameDuration: format.frameDuration,
            tcStart,
            clipOrigins,
            transitionOrigins,
            scopeOrigins: scopeCollector.origins,
            audioLayout: sequence.getAttribute("audioLayout"),
        },
        reasons: [
            ...activeTimelineReasons(project, spine),
            ...(sequence.hasAttribute("audioLayout")
                && sequence.getAttribute("audioLayout") !== "mono"
                && sequence.getAttribute("audioLayout") !== "stereo"
                ? [`Project audio layout ${sequence.getAttribute("audioLayout")} is unsupported; use mono or stereo.`]
                : []),
        ],
    };
}
function parseConnectedChildren(parent, anchor, assets, effects, output, origins, projectRef, identityCounts, scopeCollector, parentScopeId = null) {
    for (const element of childElements(parent)) {
        const tag = tagName(element);
        if (!TIMELINE_TAGS.has(tag)) {
            continue;
        }
        const id = timelineItemId(projectRef, element, identityCounts);
        let parsed = parseTimelineClip(element, id, "connected", assets, effects);
        if (scopeCollector)
            parsed = { ...parsed, container: collectContainerScopes(scopeCollector, element, parsed, parentScopeId) };
        const relativeOffset = seconds(element.getAttribute("offset"), 0, `${id} offset`) - anchor.sourceStart;
        const lane = Math.trunc(finiteNumber(element.getAttribute("lane"), parsed.kind === "audio" ? -1 : 1));
        const connected = {
            ...parsed,
            role: parsed.kind === "audio" ? "connected-audio" : parsed.kind === "title" || parsed.kind === "caption" ? "title" : "connected-video",
            anchorId: anchor.id,
            anchorOffset: relativeOffset,
            timelineStart: anchor.timelineStart + relativeOffset,
            lane: lane === 0 ? (parsed.kind === "audio" ? -1 : 1) : lane,
        };
        output.push(connected);
        origins.set(id, {
            element,
            model: connected,
            offset: element.getAttribute("offset") ?? "0s",
            start: element.getAttribute("start") ?? "0s",
            duration: requireAttribute(element, "duration", id),
        });
    }
}
function timelineItemId(projectRef, element, counts) {
    const explicit = element.getAttribute("uid");
    const signature = [
        tagName(element),
        element.getAttribute("ref") ?? "",
        element.getAttribute("name") ?? "",
        element.getAttribute("role") ?? element.getAttribute("audioRole") ?? "",
    ].join("\u001f");
    const occurrence = (counts.get(signature) ?? 0) + 1;
    counts.set(signature, occurrence);
    const stableUid = explicit ?? deterministicTimelineUid(`${projectRef}\u001f${signature}\u001f${occurrence}`);
    if (!explicit)
        element.setAttribute("uid", stableUid);
    return `${projectRef}/item[${stableUid}]`;
}
function deterministicTimelineUid(value) {
    const words = [2166136261, 2246822507, 3266489909, 668265263].map((seed, index) => {
        let hash = seed >>> 0;
        for (const character of `${index}\u001f${value}`) {
            hash ^= character.charCodeAt(0);
            hash = Math.imul(hash, 16777619) >>> 0;
        }
        return hash.toString(16).padStart(8, "0");
    }).join("");
    return `${words.slice(0, 8)}-${words.slice(8, 12)}-4${words.slice(13, 16)}-a${words.slice(17, 20)}-${words.slice(20, 32)}`;
}
function scopeClock(element, frameDuration) {
    const tcStart = element.getAttribute("tcStart") ?? element.getAttribute("start") ?? "0s";
    const duration = element.getAttribute("duration") ?? "0s";
    return {
        tcStart: rationalTime(tcStart, "nested scope tcStart"),
        duration: rationalTime(duration, "nested scope duration"),
        frameDuration: rationalTime(rationalString(frameDuration), "nested scope frame duration"),
    };
}
function collectScope(collector, descriptor, element, children) {
    const existing = collector.scopes[descriptor.id];
    if (existing)
        return existing;
    if (collector.visiting.has(descriptor.id))
        throw new Error(`Recursive scope cycle references ${descriptor.id}.`);
    collector.visiting.add(descriptor.id);
    const origins = new Map();
    const transitionOrigins = new Map();
    const identityCounts = new Map();
    const spine = [];
    const connected = [];
    const transitions = [];
    const reasons = new Set();
    const order = [];
    let previousDirectClip = null;
    let pendingTransition = null;
    let transitionIndex = 0;
    const findUnsafeNestedAttachments = (parent, depth) => {
        for (const nested of childElements(parent)) {
            if (!TIMELINE_TAGS.has(tagName(nested)))
                continue;
            if (depth >= 1 && tagName(parent) !== "sync-clip" && tagName(parent) !== "audition") {
                reasons.add("Attachments nested beneath connected scope clips are preserved read-only.");
            }
            findUnsafeNestedAttachments(nested, depth + 1);
        }
    };
    for (const child of children) {
        if (tagName(child) === "transition") {
            if (!previousDirectClip)
                reasons.add("Nested scope contains a transition without a preceding clip.");
            if (pendingTransition)
                reasons.add("Nested scope contains consecutive transitions.");
            transitionIndex += 1;
            pendingTransition = {
                id: `${descriptor.id}/transition[${transitionIndex}]`,
                element: child,
                duration: seconds(child.getAttribute("duration"), 0, `${descriptor.id} transition duration`),
            };
            continue;
        }
        if (!TIMELINE_TAGS.has(tagName(child))) {
            reasons.add(`Nested scope contains unsupported <${tagName(child)}> data.`);
            continue;
        }
        const id = timelineItemId(descriptor.id, child, identityCounts);
        let clip = parseTimelineClip(child, id, child.hasAttribute("lane") ? "connected" : "storyline", collector.assets, collector.effects);
        clip = { ...clip, container: collectContainerScopes(collector, child, clip, descriptor.id) };
        findUnsafeNestedAttachments(child, 0);
        for (const nested of childElements(child)) {
            const nestedTag = tagName(nested);
            if (nestedTag === "mute"
                && (nested.hasAttribute("start") || nested.hasAttribute("duration") || childElements(nested).length > 0)) {
                reasons.add(`${clip.name} contains a partial or animated mute range.`);
            }
            if (child.hasAttribute("lane") && TIMELINE_TAGS.has(nestedTag)) {
                reasons.add(`${clip.name} contains a nested attachment that cannot be edited safely.`);
            }
            if (!TIMELINE_TAGS.has(nestedTag)
                && (UNSAFE_ACTIVE_TAGS.has(nestedTag) || !PASSIVE_CLIP_TAGS.has(nestedTag))) {
                reasons.add(`${clip.name} contains unsupported <${nestedTag}> internals.`);
            }
        }
        order.push(id);
        const lane = Math.trunc(finiteNumber(child.getAttribute("lane"), 0));
        if (lane !== 0) {
            const timelineStart = clip.timelineStart - clip.sourceStart;
            const anchor = [...spine].reverse().find((candidate) => timelineStart >= candidate.timelineStart - EPSILON);
            if (!anchor) {
                reasons.add(`Connected nested clip ${clip.name} has no preceding scope anchor.`);
                spine.push(clip);
                origins.set(id, {
                    element: child, model: clip,
                    offset: child.getAttribute("offset") ?? "0s",
                    start: child.getAttribute("start") ?? "0s",
                    duration: requireAttribute(child, "duration", id),
                });
                continue;
            }
            const relativeOffset = timelineStart - anchor.timelineStart;
            const connectedClip = {
                ...clip,
                role: clip.kind === "audio" ? "connected-audio" : "connected-video",
                anchorId: anchor.id,
                anchorOffset: relativeOffset,
                timelineStart: anchor.timelineStart + relativeOffset,
                lane,
            };
            connected.push(connectedClip);
            origins.set(id, {
                element: child, model: connectedClip,
                offset: child.getAttribute("offset") ?? "0s",
                start: child.getAttribute("start") ?? "0s",
                duration: requireAttribute(child, "duration", id),
            });
            if (pendingTransition && previousDirectClip) {
                const transition = parseTransition(pendingTransition.id, pendingTransition.element, pendingTransition.duration, previousDirectClip, connectedClip, collector.effects);
                transitions.push(transition);
                transitionOrigins.set(transition.id, transitionOrigin(pendingTransition.element, transition));
            }
            pendingTransition = null;
            previousDirectClip = connectedClip;
        }
        else {
            spine.push(clip);
            origins.set(id, {
                element: child, model: clip,
                offset: child.getAttribute("offset") ?? "0s",
                start: child.getAttribute("start") ?? "0s",
                duration: requireAttribute(child, "duration", id),
            });
            if (clip.container?.kind !== "sync" && clip.container?.kind !== "audition") {
                const connectedStart = connected.length;
                parseConnectedChildren(child, clip, collector.assets, collector.effects, connected, origins, descriptor.id, identityCounts, collector, descriptor.id);
                order.push(...connected.slice(connectedStart).map((nestedClip) => nestedClip.id));
            }
            if (pendingTransition && previousDirectClip) {
                const transition = parseTransition(pendingTransition.id, pendingTransition.element, pendingTransition.duration, previousDirectClip, clip, collector.effects);
                transitions.push(transition);
                transitionOrigins.set(transition.id, transitionOrigin(pendingTransition.element, transition));
            }
            pendingTransition = null;
            previousDirectClip = clip;
        }
    }
    if (pendingTransition)
        reasons.add("Nested scope contains a transition without a following clip.");
    const scope = {
        ...descriptor,
        editable: reasons.size === 0,
        reasons: [...reasons],
        clock: scopeClock(element, collector.frameDuration),
        spine,
        connected,
        transitions,
        childOrder: order,
    };
    collector.scopes[scope.id] = scope;
    collector.order.push(scope.id);
    collector.origins.set(scope.id, {
        model: scope, element, childOrigins: origins, transitionOrigins, childOrder: order,
    });
    collector.visiting.delete(scope.id);
    return scope;
}
function collectContainerScopes(collector, element, clip, parentScopeId) {
    const tag = tagName(element);
    if (tag === "ref-clip") {
        const resourceId = requireAttribute(element, "ref", clip.id);
        const media = collector.media.get(resourceId);
        const sequence = media ? firstChild(media, "sequence") : null;
        const spine = sequence ? firstChild(sequence, "spine") : null;
        if (!media || !sequence || !spine)
            throw new Error(`Compound ${clip.name} refers to missing media resource ${resourceId}.`);
        const scopeId = `media[${resourceId}]/sequence`;
        collectScope(collector, {
            id: scopeId, kind: "compound", name: media.getAttribute("name") || clip.name,
            parentScopeId, viaClipId: clip.id, resourceId, angleId: null,
        }, sequence, childElements(spine));
        return { kind: "compound", resourceId, scopeId };
    }
    if (tag === "mc-clip") {
        const resourceId = requireAttribute(element, "ref", clip.id);
        const media = collector.media.get(resourceId);
        const multicam = media ? firstChild(media, "multicam") : null;
        if (!media || !multicam)
            throw new Error(`Multicam ${clip.name} refers to missing media resource ${resourceId}.`);
        const angleScopeIds = {};
        for (const angle of childrenNamed(multicam, "mc-angle")) {
            const angleId = requireAttribute(angle, "angleID", `${clip.id} multicam angle`);
            const scopeId = `media[${resourceId}]/multicam/angle[${angleId}]`;
            angleScopeIds[angleId] = scopeId;
            collectScope(collector, {
                id: scopeId, kind: "multicam-angle", name: angle.getAttribute("name") || angleId,
                parentScopeId, viaClipId: clip.id, resourceId, angleId,
            }, multicam, childElements(angle));
        }
        const selections = childrenNamed(element, "mc-source");
        return {
            kind: "multicam", resourceId, angleScopeIds,
            videoAngleId: selections.find((source) => (source.getAttribute("srcEnable") || "").split(/[, ]+/).includes("video"))?.getAttribute("angleID") ?? null,
            audioAngleId: selections.find((source) => (source.getAttribute("srcEnable") || "").split(/[, ]+/).includes("audio"))?.getAttribute("angleID") ?? null,
        };
    }
    if (tag === "sync-clip") {
        const scopeId = `${clip.id}/sync`;
        collectScope(collector, {
            id: scopeId, kind: "sync", name: clip.name, parentScopeId,
            viaClipId: clip.id, resourceId: null, angleId: null,
        }, element, childElements(element).filter((child) => TIMELINE_TAGS.has(tagName(child))));
        const sources = childrenNamed(element, "sync-source").map((source) => {
            const role = firstChild(source, "audio-role-source");
            return {
                sourceId: source.getAttribute("sourceID") || "",
                role: role?.getAttribute("role") ?? null,
                enabled: role ? role.getAttribute("enabled") !== "0" : true,
                active: role ? role.getAttribute("active") !== "0" : true,
            };
        });
        return { kind: "sync", scopeId, sources };
    }
    if (tag === "audition") {
        const choices = childElements(element).filter((child) => TIMELINE_TAGS.has(tagName(child)));
        const choiceScopeIds = choices.map((choice, index) => {
            const scopeId = `${clip.id}/audition-choice[${index + 1}]`;
            collectScope(collector, {
                id: scopeId, kind: "audition-choice", name: choice.getAttribute("name") || `Choice ${index + 1}`,
                parentScopeId, viaClipId: clip.id, resourceId: null, angleId: null,
            }, choice, [choice]);
            return scopeId;
        });
        const activeIndex = choices.findIndex((choice) => choice.getAttribute("enabled") !== "0");
        return { kind: "audition", choiceScopeIds, activeChoiceId: choiceScopeIds[activeIndex < 0 ? 0 : activeIndex] ?? null };
    }
    return null;
}
function parseTimelineClip(element, id, placement, assets, effects) {
    const tag = tagName(element);
    const ref = element.getAttribute("ref");
    const asset = ref ? assets.get(ref) : undefined;
    const template = ref ? effects.get(ref) : undefined;
    const kind = tag === "gap"
        ? "gap"
        : tag === "title"
            ? "title"
            : tag === "caption"
                ? "caption"
                : tag === "video" && template?.uid === CUSTOM_SOLID_UID
                    ? "generator"
                    : tag === "audio" || asset?.kind === "audio"
                        ? "audio"
                        : asset?.kind === "image"
                            ? "image"
                            : "video";
    const transform = defaultTransform();
    const video = defaultVideo();
    const audio = defaultAudio();
    const transformElement = firstChild(element, "adjust-transform");
    if (transformElement) {
        transform.enabled = transformElement.getAttribute("enabled") !== "0";
        const position = pair(transformElement.getAttribute("position"), [0, 0]);
        const scale = pair(transformElement.getAttribute("scale"), [1, 1]);
        const anchor = pair(transformElement.getAttribute("anchor"), [0, 0]);
        transform.x = position[0];
        transform.y = position[1];
        if (Math.abs(scale[0] - scale[1]) <= EPSILON) {
            transform.scale = scale[0];
        }
        else {
            transform.scaleX = scale[0];
            transform.scaleY = scale[1];
        }
        transform.rotation = finiteNumber(transformElement.getAttribute("rotation"), 0);
        transform.anchorX = anchor[0];
        transform.anchorY = anchor[1];
    }
    const blend = firstChild(element, "adjust-blend");
    if (blend) {
        video.blendEnabled = blend.getAttribute("enabled") !== "0";
        transform.opacity = finiteNumber(blend.getAttribute("amount"), 1);
        try {
            video.blendMode = supportedBlendMode(blend.getAttribute("mode"));
        }
        catch {
            // activeTimelineReasons marks the Project read-only while preserving XML.
        }
    }
    const crop = firstChild(element, "adjust-crop");
    if (crop) {
        video.crop.enabled = crop.getAttribute("enabled") !== "0";
        const mode = crop.getAttribute("mode");
        video.crop.type = mode === "kenburns" || mode === "kenBurns" || mode === "pan"
            ? "ken-burns" : mode === "crop" ? "crop" : "trim";
        const rectName = video.crop.type === "trim" ? "trim-rect" : "crop-rect";
        const rect = firstChild(crop, rectName) ?? firstChild(crop, "crop-rect");
        if (rect) {
            video.crop.left = finiteNumber(rect.getAttribute("left"), 0);
            video.crop.right = finiteNumber(rect.getAttribute("right"), 0);
            video.crop.top = finiteNumber(rect.getAttribute("top"), 0);
            video.crop.bottom = finiteNumber(rect.getAttribute("bottom"), 0);
        }
        if (video.crop.type === "ken-burns") {
            const windows = childrenNamed(crop, "pan-rect");
            const toWindow = (window, fallback) => {
                if (!window)
                    return fallback;
                const left = finiteNumber(window.getAttribute("left"), 0);
                const right = finiteNumber(window.getAttribute("right"), 0);
                const top = finiteNumber(window.getAttribute("top"), 0);
                const bottom = finiteNumber(window.getAttribute("bottom"), 0);
                return {
                    x: left + ((100 - left - right) / 2),
                    y: top + ((100 - top - bottom) / 2),
                    width: 100 - left - right,
                    height: 100 - top - bottom,
                };
            };
            video.crop.kenStart = toWindow(windows[0], video.crop.kenStart);
            video.crop.kenEnd = toWindow(windows[1], video.crop.kenEnd);
        }
    }
    const conform = firstChild(element, "adjust-conform");
    const conformType = conform?.getAttribute("type");
    if (conformType === "fit" || conformType === "fill" || conformType === "none") {
        video.spatialConform = conformType;
    }
    video.stabilization = firstChild(element, "adjust-stabilization") !== null;
    video.rollingShutter = firstChild(element, "adjust-rollingShutter") !== null;
    const colorConform = firstChild(element, "adjust-colorConform");
    if (colorConform) {
        video.colorConform = colorConform.getAttribute("enabled") !== "0";
        const type = colorConform.getAttribute("type")?.toLowerCase();
        video.colorConformType = type === "sdr" || type === "hdr" ? type : "automatic";
    }
    const cornerPin = firstChild(element, "adjust-corners");
    if (cornerPin) {
        const topLeft = pair(cornerPin.getAttribute("topLeft"), [0, 0]);
        const topRight = pair(cornerPin.getAttribute("topRight"), [0, 0]);
        const bottomLeft = pair(cornerPin.getAttribute("bottomLeft"), [0, 0]);
        const bottomRight = pair(cornerPin.getAttribute("bottomRight"), [0, 0]);
        video.distort = {
            ...defaultDistort(),
            enabled: cornerPin.getAttribute("enabled") !== "0",
            topLeftX: topLeft[0], topLeftY: topLeft[1],
            topRightX: topRight[0], topRightY: topRight[1],
            bottomLeftX: bottomLeft[0], bottomLeftY: bottomLeft[1],
            bottomRightX: bottomRight[0], bottomRightY: bottomRight[1],
        };
    }
    const volume = firstChild(element, "adjust-volume");
    if (volume) {
        audio.gainDb = finiteNumber(volume.getAttribute("amount"), 0);
    }
    audio.muted = Boolean(firstChild(element, "mute"));
    const panner = firstChild(element, "adjust-panner");
    audio.pan = panner
        ? normalizedPannerValue(parseParameterValue(panner.getAttribute("amount") ?? "0"), panner, `${id} pan`)
        : 0;
    audio.fadeIn = seconds(findAudioFade(volume, "fadeIn")?.getAttribute("duration") ?? null, 0, `${id} fadeIn`);
    audio.fadeOut = seconds(findAudioFade(volume, "fadeOut")?.getAttribute("duration") ?? null, 0, `${id} fadeOut`);
    audio.loudness = finiteNumber(firstChild(element, "adjust-loudness")?.getAttribute("amount") ?? null, 0);
    audio.noiseRemoval = finiteNumber(firstChild(element, "adjust-noiseReduction")?.getAttribute("amount") ?? null, 0);
    const duration = seconds(element.getAttribute("duration"), 0, `${id} duration`);
    const sourceStart = seconds(element.getAttribute("start"), 0, `${id} start`);
    const keyframes = parseKeyframes(element);
    const timeMapElement = firstChild(element, "timeMap");
    const timeMap = timeMapElement ? {
        frameSampling: timeMapElement.getAttribute("frameSampling"),
        preservesPitch: timeMapElement.hasAttribute("preservesPitch")
            ? timeMapElement.getAttribute("preservesPitch") !== "0"
            : null,
        points: childrenNamed(timeMapElement, "timept").map((point, index) => ({
            time: rationalTime(point.getAttribute("time"), `${id} retime point ${index + 1} time`),
            value: rationalTime(point.getAttribute("value"), `${id} retime point ${index + 1} value`),
            interpolation: point.getAttribute("interp") || "smooth2",
        })),
    } : null;
    const markers = parseMarkers(element, id, sourceStart);
    const effectStack = [];
    let plainEffectIndex = 0;
    let maskedEffectIndex = 0;
    for (const child of childElements(element)) {
        if (tagName(child) === "filter-video") {
            plainEffectIndex += 1;
            effectStack.push({ kind: "effect", effect: parseClipEffect(child, `${id}/effect[${plainEffectIndex}]`, effects) });
        }
        else if (tagName(child) === "filter-video-mask") {
            maskedEffectIndex += 1;
            effectStack.push({
                kind: "masked-effect",
                maskedEffect: parseMaskedEffect(child, `${id}/masked-effect[${maskedEffectIndex}]`, effects),
            });
        }
    }
    const clipEffects = effectStack.flatMap((item) => item.kind === "effect" ? [item.effect] : []);
    const textStyle = parseTextStyle(element);
    const textElement = firstChild(element, "text");
    const generatorParameters = parseParameters(element);
    return {
        id,
        assetId: kind === "gap" ? null : ref,
        name: element.getAttribute("name") || asset?.name || (kind === "gap" ? "Gap" : id),
        kind,
        role: placement === "storyline" ? "storyline" : kind === "audio" ? "connected-audio" : kind === "title" || kind === "caption" ? "title" : "connected-video",
        roleName: element.getAttribute("role"),
        audioRole: element.getAttribute("audioRole"),
        audioStart: element.hasAttribute("audioStart") ? rationalTime(element.getAttribute("audioStart"), `${id} audioStart`) : null,
        audioDuration: element.hasAttribute("audioDuration") ? rationalTime(element.getAttribute("audioDuration"), `${id} audioDuration`) : null,
        ...(asset ? { sourceRangeStart: asset.start } : {}),
        ...(asset && asset.duration > 0 ? { sourceDuration: asset.duration } : {}),
        sourceStart,
        duration,
        timelineStart: seconds(element.getAttribute("offset"), 0, `${id} offset`),
        colors: asset ? mediaAssetFromResource(asset).colors : { a: "#3b414b", b: "#697384" },
        transform,
        video,
        audio,
        keyframes,
        timeMap,
        text: kind === "title" || kind === "caption" ? textElement?.textContent ?? element.getAttribute("name") : null,
        textStyle,
        caption: kind === "caption" ? {
            displayStyle: textElement?.getAttribute("display-style") || "pop-on",
            placement: textElement?.getAttribute("placement") === "top" ? "top" : "bottom",
            alignment: textElement?.getAttribute("alignment") === "left" || textElement?.getAttribute("alignment") === "right"
                ? textElement.getAttribute("alignment") : "center",
            role: element.getAttribute("role") || "iTT.en-US.dialogue",
        } : null,
        generatorColor: kind === "generator"
            ? parseColorValue(generatorParameters.values[CUSTOM_SOLID_COLOR_KEY], `${id} Custom Solid Color`)
            : null,
        markers,
        effects: clipEffects,
        effectStack,
        container: null,
    };
}
function parseClipEffect(filter, id, effects) {
    const effectRef = filter.getAttribute("ref");
    const resource = effectRef ? effects.get(effectRef) : undefined;
    const parameters = parseParameters(filter);
    return {
        id,
        name: filter.getAttribute("name") || resource?.name || "Video Effect",
        category: "Video",
        enabled: filter.getAttribute("enabled") !== "0",
        resourceId: effectRef ?? "",
        resourceUid: resource?.uid ?? null,
        handler: null,
        support: "partial",
        parameters: parameters.values,
        parameterNames: parameters.names,
        parameterKeyframes: parameters.keyframes,
    };
}
function parseMaskedEffect(group, id, effects) {
    let maskIndex = 0;
    let filterIndex = 0;
    const masks = [];
    const filters = [];
    for (const child of childElements(group)) {
        const tag = tagName(child);
        if (tag === "mask-shape" || tag === "mask-isolation") {
            maskIndex += 1;
            masks.push(parseMaskSource(child, `${id}/mask[${maskIndex}]`));
        }
        else if (tag === "filter-video") {
            filterIndex += 1;
            filters.push(parseClipEffect(child, `${id}/filter[${filterIndex}]`, effects));
        }
    }
    return {
        id,
        enabled: group.getAttribute("enabled") !== "0",
        inverted: group.getAttribute("inverted") === "1",
        masks,
        filters,
    };
}
function parseMaskSource(element, id) {
    const parameters = parseParameters(element);
    const data = firstChild(element, "data")?.textContent?.trim() || null;
    const hasPoints = Object.keys(parameters.values).some((key) => new Set(["points", "vertices", "path", "300"]).has(key.toLowerCase()));
    let kind = hasPoints ? "draw" : "shape";
    if (tagName(element) === "mask-isolation") {
        const hasColor = Object.keys(parameters.values).some((key) => new Set(["color", "sample color", "bladeworks/color"]).has(key.toLowerCase()))
            || (data ? /"color"\s*:/.test(data) : false);
        kind = hasColor ? "color" : "luma";
    }
    const blend = element.getAttribute("blendMode") || (tagName(element) === "mask-isolation" ? "multiply" : "add");
    if (blend !== "add" && blend !== "subtract" && blend !== "multiply") {
        throw new Error(`Mask ${id} uses unsupported blend mode ${blend}.`);
    }
    return {
        id,
        kind,
        name: element.getAttribute("name") || (kind === "draw" ? "Draw Mask" : kind === "shape" ? "Shape Mask" : kind === "color" ? "Color Mask" : "Luma Mask"),
        enabled: element.getAttribute("enabled") !== "0",
        blendMode: blend,
        parameters: parameters.values,
        parameterNames: parameters.names,
        parameterKeyframes: parameters.keyframes,
        data,
    };
}
function parseColorValue(value, context) {
    if (value && typeof value === "object" && !Array.isArray(value) && "red" in value) {
        return value;
    }
    const pieces = Array.isArray(value) ? value : typeof value === "string" ? value.trim().split(/\s+/).map(Number) : [];
    if (pieces.length !== 4 || pieces.some((piece) => !Number.isFinite(piece))) {
        throw new Error(`${context} requires four finite RGBA components.`);
    }
    return { red: pieces[0], green: pieces[1], blue: pieces[2], alpha: pieces[3] };
}
function parseTextStyle(element) {
    const runReference = firstChild(firstChild(element, "text") ?? element, "text-style")?.getAttribute("ref");
    const definition = childrenNamed(element, "text-style-def").find((candidate) => candidate.getAttribute("id") === runReference) ?? firstChild(element, "text-style-def");
    const style = definition ? firstChild(definition, "text-style") : null;
    if (!style)
        return null;
    return {
        font: style.getAttribute("font") || "Helvetica",
        fontFace: style.getAttribute("fontFace") || "Regular",
        fontSize: finiteNumber(style.getAttribute("fontSize"), 48),
        fontColor: parseColorValue(style.getAttribute("fontColor") || "1 1 1 1", "text fontColor"),
        alignment: style.getAttribute("alignment") === "left" || style.getAttribute("alignment") === "right"
            ? style.getAttribute("alignment") : "center",
    };
}
function parseKeyframes(element) {
    const result = {};
    const adjustments = [
        ["adjust-transform", "transform"],
        ["adjust-blend", "transform"],
        ["adjust-crop", "video.crop"],
        ["adjust-corners", "video.distort"],
        ["adjust-volume", "audio"],
        ["adjust-panner", "audio"],
    ];
    for (const [adjustmentName, prefix] of adjustments) {
        const adjustment = firstChild(element, adjustmentName);
        if (!adjustment) {
            continue;
        }
        for (const parameter of childrenNamed(adjustment, "param")) {
            const rawName = parameter.getAttribute("name") || "value";
            const field = rawName.replace(/\s+/g, "-").toLowerCase();
            const path = adjustmentName === "adjust-blend" && field === "amount"
                ? "transform.opacity"
                : adjustmentName === "adjust-volume" && field === "amount"
                    ? "audio.gainDb"
                    : adjustmentName === "adjust-panner" && field === "amount"
                        ? "audio.pan"
                        : `${prefix}.${field}`;
            const animation = firstChild(parameter, "keyframeAnimation");
            if (!animation) {
                continue;
            }
            const frames = parseParameterKeyframes(parameter, path);
            result[path] = adjustmentName === "adjust-panner" && path === "audio.pan"
                ? frames.map((frame) => ({
                    ...frame,
                    value: normalizedPannerValue(frame.value, adjustment, `${path} keyframe`),
                }))
                : frames;
        }
    }
    return result;
}
function parseMarkers(element, clipId, sourceStart) {
    const result = [];
    let index = 0;
    for (const marker of childElements(element)) {
        const tag = tagName(marker);
        if (!MARKER_TAGS.has(tag)) {
            continue;
        }
        index += 1;
        result.push({
            id: `${clipId}/marker[${index}]`,
            offset: seconds(marker.getAttribute("start"), sourceStart, `${clipId} marker`) - sourceStart,
            name: marker.getAttribute("value") || "Marker",
            type: tag === "chapter-marker" ? "chapter" : tag === "todo-marker" ? "todo" : "standard",
            completed: marker.getAttribute("completed") === "1",
        });
    }
    return result;
}
/**
 * Parse a complete FCPXML library into the browser editor model.
 *
 * Main callers:
 * - LocalhostEditorRuntime during source bootstrap, PUT, undo, and redo.
 * - replaceProjectInFCPXML after serializing one accepted edit.
 */
export function parseFCPXMLLibrary(xml) {
    const document = parseDocument(xml);
    const resources = parseResources(document);
    const projects = {};
    const projectRefs = {};
    const editableProjects = {};
    const projectOrigins = new Map();
    const libraries = childrenNamed(document.documentElement, "library");
    const librarySummaries = libraries.map((library, libraryOffset) => {
        const libraryIndex = libraryOffset + 1;
        const libraryId = `library[${libraryIndex}]`;
        const events = childrenNamed(library, "event").map((event, eventOffset) => {
            const eventIndex = eventOffset + 1;
            const eventId = `${libraryId}/event[${eventIndex}]`;
            const projectSummaries = childrenNamed(event, "project").map((project, projectOffset) => {
                const projectIndex = projectOffset + 1;
                const ref = `${eventId}/project[${projectIndex}]`;
                const parsed = parseProject(ref, project, libraryId, eventId, resources.formats, resources.assets, resources.effects, resources.media);
                projects[ref] = parsed.snapshot;
                projectRefs[ref] = ref;
                editableProjects[ref] = { editable: parsed.reasons.length === 0, reasons: parsed.reasons };
                projectOrigins.set(ref, parsed.origin);
                return {
                    id: ref,
                    eventId,
                    name: parsed.snapshot.name,
                    duration: projectDuration(parsed.snapshot),
                    proposal: null,
                };
            });
            return {
                id: eventId,
                libraryId,
                name: event.getAttribute("name") || `Event ${eventIndex}`,
                projects: projectSummaries,
            };
        });
        return {
            id: libraryId,
            name: library.getAttribute("name") || `Library ${libraryIndex}`,
            events,
        };
    });
    const activeProjectId = Object.keys(projects)[0];
    if (!activeProjectId) {
        throw new Error("FCPXML does not contain a Project inside a Library Event.");
    }
    const workspace = {
        xml,
        bootstrap: {
            libraries: librarySummaries,
            assets: resources.mediaAssets,
            projects,
            activeProjectId,
        },
        projectRefs,
        editableProjects,
    };
    workspaceOrigins.set(workspace, {
        document,
        projects: projectOrigins,
        formats: resources.formats,
        assets: resources.assets,
        effects: resources.effects,
        media: resources.media,
    });
    return workspace;
}
function requireConstructorWorkspace(workspace) {
    if (!workspaceOrigins.has(workspace)) {
        throw new Error("Clip constructors require a workspace returned by parseFCPXMLLibrary().");
    }
}
function defaultTextStyle() {
    return {
        font: "Helvetica",
        fontFace: "Regular",
        fontSize: 72,
        fontColor: { red: 1, green: 1, blue: 1, alpha: 1 },
        alignment: "center",
    };
}
function validateNewClip(options) {
    if (!options.id.trim())
        throw new Error("Inserted clip ID must be non-empty.");
    if (!Number.isFinite(options.duration) || options.duration <= 0) {
        throw new Error("Inserted clip duration must be positive and finite.");
    }
    if (options.timelineStart !== undefined && (!Number.isFinite(options.timelineStart) || options.timelineStart < 0)) {
        throw new Error("Inserted clip timelineStart must be finite and non-negative.");
    }
}
function constructedClip(options, kind, assetId) {
    validateNewClip(options);
    return {
        id: options.id,
        assetId,
        name: options.name || (kind === "title" ? "Basic Title" : kind === "caption" ? "Caption" : "Custom Solid"),
        kind,
        role: "storyline",
        sourceStart: 0,
        duration: options.duration,
        timelineStart: options.timelineStart ?? 0,
        colors: kind === "generator" ? { a: "#606060", b: "#303030" } : { a: "#594b78", b: "#8872b5" },
        transform: defaultTransform(),
        video: defaultVideo(),
        audio: defaultAudio(),
        keyframes: {},
        timeMap: null,
        text: null,
        textStyle: null,
        caption: null,
        generatorColor: null,
        markers: [],
        effects: [],
        effectStack: [],
    };
}
/** Create the certified Basic Title model; resource insertion happens atomically with Project serialization. */
export function createBasicTitleClip(workspace, options) {
    requireConstructorWorkspace(workspace);
    return {
        ...constructedClip(options, "title", `template:${BASIC_TITLE_UID}`),
        text: options.text,
        textStyle: options.style ?? defaultTextStyle(),
    };
}
/** Create a native pop-on caption with explicit interchange metadata. */
export function createCaptionClip(workspace, options) {
    requireConstructorWorkspace(workspace);
    return {
        ...constructedClip(options, "caption", null),
        text: options.text,
        textStyle: options.style ?? { ...defaultTextStyle(), fontSize: 48 },
        caption: options.caption ?? {
            displayStyle: "pop-on",
            placement: "bottom",
            alignment: "center",
            role: "iTT.en-US.dialogue",
        },
    };
}
/** Create the only renderer-certified generator, Custom Solid, with bounded straight RGBA. */
export function createCustomSolidClip(workspace, options) {
    requireConstructorWorkspace(workspace);
    for (const [channel, value] of Object.entries(options.color)) {
        if (!Number.isFinite(value) || value < 0 || value > 1) {
            throw new Error(`Custom Solid ${channel} must be in [0, 1].`);
        }
    }
    return {
        ...constructedClip(options, "generator", `template:${CUSTOM_SOLID_UID}`),
        generatorColor: options.color,
    };
}
function maskBase(options, kind, parameters, parameterNames, allowedAnimatedKeys, validateAnimatedValue, data = null) {
    if (!options.id.trim())
        throw new Error("Mask ID must be non-empty.");
    const keyframes = options.parameterKeyframes ?? {};
    for (const key of Object.keys(keyframes)) {
        if (!allowedAnimatedKeys.has(key) || !Object.hasOwn(parameters, key)) {
            throw new Error(`${kind} mask parameter ${key} is not animatable.`);
        }
        let previous = -Infinity;
        for (const frame of keyframes[key] ?? []) {
            if (!Number.isFinite(frame.time.seconds) || frame.time.seconds <= previous) {
                throw new Error(`${kind} mask parameter ${key} keyframes must use finite, increasing times.`);
            }
            validateAnimatedValue[key]?.(frame.value, `${kind} mask parameter ${key} keyframe`);
            previous = frame.time.seconds;
        }
    }
    return {
        id: options.id,
        kind,
        name: options.name || (kind === "shape" ? "Shape Mask" : kind === "draw" ? "Draw Mask" : kind === "color" ? "Color Mask" : "Luma Mask"),
        enabled: options.enabled ?? true,
        blendMode: options.blendMode ?? "add",
        parameters,
        parameterNames,
        parameterKeyframes: keyframes,
        data,
    };
}
function boundedMaskNumber(value, label, minimum, maximum) {
    if (typeof value !== "number" || !Number.isFinite(value) || value < minimum || value > maximum) {
        throw new Error(`${label} must be a finite number in [${minimum}, ${maximum}].`);
    }
}
function boundedMaskPoint(value, label, minimum, maximum) {
    if (!value || typeof value !== "object" || Array.isArray(value) || !("x" in value) || !("y" in value)) {
        throw new Error(`${label} must be an x/y point.`);
    }
    boundedMaskNumber(value.x, `${label} x`, minimum, maximum);
    boundedMaskNumber(value.y, `${label} y`, minimum, maximum);
}
/** Create the complete numeric Shape Mask contract admitted by Bladeworks. */
export function createShapeMask(options) {
    const parameters = {
        "160": options.radius ?? { x: 160, y: 160 },
        "201": options.position ?? { x: 0, y: 0 },
        "202": options.rotation ?? 0,
        "159": options.curvature ?? 1,
        "102": options.feather ?? 0,
        "103": options.opacity ?? 1,
        "104": options.falloff ?? 1,
    };
    const validators = {
        "160": (value, label) => boundedMaskPoint(value, label, 0, 32768),
        "201": (value, label) => boundedMaskPoint(value, label, -32768, 32768),
        "202": (value, label) => boundedMaskNumber(value, label, -3600, 3600),
        "159": (value, label) => boundedMaskNumber(value, label, 0, 1),
        "102": (value, label) => boundedMaskNumber(value, label, 0, 8192),
        "103": (value, label) => boundedMaskNumber(value, label, 0, 1),
        "104": (value, label) => boundedMaskNumber(value, label, 0.1, 8),
    };
    for (const [key, value] of Object.entries(parameters))
        validators[key](value, `Shape Mask ${key}`);
    return maskBase(options, "shape", parameters, {
        "160": "Radius", "201": "Position", "202": "Rotation", "159": "Curvature",
        "102": "Feather", "103": "Opacity", "104": "Falloff",
    }, new Set(Object.keys(parameters)), validators);
}
/** Create Bladeworks's bounded convex polygon Draw Mask. */
export function createDrawMask(options) {
    if (options.points.length < 3 || options.points.length > 64) {
        throw new Error("Draw Mask requires 3..64 points.");
    }
    if (options.points.some((point) => !Number.isFinite(point.x) || !Number.isFinite(point.y)
        || Math.abs(point.x) > 32768 || Math.abs(point.y) > 32768)) {
        throw new Error("Draw Mask points must be finite and inside the portable image plane.");
    }
    const crosses = options.points.map((point, index) => {
        const next = options.points[(index + 1) % options.points.length];
        const after = options.points[(index + 2) % options.points.length];
        return ((next.x - point.x) * (after.y - next.y)) - ((next.y - point.y) * (after.x - next.x));
    }).filter((value) => Math.abs(value) > EPSILON);
    if (crosses.length === 0 || crosses.some((value) => value * crosses[0] < 0)) {
        throw new Error("Draw Mask points must form a non-degenerate convex polygon.");
    }
    const points = options.points.map((point) => `${point.x},${point.y}`).join(";");
    const opacity = options.opacity ?? 1;
    boundedMaskNumber(opacity, "Draw Mask opacity", 0, 1);
    return maskBase(options, "draw", { points, opacity }, {
        points: "Points", opacity: "Opacity",
    }, new Set(["opacity"]), {
        opacity: (value, label) => boundedMaskNumber(value, label, 0, 1),
    });
}
function boundedUnit(value, name) {
    if (!Number.isFinite(value) || value < 0 || value > 1)
        throw new Error(`${name} must be in [0, 1].`);
    return value;
}
/** Create the reviewed portable RGB-distance isolation payload. */
export function createColorMask(options) {
    const data = JSON.stringify({
        abi: "spell-mask-isolation-v1",
        color: options.color.map((value, index) => boundedUnit(value, `Color Mask channel ${index + 1}`)),
        tolerance: boundedUnit(options.tolerance ?? 0.12, "Color Mask tolerance"),
        softness: boundedUnit(options.softness ?? 0.05, "Color Mask softness"),
        opacity: boundedUnit(options.opacity ?? 1, "Color Mask opacity"),
    });
    return maskBase(options, "color", {}, {}, new Set(), {}, data);
}
/** Create the reviewed portable luminance-range isolation payload. */
export function createLumaMask(options) {
    const minimum = boundedUnit(options.minimum ?? 0, "Luma Mask minimum");
    const maximum = boundedUnit(options.maximum ?? 1, "Luma Mask maximum");
    if (minimum > maximum)
        throw new Error("Luma Mask minimum cannot exceed maximum.");
    const data = JSON.stringify({
        abi: "spell-mask-isolation-v1",
        luma_min: minimum,
        luma_max: maximum,
        softness: boundedUnit(options.softness ?? 0.05, "Luma Mask softness"),
        opacity: boundedUnit(options.opacity ?? 1, "Luma Mask opacity"),
    });
    return maskBase(options, "luma", {}, {}, new Set(), {}, data);
}
/** Bind one or two renderer-certified filters to an ordered mask group. */
export function createMaskedEffect(id, masks, inside, outside, inverted = false) {
    if (!id.trim())
        throw new Error("Masked effect ID must be non-empty.");
    if (masks.length < 1 || masks.length > 32)
        throw new Error("Masked effect requires 1..32 masks.");
    return {
        kind: "masked-effect",
        maskedEffect: { id, enabled: true, inverted, masks, filters: outside ? [inside, outside] : [inside] },
    };
}
/**
 * Replace one editable Project in a cloned complete-library XML document.
 *
 * Main callers:
 * - LocalhostEditorRuntime after the pure magnetic reducer accepts a gesture.
 *
 * Why this exists:
 * Bladeworks's source endpoint accepts complete Info.fcpxml bytes. Rewriting
 * only the selected Project subtree preserves unrelated Events and Projects
 * while retaining one atomic whole-library PUT and history entry.
 */
export function replaceProjectInFCPXML(workspace, project) {
    const editability = workspace.editableProjects[project.id];
    if (!editability) {
        throw new Error(`Project ${project.id} is not part of this FCPXML workspace.`);
    }
    if (!editability.editable) {
        throw new Error(`Project ${project.id} is read-only: ${editability.reasons.join(" ")}`);
    }
    const freshDocument = parseDocument(workspace.xml);
    const reparsedWorkspace = workspaceOrigins.has(workspace) ? null : parseFCPXMLLibrary(workspace.xml);
    const originWorkspace = workspaceOrigins.get(workspace)
        ?? (reparsedWorkspace ? workspaceOrigins.get(reparsedWorkspace) : undefined);
    if (!originWorkspace) {
        throw new Error("FCPXML workspace provenance could not be reconstructed from its XML.");
    }
    const freshResources = parseResources(freshDocument);
    const freshProject = locateProject(freshDocument, project.id);
    const originalProject = originWorkspace.projects.get(project.id);
    if (!freshProject || !originalProject) {
        throw new Error(`Project ${project.id} is no longer present in this FCPXML workspace.`);
    }
    const sequence = firstChild(freshProject, "sequence");
    const spine = sequence ? firstChild(sequence, "spine") : null;
    if (!sequence || !spine) {
        throw new Error(`Project ${project.id} no longer contains an editable sequence spine.`);
    }
    const originalById = originalProject.clipOrigins;
    const elementByClipId = new Map();
    for (const clip of [...project.spine, ...project.connected]) {
        const directOrigin = originalById.get(clip.id);
        const inheritedOrigin = clip.xmlOriginId ? originalById.get(clip.xmlOriginId) : undefined;
        const origin = directOrigin ?? inheritedOrigin;
        const element = origin
            ? freshDocument.importNode(origin.element, true)
            : createTimelineElement(freshDocument, clip, freshResources, sequence);
        if (!directOrigin && inheritedOrigin && element.hasAttribute("uid")) {
            element.setAttribute("uid", crypto.randomUUID());
        }
        removeConnectedTimelineChildren(element);
        updateClipElement(element, clip, origin, originalProject.frameDuration, freshDocument, freshResources, sequence, originalProject.tcStart.seconds);
        elementByClipId.set(clip.id, element);
    }
    for (const clip of project.connected) {
        const anchor = project.spine.find((candidate) => candidate.id === clip.anchorId);
        const anchorElement = elementByClipId.get(clip.anchorId);
        const connectedElement = elementByClipId.get(clip.id);
        if (!anchor || !anchorElement || !connectedElement) {
            throw new Error(`Connected clip ${clip.id} has missing anchor ${clip.anchorId}.`);
        }
        connectedElement.setAttribute("lane", String(clip.lane || (clip.kind === "audio" ? -1 : 1)));
        connectedElement.setAttribute("offset", quantizedTime(anchor.sourceStart + clip.anchorOffset, originalProject.frameDuration));
        anchorElement.appendChild(connectedElement);
    }
    applyScopeEdits(project, originalProject, freshResources, elementByClipId, freshDocument, sequence);
    while (spine.firstChild) {
        spine.removeChild(spine.firstChild);
    }
    for (let index = 0; index < project.spine.length; index += 1) {
        const clip = project.spine[index];
        const element = elementByClipId.get(clip.id);
        if (!element) {
            throw new Error(`Primary clip ${clip.id} could not be serialized.`);
        }
        spine.appendChild(element);
        const next = project.spine[index + 1];
        if (!next) {
            continue;
        }
        const transition = project.transitions.find((candidate) => candidate.leftItemId === clip.id && candidate.rightItemId === next.id);
        if (transition) {
            spine.appendChild(createOrUpdateTransition(freshDocument, transition, originalProject.transitionOrigins.get(transition.id), clip, originalProject.frameDuration, freshResources, originalProject.tcStart.seconds));
        }
    }
    freshProject.setAttribute("name", project.name);
    sequence.setAttribute("duration", quantizedTime(projectDuration(project), originalProject.frameDuration));
    if (project.audioLayout && (originalProject.audioLayout !== null || project.audioLayout !== "stereo")) {
        sequence.setAttribute("audioLayout", project.audioLayout);
    }
    const xml = serializeDocument(freshDocument, workspace.xml);
    return parseFCPXMLLibrary(xml);
}
/**
 * Append one new Project to an Event in the complete-library XML.
 *
 * The Project is an empty sequence: duration 0s and no spine children.
 * The first edit inserts the first clip onto that empty storyline.
 *
 * Sequence format, timecode, and audio layout are copied from
 * `templateProjectId` (the Project the editor currently has open).
 *
 * Main callers:
 * - LocalhostEditorRuntime.createProject, which PUTs the result and then
 *   selects the new structural Project ref.
 */
export function addProjectToFCPXML(workspace, eventId, templateProjectId) {
    const document = parseDocument(workspace.xml);
    const event = locateEvent(document, eventId);
    if (!event) {
        throw new Error(`Event ${eventId} does not exist in this library.`);
    }
    const template = locateProject(document, templateProjectId);
    const templateSequence = template ? firstChild(template, "sequence") : null;
    const resources = parseResources(document);
    const formatId = templateSequence?.getAttribute("format")
        ?? [...resources.formats.keys()][0];
    if (!formatId) {
        throw new Error("This library has no sequence format to copy onto a new Project.");
    }
    const format = resources.formats.get(formatId);
    if (!format) {
        throw new Error(`Sequence format ${formatId} is missing from <resources>.`);
    }
    const existingNames = childrenNamed(event, "project").map((project) => project.getAttribute("name") || "");
    const name = uniqueUntitledProjectName(existingNames);
    const project = document.createElement("project");
    project.setAttribute("name", name);
    project.setAttribute("uid", crypto.randomUUID());
    const sequence = document.createElement("sequence");
    sequence.setAttribute("format", formatId);
    sequence.setAttribute("duration", "0s");
    sequence.setAttribute("tcStart", templateSequence?.getAttribute("tcStart") || "0s");
    sequence.setAttribute("tcFormat", templateSequence?.getAttribute("tcFormat") || "NDF");
    sequence.setAttribute("audioLayout", templateSequence?.getAttribute("audioLayout") === "mono" ? "mono" : "stereo");
    sequence.setAttribute("audioRate", templateSequence?.getAttribute("audioRate") || "48k");
    const renderFormat = templateSequence?.getAttribute("renderFormat");
    if (renderFormat) {
        sequence.setAttribute("renderFormat", renderFormat);
    }
    sequence.appendChild(document.createElement("spine"));
    project.appendChild(sequence);
    event.appendChild(project);
    const projectIndex = childrenNamed(event, "project").length;
    const projectId = `${eventId}/project[${projectIndex}]`;
    const updated = parseFCPXMLLibrary(serializeDocument(document, workspace.xml));
    if (!updated.bootstrap.projects[projectId]) {
        throw new Error(`Created Project ${projectId} but the reparsed library does not contain it.`);
    }
    return { workspace: updated, projectId };
}
/** Append one empty Event to a Library and return its structural Event ref. */
export function addEventToFCPXML(workspace, libraryId) {
    const document = parseDocument(workspace.xml);
    const match = /^library\[(\d+)]$/.exec(libraryId);
    const library = match?.[1]
        ? childrenNamed(document.documentElement, "library")[Number(match[1]) - 1] ?? null
        : null;
    if (!library) {
        throw new Error(`Library ${libraryId} does not exist.`);
    }
    const existingNames = childrenNamed(library, "event").map((event) => event.getAttribute("name") || "");
    const base = "New Event";
    let name = base;
    let suffix = 2;
    while (existingNames.includes(name)) {
        name = `${base} ${suffix}`;
        suffix += 1;
    }
    const event = document.createElement("event");
    event.setAttribute("name", name);
    library.appendChild(event);
    const eventId = `${libraryId}/event[${childrenNamed(library, "event").length}]`;
    const updated = parseFCPXMLLibrary(serializeDocument(document, workspace.xml));
    const created = updated.bootstrap.libraries
        .flatMap((candidate) => candidate.events)
        .find((candidate) => candidate.id === eventId);
    if (!created) {
        throw new Error(`Created Event ${eventId} but the reparsed library does not contain it.`);
    }
    return { workspace: updated, eventId };
}
function uniqueUntitledProjectName(existing) {
    const base = "Untitled Project";
    if (!existing.includes(base)) {
        return base;
    }
    let index = 2;
    while (existing.includes(`${base} ${index}`)) {
        index += 1;
    }
    return `${base} ${index}`;
}
function applyScopeEdits(project, projectOrigin, resources, rootElements, document, projectSequence) {
    const changedCompoundResourceIds = new Set();
    for (const [scopeId, scope] of Object.entries(project.scopes ?? {})) {
        const origin = projectOrigin.scopeOrigins.get(scopeId);
        if (!origin)
            throw new Error(`Nested scope ${scopeId} is not part of the source Project.`);
        if (JSON.stringify(scope) === JSON.stringify(origin.model))
            continue;
        if (!scope.editable)
            throw new Error(`Nested scope ${scopeId} is read-only: ${scope.reasons.join(" ")}`);
        const immutable = (value) => ({
            id: value.id, kind: value.kind, name: value.name, parentScopeId: value.parentScopeId,
            viaClipId: value.viaClipId, resourceId: value.resourceId, angleId: value.angleId,
            clock: value.clock,
        });
        if (JSON.stringify(immutable(scope)) !== JSON.stringify(immutable(origin.model))) {
            throw new Error(`Nested scope ${scopeId} changed immutable identity or clock metadata.`);
        }
        const target = locateFreshScope(scope, projectOrigin, resources, rootElements);
        if (!target)
            throw new Error(`Nested scope ${scopeId} could not be located in the fresh library XML.`);
        if (scope.kind === "audition-choice") {
            const clip = scope.spine[0];
            const clipOrigin = clip ? origin.childOrigins.get(clip.id) : undefined;
            if (!clip || !clipOrigin || scope.spine.length !== 1 || scope.connected.length !== 0) {
                throw new Error(`Audition choice ${scopeId} must preserve one complete choice.`);
            }
            updateClipElement(target, clip, clipOrigin, projectOrigin.frameDuration, document, resources, projectSequence);
            continue;
        }
        rebuildEditableScope(scope, origin, target, projectOrigin.frameDuration, document, resources, projectSequence);
        if (scope.kind === "compound" && scope.resourceId)
            changedCompoundResourceIds.add(scope.resourceId);
    }
    for (const resourceId of changedCompoundResourceIds) {
        reconcileCompoundSourceWindows(document, rootElements, resources, resourceId);
    }
}
/**
 * Keep every instance source window inside a newly resized compound resource.
 *
 * A resource timeline edit can shorten its `<sequence duration>`, while its
 * `<ref-clip>` instances live elsewhere in the library. FCP validates each
 * instance against that shared source range. We retain an instance duration
 * whenever it still fits and move only its source start. If the instance is
 * longer than the complete resource, its duration must shrink to the largest
 * valid window rather than leaving XML that Bladeworks will reject.
 *
 * Main callers:
 * - `applyScopeEdits`, after every changed scope has been rebuilt so nested and
 *   top-level instances are reconciled against the final resource durations.
 */
function reconcileCompoundSourceWindows(document, rootElements, resources, resourceId) {
    const media = resources.media.get(resourceId);
    const sequence = media ? firstChild(media, "sequence") : null;
    if (!sequence)
        throw new Error(`Compound resource ${resourceId} has no sequence to reconcile.`);
    const resourceStartRaw = sequence.getAttribute("tcStart") ?? sequence.getAttribute("start") ?? "0s";
    const resourceDurationRaw = requireAttribute(sequence, "duration", `compound resource ${resourceId}`);
    const resourceStart = parseRationalTime(resourceStartRaw, `${resourceId} source start`);
    const resourceDuration = parseRationalTime(resourceDurationRaw, `${resourceId} source duration`);
    if (resourceDuration.numerator <= 0n) {
        throw new Error(`Compound resource ${resourceId} has no positive source range for its instances.`);
    }
    const resourceEnd = addRational(resourceStart, resourceDuration);
    const instances = new Set(descendantsNamed(document, "ref-clip"));
    for (const root of rootElements.values()) {
        if (tagName(root) === "ref-clip")
            instances.add(root);
        for (const nested of descendantsNamed(root, "ref-clip"))
            instances.add(nested);
    }
    for (const instance of instances) {
        if (instance.getAttribute("ref") !== resourceId)
            continue;
        const name = instance.getAttribute("name") || resourceId;
        const startRaw = instance.getAttribute("start") ?? resourceStartRaw;
        const durationRaw = requireAttribute(instance, "duration", `compound instance ${name}`);
        const start = parseRationalTime(startRaw, `${name} source start`);
        const duration = parseRationalTime(durationRaw, `${name} source duration`);
        if (duration.numerator <= 0n) {
            throw new Error(`Compound instance ${name} has a non-positive duration.`);
        }
        if (rationalSeconds(duration) > rationalSeconds(resourceDuration) + EPSILON) {
            instance.setAttribute("start", resourceStartRaw);
            instance.setAttribute("duration", resourceDurationRaw);
        }
        else {
            const maximumStart = subtractRational(resourceEnd, duration);
            if (rationalSeconds(start) < rationalSeconds(resourceStart) - EPSILON) {
                instance.setAttribute("start", resourceStartRaw);
            }
            else if (rationalSeconds(start) > rationalSeconds(maximumStart) + EPSILON) {
                instance.setAttribute("start", rationalString(maximumStart));
            }
        }
        if (instance.hasAttribute("audioStart") || instance.hasAttribute("audioDuration")) {
            const videoStartRaw = instance.getAttribute("start") ?? resourceStartRaw;
            const videoDurationRaw = requireAttribute(instance, "duration", `compound instance ${name}`);
            const audioStartRaw = instance.getAttribute("audioStart") ?? videoStartRaw;
            const audioDurationRaw = instance.getAttribute("audioDuration") ?? videoDurationRaw;
            const audioStart = parseRationalTime(audioStartRaw, `${name} audio source start`);
            const audioDuration = parseRationalTime(audioDurationRaw, `${name} audio source duration`);
            if (audioDuration.numerator <= 0n) {
                throw new Error(`Compound instance ${name} has a non-positive audio duration.`);
            }
            if (rationalSeconds(audioDuration) > rationalSeconds(resourceDuration) + EPSILON) {
                instance.setAttribute("audioStart", resourceStartRaw);
                instance.setAttribute("audioDuration", resourceDurationRaw);
            }
            else {
                const maximumAudioStart = subtractRational(resourceEnd, audioDuration);
                if (rationalSeconds(audioStart) < rationalSeconds(resourceStart) - EPSILON) {
                    instance.setAttribute("audioStart", resourceStartRaw);
                }
                else if (rationalSeconds(audioStart) > rationalSeconds(maximumAudioStart) + EPSILON) {
                    instance.setAttribute("audioStart", rationalString(maximumAudioStart));
                }
            }
        }
    }
}
function rebuildEditableScope(scope, origin, target, frameDuration, document, resources, projectSequence) {
    const elements = new Map();
    for (const clip of [...scope.spine, ...scope.connected]) {
        const clipOrigin = origin.childOrigins.get(clip.id);
        const element = clipOrigin
            ? document.importNode(clipOrigin.element, true)
            : createTimelineElement(document, clip, resources, projectSequence);
        removeConnectedTimelineChildren(element);
        updateClipElement(element, clip, clipOrigin, frameDuration, document, resources, projectSequence);
        elements.set(clip.id, element);
    }
    if (scope.kind !== "sync") {
        for (const clip of scope.connected) {
            const anchor = elements.get(clip.anchorId);
            const element = elements.get(clip.id);
            if (!anchor || !element || !scope.spine.some((candidate) => candidate.id === clip.anchorId)) {
                throw new Error(`Nested connected clip ${clip.id} has missing scope anchor ${clip.anchorId}.`);
            }
            element.setAttribute("lane", String(clip.lane || (clip.kind === "audio" ? -1 : 1)));
            element.setAttribute("offset", quantizedTime(scope.spine.find((candidate) => candidate.id === clip.anchorId).sourceStart + clip.anchorOffset, frameDuration));
            anchor.appendChild(element);
        }
    }
    const timelineChildren = childElements(target).filter((child) => TIMELINE_TAGS.has(tagName(child)) || tagName(child) === "transition");
    let boundary = null;
    const lastTimeline = timelineChildren.at(-1);
    if (lastTimeline) {
        let sibling = lastTimeline.nextSibling;
        while (sibling && sibling.nodeType === 3 && !(sibling.textContent ?? "").trim())
            sibling = sibling.nextSibling;
        boundary = sibling;
    }
    for (const child of timelineChildren)
        target.removeChild(child);
    const directById = new Map([...scope.spine, ...scope.connected].map((clip) => [clip.id, clip]));
    const direct = scope.kind === "sync"
        ? scope.childOrder.map((id) => {
            const clip = directById.get(id);
            if (!clip)
                throw new Error(`Sync scope ${scope.id} child order refers to missing clip ${id}.`);
            return clip;
        })
        : [...scope.spine];
    if (scope.kind === "sync" && new Set(scope.childOrder).size !== directById.size) {
        throw new Error(`Sync scope ${scope.id} child order must contain every direct clip exactly once.`);
    }
    const serializedTransitions = new Set();
    for (const [index, clip] of direct.entries()) {
        const element = elements.get(clip.id);
        if (!element)
            throw new Error(`Nested scope ${scope.id} could not serialize child ${clip.id}.`);
        if (scope.kind === "sync" && "anchorId" in clip) {
            const connected = clip;
            element.setAttribute("lane", String(connected.lane));
            element.setAttribute("offset", quantizedTime(scope.clock.tcStart.seconds + connected.sourceStart + connected.anchorOffset, frameDuration));
        }
        target.insertBefore(element, boundary);
        const next = direct[index + 1];
        if (!next)
            continue;
        const transition = scope.transitions.find((candidate) => candidate.leftItemId === clip.id && candidate.rightItemId === next.id);
        if (transition) {
            if (serializedTransitions.has(transition.id)) {
                throw new Error(`Nested transition ${transition.id} is duplicated.`);
            }
            serializedTransitions.add(transition.id);
            target.insertBefore(createOrUpdateTransition(document, transition, origin.transitionOrigins.get(transition.id), clip, frameDuration, resources, scope.clock.tcStart.seconds), boundary);
        }
    }
    if (serializedTransitions.size !== scope.transitions.length) {
        const dangling = scope.transitions.find((transition) => !serializedTransitions.has(transition.id));
        throw new Error(`Nested transition ${dangling?.id ?? "unknown"} does not connect adjacent direct clips in ${scope.id}.`);
    }
    if (scope.kind === "compound") {
        const sequence = target.parentElement;
        const duration = scope.spine.reduce((maximum, clip) => Math.max(maximum, clip.timelineStart + clip.duration), 0);
        if (sequence && tagName(sequence) === "sequence")
            sequence.setAttribute("duration", quantizedTime(duration, frameDuration));
    }
}
function locateFreshScope(scope, projectOrigin, resources, rootElements, visiting = new Set()) {
    if (scope.kind === "compound" && scope.resourceId) {
        const media = resources.media.get(scope.resourceId);
        const sequence = media ? firstChild(media, "sequence") : null;
        return sequence ? firstChild(sequence, "spine") : null;
    }
    if (scope.kind === "multicam-angle" && scope.resourceId && scope.angleId) {
        const media = resources.media.get(scope.resourceId);
        const multicam = media ? firstChild(media, "multicam") : null;
        return multicam
            ? childrenNamed(multicam, "mc-angle").find((angle) => angle.getAttribute("angleID") === scope.angleId) ?? null
            : null;
    }
    const root = locateFreshClipElement(scope.viaClipId, projectOrigin, resources, rootElements, visiting);
    if (!root)
        return null;
    if (scope.kind === "sync")
        return root;
    if (scope.kind === "audition-choice") {
        const baseline = projectOrigin.clipOrigins.get(scope.viaClipId)?.model.container;
        if (!baseline || baseline.kind !== "audition")
            return null;
        const index = baseline.choiceScopeIds.indexOf(scope.id);
        return childElements(root).filter((child) => TIMELINE_TAGS.has(tagName(child)))[index] ?? null;
    }
    return null;
}
function locateFreshClipElement(clipId, projectOrigin, resources, rootElements, visiting) {
    const root = rootElements.get(clipId);
    if (root)
        return root;
    if (visiting.has(clipId))
        throw new Error(`Recursive scope cycle encountered while locating ${clipId}.`);
    visiting.add(clipId);
    for (const scopeOrigin of projectOrigin.scopeOrigins.values()) {
        if (!scopeOrigin.childOrigins.has(clipId))
            continue;
        const parent = locateFreshScope(scopeOrigin.model, projectOrigin, resources, rootElements, visiting);
        if (!parent)
            return null;
        if (scopeOrigin.model.kind === "audition-choice")
            return parent;
        const index = scopeOrigin.childOrder.indexOf(clipId);
        return childElements(parent).filter((child) => TIMELINE_TAGS.has(tagName(child)))[index] ?? null;
    }
    return null;
}
function locateEvent(document, eventId) {
    const match = /^library\[(\d+)]\/event\[(\d+)]$/.exec(eventId);
    if (!match?.[1] || !match[2]) {
        return null;
    }
    const library = childrenNamed(document.documentElement, "library")[Number(match[1]) - 1];
    return library ? childrenNamed(library, "event")[Number(match[2]) - 1] ?? null : null;
}
/** Encode one validated bundle-relative filesystem path as an FCPXML URI path. */
function bundleMediaLocator(rawPath, context) {
    const segments = rawPath.split("/");
    if (segments.some((segment) => segment === "" || segment === "." || segment === "..")) {
        throw new Error(`${context} has unsafe media path ${JSON.stringify(rawPath)}.`);
    }
    return segments.map((segment) => encodeURIComponent(segment)).join("/");
}
/** Return resource IDs carrying at least one Favorite rating in an Event. */
export function favoriteAssetIdsForEvent(workspace, eventId) {
    const origin = workspaceOrigins.get(workspace);
    if (!origin)
        throw new Error("Favorite lookup requires a parsed FCPXML workspace.");
    const event = locateEvent(origin.document, eventId);
    if (!event)
        throw new Error(`Event ${eventId} does not exist.`);
    const resources = firstChild(origin.document.documentElement, "resources");
    const durations = new Map((resources ? childrenNamed(resources, "asset") : []).map((asset) => [
        asset.getAttribute("id"),
        { start: asset.getAttribute("start") || "0s", duration: asset.getAttribute("duration") },
    ]));
    return new Set(childrenNamed(event, "asset-clip").flatMap((clip) => {
        const ref = clip.getAttribute("ref");
        const resource = ref ? durations.get(ref) : undefined;
        const resourceDuration = resource?.duration;
        if (!ref || !resource || !resourceDuration)
            return [];
        const favorite = childrenNamed(clip, "rating").some((rating) => rating.getAttribute("value") === "favorite"
            && sameRationalTime(rating.getAttribute("start") || resource.start, resource.start, `${ref} Favorite start`)
            && sameRationalTime(rating.getAttribute("duration") || resourceDuration, resourceDuration, `${ref} Favorite duration`));
        return favorite ? [ref] : [];
    }));
}
/**
 * Add or remove one whole-clip Favorite rating in the selected Event.
 *
 * Ratings belong to Event clip instances, never resource `<asset>` elements.
 * If media exists only as a resource, create the minimal Event-level
 * `asset-clip` required to carry its editorial metadata. Reject ratings and
 * unrelated Favorite ranges remain untouched.
 */
export function setEventAssetFavorite(workspace, eventId, assetId, favorite, mediaAsset) {
    const origin = workspaceOrigins.get(workspace);
    if (!origin)
        throw new Error("Favorite editing requires a parsed FCPXML workspace.");
    const document = origin.document.cloneNode(true);
    const event = locateEvent(document, eventId);
    if (!event)
        throw new Error(`Event ${eventId} does not exist.`);
    const resources = firstChild(document.documentElement, "resources");
    let asset = resources
        ? childrenNamed(resources, "asset").find((candidate) => candidate.getAttribute("id") === assetId)
        : undefined;
    if (!asset && favorite && mediaAsset?.sourcePath && resources) {
        const mediaLocator = bundleMediaLocator(mediaAsset.sourcePath, "Media resource");
        let offset = 1;
        let resourceId = `bf_asset_${offset}`;
        const ids = new Set(childElements(resources).flatMap((element) => {
            const id = element.getAttribute("id");
            return id ? [id] : [];
        }));
        while (ids.has(resourceId))
            resourceId = `bf_asset_${++offset}`;
        asset = document.createElement("asset");
        asset.setAttribute("id", resourceId);
        asset.setAttribute("name", mediaAsset.name);
        asset.setAttribute("start", "0s");
        asset.setAttribute("duration", `${mediaAsset.duration}s`);
        asset.setAttribute("hasVideo", mediaAsset.kind === "audio" ? "0" : "1");
        asset.setAttribute("hasAudio", mediaAsset.kind === "audio" || (mediaAsset.kind === "video" && mediaAsset.hasAudio !== false) ? "1" : "0");
        if (mediaAsset.kind !== "audio" && mediaAsset.width && mediaAsset.height) {
            const parsedResources = parseResources(document);
            const frameDuration = mediaAsset.frameDurationRaw
                ? parseRationalTime(mediaAsset.frameDurationRaw, `Media resource ${mediaAsset.name}`)
                : { numerator: 1n, denominator: 30n };
            let format = [...parsedResources.formats.values()].find((candidate) => candidate.width === mediaAsset.width
                && candidate.height === mediaAsset.height
                && candidate.frameDuration.numerator === frameDuration.numerator
                && candidate.frameDuration.denominator === frameDuration.denominator);
            if (!format) {
                let formatOffset = 1;
                let formatId = `bf_fmt_${formatOffset}`;
                const resourceIds = new Set(childElements(resources).flatMap((element) => {
                    const id = element.getAttribute("id");
                    return id ? [id] : [];
                }));
                while (resourceIds.has(formatId))
                    formatId = `bf_fmt_${++formatOffset}`;
                const formatElement = document.createElement("format");
                formatElement.setAttribute("id", formatId);
                formatElement.setAttribute("frameDuration", rationalString(frameDuration));
                formatElement.setAttribute("width", String(mediaAsset.width));
                formatElement.setAttribute("height", String(mediaAsset.height));
                resources.appendChild(formatElement);
                format = { id: formatId, frameDuration, width: mediaAsset.width, height: mediaAsset.height };
            }
            asset.setAttribute("format", format.id);
        }
        const mediaRep = document.createElement("media-rep");
        mediaRep.setAttribute("kind", "original-media");
        mediaRep.setAttribute("src", mediaLocator);
        asset.appendChild(mediaRep);
        resources.appendChild(asset);
        assetId = resourceId;
    }
    if (!asset)
        throw new Error(`Media resource ${assetId} does not exist in FCPXML.`);
    const duration = asset.getAttribute("duration");
    if (!duration)
        throw new Error(`Media resource ${assetId} has no duration for a whole-clip Favorite.`);
    const matchingClips = childrenNamed(event, "asset-clip").filter((candidate) => candidate.getAttribute("ref") === assetId);
    let clip = matchingClips[0];
    if (!clip && favorite) {
        clip = document.createElement("asset-clip");
        clip.setAttribute("ref", assetId);
        clip.setAttribute("name", asset.getAttribute("name") || assetId);
        clip.setAttribute("start", asset.getAttribute("start") || "0s");
        clip.setAttribute("duration", duration);
        event.appendChild(clip);
    }
    if (!clip)
        return workspace;
    const wholeStart = asset.getAttribute("start") || "0s";
    for (const matchingClip of matchingClips.length > 0 ? matchingClips : [clip]) {
        const wholeFavorites = childrenNamed(matchingClip, "rating").filter((rating) => rating.getAttribute("value") === "favorite"
            && sameRationalTime(rating.getAttribute("start") || wholeStart, wholeStart, `${assetId} Favorite start`)
            && sameRationalTime(rating.getAttribute("duration") || duration, duration, `${assetId} Favorite duration`));
        for (const rating of wholeFavorites)
            matchingClip.removeChild(rating);
    }
    if (favorite) {
        const rating = document.createElement("rating");
        rating.setAttribute("start", wholeStart);
        rating.setAttribute("duration", duration);
        rating.setAttribute("value", "favorite");
        clip.insertBefore(rating, firstChild(clip, "metadata"));
    }
    return parseFCPXMLLibrary(serializeDocument(document, workspace.xml));
}
function locateProject(document, ref) {
    const match = /^library\[(\d+)]\/event\[(\d+)]\/project\[(\d+)]$/.exec(ref);
    if (!match?.[1] || !match[2] || !match[3]) {
        return null;
    }
    const library = childrenNamed(document.documentElement, "library")[Number(match[1]) - 1];
    const event = library ? childrenNamed(library, "event")[Number(match[2]) - 1] : undefined;
    return event ? childrenNamed(event, "project")[Number(match[3]) - 1] ?? null : null;
}
function removeConnectedTimelineChildren(element) {
    if (tagName(element) === "sync-clip" || tagName(element) === "audition")
        return;
    for (const child of childElements(element)) {
        if (TIMELINE_TAGS.has(tagName(child))) {
            element.removeChild(child);
        }
    }
}
function createTimelineElement(document, clip, resources, sequence) {
    if (clip.container)
        throw new Error(`Nested container ${clip.id} cannot be inserted without a canonical source scope.`);
    const tag = clip.kind === "gap" ? "gap"
        : clip.kind === "title" ? "title"
            : clip.kind === "caption" ? "caption"
                : clip.kind === "generator" ? "video"
                    : clip.kind === "audio" ? "audio" : "asset-clip";
    const element = document.createElement(tag);
    if (clip.kind !== "gap" && clip.kind !== "caption") {
        element.setAttribute("ref", ensureAssetResource(document, clip, resources, sequence));
    }
    return element;
}
function ensureAssetResource(document, clip, resources, sequence) {
    const assetId = clip.assetId;
    if (assetId && resources.assets.has(assetId)) {
        return assetId;
    }
    if (assetId && resources.effects.has(assetId)) {
        return assetId;
    }
    if (assetId?.startsWith("template:")) {
        const uid = assetId.slice("template:".length);
        const expectedName = uid === BASIC_TITLE_UID ? "Basic Title" : uid === CUSTOM_SOLID_UID ? "Custom" : clip.name;
        return ensureEffectResource(document, resources.effects, expectedName, "", uid);
    }
    if (!assetId?.startsWith("media:")) {
        throw new Error(`Clip ${clip.id} refers to unknown asset ${String(assetId)}.`);
    }
    // The ``media:`` asset ID carries the whole bundle-relative filesystem path that the
    // backend media inventory reports (e.g. ``Media/clip.mp4``), including the
    // ``Media/`` directory segment. Do not prepend ``Media/`` again; encode each
    // existing segment for the FCPXML URI instead. Prepending the directory
    // yields ``Media/Media/clip.mp4`` and the accepted Project opens with missing
    // media.
    const bundlePath = assetId.slice("media:".length);
    const mediaLocator = bundleMediaLocator(bundlePath, `Clip ${clip.id}`);
    const resourcesElement = firstChild(document.documentElement, "resources");
    if (!resourcesElement) {
        throw new Error("FCPXML has no <resources> element for inserted media.");
    }
    const existingForPath = [...resources.assets.values()].find((asset) => {
        const existingLocator = asset.source?.replace(/^\.\//, "");
        return existingLocator === bundlePath || existingLocator === mediaLocator;
    });
    if (existingForPath) {
        return existingForPath.id;
    }
    let offset = 1;
    let id = `bf_asset_${offset}`;
    const resourceIds = new Set(childElements(resourcesElement).flatMap((element) => {
        const resourceId = element.getAttribute("id");
        return resourceId ? [resourceId] : [];
    }));
    while (resourceIds.has(id)) {
        offset += 1;
        id = `bf_asset_${offset}`;
    }
    const asset = document.createElement("asset");
    asset.setAttribute("id", id);
    asset.setAttribute("name", clip.name);
    asset.setAttribute("start", "0s");
    const sourceDuration = clip.kind === "image" ? 0 : clip.sourceDuration ?? clip.duration;
    asset.setAttribute("duration", quantizedTime(sourceDuration, projectFrameDuration(sequence, resources.formats)));
    asset.setAttribute("hasVideo", clip.kind === "audio" ? "0" : "1");
    // Audio-only clips always have audio; images never do. For a video clip use
    // the probed flag carried on the clip: a video-only source (hasAudio === false)
    // must declare hasAudio="0", or the backend audio compiler builds a source for
    // a stream that does not exist. When the flag is absent (fixtures, or clips not
    // sourced from a probe) fall back to the historical "video implies audio".
    const videoHasAudio = clip.hasAudio === false ? "0" : "1";
    const hasAudioAttr = clip.kind === "audio" ? "1" : clip.kind === "video" ? videoHasAudio : "0";
    asset.setAttribute("hasAudio", hasAudioAttr);
    const formatId = ensureMediaFormat(document, resourcesElement, resources, sequence, clip);
    if (formatId) {
        asset.setAttribute("format", formatId);
    }
    const mediaRep = document.createElement("media-rep");
    mediaRep.setAttribute("kind", "original-media");
    mediaRep.setAttribute("src", mediaLocator);
    asset.appendChild(mediaRep);
    resourcesElement.appendChild(asset);
    resources.assets.set(id, {
        id,
        name: clip.name,
        kind: clip.kind === "audio" ? "audio" : clip.kind === "image" ? "image" : "video",
        start: 0,
        duration: sourceDuration,
        source: mediaLocator,
        formatId,
    });
    return id;
}
function ensureMediaFormat(document, resourcesElement, resources, sequence, clip) {
    const sequenceFormatId = sequence.getAttribute("format");
    const sequenceFormat = sequenceFormatId ? resources.formats.get(sequenceFormatId) : undefined;
    const width = clip.sourceWidth;
    const height = clip.sourceHeight;
    if (!width || !height) {
        return sequenceFormatId;
    }
    const frameDuration = clip.sourceFrameDurationRaw
        ? parseRationalTime(clip.sourceFrameDurationRaw, `inserted media ${clip.name}`)
        : (sequenceFormat?.frameDuration ?? { numerator: 1n, denominator: 30n });
    for (const existing of resources.formats.values()) {
        if (existing.width === width
            && existing.height === height
            && existing.frameDuration.numerator === frameDuration.numerator
            && existing.frameDuration.denominator === frameDuration.denominator) {
            return existing.id;
        }
    }
    let offset = 1;
    let id = `bf_fmt_${offset}`;
    const resourceIds = new Set(childElements(resourcesElement).flatMap((element) => {
        const resourceId = element.getAttribute("id");
        return resourceId ? [resourceId] : [];
    }));
    while (resourceIds.has(id)) {
        offset += 1;
        id = `bf_fmt_${offset}`;
    }
    const format = document.createElement("format");
    format.setAttribute("id", id);
    format.setAttribute("frameDuration", rationalString(frameDuration));
    format.setAttribute("width", String(width));
    format.setAttribute("height", String(height));
    resourcesElement.appendChild(format);
    resources.formats.set(id, { id, frameDuration, width, height });
    return id;
}
function projectFrameDuration(sequence, formats) {
    return projectFormat(sequence, formats).frameDuration;
}
function updateClipElement(element, clip, origin, frameDuration, document, resources, sequence, timelineOffsetBase = 0) {
    assertRepresentableChanges(clip, origin?.model);
    element.setAttribute("name", clip.name);
    element.setAttribute("start", origin
        ? preservedOrQuantized(clip.sourceStart, origin.model.sourceStart, origin.start, frameDuration)
        : quantizedTime(clip.sourceStart, frameDuration));
    element.setAttribute("duration", origin
        ? preservedOrQuantized(clip.duration, origin.model.duration, origin.duration, frameDuration)
        : quantizedTime(clip.duration, frameDuration));
    if (!("anchorId" in clip)) {
        element.setAttribute("offset", origin && Math.abs(clip.timelineStart - origin.model.timelineStart) <= EPSILON
            ? origin.offset
            : quantizedTime(clip.timelineStart + timelineOffsetBase, frameDuration));
        element.removeAttribute("lane");
    }
    if (clip.container) {
        if (!origin)
            throw new Error(`Nested container ${clip.id} has no source XML to preserve.`);
    }
    else if (clip.kind !== "gap" && clip.kind !== "caption") {
        element.setAttribute("ref", ensureAssetResource(document, clip, resources, sequence));
    }
    else {
        element.removeAttribute("ref");
    }
    updateContainerState(element, clip, origin?.model, document);
    updateSemanticAudioAttributes(element, clip, frameDuration);
    updateTransform(element, clip);
    updateCrop(element, clip, document);
    updateConform(element, clip, document);
    updateDistort(element, clip, document);
    updateAudio(element, clip, origin?.model, document, frameDuration);
    updateTimeMap(element, clip, document, frameDuration);
    updateMarkers(element, clip, origin, document, frameDuration);
    updateEffectStack(element, clip, origin, document, resources.effects, frameDuration);
    updateKeyframes(element, clip, document, frameDuration);
    if (clip.kind === "title" || clip.kind === "caption") {
        updateTitleText(element, clip, document);
    }
    if (clip.kind === "generator") {
        updateGeneratorColor(element, clip, document);
    }
}
function updateContainerState(element, clip, original, document) {
    const state = clip.container ?? null;
    const baseline = original?.container ?? null;
    if (JSON.stringify(state) === JSON.stringify(baseline))
        return;
    if (!state || !baseline || state.kind !== baseline.kind) {
        throw new Error(`Clip ${clip.id} changed its recursive container identity.`);
    }
    if (state.kind === "compound") {
        if (baseline.kind !== "compound")
            throw new Error(`Compound ${clip.id} changed container kind.`);
        if (state.resourceId !== baseline.resourceId || state.scopeId !== baseline.scopeId) {
            throw new Error(`Compound ${clip.id} changed its shared resource identity.`);
        }
        return;
    }
    if (state.kind === "multicam") {
        if (baseline.kind !== "multicam")
            throw new Error(`Multicam ${clip.id} changed container kind.`);
        if (state.resourceId !== baseline.resourceId
            || JSON.stringify(state.angleScopeIds) !== JSON.stringify(baseline.angleScopeIds)) {
            throw new Error(`Multicam ${clip.id} changed its angle resource catalog.`);
        }
        const originalSources = childrenNamed(element, "mc-source");
        removeDirectChildren(element, new Set(["mc-source"]));
        for (const [angleId, srcEnable] of [[state.videoAngleId, "video"], [state.audioAngleId, "audio"]]) {
            if (!angleId || !Object.hasOwn(state.angleScopeIds, angleId)) {
                throw new Error(`Multicam ${clip.id} selects unknown ${srcEnable} angle ${String(angleId)}.`);
            }
            const originalSource = originalSources.find((candidate) => (candidate.getAttribute("srcEnable") || "").split(/[, ]+/).includes(srcEnable));
            const source = originalSource
                ? document.importNode(originalSource, true)
                : document.createElement("mc-source");
            source.setAttribute("angleID", angleId);
            source.setAttribute("srcEnable", srcEnable);
            element.appendChild(source);
        }
        return;
    }
    if (state.kind === "sync") {
        if (baseline.kind !== "sync")
            throw new Error(`Sync clip ${clip.id} changed container kind.`);
        if (state.scopeId !== baseline.scopeId
            || state.sources.map((source) => source.sourceId).join("\u001f") !== baseline.sources.map((source) => source.sourceId).join("\u001f")) {
            throw new Error(`Sync clip ${clip.id} changed its source catalog.`);
        }
        const sourceElements = childrenNamed(element, "sync-source");
        for (const [index, source] of state.sources.entries()) {
            const target = sourceElements[index];
            if (!target || target.getAttribute("sourceID") !== source.sourceId) {
                throw new Error(`Sync clip ${clip.id} source order no longer matches its XML.`);
            }
            const role = firstChild(target, "audio-role-source");
            if (source.role === null) {
                if (role)
                    throw new Error(`Sync source ${source.sourceId} cannot drop its role metadata.`);
            }
            else {
                const roleTarget = role ?? document.createElement("audio-role-source");
                roleTarget.setAttribute("role", source.role);
                roleTarget.setAttribute("enabled", source.enabled ? "1" : "0");
                roleTarget.setAttribute("active", source.active ? "1" : "0");
                if (!role)
                    target.appendChild(roleTarget);
            }
        }
        return;
    }
    if (baseline.kind !== "audition")
        throw new Error(`Audition ${clip.id} changed container kind.`);
    if (state.choiceScopeIds.join("\u001f") !== baseline.choiceScopeIds.join("\u001f")
        || !state.activeChoiceId || !state.choiceScopeIds.includes(state.activeChoiceId)) {
        throw new Error(`Audition ${clip.id} changed or selected an unknown choice catalog.`);
    }
    const choices = childElements(element).filter((child) => TIMELINE_TAGS.has(tagName(child)));
    for (const [index, choice] of choices.entries()) {
        choice.setAttribute("enabled", state.choiceScopeIds[index] === state.activeChoiceId ? "1" : "0");
    }
}
function assertRepresentableChanges(clip, original) {
    const baselineVideo = original?.video ?? defaultVideo();
    const baselineAudio = original?.audio ?? defaultAudio();
    const unsupported = [
        ["video color controls", clip.video.color, baselineVideo.color],
        ["color conform controls", [clip.video.colorConform, clip.video.colorConformType], [baselineVideo.colorConform, baselineVideo.colorConformType]],
        ["stabilization", clip.video.stabilization, baselineVideo.stabilization],
        ["rolling shutter", clip.video.rollingShutter, baselineVideo.rollingShutter],
        ["audio solo", clip.audio.solo, baselineAudio.solo],
        ["audio loudness", clip.audio.loudness, baselineAudio.loudness],
        ["audio noise removal", clip.audio.noiseRemoval, baselineAudio.noiseRemoval],
    ];
    for (const [label, value, baseline] of unsupported) {
        if (JSON.stringify(value) !== JSON.stringify(baseline)) {
            throw new Error(`Clip ${clip.id} changed unsupported ${label}; the edit was not serialized.`);
        }
    }
    for (const path of Object.keys(clip.keyframes)) {
        if (!path.startsWith("transform.") && !path.startsWith("audio.")
            && !path.startsWith("video.crop.") && !path.startsWith("video.distort.")) {
            throw new Error(`Clip ${clip.id} has unsupported keyframe path ${path}.`);
        }
    }
}
function directChildOrCreate(parent, name, document) {
    const existing = firstChild(parent, name);
    if (existing) {
        return existing;
    }
    const created = document.createElement(name);
    parent.appendChild(created);
    return created;
}
function removeDirectChildren(parent, names) {
    for (const child of childElements(parent)) {
        if (names.has(tagName(child))) {
            parent.removeChild(child);
        }
    }
}
function updateTransform(element, clip) {
    const effectiveX = clip.transform.scale * clip.transform.scaleX;
    const effectiveY = clip.transform.scale * clip.transform.scaleY;
    const hasDefaultValues = Math.abs(clip.transform.x) <= EPSILON
        && Math.abs(clip.transform.y) <= EPSILON
        && Math.abs(effectiveX - 1) <= EPSILON
        && Math.abs(effectiveY - 1) <= EPSILON
        && Math.abs(clip.transform.rotation) <= EPSILON
        && Math.abs(clip.transform.anchorX) <= EPSILON
        && Math.abs(clip.transform.anchorY) <= EPSILON;
    const existing = firstChild(element, "adjust-transform");
    if (clip.transform.enabled && hasDefaultValues && existing && childElements(existing).length === 0) {
        element.removeChild(existing);
    }
    else if (!clip.transform.enabled || !hasDefaultValues || existing) {
        const transform = existing ?? element.ownerDocument.createElement("adjust-transform");
        if (clip.transform.enabled)
            transform.removeAttribute("enabled");
        else
            transform.setAttribute("enabled", "0");
        transform.setAttribute("position", `${clip.transform.x} ${clip.transform.y}`);
        transform.setAttribute("scale", `${effectiveX} ${effectiveY}`);
        transform.setAttribute("rotation", String(clip.transform.rotation));
        transform.setAttribute("anchor", `${clip.transform.anchorX} ${clip.transform.anchorY}`);
        if (!existing) {
            element.appendChild(transform);
        }
    }
    const blend = firstChild(element, "adjust-blend");
    const blendDefault = Math.abs(clip.transform.opacity - 1) <= EPSILON && clip.video.blendMode === "normal";
    if (clip.video.blendEnabled && blendDefault && blend && childElements(blend).length === 0) {
        element.removeChild(blend);
    }
    else if (!clip.video.blendEnabled || !blendDefault || blend) {
        const target = blend ?? element.ownerDocument.createElement("adjust-blend");
        if (clip.video.blendEnabled)
            target.removeAttribute("enabled");
        else
            target.setAttribute("enabled", "0");
        target.setAttribute("amount", String(clip.transform.opacity));
        target.setAttribute("mode", clip.video.blendMode);
        if (!blend) {
            element.appendChild(target);
        }
    }
}
function updateCrop(element, clip, document) {
    const existing = firstChild(element, "adjust-crop");
    if (!clip.video.crop.enabled && !existing) {
        return;
    }
    const crop = existing ?? document.createElement("adjust-crop");
    if (clip.video.crop.enabled)
        crop.removeAttribute("enabled");
    else
        crop.setAttribute("enabled", "0");
    crop.setAttribute("mode", clip.video.crop.type === "ken-burns" ? "pan" : clip.video.crop.type);
    if (clip.video.crop.type === "ken-burns") {
        removeDirectChildren(crop, new Set(["crop-rect", "trim-rect", "pan-rect"]));
        for (const window of [clip.video.crop.kenStart, clip.video.crop.kenEnd]) {
            const rect = document.createElement("pan-rect");
            rect.setAttribute("left", String(window.x - (window.width / 2)));
            rect.setAttribute("right", String(100 - window.x - (window.width / 2)));
            rect.setAttribute("top", String(window.y - (window.height / 2)));
            rect.setAttribute("bottom", String(100 - window.y - (window.height / 2)));
            crop.appendChild(rect);
        }
    }
    else {
        const rectName = clip.video.crop.type === "trim" ? "trim-rect" : "crop-rect";
        removeDirectChildren(crop, new Set(rectName === "trim-rect" ? ["crop-rect", "pan-rect"] : ["trim-rect", "pan-rect"]));
        const rect = directChildOrCreate(crop, rectName, document);
        rect.setAttribute("left", String(clip.video.crop.left));
        rect.setAttribute("right", String(clip.video.crop.right));
        rect.setAttribute("top", String(clip.video.crop.top));
        rect.setAttribute("bottom", String(clip.video.crop.bottom));
    }
    if (!existing) {
        element.appendChild(crop);
    }
}
function updateConform(element, clip, document) {
    const existing = firstChild(element, "adjust-conform");
    if (clip.video.spatialConform === "fit" && !existing) {
        return;
    }
    const conform = existing ?? document.createElement("adjust-conform");
    conform.setAttribute("type", clip.video.spatialConform);
    if (!existing) {
        element.appendChild(conform);
    }
}
function updateDistort(element, clip, document) {
    const existing = firstChild(element, "adjust-corners");
    if (!clip.video.distort.enabled) {
        if (existing) {
            element.removeChild(existing);
        }
        return;
    }
    const target = existing ?? document.createElement("adjust-corners");
    target.setAttribute("topLeft", `${clip.video.distort.topLeftX} ${clip.video.distort.topLeftY}`);
    target.setAttribute("topRight", `${clip.video.distort.topRightX} ${clip.video.distort.topRightY}`);
    target.setAttribute("bottomLeft", `${clip.video.distort.bottomLeftX} ${clip.video.distort.bottomLeftY}`);
    target.setAttribute("bottomRight", `${clip.video.distort.bottomRightX} ${clip.video.distort.bottomRightY}`);
    if (!existing) {
        element.appendChild(target);
    }
}
function updateSemanticAudioAttributes(element, clip, frameDuration) {
    for (const [attribute, value] of [["role", clip.roleName], ["audioRole", clip.audioRole]]) {
        if (value === undefined)
            continue;
        if (value === null || value.trim() === "")
            element.removeAttribute(attribute);
        else
            element.setAttribute(attribute, value);
    }
    for (const [attribute, value, positive] of [
        ["audioStart", clip.audioStart, false],
        ["audioDuration", clip.audioDuration, true],
    ]) {
        if (value === undefined)
            continue;
        if (value === null) {
            element.removeAttribute(attribute);
            continue;
        }
        if (!Number.isFinite(value.seconds) || value.seconds < 0 || (positive && value.seconds <= 0)) {
            throw new Error(`Clip ${clip.id} ${attribute} must be ${positive ? "positive" : "non-negative"}.`);
        }
        element.setAttribute(attribute, authoredTime(value, frameDuration));
    }
}
function updateAudio(element, clip, original, document, frameDuration) {
    let volume = firstChild(element, "adjust-volume");
    const gain = clip.audio.gainDb;
    if (Math.abs(gain) <= EPSILON && !volume) {
        // Preserve the absence of a neutral adjustment.
    }
    else {
        const target = volume ?? document.createElement("adjust-volume");
        target.setAttribute("amount", `${gain}dB`);
        if (original && Math.abs(gain - original.audio.gainDb) > EPSILON) {
            target.removeAttribute("enabled");
        }
        if (!volume) {
            element.appendChild(target);
        }
        volume = target;
    }
    const fullMutes = childrenNamed(element, "mute");
    if (clip.audio.muted && fullMutes.length === 0) {
        element.appendChild(document.createElement("mute"));
    }
    else if (!clip.audio.muted) {
        for (const mute of fullMutes)
            element.removeChild(mute);
    }
    const panner = firstChild(element, "adjust-panner");
    if (Math.abs(clip.audio.pan) <= EPSILON && !panner) {
        // Preserve the absence of a centered panner.
    }
    else {
        const target = panner ?? document.createElement("adjust-panner");
        target.setAttribute("amount", String(pannerValueForXml(clip.audio.pan, target, `${clip.id} pan`)));
        if (original && Math.abs(clip.audio.pan - original.audio.pan) > EPSILON) {
            target.removeAttribute("enabled");
        }
        if (!panner) {
            element.appendChild(target);
        }
    }
    if (!volume && (clip.audio.fadeIn > EPSILON || clip.audio.fadeOut > EPSILON)) {
        volume = document.createElement("adjust-volume");
        volume.setAttribute("amount", `${gain}dB`);
        element.appendChild(volume);
    }
    updateFade(volume, "fadeIn", clip.audio.fadeIn, document, frameDuration);
    updateFade(volume, "fadeOut", clip.audio.fadeOut, document, frameDuration);
}
/** Serialize Bladeworks's supported linear retime map without averaging ramps. */
function updateTimeMap(element, clip, document, frameDuration) {
    const existing = firstChild(element, "timeMap");
    if (!clip.timeMap) {
        if (existing)
            element.removeChild(existing);
        return;
    }
    if (clip.timeMap.points.length < 2) {
        throw new Error(`Clip ${clip.id} retime requires at least two time points.`);
    }
    const target = existing ?? document.createElement("timeMap");
    if (clip.timeMap.frameSampling)
        target.setAttribute("frameSampling", clip.timeMap.frameSampling);
    else
        target.removeAttribute("frameSampling");
    if (clip.timeMap.preservesPitch !== null) {
        target.setAttribute("preservesPitch", clip.timeMap.preservesPitch ? "1" : "0");
    }
    else {
        target.removeAttribute("preservesPitch");
    }
    removeDirectChildren(target, new Set(["timept"]));
    let previous = -Infinity;
    for (const point of clip.timeMap.points) {
        if (point.interpolation.toLowerCase() !== "linear") {
            throw new Error(`Clip ${clip.id} uses unsupported non-linear retime interpolation ${point.interpolation}.`);
        }
        if (point.time.seconds < previous - EPSILON) {
            throw new Error(`Clip ${clip.id} retime points are not ordered by output time.`);
        }
        previous = point.time.seconds;
        const child = document.createElement("timept");
        child.setAttribute("time", authoredTime(point.time, frameDuration));
        child.setAttribute("value", authoredTime(point.value, frameDuration));
        child.setAttribute("interp", "linear");
        target.appendChild(child);
    }
    if (!existing)
        element.appendChild(target);
}
function updateFade(volume, name, duration, document, frameDuration) {
    const existing = findAudioFade(volume, name);
    if (duration <= EPSILON) {
        if (existing) {
            existing.parentNode?.removeChild(existing);
        }
        return;
    }
    const fade = existing ?? document.createElement(name);
    fade.setAttribute("type", fade.getAttribute("type") || "linear");
    fade.setAttribute("duration", quantizedTime(duration, frameDuration));
    if (!existing) {
        if (!volume)
            throw new Error("Audio fade requires an adjust-volume container.");
        volume.appendChild(fade);
    }
}
function updateMarkers(element, clip, origin, document, frameDuration) {
    const originals = origin
        ? childElements(origin.element).filter((child) => MARKER_TAGS.has(tagName(child)))
        : [];
    removeDirectChildren(element, MARKER_TAGS);
    for (const marker of clip.markers) {
        const tag = marker.type === "chapter" ? "chapter-marker" : marker.type === "todo" ? "todo-marker" : "marker";
        const originalIndex = origin?.model.markers.findIndex((candidate) => candidate.id === marker.id) ?? -1;
        const original = originalIndex >= 0 ? originals[originalIndex] : undefined;
        const child = original && tagName(original) === tag
            ? document.importNode(original, true)
            : document.createElement(tag);
        if (original && tagName(original) !== tag) {
            for (let index = 0; index < original.attributes.length; index += 1) {
                const attribute = original.attributes.item(index);
                if (attribute)
                    child.setAttribute(attribute.name, attribute.value);
            }
            for (let index = 0; index < original.childNodes.length; index += 1) {
                const nested = original.childNodes.item(index);
                if (nested)
                    child.appendChild(document.importNode(nested, true));
            }
        }
        const sourceTime = clip.sourceStart + marker.offset;
        const originalStart = original?.getAttribute("start");
        child.setAttribute("start", originalStart
            ? preservedOrQuantized(sourceTime, seconds(originalStart, origin?.model.sourceStart ?? 0, `${clip.id} original marker`), originalStart, frameDuration)
            : quantizedTime(sourceTime, frameDuration));
        if (!child.hasAttribute("duration"))
            child.setAttribute("duration", rationalString(frameDuration));
        child.setAttribute("value", marker.name);
        if (marker.type === "todo") {
            child.setAttribute("completed", marker.completed ? "1" : "0");
        }
        else {
            child.removeAttribute("completed");
        }
        element.appendChild(child);
    }
}
function updateEffectStack(element, clip, origin, document, effects, frameDuration) {
    const originalsById = new Map();
    if (origin) {
        let stackIndex = 0;
        for (const child of childElements(origin.element)) {
            if (tagName(child) !== "filter-video" && tagName(child) !== "filter-video-mask")
                continue;
            const item = origin.model.effectStack[stackIndex];
            if (item) {
                originalsById.set(item.kind === "effect" ? item.effect.id : item.maskedEffect.id, child);
            }
            stackIndex += 1;
        }
    }
    const plainChanged = origin
        ? JSON.stringify(clip.effects) !== JSON.stringify(origin.model.effects)
        : clip.effectStack.length === 0 && clip.effects.length > 0;
    const stack = plainChanged ? reconcilePlainEffects(clip.effectStack, clip.effects) : clip.effectStack;
    for (const child of childElements(element)) {
        if (tagName(child) === "filter-video" || tagName(child) === "filter-video-mask")
            element.removeChild(child);
    }
    for (const item of stack) {
        if (item.kind === "effect") {
            element.appendChild(serializeClipEffect(item.effect, originalsById.get(item.effect.id), document, effects, frameDuration));
        }
        else {
            element.appendChild(serializeMaskedEffect(item.maskedEffect, originalsById.get(item.maskedEffect.id), document, effects, frameDuration));
        }
    }
}
function reconcilePlainEffects(stack, effects) {
    let index = 0;
    const output = [];
    for (const item of stack) {
        if (item.kind === "masked-effect") {
            output.push(item);
            continue;
        }
        const effect = effects[index];
        if (effect)
            output.push({ kind: "effect", effect });
        index += 1;
    }
    for (; index < effects.length; index += 1)
        output.push({ kind: "effect", effect: effects[index] });
    return output;
}
function serializeClipEffect(effect, source, document, effects, frameDuration) {
    const filter = source && tagName(source) === "filter-video"
        ? document.importNode(source, true) : document.createElement("filter-video");
    const ref = ensureEffectResource(document, effects, effect.name, effect.resourceId, effect.resourceUid);
    filter.setAttribute("ref", ref);
    filter.setAttribute("name", effect.name);
    filter.setAttribute("enabled", effect.enabled ? "1" : "0");
    updateParameterElements(filter, effect.parameters, effect.parameterNames, effect.parameterKeyframes, document, frameDuration);
    return filter;
}
function serializeMaskedEffect(masked, source, document, effects, frameDuration) {
    if (masked.masks.length === 0 || masked.masks.length > 32) {
        throw new Error(`Masked effect ${masked.id} requires 1..32 masks.`);
    }
    if (masked.filters.length < 1 || masked.filters.length > 2) {
        throw new Error(`Masked effect ${masked.id} requires one inside filter and at most one outside filter.`);
    }
    const group = source && tagName(source) === "filter-video-mask"
        ? document.importNode(source, true) : document.createElement("filter-video-mask");
    const originalMasks = childElements(group).filter((child) => tagName(child) === "mask-shape" || tagName(child) === "mask-isolation");
    const originalFilters = childrenNamed(group, "filter-video");
    group.setAttribute("enabled", masked.enabled ? "1" : "0");
    group.setAttribute("inverted", masked.inverted ? "1" : "0");
    while (group.firstChild)
        group.removeChild(group.firstChild);
    for (const mask of masked.masks) {
        const originalIndex = /\/mask\[(\d+)]$/.exec(mask.id)?.[1];
        const original = originalIndex ? originalMasks[Number(originalIndex) - 1] : undefined;
        group.appendChild(serializeMaskSource(mask, original, document, frameDuration));
    }
    for (const [index, filter] of masked.filters.entries()) {
        group.appendChild(serializeClipEffect(filter, originalFilters[index], document, effects, frameDuration));
    }
    return group;
}
function serializeMaskSource(mask, source, document, frameDuration) {
    if (!new Set(["add", "subtract", "multiply"]).has(mask.blendMode)) {
        throw new Error(`Mask ${mask.id} uses unsupported blend mode ${mask.blendMode}.`);
    }
    const expectedTag = mask.kind === "color" || mask.kind === "luma" ? "mask-isolation" : "mask-shape";
    const element = source && tagName(source) === expectedTag
        ? document.importNode(source, true) : document.createElement(expectedTag);
    element.setAttribute("name", mask.name);
    element.setAttribute("enabled", mask.enabled ? "1" : "0");
    element.setAttribute("blendMode", mask.blendMode);
    updateParameterElements(element, mask.parameters, mask.parameterNames, mask.parameterKeyframes, document, frameDuration);
    if (mask.kind === "draw") {
        const key = Object.keys(mask.parameters).find((candidate) => new Set(["points", "vertices", "path", "300"]).has(candidate.toLowerCase()));
        if (!key)
            throw new Error(`Draw Mask ${mask.id} requires an explicit polygon points parameter.`);
    }
    if (mask.kind === "color" || mask.kind === "luma") {
        if (!mask.data)
            throw new Error(`${mask.name} requires explicit spell-mask-isolation-v1 data.`);
        let decoded;
        try {
            decoded = JSON.parse(mask.data);
        }
        catch {
            throw new Error(`${mask.name} has invalid JSON isolation data.`);
        }
        if (decoded.abi !== "spell-mask-isolation-v1") {
            throw new Error(`${mask.name} requires abi='spell-mask-isolation-v1'.`);
        }
        removeDirectChildren(element, new Set(["data"]));
        const data = document.createElement("data");
        data.setAttribute("key", "bladeworks-mask");
        data.textContent = mask.data;
        element.appendChild(data);
    }
    return element;
}
function ensureEffectResource(document, effects, name, requestedId, uid) {
    if (requestedId && effects.has(requestedId))
        return requestedId;
    const existing = [...effects.values()].find((effect) => uid ? effect.uid === uid : effect.name === name);
    if (existing)
        return existing.id;
    if (!uid) {
        throw new Error(`Effect ${name} has no existing resource and no renderer-owned FCPXML UID.`);
    }
    const resources = firstChild(document.documentElement, "resources");
    if (!resources)
        throw new Error("FCPXML has no <resources> element for an inserted effect.");
    let index = 1;
    let id = `bf_effect_${index}`;
    const idExists = (candidate) => childElements(resources).some((element) => element.getAttribute("id") === candidate);
    while (idExists(id)) {
        index += 1;
        id = `bf_effect_${index}`;
    }
    const resource = document.createElement("effect");
    resource.setAttribute("id", id);
    resource.setAttribute("name", name);
    resource.setAttribute("uid", uid);
    resources.appendChild(resource);
    effects.set(id, { id, name, uid });
    return id;
}
function updateParameterElements(container, values, names, keyframes, document, frameDuration) {
    const existing = new Map(childrenNamed(container, "param").map((parameter) => [parameterIdentity(parameter), parameter]));
    for (const parameter of childrenNamed(container, "param"))
        container.removeChild(parameter);
    for (const [key, value] of Object.entries(values)) {
        const parameter = existing.get(key) ?? document.createElement("param");
        parameter.setAttribute("name", names[key] || parameter.getAttribute("name") || key);
        if (key !== names[key] || parameter.hasAttribute("key"))
            parameter.setAttribute("key", key);
        parameter.setAttribute("value", parameterValueString(value));
        const oldAnimation = firstChild(parameter, "keyframeAnimation");
        if (oldAnimation)
            parameter.removeChild(oldAnimation);
        const frames = keyframes[key] ?? [];
        if (frames.length > 0) {
            const animation = document.createElement("keyframeAnimation");
            for (const frame of frames) {
                const child = document.createElement("keyframe");
                child.setAttribute("time", frameDuration ? authoredTime(frame.time, frameDuration) : frame.time.raw);
                child.setAttribute("value", parameterValueString(frame.value));
                child.setAttribute("interp", frame.interpolation);
                animation.appendChild(child);
            }
            parameter.appendChild(animation);
        }
        container.appendChild(parameter);
    }
    for (const key of Object.keys(keyframes)) {
        if (!Object.hasOwn(values, key)) {
            throw new Error(`Animated parameter ${key} has no static parameter value.`);
        }
    }
}
function updateKeyframes(element, clip, document, frameDuration) {
    for (const [path, frames] of Object.entries(clip.keyframes)) {
        const parts = path.split(".");
        const field = parts.at(-1);
        if (!field) {
            continue;
        }
        const adjustmentName = path === "transform.opacity"
            ? "adjust-blend"
            : path.startsWith("transform.")
                ? "adjust-transform"
                : path.startsWith("audio.")
                    ? (field === "pan" ? "adjust-panner" : "adjust-volume")
                    : path.startsWith("video.crop.")
                        ? "adjust-crop"
                        : path.startsWith("video.distort.")
                            ? "adjust-corners"
                            : null;
        if (!adjustmentName) {
            continue;
        }
        const adjustment = directChildOrCreate(element, adjustmentName, document);
        let parameter = childrenNamed(adjustment, "param").find((candidate) => (candidate.getAttribute("name") || "").toLowerCase()
            === keyframeParameterName(path, field).toLowerCase());
        if (!parameter) {
            parameter = document.createElement("param");
            parameter.setAttribute("name", keyframeParameterName(path, field));
            adjustment.appendChild(parameter);
        }
        const existingAnimation = firstChild(parameter, "keyframeAnimation");
        if (existingAnimation) {
            parameter.removeChild(existingAnimation);
        }
        if (frames.length === 0) {
            continue;
        }
        const animation = document.createElement("keyframeAnimation");
        for (const frame of frames) {
            const keyframe = document.createElement("keyframe");
            keyframe.setAttribute("time", authoredTime(frame.time, frameDuration));
            const value = path === "audio.pan"
                ? pannerValueForXml(frame.value, adjustment, `${clip.id} pan keyframe`)
                : frame.value;
            keyframe.setAttribute("value", parameterValueString(value));
            keyframe.setAttribute("interp", frame.interpolation);
            animation.appendChild(keyframe);
        }
        parameter.appendChild(animation);
    }
}
function keyframeParameterName(path, fallback) {
    if (path === "transform.opacity" || path === "audio.gainDb" || path === "audio.pan") {
        return "amount";
    }
    const names = {
        "video.distort.topleft": "topLeft",
        "video.distort.topright": "topRight",
        "video.distort.bottomleft": "bottomLeft",
        "video.distort.bottomright": "bottomRight",
    };
    return names[path] ?? fallback;
}
function updateTitleText(element, clip, document) {
    const value = clip.text ?? clip.name;
    const text = firstChild(element, "text") ?? document.createElement("text");
    if (!text.parentNode) {
        element.appendChild(text);
    }
    if (clip.kind === "caption" && clip.caption) {
        text.setAttribute("display-style", clip.caption.displayStyle);
        text.setAttribute("placement", clip.caption.placement);
        text.setAttribute("alignment", clip.caption.alignment);
        element.setAttribute("role", clip.caption.role);
    }
    const existingStyle = firstChild(text, "text-style");
    if (text.textContent === value && existingStyle) {
        updateTextStyleDefinition(element, clip, document);
        return;
    }
    while (text.firstChild)
        text.removeChild(text.firstChild);
    const style = existingStyle ?? document.createElement("text-style");
    const styleId = existingStyle?.getAttribute("ref")
        || firstChild(element, "text-style-def")?.getAttribute("id")
        || "bf_text_style";
    style.setAttribute("ref", styleId);
    style.textContent = value;
    text.appendChild(style);
    updateTextStyleDefinition(element, clip, document);
}
function updateTextStyleDefinition(element, clip, document) {
    if (!clip.textStyle)
        return;
    const runReference = firstChild(firstChild(element, "text") ?? element, "text-style")?.getAttribute("ref");
    const definition = childrenNamed(element, "text-style-def").find((candidate) => candidate.getAttribute("id") === runReference) ?? firstChild(element, "text-style-def") ?? document.createElement("text-style-def");
    if (!definition.parentNode)
        element.appendChild(definition);
    const id = definition.getAttribute("id") || "bf_text_style";
    definition.setAttribute("id", id);
    definition.setAttribute("name", definition.getAttribute("name") || (clip.kind === "caption" ? "Caption" : "Main"));
    const style = firstChild(definition, "text-style") ?? document.createElement("text-style");
    if (!style.parentNode)
        definition.appendChild(style);
    style.setAttribute("font", clip.textStyle.font);
    style.setAttribute("fontFace", clip.textStyle.fontFace);
    style.setAttribute("fontSize", String(clip.textStyle.fontSize));
    style.setAttribute("fontColor", parameterValueString(clip.textStyle.fontColor));
    style.setAttribute("alignment", clip.textStyle.alignment);
    const text = firstChild(element, "text");
    const run = text ? firstChild(text, "text-style") : null;
    if (run)
        run.setAttribute("ref", id);
}
function updateGeneratorColor(element, clip, document) {
    if (!clip.generatorColor) {
        throw new Error(`Custom Solid clip ${clip.id} requires an RGBA color.`);
    }
    let parameter = childrenNamed(element, "param").find((candidate) => parameterIdentity(candidate) === CUSTOM_SOLID_COLOR_KEY);
    if (!parameter) {
        parameter = document.createElement("param");
        element.appendChild(parameter);
    }
    parameter.setAttribute("name", "Color");
    parameter.setAttribute("key", CUSTOM_SOLID_COLOR_KEY);
    parameter.setAttribute("value", parameterValueString(clip.generatorColor));
}
function createOrUpdateTransition(document, transition, origin, leftClip, frameDuration, resources, timelineOffsetBase = 0) {
    const element = origin ? document.importNode(origin.element, true) : document.createElement("transition");
    const boundResourceId = ensureEffectResource(document, resources.effects, transition.name, transition.resourceId, transition.resourceUid);
    element.setAttribute("name", transition.name);
    element.setAttribute("duration", origin
        ? preservedOrQuantized(transition.duration, origin.model.duration, origin.duration, frameDuration)
        : quantizedTime(transition.duration, frameDuration));
    const center = timelineOffsetBase + leftClip.timelineStart + leftClip.duration - (transition.duration / 2);
    element.setAttribute("offset", origin
        ? preservedOrQuantized(center, origin.offsetSeconds, origin.offset, frameDuration)
        : quantizedTime(center, frameDuration));
    if (!origin) {
        const effect = resources.effects.get(boundResourceId);
        const filter = document.createElement("filter-video");
        filter.setAttribute("ref", effect.id);
        filter.setAttribute("name", effect.name);
        element.appendChild(filter);
    }
    const filter = firstChild(element, "filter-video");
    if (!filter) {
        throw new Error(`Transition ${transition.name} does not contain a filter-video resource binding.`);
    }
    filter.setAttribute("ref", boundResourceId);
    updateParameterElements(filter, transition.parameters, transition.parameterNames, transition.parameterKeyframes, document, frameDuration);
    return element;
}
