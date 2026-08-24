/**
 * Unified mock-capability registry for the browser editor UI prototype.
 *
 * Architecture map:
 * every intentionally simulated UI surface
 *   -> one stable capability ID in MOCK_CAPABILITIES
 *   -> visible inventory in the Fixture badge
 *   -> explicit “(is still a mock)” notice when the user invokes it
 *
 * Product invariant:
 * No simulated behavior may be introduced ad hoc in app.ts. Add it here first,
 * then reference the capability ID from the action or parameter path. This
 * keeps the prototype honest and gives the future localhost renderer team one
 * concrete checklist of integrations to replace.
 */
export const MOCK_CAPABILITIES = Object.freeze({
    "fixture-projects": {
        label: "Library, Event, and Project contents",
        detail: "The visible hierarchy, clip names, proposal Projects, and media rows come from deterministic fixture data until localhost supplies canonical FCPXML snapshots.",
        availability: "fixture-only",
        category: "Library",
    },
    "media-import": {
        label: "Media import",
        detail: "Fixture mode creates a synthetic media row instead of opening the native file picker and indexing a real asset.",
        availability: "fixture-only",
        category: "Library",
    },
    "event-creation": {
        label: "Event creation",
        detail: "Fixture mode appends an in-memory Event; no Final Cut Library package or persistent store is changed.",
        availability: "fixture-only",
        category: "Library",
    },
    "realtime-preview": {
        label: "Realtime program monitor",
        detail: "The concert frame is a CSS scene. Bladeworks's WebRTC/Bladeworks preview replaces it in localhost mode.",
        availability: "fixture-only",
        category: "Playback",
    },
    "transition-editor": {
        label: "Transition editor",
        detail: "Transition blocks are visual fixtures. Selection and hover feedback work, but duration, handles, and FCPXML mutation are not wired yet.",
        availability: "fixture-only",
        category: "Timeline",
    },
    "effects-browser": {
        label: "Effects browser",
        detail: "Browsing, applying, reordering, and disabling clip effects mutates the real effect stack, but the rendered pixel result is simulated until the localhost renderer supplies FxPlug/Motion effects.",
        availability: "fixture-only",
        category: "Inspector",
    },
    "transitions-browser": {
        label: "Transitions browser",
        detail: "Dropping a transition on an edit point creates a real transition record on the storyline, but the animated blend is simulated until the localhost renderer is connected.",
        availability: "fixture-only",
        category: "Timeline",
    },
    "color-wheels": {
        label: "Shadows, Midtones, and Highlights wheels",
        detail: "Wheel selection and pointer movement are UI-only. The portable renderer parameter contract has not been connected to these zone controls.",
        availability: "fixture-only",
        category: "Inspector",
    },
    "audio-enhancements": {
        label: "Audio Enhancements",
        detail: "Loudness and noise-removal controls currently preview local UI state only.",
        availability: "fixture-only",
        category: "Inspector",
    },
    "portable-video-analysis": {
        label: "Color conform, stabilization, and rolling shutter",
        detail: "Bladeworks preserves these FCPXML settings but Studio does not author them in the current portable edit profile.",
        availability: "fixture-only",
        category: "Inspector",
    },
    "ken-burns-editor": {
        label: "Ken Burns authoring",
        detail: "Bladeworks preserves existing Ken Burns timing, but Studio does not author or replace time-map windows in the current edit profile.",
        availability: "fixture-only",
        category: "Inspector",
    },
    trackers: {
        label: "Trackers",
        detail: "Tracker creation and analysis require the native/FCP runtime and are not implemented in the browser prototype.",
        availability: "fixture-only",
        category: "Inspector",
    },
    "generators-browser": {
        label: "Titles and Generators catalog",
        detail: "The source tab and generated items are fixture content; Motion template enumeration is not connected.",
        availability: "fixture-only",
        category: "Library",
    },
    "effects-presets": {
        label: "Save Effects Preset",
        detail: "The button demonstrates placement only; no reusable preset package is persisted.",
        availability: "fixture-only",
        category: "Inspector",
    },
    "final-export": {
        label: "Final export",
        detail: "Fixture mode emits the intended request contract but cannot render media without fcpxml2ffmpeg.",
        availability: "fixture-only",
        category: "Export",
    },
});
const ACTION_CAPABILITIES = Object.freeze({
    "import-media": "media-import",
    "new-event": "event-creation",
    transition: "transition-editor",
    "apply-effect": "effects-browser",
    "apply-transition": "transitions-browser",
    "select-color-zone": "color-wheels",
    "add-tracker": "trackers",
    "show-generators": "generators-browser",
    "toggle-play": "realtime-preview",
    "mock-preview": "realtime-preview",
    "select-project": "fixture-projects",
    "select-event": "fixture-projects",
    "library-source": "fixture-projects",
    "save-effects-preset": "effects-presets",
    "toggle-item-solo": "audio-enhancements",
    "swap-ken-burns": "ken-burns-editor",
    export: "final-export",
});
const PARAMETER_CAPABILITIES = Object.freeze({
    "video.color": "color-wheels",
    "video.colorConform": "portable-video-analysis",
    "video.colorConformType": "portable-video-analysis",
    "video.stabilization": "portable-video-analysis",
    "video.rollingShutter": "portable-video-analysis",
    "audio.loudness": "audio-enhancements",
    "audio.noiseRemoval": "audio-enhancements",
    "audio.solo": "audio-enhancements",
});
function applies(capability, connectionMode) {
    return capability.availability === "always" || connectionMode !== "localhost";
}
export function mockCapabilityForAction(action, connectionMode) {
    if (!action) {
        return null;
    }
    const id = ACTION_CAPABILITIES[action];
    if (!id) {
        return null;
    }
    const capability = MOCK_CAPABILITIES[id];
    return capability && applies(capability, connectionMode) ? id : null;
}
export function mockCapabilityForParameter(path, connectionMode) {
    if (!path) {
        return null;
    }
    const direct = PARAMETER_CAPABILITIES[path];
    const id = direct ?? Object.entries(PARAMETER_CAPABILITIES)
        .find(([prefix]) => path.startsWith(`${prefix}.`))?.[1];
    if (!id) {
        return null;
    }
    const capability = MOCK_CAPABILITIES[id];
    return capability && applies(capability, connectionMode) ? id : null;
}
export function activeMockCapabilities(connectionMode) {
    return Object.entries(MOCK_CAPABILITIES)
        .filter(([, capability]) => applies(capability, connectionMode))
        .map(([id, capability]) => ({ id, capability }));
}
export function mockNoticeTitle(id) {
    const capability = MOCK_CAPABILITIES[id];
    return capability ? `${capability.label} (is still a mock)` : "This control (is still a mock)";
}
/**
 * Capability-shaped catalog for standalone fixture mode.
 *
 * Localhost never reads this value. It always asks Bladeworks for the live
 * tensor registry. Keeping fixture mode behind the same schema lets the
 * catalog and Inspector exercise the production UI without claiming that a
 * fixture effect was discovered from a running renderer.
 */
export const MOCK_EDITOR_CAPABILITIES = {
    schemaVersion: 1,
    renderer: "tensor",
    mechanics: [
        { id: "preview", name: "Preview", support: "exact", authorable: false, qualities: [720, 540, 480] },
        { id: "crop", name: "Crop", support: "exact", authorable: true, modes: ["trim", "crop", "kenBurns"] },
        {
            id: "masks",
            name: "Numeric shape, draw, color and luma masks",
            support: "approximate",
            authorable: true,
            animatable: true,
            maximumMasks: 32,
            blendModes: ["add", "subtract", "multiply"],
            invert: true,
            sourceKinds: [
                {
                    id: "shape", fcpxmlKind: "mask-shape", support: "approximate",
                    parameters: [
                        { key: "160", name: "Radius", type: "point", min: 0, max: 32768, components: ["x", "y"], units: "image_plane_pixels", animatable: true },
                        { key: "159", name: "Curvature", type: "number", default: 1, min: 0, max: 1, components: [], units: "normalized", animatable: true },
                        { key: "102", name: "Feather", type: "number", default: 0, min: 0, max: 8192, components: [], units: "image_plane_pixels", animatable: true },
                        { key: "201", name: "Position", type: "point", min: -32768, max: 32768, components: ["x", "y"], units: "image_plane_pixels", animatable: true },
                        { key: "202", name: "Rotation", type: "number", default: 0, min: -3600, max: 3600, components: [], units: "degrees", animatable: true },
                        { key: "103", name: "Opacity", type: "number", default: 1, min: 0, max: 1, components: [], units: "normalized", animatable: true },
                        { key: "104", name: "Falloff", type: "number", default: 1, min: 0.1, max: 8, components: [], units: "exponent", animatable: true },
                    ],
                    notes: "Curvature and feather use a portable superellipse and linear-ramp approximation.",
                },
                {
                    id: "draw", fcpxmlKind: "mask-shape", support: "approximate",
                    parameters: [
                        { key: "points", name: "Points", type: "point_list", min: -32768, max: 32768, components: ["x", "y"], units: "image_plane_pixels", minimumItems: 3, maximumItems: 64, convex: true, animatable: false },
                        { key: "opacity", name: "Opacity", type: "number", default: 1, min: 0, max: 1, components: [], units: "normalized", animatable: true },
                    ],
                    notes: "Polygon edges are linear; Final Cut Bezier handles are not reproduced.",
                },
                {
                    id: "color", fcpxmlKind: "mask-isolation", dataAbi: "spell-mask-isolation-v1", support: "approximate",
                    parameters: [
                        { key: "color", name: "Color", type: "color", min: 0, max: 1, components: ["red", "green", "blue"], units: "normalized", animatable: false },
                        { key: "tolerance", name: "Tolerance", type: "number", default: 0.12, min: 0.00001, max: 1, components: [], units: "normalized", animatable: false },
                        { key: "softness", name: "Softness", type: "number", default: 0.05, min: 0, max: 1, components: [], units: "normalized", animatable: false },
                        { key: "opacity", name: "Opacity", type: "number", default: 1, min: 0, max: 1, components: [], units: "normalized", animatable: false },
                    ],
                    notes: "Color isolation uses a portable RGB-distance approximation.",
                },
                {
                    id: "luma", fcpxmlKind: "mask-isolation", dataAbi: "spell-mask-isolation-v1", support: "approximate",
                    parameters: [
                        { key: "luma_min", name: "Luma Minimum", type: "number", default: 0, min: 0, max: 1, components: [], units: "normalized", animatable: false },
                        { key: "luma_max", name: "Luma Maximum", type: "number", default: 1, min: 0, max: 1, components: [], units: "normalized", animatable: false },
                        { key: "softness", name: "Softness", type: "number", default: 0.05, min: 0, max: 1, components: [], units: "normalized", animatable: false },
                        { key: "opacity", name: "Opacity", type: "number", default: 1, min: 0, max: 1, components: [], units: "normalized", animatable: false },
                    ],
                    notes: "Luma isolation uses a portable bounded luma ramp.",
                },
            ],
        },
        { id: "titles", name: "Titles", support: "approximate", authorable: true },
        { id: "customSolid", name: "Custom Solid", support: "exact", authorable: true },
    ],
    blendModes: [
        "normal", "behind", "add", "subtract", "darken", "lighten", "multiply",
        "screen", "overlay", "soft-light", "hard-light", "difference", "exclusion",
        "color-burn", "color-dodge", "divide", "linear-light", "pin-light", "hard-mix",
        "stencil-alpha", "silhouette-alpha", "stencil-luma", "silhouette-luma",
    ].map((fcpxmlValue) => ({
        id: fcpxmlValue.replaceAll("-", ""),
        name: fcpxmlValue.replace(/(^|-)([a-z])/g, (_match, separator, letter) => `${separator ? " " : ""}${letter.toUpperCase()}`),
        fcpxmlValue,
        support: fcpxmlValue === "normal" || fcpxmlValue === "behind" ? "exact" : "approximate",
        authorable: true,
    })),
    retime: {
        support: "exact",
        authorable: true,
        modes: ["constant", "reverse", "freeze", "piecewiseLinear"],
        frameSampling: ["floor"],
        preservePitch: true,
        notes: "Fixture mode mirrors the portable Bladeworks retime surface.",
    },
    effects: [
        {
            id: "color-adjustments",
            name: "Color Adjustments",
            handler: "color_adjustments",
            resource: { uid: "FxPlug:7E2022A5-202B-4EEB-A311-AC2B585D01B0", xfadeId: null },
            authorable: true,
            support: "exact",
            parameters: [
                { key: "3", name: "Exposure", type: "number", default: 0, min: -100, max: 100, animatable: false },
                { key: "17", name: "Contrast", type: "number", default: 0, min: -100, max: 100, animatable: false },
                { key: "16", name: "Saturation", type: "number", default: 0, min: -100, max: 100, animatable: false },
            ],
            notes: [],
        },
        {
            id: "gaussian-blur",
            name: "Gaussian Blur",
            handler: "gaussian",
            resource: { uid: null, xfadeId: null },
            authorable: true,
            support: "exact",
            parameters: [
                { key: "9999/986883376/2/100", name: "Amount", type: "number", default: 25, min: 0, max: 100, animatable: false },
                { key: "9999/986884620/2/100", name: "Boost", type: "number", default: 0, min: 0, max: 100, animatable: false },
            ],
            notes: [],
        },
        {
            id: "sharpen",
            name: "Sharpen",
            handler: "sharpen",
            resource: { uid: null, xfadeId: null },
            authorable: true,
            support: "exact",
            parameters: [
                { key: "9999/986883554/2/100", name: "Amount", type: "number", default: 25, min: 0, max: 100, animatable: false },
            ],
            notes: [],
        },
        {
            id: "vignette",
            name: "Vignette",
            handler: "vignette",
            resource: { uid: null, xfadeId: null },
            authorable: true,
            support: "exact",
            parameters: [
                { key: "9999/200/202", name: "Strength", type: "number", default: 0.65, min: 0, max: 1, animatable: false },
                { key: "9999/987213589/1", name: "Size", type: "number", default: 1.5, min: 0.01, max: 1.57, animatable: false },
            ],
            notes: [],
        },
        {
            id: "green-screen-keyer",
            name: "Green Screen Keyer",
            handler: "green_screen_keyer",
            resource: { uid: null, xfadeId: null },
            authorable: true,
            support: "approximate",
            parameters: [],
            notes: ["Portable RGB key and despill, not Apple's private Keyer math."],
        },
        {
            id: "pixellate",
            name: "Pixellate",
            handler: "pixellate",
            resource: { uid: null, xfadeId: null },
            authorable: true,
            support: "default_only",
            parameters: [],
            notes: ["Uses the calibrated default; authored controls are not admitted."],
        },
    ],
    transitions: [
        {
            id: "cross-dissolve",
            name: "Cross Dissolve",
            handler: "cross_dissolve",
            resource: { uid: "FxPlug:4731E73A-8DAC-4113-9A30-AE85B1761265", xfadeId: null },
            authorable: true,
            support: "approximate",
            parameters: [],
            notes: ["Composes both transition sides before the dissolve."],
        },
        {
            id: "fade-to-color",
            name: "Fade to Color",
            handler: "fade_color",
            resource: { uid: "FxPlug:F779C565-486D-4633-8035-0374B4DB8F5C", xfadeId: null },
            authorable: true,
            support: "partial",
            parameters: [
                { key: "3", name: "Color", type: "color", default: { red: 0, green: 0, blue: 0, alpha: 1 }, animatable: false },
            ],
            notes: [],
        },
        {
            id: "wipe-left",
            name: "Wipe Left",
            handler: "wipe",
            resource: { uid: null, xfadeId: null },
            authorable: true,
            support: "exact",
            parameters: [],
            notes: [],
        },
        {
            id: "bloom",
            name: "Bloom",
            handler: "xfade",
            resource: { uid: null, xfadeId: "bloom" },
            authorable: true,
            support: "approximate",
            parameters: [],
            notes: ["Calibrated tensor xfade expression."],
        },
    ],
    audio: {
        support: "exact",
        authorable: true,
        controls: ["gain", "pan", "fadeIn", "fadeOut", "mute"],
        outputLayouts: ["mono", "stereo"],
    },
    media: {
        support: "exact",
        decodedPixelFormats: ["rgb24", "rgba"],
        colorMatrices: ["bt709"],
        hdrInputTransfers: [],
        missingMedia: "Red checker placeholder for video and exact-duration silence for audio.",
    },
    export: {
        support: "exact",
        supportedResolutions: [1080, 720, 540, 480],
        defaultResolution: 1080,
        profiles: [
            { id: "delivery", name: "Delivery", resolution: "1080p" },
            { id: "delivery_alpha", name: "Delivery with Alpha", resolution: "1080p" },
        ],
    },
    unsupported: [
        { id: "stabilization", category: "video", reason: "Not implemented by the Tensor renderer." },
        { id: "rollingShutter", category: "video", reason: "Not implemented by the Tensor renderer." },
        { id: "tracking", category: "analysis", reason: "Tracking analysis is outside the portable edit profile." },
        { id: "opticalFlow", category: "retime", reason: "Only Fast (Floor) frame sampling is supported." },
    ],
};
