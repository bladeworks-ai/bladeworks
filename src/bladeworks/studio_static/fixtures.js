/**
 * Synthetic Library, media, and Project fixtures for standalone UI work.
 *
 * Architecture map:
 * curated MediaAsset catalog
 *   -> reusable TimelineClip factories
 *   -> normalized ProjectSnapshots
 *   -> Library/Event/Project summaries
 *
 * Main callers:
 * - MockEditorRuntime.bootstrap
 * - browser interaction demos
 *
 * Why this exists:
 * The browser editor must remain runnable before the localhost FCPXML/media
 * runtime is available. Fixtures exercise the real reducer and UI without
 * pretending synthetic state is canonical production data.
 */
import { defaultAudio, defaultKeyframes, defaultTransform, defaultVideo } from "./clip-state.js";
import { normalizeProject, projectDuration } from "./magnetic-timeline.js";
export { defaultTransform } from "./clip-state.js";
export const fixtureAssets = [
    {
        id: "asset-host-wide",
        name: "Host · Wide Take",
        kind: "video",
        duration: 12.8,
        colors: { a: "#334565", b: "#97665a" },
        tags: ["host", "interview", "wide"],
        glyph: "◉",
        createdAt: "Today",
        favorite: true,
    },
    {
        id: "asset-host-close",
        name: "Host · Close Up",
        kind: "video",
        duration: 9.4,
        colors: { a: "#5b354e", b: "#c48a63" },
        tags: ["host", "interview", "close"],
        glyph: "◉",
        createdAt: "Today",
        favorite: false,
    },
    {
        id: "asset-product-spin",
        name: "Product Spin",
        kind: "video",
        duration: 6.2,
        colors: { a: "#253752", b: "#53a7bb" },
        tags: ["product", "b-roll", "studio"],
        glyph: "◉",
        createdAt: "Yesterday",
        favorite: true,
    },
    {
        id: "asset-desk-detail",
        name: "Desk Detail",
        kind: "video",
        duration: 5.7,
        colors: { a: "#4c3848", b: "#b07a66" },
        tags: ["desk", "b-roll", "detail"],
        glyph: "◉",
        createdAt: "Yesterday",
        favorite: false,
    },
    {
        id: "asset-city-night",
        name: "City at Night",
        kind: "video",
        duration: 8.1,
        colors: { a: "#162741", b: "#5b3d84" },
        tags: ["city", "night", "establishing"],
        glyph: "◉",
        createdAt: "This Week",
        favorite: false,
    },
    {
        id: "asset-ui-capture",
        name: "App Walkthrough",
        kind: "video",
        duration: 13.3,
        colors: { a: "#1a4864", b: "#8057c6" },
        tags: ["screen", "product", "ui"],
        glyph: "◉",
        createdAt: "This Week",
        favorite: false,
    },
    {
        id: "asset-logo",
        name: "Bladeworks Mark",
        kind: "image",
        duration: 4,
        colors: { a: "#6d4cf4", b: "#df9fff" },
        tags: ["logo", "brand", "graphic"],
        glyph: "◆",
        createdAt: "Today",
        favorite: true,
    },
    {
        id: "asset-music-pulse",
        name: "Neon Pulse",
        kind: "audio",
        duration: 28,
        colors: { a: "#214d45", b: "#347d6d" },
        tags: ["music", "electronic", "upbeat"],
        glyph: "≋",
        createdAt: "Today",
        favorite: false,
    },
    {
        id: "asset-music-warm",
        name: "Warm Bed",
        kind: "audio",
        duration: 32,
        colors: { a: "#5b4730", b: "#8d6840" },
        tags: ["music", "warm", "ambient"],
        glyph: "≋",
        createdAt: "Yesterday",
        favorite: false,
    },
    {
        id: "asset-title-clean",
        name: "Clean Lower Third",
        kind: "title",
        duration: 3.5,
        colors: { a: "#2c2637", b: "#6e5d8a" },
        tags: ["title", "lower third", "clean"],
        glyph: "T",
        createdAt: "Today",
        favorite: false,
    },
    {
        id: "asset-title-hero",
        name: "Hero Title",
        kind: "title",
        duration: 4,
        colors: { a: "#2b2233", b: "#8a5da0" },
        tags: ["title", "hero", "intro"],
        glyph: "T",
        createdAt: "Today",
        favorite: true,
    },
    {
        id: "asset-transition-dissolve",
        name: "Cross Dissolve",
        kind: "transition",
        duration: 1,
        colors: { a: "#596b8c", b: "#8a557a" },
        tags: ["transition", "dissolve"],
        glyph: "⋈",
        createdAt: "Today",
        favorite: false,
    },
    {
        id: "asset-transition-zoom",
        name: "Radial Zoom",
        kind: "transition",
        duration: 0.8,
        colors: { a: "#385e91", b: "#9a50a4" },
        tags: ["transition", "zoom", "libplacebo"],
        glyph: "✦",
        createdAt: "Today",
        favorite: false,
    },
];
const assetById = new Map(fixtureAssets.map((asset) => [asset.id, asset]));
function requireAsset(assetId) {
    const asset = assetById.get(assetId);
    if (!asset) {
        throw new Error(`Fixture asset ${assetId} is missing.`);
    }
    return asset;
}
export function makeTimelineClip(id, assetId, duration, overrides = {}) {
    const asset = requireAsset(assetId);
    // role is derived by normalizeProject; seed a placeholder for type completeness.
    return {
        id,
        assetId,
        name: asset.name,
        kind: asset.kind,
        role: "storyline",
        sourceStart: 0,
        duration: duration ?? Math.min(asset.duration, 5),
        timelineStart: 0,
        colors: { ...asset.colors },
        transform: defaultTransform(),
        video: defaultVideo(),
        audio: defaultAudio(),
        keyframes: defaultKeyframes(),
        timeMap: null,
        text: asset.kind === "title" ? asset.name : null,
        textStyle: null,
        caption: null,
        generatorColor: null,
        markers: [],
        effects: [],
        effectStack: [],
        ...overrides,
    };
}
export function makeConnectedClip(id, assetId, anchorId, anchorOffset, duration, overrides = {}) {
    return {
        ...makeTimelineClip(id, assetId, duration, overrides),
        anchorId,
        anchorOffset,
        lane: 1,
        ...overrides,
    };
}
export function makeGapClip(id, duration) {
    return {
        id,
        assetId: null,
        name: "Gap",
        kind: "gap",
        role: "storyline",
        sourceStart: 0,
        duration,
        timelineStart: 0,
        colors: { a: "#2b2e34", b: "#1c1f24" },
        transform: defaultTransform(),
        video: defaultVideo(),
        audio: { ...defaultAudio(), muted: true, gainDb: -96 },
        keyframes: defaultKeyframes(),
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
function project(id, eventId, name, spine, connected, proposal = null, transitions = []) {
    return normalizeProject({
        revision: 1,
        id,
        libraryId: "library-demo",
        eventId,
        name,
        fps: 30,
        width: 1920,
        height: 1080,
        spine,
        connected,
        transitions,
        proposal,
    });
}
const sourceProject = project("project-product-story", "event-launch-film", "Product Story", [
    makeTimelineClip("clip-intro", "asset-city-night", 3.8),
    makeTimelineClip("clip-host-wide", "asset-host-wide", 5.1, { sourceStart: 1.2 }),
    makeTimelineClip("clip-product", "asset-product-spin", 4.2),
    makeTimelineClip("clip-ui", "asset-ui-capture", 5.6, { sourceStart: 2.1 }),
    makeTimelineClip("clip-host-close", "asset-host-close", 4.1, { sourceStart: 0.8 }),
], [
    makeConnectedClip("connected-hero", "asset-title-hero", "clip-intro", 0.3, 2.8, {
        text: "Editing, accelerated.",
        transform: { ...defaultTransform(), y: 0.22, scale: 0.92 },
    }),
    makeConnectedClip("connected-desk", "asset-desk-detail", "clip-host-wide", 1.4, 2.5, {
        transform: { ...defaultTransform(), scale: 0.42, x: 0.25, y: -0.2 },
    }),
    makeConnectedClip("connected-logo", "asset-logo", "clip-host-close", 2.6, 1.2, {
        transform: { ...defaultTransform(), scale: 0.28, x: 0.28, y: 0.25 },
    }),
    makeConnectedClip("connected-music", "asset-music-pulse", "clip-intro", 0, 22.5, {
        audio: { ...defaultAudio(), gainDb: -5.4 },
    }),
], null, [
    {
        id: "transition-intro-host", name: "Cross Dissolve", category: "Dissolves",
        leftItemId: "clip-intro", rightItemId: "clip-host-wide", duration: 1,
        resourceId: "fixture-cross-dissolve", resourceUid: "FxPlug:4731E73A-8DAC-4113-9A30-AE85B1761265",
        handler: "cross_dissolve", support: "approximate", parameters: {}, parameterNames: {}, parameterKeyframes: {},
    },
    {
        id: "transition-product-ui", name: "Cross Dissolve", category: "Dissolves",
        leftItemId: "clip-product", rightItemId: "clip-ui", duration: 0.8,
        resourceId: "fixture-cross-dissolve", resourceUid: "FxPlug:4731E73A-8DAC-4113-9A30-AE85B1761265",
        handler: "cross_dissolve", support: "approximate", parameters: {}, parameterNames: {}, parameterKeyframes: {},
    },
]);
const proposalMetadata = {
    baseProjectId: sourceProject.id,
    baseProjectName: sourceProject.name,
    agentName: "Bladeworks Agent",
    prompt: "Make the opening more immediate and keep the product visible during the explanation.",
    createdAt: "2026-08-11T18:42:00Z",
};
const tighterProposal = project("project-tighter-opening", "event-launch-film", "Product Story · Tighter Opening", [
    makeTimelineClip("proposal-product", "asset-product-spin", 2.1, { sourceStart: 0.4 }),
    makeTimelineClip("proposal-host", "asset-host-close", 3.9, { sourceStart: 1.1 }),
    makeTimelineClip("proposal-ui", "asset-ui-capture", 5.1, { sourceStart: 2.6 }),
    makeTimelineClip("proposal-wide", "asset-host-wide", 3.7, { sourceStart: 6.2 }),
], [
    makeConnectedClip("proposal-title", "asset-title-hero", "proposal-product", 0.15, 1.9, {
        text: "Meet Bladeworks",
        transform: { ...defaultTransform(), scale: 1.08 },
    }),
    makeConnectedClip("proposal-desk", "asset-desk-detail", "proposal-host", 0.8, 2.2, {
        transform: { ...defaultTransform(), scale: 0.46, x: 0.24, y: -0.2 },
    }),
    makeConnectedClip("proposal-music", "asset-music-pulse", "proposal-product", 0, 14.8, {
        audio: { ...defaultAudio(), gainDb: -4.2 },
    }),
], proposalMetadata);
const socialCut = project("project-social-cut", "event-social-cuts", "30s Vertical Teaser", [
    makeTimelineClip("social-host", "asset-host-close", 4.3),
    makeTimelineClip("social-product", "asset-product-spin", 3.4),
    makeTimelineClip("social-ui", "asset-ui-capture", 5.2, { sourceStart: 4.5 }),
    makeTimelineClip("social-detail", "asset-desk-detail", 3.1),
], [
    makeConnectedClip("social-lower", "asset-title-clean", "social-host", 0.5, 2.8, {
        text: "Tony · Bladeworks",
        transform: { ...defaultTransform(), x: -0.2, y: 0.26, scale: 0.72 },
    }),
    makeConnectedClip("social-music", "asset-music-warm", "social-host", 0, 16, {
        audio: { ...defaultAudio(), gainDb: -7 },
    }),
]);
const selects = project("project-selects", "event-archive", "Interview Selects", [
    makeTimelineClip("select-wide", "asset-host-wide", 6.4, { sourceStart: 0.6 }),
    makeGapClip("select-gap", 1),
    makeTimelineClip("select-close", "asset-host-close", 5.8, { sourceStart: 1.9 }),
], []);
const projects = {
    [sourceProject.id]: sourceProject,
    [tighterProposal.id]: tighterProposal,
    [socialCut.id]: socialCut,
    [selects.id]: selects,
};
function summary(projectSnapshot) {
    return {
        id: projectSnapshot.id,
        eventId: projectSnapshot.eventId,
        name: projectSnapshot.name,
        duration: projectDuration(projectSnapshot),
        proposal: projectSnapshot.proposal,
        openError: null,
    };
}
export function createFixtureBootstrap() {
    return {
        libraries: [
            {
                id: "library-demo",
                name: "Bladeworks Demo Library",
                events: [
                    {
                        id: "event-launch-film",
                        libraryId: "library-demo",
                        name: "Launch Film",
                        projects: [summary(sourceProject), summary(tighterProposal)],
                    },
                    {
                        id: "event-social-cuts",
                        libraryId: "library-demo",
                        name: "Social Cuts",
                        projects: [summary(socialCut)],
                    },
                    {
                        id: "event-archive",
                        libraryId: "library-demo",
                        name: "Archive",
                        projects: [summary(selects)],
                    },
                ],
            },
        ],
        assets: fixtureAssets,
        projects,
        activeProjectId: sourceProject.id,
    };
}
export function kindLabel(kind) {
    switch (kind) {
        case "video": return "Video";
        case "audio": return "Audio";
        case "image": return "Image";
        case "title": return "Title";
        case "caption": return "Caption";
        case "generator": return "Generator";
        case "transition": return "Transition";
        case "gap": return "Gap Clip";
        default: {
            const exhaustive = kind;
            return exhaustive;
        }
    }
}
