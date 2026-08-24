/**
 * UI projection of Bladeworks's renderer-owned capability document.
 *
 * Architecture map
 * ================
 *
 * GET /api/editor/capabilities
 *   -> EffectCapability / TransitionCapability
 *   -> FCP-style catalog category and fidelity badge
 *   -> default parameter records used when an item is applied
 *
 * This module never invents localhost capabilities. The only classification
 * it adds is presentation-only grouping such as Color, Blur, or Wipes. Exact
 * parameter keys, value ranges, resource identities, and support labels stay
 * owned by Bladeworks's response.
 */
const EFFECT_CATEGORIES = [
    [/color|tint|negative|vibrancy|board|wheel/i, "Color"],
    [/blur|sharpen|focus|vignette/i, "Blur and Sharpen"],
    [/key|mask/i, "Keying and Masks"],
    [/warp|fisheye|droplet|kaleido|tile|earthquake|crop.*feather/i, "Distortion"],
    [/cartoon|camcorder|pixellate|threshold|noise|mirror|flip/i, "Stylize"],
    [/shadow|callout/i, "Utility"],
];
const TRANSITION_CATEGORIES = [
    [/dissolve|fade|bloom|flash|flare/i, "Dissolves and Light"],
    [/wipe|circle|clock|curtain|reveal/i, "Wipes"],
    [/slide|push|drop|swap|page|pan/i, "Movements"],
    [/blur|smear|warp|zoom/i, "Blurs and Distortion"],
    [/360|equirect/i, "360°"],
];
function categoryFor(capability, rules, fallback) {
    const text = `${capability.id} ${capability.name} ${capability.handler ?? ""} ${capability.resource.xfadeId ?? ""}`;
    return rules.find(([pattern]) => pattern.test(text))?.[1] ?? fallback;
}
export function effectCatalogItem(capability) {
    return { capability, category: categoryFor(capability, EFFECT_CATEGORIES, "Other") };
}
export function transitionCatalogItem(capability) {
    return { capability, category: categoryFor(capability, TRANSITION_CATEGORIES, "Other") };
}
export function supportLabel(support) {
    switch (support) {
        case "exact": return "Exact";
        case "approximate": return "Approximation";
        case "partial": return "Partial controls";
        case "default_only": return "Default only";
        case "unsupported": return "Unsupported";
    }
}
export function supportDetail(support) {
    switch (support) {
        case "exact": return "Bladeworks renders this authored surface exactly.";
        case "approximate": return "Bladeworks renders a documented portable approximation.";
        case "partial": return "Bladeworks honors only the controls shown here.";
        case "default_only": return "Bladeworks renders its calibrated default and exposes no authored controls.";
        case "unsupported": return "Bladeworks rejects this construct.";
    }
}
/** Exact capability-keyed values for a newly applied effect or transition. */
export function defaultCapabilityParameters(capability) {
    return Object.fromEntries(capability.parameters
        .filter((parameter) => parameter.default !== undefined)
        .map((parameter) => [parameter.key, structuredClone(parameter.default)]));
}
export function capabilityParameterNames(capability) {
    return Object.fromEntries(capability.parameters.map((parameter) => [parameter.key, parameter.name]));
}
