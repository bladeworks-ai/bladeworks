/**
 * Safe UI projection for Bladeworks's non-tracked mask model.
 *
 * The capability document owns labels, ranges, units, and animation flags.
 * The codec constructors own exact FCPXML keys and isolation JSON. This module
 * only translates edited UI values back through those constructors.
 */
import { createColorMask, createDrawMask, createLumaMask, createShapeMask, } from "./fcpxml.js";
export function defaultMask(kind, id) {
    switch (kind) {
        case "shape": return createShapeMask({ id });
        case "draw": return createDrawMask({
            id,
            points: [{ x: -160, y: 120 }, { x: 0, y: -160 }, { x: 160, y: 120 }],
        });
        case "color": return createColorMask({ id, color: [0, 1, 0] });
        case "luma": return createLumaMask({ id });
    }
}
function isolationData(mask) {
    if (!mask.data)
        return {};
    const value = JSON.parse(mask.data);
    if (!value || typeof value !== "object" || Array.isArray(value)) {
        throw new Error(`${mask.name} has invalid isolation data.`);
    }
    return value;
}
function drawPoints(mask) {
    const alias = Object.entries(mask.parameters).find(([key]) => ["points", "vertices", "path", "300"].includes(key.toLowerCase()));
    const raw = alias?.[1];
    if (typeof raw !== "string")
        return [];
    return raw.split(";").map((pair) => {
        const [x, y] = pair.split(",").map(Number);
        if (!Number.isFinite(x) || !Number.isFinite(y))
            throw new Error(`${mask.name} has invalid draw points.`);
        return { x: x, y: y };
    });
}
function canonicalKeyframes(mask) {
    if (mask.kind === "shape") {
        const output = {};
        for (const [key, frames] of Object.entries(mask.parameterKeyframes)) {
            const canonical = Object.entries(SHAPE_PARAMETER_ALIASES).find(([, aliases]) => aliases.includes(key.toLowerCase()))?.[0] ?? key;
            output[canonical] = frames;
        }
        return output;
    }
    if (mask.kind !== "draw" || !mask.parameterKeyframes["103"]) {
        return mask.parameterKeyframes;
    }
    const { "103": legacyOpacity, ...rest } = mask.parameterKeyframes;
    return { ...rest, opacity: mask.parameterKeyframes.opacity ?? legacyOpacity };
}
const SHAPE_PARAMETER_ALIASES = {
    "160": ["160", "radius", "shape radius"],
    "201": ["201", "position", "center"],
    "202": ["202", "rotation"],
    "159": ["159", "curvature", "roundness"],
    "102": ["102", "feather"],
    "103": ["103", "opacity", "amount"],
    "104": ["104", "falloff"],
};
function shapeValues(mask) {
    const entries = Object.entries(mask.parameters);
    return Object.fromEntries(Object.entries(SHAPE_PARAMETER_ALIASES).flatMap(([canonical, aliases]) => {
        const match = entries.find(([key]) => aliases.includes(key.toLowerCase()));
        return match ? [[canonical, match[1]]] : [];
    }));
}
function canonicalShapeKey(key) {
    const normalized = key.toLowerCase();
    return Object.entries(SHAPE_PARAMETER_ALIASES).find(([, aliases]) => aliases.includes(normalized))?.[0] ?? key;
}
function validateShapeValues(mask, values, parameterKeyframes) {
    createShapeMask({
        id: mask.id,
        name: mask.name,
        enabled: mask.enabled,
        blendMode: mask.blendMode,
        parameterKeyframes,
        ...(values["160"] === undefined ? {} : { radius: values["160"] }),
        ...(values["201"] === undefined ? {} : { position: values["201"] }),
        ...(values["202"] === undefined ? {} : { rotation: Number(values["202"]) }),
        ...(values["159"] === undefined ? {} : { curvature: Number(values["159"]) }),
        ...(values["102"] === undefined ? {} : { feather: Number(values["102"]) }),
        ...(values["103"] === undefined ? {} : { opacity: Number(values["103"]) }),
        ...(values["104"] === undefined ? {} : { falloff: Number(values["104"]) }),
    });
}
export function maskUiValues(mask) {
    if (mask.kind === "shape")
        return shapeValues(mask);
    if (mask.kind === "draw")
        return {
            points: drawPoints(mask),
            opacity: mask.parameters.opacity ?? mask.parameters["103"] ?? 1,
        };
    const data = isolationData(mask);
    if (mask.kind === "color") {
        const color = Array.isArray(data.color) ? data.color.map(Number) : [0, 1, 0];
        return {
            color: { red: color[0] ?? 0, green: color[1] ?? 1, blue: color[2] ?? 0, alpha: 1 },
            tolerance: Number(data.tolerance ?? 0.12),
            softness: Number(data.softness ?? 0.05),
            opacity: Number(data.opacity ?? 1),
        };
    }
    return {
        luma_min: Number(data.luma_min ?? 0),
        luma_max: Number(data.luma_max ?? 1),
        softness: Number(data.softness ?? 0.05),
        opacity: Number(data.opacity ?? 1),
    };
}
/** Rebuild one source through its certified constructor after a UI edit. */
export function updateMaskValue(mask, key, value) {
    const common = {
        id: mask.id,
        name: mask.name,
        enabled: mask.enabled,
        blendMode: mask.blendMode,
        parameterKeyframes: canonicalKeyframes(mask),
    };
    const editedKey = mask.kind === "shape" ? canonicalShapeKey(key) : key;
    const values = { ...maskUiValues(mask), [editedKey]: value };
    switch (mask.kind) {
        case "shape": {
            validateShapeValues(mask, values, common.parameterKeyframes);
            const authoredKey = maskKeyframeKey(mask, editedKey);
            return { ...mask, parameters: { ...mask.parameters, [authoredKey]: value } };
        }
        case "draw": return createDrawMask({
            ...common,
            points: values.points,
            opacity: Number(values.opacity),
        });
        case "color": {
            const color = values.color;
            return createColorMask({
                ...common,
                color: [color.red, color.green, color.blue],
                tolerance: Number(values.tolerance),
                softness: Number(values.softness),
                opacity: Number(values.opacity),
            });
        }
        case "luma": return createLumaMask({
            ...common,
            minimum: Number(values.luma_min),
            maximum: Number(values.luma_max),
            softness: Number(values.softness),
            opacity: Number(values.opacity),
        });
    }
}
export function maskKeyframeKey(mask, capabilityKey) {
    if (mask.kind !== "shape")
        return capabilityKey;
    const aliases = SHAPE_PARAMETER_ALIASES[capabilityKey] ?? [capabilityKey];
    return Object.keys(mask.parameters).find((key) => aliases.includes(key.toLowerCase())) ?? capabilityKey;
}
export function withMaskKeyframes(mask, key, frames) {
    const parameterKeyframes = {
        ...canonicalKeyframes(mask),
        [mask.kind === "shape" ? canonicalShapeKey(key) : key]: frames,
    };
    switch (mask.kind) {
        case "shape": {
            const values = shapeValues(mask);
            const authoredKey = maskKeyframeKey(mask, key);
            if (!Object.hasOwn(mask.parameters, authoredKey)) {
                throw new Error(`${mask.name} does not author the ${key} parameter.`);
            }
            validateShapeValues(mask, values, parameterKeyframes);
            return {
                ...mask,
                parameterKeyframes: { ...mask.parameterKeyframes, [authoredKey]: frames },
            };
        }
        case "draw": return createDrawMask({
            id: mask.id, name: mask.name, enabled: mask.enabled, blendMode: mask.blendMode,
            points: drawPoints(mask), opacity: Number(mask.parameters.opacity ?? mask.parameters["103"]), parameterKeyframes,
        });
        case "color":
        case "luma": throw new Error(`${mask.name} parameters are not animatable.`);
    }
}
