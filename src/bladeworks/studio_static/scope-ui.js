/**
 * Projection helpers for editing one nested FCPXML timeline at a time.
 *
 * Architecture map:
 * canonical root ProjectSnapshot
 *   -> projectForScope creates a reducer-compatible timeline projection
 *   -> ordinary magnetic-timeline operations edit that projection
 *   -> replaceScopeProject grafts it into ProjectSnapshot.scopes
 *   -> EditorRuntime.restoreProject performs one complete-library PUT
 *
 * The root Project id and projectRef never change. Scope ids only identify an
 * editable timeline inside that Project and are never sent to preview/render.
 */
export function projectForScope(root, scopeId) {
    if (!scopeId)
        return structuredClone(root);
    const scope = root.scopes?.[scopeId];
    if (!scope)
        throw new Error(`Nested timeline ${scopeId} no longer exists.`);
    return {
        ...structuredClone(root),
        name: scope.name,
        fps: 1 / Math.max(scope.clock.frameDuration.seconds, 1 / 240),
        spine: structuredClone(scope.spine),
        connected: structuredClone(scope.connected),
        transitions: structuredClone(scope.transitions),
    };
}
/** Graft a reducer-edited projection back into the immutable root Project. */
export function replaceScopeProject(root, scopeId, edited) {
    const current = root.scopes?.[scopeId];
    if (!current)
        throw new Error(`Nested timeline ${scopeId} no longer exists.`);
    if (!current.editable)
        throw new Error(`Nested timeline is read-only: ${current.reasons.join(" ")}`);
    const spineIds = edited.spine.map((clip) => clip.id);
    const connectedIds = edited.connected.map((clip) => clip.id);
    const originalSpine = new Set(current.spine.map((clip) => clip.id));
    const originalConnected = new Set(current.connected.map((clip) => clip.id));
    const live = new Set([...spineIds, ...connectedIds]);
    const remainingSpine = [...spineIds];
    const remainingConnected = [...connectedIds];
    const childOrder = current.kind === "sync"
        ? [
            ...current.childOrder.filter((id) => live.has(id)).map((id) => {
                const queue = originalSpine.has(id) ? remainingSpine : originalConnected.has(id) ? remainingConnected : [];
                return queue.shift() ?? id;
            }),
            ...remainingSpine,
            ...remainingConnected,
        ]
        : spineIds;
    const scope = {
        ...current,
        spine: structuredClone(edited.spine),
        connected: structuredClone(edited.connected),
        transitions: structuredClone(edited.transitions),
        childOrder,
    };
    return {
        ...structuredClone(root),
        scopes: { ...structuredClone(root.scopes ?? {}), [scopeId]: scope },
    };
}
export function scopeBreadcrumbs(root, activeScopeId, explicitPath) {
    const result = [{ scopeId: "", label: root.name, kind: "compound" }];
    if (explicitPath) {
        for (const scopeId of explicitPath) {
            const scope = root.scopes?.[scopeId];
            if (scope)
                result.push({ scopeId, label: scope.name, kind: scope.kind });
        }
        return result;
    }
    const chain = [];
    let cursor = activeScopeId;
    const seen = new Set();
    while (cursor) {
        if (seen.has(cursor))
            throw new Error(`Nested timeline ancestry loops at ${cursor}.`);
        seen.add(cursor);
        const scope = root.scopes?.[cursor];
        if (!scope)
            break;
        chain.push(scope);
        cursor = scope.parentScopeId;
    }
    for (const scope of chain.reverse())
        result.push({ scopeId: scope.id, label: scope.name, kind: scope.kind });
    return result;
}
export function scopeTargets(root, clip) {
    const container = clip.container;
    if (!container)
        return [];
    const target = (scopeId) => {
        const scope = root.scopes?.[scopeId];
        return scope ? { scopeId, label: scope.name, kind: scope.kind } : null;
    };
    if (container.kind === "compound" || container.kind === "sync") {
        const item = target(container.scopeId);
        return item ? [item] : [];
    }
    if (container.kind === "multicam") {
        return Object.values(container.angleScopeIds).map(target).filter((item) => Boolean(item));
    }
    return container.choiceScopeIds.map(target).filter((item) => Boolean(item));
}
export function defaultScopeTarget(root, clip) {
    const container = clip.container;
    if (!container)
        return null;
    if (container.kind === "compound" || container.kind === "sync")
        return container.scopeId;
    if (container.kind === "multicam") {
        const angle = container.videoAngleId ?? container.audioAngleId;
        return (angle ? container.angleScopeIds[angle] : null) ?? Object.values(container.angleScopeIds)[0] ?? null;
    }
    return container.activeChoiceId ?? container.choiceScopeIds[0] ?? null;
}
export function replaceClipContainer(project, clipId, container) {
    let found = false;
    const update = (clip) => {
        if (clip.id !== clipId)
            return structuredClone(clip);
        found = true;
        return { ...structuredClone(clip), container };
    };
    const next = {
        ...structuredClone(project),
        spine: project.spine.map(update),
        connected: project.connected.map(update),
    };
    if (!found)
        throw new Error(`Container clip ${clipId} does not exist in the active timeline.`);
    return next;
}
export function replaceClipMetadata(project, clipId, patch) {
    let found = false;
    const update = (clip) => {
        if (clip.id !== clipId)
            return structuredClone(clip);
        found = true;
        return { ...structuredClone(clip), ...patch };
    };
    const next = { ...structuredClone(project), spine: project.spine.map(update), connected: project.connected.map(update) };
    if (!found)
        throw new Error(`Clip ${clipId} does not exist in the active timeline.`);
    return next;
}
