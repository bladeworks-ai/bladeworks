/**
 * Transaction-level undo/redo state for one active Project.
 *
 * Architecture map:
 * accepted runtime ProjectSnapshot
 *   -> push prior snapshot into `past`
 *   -> clear `future`
 * undo/redo
 *   -> swap complete snapshots between stacks
 *
 * Why this exists:
 * Pointer movement can produce dozens of ephemeral visual updates. The editor
 * must store one history entry per accepted edit transaction, never one entry
 * per DOM event.
 */
export function emptyHistory() {
    return { past: [], future: [] };
}
export function recordHistory(history, previous) {
    return {
        past: [...history.past, structuredClone(previous)].slice(-100),
        future: [],
    };
}
export function undoHistory(history, current) {
    const previous = history.past[history.past.length - 1];
    if (!previous || previous.id !== current.id) {
        return null;
    }
    return {
        project: structuredClone(previous),
        history: {
            past: history.past.slice(0, -1),
            future: [structuredClone(current), ...history.future].slice(0, 100),
        },
    };
}
export function redoHistory(history, current) {
    const [next, ...remaining] = history.future;
    if (!next || next.id !== current.id) {
        return null;
    }
    return {
        project: structuredClone(next),
        history: {
            past: [...history.past, structuredClone(current)].slice(-100),
            future: remaining,
        },
    };
}
