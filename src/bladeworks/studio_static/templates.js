/**
 * FCP-inspired HTML projections for the browser editor.
 *
 * Architecture map
 * ================
 *
 * EditorState + canonical browser snapshot
 *   -> stateless template functions, one file per panel under templates/
 *   -> dense, native-editor-like DOM surface
 *   -> event delegation in app.ts
 *
 * Why this exists:
 * The public import path stays `./templates.js` so app.ts and tests do not
 * churn. Chrome edits go to the matching templates/*.ts file.
 *
 * The templates intentionally expose stable `data-action` and
 * `data-parameter-path` attributes. Those attributes are the browser-side ABI:
 * both inspector controls and viewer onscreen controls write the same dotted
 * parameter paths, so the localhost FCPXML runtime can map one command surface
 * instead of learning every widget independently.
 */
export { escapeHtml } from "./ui.js";
export { icon } from "./templates/icons.js";
export { shellTemplate } from "./templates/shell.js";
export { topbarTemplate } from "./templates/topbar.js";
export { libraryTemplate } from "./templates/library.js";
export { mediaGridTemplate, mediaTemplate } from "./templates/media.js";
export { canvasControlsTemplate, transportTemplate, viewerControlStripTemplate, viewerToolbarTemplate, } from "./templates/viewer.js";
export { inspectorTemplate } from "./templates/inspector.js";
export { selectedStorylineEdit, selectedTimelineClip, timelineIndexTemplate, timelineTemplate, timelineToolbarTemplate, } from "./templates/timeline.js";
export { catalogBrowserTemplate } from "./templates/catalog.js";
