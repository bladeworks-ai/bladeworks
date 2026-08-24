/**
 * Top toolbar: import, project title, panel toggles, export.
 *
 * Architecture map:
 * EditorState + capabilities
 *   -> topbarTemplate HTML
 *   -> data-action buttons handled in app.ts
 *
 * Main callers: BladeworksEditorApp.renderTopbar
 */
import { activeMockCapabilities } from "../mock-capabilities.js";
import { escapeHtml } from "../ui.js";
import { icon } from "./icons.js";
import { createPopover, studioBuildId } from "./shared.js";
export function topbarTemplate(state) {
    const local = state.connectionMode === "localhost";
    return `
    <div class="topbar-left toolbar-cluster">
      <div class="create-menu-wrap">
        ${toolbarButton("toggle-create-menu", "plus", "Create Event or Project", `create-button ${state.activePopover === "create" ? "active" : ""}`)}
        ${createPopover(state.activePopover, Boolean(state.library))}
      </div>
      ${toolbarButton("import-media", "import", "Import Media (⌘I)")}
    </div>
    <div class="topbar-title">
      <span class="app-glyph">${icon("clapper")}</span>
      <strong>${escapeHtml(state.project.name)}</strong>
      <span class="project-format">${state.project.width}×${state.project.height} · ${state.project.fps.toFixed(2)}p</span>
      ${!state.projectEditable ? `<span class="save-state read-only" title="${escapeHtml(state.projectEditReasons.join("; "))}">Read only</span>` : state.isSaving ? `<span class="save-state saving">Saving…</span>` : local ? `<span class="save-state">Library source</span>` : `<button class="save-state fixture-state" data-action="show-mock-inventory" title="Show every remaining mock">Fixture · ${activeMockCapabilities(state.connectionMode).length} mocks</button>`}
      <span class="studio-build-id" title="Bladeworks Studio browser bundle ${studioBuildId()}">Build ${studioBuildId()}</span>
    </div>
    <div class="topbar-right toolbar-cluster">
      <span class="runtime-light ${local ? (state.connectionHealthy ? "local" : "offline") : ""}" title="${escapeHtml(state.connectionMessage)}"></span>
      ${panelButton("library", "sidebar-left", state.panels.library, "Libraries")}
      ${panelButton("browser", "browser", state.panels.browser, "Browser")}
      ${panelButton("timeline", "timeline", state.panels.timeline, "Timeline")}
      ${panelButton("inspector", "sliders", state.panels.inspector, "Inspector")}
      <div class="export-menu-wrap">
        ${toolbarButton("export", "share", state.exportProgress ? "Show export progress" : "Share / Export", `share-button ${state.activePopover === "export" ? "active" : ""} ${state.exportProgress ? "exporting" : ""}`)}
        ${exportPopover(state.activePopover, state.capabilities, state.exportResolution ?? 1080, state.exportProfile ?? "delivery", state.exportProgress ?? null)}
      </div>
    </div>
  `;
}
function exportPopover(activePopover, capabilities, resolution, profile, progress) {
    if (activePopover !== "export")
        return "";
    if (progress) {
        const ratio = progress.totalFrames > 0
            ? Math.min(1, progress.completedFrames / progress.totalFrames)
            : 0;
        const percent = Math.round(ratio * 100);
        const label = progress.status === "cancelling"
            ? "Cancelling export..."
            : progress.totalFrames > 0
                ? `Exporting ${percent}%`
                : "Preparing export...";
        return `<div class="export-popover export-progress" role="dialog" aria-label="Export Progress">
      <header><strong>${label}</strong></header>
      <progress max="1" value="${ratio}" aria-label="${label}"></progress>
      <button class="export-cancel" data-action="export-cancel" ${progress.status === "cancelling" ? "disabled" : ""}>${progress.status === "cancelling" ? "Cancelling..." : "Cancel"}</button>
    </div>`;
    }
    const resolutions = capabilities?.export.supportedResolutions ?? [1080];
    const formats = [
        { id: "delivery", name: "H.264" },
        { id: "delivery_alpha", name: "ProRes 4444" },
    ];
    const selectedProfile = formats.some((format) => format.id === profile) ? profile : "delivery";
    return `
    <div class="export-popover" role="dialog" aria-label="Export Video">
      <header><strong>Export Video</strong></header>
      <label class="export-field"><span>Resolution</span><select data-action="export-resolution">${resolutions.map((height) => `<option value="${height}" ${height === resolution ? "selected" : ""}>${height}p</option>`).join("")}</select></label>
      <label class="export-field"><span>Format</span><select data-action="export-format">${formats.map((format) => `<option value="${format.id}" data-profile="${format.id}" ${format.id === selectedProfile ? "selected" : ""}>${escapeHtml(format.name)}</option>`).join("")}</select></label>
      <button class="export-start" data-action="export-start">Export</button>
    </div>`;
}
function toolbarButton(action, iconName, title, extraClass = "", disabled = false) {
    return `<button class="chrome-button ${extraClass}" data-action="${action}" title="${escapeHtml(title)}" ${disabled ? "disabled" : ""}>${icon(iconName)}</button>`;
}
function panelButton(panel, iconName, active, title) {
    return `<button class="chrome-button panel-toggle ${active ? "active" : ""}" data-action="toggle-panel" data-panel="${panel}" title="Show or hide ${escapeHtml(title)}">${icon(iconName)}</button>`;
}
