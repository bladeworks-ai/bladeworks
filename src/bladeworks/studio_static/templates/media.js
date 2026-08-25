/**
 * Media browser: search, filters, list/grid, Project cards, asset cards.
 *
 * Architecture map:
 * MediaAsset[] + browser options
 *   -> mediaTemplate chrome + mediaGridTemplate cards
 *   -> data-action browser filters and select-project
 *
 * Main callers: BladeworksEditorApp.renderAll / refreshBrowser
 */
import { escapeHtml, formatTimecode } from "../ui.js";
import { icon } from "./icons.js";
import { assetVisualIndex, mediaVisualAttributes } from "./shared.js";
export function mediaTemplate(assets, options) {
    const filterTabs = [
        ["media", "Media"],
        ["audio", "Audio"],
        ["titles", "Titles"],
    ];
    const showsGeneratedItems = options.activeTab === "titles";
    return `
    <div class="browser-titlebar">
      ${showsGeneratedItems ? '<strong class="generated-browser-title">Titles and Generators</strong>' : `<div class="browser-scope">
        <button class="scope-button active ${options.activePopover === "browser-scope" ? "open" : ""}" data-action="browser-scope">${browserScopeLabel(options.scope)} ${icon("chevron-down")}</button>${browserScopePopover(options)}
      </div>
      <div class="browser-view-controls">
        <button class="mini-chrome ${options.activePopover === "browser-sort" ? "active" : ""}" data-action="browser-sort" title="Sort and group">${icon("list-sort")}</button>${browserSortPopover(options)}
        <button class="mini-chrome ${options.view === "list" ? "active" : ""}" data-action="browser-view" data-view="list" title="List view">${icon("list")}</button>
        <button class="mini-chrome ${options.view === "grid" ? "active" : ""}" data-action="browser-view" data-view="grid" title="Filmstrip view">${icon("grid")}</button>
      </div>`}
    </div>
    ${showsGeneratedItems ? "" : `<div class="browser-search-row">
      <label class="fcp-search">${icon("search")}<input id="asset-search" type="search" placeholder="Search names and tags" value="${escapeHtml(options.query)}">${options.query ? '<button class="search-clear" data-action="clear-browser-search" title="Clear Search">×</button>' : ""}</label>
      <button class="mini-chrome" data-action="refresh-media" title="Refresh Media">${icon("reset")}</button>
    </div>`}
    <div class="browser-filter-row">
      ${filterTabs.map(([id, label]) => `<button class="browser-filter ${options.activeTab === id ? "active" : ""}" data-action="media-tab" data-tab="${id}">${label}</button>`).join("")}
      <span class="browser-event-name">${escapeHtml(options.eventName)}</span>
    </div>
    ${showsGeneratedItems ? generatedClipShelf() : `<div class="asset-browser ${options.view}" id="asset-grid">${mediaGridTemplate(assets, options)}</div>
    <div class="browser-status" id="browser-status">${browserStatus(assets, options)}</div>`}
  `;
}
function generatedClipShelf() {
    return `<div class="generated-clip-shelf">
    <button data-action="insert-basic-title"><span class="generated-icon">T</span><strong>Basic Title</strong><small>Connected at playhead</small></button>
    <button data-action="insert-caption"><span class="generated-icon">CC</span><strong>Caption</strong><small>Pop-on dialogue</small></button>
    <button data-action="insert-custom-solid"><span class="generated-icon solid"></span><strong>Custom Solid</strong><small>Primary storyline</small></button>
  </div>`;
}
function browserSortPopover(options) {
    if (options.activePopover !== "browser-sort")
        return "";
    const rows = [
        ["date", "Date Created"],
        ["name", "Name"],
        ["duration", "Duration"],
        ["favorites", "Favorites First"],
    ];
    return `<div class="browser-sort-popover">${rows.map(([value, label]) => `<button data-action="set-browser-sort" data-value="${value}" class="${options.sort === value ? "active" : ""}">${options.sort === value ? icon("check") : '<span class="check-placeholder"></span>'}<span>${label}</span></button>`).join("")}</div>`;
}
function browserScopeLabel(scope) {
    const labels = {
        all: "All Clips",
        favorites: "Favorites",
        video: "Video Only",
        stills: "Stills",
    };
    return labels[scope];
}
function browserScopePopover(options) {
    if (options.activePopover !== "browser-scope")
        return "";
    const rows = [
        ["all", "All Clips"],
        ["favorites", "Favorites"],
        ["video", "Video Only"],
        ["stills", "Stills"],
    ];
    return `<div class="browser-scope-popover">${rows.map(([value, label]) => `<button data-action="set-browser-scope" data-value="${value}" class="${options.scope === value ? "active" : ""}">${options.scope === value ? icon("check") : '<span class="check-placeholder"></span>'}<span>${label}</span></button>`).join("")}</div>`;
}
function filteredMediaAssets(assets, options) {
    const query = options.query.toLowerCase();
    return assets.filter((asset) => {
        const tabMatches = options.activeTab === "media"
            ? ["video", "image"].includes(asset.kind)
            : options.activeTab === "audio"
                ? asset.kind === "audio"
                : asset.kind === "title";
        const scope = options.scope;
        const scopeMatches = scope === "all"
            || (scope === "favorites" && Boolean(asset.favorite))
            || (scope === "video" && asset.kind === "video")
            || (scope === "stills" && asset.kind === "image");
        const haystack = `${asset.name} ${asset.tags.join(" ")}`.toLowerCase();
        return tabMatches && scopeMatches && haystack.includes(query);
    });
}
function visibleAssetCount(assets, options) {
    return filteredMediaAssets(assets, options).length;
}
/**
 * Return Projects that belong in the selected Event's browser.
 *
 * Projects are first-class FCP browser items. They appear with visual media in
 * the Media/All Clips view, participate in search, and open through the same
 * select-project action as the Library sidebar. Audio, Titles, and clip-only
 * smart filters intentionally exclude them.
 *
 * Main callers: mediaGridTemplate and browserStatus.
 */
function filteredBrowserProjects(options) {
    if (options.activeTab !== "media" || options.scope !== "all")
        return [];
    const query = options.query.trim().toLowerCase();
    return options.eventProjects.filter((project) => project.name.toLowerCase().includes(query));
}
function browserStatus(assets, options) {
    const clips = visibleAssetCount(assets, options);
    const projects = filteredBrowserProjects(options).length;
    const visible = clips + projects;
    const totalProjects = options.activeTab === "media" && options.scope === "all" ? options.eventProjects.length : 0;
    return `${visible} of ${assets.length + totalProjects} items; ${clips} ${clips === 1 ? "clip" : "clips"}, ${projects} ${projects === 1 ? "Project" : "Projects"}`;
}
export function mediaGridTemplate(assets, options) {
    const filtered = filteredMediaAssets(assets, options);
    const projects = filteredBrowserProjects(options);
    if (!filtered.length && !projects.length) {
        return `<div class="empty-browser">${icon("media")}<strong>No matching items</strong><span>Try another search or browser filter.</span></div>`;
    }
    const ordered = [...filtered].sort((a, b) => {
        if (options.sort === "name")
            return a.name.localeCompare(b.name);
        if (options.sort === "duration")
            return b.duration - a.duration;
        if (options.sort === "favorites") {
            return Number(Boolean(b.favorite)) - Number(Boolean(a.favorite)) || a.name.localeCompare(b.name);
        }
        return String(b.createdAt ?? "").localeCompare(String(a.createdAt ?? "")) || a.name.localeCompare(b.name);
    });
    const orderedProjects = [...projects].sort((a, b) => {
        if (options.sort === "duration")
            return b.duration - a.duration;
        return a.name.localeCompare(b.name);
    });
    return [
        ...orderedProjects.map((project) => projectBrowserCardTemplate(project, options.view, project.id === options.selectedProjectId)),
        ...ordered.map((asset, index) => assetCardTemplate(asset, options.view, asset.id === options.selectedAssetId, index)),
    ].join("");
}
function projectBrowserCardTemplate(project, view, selected) {
    const previewBars = Array.from({ length: view === "list" ? 7 : 5 }, (_, index) => `<i style="--project-bar:${index}"></i>`).join("");
    // Mirrors the Libraries sidebar: an uncompilable Project is greyed, carries
    // its error as the tooltip, and still answers a click with an explanation.
    const unopenable = project.openError !== null;
    const title = unopenable ? `Cannot open ${project.name}: ${project.openError}` : `Open Project ${project.name}`;
    return `<article class="asset-card project-browser-card ${selected ? "selected" : ""} ${unopenable ? "unopenable" : ""}" data-action="select-project" data-project-id="${escapeHtml(project.id)}" tabindex="0" title="${escapeHtml(title)}" aria-disabled="${unopenable}">
    <div class="asset-thumb project-browser-thumb">
      <span class="project-browser-icon">${icon("project")}</span>
      <div class="project-browser-timeline">${previewBars}</div>
    </div>
    <div class="asset-copy">
      <strong>${escapeHtml(project.name)}</strong>
      <span>Project</span>
      <span class="asset-time">${formatTimecode(project.duration, 29.97)}</span>
    </div>
    <span class="project-browser-open">${icon("disclosure-closed")}</span>
  </article>`;
}
function assetCardTemplate(asset, view, selected, index) {
    const duration = asset.kind === "image" || asset.kind === "title" ? "" : formatTimecode(asset.duration, 29.97);
    const visualAttributes = asset.kind === "title" || asset.kind === "transition"
        ? ""
        : mediaVisualAttributes(asset, 0, Math.max(0.01, asset.duration), asset.kind === "audio" ? 0 : (view === "list" ? 7 : 5), asset.kind === "audio" ? 28 : 0);
    return `
    <article class="asset-card ${selected ? "selected" : ""}" draggable="true" data-asset-id="${asset.id}" data-kind="${asset.kind}" tabindex="0" ${visualAttributes}>
      <div class="asset-thumb ${asset.kind === "audio" ? "audio-asset-thumb" : ""}" style="--asset-color:${asset.colors.a};--asset-accent:${asset.colors.b};--scene:${assetVisualIndex(asset.id)}">
        <div class="asset-filmstrip"></div>
        ${asset.kind === "audio" ? '<div class="browser-waveform"></div>' : ""}
      </div>
      <div class="asset-copy">
        <strong>${escapeHtml(asset.name)}</strong>
        <span>${escapeHtml(asset.createdAt ?? "Today")}</span>
        ${duration ? `<span class="asset-time">${duration}</span>` : ""}
      </div>
      <button class="favorite-dot ${asset.favorite ? "favorite" : ""}" data-action="toggle-favorite" data-asset-id="${asset.id}" title="Favorite">${asset.favorite ? "★" : "☆"}</button>
    </article>
  `;
}
