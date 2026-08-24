/**
 * Effects and Transitions browser docked on the timeline's right edge.
 *
 * Architecture map:
 * capabilities catalog + selection
 *   -> catalogBrowserTemplate HTML
 *   -> apply-effect / apply-transition actions
 *
 * Main callers: BladeworksEditorApp.renderTimeline
 */
import { effectCatalogItem, transitionCatalogItem } from "../capability-ui.js";
import { escapeHtml } from "../ui.js";
import { icon } from "./icons.js";
import { selectedStorylineEdit, selectedTimelineClip } from "./timeline.js";
export function catalogBrowserTemplate(state) {
    if (!state.activeBrowser) {
        return "";
    }
    const isEffects = state.activeBrowser === "effects";
    const certifiedIds = isEffects
        ? new Set(["effect-color-adjustments"])
        : new Set(["transition-cross-dissolve"]);
    const catalog = (isEffects ? state.capabilities.effects : state.capabilities.transitions)
        .filter((item) => item.authorable && item.support !== "unsupported")
        .filter((item) => state.connectionMode !== "localhost" || certifiedIds.has(item.id))
        .map(isEffects ? effectCatalogItem : transitionCatalogItem);
    const title = isEffects ? "Effects" : "Transitions";
    const applyAction = isEffects ? "apply-effect" : "apply-transition";
    const categories = [...new Set(catalog.map((item) => item.category))];
    const query = state.browserQuery.trim().toLowerCase();
    const activeCategory = state.browserCategory ?? "";
    const items = catalog.filter((item) => {
        if (activeCategory && item.category !== activeCategory) {
            return false;
        }
        if (query && !`${item.capability.name} ${item.capability.handler ?? ""}`.toLowerCase().includes(query)) {
            return false;
        }
        return true;
    });
    const target = isEffects ? effectsTargetLabel(state) : transitionsTargetLabel(state);
    const chips = [["", "All"], ...categories.map((c) => [c, c])];
    const categoryChips = chips
        .map(([id, label]) => `<button class="catalog-chip ${activeCategory === id ? "active" : ""}" data-action="browser-category" data-category="${escapeHtml(id)}">${escapeHtml(label)}</button>`)
        .join("");
    const tiles = items.map((item) => catalogTile(item, applyAction, isEffects)).join("")
        || `<p class="catalog-empty">No ${title.toLowerCase()} match “${escapeHtml(state.browserQuery)}”.</p>`;
    return `
    <header class="catalog-header">
      <div class="catalog-title">${icon(isEffects ? "effects" : "transitions")}<strong>${title}</strong></div>
      <button class="catalog-close" data-action="close-browser" aria-label="Close ${title} browser">×</button>
    </header>
    ${catalog.length > 5 ? `<div class="catalog-search"><span class="catalog-search-icon">${icon("search")}</span><input type="search" placeholder="Search ${title}" value="${escapeHtml(state.browserQuery)}" data-action="browser-search"></div><div class="catalog-chips">${categoryChips}</div>` : ""}
    <div class="catalog-target ${target.ready ? "ready" : "empty"}">${escapeHtml(target.text)}</div>
    <div class="catalog-grid">${tiles}</div>
    <footer class="catalog-footer">${icon("magic")}<span>${state.connectionMode === "localhost" ? `Certified ${title.toLowerCase()}` : "Fixture-mode catalog; localhost loads the live tensor registry."}</span></footer>
  `;
}
function catalogTile(item, applyAction, isEffects) {
    const swatch = `cat-${item.category.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`;
    const capability = item.capability;
    return `<button class="catalog-tile" data-action="${applyAction}" data-capability-id="${escapeHtml(capability.id)}" title="Apply ${escapeHtml(capability.name)}"><span class="catalog-thumb ${swatch}">${icon(isEffects ? "effects" : "transitions")}</span><span class="catalog-meta"><span class="catalog-name">${escapeHtml(capability.name)}</span><span class="catalog-cat">${escapeHtml(item.category)}</span></span></button>`;
}
function effectsTargetLabel(state) {
    const item = selectedTimelineClip(state);
    if (!item) {
        return { ready: false, text: "Select a clip, then choose an effect to apply" };
    }
    return { ready: true, text: `Applies to: ${item.name}` };
}
function transitionsTargetLabel(state) {
    const edit = selectedStorylineEdit(state);
    if (!edit) {
        return { ready: false, text: "Select a storyline clip that has a clip after it" };
    }
    return { ready: true, text: `Applies between: ${edit.left.name} → ${edit.right.name}` };
}
