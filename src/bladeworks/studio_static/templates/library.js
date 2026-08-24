/**
 * Libraries sidebar: Library / Event / Project tree and source shelves.
 *
 * Architecture map:
 * LibrarySummary[] + selected ids
 *   -> libraryTemplate HTML
 *   -> data-action select-event / select-project / library-source
 *
 * Main callers: BladeworksEditorApp.renderAll
 */
import { escapeHtml } from "../ui.js";
import { icon } from "./icons.js";
export function libraryTemplate(libraries, selectedProjectId, selectedEventId, source = "libraries", connectionMode = "mock", expandedEventIds = libraries.flatMap((library) => library.events.map((event) => event.id))) {
    const sourceTabs = [
        ["libraries", "clapper", "Libraries"],
        ["titles-generators", "text", "Titles and Generators"],
    ];
    const eventCount = libraries.reduce((total, library) => total + library.events.length, 0);
    return `
    <div class="library-source-tabs">
      ${sourceTabs.map(([id, ico, title]) => `<button class="source-tab ${source === id ? "active" : ""}" data-action="library-source" data-source="${id}" title="${title}">${icon(ico)}</button>`).join("")}
    </div>
    <div class="library-tree-scroll">
      ${source === "libraries" ? libraries.map((library) => libraryTreeTemplate(library, selectedProjectId, selectedEventId, expandedEventIds)).join("") : sourceShelfTemplate(source)}
    </div>
    <div class="library-bottom-bar">
      ${source === "libraries" ? `<span>${eventCount} Events</span>` : `<span>Titles and Generators</span>`}
    </div>
  `;
}
function libraryTreeTemplate(library, selectedProjectId, selectedEventId, expandedEventIds) {
    return `
    <div class="library-root-row library-title-row" role="heading" aria-level="2">
      <span class="library-cube">${icon("library")}</span><strong>${escapeHtml(library.name)}</strong>
    </div>
    ${library.events.map((event) => {
        const expanded = expandedEventIds.includes(event.id);
        return `
      <section class="event-node">
        <button class="event-tree-row ${event.id === selectedEventId ? "selected" : ""}" data-action="toggle-event" data-event-id="${event.id}" aria-expanded="${expanded}">
          <span class="event-disclosure">${icon(expanded ? "disclosure-open" : "disclosure-closed")}</span>${icon("event-grid")}<span>${escapeHtml(event.name)}</span>
        </button>
        ${expanded && event.projects.length ? `<div class="project-tree-list">
          ${event.projects.map((project) => `
            <button class="project-tree-row ${project.id === selectedProjectId ? "selected-project" : ""}" data-action="select-project" data-project-id="${project.id}" title="${escapeHtml(project.name)}">
              ${project.proposal ? icon("sparkle") : icon("project")}
              <span>${escapeHtml(project.name)}</span>
            </button>
          `).join("")}
        </div>` : ""}
      </section>
    `;
    }).join("")}
  `;
}
function sourceShelfTemplate(_source) {
    return `<div class="source-shelf">
    <div class="source-shelf-row">${icon("text")}<span>Titles</span><small>Basic Title and captions</small></div>
    <div class="source-shelf-row">${icon("media")}<span>Generators</span><small>Custom Solid</small></div>
  </div>`;
}
