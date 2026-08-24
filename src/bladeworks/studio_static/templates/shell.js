/**
 * One-time workspace shell for Bladeworks Studio.
 *
 * Architecture map:
 * empty #app
 *   -> shellTemplate paints named panel slots and the preview <video>
 *   -> later panel templates fill those slots
 *   -> BladeworksEditorApp.bindEvents attaches one delegated listener
 *
 * Why this exists:
 * The shell is painted once at boot. Keeping it separate from per-panel HTML
 * means chrome edits do not recreate the WebRTC <video> node.
 *
 * Main callers: BladeworksEditorApp.start
 */
export function shellTemplate() {
    return `
    <div class="editor-app" id="editor-app">
      <header class="topbar" id="topbar"></header>
      <aside class="media-warning-banner" id="media-warning" hidden role="alert">
        <div class="media-warning-copy">
          <strong>Media files could not be read</strong>
          <p id="media-warning-detail"></p>
        </div>
        <button class="media-warning-dismiss" data-action="dismiss-media-warning" type="button">Dismiss</button>
      </aside>
      <main class="workspace-grid" id="workspace-grid">
        <nav class="library-panel panel" id="library-panel" aria-label="Libraries and projects">
          <div id="library-content"></div>
        </nav>
        <div class="panel-splitter splitter-library" data-resize="library" aria-hidden="true"></div>
        <section class="browser-panel panel" id="browser-panel" aria-label="Media browser">
          <div id="media-content"></div>
        </section>
        <div class="panel-splitter splitter-browser" data-resize="browser" aria-hidden="true"></div>
        <section class="viewer-panel panel" id="viewer-panel" aria-label="Video viewer">
          <div class="viewer-toolbar" id="viewer-toolbar"></div>
          <div class="viewer-wrap" id="viewer-wrap">
            <video id="preview-video" class="preview-video" playsinline hidden></video>
            <div class="viewer-warmup" id="viewer-warmup" hidden><span class="viewer-warmup-spinner"></span></div>
            <div class="mock-stage" id="mock-stage" aria-label="Preview loading">
              <div class="viewer-black-matte">
                <div class="program-frame" id="program-frame">
                  <div class="program-art" id="program-art">
                    <div class="title-preview" id="title-preview" hidden></div>
                  </div>
                  <div class="viewer-safe-guides" id="viewer-guides">
                    <div class="safe-guide action-safe"></div>
                    <div class="safe-guide title-safe"></div>
                    <div class="center-guide vertical"></div>
                    <div class="center-guide horizontal"></div>
                  </div>
                  <div class="canvas-controls" id="canvas-controls"></div>
                </div>
              </div>
              <div class="viewer-control-strip" id="viewer-control-strip"></div>
            </div>
            <div class="live-canvas-overlay" id="live-canvas-overlay">
              <div class="canvas-controls" id="live-canvas-controls"></div>
            </div>
            <div class="viewer-control-strip live-viewer-control-strip" id="live-viewer-control-strip"></div>
          </div>
          <div class="transport" id="transport"></div>
        </section>
        <div class="panel-splitter splitter-inspector" data-resize="inspector" aria-hidden="true"></div>
        <aside class="inspector-panel panel" id="inspector-panel" aria-label="Inspector">
          <div id="inspector-content"></div>
        </aside>
      </main>
      <div class="horizontal-splitter" data-resize="timeline" aria-hidden="true"></div>
      <section class="timeline-panel panel" id="timeline-panel" aria-label="Magnetic timeline">
        <div class="timeline-toolbar" id="timeline-toolbar"></div>
        <div class="timeline-body" id="timeline-body">
          <aside class="timeline-index" id="timeline-index" aria-label="Timeline index"></aside>
          <div class="timeline-scroller" id="timeline-scroller">
            <div class="timeline-content" id="timeline-content"></div>
          </div>
          <aside class="catalog-browser" id="catalog-browser" aria-label="Effects and transitions browser"></aside>
        </div>
      </section>
      <div class="toast" id="toast" role="status" aria-live="polite"></div>
      <div class="mock-notice" id="mock-notice" role="status" aria-live="polite">
        <strong id="mock-notice-title"></strong>
        <span id="mock-notice-detail"></span>
      </div>
      <div class="mock-modal-backdrop" id="mock-inventory" hidden>
        <section class="mock-modal" role="dialog" aria-modal="true" aria-labelledby="mock-inventory-title">
          <header><div><strong id="mock-inventory-title">Prototype mock inventory</strong><span>One registry · every intentionally simulated surface</span></div><button data-action="close-mock-inventory" aria-label="Close">×</button></header>
          <div id="mock-inventory-content"></div>
        </section>
      </div>
    </div>
  `;
}
