/**
 * Convert Final Cut transform coordinates into browser overlay geometry.
 *
 * Architecture map
 * ================
 *
 * FCPXML transform values
 *   -> convert project-height units into viewer percentages
 *   -> invert Final Cut's Y-up axis and counterclockwise rotation
 *   -> account for the anchor before positioning the source rectangle
 *   -> return CSS-ready geometry for the onscreen controls
 *
 * The renderer's authoritative transform kernel lives in
 * `src/bladeworks/core/geometry.py`. Keep this conversion equivalent to
 * its `transform_points` operation so the blue overlay describes the rendered
 * source rectangle instead of a second, approximate transform model.
 */
/**
 * Project the un-cropped project-sized source rectangle into viewer space.
 *
 * Final Cut expresses both X and Y in one percent of project height. Browser
 * percentages use element width for X and element height for Y, so X must be
 * multiplied by the project height-to-width ratio. Screen Y and CSS rotation
 * also have the opposite sign from Final Cut.
 *
 * Main callers:
 * - `BladeworksEditorApp.updateCanvasControls`, after every inspector or pointer
 *   transform change.
 */
export function transformOverlayGeometry(projectWidth, projectHeight, transform) {
    if (!(projectWidth > 0) || !(projectHeight > 0)) {
        throw new Error("Project dimensions must be positive for viewer geometry.");
    }
    const scaleX = transform.scale * transform.scaleX;
    const scaleY = transform.scale * transform.scaleY;
    const heightToWidth = projectHeight / projectWidth;
    const angle = (-transform.rotation * Math.PI) / 180;
    const cosine = Math.cos(angle);
    const sine = Math.sin(angle);
    // This is the renderer's C + T + R(-rotation) * S * (C - (C + A))
    // evaluated at the source rectangle's center and normalized to percentages.
    const anchoredX = -transform.anchorX * scaleX;
    const anchoredY = transform.anchorY * scaleY;
    const rotatedAnchorX = anchoredX * cosine - anchoredY * sine;
    const rotatedAnchorY = anchoredX * sine + anchoredY * cosine;
    return {
        widthPercent: 100 * scaleX,
        heightPercent: 100 * scaleY,
        centerXPercent: 50 + heightToWidth * (transform.x + rotatedAnchorX),
        centerYPercent: 50 - transform.y + rotatedAnchorY,
        rotationDegrees: -transform.rotation,
        anchorXPercent: 50 + transform.anchorX * heightToWidth,
        anchorYPercent: 50 - transform.anchorY,
    };
}
