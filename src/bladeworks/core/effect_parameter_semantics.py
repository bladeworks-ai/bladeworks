"""Reviewed static-semantics decisions for the cohort effect controls.

Architecture map
================

genuine Final Cut round-trip + per-setting A/B contact sheets
    -> one explicit decision per exact operation/parameter identity
    -> evidence generator writes the approved note into every setting finding
    -> inventory generator admits only those reviewed controls as executable

This module intentionally contains no guessed aliases or fuzzy name matching.
An executable identity must appear in ``APPROVED_STATIC_REVIEWS``.  Controls in
``BLOCKED_STATIC_REVIEWS`` retain their genuine serialization evidence but are
not presented as portable parameter support.

Why this exists
---------------

Static FCPXML transport and a runnable FFmpeg expression do not prove that a
slider means the same thing in both renderers.  Keeping the human-reviewed
decision in one small module prevents the evidence bundle and inventory from
drifting apart.

Main callers:
- ``effect_parameter_evidence`` when writing setting findings.
- ``effect_parameter_inventory`` when building ``effects.v1.json``.
"""

from __future__ import annotations


ParameterIdentity = tuple[str, str]


APPROVED_STATIC_REVIEWS: dict[ParameterIdentity, str] = {
    ("effect-vibrancy", "amount"): (
        "Approved for static use: both response sweeps move from reduced color "
        "intensity through the default toward stronger color intensity. The "
        "portable low/high brightness differs, so this remains a bounded approximation."
    ),
    ("effect-focus-blur", "amount"): (
        "Approved for static use: both renderers increase blur outside the focus "
        "region as Amount rises; the portable blur kernel is less diffuse."
    ),
    ("effect-focus-blur", "softness"): (
        "Approved for static use: both renderers broaden the transition between "
        "the sharp focus region and blurred exterior; their falloff profiles differ."
    ),
    ("effect-focus-blur", "emphasis"): (
        "Approved for static use: both renderers increase exterior blur strength "
        "as Emphasis rises; exact blur radius is approximate."
    ),
    ("effect-focus-blur", "width"): (
        "Approved for static use: both renderers expand the horizontal sharp "
        "focus region as Width rises; edge falloff is approximate."
    ),
    ("effect-focus-blur", "height"): (
        "Approved for static use: both renderers expand the vertical sharp focus "
        "region as Height rises; edge falloff is approximate."
    ),
    ("effect-radial-blur", "amount"): (
        "Approved for static use: both renderers increase center-out radial motion "
        "blur as Amount rises; the portable sampling kernel is visibly different."
    ),
    ("effect-crop-feather", "width"): (
        "Approved for static use: both renderers expand the retained horizontal "
        "alpha region as Width rises; their edge geometry differs slightly."
    ),
    ("effect-crop-feather", "height"): (
        "Approved for static use: both renderers expand the retained vertical "
        "alpha region as Height rises; their edge geometry differs slightly."
    ),
    ("effect-crop-feather", "feather"): (
        "Approved for static use: both renderers move from a broad alpha feather "
        "at value 0 toward a hard edge at value 1; ramp shape is approximate."
    ),
    ("effect-droplet", "intensity"): (
        "Approved for static use: both renderers increase circular ring distortion "
        "as Intensity rises; portable output uses a single bounded Gaussian ring."
    ),
    ("effect-earthquake", "amount"): (
        "Approved for static use: both renderers increase displacement and temporal "
        "echo as Amount rises; the deterministic portable shake phase differs."
    ),
    ("effect-camcorder", "amount"): (
        "Approved for static use: both renderers increase HUD visibility from absent "
        "at zero toward fully visible at one."
    ),
    ("effect-camcorder", "size"): (
        "Approved for static use: both renderers scale the camcorder HUD elements "
        "upward as Size rises; font and icon geometry are approximate."
    ),
    ("effect-camcorder", "battery-level"): (
        "Approved for static use: both renderers increase the battery icon fill as "
        "Battery Level rises; icon geometry is approximate."
    ),
    ("effect-camcorder", "recording"): (
        "Approved for static use: Final Cut's false-to-true response is localized "
        "to the REC glyph (98 of 14,400 pixels per frame at the >=8-code-value "
        "threshold, 0.6806%, within a 19x9 union bounding box) with 0.1936 normalized changed-region "
        "MAE. The full-frame response is only 0.001834 because the glyph occupies "
        "less than one percent of the frame. The earlier approximately 0.44 raw "
        "full-frame MAE was therefore dilution, not absence of Final Cut response. "
        "The false/true contact sheets visibly confirm absent/present REC graphics; "
        "the cited artifacts are effect-camcorder/recording/metrics.json plus "
        "effect-camcorder/recording/0/contact-sheet.png and "
        "effect-camcorder/recording/1/contact-sheet.png under the calibration cases."
    ),
    ("effect-cartoon", "amount"): (
        "Approved for static use: both renderers increase smoothing and tonal "
        "quantization as Amount rises; the portable stylization remains subtle."
    ),
}


BLOCKED_STATIC_REVIEWS: dict[ParameterIdentity, str] = {
    ("effect-vibrancy", "protect-skin"): (
        "Blocked: all static A/B settings were visually flat on the calibration "
        "source, so the portable red-balance approximation is not semantically "
        "established. Final Cut also rejects explicit interpolation and drops the "
        "same channel when authored with implicit interpolation."
    ),
    ("effect-drop-shadow", "opacity"): (
        "Blocked: the opaque full-frame calibration source completely occludes the "
        "shadow, so the five transported values do not establish visible semantics."
    ),
    ("effect-drop-shadow", "position"): (
        "Blocked: the opaque full-frame calibration source completely occludes the "
        "shadow, so the transported positions do not establish visible semantics."
    ),
    ("effect-drop-shadow", "blur"): (
        "Blocked: the opaque full-frame calibration source completely occludes the "
        "shadow, and only four distinct exported blur values survived."
    ),
    ("effect-drop-shadow", "blur-falloff"): (
        "Blocked: only the fixed exported value 1 was established; no alternative "
        "editor value or visible shadow response was captured."
    ),
    ("effect-drop-shadow", "perspective-amount"): (
        "Blocked: the opaque full-frame calibration source completely occludes the "
        "shadow, and only three distinct exported values survived."
    ),
    ("effect-drop-shadow", "presets"): (
        "Blocked: only the fixed Custom preset serialization was established; no "
        "alternative editor preset or parameter-specific response was captured."
    ),
    ("effect-fisheye", "amount"): (
        "Blocked: strong-value A/B frames show grossly different lens geometry and "
        "the sweep contains only four distinct exported values."
    ),
    ("effect-perspective-tile", "amount"): (
        "Blocked: strong-value A/B frames show grossly different tiling geometry; "
        "the portable mapping does not preserve Final Cut semantics."
    ),
    ("effect-perspective-tile", "mix"): (
        "Blocked: strong-value A/B frames show grossly different blend and tiling "
        "geometry; the portable mapping does not preserve Final Cut semantics."
    ),
}


def approved_static_note(operation_id: str, parameter_id: str) -> str | None:
    """Return the reviewed note only for an admitted static control."""

    return APPROVED_STATIC_REVIEWS.get((operation_id, parameter_id))
