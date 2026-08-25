---
name: bladeworks-fcpxml
description: Author, inspect, repair, or validate FCPXML intended for the Bladeworks renderer. Use for Bladeworks timeline structure, timing, geometry, audio, titles, effects, transitions, capability checks, and render compatibility; not for general Final Cut Pro UI questions.
---

# Bladeworks FCPXML

Use the bundled reference as the authority for FCPXML that must render through
the `bladeworks` CLI. Bladeworks implements a deliberate subset of Final Cut
Pro's FCPXML, so native FCPXML validity alone does not prove renderability.

## Required workflow

1. Read [references/FCPXML_BLADEWORKS_SPEC.md](references/FCPXML_BLADEWORKS_SPEC.md)
   completely before authoring or changing a document.
2. Read the focused reference pages for every domain the task touches:

   - document structure, formats, assets, and delivery: [CORE.md](references/FCPXML_BLADEWORKS/CORE.md)
   - timing, lanes, compounds, multicam, retiming, and transitions: [TIMELINE.md](references/FCPXML_BLADEWORKS/TIMELINE.md)
   - transforms, crop, opacity, blend, and compositing: [GEOMETRY.md](references/FCPXML_BLADEWORKS/GEOMETRY.md)
   - roles, gain, fades, panning, and routing: [AUDIO.md](references/FCPXML_BLADEWORKS/AUDIO.md)
   - titles, captions, and automatic runtime rasterization: [TITLES_AND_CAPTIONS.md](references/FCPXML_BLADEWORKS/TITLES_AND_CAPTIONS.md)
   - effects, masks, and transition parameters: [EFFECTS_AND_TRANSITIONS.md](references/FCPXML_BLADEWORKS/EFFECTS_AND_TRANSITIONS.md)
   - certified support status: [INVENTORY.md](references/FCPXML_BLADEWORKS/INVENTORY.md)
   - complete documents: [EXAMPLES.md](references/FCPXML_BLADEWORKS/EXAMPLES.md)

3. Check [references/FCPXML_RENDER_CAPABILITIES.yaml](references/FCPXML_RENDER_CAPABILITIES.yaml)
   when exact effect, transition, mask, title, alias, handler, or parameter
   support matters. This machine-readable registry wins over prose.
4. Preserve exact rational FCPXML timing. Do not replace rational times with
   floating-point approximations.
5. Validate grammar when `xmllint` is available. Resolve `SKILL_DIR` to the
   absolute directory containing this `SKILL.md`, and pass the project as an
   absolute path so validation does not depend on the caller's working
   directory:

   ```bash
   xmllint --noout \
     --dtdvalid "$SKILL_DIR/references/FCPXML_BLADEWORKS/FCPXMLv1_14.dtd" \
     "/absolute/path/to/project.fcpxml"
   ```

6. Validate Bladeworks compatibility separately:

   ```bash
   bladeworks inspect project.fcpxml --strict
   ```

7. Render a representative output when the task requires proof beyond static
   inspection. Report unsupported or approximated constructs explicitly; do
   not silently substitute different behavior.

## Boundaries

- Treat the reference bundle as Bladeworks-specific, not as a complete guide to
  everything Final Cut Pro accepts.
- Do not infer support from DTD validity. The DTD verifies XML grammar only.
- Do not invent effect identifiers or parameter names. Consult the inventory
  and capability registry.
- Keep media paths and project selection explicit when a document contains
  multiple projects or external assets.
