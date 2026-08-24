/**
 * Browser-facing editor model.
 *
 * Architecture map:
 * canonical FCPXML in localhost runtime
 *   -> stable browser ProjectSnapshot
 *   -> pure EditOperation reducer
 *   -> accepted snapshot revision
 *
 * Product invariants:
 * - The primary storyline is contiguous after every accepted edit.
 * - Connected clips reference a primary-storyline anchor by stable ID.
 * - Project times use seconds in this UI prototype; the localhost adapter will
 *   translate exact FCPXML rational times at the boundary.
 * - Agent proposals are separate Projects and never overwrite their source.
 * - Inspector and onscreen controls share one dotted parameter-path ABI.
 */
export {};
