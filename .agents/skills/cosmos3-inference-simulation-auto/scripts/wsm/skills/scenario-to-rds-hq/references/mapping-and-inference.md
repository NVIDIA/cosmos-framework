# Mapping and inference decisions

Use this reference when the source vocabulary is richer, coarser, or incomplete.
The [RDS-HQ reference](../../../docs/rds-hq.md) defines exact target values; the
renderer determines which distinctions matter. These are reasoning criteria,
not a fixed source taxonomy or a list of universal numeric thresholds.

## Reduce classes without losing supported meaning

| Target component             | Useful evidence and mapping policy                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Objects                      | Include actors and other potentially moving objects. Passenger-car subtypes usually collapse to Car; heavy goods/passenger vehicles to Truck; walking people to Pedestrian; cycle/rider categories to Cyclist. Others includes protruding objects, strollers, and animals. Exclude standalone static props such as cones and construction objects; unknown labels do not automatically qualify for Others. Preserve stopped or parked actors in their semantic class. Resolve ambiguous vehicle labels with dimensions, shape, and source hierarchy rather than one noisy size threshold.                  |
| Lanelines                    | Preserve physical color, single/double, solid/dashed, and short/long dashed meanings using source class definitions and visible paint where available. A continuous sampled polyline describes the path and can represent dashed paint. Translate synonymous or richer styles into the nearest documented pair; keep a known color when style is uncertain. Write one string in each `colors` and `styles` enum list; split the line where either changes. Do not treat an unpainted lane centerline as a painted lane boundary. Use ordered geometry or lane topology to resolve asymmetric double lines. |
| Road boundaries              | Group physical barriers/fences, curbs, surface seams, guide posts, walls, and other limits into the six documented types. A road-to-grass transition is a seam; a raised lip is a curb. If the source establishes a physical boundary but its subtype remains unresolved, default to `curb` and record `method: defaulted`. Preserve stronger source or visual evidence for another type. Target group headings are not necessarily accepted input enum strings.                                                                                                                                           |
| Crosswalks, waitlines, poles | Infer component identity from explicit semantics and corroborating geometry/topology. A navigation stopping condition alone does not prove a painted waitline. Virtual stopping boundaries retain their implicit status. Pole geometry follows its upright axis.                                                                                                                                                                                                                                                                                                                                           |
| Signs                        | Map clear synonyms to the named classes, preserving one-way direction. Preserve recognized raw classes outside the named color categories when the renderer uses them for facing. Check coarse sign labels against RGB where available, especially low boxes around temporary construction objects. Establish whether the box describes a sign panel, its support, or another prop before mapping it. A low position alone is not grounds to delete or raise a real temporary sign.                                                                                                                        |
| Road markings                | Use the dedicated seven arrow classes, STOP, KEEP CLEAR, and text-paint class when supported by the source meaning. Additional arrow combinations remain generic rather than being silently changed into a different instruction. An arrow needs four enclosing corners in the documented order, oriented along lane travel; its class supplies the turn direction. Use the enclosing footprint rather than tracing the arrow outline. Exclude support masks and road-area polygons.                                                                                                                       |
| Signals                      | Combine observed color and direction or pedestrian indication into a supported state. `white_vru` is the white pedestrian go signal. Preserve simultaneous red-yellow where observed. Map richer indications to a meaningful basic color while retaining their state string. `unknown` means no reliable observation; `stateless` means an observed signal with no active indication. Keep the required basic-state map even though its field name contains `legacy`.                                                                                                                                      |

Source appearance may disambiguate a class, but ambient object color is not a
semantic class. Text recognition or image inspection is evidence only when the
feature is visible and identifiable in that frame. Record inferred semantics.

## Fill required attributes from the strongest available evidence

| Missing information           | Preferred derivation; fallback when evidence is insufficient                                                                                                                                                                                                                                                                           |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Object dimensions             | Measure an oriented box from source corners, mesh, or sufficiently complete points; otherwise use source class dimensions or a justified class prior. State whether the prior represents a measured object or an approximation. Reject invalid zero/negative dimensions.                                                               |
| Box center                    | Use the source origin definition and dimensions to move a bottom or axle origin to the geometric center in the object's local frame. Preserve the original heading convention during the offset.                                                                                                                                       |
| Object heading                | Prefer annotated orientation or geometry. Motion over several observations can support heading for a moving object; velocity alone is ambiguous for reversing vehicles or pedestrians. Keep heading stable for stationary tracks.                                                                                                      |
| Motion flag                   | Derive motion in the fixed world frame over the whole clip, accounting for annotation jitter and gaps. Sensor-relative motion includes ego motion and is unsuitable. A parked car that later pulls out needs `true` in every record, including the parked portion.                                                                     |
| Track ID                      | Associate consistent observations using time, location, dimensions, and class. Prevent reuse across incompatible objects. Do not join observations merely because the source ID is null or zero.                                                                                                                                       |
| Signal/sign thickness         | A small positive thickness can complete a planar annotation; record the value. Infer front normal from annotation orientation, approach direction, topology, or clear imagery. If facing cannot be resolved, disclose the ambiguity rather than pretending it was measured.                                                            |
| Map elevation                 | Derive from source world geometry, a local road surface, or nearby reliable ground points. A flat-ground assumption can serve a flat, ungraded scenario but must not flatten an overpass onto the road beneath it.                                                                                                                     |
| Semantic subtype              | First establish that the feature belongs in the target component. Use documented hierarchy and corroborating attributes, then a documented component prior or its exact Other/unknown representation. Unresolved physical road boundaries use the nominal `curb` default above. Preserve known color, geometry, or basic signal state. |
| Entire component              | Write its required empty-layer envelope and report unavailable source coverage. Infer geometry only when relationships or observations actually constrain it; do not create illustrative features just to populate a layer.                                                                                                            |
| Calibration or clock relation | Derive from metadata, known rig geometry, image transforms, synchronized events, or measured correspondences. Measure the residual. If no defensible relation exists, explain the specific missing fact and ask for it.                                                                                                                |

Choose priors per scenario rather than copying dimensions or thresholds from a
previous dataset. Distinguish a reversible representation choice (such as a
small sign thickness) from a missing fact that changes the entire projection.

## Record and check information loss

Keep a report next to the output, separate from the rendering tars:

- Source interval, source and target rates, selected views, transform chain,
  world origin, units, and image calibration transforms.
- Source-to-target class mapping with counts and evidence; unknown numeric
  values; inferred attributes and assumptions with their scope.
- Class distributions that need review, such as a whole component mapped to
  Other. Distinguish observed classes from nominal defaults.
- Counts before and after temporal selection, invalid-geometry filtering,
  static consolidation, object association, and class reduction.
- Empty or omitted components and why; unsupported source detail retained in
  the report; effects of collapsing a time-varying map into a static map.
- Structural checks, actual loader checks, projection residuals where available,
  render command/settings, and decoded/visually inspected output evidence.
  Keep these separate from the semantic findings in the RGB/render review.

When refinement reveals a reusable target constraint, update the format reference
or the general skill. Keep a source-specific workaround in that conversion's
adapter and report. Do not turn one example's schema or operating environment
into a requirement for future inputs.
