---
name: stl-to-drawing-and-step
description: Use when given an STL mesh (a 3D-print file) and asked for dimensioned drawings, a technical drawing set, a reverse-engineered CAD model, or a STEP file of the part.
---

# STL to Dimensioned PDF + Clean STEP

## 0. Scope (read first)
Work only on files directly in the folder the user names (or the current folder). **Ignore every sub-folder**: do not read, list, search or reuse anything inside them (earlier results, `step/`, test or archive folders). Never treat a file found in a sub-folder as input or as a reference answer.

## Deliverables (exactly two generated files, nothing else)
1. `<part>_drawing.pdf`: ONE multipage vector PDF containing every sheet.
2. `<part>_clean.step`: parametric-style B-rep solid (not a triangle mesh) that re-imports in any CAD program.

No PNGs, CSVs or intermediates next to the part (the final step creates the `step/` folder, see step 5). Keep scripts and scratch renders in the scratchpad. Work autonomously; do not stop to ask questions. Record assumptions in the PDF notes instead.

## Priority rule
**Dimensional accuracy of the original part beats everything else, including looks and design-intent tidiness.**
- Every dimension comes from the mesh, never from assumption or "nice" nominal values.
- Do not round or snap to nominals unless the deviation is smaller than the stated measurement accuracy. Say so in the notes when you do.
- Never invent geometry the mesh does not show.

## Tools by task
Environment: Windows, Python 3.12 (`pip install numpy scipy matplotlib shapely cadquery pymupdf`; trimesh optional). Claude Code tools: Bash to run scripts (use the scratchpad for scripts and scratch renders), Write for scripts, Read to LOOK at rendered PNGs, Artifact is not used.

| Task | Tool | Used for |
|---|---|---|
| 1 Load, sanity check | numpy | Parse binary/ASCII STL (`frombuffer` with a structured dtype), bbox, signed volume, edge-use counts (`np.unique` on sorted edges), face-normal histograms |
| 2 Sections | numpy | Plane-triangle intersection (vectorised), ray probes along an axis (Moller-Trumbore or barycentric test, list all crossings), local thickness (ray along inward normal, chunks of 400 triangles) |
| 2 Section outlines | numpy + shapely | Chain segments into loops, even-odd polygons, `simplify(0.03)` to print vertices, circle fits (least squares), ring area filters |
| 2 Connectivity | scipy.sparse.csgraph | `connected_components` on vertex/edge graphs to cluster thin triangles and find shells |
| 2 Look at geometry | matplotlib (Agg) + Read | `LineCollection` section plots with zoom windows and grid, saved to PNG and opened with Read; feature-edge line views |
| 3 Views | matplotlib | `PdfPages`, A3 figure 420 x 297 mm, one full-page axes in paper mm; helper functions for dimensions, leaders, centrelines, title block |
| 3 Hidden lines | numpy | Feature edges (dihedral > 30 deg) and a triangle z-buffer rasteriser |
| 3 Hatching | shapely | Polygon from section loops, parallel lines clipped to the polygon |
| 3 Check the PDF | pymupdf (`import pymupdf`) | Rasterise each PDF page to PNG at 75 dpi, then Read the PNG |
| 4 STEP | cadquery (OCP kernel) | `Workplane` boxes, `polyline().close().extrude()`, `loft`, `fillet`, `chamfer`, `union`, `cut`, `exporters.export` |
| 4 Verify STEP | `scripts/verify_step.py` | Size test below (numpy, scipy `cKDTree`, cadquery import and tessellate) |
| Avoid | trimesh `closest_point`, pymeshfix, OCC sewing of the STL, 1.5M-point nearest-neighbour clouds | Memory blow-ups, deleted bodies, invalid solids, or meaningless distances |

## Fidelity split
| Output | Content |
|---|---|
| PDF | **Complete mirror of the STL.** Every feature is measured and drawn, including thin webs, bridging layers, membranes and slits. Mark suspected print supports on the sheet as "print support (omitted from STEP)". |
| STEP | The part only. Omit what is **unambiguously** a custom support addition (e.g. sacrificial bridging layers 0.2 to 1 mm thick that seal a hole or span a void, with no functional purpose). If there is any doubt, keep it in the STEP and list it in the notes. |

## 1. Load and sanity-check the mesh
- Parse the STL with numpy (binary: 80-byte header, uint32 count, 50 bytes per triangle; fall back to ASCII). Use trimesh/numpy-stl only if they install easily.
- Report bounding box, size, units (assume mm and say so), triangle count, signed volume, and watertightness (edge-use counts).
- If the model is rotated about Z, check the dominant face normals and align it first. The model may sit far from the origin.
- Express all coordinates relative to the outer bounding-box minimum corner (X right, Y back, Z up). State this origin on the drawing.
- Also count zero-area triangles and edges used 3+ times. A non-manifold mesh is expected when supports are present. Note it and carry on; do not repair it (see step 4).

## 2. Understand the geometry before measuring
- Render cross-sections (plane-triangle intersection, matplotlib) at several Z levels and through X and Y planes. LOOK at them before deciding what the features are.
- Find flat-face levels (histogram of face-normal Z weighted by area) for floor, wall, rim and step heights.
- Measure thicknesses with ray probes along an axis (list all triangle crossings). Probe many positions: mid-wall, near corners, inside and outside pockets.
- Holes: chain section segments into closed loops, fit circles by least squares, report diameter, centre, pitch, count. Cluster centres to detect grids. Probe along the axis to classify through vs blind.
- Also measure chamfers, fillet/corner radii (fit from first-crossing offsets), slots (stadium vs rectangle), steps, ramps, angles.
- When a reading looks odd, zoom the section on the feature and extract the exact outline coordinates. Never describe a profile from coarse probes alone.
- Find thin features objectively: for every triangle, cast a ray along its inward normal and record the distance to the first hit (local thickness; vectorise in chunks, about 3 min for 40k triangles). Cluster triangles under 1 mm by vertex connectivity and list each cluster's bbox, thickness and area.
- Classify each thin cluster (0.2 to 1 mm) as **support** or **feature/uncertain**. Support = floating or comb-segmented plates inside a groove or slot with clearance on all sides, or a 0.2 mm skin closing a hole. Anything else stays uncertain. Measure all of them regardless; this classification drives only the STEP.
- State the measurement accuracy (driven by mesh tessellation and probe spacing).

## 3. PDF sheet set (A3 landscape, matplotlib, `PdfPages`)
Axes sized in paper mm at true scale so dimension text is a consistent size. One page per sheet, in this order:
1. **Overview**: title block, mesh sanity report (bbox, units, triangles, volume, watertight), origin definition, assumptions.
2. **Orthographic views** (third-angle; top, front, back, side as the part requires), all dimensioned: chained dimensions for hole positions and pitches, overall dims outside, hidden features dashed, centrelines on holes, view titles. Supports drawn and labelled.
3. **Section view** through the most informative plane, plus a zoomed detail (e.g. hole-grid corner) at larger scale.
4. **Special-profile sheet(s)** for any recess, groove, rail or undercut: 5:1 to 10:1, exact outline polyline extracted from the mesh, hatching, break lines, all profile dimensions (mouth, throat, depth, radii, flank angles), key plan showing the cut, profile table.
5. **Feature table**: grouped as overall size; inside cavity; wall/floor thicknesses; chamfers and radii; holes (diameter, count, pattern, pitch, first/last position, depth); slots/openings; pockets/recesses; **print supports** (separate group, with location and thickness). Label every interpretation (as opposed to measurement) as such. Include a column or note stating what is omitted from the STEP.

Title block on every sheet: part name, units, origin, scale, projection, date, "as-measured" revision, tolerance = measurement accuracy, sheet n of N.

Hidden lines: project mesh feature edges (dihedral > 30 deg), split into 0.6 mm pieces and test each against a numpy z-buffer of the triangles (about 6 s per view). Draw hidden pieces dashed and drop those shorter than 0.8 mm. Section views: hatch shapely polygons built from the chained section loops (even-odd). Place every leader's text in the view's data coordinates, not in absolute paper mm.

Then render the PDF pages to images (`pip install pymupdf`), LOOK at them, fix overlaps (labels over geometry, text over text, leaders crossing dimensions) and re-render until clean. Verify every number on the drawing against the measured data.

## 4. Clean STEP
Build the STEP as a parametric CSG model from the measurements in step 2. **Never repair, sew or boolean the mesh.** STL files with supports are routinely non-manifold (zero-area slivers where supports touch the part), and repairing them wastes hours. Sections, ray probes and loop fits already work on a broken mesh, so everything needed for the CSG comes from those.

- Use CadQuery (`pip install cadquery`; build123d as an alternative). If no CAD kernel installs, say so and deliver the PDF only.
- Same origin and orientation as the drawing. Omit only unambiguous supports (see Fidelity split).
- Recipe, in this order:
  1. **Main body**: extrude the outer section outline along the dominant prismatic axis (a U-shaped tray profile, for example). Take the outline from a section plane that misses holes and supports.
  2. **Regions that differ** (front or back end blocks, ears, notches): extrude their own section outlines over their own axis ranges, and union or cut them.
  3. **Grooves, dovetails, rails**: cut a prism of the measured profile (mouth, throat, depth, flank angles, lip radii) over the measured length. Draw the profile from a few polyline points, with arcs for radii. Do not trace the faceted mesh outline.
  4. **Slots, pockets, openings**: cut boxes, stadiums and prisms from measured positions.
  5. **Holes**: true cylinders at the fitted centres and diameter, through or blind as probed. Grids are one loop over the measured pitch.
  6. **Chamfers and fillets**: apply with the measured size, or build them into the profile.
- Get coordinates fast: chain section segments into loops, simplify with shapely (`simplify(0.03)`), print the vertices, and type the polygons straight into the CSG script. Mirror symmetric features about the part centre line instead of measuring both sides, then check the mirrored side in the overlay.
- Features that sit inside a pocket (bosses on a cavity wall) go in after the pocket cut, or the cut removes them. Sloped or wedge pockets: `loft` between two measured section polygons.
- Section outlines may be simplified to a few points per straight run or arc. No tracing hundreds of mesh vertices and no stair-step slabs.
- Skip features smaller than about 0.5 mm that are not holes or edge breaks. List them in the notes.
- Time-box: one focused pass per feature group. If a feature resists, leave it out of the STEP and list it under "not modelled" in the notes and the reply.

### Verify before delivering (all required, none need a repaired mesh)
Run the size test script. It does steps 1 to 5 below and exits non-zero on any failure:

```
python scripts/verify_step.py "<part>.stl" "<part>_clean.step" --tol 0.05 --supports supports.json --out <scratch>/fail [--planes "x=15.9,y=222"]
```

- `supports.json` lists the removed supports as boxes and cylinder sets (see the script header; example: `scripts/supports_S5Box2F.json`). Write it from the support table of step 2. Points inside these regions are ignored in both directions.
- Add `--planes` for every feature that the automatic planes (8 per axis at fixed irregular fractions) are unlikely to cut: small pockets, bosses, chamfers, thin walls.
- The script prints one line per plane (max STEP to STL, max STL to STEP, p99, and the 3D location of the worst point when it fails) and writes overlay PNGs (STL blue, STEP red, over-tolerance points green) for failing planes. Read them before changing the model.

Pass criteria (all required):
1. STEP re-imports; one solid; valid.
2. Bounding box within 0.05 mm of the STL bbox.
3. **Every feature within 0.05 mm of the STL on every tested plane, except the removed supports.** This is the size tolerance. A failing plane means a missing or mis-sized feature: fix the model, re-run, repeat until all planes pass.
4. Volume within about 1% of the STL signed volume once the omitted support volume is accounted for; report the difference.
5. If a failure is only the sag of a faceted STL arc (R x (1 - cos(pi / facets)), compute it), list it in the notes with the arc radius and facet count rather than distorting the model. Nothing else may be excused.

How the script avoids false failures: a failing plane is re-cut 0.31 mm away, because a plane lying exactly on a wall, chamfer end or cylinder axis gives false deviations of several mm; a real error persists, an artefact does not. It passes if either cut passes. It also never builds a nearest-point cloud on the whole surface.
Kill stray python processes after any timeout.

## 5. Publish to the `step` folder (last step, after verification passes)
Create `step/` inside the part's folder (reuse it if present; overwrite same-named files) and copy exactly three files into it:
- `<part>.stl` (the original, unmodified)
- `<part>_clean.step`
- `<part>_drawing.pdf`

Copy, do not move the STL. Nothing else goes in `step/`. Read the copies back (file sizes, STEP re-import) to confirm.

## 6. Reply
Short summary only: the paths of the three files in `step/`, mesh sanity numbers, what was omitted from the STEP and why, verification results (script output: planes tested, planes failing, worst deviation, bbox, volume), and the list of assumptions and uncertain interpretations.

## Common mistakes
- Putting supports in the drawings' omit list: the PDF must mirror the STL fully.
- Dropping an ambiguous thin feature from the STEP: keep it when in doubt.
- Rounding to nominal values that exceed measurement accuracy.
- Declaring the STEP good without running `scripts/verify_step.py`, or loosening its tolerance to make it pass.
- Hiding a real feature error inside the support mask: masks cover only the removed supports.
- Spending time on mesh repair, sewing or mesh booleans. The STEP comes from measurements only.
- Leaving PNGs or CSVs in the output folder, or using files from sub-folders (including a previous `step/`).
- Writing the drawings' dimensions before the STEP is verified: the overlay often exposes missed features (bosses, wedge pockets) whose dimensions then need adding to the PDF.
