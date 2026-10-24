# stl-to-drawing-and-step

A Claude Code skill that turns an STL mesh (typically a 3D-print file) into:

- `<part>_drawing.pdf`: one multipage vector PDF (A3 landscape) of dimensioned views, sections, feature tables and a support table. It mirrors the whole STL, including print supports, which are labelled.
- `<part>_clean.step`: a B-rep solid that re-imports in a CAD program. It omits only what is unambiguously a custom print-support addition.

Both files, plus the original STL, are copied into a `step/` folder next to the part.

Dimensional accuracy comes first. The STEP is built by CSG (extrusions, cylinders, lofts, fillets) from measurements taken on the mesh. The mesh is never repaired, sewn or boolean-ed, because print supports make it non-manifold.

## Contents

| Path | Purpose |
|---|---|
| `SKILL.md` | The skill: workflow, tools per task, pass criteria |
| `scripts/verify_step.py` | Size test: compares STL and STEP section outlines on many planes (default tolerance 0.05 mm) |
| `scripts/gpu_kernels.py` | Ray kernels: local thickness and hidden-line visibility. Backends: numba (default, all cores, float64), CUDA via torch (opt-in), numpy fallback |
| `scripts/supports_S5Box2F.json` | Example list of removed supports for the verifier |

## Install in Claude Code

Requirements: Claude Code CLI, Python 3.12, then:

```
pip install numpy scipy matplotlib shapely cadquery pymupdf numba
```

`numba` is optional (the ray kernels fall back to a slow numpy path without it). `torch` with CUDA is optional and only used if you pass `backend="cuda"`.

Install the skill by cloning this repo into your skills folder. The folder name must match the skill name.

```
# personal (all projects)
git clone https://github.com/jpb-github/stl-to-step-converter.git ~/.claude/skills/stl-to-drawing-and-step

# or per project
git clone https://github.com/jpb-github/stl-to-step-converter.git <project>/.claude/skills/stl-to-drawing-and-step
```

On Windows PowerShell, `~` is `$env:USERPROFILE`. Restart Claude Code (or start a new session) and check that `stl-to-drawing-and-step` is listed.

## Use

Put the STL in a folder and start `claude` there. Then ask, for example:

```
/stl-to-drawing-and-step "My Part.stl"
```

or just: "make dimensioned drawings and a STEP file from My Part.stl".

The skill works autonomously. Only files directly in that folder are used; sub-folders are ignored. Outputs:

```
My Part.stl            (input, untouched)
step/My Part.stl       copy of the input
step/My Part_clean.step
step/My Part_drawing.pdf
```

To re-run the size test by hand:

```
python scripts/verify_step.py "My Part.stl" "My Part_clean.step" --tol 0.05 --supports supports.json --out fail_overlays [--planes "x=15.9,y=222"] [--jobs 6]
```

Exit code 0 = all planes within tolerance, 1 = a plane fails (see overlays), 2 = input error. `supports.json` format is in the script header.

## Known limits and caveats

- **Accuracy:** the target is 0.05 mm on every feature except removed supports. It is checked on sampled planes only (8 per axis plus the ones you add), so a feature that no plane cuts is not tested. Add `--planes` for small features.
- **First real test (S5 Box 2F) did not pass 0.05 mm.** The shipped STEP for that part has known deviations of up to 0.7 mm (missing 1 mm chamfers on the back ears, ear-slot detents, a latch wedge, front-slot tabs). Its PDF verification note still quotes the earlier, looser check. Treat the S5 outputs as unfinished.
- **Faceted arcs:** STL arcs are polygons. Deviation equal to the facet sag is excused and must be noted, nothing else.
- **Support detection is a judgement call.** Only clear cases (0.2 mm hole skins, floating strips with clearance, thin bridging layers) are omitted. Anything doubtful stays in the STEP and is listed in the notes. Review that list.
- **Prismatic parts fit best.** The CSG approach suits extrusions, holes, slots, chamfers and simple lofts. Organic or heavily sculpted shapes are not suited; there is no mesh-to-NURBS fitting.
- **No mesh repair.** Watertight checks only report problems.
- **Units and orientation:** mm assumed. A part rotated about Z is aligned first; the origin is the bounding-box minimum corner.
- **Time and effort:** a part like the S5 box took many hand-built measurement scripts. The skill is a guide for the agent, not a one-command converter. The verifier alone takes about 40 s.
- **Hardware/OS:** developed on Windows with CadQuery (OpenCascade). Large probes (nearest-point clouds, trimesh `closest_point`) can use tens of GB of memory and are explicitly avoided.
- **The `step/` folder is overwritten** on each run for same-named files.

## Performance

Measured on a 6-core CPU (12 threads) and an RTX 4070, on a 73,724-triangle part:

| Task | Before | After |
|---|---|---|
| Local thickness, whole mesh | about 15 min (numpy, extrapolated from 2000 rays) | 5 to 7 s (numba); 4.5 s (CUDA) |
| Hidden-line visibility, 100k points | not measured | 2.5 s (numba); 6 s (CUDA) |
| `verify_step.py`, 24 planes, STEP of 1.9M triangles | 152 s | 61 s with 6 jobs |

CUDA is not the default: it is no faster than numba here and float32 gave a wrong result on 2 of 73,723 grazing rays (numba matches the float64 numpy reference exactly). Verifier output is identical except p99 values in the third decimal.
