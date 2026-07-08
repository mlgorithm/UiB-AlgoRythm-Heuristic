# PACE 2026 solver description (UiB-AlgoRythm)

`solver-description.pdf` is the LIPIcs-formatted solver description for the PACE 2026
Heuristic Track submission. `solver-description.tex` is its source.

## Editing on Overleaf (recommended for the team)

The whole team can co-edit on Overleaf:

1. [overleaf.com](https://www.overleaf.com) → **New Project → Upload Project**
2. Select **`solver-description-overleaf.zip`** (in this folder) — it bundles the `.tex`,
   the LIPIcs class, the logo assets, and the `fontawesome5` shim, all at the root.
3. Overleaf auto-detects `solver-description.tex` as the main document. The compiler
   defaults to **pdfLaTeX** (works); if a font error ever appears, switch
   Menu → **Compiler → XeLaTeX**.
4. **Recompile** → the same 4-page PDF.
5. **Share** (top-right) → invite the co-authors by email, or enable "Anyone with the
   link can edit" and post the link in the team chat.

Keep `fontawesome5.sty` in the project so Overleaf renders exactly the committed PDF
(the author emails are already spelled out on the title page, so the envelope/ORCID
icons are not needed). Delete that one file only if you want LIPIcs' native icons.

**Premium alternative:** New Project → **Import from GitHub** → `mlgorithm/UiB-AlgoRythm-Heuristic`
syncs the `description/` folder directly.

## Rebuild locally (tectonic)

```bash
tectonic -X compile solver-description.tex
```

This folder is self-contained: it bundles `lipics-v2021.cls`, the logo assets
(`cc-by.pdf`, `lipics-logo-bw.pdf`, `orcid.pdf`), and a tiny local `fontawesome5.sty`
**stub**. The stub exists only because loading the real FontAwesome OTF aborts
tectonic's XeTeX engine; it blanks two decorative title-page glyphs but keeps the
genuine LIPIcs layout (all author emails are printed as text anyway).

To regenerate the Overleaf zip after editing locally, from this folder:

```bash
zip -j solver-description-overleaf.zip solver-description.tex \
  lipics-v2021.cls cc-by.pdf lipics-logo-bw.pdf orcid.pdf fontawesome5.sty
```

## Notes for submission

- **Contact email on the title page**: Sam Urmian, `sam.urmian@uib.no` (required by
  PACE); all seven author emails are shown.
- Length is **4 pages** including the title page and bibliography (the PACE limit is 4).
- `lipics-v2021.cls` is © Schloss Dagstuhl, redistributed here only to build the
  submission source; it is not part of the solver.
