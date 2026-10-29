# Data migration validation and manual handoff

## Outcome

All **14 executable scientific entry points** completed with their real inputs before migration and after migration from both the project root and another working directory. All **28 before/after comparisons matched exactly** for the captured numerical results, processed tables, exported results, plotted data, and rendered figures. This establishes equivalence for these executed workflows under the settings below; it does not validate the unavailable upstream workflow described only by the unused Isabel configuration.

The initial migration was left unstaged for review. At the user's subsequent request, obsolete text artifacts were removed and the required code, inputs, and documentation were staged for manual commit. No commit, push, tag, release, visibility change, or history rewrite was performed.

## Baseline protection and comparison method

- Baselines were captured before any migration path edits. Original source scripts, the prior inventory, Git status/diff, execution logs, loaded package versions, numerical snapshots, plotted artist data, and exported files were preserved outside the repository.
- Validation executed the actual complete scripts with Agg and a write-isolation harness. Python file writes, figure exports, pandas CSV writes, directory creation, and native PyMuPDF image saves were routed to a separate validation directory. Other filesystem mutations outside validation were blocked. The analysis CSV therefore never overwrote its original reference input.
- **95 pre-existing input/output files** retained their original sizes and SHA-256 during the migration validation. The intentionally migrated audit script was excluded from that check. Subsequent user-requested cleanup archived scientific outputs with verified checksums before removing obsolete text outputs from the repository; original external datasets remain unchanged.
- Tables were compared with exact pandas checks, including values, shape, dtype, row/column/index ordering, identifiers, and missing values. Captured numerical arrays and structured results used exact equality. No numerical tolerance was introduced for the migration comparison.
- Only known input/output location strings were normalized for the migration. Provenance columns were otherwise retained. Path declarations were excluded from scientific snapshots, and their actual input destinations were checked separately.
- PNGs were compared as exact RGBA pixels. Generated PDFs had identical page dimensions and exact rendered RGBA pixels at **144 dpi**; irrelevant PDF creation metadata was ignored. Plotted coordinates, limits, positions, labels, colors, patches, and collections were also compared.
- Reported statistical console lines matched exactly, after removing progress/timing/source-location diagnostics and normalizing whitespace. The full original logs remain available locally. Existing code's own reference-test tolerances were preserved; in particular, the audit still performs its original 42 lifelines comparisons.
- The relocated native readers were observed opening **all 7 CZI files** and **all 179 selected HDF5 files** from Data in both working directories. Recorded tabular reads were also inside Data. No fallback to original data locations was added.
- AST comparisons of the **17 original scripts/helpers**, excluding only location declarations/imports and style-location resolution, confirmed unchanged scientific code and parameters. The plotting style file and audit support code/data were preserved byte for byte.

## Per-script results

In the table, “both” means project-root and other-working-directory runs. Snapshot counts describe captured function/module calculations and tables; artifact comparisons include rendered appearance as applicable.

| Script | Baseline | Migrated (both) | Comparison method | Outcome |
| --- | --- | --- | --- | --- |
| `Figure 2.3.py` | Pass | Pass | 3 calculation/table snapshots; 0 statistic log lines; 2 artifacts; plotted data + exact PNG/PDF rendering | Exact match |
| `Figure 3.3.py` | Pass | Pass | 1 calculation/table snapshots; 0 statistic log lines; 2 artifacts; plotted data + exact PNG/PDF rendering | Exact match |
| `Figure 3.8 Elias.py` | Pass | Pass | 43 calculation/table snapshots; 16 statistic log lines; 2 artifacts; plotted data + exact PNG/PDF rendering | Exact match |
| `Figure 4.1.py` | Pass | Pass | 1 calculation/table snapshots; 1 statistic log lines; 2 artifacts; plotted data + exact PNG/PDF rendering | Exact match |
| `Figure 4.1_analysis.py` | Pass | Pass | 9 calculation/table snapshots; 0 statistic log lines; 1 artifacts; CZI metadata and ordered channel/scene CSV | Exact match |
| `Figure 4.2.py` | Pass | Pass | 3 calculation/table snapshots; 3 statistic log lines; 5 artifacts; plotted data + exact PNG/PDF rendering | Exact match |
| `Figure 4.8 Low Adhesion KM.py` | Pass | Pass | 10 calculation/table snapshots; 37 statistic log lines; 2 artifacts; plotted data + exact PNG/PDF rendering | Exact match |
| `Figure 4.9.py` | Pass | Pass | 7 calculation/table snapshots; 4 statistic log lines; 3 artifacts; plotted data + exact PNG/PDF rendering | Exact match |
| `Figure 4.10 Global Population Competition.py` | Pass | Pass | 8 calculation/table snapshots; 4 statistic log lines; 5 artifacts; plotted data + exact PNG/PDF rendering | Exact match |
| `Figure 4.11 Spheroid Disruption.py` | Pass | Pass | 13 calculation/table snapshots; 4 statistic log lines; 3 artifacts; plotted data + exact PNG/PDF rendering | Exact match |
| `Figure 4.13 Isabel 1.py` | Pass | Pass | 5 calculation/table snapshots; 1 statistic log lines; 2 artifacts; plotted data + exact PNG/PDF rendering | Exact match |
| `Figure 4.16 Isabel 2.py` | Pass | Pass | 1 calculation/table snapshots; 1 statistic log lines; 2 artifacts; plotted data + exact PNG/PDF rendering | Exact match |
| `Figure 5.5 S100.py` | Pass | Pass | 2 calculation/table snapshots; 1 statistic log lines; 2 artifacts; plotted data + exact PNG/PDF rendering | Exact match |
| `output/logrank_audit/verify_logrank.py` | Pass | Pass | 1 calculation/table snapshots; 43 statistic log lines; 9 artifacts; exact CSVs, rendered reference images, text | Exact match |

| Helper / utility | Validation | Limit |
| --- | --- | --- |
| `figtools.py` | Exercised through all 12 figure scripts in every phase; exact plotted data and rendering | No plotting setting changes |
| `experiment_filters.py` | Original module unchanged; imported by Figure 4.1. The audit's original separate filter module was byte-preserved and exercised through the audit | Figure 4.1 does not apply exclusions to its scene-level data |
| `experiment_info_isabel.py` | Returned keys, all non-path values, calibration/background numbers, channel map, and day labels exactly preserved; paths now point below Data | Missing original upstream experiment inputs; no full workflow comparison possible |
| `data_paths.py` | Exercised through relocated workflows from both working directories | New helper; no pre-migration baseline |
| `verify_data.py` | Verified all 254 required files, including capitalization, size, SHA-256, from both working directories | New utility; no scientific baseline |

## Environment and execution settings

Python **3.12.7**, NumPy **2.3.4**, pandas **2.3.3**, SciPy **1.16.3**, Matplotlib **3.10.7**, statsmodels **0.14.6**, openpyxl **3.1.5**, lifelines **0.30.0**, aicsimageio **4.14.0**, aicspylibczi **3.3.1**, h5py **3.16.0**, PyMuPDF **1.28.2**, and tqdm **4.67.1** were used.

Matched settings: Agg backend; `PYTHONHASHSEED=0`; NumPy legacy RNG and Python `random` seeded to **1729**; OpenMP/OpenBLAS/MKL/NumExpr threads set to **1**. Existing explicit RNG seeds in the scripts were retained. No analysis parameters were overridden.

The original audit loaded h5py 3.16.0 from a sibling environment. The same missing version was installed into the configured project environment with `python -m pip install --no-deps h5py==3.16.0`. **92 h5py package source/binary files were verified identical** to the previously used installation. No existing dependency was upgraded. Both final audit runs loaded h5py from the project environment, and no sibling data/code checkout was needed by the relocated scripts.

## Copy integrity and microscopy metadata

All **257 source-to-destination mappings** passed file-length and SHA-256 checks, producing **254 distinct input files**. Four identical ME_A2-3 workbook sources share one destination. No differing file was silently overwritten or merged. Original files were copied, not moved.

The seven microscopy files have actual **ZISRAWFILE / Zeiss CZI** signatures. The extracted CZI scene lists, dimension orders/shapes, channel metadata, physical pixel sizes, and computed channel/scene intensity results matched the baseline exactly. Their complete source bytes, including metadata, were preserved; no TIFF conversion or image transformation occurred. Their metadata snapshots are included in the local evidence.

- Normal Git inputs: **247 files, 2,337,049,881 bytes**.
- Ignored release images: **7 files, 2,090,635,552 bytes**; each below **2 GiB**.
- Other inputs above **100 MiB**: **none**. The largest ordinary HDF5 input is 37,626,096 bytes.
- Large data already tracked by Git at the beginning: **none**. The images remain untracked and ignored.

Checksums and consuming scripts: [DATA_SOURCE_MANIFEST.json](DATA_SOURCE_MANIFEST.json). Release asset names, exact sizes, SHA-256, and local destinations: [Data/RELEASE_ASSETS.json](Data/RELEASE_ASSETS.json).

## Remaining limits and recorded failures

- All 14 baseline entry points passed; no existing execution failure was observed for those workflows.
- One migrated verification attempt failed because the isolation harness redirected file opening after pandas had already checked a new CSV output directory. The attempt was preserved locally, the harness was corrected, and the complete extraction was rerun successfully. No scientific code was changed to accommodate the harness.
- The unused Isabel configuration's original images, metadata files, pickles, and derivative workbooks were unavailable. They remain explicitly missing under `Data/isabel/August_experiment_1x/`; nothing was guessed or substituted. Its full upstream workflow is **not validated**.
- Acquisition stacks preceding the available MIPs, the earlier workbook-consolidation workflows, and Figure 4.16's scoring provenance remain outside the evidence available in this repository.
- Execution/rendering was checked on the current Windows/Python environment. Actual filename capitalization was verified explicitly, and both working directories were tested; Linux/macOS runtime environments were not exercised.
- Normal interactive GUI display behavior was not tested. Scientific results and exported appearance were compared with the same headless backend in all phases.

## Manual commit and release handoff

The required code, documentation, relative manifest, release metadata, integrity utility, and all 247 ordinary input files are staged for manual review and commit. Review the staged changes in Git or the IDE. The temporary MIGRATION_COMMIT_FILES.txt checklist was removed once its reviewed path list was saved in the local validation evidence. Computer-specific IDE settings and the seven microscopy assets are excluded.

Intentionally ignored:

- `/Data/microscopy/` — original-format images to distribute as manual release assets.
- `/output/figures/` and `/output/data/` — generated figures and analysis tables.
- `/output/logrank_audit/*`, except `verify_logrank.py` — existing/generated audit evidence.

All ordinary input files were checked to remain eligible for Git tracking. The necessary files were subsequently staged at the user's explicit request. The release URL is an explicit TODO in README.md and Data/RELEASE_ASSETS.json; publication and asset upload remain manual.

The obsolete root figstyle.mplstyle was removed after confirming that it matched Data/config/figstyle.mplstyle and the local baseline archive. No redundant copies of the four shared workbook sources remain under Data. Distinct replicate filenames and immutable audit source files were deliberately retained. All 19 old generated output files were copied to the local validation cleanup archive and verified by SHA-256. The 14 text outputs were removed using explicit editor deletions. Automatic approval review blocked shell deletion of the remaining five binary outputs and 14 bytecode caches with the stated reason "blocked by policy"; they remain ignored. They are not input data or normal commit candidates. The optional 40 unused plate-reader exports were not copied.

The complete local validation evidence, including original absolute source mappings, is outside this repository in the sibling `Dissertation_Figures_validation` directory, run `migration_20261029_093807`. It includes the before snapshot, source_mappings.json, copy_summary.json, baseline/migrated logs and artifacts, numerical snapshots, plotted-data snapshots, comparison_results.json, console_comparison.json, structural_integrity_checks.json, and the protected runner/check scripts. Machine-specific source locations are intentionally absent from these public-facing documents.

The cleanup also removed the two obsolete IDE migration-marker files. Local interpreter/IDE settings and the active virtual environment remain ignored because they are computer-specific. Python cache patterns are ignored to prevent regenerated caches appearing as untracked files. Data/** is marked -text in .gitattributes so Git preserves the exact input bytes recorded in the manifest.
