# Script input inventory after the Data migration

All runtime input locations below are relative to the repository root. `Data` has a capital D. Exact filenames, SHA-256 values, source identifiers, byte sizes, distribution choices, and consuming scripts are recorded in [DATA_SOURCE_MANIFEST.json](DATA_SOURCE_MANIFEST.json). Machine-specific original source mappings and the pre-migration inventory were preserved in the separate local validation folder.

## Per-script inputs

| Script | Required inputs | Notes |
| --- | --- | --- |
| `Figure 2.3.py` | `Data/plate_reader/2025_01_Plate_Reader_Results.xlsx` | Reads the original alpha-2/alpha-4 sheets and positional mean/SD ranges. Calibration and other workbook sheets remain intact. |
| `Figure 3.3.py` | `Data/thawing/Thawing_time_dependency_delay_quantification.xlsx` | Retains `Sheet1`, AA8:AC12. |
| `Figure 3.8 Elias.py` | `Data/elias_competition/*.txt` | All 46 original files copied; unchanged selection uses 28 files. Sorted discovery, strategy mapping, and day filtering preserved. |
| `Figure 4.1.py` | `Data/spheroid_sinking/Spheroid Height DropSeeding.xlsx` and `Data/spheroid_sinking/intensity_sum2d_per_channel_per_scene.csv` | Retains `Tabelle2`, scene E4, channels C0/C1, and normalization. |
| `Figure 4.1_analysis.py` | `Data/microscopy/figure_4_1/*.czi` | Seven actual Zeiss CZI/MIP files, required to reproduce extraction; no TIFF substitutions. Unchanged sorted discovery, scenes, CYX reads, channel fallback, time parsing, and sums. |
| `Figure 4.2.py` | `Data/embedded_spheroids/ME_A2-4 Data.xlsx` | Retains Data/Treatment sheets and all correction/normalization settings. |
| `Figure 4.8 Low Adhesion KM.py` | `Data/low_adhesion/ME_A2-3 Data.xlsx` | Plate 1/2 and treatment sheets; KM and treatment fractions unchanged. |
| `Figure 4.9.py` | Same shared ME_A2-3 workbook | Treatment-effect inputs and clustered/paired analyses unchanged. |
| `Figure 4.10 Global Population Competition.py` | Same shared ME_A2-3 workbook | Population competition inputs, GEE, and KDE figures unchanged. |
| `Figure 4.11 Spheroid Disruption.py` | Same shared ME_A2-3 workbook | Includes both Treatment_fallapart sheets, which were retained. |
| `Figure 4.13 Isabel 1.py` | `Data/isabel/Data_main_all_wells.xlsx` | Meta and the three original B-prefixed normalized metric sheets, well mapping, and threshold retained. |
| `Figure 4.16 Isabel 2.py` | Original arrays in the script: 12 AT and 8 CT scores | No external data file; no score changes. Upstream scoring provenance remains undocumented. |
| `Figure 5.5 S100.py` | `Data/s100/S100_Analysis.xlsx` | Tabelle2, units, three-day lag, censoring, and statistics retained. |
| `output/logrank_audit/verify_logrank.py` | `Data/audit/` and `Data/audit_support/` | 72 experimental HDF5s, 107 simulation HDF5s, four input CSVs, a thesis PDF, and eight immutable code-support files. Original strategy/run selection and statistical-function extraction retained. |
| `experiment_filters.py` | Embedded exclusion/truncation rules | Unchanged; imported by Figure 4.1, which does not apply exclusions to its scene-level table. The audit uses its byte-preserved original filter module under Data/audit_support. |
| `figtools.py` | `Data/config/figstyle.mplstyle` | Style resolves relative to the helper, including when the working directory differs. No plotting setting changes. |
| `experiment_info_isabel.py` | Expected root `Data/isabel/August_experiment_1x/` | Original upstream inputs unavailable. No figure script consumes this configuration. Numerical settings and channel mapping retained; paths rebased without fallback. |
| `data_paths.py` | Repository/script location | Shared Data/output locations; supplies the preserved audit module's input-path interface. |
| `verify_data.py` | `DATA_SOURCE_MANIFEST.json` and every listed input | Checks actual file sizes, SHA-256, and capitalization. This is an integrity utility, not an analysis. |

## Copied data and distribution

- **257 original source mappings** were checked and resolve to **254 distinct destination files**. All copied files were checked by exact byte length and SHA-256 against the original sources.
- **54 figure-table/TXT inputs:** 2,141,219 bytes. The four ME_A2-3 workbooks were byte-identical and now share one complete copy; distinct replicate filenames were not deduplicated.
- **One plotting configuration file:** 1,797 bytes, copied to Data/config without changing any setting.
- **Seven microscopy assets:** 2,090,635,552 bytes, distributed separately as GitHub Release attachments. Their complete paths, checksums, and sizes are in [Data/RELEASE_ASSETS.json](Data/RELEASE_ASSETS.json). Each is below 2 GiB.
- **184 audit data inputs:** 2,334,804,218 bytes. Every individual audit input is below 100 MiB.
- **Eight byte-preserved audit code inputs:** 102,647 bytes. These avoid dependence on a sibling checkout while retaining the imported analysis and selected statistical-function source code.
- All **247 non-microscopy input files** total **2,337,049,881 bytes** and are eligible for normal Git tracking. No other input exceeds 100 MiB. No large data file was already tracked by Git when the migration began.

Original datasets were copied, never moved. No image or other input content was transformed. The copied workbooks retain all sheets and cached formula values. Original provenance strings inside CSVs remain unchanged and are not used as fallback input paths.

## Generated outputs and missing upstream inputs

Generated data and figures go to `output/`, separately from Data. In particular, microscopy extraction writes `output/data/figure_4_1/intensity_sum2d_per_channel_per_scene.csv`, preserving the reference CSV consumed by Figure 4.1.

The unavailable upstream Isabel configuration expects image/time-stack/video directories, two metadata workbooks, three pickles, and five fluorescence/derivative workbooks below `Data/isabel/August_experiment_1x/`. Their sizes and contents remain unknown. The included `Data_main_all_wells.xlsx` is not silently substituted for those missing inputs.

The optional 40 adjacent plate-reader exports from the original inventory were not runtime inputs of any script and were not copied. Their source records remain in the local pre-migration inventory. No trained models or additional loaded calibration files were identified; the actual calibration data/settings are retained in the copied workbooks and unchanged code.

See [README.md](README.md) for download/placement instructions and [MIGRATION_VALIDATION.md](MIGRATION_VALIDATION.md) for the baseline comparisons, settings, blockers, and handoff.

The obsolete root figstyle.mplstyle was removed during the requested cleanup; its original bytes remain in the local baseline archive. The verified Data/config copy is the sole active plotting configuration.
