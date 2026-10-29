# Dissertation figures

The figure and analysis scripts read repository-relative inputs from **`Data/`** (capital D). Generated figures and tables are written under `output/`. Paths are resolved from the script location, so running a script does not require the repository to be the current working directory.

## Input layout

| Folder | Contents | Consumers |
| --- | --- | --- |
| `Data/config/` | Byte-preserved Matplotlib style configuration | All 12 figure scripts via figtools |
| `Data/plate_reader/` | Consolidated plate-reader workbook, including its calibration and summary sheets | Figure 2.3 |
| `Data/thawing/` | Thawing/passage measurement workbook | Figure 3.3 |
| `Data/elias_competition/` | All 46 original population-count TXT files; the existing selection rules use 28 | Figure 3.8 |
| `Data/spheroid_sinking/` | Spheroid-height workbook and the existing fluorescence-summary CSV | Figure 4.1 |
| `Data/embedded_spheroids/` | `ME_A2-4 Data.xlsx` | Figure 4.2 |
| `Data/low_adhesion/` | One complete `ME_A2-3 Data.xlsx`, shared by four scripts | Figures 4.8, 4.9, 4.10, 4.11 |
| `Data/isabel/` | `Data_main_all_wells.xlsx` | Figure 4.13 |
| `Data/s100/` | `S100_Analysis.xlsx` | Figure 5.5 |
| `Data/microscopy/figure_4_1/` | Seven original-format processed CZI/MIP files, obtained separately from release assets | `Figure 4.1_analysis.py` |
| `Data/audit/` | Experimental/simulation HDF5 inputs, event/escape CSVs, and the reference thesis PDF | `output/logrank_audit/verify_logrank.py` |
| `Data/audit_support/` | Byte-preserved helper modules and statistical-function source files used by the audit | Same audit utility |

Figure 4.16 retains its 20 original scores in its script. Exclusion rules, well mappings, channel assignments, background corrections, and calibration factors remain in their existing scripts/helpers or input workbooks. The plotting style is copied unchanged to `Data/config/figstyle.mplstyle`. No trained-model dependency was found.

The figure tables and TXT files total **2,141,219 bytes (2.14 MB)** after deduplicating the four identical ME_A2-3 workbooks. The audit utility additionally requires about **2.33 GB** of ordinary input files. Including its helper-code inputs, the **247 input files intended for normal Git tracking total 2,337,049,881 bytes**. Every one is below 100 MiB. These audit inputs are included for completeness because the existing audit script consumes them; they are not used by the 12 figure scripts.

## Download the microscopy release assets

**Release URL: TODO — insert the GitHub Release URL after manual publication.**

Download each asset listed in [Data/RELEASE_ASSETS.json](Data/RELEASE_ASSETS.json) and place it directly in `Data/microscopy/figure_4_1/`, retaining its filename, capitalization, extension, and bytes:

| Required asset | Exact bytes |
| --- | ---: |
| `2024_04_04 d0 Doubledropdrugdilution-01_processed_MIPs.czi` | 246,798,848 |
| `2024_04_05 d1 Doubledropdrugdilution-01_processed_MIPs.czi` | 246,763,328 |
| `2024_04_06 d2 Doublelayerdropdrugdilution-01_processed_MIPs.czi` | 60,933,120 |
| `2024_04_08 d4 Doublelayerdropdrugdilution-01_processed_MIPs.czi` | 60,959,296 |
| `2024_04_10 d6 Doubledropdrugdilution-01_processed_MIPs.czi` | 246,772,480 |
| `2024_04_15 d11 Doubledropdrugdilution-01_processed_MIPs.czi` | 614,171,232 |
| `2024_04_22 d18 Doubledropdrugdilution-01_processed_MIPs.czi` | 614,237,248 |

The seven assets total **2,090,635,552 bytes (1.947 GiB)**. Each is below 2 GiB. Their formats are **Zeiss CZI**, not TIFF. They are already maximum-intensity projections in the source dataset; this migration did not convert, recompress, rescale, project, or otherwise alter them. The extraction retains `AICSImage` scene handling, channel naming/fallback order, `CYX` reads, filename-derived days, and pixel-intensity summation.

Only `/Data/microscopy/` is ignored for large input distribution. The images will not be included automatically in a normal repository checkout or source archive. Ordinary processed datasets remain eligible for Git tracking. GitHub's [normal file limit](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github) is 100 MiB; its [release documentation](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases#storage-and-bandwidth-quotas) requires each release asset to be under 2 GiB.

After downloading all assets, check the inputs from any working directory:

```text
python /path/to/Dissertation_Figures/verify_data.py
```

This verifies capitalization, file sizes, and SHA-256 for all 254 required input files. Missing microscopy assets are reported as missing; there is no fallback to a laboratory drive.

## Run scripts

Use a compatible Python environment with NumPy, pandas, SciPy, Matplotlib, openpyxl, statsmodels, and lifelines. Microscopy extraction additionally uses `aicsimageio` and `aicspylibczi`; tqdm is optional. The audit additionally uses h5py and PyMuPDF. The tested versions and execution settings are recorded in [MIGRATION_VALIDATION.md](MIGRATION_VALIDATION.md). The previously borrowed `h5py 3.16.0` was installed into the project environment with `--no-deps`; no existing dependency was upgraded.

For example, from the repository root:

```text
python "Figure 4.1.py"
python "Figure 4.1_analysis.py"
```

From another directory, supply the full path to the script. The code locates `Data/`, including `Data/config/figstyle.mplstyle`, relative to the repository. Existing output filenames and plotting settings are retained, with outputs grouped under `output/figures/figure_*`, `output/data/figure_4_1/`, and `output/logrank_audit/`.

The extraction writes `output/data/figure_4_1/intensity_sum2d_per_channel_per_scene.csv`. Figure 4.1 continues to consume the preserved reference CSV under `Data/spheroid_sinking/`. Rerunning extraction therefore does **not** overwrite that input or silently replace the reference data. Original absolute paths stored inside copied CSVs are retained as provenance fields; the scripts do not follow those strings to load additional files.

The files in `Data/audit_support/figures/` are immutable source inputs: the audit extracts selected statistical functions from them. They are not additional figure entry points. Their original top-level path declarations are not executed. The imported audit-analysis module gets its migrated data locations from the repository's `data_paths.py`.

## Known missing upstream inputs

`experiment_info_isabel.py` also describes an unused upstream experiment. Its paths now resolve below `Data/isabel/August_experiment_1x/`, but its original image directories, metadata workbooks, pickles, and derivative workbooks were unavailable and have not been fabricated or substituted. This configuration is not used by Figure 4.13, whose consolidated workbook is included. Its calibration/background numbers and channel mapping are preserved. The upstream workflow described only by that configuration remains unverified.

The original acquisition stacks preceding the available CZI/MIPs, the workflow that consolidated some workbooks, and the scoring provenance of Figure 4.16 are not established by this migration.

See [DATA_SOURCE_INVENTORY.md](DATA_SOURCE_INVENTORY.md) for per-script dependencies, [DATA_SOURCE_MANIFEST.json](DATA_SOURCE_MANIFEST.json) for relative destinations and checksums, and [MIGRATION_VALIDATION.md](MIGRATION_VALIDATION.md) for baseline comparisons and limitations. Machine-specific source mappings and the complete execution evidence are kept in the separate local validation report, outside the repository.

The obsolete root `figstyle.mplstyle` was removed after verifying that it matched the active `Data/config/figstyle.mplstyle` and the local baseline archive. All figures use the Data/config copy. Data files are committed with Git text conversion disabled in `.gitattributes`, preserving the manifest checksums across checkout platforms.
