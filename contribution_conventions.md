# Contribution Conventions

This document describes the project structure and conventions for the World of Energy repository.

## Project Structure

```
world-of-energy/
├── woe/                    # Python package with shared utilities
│   ├── paths.py            # Centralized path configuration
│   └── smard/              # SMARD API client and config
├── pipeline/               # Data pipeline scripts
├── book/                   # Jupyter Book source
│   ├── notebooks/          # Generated .md files (from pipeline)
│   ├── markdown/           # Static markdown content
│   └── assets/             # Images and other assets
├── data/
│   ├── downloads/          # Raw downloaded data
│   └── processed/          # Processed/transformed data
├── output/                 # Generated outputs
│   ├── images/             # Chart images
│   └── reports/            # Report files
├── dvc.yaml                # DVC pipeline definition
└── pyproject.toml          # Project dependencies (uv)
```

## Tools

- **uv** - Package and environment management
- **DVC** - Data pipeline orchestration
- **jupytext** - Convert `.py` to MyST Markdown (`.md`) for the book
- **MyST** - Build the book from markdown and notebooks

## Pipeline Conventions

### Two Types of Pipeline Scripts

1. **Pure data scripts** (no charts/visualizations)
   - Simple `.py` files in `pipeline/`
   - Should use jupytext cell markers when it helps for interactive debugging
   - Not converted to notebooks
   - Example: `01_download_smard_DE_prices.py`

2. **Analysis scripts** (with charts/visualizations)
   - `.py` files with jupytext `# %%` cell markers and a jupytext/kernelspec header
   - Save all figures to `output/images/` using `fig.savefig()`
   - Include `{figure}` directives in markdown cells referencing the saved images
   - DVC runs the script, then converts `.py` → `.md` via jupytext
   - Generated `.md` files stored in `book/notebooks/`
   - Example: `03_smard_DE_prices.py`

### Jupytext Format

Analysis files use the percent format with a jupytext/kernelspec header:

```python
# ---
# jupytext:
#   text_representation:
#     format_name: percent
# kernelspec:
#   display_name: Python 3
#   language: python
#   name: python3
# ---

# %% [markdown]
# # Title
# Description text

# %%
import pandas as pd
```

### Images and Figure Directives

Analysis scripts save all figures to disk and reference them via MyST `{figure}`
directives. The scripts are **not executed** during the book build — images must
already exist on disk.

```python
# %%
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(...)
fig.tight_layout()
fig.savefig(paths.images_path / "03_my_chart.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ```{figure} ../../output/images/03_my_chart.png
# :name: fig-03-my-chart
# Caption for the figure
# ```
```

Naming convention: `<script_number>_<descriptive_name>.png` (e.g. `03_hourly_prices.png`).

The `{figure}` path is relative to `book/notebooks/` where the generated `.md` lives,
so images in `output/images/` are referenced as `../../output/images/<name>.png`.

### DVC Stages

- Download stages: run pure Python scripts, output raw data
- Process stages: run the script (saves images), convert `.py` → `.md` via jupytext, post-process the `.md`

Example from `dvc.yaml`:
```yaml
process_smard_DE_prices:
  cmd: >
    MPLBACKEND=Agg python pipeline/03_smard_DE_prices.py &&
    uv run jupytext --to md:myst --output book/notebooks/03_smard_DE_prices.md pipeline/03_smard_DE_prices.py &&
    sed -i -e 's/```{code-cell}/```{code-cell} python/g' -e '/root_level_metadata_filter/d' book/notebooks/03_smard_DE_prices.md
```

Notes:
- `MPLBACKEND=Agg` ensures `plt.show()` is a no-op in headless mode
- The `sed` step adds the `python` language to code cells and removes a jupytext metadata line that MyST doesn't recognize

## Path Conventions

All scripts must be **runnable from any working directory**. Use the centralized `ProjPaths` class from `woe/paths.py`:

```python
from woe.paths import ProjPaths

paths = ProjPaths()

# Use path properties instead of hardcoded strings
df = pd.read_parquet(paths.smard_prices_file)
fig.savefig(paths.images_path / "my_chart.png")
```

Key path properties:
- `paths.data_path` - Main data directory
- `paths.downloads_path` - Raw downloaded data
- `paths.processed_data_path` - Processed data
- `paths.output_path` - Generated outputs
- `paths.images_path` - Chart images
- `paths.pipeline_path` - Pipeline scripts

## Adding New Data Files

When introducing new data files or pipeline outputs, update these files:

1. **`woe/paths.py`** - Add a property for the new file path:
   ```python
   @property
   def my_new_data_file(self) -> Path:
       """Description of the data file."""
       return self.downloads_path / "my_data.parquet"
   ```

2. **`dvc.yaml`** - Register the file in the appropriate stage:
   - As `deps:` if the file is an input to a stage
   - As `outs:` if the file is produced by a stage
   - Use `persist: true` for downloaded data that shouldn't be deleted on `dvc repro`
   - Use `cache: false` for generated files that should be tracked in git (`.md` files, images)

This ensures:
- All file paths remain centralized and consistent
- DVC tracks dependencies correctly and rebuilds only what's needed
- Scripts remain portable across different working directories

## Workflow

1. Create/edit pipeline script in `pipeline/`
2. Add DVC stage in `dvc.yaml`
3. Run pipeline: `dvc repro`
4. Images appear in `output/images/`, generated `.md` files in `book/notebooks/`
5. Build book: `cd book && uv run myst build`

## Git Conventions

- **Tracked**: Source `.py` files, generated `.md` files, generated chart images, book content
- **Not tracked**: Raw data files (managed by DVC), virtual environment
