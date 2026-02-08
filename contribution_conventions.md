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
│   ├── notebooks/          # Generated .ipynb files (from pipeline)
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
- **jupytext** - Convert between .py and .ipynb formats
- **Jupyter Book** - Build the book from notebooks

## Pipeline Conventions

### Two Types of Pipeline Scripts

1. **Pure data scripts** (no charts/visualizations)
   - Simple `.py` files in `pipeline/`
   - Should use jupytext cell markers when it helps for interactive debugging
   - Not converted to notebooks
   - Example: `01_download_smard_DE_prices.py`

2. **Analysis scripts** (with charts/visualizations)
   - `.py` files with jupytext `# %%` cell markers
   - Get converted to `.ipynb` and executed by DVC
   - Final notebooks stored in `book/notebooks/`
   - Example: `03_smard_DE_prices.py`

### Jupytext Format

Analysis files use the percent format with cell markers:

```python
# %% [markdown]
# # Title
# Description text

# %%
import pandas as pd

# %% [markdown]
# ## Section Header

# %%
# Code cell
df = pd.read_parquet(paths.smard_prices_file)
```

### DVC Stages

- Download stages: run pure Python scripts, output raw data
- Process stages: convert `.py` → `.ipynb`, execute notebook, output to `book/notebooks/`

Example from `dvc.yaml`:
```yaml
process_smard_DE_prices:
  cmd: >
    uv run jupytext --to notebook --output book/notebooks/03_smard_DE_prices.ipynb pipeline/03_smard_DE_prices.py &&
    uv run jupyter execute book/notebooks/03_smard_DE_prices.ipynb --inplace
```

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
   - Use `cache: false` for notebooks that should be tracked in git

This ensures:
- All file paths remain centralized and consistent
- DVC tracks dependencies correctly and rebuilds only what's needed
- Scripts remain portable across different working directories

## Workflow

1. Create/edit pipeline script in `pipeline/`
2. Add DVC stage in `dvc.yaml`
3. Run pipeline: `dvc repro`
4. Generated notebooks appear in `book/notebooks/`
5. Build book: `uv run myst build`

## Git Conventions

- **Tracked**: Source `.py` files, generated `.ipynb` notebooks, book content
- **Not tracked**: Raw data files (managed by DVC), virtual environment
