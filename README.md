# ONEE Electricity Forecast

## Entry Points
- **Short-Term Forecasts**: `run_stf_cd.py`, `run_stf_srm.py`
- **Long-Term Forecasts**: `run_ltf_cd.py`, `run_ltf_srm.py`
- **Full Pipeline**: `run_full_cd.py`, `run_full_srm.py`

Configure forecasts via YAML files in `configs/` (e.g., `stf_cd.yaml`, `ltf_srm.yaml`).

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -e .
```

## Data
SQLite databases in `data/`:
- `all_data.db`: main database for forecasting

## Outputs
Results are written to `outputs/`:
- `ltf_cd_results.csv`, `ltf_srm_results.csv`
- `stf_cd_results.csv`, `stf_srm_results.csv`

## Package Structure
The `src/onee` package contains:
- `config/`: configuration classes
- `data/`: data loading and entity handling
- `short_term_forecast_strategies.py`, `long_term_forecast_strategies.py`: forecasting logic
- `utils.py`, `full_forecast_utils.py`: helper functions
