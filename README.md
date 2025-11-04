# ONEE Electricity Forecast

## Key Entry Points
- `run_srm_strategies.py`: primary orchestration script; edit its configuration block to choose regions, forecast horizons, feature sets, and to trigger the different SRM analysis levels.
- `srm_strategies.py`: forecasting toolkit that loads data, assembles features, trains candidate models, and writes intermediate results or summaries back to disk.

## Source Package
The `src/onee` package collects the reusable pieces that support the strategies, including shared constants (`src/onee/constants/values.py`) and utilities (`src/onee/utils.py`). Install the project in editable mode to make the package importable when running scripts locally.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -e .
```

## Data Dependencies
Forecasts rely on SQLite databases stored under `data/`:
- `data/cd_database_2013_2023.db`: consolidated consumption and client history for the central directorate.
- `data/ONEE_Distributeurs_consumption.db`: distributor level energy usage used when `RUN_LEVELS` includes detailed distributor analyses.
- `data/ONEE_Regional_COMPLETE.db` and `data/ONEE_Regional_COMPLETE_2007_2023.db`: regional aggregates and long horizon time series used for SRM regional modelling.

Ensure the `.db` files remain in place; the scripts connect to them directly via `sqlite3`.

## Running Forecasts
1. Activate your environment and install the project (see above).
2. Review the configuration constants near the top of `run_srm_strategies.py` to set the unit, variable, regions, and model search space.
3. Launch the pipeline:

   ```bash
   python run_srm_strategies.py
   ```

Generated artefacts are written under `outputs_srm/` (SRM runs) and related directories configured inside the scripts.

## Working Notes
- Use the helper functions in `srm_strategies.py` if you need to call parts of the workflow from notebooks or other automation.
- Keep the `.db` datasets synchronized with your desired reporting window before triggering new runs; missing tables or mismatched schemas will surface as runtime errors during the SQLite queries.
