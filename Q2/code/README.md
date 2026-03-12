# Q2 Code (Direction A)

## Environment
- Python 3.10+
- Install dependencies:
  ```bash
  pip install -r Q2/code/requirements.txt
  ```

## Run
```bash
PYTHONPATH=Q2/code python Q2/code/scripts/run_q2_pipeline.py --data-path data/data.pq --out-dir Q2/outputs
```

If `data/data.pq` is unavailable, the loader automatically falls back to `data/split_chunk_*.pq`.

## Outputs
- `Q2/outputs/feature_diagnosis.csv`
- `Q2/outputs/feature_cleaning_log.csv`
- `Q2/outputs/feature_scores.csv`
- `Q2/outputs/selected_top50.csv`
- `Q2/outputs/model_metrics.csv`
- `Q2/outputs/agent_execution_log.json`
- `Q2/outputs/meta.json`
