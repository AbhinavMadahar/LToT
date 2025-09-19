
# Artifact schema (results/artifact.jsonl)

Each line is a JSON object with fields:

- `run_id`: unique string for the pipeline run (timestamped).
- `task_id`: e.g., `gsm_hard`, `math_500`, `humaneval`, `mbpp_lite`, `game24`.
- `model_id`: e.g., `S`, `M`, `L`.
- `budget`: `low` | `med` | `high`.
- `method`: `CoT` | `ToT` | `LToT`.
- `metrics`: dictionary (Success@1 / Pass@1 / time_to_first_hit / false_promotion_rate ...).
- `tables`: optional dict-of-tables, each a list-of-rows.
- `figures`: list of figures: `{ "name": str, "png_base64": str, "caption": str }`.
- `notes`: free-form diagnostic info.

Use these IDs to programmatically update LaTeX (or ask ChatGPT to do it, pointing at this file).
