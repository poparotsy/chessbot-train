# Project Memory (Persistent)

1. Use one canonical dependency install command everywhere:
`pip install -r requirements.txt`

2. Keep only one dependency file: `requirements.txt`

3. Keep generator profile-mix logging enabled.
Example output:
`Created val_0 | mix[default:..., screenshot_clutter:..., edge_rook:..., hard_mix:...]`

4. v5 must remain standalone scripts:
- `generate_hybrid_v5.py`
- `train_hybrid_v5.py`
- `recognizer_v5.py`
No direct module imports from v4 wrappers.

5. Current acceptance baseline:
`python3 scripts/evaluate_v5.py` => `29/30`, only `puzzle-00028` failing.
Treat 00028 as model/data issue unless proven otherwise.

6. Documentation is mandatory for every code change/commit.
Before or immediately after each commit, update `ENGINEERING_LOG.md` with:
- commit hash
- objective/rationale
- files changed
- behavior changes
- validation commands and outcome
