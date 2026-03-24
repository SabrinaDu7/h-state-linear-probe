@README.md

## Project-Specific Rules

### CRITICAL: Metrics
- All metric computation logic lives in `src/metrics.py` - SINGLE SOURCE OF TRUTH
- Threshold value (0.5) is in `src/config.py` as `MASK_THRESHOLD` - use `binarize()`, never hardcode
- Use `upsample_proba_and_binarize()` for predictions, never upsample binary masks

### Style
- DRY principles
- Reuse functions as much as possible and ask ME questions if confused.
