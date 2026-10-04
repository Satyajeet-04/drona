"""Global paths and constants for the DRONA Smart Ticket Understanding Engine."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
ARTIFACTS = ROOT / "artifacts"
MODELS = ARTIFACTS / "models"
REPORTS = ARTIFACTS / "reports"
RESEARCH = ROOT / "research"
DOCS = ROOT / "docs"

for _p in (RAW, PROCESSED, ARTIFACTS, MODELS, REPORTS, RESEARCH, DOCS):
    _p.mkdir(parents=True, exist_ok=True)

# Reproducibility
SEED = 20261003

# Split ratios mandated by the synopsis: 70/15/15 stratified by category
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15