"""Shared switches. Add --smoke to any script for a tiny fast run that writes to results_smoke."""
import sys

SMOKE = "--smoke" in sys.argv
RESULTS = "results_smoke" if SMOKE else "results"
