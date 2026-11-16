import os, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# keep API prediction logs out of the project folder during tests
os.environ["PREDICTION_LOG"] = str(Path(tempfile.mkdtemp()) / "predictions.jsonl")
