import glob
import os
import shutil
import sys

FILES = ["lstm_weather_model.keras", "scaler.pkl", "config.json", "metrics.json"]

run_id = sys.argv[1]                                       # the Run ID shown on the run page in MLflow
matches = glob.glob(f"mlruns/*/{run_id}/artifacts/files")
if not matches:
    sys.exit(f"No run {run_id} found in mlruns/")

os.makedirs("production_model", exist_ok=True)
for name in FILES:
    shutil.copy(os.path.join(matches[0], name), "production_model")
with open("production_model/run_id.txt", "w") as f:        
    f.write(run_id)
print(f"Copied run {run_id} into production_model/")
