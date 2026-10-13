import json
import os
import sys

MODEL_DIR = "artifacts/forecast-model/_work"


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else MODEL_DIR

    with open(os.path.join(model_dir, "metrics.json")) as f:
        metrics = json.load(f)

    baseline = metrics["baseline_rmse"]
    lstm = metrics["lstm_rmse"]
    worse_days = [day + 1 for day in range(len(lstm)) if lstm[day] >= baseline[day]]

    if worse_days:
        print(f"Gate FAILED: LSTM is not better than the baseline on day(s) {worse_days}")
        sys.exit(1)
    print(f"Gate passed: LSTM beats the baseline on all {len(lstm)} days "
          f"({metrics['lstm_avg']} vs {metrics['baseline_avg']})")


if __name__ == "__main__":
    main()
