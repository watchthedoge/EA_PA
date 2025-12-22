import json
import numpy as np
from pathlib import Path

ROOT = Path("data_bench")

def read_best_y_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    runs = data["scenarios"][0]["runs"]
    best_y = [run["best"]["y"] for run in runs]
    return np.array(best_y)

results = []

for run_dir in ROOT.iterdir():
    if not run_dir.is_dir():
        continue
    if not run_dir.name.startswith("run_"):
        continue

    json_files = list(run_dir.glob("*.json"))
    if not json_files:
        continue

    json_path = json_files[0]  # only one per folder

    try:
        best_y = read_best_y_from_json(json_path)
    except Exception as e:
        print(f"Failed reading {json_path}: {e}")
        continue

    median_y = np.median(best_y)
    mean_y = np.mean(best_y)

    results.append({
        "folder": run_dir.name,
        "json": json_path.name,
        "median": median_y,
        "mean": mean_y,
    })

# sort by median (lower is better)
results.sort(key=lambda x: x["median"])

# ==========================
# PRINT SUMMARY
# ==========================
print("\n=== All configurations (sorted by median best y) ===\n")

for r in results:
    print(
        f"{r['folder']:30s} | "
        f"median = {r['median']:.6e} | "
        f"mean = {r['mean']:.6e}"
    )

# ==========================
# BEST PERFORMER
# ==========================
best = results[0]

print("\n=== BEST OVERALL CONFIGURATION ===\n")
print(f"Folder      : {best['folder']}")
print(f"JSON file   : {best['json']}")
print(f"Median best : {best['median']:.6e}")
print(f"Mean best   : {best['mean']:.6e}")
