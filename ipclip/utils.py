import os
import pandas as pd

def append_results_csv(dataset: str, task: str, results: list[dict], extra: dict | None = None):
    extra = extra or {}
    for r in results:
        r.update(extra)
    out_dir = os.environ.get("PC_RESULTS_FOLDER", "./results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"results_{task}_{dataset}.csv")
    df = pd.DataFrame(results)
    if os.path.exists(out_path):
        old = pd.read_csv(out_path, index_col=0)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(out_path)
    return out_path
