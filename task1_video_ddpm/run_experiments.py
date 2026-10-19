"""Run controlled experiments (linear vs cosine, lr change) and write results/experiments.md + results.csv."""
import json, os, sys
from train import main, parse

EXPS = [("linear_lr5e-4", ["--schedule", "linear", "--lr", "5e-4"]),
        ("cosine_lr5e-4", ["--schedule", "cosine", "--lr", "5e-4"]),
        ("cosine_lr2e-3", ["--schedule", "cosine", "--lr", "2e-3"])]
extra = sys.argv[1:]  # e.g. --epochs 30 --n_clips 1500
res = []
for name, args in EXPS:
    res.append(main(parse(["--out", f"runs/{name}"] + args + extra)) | {"name": name})
os.makedirs("results", exist_ok=True)
cols = ["name", "schedule", "lr", "test_loss", "motion_real", "motion_gen", "nn_dist_gen", "nn_dist_real", "train_s", "sample_s", "peak_mem_mb"]
md = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
for r in res:
    md += "| " + " | ".join(f"{r[c]:.4g}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |\n"
open("results/experiments.md", "w").write(md); json.dump(res, open("results/results.json", "w"), indent=1); print(md)
