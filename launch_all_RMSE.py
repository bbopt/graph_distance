import subprocess
import os
import sys

if __name__ == "__main__":

    # User-defined parameters
    nb_seeds = 5
    budget_per_param = 200

    variants = [f"RMSE_variant{i}" for i in range(1, 6)]
    architectures = ["MLP", "CNN"]

    for variant in variants:
        for architecture in architectures:
            for seed in range(nb_seeds):
                print(
                    f"Running {variant}/RMSE.py with architecture={architecture}, seed_setup={seed}, budget_per_param={budget_per_param}")
                subprocess.run(
                    [
                        sys.executable,
                        "RMSE.py",
                        "--architecture", architecture,
                        "--seed_setup", str(seed),
                        "--budget_per_param", str(budget_per_param)
                    ],
                    cwd=variant
                )
