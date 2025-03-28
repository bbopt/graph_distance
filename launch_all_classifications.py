import subprocess
import os
import sys

if __name__ == "__main__":

    # User-defined parameters
    budget_per_param = 100
    nb_seeds = 20
    nb_of_classes = 5

    variants = [f"classification_variant{i}" for i in range(1, 6)]
    sizes = [1, 2]

    # Launch each script with each size
    for variant in variants:
        for size in sizes:
            print(f"Running {variant}/ACCURACY.py with size={size}, budget_per_param={budget_per_param}")

            subprocess.run(
                [
                    sys.executable,
                    "ACCURACY.py",
                    "--size", str(size),
                    "--budget_per_param", str(budget_per_param),
                    "--nb_seeds", str(nb_seeds),
                    "--nb_of_classes", str(nb_of_classes)
                ],
                cwd=variant  # Run from within the local directory
            )

