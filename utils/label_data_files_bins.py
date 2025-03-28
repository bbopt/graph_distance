import os
import pandas as pd
from collections import defaultdict
import xlsxwriter
import numpy as np


# Function to extract accuracy values from all sheets in a given file
def extract_accuracy_values(filepath):
    accuracy_values = []

    xls = pd.ExcelFile(filepath)  # Load Excel file
    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name)  # Read sheet
        if "accuracy" in df.columns:  # Ensure column exists
            accuracy_values.extend(df["accuracy"].dropna().tolist())  # Collect values

    return np.array(accuracy_values)  # Convert to NumPy array for computations


# Function to replace accuracy values with labels based on fixed bins (0 to 100)
def replace_accuracy_with_labels(filepath, bins, output_filename):
    xls = pd.ExcelFile(filepath)
    with pd.ExcelWriter(output_filename, engine="xlsxwriter") as writer:
        for sheet_name in xls.sheet_names:
            df = xls.parse(sheet_name)
            if "accuracy" in df.columns:
                df["label"] = np.digitize(df["accuracy"], bins, right=True) - 1

                # Handle edge cases where values exactly match the upper bound of the last bin
                df.loc[df["accuracy"] == bins[-1], "label"] = len(bins) - 2

                # Handle edge cases where values exactly match the lower bound of the first bin
                df.loc[df["accuracy"] == bins[0], "label"] = 0

                df.drop(columns=["accuracy"], inplace=True)
            df.to_excel(writer, sheet_name=sheet_name, index=False)


def create_labeled_variant_size(variant, size, architecture, N=5):
    # Directory where files are stored
    directory = "C:/Users/edwar/Desktop/Meta_distance/article_distance/" + variant + "_classification"

    # Files to process
    data_file = os.path.join(directory, "data_" + variant + "_size" + str(size) + "_" + architecture + ".xlsx")

    # Set bins from 0 to 100
    bins = np.linspace(0, 100, N + 1)

    # Display computed bins
    print(f"Fixed {N}-Class Bins (0-100): {bins}")

    # Replace accuracy values with labels
    replace_accuracy_with_labels(data_file, bins,
            os.path.join(directory, "labeled_data_" + variant + "_size" + str(size) + "_" + architecture + ".xlsx"))

    labeled_file = os.path.join(directory, "labeled_data_" + variant + "_size" + str(size) + "_" + architecture + ".xlsx")

    # Function to count occurrences of each label
    def count_labels(filepath):
        label_counts = defaultdict(int)
        xls = pd.ExcelFile(filepath)
        for sheet_name in xls.sheet_names:
            df = xls.parse(sheet_name)
            if "label" in df.columns:
                counts = df["label"].value_counts().to_dict()
                for label, count in counts.items():
                    label_counts[label] += count
        return label_counts

    label_counts = count_labels(labeled_file)

    print("Label Counts:", dict(label_counts))



