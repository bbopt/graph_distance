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

    return np.array(accuracy_values)  # Convert to NumPy array for quantile computation


# Function to replace accuracy values with labels based on quantiles
def replace_accuracy_with_labels(filepath, quantiles, output_filename):
    xls = pd.ExcelFile(filepath)
    with pd.ExcelWriter(output_filename, engine="xlsxwriter") as writer:
        for sheet_name in xls.sheet_names:
            df = xls.parse(sheet_name)
            if "accuracy" in df.columns:
                df["label"] = np.digitize(df["accuracy"], quantiles, right=True) - 1

                # Handle edge cases where values exactly match the upper bound of the last quantile
                df.loc[df["accuracy"] == quantiles[-1], "label"] = len(quantiles) - 2

                # Handle edge cases where values exactly match the lower bound of the first quantile
                df.loc[df["accuracy"] == quantiles[0], "label"] = 0

                df.drop(columns=["accuracy"], inplace=True)
            df.to_excel(writer, sheet_name=sheet_name, index=False)


if __name__ == "__main__":
    # Define directory where files are stored
    directory = "C:/Users/edwar/Desktop/Meta_distance/article_distance/variant1_classification"
    # Number of quantiles
    N = 10  # Number of classes


    # Delete existing merged and labeled files if they exist
    files_to_delete = [
        os.path.join(directory, "data_variant1_MLP.xlsx"),
        os.path.join(directory, "data_variant1_CNN.xlsx"),
        os.path.join(directory, "labeled_data_variant1_MLP.xlsx"),
        os.path.join(directory, "labeled_data_variant1_CNN.xlsx")]

    for file in files_to_delete:
        if os.path.exists(file):
            os.remove(file)
            print(f"Deleted: {file}")

    # Get all .xlsx files in the directory
    files = [f for f in os.listdir(directory) if f.endswith(".xlsx")]

    # Separate files into CNN and MLP groups
    cnn_files = [os.path.join(directory, f) for f in files if "_CNN" in f]
    mlp_files = [os.path.join(directory, f) for f in files if "_MLP" in f]

    # Function to merge sheets from multiple Excel files
    def merge_sheets(file_list, output_filename):
        merged_data = defaultdict(list)

        # Read each file and store sheets
        for file in file_list:
            xls = pd.ExcelFile(file)  # Load Excel file
            for sheet_name in xls.sheet_names:
                df = xls.parse(sheet_name)  # Read sheet
                merged_data[sheet_name].append(df)  # Store without adding "Source_File"

        # Save merged sheets to a new Excel file
        with pd.ExcelWriter(os.path.join(directory, output_filename), engine="xlsxwriter") as writer:
            for sheet_name, dfs in merged_data.items():
                merged_df = pd.concat(dfs, ignore_index=True)  # Concatenate all DataFrames
                merged_df.to_excel(writer, sheet_name=sheet_name, index=False)  # Write to sheet

    # Merge CNN and MLP files separately
    merge_sheets(cnn_files, "data_variant1_CNN.xlsx")
    merge_sheets(mlp_files, "data_variant1_MLP.xlsx")

    # Save merged files
    mlp_file = os.path.join(directory, "data_variant1_MLP.xlsx")
    cnn_file = os.path.join(directory, "data_variant1_CNN.xlsx")

    # Extract accuracy values from MLP and CNN files
    accuracy_mlp = extract_accuracy_values(mlp_file)
    accuracy_cnn = extract_accuracy_values(cnn_file)

    # Compute quantiles
    quantiles_mlp = np.quantile(accuracy_mlp, np.linspace(0, 1, N + 1))
    quantiles_cnn = np.quantile(accuracy_cnn, np.linspace(0, 1, N + 1))

    # Display computed quantiles
    print(f"MLP {N}-Quantiles: {quantiles_mlp}")
    print(f"CNN {N}-Quantiles: {quantiles_cnn}")

    # Replace accuracy values with labels in new Excel files
    replace_accuracy_with_labels(mlp_file, quantiles_mlp, os.path.join(directory, "labeled_data_variant1_MLP.xlsx"))
    replace_accuracy_with_labels(cnn_file, quantiles_cnn, os.path.join(directory, "labeled_data_variant1_CNN.xlsx"))

    labeled_mlp_file = os.path.join(directory, "labeled_data_variant1_MLP.xlsx")
    labeled_cnn_file = os.path.join(directory, "labeled_data_variant1_CNN.xlsx")

    # Count occurrences of each label in the labeled files
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


    mlp_label_counts = count_labels(labeled_mlp_file)
    cnn_label_counts = count_labels(labeled_cnn_file)

    print("MLP Label Counts:", dict(mlp_label_counts))
    print("CNN Label Counts:", dict(cnn_label_counts))