import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys


def return_columns(data: pd.DataFrame):
    print(data.columns)


def get_column_data(data: pd.DataFrame, *args):

    if len(args) != 1:
        print(
            "ERROR: Must Either Pass A String Or An Integer With Data Frame To `get_column_data()`"
        )
        sys.exit(1)

    if isinstance(args[0], int):
        index = args[0]
        num_columns = len(data.columns)

        if index < 0 or index > num_columns:
            print(
                f"ERROR: Given index: {index} is out of range. Must be between 0 and {num_columns-1}"
            )
            sys.exit(1)
        return data[index]

    elif isinstance(args[0], str):
        col_name = args[0]

        if not col_name in data.columns:
            print(f"ERROR: Given Column Name: {col_name} does not exist...")
            sys.exit(1)
        return data[col_name]

    else:
        print(
            "ERROR: Second Parameter passed to `get_column_data()` is nor a string or an int..."
        )
        sys.exit(1)


def checkNull(data, column_number):

    column_data = get_column_data(data, column_number)


def print_unique_vals(data, column_number):
    pass


def handle_missing_vals(df):
    # Check for missing values
    print(df.isnull().sum())

    # Visualize missing values
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title("Missing Values Heatmap")
    plt.show()

    # Handle missing values (choose appropriate methods)
    # df = df.dropna()  # Drop rows with missing values
    # df['column'] = df['column'].fillna(df['column'].mean())  # Fill with mean
    # df['column'] = df['column'].fillna(df['column'].mode()[0])  # Fill with mode


def main():

    csv_path = "./data/SBSA_DS_2026.csv"
    data = pd.read_csv(csv_path)

    handle_missing_vals(data)
    return
    # print(data.head())

    # Get dataframe info (data types, non-null counts)
    print(data.info())

    print(data.describe(include="all"))

    return

    column_data = get_column_data(data, "CustomerID")
    print(type(column_data))

    return
    column_names = list(data.columns)

    for idx, name in enumerate(column_names):
        print(f"Feature {idx + 1}: {name}")


if __name__ == "__main__":
    main()
