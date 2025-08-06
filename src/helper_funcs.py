import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import sys
from tabulate import tabulate
import os
import shutil


def debug_print(text, debug):
    """
    Easy function to help me print with debug option
    """
    if debug:
        print(text)


def print_heading(text):
    """
    Helper function to print headings to console
    """

    print("")
    print("=" * 20, text, "=" * 20)


# TODO: Check if this is needed
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


# TODO: Come back here and use this function
def handle_missing_vals(df):
    """ """
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


def get_cardinality(data: pd.DataFrame, debug=False):
    """
    Returns dictionary of unique counts for all features in `data`
    """

    debug_print("Counting Cardinalities of Features...", debug)

    col_names = data.columns.tolist()
    unique_counts = {name: 0 for name in col_names}

    for name in col_names:

        debug_print(f"   Counting: `{name}`", debug)

        column_data = data.loc[:, name].tolist()

        vals_seen = []
        for elem in column_data:
            if elem in vals_seen:
                continue

            vals_seen.append(elem)
            unique_counts[name] += 1

        debug_print("Done!", debug)

    if debug:
        print("Final Counts:")
        for key, value in unique_counts.items():
            print(f"   {key}: {value}")
        print("")

    return unique_counts


def get_sample_vals(feature: pd.Series, num_samples, debug=False):
    """
    Returns a list of randomly sampled data from column data `feature`
    """

    debug_print(f"Sampling From `{feature.name}`...", debug)

    sampled_vals = []

    for _ in range(num_samples):
        rand_num = random.randint(0, feature.shape[0] - 1)

        value = feature[rand_num]
        sampled_vals.append(value)
        debug_print(f"    {value}", debug)

    debug_print("", debug)
    return sampled_vals


def prepare_directory(base_dir):
    """
    Checks if a directory exists, clears its contents if it does,
    and creates subdirectories 'tables' & 'graphs'.

    Parameters:
        - base_dir (str): Path to the base directory to prepare

    Returns:
        - tuple (str, str):
            - 1. Path to the created 'graphs' directory
            - 2. Path to the created 'tables' directory

    Note:
        - Will exit program if given path does not exist
    """

    graphs_dir = os.path.join(base_dir, "graphs")
    tables_dir = os.path.join(base_dir, "tables")

    try:
        # Check if base directory exists
        if os.path.exists(base_dir):
            print(f"Directory '{base_dir}' exists. Clearing contents...")

            # Remove all contents of the directory
            for filename in os.listdir(base_dir):
                file_path = os.path.join(base_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
        else:
            print(f"Directory '{base_dir}' doesn't exist. Creating it...")
            os.makedirs(base_dir)

        # Create 'graphs' & 'tables' directories (will recreate if it existed before)
        os.makedirs(graphs_dir, exist_ok=True)
        os.makedirs(tables_dir, exist_ok=True)
        print(f"Created directory '{graphs_dir}' & '{tables_dir}'")

        return (graphs_dir, tables_dir)

    except Exception as e:
        print(f"Error preparing directory: {e}")
        sys.exit(1)


def save_data_quality_report(df_report: pd.DataFrame, output_dir: str):
    """
    Saves a pandas DataFrame containing data quality metrics in multiple formats.

    Args:
        df_report (pd.DataFrame): The data quality report DataFrame
        output_dir (str): Directory to save the report
    """

    # Base filename without extension
    base_filename = f"{output_dir}/data_quality_report"

    # 1. Save as raw text (pretty formatted table)
    txt_file = f"{base_filename}.txt"
    with open(txt_file, "w") as f:
        f.write(tabulate(df_report, headers="keys", tablefmt="psql", showindex=False))
    print(f"Saved text version to: {txt_file}")

    # 2. Save as CSV (machine-readable)
    csv_file = f"{base_filename}.csv"
    df_report.to_csv(csv_file, index=False)
    print(f"Saved CSV version to: {csv_file}")

    # 3. Save as Markdown (for documentation)
    md_file = f"{base_filename}.md"
    with open(md_file, "w") as f:
        f.write(tabulate(df_report, headers="keys", tablefmt="github", showindex=False))
    print(f"Saved Markdown version to: {md_file}")

    # 5. Save as HTML (for web viewing)
    html_file = f"{base_filename}.html"
    df_report.to_html(html_file, index=False, border=1, justify="center")
    print(f"Saved HTML version to: {html_file}")


def discern_cat_vs_cts(data: pd.DataFrame):
    """
    Prints Nice Table with headings `[feature name, Data Type, Cardiniality]`
    """
    data_types = data.dtypes
    card_counts = get_cardinality(data, debug=False)

    data_quality_report = {
        "Feature": [],
        "Data Type": [],
        "Cardinality": [],
    }
    for name in data.columns.tolist():
        data_quality_report["Feature"].append(name)
        data_quality_report["Data Type"].append(data_types[name])
        data_quality_report["Cardinality"].append(card_counts[name])

    df_dqr = pd.DataFrame(data_quality_report)
    print(tabulate(df_dqr, headers="keys", tablefmt="psql", showindex=False))


def get_mode_info(df, top_n=2):
    """
    Calculate mode statistics for all columns in a DataFrame.

    Args:
        df: pandas DataFrame
        top_n: Number of modes to calculate (default: 2)

    Returns:
        DataFrame with mode statistics for each column
    """
    results = []

    for col in df.columns:
        value_counts = df[col].value_counts()
        total = len(df[col])

        col_stats = {}

        # Add stats for each mode requested
        for i in range(1, top_n + 1):
            if i <= len(value_counts):
                mode = value_counts.index[i - 1]
                count = value_counts.iloc[i - 1]
                percent = (count / total) * 100

                col_stats.update(
                    {
                        f"Mode {i}": mode,
                        f"Mode {i} Frequency": count,
                        f"Mode {i} %": round(percent, 2),
                    }
                )
            else:
                col_stats.update(
                    {
                        f"Mode {i}": None,
                        f"Mode {i} Frequency": None,
                        f"Mode {i} %": None,
                    }
                )

        results.append(col_stats)

    return pd.DataFrame(results)


def get_numerical_stats(df):
    """
    Calculate descriptive statistics for numerical columns in a DataFrame.

    Args:
        df: pandas DataFrame

    Returns:
        DataFrame with statistics for each numerical column
    """

    results = []

    for col in df.columns:

        col_stats = {}

        series_data = df[col]

        col_stats["Minimum"] = series_data.min()
        col_stats["1st Quartile"] = series_data.quantile(0.25)
        col_stats["Mean"] = series_data.mean()
        col_stats["Median"] = series_data.median()
        col_stats["3rd Quartile"] = series_data.quantile(0.75)
        col_stats["Maximum"] = series_data.max()
        col_stats["Std. Dev"] = series_data.std()

        results.append(col_stats)

    return pd.DataFrame(results)


def plot_histograms(df, save_dir="histograms"):
    """
    Plot histograms for all features in a DataFrame and save them to a directory.

    Args:
        df: pandas DataFrame
        save_dir: Directory to save histogram images (default: "histograms")
    """

    # Loop through each column in the DataFrame
    for column in df.columns:
        try:
            plt.figure(figsize=(10, 6))

            # For numeric columns
            df[column].hist(bins=30, edgecolor="black", color="skyblue")
            plt.axvline(
                df[column].mean(),
                color="red",
                linestyle="dashed",
                linewidth=1,
                label=f"Mean: {df[column].mean():.2f}",
            )
            plt.axvline(
                df[column].median(),
                color="green",
                linestyle="dashed",
                linewidth=1,
                label=f"Median: {df[column].median():.2f}",
            )
            plt.legend()

            # Customize plot appearance
            plt.title(f"Distribution of {column}", fontsize=14)
            plt.xlabel(column, fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            plt.grid(axis="y", alpha=0.5)
            plt.xticks(
                rotation=45 if not pd.api.types.is_numeric_dtype(df[column]) else 0
            )
            plt.tight_layout()

            # Save the plot
            filename = os.path.join(save_dir, f"{column}.png")
            plt.savefig(filename, dpi=100, bbox_inches="tight")
            plt.close()

            print(f"Saved histogram for {column} to {filename}")

        except Exception as e:
            print(f"Could not create histogram for {column}: {str(e)}")
            plt.close()

    # Create combined histogram plot
    figsize = (20, 15)
    bins = 30
    ax = df.hist(
        figsize=figsize,
        bins=bins,
        edgecolor="black",
        xlabelsize=10,
        ylabelsize=10,
        grid=False,
    )

    # Customize subplots
    for axis in ax.flatten():
        axis.set_ylabel("Frequency", fontsize=10)
        axis.tick_params(axis="x", rotation=45)

    filename = os.path.join(save_dir, "all_graphs.png")
    plt.suptitle("Feature Distributions", y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig(filename, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved combined histograms to {filename}")


def plot_categorical_bars(df, save_dir="bar_plots"):
    """
    Create bar plots for categorical features and save to directory.

    Args:
        df: pandas DataFrame with categorical features
        save_dir: Directory to save bar plot images
    """
    for column in df.columns:
        try:
            plt.figure(figsize=(12, 6))

            # Get value counts and sort by frequency
            value_counts = df[column].value_counts().sort_values(ascending=False)

            # Create bar plot
            ax = value_counts.plot(
                kind="bar", color="#1f77b4", edgecolor="black", width=0.8
            )

            # Add value labels on top of bars
            for p in ax.patches:
                ax.annotate(
                    f"{p.get_height(): }",
                    (p.get_x() + p.get_width() / 2.0, p.get_height()),
                    ha="center",
                    va="center",
                    xytext=(0, 5),
                    textcoords="offset points",
                )

            # Customize plot
            plt.title(f"Distribution of {column}", fontsize=14, pad=20)
            plt.xlabel(column, fontsize=12)
            plt.ylabel("Count", fontsize=12)
            plt.xticks(rotation=45, ha="right")
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()

            # Save plot
            filename = f"{save_dir}/{column}_barplot.png"
            plt.savefig(filename, dpi=120, bbox_inches="tight")
            plt.close()

            print(f"Saved bar plot for {column} to {filename}")

        except Exception as e:
            print(f"Error creating bar plot for {column}: {e}")
            plt.close()

    # Create Combined Subplots
    filename = os.path.join(save_dir, "all_graphs.png")
    figsize = (20, 15)

    # Calculate grid dimensions
    n_cols = 3
    n_rows = (len(df.columns) + n_cols - 1) // n_cols

    # Create figure
    _, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i, col in enumerate(df.columns):
        ax = axes[i]
        # Get top 15 categories if many unique values
        value_counts = df[col].value_counts()[:15]
        value_counts.plot(kind="bar", ax=ax, color="#1f77b4", edgecolor="black")

        # Customizations
        ax.set_title(col, fontsize=12)
        ax.set_ylabel("Count")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)

    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Categorical Feature Distributions", y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig(filename, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved combined bar plots to {filename}")
