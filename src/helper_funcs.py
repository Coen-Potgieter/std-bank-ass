import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import sys
from tabulate import tabulate
import os
import shutil


def filter_data(df, condition, return_complement=False) -> pd.DataFrame:
    """
    Filter DataFrame based on a condition with options to:
    - Return matching records (default)
    - Return non-matching records (complement)
    - Get both sets

    Args:
        df: Input DataFrame
        condition: Boolean Series (e.g., df["Mobile_App_Registered"] == True)
        return_complement: If True, returns records NOT matching the condition

    Returns:
        Filtered DataFrame (or tuple if return_complement='both')
    """
    filtered = df[condition]

    if return_complement == "both":
        return filtered, df[~condition]
    elif return_complement:
        return df[~condition]
    return filtered


def check_and_remove_duplicates(data: pd.DataFrame):
    """
    Point of this was to check and remove duplicate records, but there are none :D
    """
    num_dups = data.duplicated().sum()
    print(num_dups)


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
            value_counts = (
                df[column].value_counts(dropna=False).sort_values(ascending=False)
            )

            value_counts.index = value_counts.index.fillna("Missing")

            # Create bar plot
            ax = value_counts.plot(
                kind="bar",
                color=[
                    "#1f77b4" if x != "Missing" else "#d62728"
                    for x in value_counts.index
                ],
                edgecolor="black",
                width=0.8,
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
            plt.title(
                f"Distribution of {column}\n(Total: {len(df)}, Missing: {df[column].isna().sum()})",
                fontsize=14,
                pad=20,
            )
            # plt.title(f"Distribution of {column}", fontsize=14, pad=20)
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
        value_counts = df[col].value_counts(dropna=False)[:15]
        value_counts.index = value_counts.index.fillna("Missing")

        colors = [
            "#1f77b4" if x != "Missing" else "#d62728" for x in value_counts.index
        ]
        value_counts.plot(kind="bar", ax=ax, color=colors, edgecolor="black")

        # Customizations
        ax.set_title(f"{col}\n(Missing: {df[col].isna().sum()})", fontsize=12)
        ax.set_ylabel("Count")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)

    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        "Categorical Feature Distributions (Missing values in red)", y=1.02, fontsize=16
    )
    plt.tight_layout()
    plt.savefig(filename, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved combined bar plots to {filename}")


def sepearate_mobile_online_offline(data):
    """
    Exploring if the differences in data between the following categories `["Mobile", "Online_Only", "Non_Digital"]`
    Ultimately Fruitless Endevour...
    """

    def check_numerical_feats():
        value_metrics = [
            "Annual_Income_Estimate",
            "Credit_Score",
            "Current_Account_Balance",
            "Savings_Account_Balance",
            "Credit_Card_Balance",
            "Mobile_App_Txns",
        ]

        segment_stats = data.groupby("Segment")[value_metrics].agg(
            ["mean", "median", "count"]
        )
        print(tabulate(segment_stats, headers="keys", tablefmt="psql", showindex=False))

        # Plot distributions
        for metric in value_metrics:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=data, x="Segment", y=metric)
            plt.title(f"{metric} by Engagement Segment")
            plt.xticks(rotation=45)
            plt.show()

    def check_single_product_distr():
        products = ["Credit_Card", "Personal_Loan", "Home_Loan", "Vehicle_Loan"]

        for product in products:
            cross_tab = (
                pd.crosstab(data["Segment"], data[product], normalize="index") * 100
            )
            print(f"\n{product} Penetration (%):")
            print(cross_tab[True].sort_values(ascending=False))

    def check_credit_score():
        # Compare transaction frequency
        sns.lmplot(
            data=data,
            x="Mobile_App_Txns",
            y="Credit_Score",
            hue="Segment",
            height=6,
            aspect=1.5,
        )
        plt.title("Credit Score vs Mobile Transactions by Segment")
        plt.show()

    def check_custom_score():
        # Normalize metrics (0-1 scale)
        value_factors = data[
            ["Annual_Income_Estimate", "Credit_Score", "Savings_Account_Balance"]
        ].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

        data["Value_Score"] = value_factors.mean(axis=1)

        # Compare across segments
        sns.violinplot(data=data, x="Segment", y="Value_Score")
        plt.title("Customer Value Score by Engagement Segment")
        plt.show()

    def check_employement():
        emp_dist = (
            pd.crosstab(
                data["Segment"], data["Employment_Type"], normalize="index"
            ).round(4)
            * 100
        )

        print("Employment Type Distribution (%):")
        print(emp_dist)

        plt.figure(figsize=(12, 6))

        sns.countplot(
            data=data,
            x="Segment",
            hue="Employment_Type",
            palette="Set2",
        )
        plt.title("Employment Type Distribution by Engagement Segment")
        plt.ylabel("Customer Count")
        plt.xlabel("Engagement Segment")
        plt.legend(title="Employment Type")
        plt.show()

    def check_number_products_held():
        # List of product columns
        products = [
            "Current_Account",
            "Savings_Account",
            "Credit_Card",
            "Personal_Loan",
            "Home_Loan",
            "Vehicle_Loan",
        ]

        # Convert boolean columns to integers (True=1, False=0)
        data_products = data[products].astype(int)

        # Create new column counting how many products each customer has
        data["Product_Count"] = data_products.sum(axis=1)

        # Verify distribution
        print(data["Product_Count"].value_counts().sort_index())

        # Cross-tabulation
        count_dist = pd.crosstab(data["Segment"], data["Product_Count"], margins=True)
        print("Product Count Distribution:")
        print(count_dist)

        # Normalized view
        count_pct = (
            pd.crosstab(
                data["Segment"], data["Product_Count"], normalize="index"
            ).round(4)
            * 100
        )

        print("\nPercentage Distribution:")
        print(count_pct)

        plt.figure(figsize=(12, 6))
        count_pct[range(data["Product_Count"].max() + 1)].plot(
            kind="bar", stacked=True, colormap="viridis"
        )
        plt.title("Product Count Distribution by Engagement Segment")
        plt.ylabel("Percentage of Customers")
        plt.xlabel("Engagement Segment")
        plt.legend(title="Number of Products", bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.heatmap(count_pct, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=0.5)
        plt.title("Product Count Distribution (%) by Segment")
        plt.ylabel("Engagement Segment")
        plt.xlabel("Number of Products")
        plt.show()

        sns.lmplot(
            data=data,
            x="Product_Count",
            y="Mobile_App_Txns",
            hue="Segment",
            height=6,
            aspect=1.3,
        )
        plt.title("Mobile Transactions vs Product Count")
        plt.show()

    def check_age_stats():
        age_stats = (
            data.groupby("Segment")["Age"]
            .agg(
                [
                    "mean",
                    "median",
                    "std",
                    "min",
                    "max",
                    lambda x: x.quantile(0.75) - x.quantile(0.25),
                ]
            )
            .rename(columns={"<lambda>": "IQR"})
        )

        print(age_stats.round(1))
        plt.figure(figsize=(12, 6))

        # Boxplot
        sns.boxplot(
            data=data,
            x="Segment",
            y="Age",
            palette="Set2",
            showfliers=False,
        )
        plt.title("Age Distribution by Customer Segment")
        plt.ylabel("Age")
        plt.xlabel("")
        plt.show()

        # Density plot
        plt.figure(figsize=(12, 6))
        sns.kdeplot(
            data=data,
            x="Age",
            hue="Segment",
            common_norm=False,
            palette="Set2",
            alpha=0.6,
            fill=True,
        )
        plt.title("Age Density Across Segments")
        plt.xlabel("Age")
        plt.ylabel("Density")
        plt.show()

    def check_transaction_channels():
        """
        Analyze transaction channels including in-person (non-digital) activity
        """

        # Create a comprehensive channel segment
        data["Transaction_Channel"] = np.where(
            (data["Mobile_App_Txns"] > 0) & (data["Online_Banking_Txns"] > 0),
            "Omnichannel",
            np.where(
                data["Mobile_App_Txns"] > 0,
                "Mobile_Only",
                np.where(
                    data["Online_Banking_Txns"] > 0,
                    "Online_Only",
                    "In_Person",  # Neither mobile nor online transactions
                ),
            ),
        )

        # 1. Transaction Volume Analysis
        channel_stats = data.groupby("Segment")[
            ["Mobile_App_Txns", "Online_Banking_Txns"]
        ].agg(["mean", "sum", "count"])

        # Add in-person stats (count of customers with zero digital transactions)
        in_person_stats = (
            data[data["Transaction_Channel"] == "In_Person"].groupby("Segment").size()
        )
        channel_stats[("In_Person", "count")] = in_person_stats

        print("\n=== Transaction Volume Analysis ===")
        print(channel_stats.round(1))

        # 2. Channel Preference Heatmap
        channel_pct = data.groupby("Segment").apply(
            lambda x: pd.Series(
                {
                    "Mobile_%": x["Mobile_App_Txns"].sum()
                    / max(
                        1, (x["Mobile_App_Txns"].sum() + x["Online_Banking_Txns"].sum())
                    )
                    * 100,
                    "Online_%": x["Online_Banking_Txns"].sum()
                    / max(
                        1, (x["Mobile_App_Txns"].sum() + x["Online_Banking_Txns"].sum())
                    )
                    * 100,
                    "In_Person_%": (x["Transaction_Channel"] == "In_Person").mean()
                    * 100,
                }
            )
        )

        plt.figure(figsize=(12, 6))
        sns.heatmap(channel_pct, annot=True, fmt=".0f", cmap="YlGnBu", linewidths=0.5)
        plt.title(
            "Channel Preference by Segment (%)\n(In-Person = % with zero digital transactions)"
        )
        plt.show()

        # 3. Detailed Channel Distribution
        print("\n=== Detailed Channel Distribution ===")
        channel_dist = (
            pd.crosstab(
                data["Segment"], data["Transaction_Channel"], normalize="index"
            ).round(3)
            * 100
        )
        print(channel_dist)

        # 4. Demographic Comparison
        print("\n=== Demographic Profile by Channel ===")
        demo_stats = (
            data.groupby("Transaction_Channel")
            .agg(
                {
                    "Age": "median",
                    "Annual_Income_Estimate": "median",
                    "Credit_Score": "mean",
                }
            )
            .round(1)
        )
        print(demo_stats)

    def use_mobile_online_offline():
        conditions = [
            (data["Mobile_App_Registered"] == True),  # Mobile users
            (data["Online_Banking_Registered"] == True)
            & (data["Mobile_App_Registered"] == False),  # Online-only
            (data["Online_Banking_Registered"] == False)
            & (data["Mobile_App_Registered"] == False),  # Non-digital
        ]

        segments = ["Mobile", "Online_Only", "Non_Digital"]
        data["Segment"] = np.select(conditions, segments, default="Other")

    def use_mobile_non_mobile():
        conditions = [
            (data["Mobile_App_Registered"] == True),  # Mobile users
            (data["Mobile_App_Registered"] == False),  # Non-digital
        ]

        segments = ["Mobile", "Non_Mobile"]
        data["Segment"] = np.select(conditions, segments, default="Other")

    def use_suburbs():
        conditions = [
            (data["Suburb"] == "Sandton"),
            (data["Suburb"] == "Fourways"),
            (data["Suburb"] == "Newlands"),
        ]

        segments = ["Sandton", "Fourways", "Newlands"]
        data["Segment"] = np.select(conditions, segments, default="Other")

    def use_province():
        conditions = [
            (data["Province"] == "Gauteng"),
            (data["Province"] == "Western Cape"),
            (data["Province"] == "KwaZulu-Natal"),
        ]

        segments = ["G", "CP", "KZN"]
        data["Segment"] = np.select(conditions, segments, default="Other")

    def use_product_count_segments():
        products = [
            "Current_Account",
            "Savings_Account",
            "Credit_Card",
            "Personal_Loan",
            "Home_Loan",
            "Vehicle_Loan",
        ]

        # Calculate number of products held (converting booleans to integers)
        product_count = data[products].astype(int).sum(axis=1)

        # Define segments
        conditions = [
            (product_count == 0),
            (product_count == 1),
            (product_count == 2),
            (product_count == 3),
            (product_count >= 4),
        ]

        segments = [
            "0_Products",
            "1_Product",
            "2_Products",
            "3_Products",
            "4+_Products",
        ]
        data["Segment"] = np.select(conditions, segments, default="Other")

    def use_employement():
        conditions = [
            (data["Employment_Type"] == "Part-time employed"),
            (data["Employment_Type"] == "Self-employed"),
            (data["Employment_Type"] == "Full-time employed"),
        ]

        segments = ["PART", "SELF", "FULL"]
        data["Segment"] = np.select(conditions, segments, default="Other")

    def use_age_segments():
        conditions = [
            (data["Age"] < 20),
            ((data["Age"] >= 20) & (data["Age"] < 30)),
            ((data["Age"] >= 30) & (data["Age"] < 40)),
            ((data["Age"] >= 40) & (data["Age"] < 50)),
            ((data["Age"] >= 50) & (data["Age"] < 60)),
            (data["Age"] >= 60),
        ]

        segments = ["Under_20", "20s", "30s", "40s", "50s", "60+"]

        data["Segment"] = np.select(conditions, segments, default="Unknown")

    def use_app_active():
        nonlocal data
        data = filter_data(data, data["Mobile_App_Registered"] == True)

        conditions = [
            (data["Mobile_App_Txns"] == 0),
            (data["Mobile_App_Txns"] != 0),
        ]

        segments = ["Mobile_Not_Active", "Mobile_Active"]

        data["Segment"] = np.select(conditions, segments, default="Unknown")

    # use_mobile_online_offline()
    use_mobile_non_mobile()
    # use_suburbs()
    # use_product_count_segments()
    # use_province()
    # use_employement()
    # use_age_segments()
    # use_app_active()

    # # Verify distribution
    print(data["Segment"].value_counts())

    check_numerical_feats()
    check_single_product_distr()
    check_credit_score()
    check_custom_score()
    check_employement()
    check_number_products_held()
    check_age_stats()
    check_transaction_channels()

    # Calculate mean transactions by channel and segment


def inspect_low_age_records(data, cutoff_age=13, show_columns=None):
    """
    Inspect records with ages below threshold

    Args:
        cutoff_age: Age threshold (default 13)
        show_columns: List of columns to display (default shows key columns)
    """
    if show_columns is None:
        show_columns = [
            "CustomerID",
            "Age",
            "Home_Loan",
            "Vehicle_Loan",
            "Annual_Income_Estimate",
            "Employment_Type",
        ]

    low_age_mask = data["Age"] < cutoff_age
    low_age_count = low_age_mask.sum()

    print(f"\nFound {low_age_count} records under {cutoff_age} years old")

    if low_age_count > 0:
        # Display records
        print("\nRecord Details:")
        print(
            tabulate(
                data.loc[low_age_mask, show_columns],
                headers="keys",
                tablefmt="psql",
                showindex=False,
            )
        )
    else:
        print("No suspicious age records found")


def clean_age_anomalies(data, debug=False) -> pd.DataFrame:
    """
    Remove records with invalid age-based combinations:
    1. Ages under 16
    2. Ages under 18 with home/vehicle loans

    Returns:
        Cleaned DataFrame
        Report of removed records
    """
    # Create masks for each condition
    condition1 = data["Age"] < 16  # Under 13
    condition2 = (data["Age"] < 18) & (
        (data["Home_Loan"] == True) | (data["Vehicle_Loan"] == True)
    )

    # Combine all conditions
    remove_mask = condition1 | condition2

    # Create report before removal
    report = {
        "total_removed": remove_mask.sum(),
        "under_13": condition1.sum(),
        "loans_under_18": condition2.sum(),
    }

    # Clean the data
    cleaned_data = data[~remove_mask].copy()

    debug_print(report, debug)

    return cleaned_data


def explore_non_active_mobile_users(data: pd.DataFrame):
    """
    WE FOUND SOMETHING INETERESTING
    """
    # Split into Segments
    mobile_app_registered = filter_data(data, data["Mobile_App_Registered"] == True)
    mobile_inactive = filter_data(
        mobile_app_registered, mobile_app_registered["Mobile_App_Txns"] == 0
    )

    num_mobile = len(mobile_app_registered)
    print(num_mobile)
    num_mobile_inactive = len(mobile_inactive)
    print(num_mobile_inactive)
    return
    num_mobile_active = num_mobile - len(mobile_inactive)
    # summary_data = [
    #     [
    #         "Active",
    #         num_mobile_active,
    #         f"{round(num_mobile_active/num_mobile*100, 2)}%",
    #     ],
    #     [
    #         "Inactive",
    #         num_mobile_inactive,
    #         f"{round(num_mobile_inactive/num_mobile*100, 2)}%",
    #     ],
    #     [
    #         "Sum",
    #         num_mobile,
    #         "100%",
    #     ],
    # ]

    # # Convert to DataFrame
    # df_summary = pd.DataFrame(
    #     summary_data, columns=["Mobile Registered", "Count", "Percentage"]
    # )
    # # Print with tabulate
    # print(
    #     tabulate(
    #         df_summary,
    #         headers="keys",
    #         tablefmt="psql",
    #         showindex=False,
    #         colalign=("left", "right", "right"),
    #     )
    # )

    # Split into subgroups
    mobile_inactive_online_active = filter_data(
        mobile_inactive, mobile_inactive["Online_Banking_Txns"] > 0
    )
    fully_inactive = filter_data(
        mobile_inactive,
        mobile_inactive["Online_Banking_Txns"] > 0,
        return_complement=True,
    )

    summary_data = [
        [
            "Online Banking",
            len(mobile_inactive_online_active),
            f"{round(len(mobile_inactive_online_active)/num_mobile_inactive*100, 2)}%",
        ],
        [
            "Offline",
            len(fully_inactive),
            f"{round(len(fully_inactive)/num_mobile_inactive*100, 2)}%",
        ],
        [
            "Sum",
            num_mobile_inactive,
            "100%",
        ],
    ]

    # Convert to DataFrame
    df_summary = pd.DataFrame(
        summary_data, columns=["Mobile Registered But Inactive", "Count", "Percentage"]
    )
    # Print with tabulate
    print(
        tabulate(
            df_summary,
            headers="keys",
            tablefmt="psql",
            showindex=False,
            colalign=("left", "right", "right"),
        )
    )

    # Combine for easier analysis
    comparison_df = pd.concat(
        [
            mobile_inactive_online_active.assign(Group="Online_Active"),
            fully_inactive.assign(Group="Fully_Inactive"),
        ]
    )

    # 1. Count Comparison Visualization
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=["Using Online Banking", "Fully Inactive"],
        y=[len(mobile_inactive_online_active), len(fully_inactive)],
        palette=["#1f77b4", "#d62728"],
    )
    plt.title("Mobile-Inactive User Behavior Split")
    plt.ylabel("Number of Customers")
    plt.show()

    # 2. Demographic Comparison
    # plt.figure(figsize=(10, 6))

    # # Age distribution
    # sns.boxplot(
    #     data=comparison_df,
    #     x="Group",
    #     y="Age",
    #     palette=["#1f77b4", "#d62728"],
    # )
    # plt.title("Age Distribution")
    # plt.show()


def explore_online_vs_mobile_banking(data: pd.DataFrame):
    """
    Analyze transaction patterns between online banking and mobile app
    for customers with at least one transaction in either channel.

    Args:
        data: DataFrame containing transaction columns

    Returns:
        DataFrame with analysis results
    """
    # Filter to only customers with transactions in either channel
    active_data = data[
        (data["Online_Banking_Txns"] > 0) | (data["Mobile_App_Txns"] > 0)
    ].copy()

    print(f"Analyzing {len(active_data)} customers with transactions")
    print(f"Excluded {len(data)-len(active_data)} customers with zero transactions\n")

    # Create separate datasets for each channel (excluding zeros)
    mobile_users = active_data[active_data["Mobile_App_Txns"] > 0].copy()
    online_users = active_data[active_data["Online_Banking_Txns"] > 0].copy()

    # 1. Basic transaction stats (for users who actually used each channel)
    print("=== Transaction Summary (Mobile Users) ===")
    print(mobile_users["Mobile_App_Txns"].describe().round(1))

    print("\n=== Transaction Summary (Online Users) ===")
    print(online_users["Online_Banking_Txns"].describe().round(1))

    # 2. Transaction ratio analysis (unchanged)
    active_data["Mobile_Ratio"] = active_data["Mobile_App_Txns"] / (
        active_data["Mobile_App_Txns"] + active_data["Online_Banking_Txns"]
    )

    # 3. Visualizations of non-zero distributions
    plt.figure(figsize=(12, 5))

    # Transaction distribution (only positive values)
    sns.histplot(
        online_users["Online_Banking_Txns"],
        color="blue",
        label="Online",
        alpha=0.5,
        bins=30,
    )
    sns.histplot(
        mobile_users["Mobile_App_Txns"],
        color="green",
        label="Mobile",
        alpha=0.5,
        bins=30,
    )
    plt.title("Transaction Distribution (Non-Zero Values Only)")
    plt.xlabel("Number of Transactions")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 4. Correlation analysis (unchanged)
    print("\n=== Correlation Analysis ===")
    corr_matrix = active_data[["Online_Banking_Txns", "Mobile_App_Txns"]].corr()
    print(corr_matrix)


def nice_print_mobile_txns2(data: pd.DataFrame):

    # data = filter_data(
    #     data,
    #     (data["Mobile_App_Registered"] == True)
    #     & (data["Online_Banking_Registered"] == False),
    # )
    data = filter_data(data, data["Mobile_App_Registered"] == True)

    print_column = data.loc[:, "Mobile_App_Txns"]

    try:
        plt.figure(figsize=(10, 6))

        # Plot histogram and get bin information
        n, bins, patches = plt.hist(
            print_column, bins=30, edgecolor="black", color="skyblue"
        )

        # Find and color the 0-transactions bar red
        zero_bin_index = np.where(bins <= 0)[0][-1]  # Find the bin that contains 0
        if zero_bin_index < len(patches):
            patches[zero_bin_index].set_facecolor("#C03D3E")
            # patches[zero_bin_index].set_alpha(1.0)  # Slightly transparent

        # Add mean and median lines
        plt.axvline(
            print_column.mean(),
            color="black",
            linestyle="dashed",
            linewidth=1,
            label=f"Mean: {print_column.mean():.2f}",
        )
        plt.legend()

        # Customize plot appearance
        column = "Mobile_App_Txns"
        plt.title(f"Distribution of Mobile_App_Txns", fontsize=14)
        plt.xlabel(column, fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.grid(axis="y", alpha=0.5)
        plt.xticks(
            rotation=45 if not pd.api.types.is_numeric_dtype(print_column) else 0
        )
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Could not create histogram for {column}: {str(e)}")
        plt.close()


def nice_print_mobile_txns(data: pd.DataFrame):

    data = filter_data(data, data["Mobile_App_Registered"] == True)
    print_column = data.loc[:, "Mobile_App_Txns"]

    try:
        plt.figure(figsize=(10, 6))

        # First plot the histogram to get bin information
        n, bins, patches = plt.hist(
            print_column, bins=30, edgecolor="black", color="skyblue"
        )

        # Close the figure to start fresh (we'll replot)
        plt.close()

        # Recreate figure with proper highlighting
        plt.figure(figsize=(10, 6))

        # Create custom bins that ensure 0 gets its own bin if present
        if 0 in print_column.value_counts().index:
            # Find the smallest positive value to set bin edge
            min_positive = print_column[print_column > 0].min()
            custom_bins = np.concatenate(
                [
                    [-0.5],  # Catch all negative values (if any)
                    [0.5],  # This creates a bin just for 0
                    np.linspace(min_positive, print_column.max(), 28),
                    [print_column.max() + 1],  # Final bin edge
                ]
            )

            # Plot with custom bins
            n, bins, patches = plt.hist(
                print_column, bins=custom_bins, edgecolor="black", color="skyblue"
            )

            # Highlight only the exact 0 bin
            zero_bin_index = np.searchsorted(bins, 0) - 1
            if zero_bin_index < len(patches):
                patches[zero_bin_index].set_facecolor("#C03D3E")
        else:
            # If no zeros exist, plot normally
            plt.hist(print_column, bins=30, edgecolor="black", color="skyblue")

        # Add mean and median lines
        plt.axvline(
            print_column.mean(),
            color="green",
            linestyle="dashed",
            linewidth=1,
            label=f"Mean: {print_column.mean():.2f}",
        )
        plt.legend()

        # Customize plot appearance
        plt.title(f"Distribution of Mobile_App_Txns", fontsize=14)
        plt.xlabel("Number of Transactions", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.grid(axis="y", alpha=0.5)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Could not create histogram: {str(e)}")
        plt.close()
