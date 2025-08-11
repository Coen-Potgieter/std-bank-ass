import pandas as pd
import numpy as np
from seaborn import cubehelix_palette
from tabulate import tabulate
from helper_funcs import (
    print_heading,
    get_cardinality,
    prepare_directory,
    save_data_quality_report,
    get_mode_info,
    get_numerical_stats,
    plot_histograms,
    plot_categorical_bars,
    sepearate_mobile_online_offline,
    inspect_low_age_records,
    clean_age_anomalies,
    explore_non_active_mobile_users,
    explore_online_vs_mobile_banking,
    nice_print_mobile_txns,
)


def data_quality_report(data: pd.DataFrame, dir_path: str, is_cat: bool):
    """
    Creates DQR, and puts all data in given directory

    Params:
        - `data`: csv data
        - `dir_path`: Directory where both DQR table and graphs will be stored
        - `is_cat`: True for categorical data, False for continuous data
    """

    # Prepare Directory
    print_heading(f"Preparing Given Directory `{dir_path}`")
    graphs_dir_path, tables_dir_path = prepare_directory(dir_path)

    # Get Data
    print_heading("Creating DQR")
    card_counts = get_cardinality(data, debug=False)
    null_counts = pd.isna(data).sum(axis=0)
    additional_data = get_mode_info(data) if is_cat else get_numerical_stats(data)

    # Create Table
    dqr_column_names = ["Feature", "Count", "Missing %", "Cardinality"]
    data_quality_report = {name: [] for name in dqr_column_names}

    num_records = data.shape[0]
    for name in data.columns.tolist():
        null_percentage = round((null_counts[name] / num_records) * 100, 2)
        data_quality_report["Feature"].append(name)
        data_quality_report["Count"].append(num_records)
        data_quality_report["Missing %"].append(null_percentage)
        data_quality_report["Cardinality"].append(card_counts[name])

    # Print DQR
    df_dqr = pd.DataFrame(data_quality_report).join(additional_data)
    print(tabulate(df_dqr, headers="keys", tablefmt="psql", showindex=False))

    # Save DQR
    print_heading("Saving DQR")
    save_data_quality_report(df_dqr, tables_dir_path)

    # Save Plots
    print_heading("Creating Plots")

    if is_cat:
        plot_categorical_bars(data, graphs_dir_path)
    else:
        plot_histograms(data, graphs_dir_path)


def main():
    csv_path = "./data/SBSA_DS_2026.csv"
    data = pd.read_csv(csv_path)

    data = clean_age_anomalies(data, debug=True)

    # inspect_low_age_records(data, cutoff_age=18)

    # explore_online_vs_mobile_banking(data)
    nice_print_mobile_txns(data)
    explore_non_active_mobile_users(data)
    # sepearate_mobile_online_offline(data)
    return
    return

    # sepearate_mobile_online_offline(data)
    explore_online_vs_mobile_banking(data)
    return

    return
    categorical_columns = [
        "Employment_Type",
        "Current_Account",
        "Savings_Account",
        "Credit_Card",
        "Personal_Loan",
        "Home_Loan",
        "Vehicle_Loan",
        "Online_Banking_Registered",
        "Mobile_App_Registered",
        "Suburb",
        "City",
        "Province",
    ]

    continuous_columns = [
        "Age",
        "Annual_Income_Estimate",
        "Current_Account_Balance",
        "Savings_Account_Balance",
        "Credit_Card_Balance",
        "Personal_Loan_Balance",
        "Home_Loan_Balance",
        "Vehicle_Loan_Balance",
        "Online_Banking_Txns",
        "Mobile_App_Txns",
        "Credit_Score",
    ]

    data_quality_report(
        data.loc[:, categorical_columns], "./output/dqr/cat/", is_cat=True
    )

    # data_quality_report(
    #     data.loc[:, continuous_columns], "./output/dqr/cts/", is_cat=False
    # )
    return
    get_sample_vals(data.iloc[:, 0], 10, True)
    return
    # counts = get_cardinality(data, True)
    types = data.info()
    print(data.dtypes)
    return

    # handle_missing_vals(data)
    # return

    # Get dataframe info (data types, non-null counts)
    # print(data.info())

    target_columns = data.columns[0:5]

    # print(target_columns)

    target_data = data.iloc[:, 1:5]
    print(target_data.describe())
    # print(data[-1].describe(include="all"))

    return

    column_data = get_column_data(data, "CustomerID")
    print(type(column_data))

    return
    column_names = list(data.columns)

    for idx, name in enumerate(column_names):
        print(f"Feature {idx + 1}: {name}")


if __name__ == "__main__":
    main()
