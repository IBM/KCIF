import numpy as np
import pandas as pd

STRICT_EM_COUNT_COLUMN_NAME = "instr_strict_em_count"
LOOSE_EM_COUNT_COLUMN_NAME = "instr_loose_em_count"
INSTANCES_COUNT_COLUMN_NAME = "instr_count"

IGNORE_ROWS = ["print_correct_answer", "print_correct_answer_label"]


def micro_avg_em(instr_df: pd.DataFrame, category: str = "strict"):
    """
    Compute micro-average of exact match scores over all instances of every instruction type

    Parameters:
        instr_df (pd.DataFrame): Metrics computed for every instruction
        category (str): Optional flag to compute the loose benchmark score. Defaults to strict

    Return:
        float: micro average exact match score
    """
    df = instr_df[~instr_df["instruction"].isin(IGNORE_ROWS)]
    col_name = (
        STRICT_EM_COUNT_COLUMN_NAME
        if category == "strict"
        else LOOSE_EM_COUNT_COLUMN_NAME
    )
    numerator = df[col_name].sum()
    denominator = df[INSTANCES_COUNT_COLUMN_NAME].sum()
    return numerator / denominator


def weighted_micro_avg_instruction(class_df: pd.DataFrame, category: str = "strict"):
    """
    Compute instruction category score as arithmetic mean of the
    micro-average exact-match scores for every instance per instruction category

    Parameters:
        class_df (pd.DataFrame): Metrics computed for every instruction category
        category (str): Optional flag to compute the loose benchmark score. Defaults to strict

    Return:
        float: Instruction category score
    """
    df = class_df[~class_df["classification"].isin(IGNORE_ROWS)]
    col_name = (
        STRICT_EM_COUNT_COLUMN_NAME
        if category == "strict"
        else LOOSE_EM_COUNT_COLUMN_NAME
    )
    avg_scores = []
    for _id, row in df.iterrows():
        curr_avg = row[col_name] / row[INSTANCES_COUNT_COLUMN_NAME]
        avg_scores.append(curr_avg)
    return np.mean(avg_scores)


def weighted_em_knowledge_task(class_data_df: pd.DataFrame, category: str = "strict"):
    """
    Compute knowledge task subset score as arithmetic mean of the
    micro-average exact-match scores for every instance per knowledge-task

    Parameters:
        class_data_df (pd.DataFrame): Metrics computed for every knowledge task per instruction category
        category (str): Optional flag to compute the loose benchmark score. Defaults to strict

    Return:
        float: Knowledge task subset score
    """
    df = class_data_df[~class_data_df["classification"].isin(IGNORE_ROWS)]
    col_name = (
        STRICT_EM_COUNT_COLUMN_NAME
        if category == "strict"
        else LOOSE_EM_COUNT_COLUMN_NAME
    )
    # Group by 'dataset' column
    grouped = df.groupby("dataset")
    avg_scores = []
    for group_name, group_data in grouped:
        curr_avg = (
            group_data[col_name].sum() / group_data[INSTANCES_COUNT_COLUMN_NAME].sum()
        )
        avg_scores.append(curr_avg)
    return np.mean(avg_scores)


def compute_benchmark_scores(
    model_name: str,
    instr_df: pd.DataFrame,
    noif_instr_df: pd.DataFrame,
    class_df: pd.DataFrame,
    class_data_df: pd.DataFrame,
    category: str = "strict",
):
    """
    Computes the benchmark score for a given model

    Parameters:
        model_name (str): Model for which to compute the benchmark score
        instr_df (pd.DataFrame): Metrics computed for every instruction
        noif_instr_df (pd.DataFrame): No instruction following metrics computed for every instruction
        class_df (pd.DataFrame): Metrics computed for every instruction category
        class_data_df (pd.DataFrame): Metrics computed for every knowledge task per instruction category
        category (str): Optional flag to compute loose/strict benchmark score. Defaults to strict

    Return:
        list: List of 4 scores to use for benchmarking the model
    """
    scores = [model_name]

    scores.append(micro_avg_em(instr_df, category))
    scores.append(weighted_micro_avg_instruction(class_df, category))
    scores.append(weighted_em_knowledge_task(class_data_df, category))
    scores.append(micro_avg_em(noif_instr_df, category))
    return scores


def write_benchmark_results(benchmark_scores: list, output_file: str):
    """
    Compute the average benchmark score, sort the models and write the benchmark results.

    Parameters:
        benchmark_scores (list): Each element contains the 4 scores obtained per model
        output_file (str): Output file path for the benchmark results
    """
    columns = [
        "Models",
        "Micro-Avg EM",
        "Instr. Category EM",
        "Knowledge Task EM",
        "Micro-Avg EM(No Follow)",
    ]
    benchmark_df = pd.DataFrame(benchmark_scores, columns=columns)
    benchmark_df["Avg scores"] = benchmark_df[list(benchmark_df.columns)[1:]].mean(
        axis=1
    )
    sorted_df = benchmark_df.sort_values(by="Avg scores", ascending=False)
    sorted_df.to_excel(output_file, index=False)
    return
