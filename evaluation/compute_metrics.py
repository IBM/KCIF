import argparse
import copy
import json
import string
from pathlib import Path
import re

import pandas as pd
from Levenshtein import distance
from tqdm import tqdm
import traceback

import evaluation.metric_utils as metric_utils
from evaluation.benchmark import compute_benchmark_scores, write_benchmark_results

LEVENSHTEIN_DISTANCE_THRESHOLD = 2
LEVENSHTEIN_ERROR_SETS_DISTANCE_THRESHOLD = 4
REASONING_OR_IF_CANDIDATE_LENGTH_THRESHOLD = 6
LEVENSHTEIN_WEIGHTS = (1, 1, 2)


def postprocess_output(prediction: str):
    """
    Post-process ground truth or prediction strings.

    Parameters:
        prediction (str): String corresponding to ground truth or LLM prediction

    Return:
        str: post-processed output
    """
    llmoutput = str(copy.deepcopy(prediction))
    llmoutput = llmoutput.replace("[", "")
    llmoutput = llmoutput.replace("]", "")
    llmoutput = llmoutput.replace("$", "")
    llmoutput = llmoutput.replace("'", "")
    if llmoutput.endswith("."):
        llmoutput = llmoutput[:-1]
    llmoutput = llmoutput.strip()
    return llmoutput


def em_parse_llm_output(test_instance: dict):
    """
    Strict and loose parsing of the LLM output for Exact Match computation

    Parameters:
        test_instance (dict): Dictionary corresponding to a single inference test instance

    Return:
        Tuple(str, str): strict and loose LLM prediction strings for exact match computation
    """
    if "Response:" not in test_instance["output"]:
        strict_response_prediction = postprocess_output(test_instance["output"].strip())
        truncated_output = test_instance["output"].rsplit("\n", 1)[-1]
        if "answer is:" in truncated_output:
            truncated_output = truncated_output.rsplit("answer is:", 1)[-1]
        elif "answer is :" in truncated_output:
            truncated_output = truncated_output.rsplit("answer is :", 1)[-1]
        loose_response_prediction = postprocess_output(truncated_output.strip())
    else:
        strict_response_prediction = postprocess_output(
            test_instance["output"].split("Response:")[-1].strip()
        )
        loose_response_prediction = strict_response_prediction
    return strict_response_prediction, loose_response_prediction


def analysis_parse_llm_output(test_instance: dict):
    """
    Strict and loose parsing of the LLM output for analyzing errors

    Parameters:
        test_instance (dict): Dictionary corresponding to a single inference test instance

    Return:
        Tuple(str, str): strict and loose LLM prediction strings for error analysis
    """
    if "Response:" not in test_instance["output"]:
        if "answer is" in test_instance["output"]:
            truncated_output = (
                test_instance["output"].strip().rsplit("answer is", 1)[-1]
            )
            truncated_output = truncated_output.replace(":", "").strip()
        elif ":" in test_instance["output"]:
            truncated_output = test_instance["output"].strip().rsplit(":", 1)[-1]
        elif "\n" in test_instance["output"]:
            truncated_output = test_instance["output"].strip().rsplit("\n", 1)[-1]
        else:
            truncated_output = test_instance["output"].strip()
        strict_response_prediction = postprocess_output(truncated_output.strip())
        loose_response_prediction = strict_response_prediction
    else:
        strict_response_prediction = postprocess_output(
            test_instance["output"].split("Response:")[-1].strip()
        )
        loose_response_prediction = postprocess_output(test_instance["output"].strip())
    return strict_response_prediction, loose_response_prediction


def parse_llm_output(test_instance: dict):
    if "output" not in test_instance:
        return ""
    response = test_instance["output"]
    if any(resp in response for resp in ["Response:", "response:"]):
        response = re.split("Response:|response:", test_instance["output"])[-1].strip()
    elif any(resp.lower() in response for resp in ["answer is:", "answer is :"]):
        response = re.split(
            "answer is:|answer is :|Answer is:|Answer is :", test_instance["output"]
        )[-1].strip()
    elif "\n" in response:
        response = test_instance["output"].rsplit("\n", 1)[-1]

    response = postprocess_output(response)
    return response


def compare_removing_whitespace(ground_truth, prediction):
    """
    String comparison after removing whitespaces

    Parameters:
        ground_truth (str): Ground truth
        prediction (str): LLM prediction

    Return:
        Bool: 0 or 1 corresponding to match
    """
    return ground_truth.translate(
        str.maketrans("", "", string.whitespace)
    ) == prediction.translate(str.maketrans("", "", string.whitespace))


def compute_exact_match(
    em_metrics: dict,
    test_instance: dict,
    pca_instance: dict,
    pca_label_instance: dict,
):
    """
    Function to compute exact match for each instance.
    Exact match is computed for the applied instruction,
    along with the corresponding print_correct_answer and print_correct_answer_label

    Parameters:
        em_metrics (dict): Dictionary to store metrics
        test_instance (dict): Dictionary corresponding to a single inference test instance
        pca_instance (dict): print_correct_answer instance of current test utterance
        pca_label_instance (dict): print_correct_answer_label instance of current test utterance

    Return:
        dict: updated em_metrics dict with computed exact match metrics
    """
    em_metrics["strict"] = 0
    em_metrics["loose"] = 0
    em_metrics["pca_strict"] = 0
    em_metrics["pca_loose"] = 0
    em_metrics["pca_label_strict"] = 0
    em_metrics["pca_label_loose"] = 0

    ground_truth = postprocess_output(test_instance["instruction_output"][-1])
    prediction = parse_llm_output(test_instance)

    if (prediction == ground_truth) or compare_removing_whitespace(
        ground_truth, prediction
    ):
        em_metrics["strict"] = em_metrics["loose"] = 1

    if pca_instance:
        pca_gt = (
            postprocess_output(str(pca_instance["instruction_output"][-1]))
            if pca_instance
            else ""
        )
        pca_pred = parse_llm_output(pca_instance)
        if (pca_pred == pca_gt) or compare_removing_whitespace(pca_gt, pca_pred):
            em_metrics["pca_strict"] = em_metrics["pca_loose"] = 1

    if pca_label_instance:
        pca_label_gt = (
            postprocess_output(str(pca_label_instance["instruction_output"][-1]))
            if pca_label_instance
            else ""
        )
        pca_label_pred = parse_llm_output(pca_label_instance)
        if (pca_label_pred == pca_label_gt) or compare_removing_whitespace(
            pca_label_gt, pca_label_pred
        ):
            em_metrics["pca_label_strict"] = em_metrics["pca_label_loose"] = 1

    return em_metrics


def compute_levenshtein_based_match(pred, gt):
    """
    Compute a levenshtein distance based exact match (loose matching)
    We prioritize addition and removal over updates (i.e.) LEVENSHTEIN_WEIGHTS

    Parameters:
        pred (str): post-processed prediction
        gt (str): post-processed ground truth

    Return:
        Bool: 0 or 1 based on match.
    """
    loose_em = 0
    levenshtein_distance = distance(pred, gt, weights=LEVENSHTEIN_WEIGHTS)
    levenshtein_distance_no_space = distance(
        pred.replace(" ", ""), gt.replace(" ", ""), weights=LEVENSHTEIN_WEIGHTS
    )
    if (
        levenshtein_distance <= LEVENSHTEIN_DISTANCE_THRESHOLD
        or levenshtein_distance_no_space <= LEVENSHTEIN_DISTANCE_THRESHOLD
    ):
        loose_em = 1
    return loose_em


def compute_reasoning_analysis_levenshtein_based_match(pred, gt):
    """
    Compute a levenshtein distance based error analysis match (loose matching)
    Differs from exact match distance in terms of distance threshold.
    We prioritize addition and removal over updates (i.e.) LEVENSHTEIN_WEIGHTS

    Parameters:
        pred (str): post-processed prediction
        gt (str): post-processed ground truth

    Return:
        Bool: 0 or 1 based on match.
    """
    loose_em = 0
    levenshtein_distance = distance(pred, gt, weights=LEVENSHTEIN_WEIGHTS)
    levenshtein_distance_no_space = distance(
        pred.replace(" ", ""), gt.replace(" ", ""), weights=LEVENSHTEIN_WEIGHTS
    )
    if (
        levenshtein_distance <= LEVENSHTEIN_ERROR_SETS_DISTANCE_THRESHOLD
        or levenshtein_distance_no_space <= LEVENSHTEIN_ERROR_SETS_DISTANCE_THRESHOLD
    ):
        loose_em = 1
    return loose_em


def add_missing_labels_in_if_error(test_instance):
    """
    Function to add more instruction following error candidates during runtime,
    without needing to re-run inference.
    Note: Can be expanded in case other candidate possibilities are missing.
    Currently:
    -- Dynamically add in if_error_set :
    label + instruction_output[-1] ,
    label + " " + instruction_output[-1] ,
    label + "." + instruction_output[-1]
    label + ". " + instruction_output[-1]
    Parameters:
        test_instance (dict): Dictionary corresponding to a single inference test instance

    Return:
        list: new instruction following error candidates as a list
    """
    new_if_errors = []
    new_if_errors.append(
        str(test_instance["ground_truth_answer_label"]).strip()
        + str(test_instance["instruction_output"][-1]).strip()
    )
    new_if_errors.append(
        str(test_instance["ground_truth_answer_label"]).strip()
        + " "
        + str(test_instance["instruction_output"][-1]).strip()
    )
    new_if_errors.append(
        str(test_instance["ground_truth_answer_label"]).strip()
        + "."
        + str(test_instance["instruction_output"][-1]).strip()
    )
    new_if_errors.append(
        str(test_instance["ground_truth_answer_label"]).strip()
        + ". "
        + str(test_instance["instruction_output"][-1]).strip()
    )

    return new_if_errors


def compute_reasoning_if_errors(
    analysis_metrics: dict,
    test_instance: dict,
    pca_instance: dict,
    pca_label_instance: dict,
    em: int,
):
    """
    Function to compute error analysis for each instance.
    Computes reasoning, instruction following, and unclassified errors.
    This is computed for the applied instruction,
    along with the corresponding print_correct_answer and print_correct_answer_label

    Parameters:
        analysis_metrics (dict): Dictionary to store metrics
        test_instance (dict): Dictionary corresponding to a single inference test instance
        pca_instance (dict): print_correct_answer instance of current test utterance
        pca_label_instance (dict): print_correct_answer_label instance of current test utterance
        em (int): Exact match value

    Return:
        dict: updated analysis_metrics dict with computed error analysis metrics
    """
    analysis_metrics["reason_strict"] = 0
    analysis_metrics["reason_loose"] = 0
    analysis_metrics["if_strict"] = 0
    analysis_metrics["if_loose"] = 0
    analysis_metrics["unclass_strict"] = 0
    analysis_metrics["unclass_loose"] = 0

    analysis_metrics["pca_reason"] = 0
    analysis_metrics["pca_if"] = 0
    analysis_metrics["pca_unclass"] = 0
    analysis_metrics["pca_label_reason"] = 0
    analysis_metrics["pca_label_if"] = 0
    analysis_metrics["pca_label_unclass"] = 0

    ground_truth = postprocess_output(test_instance["instruction_output"][-1])
    prediction = parse_llm_output(test_instance)

    key_val = (
        test_instance["dataset"]
        + "_"
        + test_instance["dataset_input"].strip()
        + "_"
        + test_instance["instruction_id"]
    )
    reasoning_set = test_instance["reasoning_error_set"]
    if_set = test_instance["instruction_following_errors_set"]

    reasoning_set = list(map(postprocess_output, reasoning_set))
    if_set.extend(add_missing_labels_in_if_error(test_instance))
    if_set = list(map(postprocess_output, if_set))

    if em == 0:
        for candidate in reasoning_set:
            if len(candidate) == 1 and candidate.isalpha():
                if any(
                    opt in prediction
                    for opt in [candidate, f"{candidate}.", f"'{candidate}'"]
                ):
                    analysis_metrics["reason_strict"] = analysis_metrics[
                        "reason_loose"
                    ] = 1
                    break
            else:
                if len(candidate) > REASONING_OR_IF_CANDIDATE_LENGTH_THRESHOLD:
                    analysis_metrics["reason_strict"] = analysis_metrics[
                        "reason_loose"
                    ] = compute_reasoning_analysis_levenshtein_based_match(
                        prediction, candidate
                    )
                if (prediction == candidate) or compare_removing_whitespace(
                    candidate, prediction
                ):
                    analysis_metrics["reason_strict"] = analysis_metrics[
                        "reason_loose"
                    ] = 1
                if analysis_metrics["reason_strict"] == 1:
                    break
        for candidate in if_set:
            if len(candidate) == 1 and candidate.isalpha():
                if any(
                    opt in prediction
                    for opt in [candidate, f"{candidate}.", f"'{candidate}'"]
                ):
                    analysis_metrics["if_strict"] = analysis_metrics["if_loose"] = 1
                    break
            else:
                if len(candidate) > REASONING_OR_IF_CANDIDATE_LENGTH_THRESHOLD:
                    analysis_metrics["if_strict"] = analysis_metrics["if_loose"] = (
                        compute_reasoning_analysis_levenshtein_based_match(
                            prediction, candidate
                        )
                    )
                if (prediction == candidate) or compare_removing_whitespace(
                    candidate, prediction
                ):
                    analysis_metrics["if_strict"] = analysis_metrics["if_loose"] = 1
                if analysis_metrics["if_strict"] == 1:
                    break
        if (
            analysis_metrics["reason_strict"] == 0
            and analysis_metrics["if_strict"] == 0
        ):
            analysis_metrics["unclass_strict"] = analysis_metrics["unclass_loose"] = 1

    if pca_instance and em == 0:
        pca_pred = parse_llm_output(pca_instance)
        pca_reasoning_set = pca_instance["reasoning_error_set"]
        pca_if_set = pca_instance["instruction_following_errors_set"]
        pca_reasoning_set = list(map(postprocess_output, pca_reasoning_set))
        pca_if_set = list(map(postprocess_output, pca_if_set))
        for candidate in pca_reasoning_set:
            if len(candidate) > REASONING_OR_IF_CANDIDATE_LENGTH_THRESHOLD:
                analysis_metrics["pca_reason"] = (
                    compute_reasoning_analysis_levenshtein_based_match(
                        pca_pred, candidate
                    )
                )
            if candidate in pca_pred:
                analysis_metrics["pca_reason"] = 1
            if analysis_metrics["pca_reason"] == 1:
                break
        for candidate in pca_if_set:
            if len(candidate) > REASONING_OR_IF_CANDIDATE_LENGTH_THRESHOLD:
                analysis_metrics["pca_if"] = (
                    compute_reasoning_analysis_levenshtein_based_match(
                        pca_pred, candidate
                    )
                )
            if candidate in pca_pred:
                analysis_metrics["pca_if"] = 1
            if analysis_metrics["pca_if"] == 1:
                break
        if analysis_metrics["pca_reason"] == 0 and analysis_metrics["pca_if"] == 0:
            analysis_metrics["pca_unclass"] = 1

    return analysis_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Json config file, list of dictionaries",
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        required=True,
        help="Output results folder path"
    )
    parser.add_argument(
        "-l",
        "--loose",
        action=argparse.BooleanOptionalAction,
        help="Compute benchmark using loose scores"
    )
    args = parser.parse_args()
    with open(str(args.config)) as f:
        config_list = json.load(f)

    final_benchmark_results = []
    for config_entry in tqdm(config_list, desc="Processing config.."):
        assert all(
            k in config_entry for k in ["if_filepath", "noif_filepath", "model"]
        ), "Missing key in config"
        benchmark_dict = {}
        model_name = config_entry["model"]
        for idx, file_path in enumerate(
            [config_entry["if_filepath"], config_entry["noif_filepath"]]
        ):
            filename = "if" if idx == 0 else "noif"
            output_file = (
                str(args.output_folder) + "/" + model_name + "_" + filename + ".xlsx"
            )

            with open(file_path) as f:
                inference_data = [json.loads(line) for line in f]
            pca_instances_dict = {}
            pca_label_instances_dict = {}
            for instance in inference_data:
                if instance["instruction_id"] not in [
                    "print_correct_answer",
                    "print_correct_answer_label",
                ]:
                    continue
                if instance["instruction_id"] == "print_correct_answer":
                    if instance["dataset_input"] not in pca_instances_dict:
                        pca_instances_dict[instance["dataset_input"]] = instance
                elif instance["instruction_id"] == "print_correct_answer_label":
                    if instance["dataset_input"] not in pca_label_instances_dict:
                        pca_label_instances_dict[instance["dataset_input"]] = instance

            print(f"processing file --- {file_path}")
            with open(file_path) as f:
                data = [json.loads(line) for line in f]

            dataset_metrics = metric_utils.DatasetMetric(data)
            instruction_metrics = metric_utils.InstructionMetric(data)
            classification_metrics = metric_utils.ClassificationMetric(data)
            data_instr_metrics = metric_utils.DatasetInstrMetric(data)
            class_instr_metrics = metric_utils.ClassificationInstrMetric(data)
            class_data_metrics = metric_utils.ClassificationDatasetMetric(data)
            class_data_noif_metrics = metric_utils.ClassificationDatasetNoIFMetric(data)

            category_mapping = metric_utils.CATEGORY_MAPPING()
            datasets = metric_utils.DATASETS().datasets

            for test_instance in data:
                try:
                    current_dataset = test_instance["dataset"]
                    current_instruction = test_instance["instruction_id"]
                    current_class = category_mapping.mapping[current_instruction]

                    if (
                        current_dataset in ["BoolQ", "Winogrande", "Piqa"]
                        and current_class == "Operations on List"
                    ):
                        continue

                    current_parent_dataset = ""
                    current_dataset_input = test_instance["dataset_input"]
                    print_correct_answer_instance = pca_instances_dict.get(
                        current_dataset_input, {}
                    )
                    print_correct_answer_label_instance = pca_label_instances_dict.get(
                        current_dataset_input, {}
                    )

                    for dset in datasets:
                        if dset.lower() in current_dataset.lower():
                            current_parent_dataset = dset
                            break
                    assert current_parent_dataset != ""

                    em_metrics, analysis_metrics = {}, {}
                    em_metrics["pca_count"] = (
                        0 if print_correct_answer_instance == {} else 1
                    )
                    em_metrics["pca_label_count"] = (
                        0 if print_correct_answer_label_instance == {} else 1
                    )

                    em_metrics = compute_exact_match(
                        em_metrics,
                        test_instance,
                        print_correct_answer_instance,
                        print_correct_answer_label_instance,
                    )
                    analysis_metrics = compute_reasoning_if_errors(
                        analysis_metrics,
                        test_instance,
                        print_correct_answer_instance,
                        print_correct_answer_label_instance,
                        em_metrics["loose"],
                    )

                    dataset_metrics.update_metrics(
                        current_dataset, em_metrics, analysis_metrics
                    )
                    instruction_metrics.update_metrics(
                        current_instruction, em_metrics, analysis_metrics
                    )
                    classification_metrics.update_metrics(
                        current_class, current_instruction, em_metrics, analysis_metrics
                    )
                    data_instr_metrics.update_metrics(
                        current_dataset,
                        current_instruction,
                        em_metrics,
                        analysis_metrics,
                    )
                    class_instr_metrics.update_metrics(
                        current_class, current_instruction, em_metrics, analysis_metrics
                    )
                    class_data_metrics.update_metrics(
                        current_class,
                        current_parent_dataset,
                        current_instruction,
                        em_metrics,
                        analysis_metrics,
                    )
                    class_data_noif_metrics.update_metrics(
                        current_class,
                        current_parent_dataset,
                        current_instruction,
                        em_metrics,
                        analysis_metrics,
                    )
                except Exception as e:
                    traceback.print_exc()
                    continue

            try:
                dataset_csv = dataset_metrics.return_metrics()
                instruction_csv = instruction_metrics.return_metrics()
                classification_csv = classification_metrics.return_metrics()
                data_instr_csv = data_instr_metrics.return_metrics()
                class_instr_csv = class_instr_metrics.return_metrics()
                class_data_csv = class_data_metrics.return_metrics()
                class_data_noif_csv = class_data_noif_metrics.return_metrics()

                instr_df, class_df, class_data_df = metric_utils.write_result_xlsx(
                    output_file,
                    dataset_csv,
                    instruction_csv,
                    classification_csv,
                    data_instr_csv,
                    class_instr_csv,
                    class_data_csv,
                    class_data_noif_csv,
                )
                if idx == 0:
                    (
                        benchmark_dict["if_instr_df"],
                        benchmark_dict["if_class_df"],
                        benchmark_dict["if_class_data_df"],
                    ) = (instr_df, class_df, class_data_df)
                else:
                    benchmark_dict["noif_instr_df"] = instr_df
            except Exception as e:
                traceback.print_exc()
                raise
        try:
            if args.loose == True:
                benchmark_category = "loose"
            else:
                benchmark_category = "strict"
            final_benchmark_results.append(
                compute_benchmark_scores(
                    model_name,
                    benchmark_dict["if_instr_df"],
                    benchmark_dict["noif_instr_df"],
                    benchmark_dict["if_class_df"],
                    benchmark_dict["if_class_data_df"],
                    benchmark_category,
                )
            )
        except Exception as e:
            traceback.print_exc()
            print(f"Unable to compute benchmark score for {model_name}")
            raise

    benchmark_path = str(args.output_folder) + "/" + "benchmark.xlsx"
    write_benchmark_results(final_benchmark_results, benchmark_path)


if __name__ == "__main__":
    main()
