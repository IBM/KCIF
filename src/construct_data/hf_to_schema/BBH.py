from datasets import load_dataset, concatenate_datasets, Dataset
from ..constants import create_schema, SCHEMA_KEYS
import re


def regex_match_option_label_text(input: str, candidate_list: list):
    option_text_mapping = {}
    for option in candidate_list:
        target_answer_letter = option.strip().replace("(", "")
        target_answer_letter = target_answer_letter.replace(")", "")
        string_corresponding_to_option = re.search(
            rf"\({target_answer_letter}\).*([\n\(]|$)", input
        )
        if not string_corresponding_to_option:
            continue
        string_corresponding_to_option = string_corresponding_to_option.group(0).strip()
        string_corresponding_to_option = re.sub(
            r"\(.*\)", "", string_corresponding_to_option
        ).strip()
        option_text_mapping[option] = string_corresponding_to_option
    return option_text_mapping


def preprocess_task_input(
    test_instance: dict, task_parameters: dict, BBH_OPTIONS: list
):
    task_input = test_instance["input"]
    # First truncate the input and remove the options
    if "Options:" in test_instance["input"]:
        task_input = test_instance["input"].split("Options:")[0].strip()

    candidate_answer_set = [""]
    candidate_label_space = [""]
    option_candidate_text_mapping = {}
    if len(task_parameters["candidate_answers"]) > 0:
        if "(A)" in task_parameters["candidate_answers"]:
            option_candidate_text_mapping = regex_match_option_label_text(
                test_instance["input"], task_parameters["candidate_answers"]
            )
        if option_candidate_text_mapping:
            candidate_label_space = list(option_candidate_text_mapping.keys())
            candidate_answer_set = list(option_candidate_text_mapping.values())
        else:
            candidate_label_space = (
                BBH_OPTIONS[: len(task_parameters["candidate_answers"])]
                if len(task_parameters["candidate_answers"]) > 0
                else []
            )
            candidate_answer_set = task_parameters["candidate_answers"]

    if len(task_parameters["candidate_answers"]) == 0:
        ground_truth_text = test_instance["target"]
        ground_truth_answer_label = ""
    else:
        if test_instance["target"].strip().lower() in [
            "true",
            "false",
            "yes",
            "no",
            "valid",
            "invalid",
        ]:
            ground_truth_text = test_instance["target"]
            gt_label_index = task_parameters["candidate_answers"].index(
                test_instance["target"].strip()
            )
            ground_truth_answer_label = candidate_label_space[gt_label_index]
        else:
            ground_truth_text = option_candidate_text_mapping[
                test_instance["target"].strip()
            ]
            ground_truth_answer_label = test_instance["target"].strip()

    return (
        task_input,
        candidate_label_space,
        candidate_answer_set,
        ground_truth_text,
        ground_truth_answer_label,
    )


def transform_bbh(
    test_instance: dict, task_parameters: dict, task_name: str, BBH_OPTIONS: list
):
    schema = create_schema()
    schema[SCHEMA_KEYS.DATA_SET.value] = "BBH_" + task_name
    schema[SCHEMA_KEYS.HF_DATA_NAME.value] = (
        "https://huggingface.co/datasets/lukaemon/bbh"
    )

    schema[SCHEMA_KEYS.TASK_TYPE.value] = (
        "MCQ" if len(task_parameters["candidate_answers"]) > 0 else "Generative"
    )
    try:
        (
            task_input,
            candidate_label_space,
            candidate_answer_set,
            ground_truth_text,
            ground_truth_answer_label,
        ) = preprocess_task_input(test_instance, task_parameters, BBH_OPTIONS)
    except KeyError:
        schema[SCHEMA_KEYS.TASK_TYPE.value] = "invalid"
        return schema
    if len(task_parameters["candidate_answers"]) > 0:
        schema[SCHEMA_KEYS.INPUT_INSTANCE.value] = f"Question: {task_input}\nOptions:"
    else:
        schema[SCHEMA_KEYS.INPUT_INSTANCE.value] = f"Question: {task_input}\n"

    schema[SCHEMA_KEYS.CANDIDATE_ANSWER_SET.value] = candidate_answer_set
    schema[SCHEMA_KEYS.CANDIDATE_ANSWER_LABEL_SPACE.value] = candidate_label_space

    schema[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_LABEL.value] = ground_truth_answer_label
    schema[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_TEXT.value] = ground_truth_text

    # The instruction prefix which gets appended
    if len(candidate_label_space) > 0 and candidate_label_space != [""]:
        schema[SCHEMA_KEYS.TASK_PROMPT.value] = (
            f"Given a question and {len(candidate_label_space)} options namely {', '.join(candidate_label_space)} as candidate answers, "
        )
        schema[SCHEMA_KEYS.FINAL_PREFIX_TASK_INSTRUCTION.value] = (
            f"Given a question and {len(candidate_label_space)} options namely {', '.join(candidate_label_space)} as candidate answers, answer the question by selecting the value associated with the option label corresponding to the correct answer.\n"
        )
    else:
        schema[SCHEMA_KEYS.TASK_PROMPT.value] = "Answer the given question."
        schema[SCHEMA_KEYS.FINAL_PREFIX_TASK_INSTRUCTION.value] = (
            "Answer the given question."
        )

    schema[SCHEMA_KEYS.FINAL_SUFFIX_TASK_INSTRUCTION.value] = "\n"

    if ground_truth_answer_label != "":
        schema[SCHEMA_KEYS.INSTRUCTION_OUTPUT.value] = [ground_truth_answer_label]
    else:
        schema[SCHEMA_KEYS.INSTRUCTION_OUTPUT.value] = [ground_truth_text]

    schema[SCHEMA_KEYS.TASK_INSTRUCTIONS.value].append(
        schema[SCHEMA_KEYS.TASK_PROMPT.value]
    )

    assert schema[SCHEMA_KEYS.INPUT_INSTANCE.value] != ""
    assert schema[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_TEXT.value] != ""
    if schema[SCHEMA_KEYS.TASK_TYPE.value] == "MCQ":
        assert schema[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_LABEL.value] != ""
    assert len(schema[SCHEMA_KEYS.CANDIDATE_ANSWER_SET.value]) == len(
        schema[SCHEMA_KEYS.CANDIDATE_ANSWER_LABEL_SPACE.value]
    )
    assert len(schema[SCHEMA_KEYS.CANDIDATE_ANSWER_LABEL_SPACE.value]) > 0
    assert len(schema[SCHEMA_KEYS.CANDIDATE_ANSWER_SET.value]) > 0

    assert schema[SCHEMA_KEYS.TASK_PROMPT.value] != ""
    assert len(schema[SCHEMA_KEYS.INSTRUCTION_OUTPUT.value]) > 0
    return schema


class BBH:
    BBH_OPTIONS = [
        "(A)",
        "(B)",
        "(C)",
        "(D)",
        "(E)",
        "(F)",
        "(G)",
        "(H)",
        "(I)",
        "(J)",
        "(K)",
        "(L)",
        "(M)",
        "(N)",
        "(O)",
        "(P)",
        "(Q)",
        "(R)",
    ]
    BBH_TASKS = {
        "boolean_expressions": {
            "candidate_answers": ["True", "False"],
        },
        "causal_judgement": {
            "candidate_answers": ["Yes", "No"],
        },
        "date_understanding": {
            "candidate_answers": ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)"],
        },
        "disambiguation_qa": {
            "candidate_answers": ["(A)", "(B)", "(C)"],
        },
        "dyck_languages": {
            "candidate_answers": [],
        },
        "formal_fallacies": {
            "candidate_answers": ["valid", "invalid"],
        },
        "geometric_shapes": {
            "candidate_answers": [
                "(A)",
                "(B)",
                "(C)",
                "(D)",
                "(E)",
                "(F)",
                "(G)",
                "(H)",
                "(I)",
                "(J)",
                "(K)",
            ],
        },
        "hyperbaton": {
            "candidate_answers": ["(A)", "(B)"],
        },
        "logical_deduction_five_objects": {
            "candidate_answers": ["(A)", "(B)", "(C)", "(D)", "(E)"],
        },
        "logical_deduction_seven_objects": {
            "candidate_answers": [
                "(A)",
                "(B)",
                "(C)",
                "(D)",
                "(E)",
                "(F)",
                "(G)",
            ],
        },
        "logical_deduction_three_objects": {
            "candidate_answers": ["(A)", "(B)", "(C)"],
        },
        "movie_recommendation": {
            "candidate_answers": ["(A)", "(B)", "(C)", "(D)", "(E)"],
        },
        "multistep_arithmetic_two": {
            "candidate_answers": [],
        },
        "navigate": {
            "candidate_answers": ["Yes", "No"],
        },
        "object_counting": {
            "candidate_answers": [],
        },
        "penguins_in_a_table": {
            "candidate_answers": ["(A)", "(B)", "(C)", "(D)", "(E)"],
        },
        "reasoning_about_colored_objects": {
            "candidate_answers": [
                "(A)",
                "(B)",
                "(C)",
                "(D)",
                "(E)",
                "(F)",
                "(G)",
                "(H)",
                "(I)",
                "(J)",
                "(K)",
                "(L)",
                "(M)",
                "(N)",
                "(O)",
                "(P)",
                "(Q)",
                "(R)",
            ],
        },
        "ruin_names": {
            "candidate_answers": ["(A)", "(B)", "(C)", "(D)"],
        },
        "salient_translation_error_detection": {
            "candidate_answers": ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)"],
        },
        "snarks": {
            "candidate_answers": ["(A)", "(B)"],
        },
        "sports_understanding": {
            "candidate_answers": ["yes", "no"],
        },
        "temporal_sequences": {
            "candidate_answers": ["(A)", "(B)", "(C)", "(D)"],
        },
        "tracking_shuffled_objects_five_objects": {
            "candidate_answers": ["(A)", "(B)", "(C)", "(D)", "(E)"],
        },
        "tracking_shuffled_objects_seven_objects": {
            "candidate_answers": [
                "(A)",
                "(B)",
                "(C)",
                "(D)",
                "(E)",
                "(F)",
                "(G)",
            ],
        },
        "tracking_shuffled_objects_three_objects": {
            "candidate_answers": ["(A)", "(B)", "(C)"],
        },
        "web_of_lies": {
            "candidate_answers": ["Yes", "No"],
        },
        "word_sorting": {
            "candidate_answers": [],
        },
    }

    def __init__(self):
        super().__init__()
        self.intermediate_representation = {}
        self.dataset = {}
        dataset = []
        intermediate_representations = {}
        for task_name, task_parameters in self.BBH_TASKS.items():
            task_dataset = load_dataset("lukaemon/bbh", task_name, split="test")
            filtered_dataset = []
            for test_instance in task_dataset:
                if task_parameters["candidate_answers"]:
                    # Couple of cases where the input dataset is noisy and no actual labels given
                    if (
                        task_parameters["candidate_answers"]
                        and test_instance["target"].strip()
                        in task_parameters["candidate_answers"]
                    ):
                        filtered_dataset.append(test_instance)
                else:
                    filtered_dataset.append(test_instance)
            filtered_dataset = Dataset.from_list(filtered_dataset)
            dataset.append(filtered_dataset)
            intermediate_representations["BBH_" + task_name] = filtered_dataset.map(
                transform_bbh,
                fn_kwargs={
                    "task_parameters": task_parameters,
                    "task_name": task_name,
                    "BBH_OPTIONS": self.BBH_OPTIONS,
                },
                remove_columns=task_dataset.column_names,
            )

        self.dataset = concatenate_datasets([dset for dset in dataset])
        self.intermediate_representation = intermediate_representations
