import os
import json
import argparse
import importlib
import traceback
from tqdm import tqdm
from constants import SCHEMA_KEYS
from functools import partial
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))


def create_input_from_dataset(example: dict, COT=False):
    """Transform HF dataset into the schema

    Args:
        example (dict): an instance of HF dataset
        COT (bool, optional): Use CoT prompts. Defaults to False.

    Returns:
        _type_: an instance of HF dataset transformed to our schema
    """

    if example[SCHEMA_KEYS.TASK_TYPE.value] == "MCQ":
        choices_as_text = "\n"
        for each_label_index in range(
            len(example[SCHEMA_KEYS.CANDIDATE_ANSWER_SET.value])
        ):
            choices_as_text = (
                choices_as_text
                + f"{example[SCHEMA_KEYS.CANDIDATE_ANSWER_LABEL_SPACE.value][each_label_index]}. {example[SCHEMA_KEYS.CANDIDATE_ANSWER_SET.value][each_label_index]}\n"
            )
    else:
        choices_as_text = ""
    if COT:
        example["input"] = (
            example[SCHEMA_KEYS.COT_INSTRUCTION.value].strip()
            + "\n"
            + example[SCHEMA_KEYS.INPUT_INSTANCE.value]
            + choices_as_text
            + example[SCHEMA_KEYS.FINAL_SUFFIX_TASK_INSTRUCTION.value]
        )
    else:
        example["input"] = (
            example[SCHEMA_KEYS.FINAL_PREFIX_TASK_INSTRUCTION.value].strip()
            + "\n"
            + example[SCHEMA_KEYS.INPUT_INSTANCE.value]
            + choices_as_text
            + example[SCHEMA_KEYS.FINAL_SUFFIX_TASK_INSTRUCTION.value]
        )
    return example


def get_minimal_subset(instances_by_instruction, threshold):
    """Obtain minimal subset across datasets to construct the dataset

    Args:
        instances_by_instruction (_type_): an instance of dataset in our schema
        threshold (_type_): the threshold value for the given dataset

    Returns:
        _type_: the subset of examples to consider as part of the dataset
    """
    instructions_by_frequency = {}
    for each_instruction in instances_by_instruction:
        instructions_by_frequency[each_instruction] = len(
            instances_by_instruction[each_instruction]
        )

    overlapping_keys = []

    for current_instruction, frequency in sorted(
        instructions_by_frequency.items(), key=lambda x: x[1]
    ):
        keys_from_current_instruction = list(
            instances_by_instruction[current_instruction].keys()
        )

        minimal_overlapping_key = keys_from_current_instruction + overlapping_keys
        for each_instruction in instances_by_instruction:
            if each_instruction == current_instruction:
                continue

            minimal_overlapping_key = set(list(minimal_overlapping_key)) & set(
                list(instances_by_instruction[each_instruction].keys())
            )

        # there's no overlap at all
        if len(minimal_overlapping_key) == 0:
            # if the dataset has more instances compared to threshold
            if len(keys_from_current_instruction) > threshold:
                overlapping_keys.extend(list(keys_from_current_instruction)[:threshold])
            else:
                overlapping_keys.extend(list(keys_from_current_instruction))

            overlapping_keys = list(set(overlapping_keys))
        else:
            # there is already some overlap
            if len(minimal_overlapping_key) > threshold:
                overlapping_keys.extend(list(minimal_overlapping_key)[:threshold])
            else:
                overlapping_keys.extend(list(minimal_overlapping_key))

                # find the set difference between two lists
                difference_set = set(keys_from_current_instruction) - set(
                    minimal_overlapping_key
                )
                overlapping_keys.extend(
                    list(difference_set)[: (threshold - len(minimal_overlapping_key))]
                )

            overlapping_keys = list(set(overlapping_keys))

    return overlapping_keys


def write_data(
    output_path, instances, dataset_name, instances_to_consider, cot, instruct_follow
):
    """Save Instances to file

    Args:
        output_path (_type_): path to output folder
        instances (_type_): the set of instances
        dataset_name (_type_): dataset name
        instances_to_consider (_type_): the subset of instances written to file
        cot (_type_): Use CoT Prompts
        instruct_follow (_type_): Are the instances from the instruction follow subset?

    Returns:
        total_number_of_instances: Total number of instances
        total_number_of_words: Total number of words
    """
    total_number_of_instances = 0
    total_number_of_words = 0

    for each_instruction in instances:
        if instruct_follow:
            instruct_file_name = os.path.join(
                os.path.join(output_path, "instruction_follow"),
                f"{dataset_name}_{each_instruction}_cot_{cot}.jsonl",
            )
        else:
            instruct_file_name = os.path.join(
                os.path.join(output_path, "no_instruction_follow"),
                f"{dataset_name}_{each_instruction}_cot_{cot}.jsonl",
            )

        instance_list = []
        for each_key in instances_to_consider:
            if each_key in instances[each_instruction]:
                instance_list.append(instances[each_instruction][each_key])

        with open(instruct_file_name, "w", errors="ignore", encoding="utf8") as writer:
            for index in range(len(instance_list)):
                writer.write(json.dumps(instance_list[index]))
                writer.write("\n")

                total_number_of_instances = total_number_of_instances + 1
                total_number_of_words = total_number_of_words + len(
                    instance_list[index]["input"].split(" ")
                )
            writer.close()

    return total_number_of_instances, total_number_of_words


def main():
    parser = argparse.ArgumentParser(description="Construct KCIFmark")
    parser.add_argument("--config", help="Dataset name", required=True)
    parser.add_argument("--output_path", help="Instruction name", required=True)
    parser.add_argument(
        "--cot",
        help="To use CoT Prompt or Not",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--debug",
        help="Debugging the schema creation",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
    )

    args = parser.parse_args()

    config = {}
    with open(args.config, "r", errors="ignore", encoding="utf8") as reader:
        config = json.load(reader)

    from datasets import disable_caching

    disable_caching()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # create instruction following paths
    if not os.path.exists(os.path.join(args.output_path, "instruction_follow")):
        os.makedirs(os.path.join(args.output_path, "instruction_follow"))

    # create no instruction following paths
    if not os.path.exists(os.path.join(args.output_path, "no_instruction_follow")):
        os.makedirs(os.path.join(args.output_path, "no_instruction_follow"))

    stats_by_dataset_instruction = {}

    original_dataset = None
    original_dataset_name = ""

    total_number_of_instances = 0
    total_number_of_words = 0

    importlib.invalidate_caches()
    # for each dataset
    for each_dataset in tqdm(config, desc="For each dataset"):
        instances_by_instruction_follow = {}
        instances_by_instruction_no_follow = {}

        # for every instruction
        try:
            dataset_name, category = each_dataset.split("_", 1)
        except ValueError:
            dataset_name = each_dataset
            category = None

        if original_dataset_name != dataset_name:
            # Before applying instruction, let's clone the dataset
            print(f"Importing dataset class {dataset_name}")
            module = importlib.import_module(
                f"construct_data.hf_to_schema.{dataset_name}"
            )
            class_def = getattr(module, dataset_name)

            # Create a new dataset
            original_dataset = class_def()
            original_dataset_name = dataset_name

        import copy

        dataset = copy.deepcopy(original_dataset)
        if category is not None:
            dataset.intermediate_representation = dataset.intermediate_representation[
                each_dataset
            ]
            assert len(dataset.intermediate_representation) > 0

        # for every instruction transformation
        for each_instruction in config[each_dataset]:
            if each_instruction not in instances_by_instruction_follow:
                instances_by_instruction_follow[each_instruction] = {}
                instances_by_instruction_no_follow[each_instruction] = {}

            # we clone the dataset and apply transformation
            import copy

            dataset = copy.deepcopy(original_dataset)

            # select the subset
            if category is not None:
                dataset.intermediate_representation = (
                    dataset.intermediate_representation[each_dataset]
                )
                assert len(dataset.intermediate_representation) > 0

            module = importlib.import_module(
                f"construct_data.instruction.{each_instruction}"
            )
            instruction_function = getattr(module, each_instruction)

            try:
                # apply the instruction transformation
                dataset.intermediate_representation = dataset.intermediate_representation.map(
                    instruction_function,
                    desc=f"Applying {each_instruction} transformation on dataset {each_dataset}",
                    num_proc=8,
                )

                # get input and output fields
                dataset.intermediate_representation = (
                    dataset.intermediate_representation.map(
                        partial(create_input_from_dataset, COT=args.cot),
                        desc=f"Creating Input for dataset {each_dataset}...",
                        num_proc=8,
                    )
                )

                dataset.intermediate_representation = (
                    dataset.intermediate_representation.shuffle(seed=42)
                )

                for each_instance in dataset.intermediate_representation:
                    each_instance["COT"] = str(args.cot)

                    if each_instruction not in [
                        "print_correct_answer",
                        "print_correct_answer_label",
                    ]:
                        if (
                            each_instance[SCHEMA_KEYS.INSTRUCTION_OUTPUT.value][-1]
                            != each_instance[SCHEMA_KEYS.INSTRUCTION_OUTPUT.value][-2]
                        ):
                            key = each_instance["dataset_input"]
                            instances_by_instruction_follow[each_instruction][
                                key
                            ] = each_instance
                        else:
                            instances_by_instruction_no_follow[each_instruction][
                                key
                            ] = each_instance
                    else:
                        key = each_instance["dataset_input"]
                        instances_by_instruction_follow[each_instruction][
                            key
                        ] = each_instance
                        instances_by_instruction_no_follow[each_instruction][
                            key
                        ] = each_instance
            except Exception:
                print(traceback.format_exc())
                print(
                    f"Skipping dataset {each_dataset} and instruction {each_instruction}"
                )
                exit()

        threshold = 1500
        if "MMLU" in each_dataset or "BBH" in each_dataset:
            threshold = 150

        # create instruction follow subset
        overlapping_keys = get_minimal_subset(
            instances_by_instruction_follow, threshold
        )

        num_instances, num_words = write_data(
            args.output_path,
            instances_by_instruction_follow,
            each_dataset,
            overlapping_keys,
            args.cot,
            instruct_follow=True,
        )

        # create no instruction follow subset
        overlapping_keys = get_minimal_subset(
            instances_by_instruction_no_follow, threshold
        )

        num_instances, num_words = write_data(
            args.output_path,
            instances_by_instruction_no_follow,
            each_dataset,
            overlapping_keys,
            args.cot,
            instruct_follow=False,
        )

        total_number_of_instances = total_number_of_instances + num_instances
        total_number_of_words = total_number_of_words + num_words


if __name__ == "__main__":
    main()
