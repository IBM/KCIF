from datasets import load_dataset
from ..constants import create_schema, SCHEMA_KEYS
import re


def get_choices(options):
    """Get all choices

    Args:
        options (_type_): choices

    Returns:
        _type_: choices
    """
    return options


def get_label_space(options):
    """Return the label space for the given MMLU Pro task

    Args:
        options (_type_): options

    Returns:
        list: label spaces
    """
    labels = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
    ]
    return labels[: len(options)]


def transform_mmlupro(test_instance: dict):
    """Convert instance into the schema defined

    Args:
        test_instance (dict): each instance of the BoolQ dataset

    Returns:
        dict: the provided instance in the schema defined
    """

    # initialize the schema for every instance
    schema = create_schema()
    schema[SCHEMA_KEYS.DATA_SET.value] = "MMLUPro_" + test_instance["category"]
    schema[SCHEMA_KEYS.HF_DATA_NAME.value] = "TIGER-Lab/MMLU-Pro"
    schema[SCHEMA_KEYS.TASK_TYPE.value] = "MCQ"

    # Copying untransformed data fields
    schema[SCHEMA_KEYS.INPUT_INSTANCE.value] = (
        f"Question: {test_instance['question']}\n"
    )
    schema[SCHEMA_KEYS.CANDIDATE_ANSWER_SET.value] = get_choices(
        test_instance["options"]
    )
    schema[SCHEMA_KEYS.CANDIDATE_ANSWER_LABEL_SPACE.value] = get_label_space(
        test_instance["options"]
    )
    schema[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_LABEL.value] = test_instance["answer"]
    schema[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_TEXT.value] = schema[
        SCHEMA_KEYS.CANDIDATE_ANSWER_SET.value
    ][test_instance["answer_index"]]

    # The instruction prefix which gets appended
    schema[SCHEMA_KEYS.TASK_PROMPT.value] = (
        "Given a question about "
        + test_instance["category"]
        + " and "
        + str(len(schema[SCHEMA_KEYS.CANDIDATE_ANSWER_LABEL_SPACE.value]))
        + " options: "
        + ", ".join(schema[SCHEMA_KEYS.CANDIDATE_ANSWER_LABEL_SPACE.value])
        + " as candidate answers, "
    )
    schema[SCHEMA_KEYS.FINAL_PREFIX_TASK_INSTRUCTION.value] = (
        schema[SCHEMA_KEYS.TASK_PROMPT.value]
        + ", answer the question by selecting the value associated with the option label corresponding to the correct answer.\n"
    )
    schema[SCHEMA_KEYS.FINAL_SUFFIX_TASK_INSTRUCTION.value] = "\n"

    schema[SCHEMA_KEYS.INSTRUCTION_OUTPUT.value] = [test_instance["answer"]]
    assert schema[SCHEMA_KEYS.INPUT_INSTANCE.value] != ""
    assert schema[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_TEXT.value] != ""
    assert len(schema[SCHEMA_KEYS.CANDIDATE_ANSWER_SET.value]) == len(
        schema[SCHEMA_KEYS.CANDIDATE_ANSWER_LABEL_SPACE.value]
    )
    assert schema[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_LABEL.value] != ""
    assert len(schema[SCHEMA_KEYS.CANDIDATE_ANSWER_LABEL_SPACE.value]) > 0
    assert len(schema[SCHEMA_KEYS.CANDIDATE_ANSWER_SET.value]) > 0

    assert schema[SCHEMA_KEYS.TASK_PROMPT.value] != ""
    assert len(schema[SCHEMA_KEYS.INSTRUCTION_OUTPUT.value]) > 0
    return schema


class MMLUPro:
    """
    This Class holds the MMLUPro dataset post transformation into the schema required
    """

    def __init__(self):
        """
        Initialize MMLUPro Instance
        """
        super().__init__()
        # load the dataset
        self.dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

        all_categories = list(set(self.dataset["category"]))

        intermediate_representations = {}

        for each_category in all_categories:
            task_dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
            task_dataset = task_dataset.filter(
                lambda instance: instance["category"] == each_category
            )

            intermediate_representations["MMLUPro_" + each_category] = task_dataset.map(
                transform_mmlupro, remove_columns=self.dataset.column_names
            )

        self.intermediate_representation = intermediate_representations
