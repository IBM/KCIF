from datasets import load_dataset
import re
from ..constants import create_schema, SCHEMA_KEYS


def get_choices(options):
    """This functions returns the possible candidate options and their corresponding values

    Args:
        options (_type_): Dictionary containing options and labels

    Returns:
        _type_: Candidate options, Candidate values, Dictionary mapping the two
    """
    candidate_answer_labels = ["a", "b", "c", "d", "e"]

    choices = [
        c[4:].rstrip(" ,") for c in re.findall(r"[abcd] \) .*?, |e \) .*?$", options)
    ]
    return choices, candidate_answer_labels


def transform_mathqa(test_instance: dict):
    """Convert instance into the schema defined

    Args:
        test_instance (dict): each instance of the MathQA dataset

    Returns:
        dict: the provided instance in the schema defined
    """

    # initialize the schema for every instance
    schema = create_schema()
    schema[SCHEMA_KEYS.DATA_SET.value] = "MathQA"
    schema[SCHEMA_KEYS.HF_DATA_NAME.value] = "https://math-qa.github.io/math-QA/"
    schema[SCHEMA_KEYS.TASK_TYPE.value] = "MCQ"

    # Copying untransformed data fields
    schema[SCHEMA_KEYS.INPUT_INSTANCE.value] = (
        f"Question: {test_instance['Problem']}\nOptions:"
    )
    candidate_answer_list, candidate_labels = get_choices(test_instance["options"])

    schema[SCHEMA_KEYS.CANDIDATE_ANSWER_SET.value] = candidate_answer_list
    schema[SCHEMA_KEYS.CANDIDATE_ANSWER_LABEL_SPACE.value] = candidate_labels

    schema[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_LABEL.value] = test_instance["correct"]
    schema[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_TEXT.value] = schema[
        SCHEMA_KEYS.CANDIDATE_ANSWER_SET.value
    ][
        schema[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_LABEL.value].index(
            test_instance["correct"]
        )
    ]

    # The instruction prefix which gets appended
    schema[SCHEMA_KEYS.TASK_PROMPT.value] = (
        "Given a mathematical question and 5 options namely 'a', 'b', 'c', 'd', and, 'e', as candidate answers, "
    )
    schema[SCHEMA_KEYS.FINAL_PREFIX_TASK_INSTRUCTION.value] = (
        "Given a mathematical question and 5 options namely 'a', 'b', 'c', 'd', and, 'e', answer the question by selecting the value associated with the option label corresponding to the correct answer.\n"
    )
    schema[SCHEMA_KEYS.FINAL_SUFFIX_TASK_INSTRUCTION.value] = "\n"

    schema[SCHEMA_KEYS.INSTRUCTION_OUTPUT.value] = [test_instance["correct"]]

    schema[SCHEMA_KEYS.TASK_INSTRUCTIONS.value].append(
        schema[SCHEMA_KEYS.TASK_PROMPT.value]
    )

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


class MathQA:
    """
    This Class holds the MathQA dataset post transformation into the schema required
    """

    def __init__(self):
        """
        Initialize MathQA instance
        """
        super().__init__()
        # load the dataset
        self.dataset = load_dataset(
            "allenai/math_qa", split="test", trust_remote_code=True
        )

        # convert the dataset into the schema defined
        self.intermediate_representation = self.dataset.map(
            transform_mathqa,
            remove_columns=self.dataset.column_names,
            desc="Converting dataset to schema",
        )
