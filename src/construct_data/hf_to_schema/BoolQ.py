from datasets import load_dataset
from ..constants import create_schema, SCHEMA_KEYS


def get_choices():
    """This functions returns the possible candidate options and their corresponding values

    Returns:
        _type_: Candidate options, Candidate values, Dictionary mapping the two
    """
    candidate_set = []
    candidate_labels = []
    candidate_set = ["True", "False"]
    candidate_labels = ["A", "B"]

    key_dict = {}
    for text, label in zip(candidate_set, candidate_labels):
        key_dict[text] = label

    return candidate_set, candidate_labels, key_dict


def transform_boolq(test_instance: dict):
    """Convert instance into the schema defined

    Args:
        test_instance (dict): each instance of the BoolQ dataset

    Returns:
        dict: the provided instance in the schema defined
    """

    # initialize the schema for every instance
    schema = create_schema()
    schema[SCHEMA_KEYS.DATA_SET.value] = "BoolQ"
    schema[SCHEMA_KEYS.HF_DATA_NAME.value] = (
        "https://huggingface.co/datasets/google/boolq/"
    )
    schema[SCHEMA_KEYS.TASK_TYPE.value] = "MCQ"

    # Copying untransformed data fields
    schema[SCHEMA_KEYS.INPUT_INSTANCE.value] = (
        f"Passage: {test_instance['passage']}\nQuestion: {test_instance['question']}\nOptions: "
    )

    # convert Boolean value to string
    test_instance["answer"] = str(test_instance["answer"])
    candidate_answer_list, candidate_labels, key_dict = get_choices()

    schema[SCHEMA_KEYS.CANDIDATE_ANSWER_SET.value] = candidate_answer_list
    # label space is same as True or False
    # we could have 0 or 1 / A or B as label spaces
    schema[SCHEMA_KEYS.CANDIDATE_ANSWER_LABEL_SPACE.value] = candidate_labels

    schema[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_LABEL.value] = key_dict[
        test_instance["answer"]
    ]

    schema[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_TEXT.value] = test_instance["answer"]

    # The instruction prefix which gets appended
    schema[SCHEMA_KEYS.TASK_PROMPT.value] = (
        "Given a passage and a boolean question, and the possible answer candidates 'A' or 'B', "
    )
    schema[SCHEMA_KEYS.FINAL_PREFIX_TASK_INSTRUCTION.value] = (
        schema[SCHEMA_KEYS.TASK_PROMPT.value]
        + "answer the question by selecting the value associated with the option label corresponding to the correct answer.\n"
    )
    schema[SCHEMA_KEYS.FINAL_SUFFIX_TASK_INSTRUCTION.value] = "\n"

    schema[SCHEMA_KEYS.INSTRUCTION_OUTPUT.value] = [
        schema[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_TEXT.value]
    ]

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


class BoolQ:
    """
    This Class holds the BoolQ dataset post transformation into the schema required
    """

    def __init__(self):
        """
        Initialize the BoolQ instance
        """
        super().__init__()
        # load the dataset
        self.dataset = load_dataset("google/boolq", split="validation")

        # convert the dataset into the schema defined
        self.intermediate_representation = self.dataset.map(
            transform_boolq,
            remove_columns=self.dataset.column_names,
            desc="Converting dataset to schema",
        )
