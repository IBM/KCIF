from datasets import load_dataset
from ..constants import create_schema, SCHEMA_KEYS


def transform_winogrande(test_instance: dict):
    """Convert instance into the schema defined

    Args:
        test_instance (dict): each instance of the Winogrande dataset

    Returns:
        dict: the provided instance in the schema defined
    """

    # initialize the schema for every instance
    schema = create_schema()
    schema[SCHEMA_KEYS.DATA_SET.value] = "Winogrande"
    schema[SCHEMA_KEYS.HF_DATA_NAME.value] = (
        "https://huggingface.co/datasets/allenai/winogrande"
    )
    schema[SCHEMA_KEYS.TASK_TYPE.value] = "MCQ"

    # Copying untransformed data fields
    schema[SCHEMA_KEYS.INPUT_INSTANCE.value] = (
        f"Sentence: {test_instance['sentence']}\nOptions: "
    )
    schema[SCHEMA_KEYS.CANDIDATE_ANSWER_SET.value] = [
        test_instance["option1"],
        test_instance["option2"],
    ]
    schema[SCHEMA_KEYS.CANDIDATE_ANSWER_LABEL_SPACE.value] = ["A", "B"]

    if test_instance["answer"] == "1":
        schema[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_TEXT.value] = test_instance["option1"]
        schema[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_LABEL.value] = "A"
        schema[SCHEMA_KEYS.INSTRUCTION_OUTPUT.value] = ["A"]
    else:
        schema[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_TEXT.value] = test_instance["option2"]
        schema[SCHEMA_KEYS.GROUND_TRUTH_ANSWER_LABEL.value] = "B"
        schema[SCHEMA_KEYS.INSTRUCTION_OUTPUT.value] = ["B"]

    # The instruction prefix which gets appended
    schema[SCHEMA_KEYS.TASK_PROMPT.value] = (
        "Given a sentence with a blank and 2 options namely 'A' and 'B', "
    )
    schema[SCHEMA_KEYS.FINAL_PREFIX_TASK_INSTRUCTION.value] = (
        "Given a sentence with a blank and 2 options namely 'A' and 'B', complete the sentence by selecting the  values associated with the correct option labels corresponding to the blank.\n"
    )

    schema[SCHEMA_KEYS.FINAL_SUFFIX_TASK_INSTRUCTION.value] = "\n"

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


class Winogrande:
    """
    This Class holds the Winogrande dataset post transformation into the schema required
    """

    def __init__(self):
        """
        Initialize Winogrande instance
        """
        super().__init__()
        # load the dataset
        self.dataset = load_dataset(
            "allenai/winogrande",
            "winogrande_xl",
            split="validation",
            trust_remote_code=True,
        )

        # convert the dataset into the schema defined
        self.intermediate_representation = self.dataset.map(
            transform_winogrande,
            remove_columns=self.dataset.column_names,
            desc="Converting dataset to schema",
        )
